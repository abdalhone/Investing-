"""The Data Desk: turns raw data into a verified, point-in-time *fact sheet* per ticker.

Design rule (the most important one in the system): **LLM agents never supply numbers from
memory.** Every number an analyst may use is computed here, deterministically, from data
that was available on the as-of date, and is given a stable fact ID such as
``AAPL.ret_3m``. Analysts must cite fact IDs; the grounding checker
(:mod:`aifirm.agents.grounding`) flags any number in an agent's text that does not match a
fact. This addresses the hallucinated-numbers failure mode and makes every claim auditable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..schemas import NewsEvent
from .fundamentals import EIGHT_K_ITEMS, Fundamentals
from .macro import MacroValue
from .news import TIER_DESCRIPTIONS, Headline
from .prices import PriceData

TRADING_DAYS = 252

# Event facts (only present once news events are attached, see FactSheet.attach_news_events):
# ``TICKER.news_N`` = a reader-extracted, tier-labelled news event; ``TICKER.filing_N`` = a
# dated SEC filing event made citable. Their value is text (unit EVENT_UNIT).
NEWS_FACT_PREFIX = "news_"
FILING_FACT_PREFIX = "filing_"
EVENT_UNIT = "event"
FILING_TIER = "A"            # SEC filings come from EDGAR: a primary source


def fact_key(fact_id: str) -> str:
    """The part of a fact ID after the first dot ("NVDA.news_1" -> "news_1")."""
    return fact_id.split(".", 1)[-1]


def is_news_fact_id(fact_id: str) -> bool:
    return "." in fact_id and fact_key(fact_id).startswith(NEWS_FACT_PREFIX) and not fact_id.startswith(("MACRO.", "MKT."))


def is_filing_fact_id(fact_id: str) -> bool:
    return "." in fact_id and fact_key(fact_id).startswith(FILING_FACT_PREFIX) and not fact_id.startswith(("MACRO.", "MKT."))


def is_event_fact_id(fact_id: str) -> bool:
    """News-event or filing-event fact (text facts rendered in the events/news sections)."""
    return is_news_fact_id(fact_id) or is_filing_fact_id(fact_id)


@dataclass
class Fact:
    id: str
    label: str
    value: float | str
    unit: str          # "USD", "%", "x", "ratio", "count", "text", "index", "date"
    as_of: str         # date the value refers to
    source: str        # provenance (e.g. "SEC 10-Q filed 2026-07-31", "Yahoo prices", "FRED DGS10")

    def render(self) -> str:
        v = self.value
        if isinstance(v, float):
            if self.unit == "USD":
                v_s = _fmt_usd(v)
            elif self.unit == "%":
                v_s = f"{v:.2f}%"
            elif self.unit in ("x", "ratio"):
                v_s = f"{v:.2f}"
            else:
                v_s = f"{v:,.2f}"
        else:
            v_s = str(v)
        return f"[{self.id}] {self.label}: {v_s} (as of {self.as_of}; {self.source})"


@dataclass
class FactSheet:
    ticker: str
    as_of: str
    name: str = ""
    sector: str = ""
    facts: list[Fact] = field(default_factory=list)
    events: list[str] = field(default_factory=list)     # dated, point-in-time filing events
    headlines: list[Headline] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)  # what could not be loaded (told to agents)
    # Quarantined news (see attach_news_events). Once ``news_quarantined`` is set, render()
    # never prints raw headline text: analysts see only the reader-extracted event facts.
    news_quarantined: bool = False
    news_events: dict[str, NewsEvent] = field(default_factory=dict)        # news fact ID -> event (tier, provenance)
    quarantined_headlines: list[Headline] = field(default_factory=list)    # raw headlines kept for audit; never rendered

    def add(self, fact: Fact) -> None:
        if isinstance(fact.value, float) and (math.isnan(fact.value) or math.isinf(fact.value)):
            return
        self.facts.append(fact)

    def get(self, fid: str) -> Fact | None:
        for f in self.facts:
            if f.id == fid:
                return f
        return None

    def ids(self) -> set[str]:
        return {f.id for f in self.facts}

    def facts_with_prefix(self, *prefixes: str) -> list[Fact]:
        return [f for f in self.facts if any(f.id.split(".", 1)[-1].startswith(p) for p in prefixes)]

    def render(self, sections: tuple[str, ...] = ("price", "fund", "macro", "events", "news")) -> str:
        lines = [f"FACT SHEET: {self.ticker} ({self.name}) | sector: {self.sector or 'n/a'} | as of {self.as_of}"]

        def ticker_key(f: Fact) -> str | None:
            return f.id.split(".", 1)[1] if f.id.startswith(f"{self.ticker}.") and not is_event_fact_id(f.id) else None

        groups = {
            "price": ("PRICE, TREND, RISK", lambda f: ticker_key(f) in PRICE_KEYS),
            "fund": ("FUNDAMENTALS (SEC filings, point-in-time) AND VALUATION",
                     lambda f: ticker_key(f) is not None and ticker_key(f) not in PRICE_KEYS),
            "macro": ("MACRO / MARKET REGIME", lambda f: f.id.startswith("MACRO.") or f.id.startswith("MKT.")),
        }
        for key in ("price", "fund", "macro"):
            if key not in sections:
                continue
            title, pred = groups[key]
            sel = [f for f in self.facts if pred(f)]
            if sel:
                lines.append(f"\n## {title}")
                lines += [f.render() for f in sel]
        if "events" in sections:
            lines.append("\n## RECENT SEC FILING EVENTS (point-in-time)")
            filing_facts = [f for f in self.facts if is_filing_fact_id(f.id)]
            if filing_facts:   # citable form: "- [NVDA.filing_1] 2026-05-28: 8-K - ..."
                lines += [f"- [{f.id}] {f.value}" for f in filing_facts]
            else:
                lines += [f"- {e}" for e in self.events] or ["- none in window"]
        if "news" in sections:
            lines += self._render_news()
        if self.data_gaps:
            lines.append("\n## DATA GAPS (unknown - do not fill in from memory)")
            lines += [f"- {g}" for g in self.data_gaps]
        return "\n".join(lines)

    def _render_news(self) -> list[str]:
        """The news section. Quarantined: reader-extracted event facts only, never headline
        text. Otherwise: raw headlines (legacy path, used when no reader ran)."""
        news_facts = [f for f in self.facts if is_news_fact_id(f.id)]
        if self.news_quarantined or news_facts:
            out = ["\n## NEWS EVENTS (extracted from headlines by a quarantined reader; source tier set by code)",
                   "Tier A = primary source (SEC filing, company release, official newswire); tier B = established "
                   "news organisation; tier C = unverified (aggregators, blogs, unknown publishers, rumors). "
                   "Raw headline text is withheld. Cite an event by its ID."]
            out += [f.render() for f in news_facts] or ["- no reader-verified news events (see DATA GAPS if news could not be read)"]
            return out
        out = ["\n## RECENT HEADLINES"]
        if self.headlines:
            out += [f"- {h.published[:10]} | {h.publisher} | {h.title}" for h in self.headlines]
        else:
            out.append("- no headlines available (historical mode or feed unavailable) - do not assume news you have not been shown")
        return out

    def news_tier(self, fact_id: str) -> str | None:
        """Source tier of an event fact on this sheet: the code-assigned tier of a news event
        (tier C when its metadata is missing: unproven provenance is unverified), "A" for a
        filing event, None for a non-event fact or an ID not on the sheet."""
        if self.get(fact_id) is None:
            return None
        if is_filing_fact_id(fact_id):
            return FILING_TIER
        if is_news_fact_id(fact_id):
            ev = self.news_events.get(fact_id)
            return ev.tier if ev is not None else "C"
        return None

    def has_news_content(self) -> bool:
        """True if the sheet carries anything that came from news: raw headlines or news-event facts."""
        return bool(self.headlines) or any(is_news_fact_id(f.id) for f in self.facts)

    def attach_news_events(self, events: list[NewsEvent], gaps: list[str] | tuple[str, ...] = (),
                           cite_filings: bool = True) -> list[Fact]:
        """Quarantine the news on this sheet: replace raw headlines with event facts.

        * Each event becomes a citable fact ``TICKER.news_N`` (:func:`news_event_facts`),
          and its record (tier, provenance) is kept in :attr:`news_events` by fact ID.
        * Raw headlines move to :attr:`quarantined_headlines` (audit only) and
          :attr:`news_quarantined` is set, so :meth:`render` never prints headline text
          again, even if ``headlines`` is refilled later.
        * ``gaps`` (e.g. a reader failure note) are added to the data gaps.
        * ``cite_filings``: also make each filing event citable as ``TICKER.filing_N``
          (tier A), so the tier gate can tell a filing-based view from a rumor-based one.
        Calling it again replaces the previously attached event facts. Returns the facts added."""
        self.facts = [f for f in self.facts if not is_event_fact_id(f.id)]
        self.news_events = {}
        self.quarantined_headlines = [*self.quarantined_headlines, *self.headlines]
        self.headlines = []
        self.news_quarantined = True
        added: list[Fact] = []
        for fct, ev in zip(news_event_facts(events, self.ticker), events_in_fact_order(events), strict=True):
            self.add(fct)
            self.news_events[fct.id] = ev
            added.append(fct)
        if cite_filings:
            for fct in filing_event_facts(self.events, self.ticker):
                self.add(fct)
                added.append(fct)
        for g in gaps:
            if g and g not in self.data_gaps:
                self.data_gaps.append(g)
        return added


PRICE_KEYS = {
    "price", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "rel_ret_3m", "rel_ret_12m",
    "vol_20d", "vol_60d", "beta_1y", "max_dd_1y", "dist_52w_high", "sma50_sma200", "rsi_14",
    "adv_20d", "price_vs_sma200",
}


def _fmt_usd(v: float) -> str:
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:,.2f}T"
    if a >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.2f}"


# --------------------------------------------------------------------------- price facts

def _ret(s: pd.Series, n: int, skip: int = 0) -> float:
    if len(s) <= n:
        return float("nan")
    end = s.iloc[-1 - skip]
    start = s.iloc[-1 - n]
    return float(end / start - 1.0) * 100.0


def _rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n + 1:
        return float("nan")
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] > 0 else float("inf")
    return float(100 - 100 / (1 + rs))


def price_facts(ticker: str, prices: PriceData, benchmark: str | None = "SPY") -> list[Fact]:
    """Trend, momentum and risk facts computed from data up to the last row of ``prices``.

    The ``price`` fact is the **as-traded** close (what printed that day, not back-adjusted
    for later splits; see :mod:`aifirm.data.prices`), so it is consistent with as-filed SEC
    share counts and with the minimum-price rule. Returns use the adjusted close.
    Statistics that need a full year (12-month returns, momentum, 1-year drawdown, 52-week
    high) are only computed with at least 252 observations, and windowed statistics only
    with a full window: a few weeks after a listing they would be mislabelled."""
    t = ticker.upper()
    adj = prices.adj_close[t].dropna()
    close = prices.close[t].dropna()
    level = prices.price_level(t).dropna()
    if adj.empty:
        return []
    asof = str(adj.index[-1].date())
    src = "Yahoo daily prices (returns from adjusted close)"
    rets = adj.pct_change().dropna()
    out = []
    if not level.empty:
        out.append(Fact(f"{t}.price", "Last close", float(level.iloc[-1]), "USD", str(level.index[-1].date()),
                        "Yahoo daily close, as traded (not adjusted for later splits)"))
    for key, label, n, skip in [
        ("ret_1m", "Return, 1 month (21 trading days)", 21, 0),
        ("ret_3m", "Return, 3 months (63 trading days)", 63, 0),
        ("ret_6m", "Return, 6 months (126 trading days)", 126, 0),
        ("ret_12m", "Return, 12 months (252 trading days)", 252, 0),
        ("mom_12_1", "Momentum 12-1 (12-month return skipping the latest month)", 252, 21),
    ]:
        out.append(Fact(f"{t}.{key}", label, _ret(adj, n, skip), "%", asof, src))
    if benchmark and benchmark in prices.adj_close.columns and benchmark != t:
        b = prices.adj_close[benchmark].reindex(adj.index).ffill()
        out.append(Fact(f"{t}.rel_ret_3m", f"Return minus {benchmark} return, 3 months", _ret(adj, 63) - _ret(b, 63), "%", asof, src))
        out.append(Fact(f"{t}.rel_ret_12m", f"Return minus {benchmark} return, 12 months", _ret(adj, 252) - _ret(b, 252), "%", asof, src))
        br = b.pct_change().dropna()
        joined = pd.concat([rets, br], axis=1, join="inner").dropna().iloc[-TRADING_DAYS:]
        if len(joined) > 60 and joined.iloc[:, 1].var() > 0:
            beta = joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()
            window = "1 year daily" if len(joined) >= TRADING_DAYS - 1 else f"daily, only {len(joined)} trading days of history"
            out.append(Fact(f"{t}.beta_1y", f"Beta vs {benchmark} ({window})", float(beta), "ratio", asof, src))
    if len(rets) >= 20:
        out.append(Fact(f"{t}.vol_20d", "Realized volatility, 20 days (annualized)", float(rets.iloc[-20:].std() * np.sqrt(TRADING_DAYS) * 100), "%", asof, src))
    if len(rets) >= 60:
        out.append(Fact(f"{t}.vol_60d", "Realized volatility, 60 days (annualized)", float(rets.iloc[-60:].std() * np.sqrt(TRADING_DAYS) * 100), "%", asof, src))
    if len(adj) >= TRADING_DAYS:
        last_year = adj.iloc[-TRADING_DAYS:]
        dd = last_year / last_year.cummax() - 1
        out.append(Fact(f"{t}.max_dd_1y", "Maximum drawdown over the last year", float(dd.min() * 100), "%", asof, src))
        out.append(Fact(f"{t}.dist_52w_high", "Distance below 52-week high", float((adj.iloc[-1] / last_year.max() - 1) * 100), "%", asof, src))
    if len(adj) >= 200:
        sma50, sma200 = adj.iloc[-50:].mean(), adj.iloc[-200:].mean()
        out.append(Fact(f"{t}.sma50_sma200", "50-day / 200-day moving-average ratio (>1 = uptrend)", float(sma50 / sma200), "ratio", asof, src))
        out.append(Fact(f"{t}.price_vs_sma200", "Price vs 200-day moving average", float((adj.iloc[-1] / sma200 - 1) * 100), "%", asof, src))
    out.append(Fact(f"{t}.rsi_14", "RSI (14 days; >70 overbought, <30 oversold)", _rsi(adj), "index", asof, src))
    if len(close) >= 20:
        vol = prices.volume[t].reindex(close.index).fillna(0)
        adv = float((close * vol).iloc[-20:].mean())
        out.append(Fact(f"{t}.adv_20d", "Average daily dollar volume, 20 days", adv, "USD", asof, "Yahoo daily close x volume"))
    return [f for f in out if not (isinstance(f.value, float) and math.isnan(f.value))]


def price_history_gaps(ticker: str, prices: PriceData) -> list[str]:
    """Data gaps for a short price history (e.g. a recent listing)."""
    t = ticker.upper()
    adj = prices.adj_close[t].dropna() if t in prices.adj_close.columns else pd.Series(dtype=float)
    if adj.empty or len(adj) >= TRADING_DAYS:
        return []
    gap = (f"Only {len(adj)} trading days of price history (since {adj.index[0].date()}): 1-year statistics "
           "(12-month return, momentum, 1-year drawdown, 52-week high) are not computed and the stock's "
           "longer-term behaviour is unknown")
    return [gap]


# --------------------------------------------------------------------------- fundamental facts

def fundamental_facts(f: Fundamentals, last_price: float | None,
                      split_ratio: Callable[[str], float] | None = None) -> tuple[list[Fact], list[str]]:
    """Fundamental and valuation facts.

    ``last_price`` must be the as-traded close on the as-of date. ``split_ratio(date)``
    returns the product of the stock's split ratios with ex-date after ``date`` and on or
    before the as-of date; it puts an as-filed share count (dated before a split) on the
    as-of share basis so that market cap = price x shares is consistent. Market cap and
    multiples are omitted (with a data gap) for foreign private issuers, whose SEC share
    count is of ordinary shares while the US listing may be an ADR."""
    t = f.ticker
    out: list[Fact] = []
    gaps: list[str] = list(f.gaps)
    v = f.values
    ratio = split_ratio or (lambda _d: 1.0)

    def src(fv) -> str:
        filed = f" filed {fv.filed}" if fv.filed else ""
        return f"SEC XBRL {fv.concept}{filed}, {fv.basis}"

    labels = {
        "revenue": "Revenue", "gross_profit": "Gross profit", "operating_income": "Operating income",
        "net_income": "Net income", "op_cash_flow": "Operating cash flow", "capex": "Capital expenditure",
        "rnd": "R&D expense", "sbc": "Stock-based compensation", "total_assets": "Total assets",
        "total_liabilities": "Total liabilities", "equity": "Shareholders' equity", "cash": "Cash and equivalents",
        "long_term_debt": "Long-term debt", "current_assets": "Current assets", "current_liabilities": "Current liabilities",
    }
    for key, label in labels.items():
        if key in v:
            fv = v[key]
            suffix = " (TTM)" if fv.basis.startswith("TTM") else (" (fiscal year)" if fv.basis == "fiscal year" else "")
            out.append(Fact(f"{t}.{key}", label + suffix, fv.value, "USD", fv.period_end, src(fv)))
        elif key in ("revenue", "net_income", "op_cash_flow", "equity"):
            gaps.append(f"{label} not available from SEC XBRL as of {f.as_of}")

    def growth(key: str, label: str) -> None:
        if key in v and key in f.prior_year and f.prior_year[key].value:
            cur, prev = v[key].value, f.prior_year[key].value
            if prev > 0:
                out.append(Fact(f"{t}.{key}_growth_yoy", f"{label} growth, TTM year over year", (cur / prev - 1) * 100, "%", v[key].period_end, "computed from SEC XBRL TTM values"))

    growth("revenue", "Revenue")
    growth("net_income", "Net income")
    growth("op_cash_flow", "Operating cash flow")

    def margin(num: str, label: str) -> None:
        if num in v and "revenue" in v and v["revenue"].value > 0 and v[num].period_end == v["revenue"].period_end:
            out.append(Fact(f"{t}.{num}_margin", f"{label} margin (TTM)", v[num].value / v["revenue"].value * 100, "%", v[num].period_end, "computed from SEC XBRL"))

    margin("gross_profit", "Gross")
    margin("operating_income", "Operating")
    margin("net_income", "Net")

    fcf = None
    if "op_cash_flow" in v:
        fcf = v["op_cash_flow"].value - (v["capex"].value if "capex" in v else 0.0)
        out.append(Fact(f"{t}.fcf", "Free cash flow (operating cash flow minus capex, TTM)", fcf, "USD", v["op_cash_flow"].period_end, "computed from SEC XBRL"))
    if "net_income" in v and "equity" in v and v["equity"].value > 0:
        out.append(Fact(f"{t}.roe", "Return on equity (TTM net income / latest equity)", v["net_income"].value / v["equity"].value * 100, "%", v["equity"].period_end, "computed from SEC XBRL"))
    if "long_term_debt" in v and "equity" in v and v["equity"].value > 0:
        out.append(Fact(f"{t}.debt_to_equity", "Long-term debt / equity", v["long_term_debt"].value / v["equity"].value, "ratio", v["equity"].period_end, "computed from SEC XBRL"))
    if "current_assets" in v and "current_liabilities" in v and v["current_liabilities"].value > 0:
        out.append(Fact(f"{t}.current_ratio", "Current ratio", v["current_assets"].value / v["current_liabilities"].value, "ratio", v["current_assets"].period_end, "computed from SEC XBRL"))

    sh = f.shares_outstanding if f.shares_outstanding is not None and f.shares_outstanding.value > 0 else None
    shares = None
    if sh is not None:
        k = ratio(sh.period_end)
        shares = sh.value * k
        source = f"SEC cover page filed {sh.filed}"
        if k != 1.0:
            source += f"; x{k:g} for stock splits after {sh.period_end}"
        label = "Shares outstanding (ordinary shares per SEC filing)" if f.foreign_filer else "Shares outstanding"
        out.append(Fact(f"{t}.shares_out", label, shares, "count", sh.period_end, source))
        prior = f.shares_outstanding_prior_year
        if prior is not None and prior.value > 0:
            chg = (shares / (prior.value * ratio(prior.period_end)) - 1) * 100
            out.append(Fact(f"{t}.share_count_change_yoy", "Change in share count, ~1 year (negative = buybacks)", chg, "%", sh.period_end, "computed from SEC cover pages (split-adjusted)"))
    elif not any("share count" in g for g in gaps):   # fundamentals may already have said why
        gaps.append("Shares outstanding unavailable - valuation multiples not computed")

    if shares and f.foreign_filer:
        gaps.append("Foreign private issuer (20-F/40-F filer): the SEC share count is of ordinary shares while the "
                    "US listing may be an ADR representing several of them - market cap and valuation multiples "
                    "not computed")
    elif shares and last_price is not None and last_price > 0:
        mcap = shares * last_price
        out.append(Fact(f"{t}.market_cap", "Market capitalization (last close x shares outstanding)", mcap, "USD", f.as_of, "computed"))
        if "net_income" in v and v["net_income"].value > 0:
            out.append(Fact(f"{t}.pe_ttm", "Price / earnings (TTM)", mcap / v["net_income"].value, "x", f.as_of, "computed"))
        if "revenue" in v and v["revenue"].value > 0:
            out.append(Fact(f"{t}.ps_ttm", "Price / sales (TTM)", mcap / v["revenue"].value, "x", f.as_of, "computed"))
        if "equity" in v and v["equity"].value > 0:
            out.append(Fact(f"{t}.pb", "Price / book", mcap / v["equity"].value, "x", f.as_of, "computed"))
        if fcf is not None:
            out.append(Fact(f"{t}.fcf_yield", "Free-cash-flow yield (FCF / market cap)", fcf / mcap * 100, "%", f.as_of, "computed"))
        if "net_income" in v:
            out.append(Fact(f"{t}.earnings_yield", "Earnings yield (TTM net income / market cap)", v["net_income"].value / mcap * 100, "%", f.as_of, "computed"))
    return out, gaps


def macro_facts(values: list[MacroValue]) -> list[Fact]:
    out = []
    by_id = {m.series_id: m for m in values}
    for m in values:
        out.append(Fact(f"MACRO.{m.series_id}", m.label, m.value, m.unit, m.observation_date, f"FRED {m.series_id}"))
        if m.change_3m is not None:
            out.append(Fact(f"MACRO.{m.series_id}_chg_3m", f"{m.label}: change over ~3 months", m.change_3m, "pts" if m.unit != "%" else "%pts", m.observation_date, f"FRED {m.series_id}"))
    if "DGS10" in by_id and "DGS2" in by_id:
        out.append(Fact("MACRO.curve_2s10s", "Yield curve slope (10y minus 2y)", by_id["DGS10"].value - by_id["DGS2"].value, "%pts", by_id["DGS10"].observation_date, "computed from FRED"))
    return out


def market_facts(prices: PriceData, benchmark: str = "SPY") -> list[Fact]:
    if benchmark not in prices.adj_close.columns:
        return []
    s = prices.adj_close[benchmark].dropna()
    if s.empty:
        return []
    asof = str(s.index[-1].date())
    out = [
        Fact(f"MKT.{benchmark}_ret_3m", f"{benchmark} return, 3 months", _ret(s, 63), "%", asof, "Yahoo prices"),
        Fact(f"MKT.{benchmark}_ret_12m", f"{benchmark} return, 12 months", _ret(s, 252), "%", asof, "Yahoo prices"),
    ]
    if len(s) >= 200:
        out.append(Fact(f"MKT.{benchmark}_vs_sma200", f"{benchmark} vs its 200-day moving average", float((s.iloc[-1] / s.iloc[-200:].mean() - 1) * 100), "%", asof, "Yahoo prices"))
    r = s.pct_change().dropna()
    if len(r) >= 20:
        out.append(Fact(f"MKT.{benchmark}_vol_20d", f"{benchmark} realized volatility, 20 days (annualized)", float(r.iloc[-20:].std() * np.sqrt(TRADING_DAYS) * 100), "%", asof, "Yahoo prices"))
    return [f for f in out if not (isinstance(f.value, float) and math.isnan(f.value))]


def filing_events(filings: list[dict]) -> list[str]:
    events = []
    form4 = [x for x in filings if x["form"] in ("4", "4/A")]
    for x in filings:
        form = x["form"]
        if form in ("8-K", "8-K/A"):
            items = [i.strip() for i in (x.get("items") or "").split(",") if i.strip()]
            desc = "; ".join(EIGHT_K_ITEMS.get(i, f"item {i}") for i in items if i != "9.01") or "8-K"
            events.append(f"{x['filed']}: 8-K - {desc}")
        elif form in ("10-Q", "10-K", "10-K/A", "10-Q/A", "S-1", "S-3", "DEF 14A", "SC 13D", "SC 13G", "13D", "13G"):
            events.append(f"{x['filed']}: {form} filed")
    if form4:
        events.append(f"{len(form4)} insider transaction reports (Form 4) filed in the window (direction not parsed)")
    return events


# --------------------------------------------------------------------------- news and filing event facts

_SUMMARY_UNSAFE = re.compile(r"[:\[\]{}<>`|\\]")


def events_in_fact_order(events: list[NewsEvent]) -> list[NewsEvent]:
    """Events in the order their fact IDs are numbered: by headline (newest first, as the
    reader numbered them), then in the order the reader listed them (stable sort)."""
    return sorted(events, key=lambda e: e.headline_index)


def news_event_facts(events: list[NewsEvent], ticker: str | None = None) -> list[Fact]:
    """Citable facts for news events: ``TICKER.news_1``, ``TICKER.news_2``... in
    :func:`events_in_fact_order`. The label carries tier, type, direction and materiality,
    the value is the sanitized reader summary, ``as_of`` the publish date and the source
    the code-assigned source name and tier (never the raw publisher string, which is
    attacker-controlled free text). Labels contain no numbers, so they never count as
    grounding support. ``ticker`` defaults to the events' own ticker."""
    out: list[Fact] = []
    for k, ev in enumerate(events_in_fact_order(events), 1):
        t = (ticker or ev.ticker).upper()
        summary = _SUMMARY_UNSAFE.sub(" ", ev.summary)
        summary = " ".join(summary.split()) or "no summary"
        label = (f"News event (tier {ev.tier}, {ev.event_type}, {ev.direction}, "
                 f"{ev.materiality} materiality)")
        writer = "code" if ev.summary_by == "code" else "a quarantined reader"
        source = (f"{ev.source_name}, tier {ev.tier} ({TIER_DESCRIPTIONS.get(ev.tier, 'unverified')}) "
                  f"assigned by code; summary by {writer}")
        out.append(Fact(f"{t}.{NEWS_FACT_PREFIX}{k}", label, summary, EVENT_UNIT, ev.published[:10] or "unknown", source))
    return out


_EVENT_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def filing_event_facts(events: list[str], ticker: str) -> list[Fact]:
    """Citable facts ``TICKER.filing_N`` for the dated filing events of a sheet (tier A).
    The value is the event text exactly as the events section shows it."""
    out: list[Fact] = []
    for k, e in enumerate(events, 1):
        m = _EVENT_DATE.match(e)
        out.append(Fact(f"{ticker.upper()}.{FILING_FACT_PREFIX}{k}", f"SEC filing event (tier {FILING_TIER}, primary source)",
                        e, EVENT_UNIT, m.group(1) if m else "window", "SEC EDGAR filing index"))
    return out


def build_factsheet(
    ticker: str,
    as_of: date | str,
    prices: PriceData,
    fundamentals: Fundamentals | None = None,
    macro: list[MacroValue] | None = None,
    filings: list[dict] | None = None,
    headlines: list[Headline] | None = None,
    sector: str = "",
    benchmark: str = "SPY",
) -> FactSheet:
    """Assemble a FactSheet. ``prices`` must already be sliced to ``as_of`` by the caller
    (``PriceData.as_of``); this function re-slices defensively."""
    as_of_s = str(pd.Timestamp(as_of).date())
    p = prices.as_of(as_of_s)
    t = ticker.upper()
    fs = FactSheet(ticker=t, as_of=as_of_s, sector=sector)
    for fct in price_facts(t, p, benchmark):
        fs.add(fct)
    fs.data_gaps += price_history_gaps(t, p)
    price_fact = fs.get(f"{t}.price")
    last_price = float(price_fact.value) if price_fact is not None else None
    if fundamentals is not None:
        fs.name = fundamentals.entity_name
        fs.sector = fs.sector or fundamentals.sic_description
        # shares are put on the share basis of the day the price printed
        through = price_fact.as_of if price_fact is not None else as_of_s
        ff, gaps = fundamental_facts(fundamentals, last_price, split_ratio=lambda d: p.split_ratio(t, d, through))
        for fct in ff:
            fs.add(fct)
        fs.data_gaps += gaps
    else:
        fs.data_gaps.append("No SEC fundamentals (ETF, foreign issuer, or fetch failure)")
    for fct in market_facts(p, benchmark):
        fs.add(fct)
    if macro:
        for fct in macro_facts(macro):
            fs.add(fct)
    else:
        fs.data_gaps.append("Macro data unavailable")
    if filings is not None:
        fs.events = filing_events(filings)
    if headlines is not None:
        fs.headlines = headlines
    return fs
