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
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .fundamentals import EIGHT_K_ITEMS, Fundamentals
from .macro import MacroValue
from .news import Headline
from .prices import PriceData

TRADING_DAYS = 252


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
        groups = {
            "price": ("PRICE, TREND, RISK", lambda f: f.id.startswith(f"{self.ticker}.") and f.id.split(".")[1] in PRICE_KEYS),
            "fund": ("FUNDAMENTALS (SEC filings, point-in-time) AND VALUATION", lambda f: f.id.startswith(f"{self.ticker}.") and f.id.split(".")[1] not in PRICE_KEYS),
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
            lines += [f"- {e}" for e in self.events] or ["- none in window"]
        if "news" in sections:
            lines.append("\n## RECENT HEADLINES")
            if self.headlines:
                lines += [f"- {h.published[:10]} | {h.publisher} | {h.title}" for h in self.headlines]
            else:
                lines.append("- no headlines available (historical mode or feed unavailable) - do not assume news you have not been shown")
        if self.data_gaps:
            lines.append("\n## DATA GAPS (unknown - do not fill in from memory)")
            lines += [f"- {g}" for g in self.data_gaps]
        return "\n".join(lines)


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
    """Trend, momentum and risk facts computed from data up to the last row of ``prices``."""
    t = ticker.upper()
    adj = prices.adj_close[t].dropna()
    close = prices.close[t].dropna()
    if adj.empty:
        return []
    asof = str(adj.index[-1].date())
    src = "Yahoo daily prices (returns from adjusted close)"
    rets = adj.pct_change().dropna()
    out = [Fact(f"{t}.price", "Last close", float(close.iloc[-1]), "USD", asof, "Yahoo daily close")]
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
            out.append(Fact(f"{t}.beta_1y", f"Beta vs {benchmark} (1 year daily)", float(beta), "ratio", asof, src))
    if len(rets) >= 20:
        out.append(Fact(f"{t}.vol_20d", "Realized volatility, 20 days (annualized)", float(rets.iloc[-20:].std() * np.sqrt(TRADING_DAYS) * 100), "%", asof, src))
    if len(rets) >= 60:
        out.append(Fact(f"{t}.vol_60d", "Realized volatility, 60 days (annualized)", float(rets.iloc[-60:].std() * np.sqrt(TRADING_DAYS) * 100), "%", asof, src))
    last_year = adj.iloc[-TRADING_DAYS:]
    dd = last_year / last_year.cummax() - 1
    out.append(Fact(f"{t}.max_dd_1y", "Maximum drawdown over the last year", float(dd.min() * 100), "%", asof, src))
    out.append(Fact(f"{t}.dist_52w_high", "Distance below 52-week high", float((adj.iloc[-1] / last_year.max() - 1) * 100), "%", asof, src))
    if len(adj) >= 200:
        sma50, sma200 = adj.iloc[-50:].mean(), adj.iloc[-200:].mean()
        out.append(Fact(f"{t}.sma50_sma200", "50-day / 200-day moving-average ratio (>1 = uptrend)", float(sma50 / sma200), "ratio", asof, src))
        out.append(Fact(f"{t}.price_vs_sma200", "Price vs 200-day moving average", float((adj.iloc[-1] / sma200 - 1) * 100), "%", asof, src))
    out.append(Fact(f"{t}.rsi_14", "RSI (14 days; >70 overbought, <30 oversold)", _rsi(adj), "index", asof, src))
    vol = prices.volume[t].reindex(close.index).fillna(0)
    adv = float((close * vol).iloc[-20:].mean())
    out.append(Fact(f"{t}.adv_20d", "Average daily dollar volume, 20 days", adv, "USD", asof, "Yahoo daily close x volume"))
    return [f for f in out if not (isinstance(f.value, float) and math.isnan(f.value))]


# --------------------------------------------------------------------------- fundamental facts

def fundamental_facts(f: Fundamentals, last_price: float | None) -> tuple[list[Fact], list[str]]:
    t = f.ticker
    out: list[Fact] = []
    gaps: list[str] = []
    v = f.values

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

    sh = f.shares_outstanding
    if sh:
        out.append(Fact(f"{t}.shares_out", "Shares outstanding", sh.value, "count", sh.period_end, f"SEC cover page filed {sh.filed}"))
        if f.shares_outstanding_prior_year and f.shares_outstanding_prior_year.value > 0:
            chg = (sh.value / f.shares_outstanding_prior_year.value - 1) * 100
            out.append(Fact(f"{t}.share_count_change_yoy", "Change in share count, ~1 year (negative = buybacks)", chg, "%", sh.period_end, "computed from SEC cover pages"))
    else:
        gaps.append("Shares outstanding unavailable - valuation multiples not computed")

    if sh and last_price:
        mcap = sh.value * last_price
        out.append(Fact(f"{t}.market_cap", "Market capitalization (last close x shares outstanding)", mcap, "USD", f.as_of, "computed"))
        if "net_income" in v and v["net_income"].value > 0:
            out.append(Fact(f"{t}.pe_ttm", "Price / earnings (TTM)", mcap / v["net_income"].value, "x", f.as_of, "computed"))
        if "revenue" in v and v["revenue"].value > 0:
            out.append(Fact(f"{t}.ps_ttm", "Price / sales (TTM)", mcap / v["revenue"].value, "x", f.as_of, "computed"))
        if "equity" in v and v["equity"].value > 0:
            out.append(Fact(f"{t}.pb", "Price / book", mcap / v["equity"].value, "x", f.as_of, "computed"))
        if fcf is not None and mcap > 0:
            out.append(Fact(f"{t}.fcf_yield", "Free-cash-flow yield (FCF / market cap)", fcf / mcap * 100, "%", f.as_of, "computed"))
        if "net_income" in v and mcap > 0:
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
    last_price = fs.get(f"{t}.price").value if fs.get(f"{t}.price") else None
    if fundamentals is not None:
        fs.name = fundamentals.entity_name
        fs.sector = fs.sector or fundamentals.sic_description
        ff, gaps = fundamental_facts(fundamentals, last_price)
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
