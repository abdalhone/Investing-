"""One decision cycle, end to end: data -> committee -> portfolio -> risk -> orders -> journal.

``decide`` is shared by the live CLI (``aifirm run``) and the walk-forward backtest, so a
backtest exercises exactly the code that would trade (P10). The cycle, in order:

1. **Resolve and learn (P7, P3).** Forecasts whose horizon has passed by ``as_of`` are
   resolved from prices cut at ``as_of``; a few of the most informative ones get a
   process lesson. Believability weights and lessons are then read *point-in-time*:
   only outcomes resolved on or before ``as_of`` count.
2. **Screen, news, committee (P4, P5, P6).** The deterministic quant view is computed for
   the whole universe at once (it is cross-sectional). With ``universe.screen`` enabled it
   ranks the broad universe and only the top band plus every held name go to the LLM
   committee (:mod:`aifirm.quant.screen`); otherwise every universe name does. Live
   headlines of those names are read by the quarantined news reader into tier-labelled
   events (:mod:`aifirm.agents.news_reader`); no other agent ever sees raw headline text,
   and headlines that are not read are withheld. Each ticker's committee then runs
   sequentially (analysts concurrently inside it), with each LLM analyst's view passed
   through the tier gate and its point-in-time calibration map before pooling, and the
   quant core tilted by the bounded LLM overlay (``process.combination``), whose bound
   shrinks when resolved forecasts show the LLM pool ranking worse than the quant.
3. **Code sizes, LLMs may only shrink (P1, P8).** ``construct_portfolio`` turns pooled
   views into weights. The LLM risk review can only scale them down, and the
   deterministic risk engine (P9) has the last word before orders are generated.
4. **Journal everything (P10).** Every step is appended to the hash-chained ledger and
   every non-abstaining view becomes a resolvable forecast.

Fail-closed rules specific to this module:

* **Systemic LLM failure holds the book.** If half or more of the LLM analyst calls (or of
  the required IC memos) fail, or the LLM budget is exhausted, the cycle trades nothing
  (target = current book, still subject to the risk engine). An API outage is not
  information about the stocks, so it must not liquidate the book or open positions.
  Isolated failures follow the committee's own rules (an abstention; an edge of 0 for an
  unheld name whose memo failed; a held name whose memo failed keeps its current weight,
  ``CommitteeResult.hold_current``, with a warning).
* **A risk-review halt keeps the book.** ``apply_risk_review`` receives the current
  weights, so ``halt_trading`` means no trades from the committee this cycle (as the
  reviewer is told), never a sale of the active sleeve; the risk engine still applies.
* **A news reader failure is a data gap, not an outage.** The sheet gets no news events
  and a gap note; reader calls never count toward the outage rule, and with no events the
  tier gate means news cannot raise exposure anyway.
* **A data source that fails is a data gap, not a crash** (``gather_data``): the agents
  are told what is missing and must not fill it in from memory (P2). A missing
  benchmark is the exception: without it nothing can be resolved or compared, so the
  run stops.
* **No information is not a reason to trade.** A held name on which every analyst
  abstained keeps its current weight (see ``construct_portfolio``), with a warning; an
  agent that abstains on every name (e.g. the quant with fewer than three tickers) is
  reported as a warning, so an all-cash book is never left unexplained.
* **Any unexpected error stops the cycle** after writing a ``cycle_error`` ledger entry;
  no orders are produced.

The risk state sees the whole NAV path, not only decision days (P9): ``nav_marks`` (or,
for the live book, a path rebuilt from the held positions since the previous decision)
raises the high-water mark with every daily close and feeds the worst one-day loss since
the previous cycle to the daily-loss halt.

``decide`` slices ``prices`` at ``as_of`` itself, so a caller cannot leak the future
into a decision by passing the full history (P3).
"""

from __future__ import annotations

import copy
import math
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from .agents import (CommitteeResult, apply_risk_review, llm_analysts, read_headlines, review_portfolio, run_committee,
                     write_lesson)
from .config import FirmConfig
from .data.factsheet import FactSheet, build_factsheet
from .data.fundamentals import Fundamentals, load_fundamentals, recent_filings
from .data.macro import MacroValue, macro_snapshot
from .data.news import Headline, recent_headlines
from .data.prices import PriceData, load_prices
from .journal import (COMMITTEE_AGENT, Calibrator, ForecastStore, Ledger, LessonStore, believability_weights,
                      fit_calibrators, make_forecast_records)
from .llm.client import LLM, CallRecord, OfflineLLM
from .portfolio import (OverlayBound, RiskState, adaptive_bound, apply_risk_limits, construct_portfolio,
                        orders_from_target, update_risk_state)
from .quant import QUANT_AGENT, ScreenResult, quant_views, screen_universe
from .universes import file_tickers, preset_tickers
from .schemas import (AggregatedView, ForecastRecord, Lesson, Order, RiskReview, TargetPortfolio, TargetPosition,
                      ViewRecord)

PRICE_LOOKBACK_DAYS = 400       # calendar days: 252-day statistics need ~365 days plus slack
FILINGS_LOOKBACK_DAYS = 120     # SEC filing events shown to the events analyst
STALE_PRICE_DAYS = 5            # a last close older than this (calendar days) is flagged as stale
LIQUIDITY_WINDOW = 20           # trading days for ADV and daily volatility
MIN_LIQUIDITY_OBS = 5           # fewer valid days and ADV / volatility are unknown (orders blocked)
LLM_OUTAGE_FRACTION = 0.5       # this share of failed LLM calls means an outage: hold the book
DIVIDEND_MIN_YIELD = 1e-6       # adj/raw close ratio moves smaller than this are rounding noise, not dividends
MAX_LESSONS_PER_CYCLE = 8       # lesson calls cost money; learn from the most informative outcomes
_GAP_TEXT_CHARS = 160           # error text shown to agents is truncated

FundamentalsLoader = Callable[[str, str], Fundamentals | None]
MacroLoader = Callable[[str], list[MacroValue]]
FilingsLoader = Callable[[str, str], list[dict]]
HeadlinesLoader = Callable[[str, str], list[Headline]]

# A stand-in used when no LLM is configured (quant-only runs): every call fails closed.
_NO_LLM = OfflineLLM({})


def _day(d: object) -> str:
    return str(pd.Timestamp(d).date())  # type: ignore[arg-type]


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _GAP_TEXT_CHARS else text[:_GAP_TEXT_CHARS - 3] + "..."


# --------------------------------------------------------------------------- configuration helpers


def universe(cfg: FirmConfig) -> list[str]:
    """Investable tickers in config order: upper-cased, de-duplicated, benchmark excluded
    (the benchmark is the passive core, never an active name).

    With ``universe.screen`` enabled the list is ``universe.tickers`` followed by the
    screen's sources: the bundled ``preset``, the tickers in ``file`` (relative to the
    config file, :meth:`FirmConfig.screen_file_path`) and the inline ``tickers``. An
    unknown preset raises ``KeyError`` and a missing file ``FileNotFoundError`` (fail
    loudly: a silently empty universe would never trade and never say why)."""
    bench = cfg.universe.benchmark.upper()
    sources = list(cfg.universe.tickers)
    scr = cfg.universe.screen
    if scr.enabled:
        if scr.preset.strip():
            sources += preset_tickers(scr.preset)
        if scr.file.strip():
            sources += file_tickers(cfg.screen_file_path())
        sources += list(scr.tickers)
    out: list[str] = []
    for t in sources:
        u = t.strip().upper()
        if u and u != bench and u not in out:
            out.append(u)
    return out


def quant_only_config(cfg: FirmConfig) -> FirmConfig:
    """A copy of ``cfg`` whose committee is the deterministic quant analyst alone, with every
    LLM stage (challenge, memo, risk review) switched off. Used for ``mode="quant"``
    backtests and the quant-only baseline (P10): the same engine, no language model."""
    out = cfg.model_copy(deep=True)
    out.analysts.enabled = [QUANT_AGENT]
    out.process.challenge_enabled = False
    out.process.memo_enabled = False
    out.process.risk_review_enabled = False
    return out


def needs_llm(cfg: FirmConfig) -> bool:
    """True if any stage of ``cfg`` calls a language model."""
    return bool(llm_analysts(cfg)) or cfg.process.memo_enabled or cfg.process.risk_review_enabled


def sector_map(cfg: FirmConfig, factsheets: Mapping[str, FactSheet]) -> dict[str, str]:
    """Ticker -> sector: the manual ``cfg.universe.sectors`` label wins, else the fact
    sheet's sector (SEC SIC description). Unlabelled names are omitted."""
    manual = {k.upper(): v for k, v in cfg.universe.sectors.items()}
    out = {}
    for t, fs in factsheets.items():
        label = (manual.get(t.upper()) or fs.sector or "").strip()
        if label:
            out[t] = label
    return out


# --------------------------------------------------------------------------- data


def gather_data(cfg: FirmConfig, as_of: str, live: bool, prices: PriceData | None = None, *,
                fundamentals_loader: FundamentalsLoader | None = None,
                macro_loader: MacroLoader | None = None,
                filings_loader: FilingsLoader | None = None,
                headlines_loader: HeadlinesLoader | None = None) -> tuple[PriceData, dict[str, FactSheet]]:
    """Load point-in-time data and build one fact sheet per universe ticker (P2, P3).

    * ``prices``: if None, daily prices for the universe and the benchmark are downloaded
      with :data:`PRICE_LOOKBACK_DAYS` of history. The returned prices are cut at
      ``as_of`` whether or not they were passed in.
    * **The decision session.** A decision is made at a close, so ``as_of`` is first
      resolved to the last trading session on or before it (the last row of the price
      index). On a holiday or weekend that is the previous session: fundamentals, filings,
      macro and headlines are then loaded as of that close, never as of the calendar day
      (a filing accepted on Good Friday must not reach a decision at Thursday's close, P3).
      The fact sheets' ``as_of`` is that session.
    * Fundamentals (SEC XBRL as filed), filing events and macro (FRED with publication
      lags) are point-in-time. Loaders are injectable so tests and backtests can run
      offline; the defaults hit SEC and FRED (both cached on disk), and use the SEC filer
      pinned in ``cfg.universe.ciks`` when there is one.
    * Headlines are live-only feeds that cannot be replayed point-in-time, so they are
      loaded only when ``live`` and ``cfg.process.news_enabled``.
    * A loader that fails for one ticker becomes a data gap on that ticker's sheet; it
      never crashes the run. A ticker with no prices at all gets no sheet (and therefore
      no view). A missing benchmark raises: nothing can be resolved without it.
    """
    as_of_s = _day(as_of)
    bench = cfg.universe.benchmark.upper()
    tickers = universe(cfg)
    if prices is None:
        start = pd.Timestamp(as_of_s) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
        prices = load_prices([*tickers, bench], start, as_of_s)
    p = prices.as_of(as_of_s)
    if bench not in p.adj_close.columns or p.adj_close[bench].dropna().empty:
        raise RuntimeError(f"no prices for the benchmark {bench} as of {as_of_s}: cannot build fact sheets")
    as_of_s = _day(p.close.index[-1])    # the session whose close is decided on (a holiday -> the day before)

    pins = cfg.universe.cik_pins()      # explicit SEC filers (config), else resolved by ticker and date
    fund_fn = fundamentals_loader or (lambda t, d: load_fundamentals(t, d, cik=pins.get(t)))
    filings_fn = filings_loader or (lambda t, d: recent_filings(t, d, FILINGS_LOOKBACK_DAYS, cik=pins.get(t)))
    macro_fn = macro_loader or macro_snapshot
    news_fn = headlines_loader or recent_headlines
    want_news = live and cfg.process.news_enabled

    shared_gaps: list[str] = []
    try:
        macro = list(macro_fn(as_of_s) or [])
    except Exception as exc:  # a failed source is a gap the agents are told about (P2)
        macro = []
        shared_gaps.append(f"Macro data could not be loaded ({_short(exc)})")

    manual_sectors = {k.upper(): v for k, v in cfg.universe.sectors.items()}
    factsheets: dict[str, FactSheet] = {}
    for t in tickers:
        if t not in p.close.columns or p.close[t].dropna().empty:
            continue
        gaps = list(shared_gaps)
        try:
            fundamentals = fund_fn(t, as_of_s)
        except Exception as exc:
            fundamentals = None
            gaps.append(f"SEC fundamentals could not be loaded ({_short(exc)})")
        try:
            filings = list(filings_fn(t, as_of_s) or [])
        except Exception as exc:
            filings = None
            gaps.append(f"SEC filing events could not be loaded ({_short(exc)}); 'none in window' means unknown")
        headlines = None
        if want_news:
            try:
                headlines = list(news_fn(t, as_of_s) or [])
            except Exception as exc:
                gaps.append(f"Headlines could not be loaded ({_short(exc)})")
        fs = build_factsheet(t, as_of_s, p, fundamentals=fundamentals, macro=macro, filings=filings,
                             headlines=headlines, sector=manual_sectors.get(t, ""), benchmark=bench)
        last = p.close[t].dropna().index[-1]
        age = (pd.Timestamp(as_of_s) - last).days
        if age > STALE_PRICE_DAYS:
            gaps.append(f"Latest price is from {last.date()} ({age} days before the as-of date); "
                        "price facts may be stale")
        for g in gaps:
            if g not in fs.data_gaps:
                fs.data_gaps.append(g)
        factsheets[t] = fs
    return p, factsheets


@dataclass(frozen=True)
class MarketSnapshot:
    """Execution inputs as of a close: reference prices, dollar ADV and daily volatility.

    A ticker is absent from a dict when the value is unknown; downstream code then blocks
    its orders (fail closed) rather than trading on a guess."""

    prices: dict[str, float]
    adv: dict[str, float]          # average daily dollar volume over LIQUIDITY_WINDOW days
    daily_vol: dict[str, float]    # std-dev of daily returns over LIQUIDITY_WINDOW days (fraction)


def market_snapshot(prices: PriceData, as_of: str, tickers: Iterable[str]) -> MarketSnapshot:
    """Reference prices are the close on the last row on or before ``as_of`` (decide at
    the close, P3); a ticker with no close on that row has no reference price."""
    p = prices.as_of(_day(as_of))
    ref: dict[str, float] = {}
    adv: dict[str, float] = {}
    vol: dict[str, float] = {}
    rets = p.returns()
    for t in sorted(set(tickers)):
        if t not in p.close.columns or p.close.empty:
            continue
        last = p.close[t].iloc[-1]
        if math.isfinite(last) and last > 0:
            ref[t] = float(last)
        dollar = (p.close[t] * p.volume[t]).iloc[-LIQUIDITY_WINDOW:].dropna()
        if len(dollar) >= MIN_LIQUIDITY_OBS and dollar.mean() > 0:
            adv[t] = float(dollar.mean())
        r = rets[t].iloc[-LIQUIDITY_WINDOW:].dropna()
        if len(r) >= MIN_LIQUIDITY_OBS and r.std() > 0:
            vol[t] = float(r.std())
    return MarketSnapshot(prices=ref, adv=adv, daily_vol=vol)


def implied_dividends(prices: PriceData) -> pd.DataFrame:
    """Cash dividend per share on each ex-date, inferred from adjusted vs raw closes.

    On an ex-date ``t`` the adjusted close (total return) gains on the raw close by the
    dividend: ``adj_t / adj_{t-1} = (close_t + D_t) / close_{t-1}``, so
    ``D_t = close_{t-1} * (adj_t / adj_{t-1} - close_t / close_{t-1})``. Only the factor for
    the ex-date itself survives in that ratio, so no later dividend leaks in (P3). Moves
    below ``DIVIDEND_MIN_YIELD`` of the previous close are rounding noise and negative ones
    are data errors; both count as 0. Missing prints are forward-filled first, so a
    dividend whose ex-date print is missing is credited on the next print. Same index and
    columns as ``prices.close``; 0.0 where there is no dividend.

    Paper books credit these amounts (``PaperBroker.credit_dividends``) and the backtest's
    benchmark baseline uses the same amounts, so a book holding the benchmark earns
    exactly the benchmark's total return before costs.
    """
    close = prices.close.ffill()
    adj = prices.adj_close.reindex(index=close.index, columns=close.columns).ffill()
    excess = adj / adj.shift(1) - close / close.shift(1)            # D_t / close_{t-1}
    div = (close.shift(1) * excess).where(excess > DIVIDEND_MIN_YIELD, 0.0)
    return div.fillna(0.0)


# --------------------------------------------------------------------------- the decision record


@dataclass
class Decision:
    """Everything one decision cycle produced. ``risk_state`` is the state to persist for
    the next cycle (the caller's object is not modified)."""

    as_of: str
    run_id: str
    mode: str                                        # "llm", "offline" or "quant"
    benchmark: str
    config_fingerprint: str
    nav: float
    committee: dict[str, CommitteeResult]            # per ticker
    aggregates: dict[str, AggregatedView]            # final pooled view per ticker
    believability: dict[str, float]
    target_proposed: TargetPortfolio                 # construct_portfolio (before any risk layer)
    target_reviewed: TargetPortfolio                 # after the LLM risk review (can only shrink)
    target_final: TargetPortfolio                    # after the deterministic risk engine
    risk_review: RiskReview | None
    risk_messages: list[str]
    orders: list[Order]
    order_messages: list[str]
    market: MarketSnapshot
    forecasts: list[ForecastRecord]                  # written this cycle
    resolved: list[ForecastRecord]                   # resolved at the start of this cycle
    lessons_written: list[Lesson]
    risk_state: RiskState
    usage: dict
    warnings: list[str]
    data_gaps: dict[str, list[str]]
    held_by_outage: bool = False
    ledger_head: str = ""
    lessons_used: dict[str, int] = field(default_factory=dict)
    screen: ScreenResult | None = None                     # the broad-universe screen (None when disabled)
    overlay_bound: OverlayBound | None = None              # the overlay bound in force (overlay mode)
    calibrators: dict[str, Calibrator] = field(default_factory=dict)   # per LLM analyst (empty when disabled)
    news: dict[str, dict] = field(default_factory=dict)    # ticker -> news reading summary (live news only)

    def exposure(self) -> dict[str, float]:
        """The final book's exposure: ``active_gross`` (sum of |active weights|), ``core``
        (benchmark leg), ``cash`` and ``n_names`` (active names held), fractions of NAV."""
        w = self.target_final.weights()
        return {"active_gross": float(sum(abs(x) for x in w.values())), "core": float(self.target_final.core_weight),
                "cash": float(self.target_final.cash_weight), "n_names": float(len(w))}

    def final_probabilities(self) -> dict[str, float]:
        """The probability the firm holds for every name it scored this cycle: the committee's
        final probability for committee names, the quant's for screened names that were not
        deep-dived (the probability that kept them out of the book)."""
        out = dict(self.screen.quant_probs()) if self.screen is not None else {}
        out.update({t: a.prob_outperform for t, a in self.aggregates.items()})
        return out


# --------------------------------------------------------------------------- helpers


# Sealed stages, in order; the revised "pool_inputs" and "aggregate" (after a challenge) take 5 and 6.
_EVENT_STAGE = {"view": 0, "pool_inputs": 1, "aggregate": 2, "challenge": 3, "revision": 4, "memo": 7}
_REVISED_STAGE = {"pool_inputs": 5, "aggregate": 6}


class _EventBuffer:
    """Collects committee log events (some arrive from worker threads) and replays them in
    a deterministic order, so the same inputs always produce the same ledger content."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, payload: dict) -> None:
        with self._lock:
            self.events.append((event, dict(payload)))

    def ordered(self) -> list[tuple[str, dict]]:
        def key(item: tuple[int, tuple[str, dict]]) -> tuple:
            i, (event, payload) = item
            if event == "revision":     # the only events emitted concurrently
                return (_EVENT_STAGE["revision"], str(payload.get("agent", "")), int(payload.get("sample", 0)), i)
            if payload.get("stage") == "revised" and event in _REVISED_STAGE:
                return (_REVISED_STAGE[event], "", 0, i)
            return (_EVENT_STAGE.get(event, 8), "", 0, i)
        return [e for _, e in sorted(enumerate(self.events), key=key)]


def _mode(llm: LLM | None) -> str:
    if llm is None:
        return "quant"
    return "offline" if llm.is_offline else "llm"


def _usage_summary(calls: list[CallRecord]) -> dict:
    """Usage of one cycle (the same fields as ``Usage.summary``, for a slice of calls)."""
    live = [c for c in calls if not c.cached]
    return {
        "calls": len(calls),
        "live_calls": len(live),
        "cached_calls": len(calls) - len(live),
        "input_tokens": sum(c.input_tokens for c in live),
        "output_tokens": sum(c.output_tokens for c in live),
        "cache_read_tokens": sum(c.cache_read_tokens for c in live),
        "cost_usd": round(sum(c.cost_usd for c in calls), 4),
        "failures": sum(1 for c in calls if c.error),
    }


def _hold_target(current_weights: Mapping[str, float], bench: str, as_of: str, note: str) -> TargetPortfolio:
    """A target equal to the current book: no trades unless the risk engine cuts risk."""
    positions = [TargetPosition(ticker=t, weight=float(w), prob_outperform=0.5, expected_excess_return_pct=0.0,
                                rationale=note, constraints_hit=["pipeline:hold"])
                 for t, w in sorted(current_weights.items()) if t != bench and float(w) != 0.0]
    core = float(current_weights.get(bench, 0.0))
    return TargetPortfolio(as_of=as_of, positions=positions, core_weight=core,
                           cash_weight=1.0 - core - sum(p.weight for p in positions), notes=[note])


def _llm_failed(v: ViewRecord) -> bool:
    return any(i.startswith("llm failure") for i in v.grounding_issues)


def _nav_path_since_last_decision(state: RiskState, p: PriceData, shares: Mapping[str, float], nav: float,
                                  as_of: str) -> list[float]:
    """The book's closing NAVs from the previous decision (``state.last_nav``) through the
    day before ``as_of``, rebuilt from the shares held now and forward-filled closes.

    Used when the caller passes no ``nav_marks`` (the live book). Valid because a paper
    book's orders fill at the open of the day after a decision and nothing trades until the
    next decision, so the shares held now were held at every close in between, and cash is
    ``nav - shares * close(as_of)`` (a dividend credited in between shifts that by its
    amount; small next to the losses the halt is about). Returns [] when it cannot be
    rebuilt (no previous decision on record, or a held name without a price), in which
    case the caller's ``prev_nav`` is used alone."""
    if state.last_as_of is None or not _valid_nav(state.last_nav):
        return []
    last = pd.Timestamp(state.last_as_of)
    if last >= pd.Timestamp(as_of):
        return []
    closes = p.close.ffill()
    held = {t: float(q) for t, q in shares.items() if float(q) != 0.0}
    if any(t not in closes.columns for t in held) or closes.empty:
        return []
    now = closes.iloc[-1]
    if any(not (math.isfinite(now[t]) and now[t] > 0) for t in held):
        return []
    cash = float(nav) - sum(q * float(now[t]) for t, q in held.items())
    between = closes.loc[(closes.index > last) & (closes.index < pd.Timestamp(as_of))]
    path = [float(state.last_nav)]  # type: ignore[arg-type]
    for _, row in between.iterrows():
        px = [float(row[t]) for t in held]
        if not all(math.isfinite(x) and x > 0 for x in px):
            return []
        path.append(cash + sum(q * x for q, x in zip(held.values(), px)))
    return path


def _valid_nav(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def _worst_day_return(path: Sequence[float]) -> float | None:
    """The worst one-day return along consecutive NAVs (-inf if any NAV is broken)."""
    worst: float | None = None
    for a, b in zip(path, path[1:]):
        r = b / a - 1.0 if _valid_nav(a) and _valid_nav(b) else -math.inf
        worst = r if worst is None else min(worst, r)
    return worst


def _abstention_warnings(committee: Mapping[str, CommitteeResult], current_weights: Mapping[str, float]) -> list[str]:
    """Explain cycles in which views are missing, so an all-cash or unchanged book is never
    left unexplained: agents that abstained on every name (LLM failures excluded: the
    outage rule reports those), and held names on which every analyst abstained."""
    out = []
    by_agent: dict[str, list[ViewRecord]] = {}
    for r in committee.values():
        for v in r.views:
            by_agent.setdefault(v.agent, []).append(v)
    for agent, views in by_agent.items():
        if views and all(v.view.abstain for v in views) and not any(_llm_failed(v) for v in views):
            reasons = Counter(v.view.abstain_reason.strip() or "no reason given" for v in views)
            out.append(f"{agent} abstained on all {len(committee)} names: {reasons.most_common(1)[0][0]}")
    held_blind = sorted(t for t, r in committee.items()
                        if r.aggregate.n_views == 0 and float(current_weights.get(t, 0.0)) != 0.0)
    if held_blind:
        out.append(f"every analyst abstained on held name(s) {', '.join(held_blind)}: kept at the current weight "
                   f"(no information is not a reason to sell)")
    return out


def _view_summary(payload: Mapping) -> str:
    """Plain-text summary of a logged sealed view, for the lesson writer."""
    view = payload.get("view") or {}
    lines = [f"Sealed probability {float(view.get('prob_outperform', 0.5)):.3f}, "
             f"expected excess return {float(view.get('expected_excess_return_pct', 0.0)):+.2f}%."]
    if payload.get("revised_prob") is not None:
        lines.append(f"After the challenge round it revised to {float(payload['revised_prob']):.3f}.")
    for e in (view.get("evidence") or [])[:8]:
        facts = ", ".join(e.get("fact_ids") or []) or "none"
        lines.append(f"- [{e.get('direction', 'neutral')}] {e.get('point', '')} (facts: {facts})")
    risks = view.get("key_risks") or []
    if risks:
        lines.append("Key risks named: " + "; ".join(str(r) for r in risks[:4]))
    if not payload.get("grounding_ok", True):
        lines.append("The grounding checker flagged this view.")
    return "\n".join(lines)


def _committee_summary(agg: Mapping | None, memo: Mapping | None) -> str:
    lines = []
    if agg:
        lines.append(f"Pooled probability {float(agg.get('prob_outperform', 0.5)):.3f} from "
                     f"{agg.get('n_views', 0)} views (dispersion {float(agg.get('dispersion', 0.0)):.3f}, "
                     f"challenged: {bool(agg.get('challenged'))}).")
    if memo and not memo.get("error"):
        lines.append(f"Chair: {memo.get('recommendation')} with multiplier "
                     f"{float(memo.get('edge_multiplier', 1.0)):.2f}. "
                     f"Thesis: {memo.get('thesis_one_sentence', '')}")
        pre = memo.get("premortem") or []
        if pre:
            lines.append("Pre-mortem: " + "; ".join(str(x) for x in pre[:4]))
    return "\n".join(lines)


def _summaries_from_ledger(ledger: Ledger, wanted: set[str]) -> dict[str, str]:
    """Forecast id -> summary of the view behind it, read back from the ledger, so a
    lesson is written from what was actually recorded at decision time."""
    views: dict[str, str] = {}
    aggs: dict[str, dict] = {}
    memos: dict[str, dict] = {}
    for e in ledger.entries():
        p = e.get("payload") or {}
        event = e.get("event")
        if event not in ("view", "aggregate", "memo") or "run_id" not in p:
            continue
        base = f"{p['run_id']}|{p.get('as_of')}|{p.get('ticker')}"
        if event == "view":
            key = f"{base}|{p.get('agent')}|s{p.get('sample', 0)}"
            if key in wanted:
                views[key] = _view_summary(p)
        elif f"{base}|{COMMITTEE_AGENT}" in wanted:
            (aggs if event == "aggregate" else memos)[f"{base}|{COMMITTEE_AGENT}"] = p
    for key in wanted:
        if key.endswith(f"|{COMMITTEE_AGENT}") and key in aggs:
            views[key] = _committee_summary(aggs.get(key), memos.get(key))
    return views


def _write_lessons(llm: LLM, resolved: list[ForecastRecord], cfg: FirmConfig, ledger: Ledger,
                   lesson_store: LessonStore, log: Callable[[str, dict], None]) -> list[Lesson]:
    """Write process lessons for the most informative newly resolved forecasts.

    Only LLM analysts and the chair read lessons, so only their forecasts get one. The
    forecasts furthest from a coin flip's Brier score (confident hits and confident
    misses) teach the most; at most :data:`MAX_LESSONS_PER_CYCLE` are written. A lesson
    is stored with its forecast's ``resolve_on`` date, so it can never reach a decision
    made before the outcome was known (P3)."""
    readers = set(llm_analysts(cfg)) | ({COMMITTEE_AGENT} if cfg.process.memo_enabled else set())
    scored = [(r, r.brier) for r in resolved if r.agent in readers and r.outcome is not None and r.brier is not None]
    candidates = [r for r, brier in sorted(scored, key=lambda x: (-abs(x[1] - 0.25), x[0].id))]
    candidates = candidates[:MAX_LESSONS_PER_CYCLE]
    if not candidates:
        return []
    summaries = _summaries_from_ledger(ledger, {r.id for r in candidates})
    written = []
    for rec in candidates:
        summary = summaries.get(rec.id)
        if not summary:
            continue        # without the recorded view a lesson would be invented, not learned
        lesson = write_lesson(llm, rec, summary, cfg)
        if lesson is None:
            continue
        lesson_store.add(lesson, resolved_on=rec.resolve_on)
        log("lesson", {"forecast_id": rec.id, "resolved_on": rec.resolve_on, **lesson.model_dump()})
        written.append(lesson)
    return written


def _factsheet_payload(fs: FactSheet) -> dict:
    """The sheet as the agents were given it, for the ledger. Raw headline text appears only
    under ``quarantined_headlines`` (audit) once the sheet is quarantined."""
    return {"ticker": fs.ticker, "name": fs.name, "sector": fs.sector,
            "facts": [{"id": f.id, "label": f.label, "value": f.value, "unit": f.unit, "as_of": f.as_of,
                       "source": f.source} for f in fs.facts],
            "events": fs.events, "headlines": fs.headlines, "data_gaps": fs.data_gaps,
            "news_quarantined": fs.news_quarantined,
            "news_events": {fid: ev.model_dump() for fid, ev in fs.news_events.items()},
            "quarantined_headlines": fs.quarantined_headlines}


def _quarantine_news(cfg: FirmConfig, llm: LLM | None, model: LLM, as_of: str, factsheets: Mapping[str, FactSheet],
                     read_for: Iterable[str], log: Callable[[str, Mapping], None]) -> tuple[dict[str, FactSheet],
                                                                                          dict[str, dict]]:
    """Quarantine every sheet that carries raw headlines (RESEARCH 5.5, 9.3 item 3).

    Sheets of ``read_for`` tickers are read by the news reader when it is enabled and a
    language model is available: their headlines become tier-labelled event facts
    (``FactSheet.attach_news_events``). Every other sheet with headlines gets no events and
    a data-gap note, so raw headline text never reaches an analyst, the chair or the risk
    reviewer. Any reader failure, including an unexpected exception, is the same: no
    events and a gap (fail closed; it is not counted as an LLM outage). Sheets are copied
    before they are changed: the caller's sheets are never modified. Returns the sheets
    and a per-ticker summary of each reading."""
    out = dict(factsheets)
    summaries: dict[str, dict] = {}
    wanted = set(read_for)
    pc, mc = cfg.process, cfg.models
    domains = cfg.universe.issuer_domain_map()
    for t in sorted(factsheets):
        fs = factsheets[t]
        if not fs.headlines or fs.news_quarantined:
            continue
        fs = copy.deepcopy(fs)
        if t not in wanted:
            why = "not read: outside this cycle's deep-dive set"
        elif not pc.news_reader_enabled:
            why = "not read: the news reader is disabled (process.news_reader_enabled)"
        elif llm is None:
            why = "not read: no language model in this run"
        else:
            why = ""
        if why:
            fs.attach_news_events([], gaps=[f"Headlines withheld ({why}); no news events this cycle - do not "
                                            "assume any news"])
            summaries[t] = {"read": False, "reason": why, "n_headlines": len(fs.quarantined_headlines)}
            out[t] = fs
            continue
        try:
            res = read_headlines(model, t, fs.headlines, as_of, model=mc.news_reader_model,
                                 effort=mc.news_reader_effort, issuer_domains=domains.get(t, ()),
                                 max_headlines=pc.news_max_headlines, isolate_untrusted=pc.news_isolate_untrusted)
            fs.attach_news_events(res.events, res.gaps)
            payload = res.to_dict()
        except Exception as exc:  # noqa: BLE001 - any reader error is a data gap, never a crash or a trade
            reason = _short(exc)
            fs.attach_news_events([], gaps=[f"News could not be read ({reason}); no news events this cycle - do not "
                                            "assume any news"])
            payload = {"ticker": t, "as_of": as_of, "events": [], "failed": True, "error": reason}
        log("news_reading", payload)
        summaries[t] = {"read": True, "failed": bool(payload.get("failed")), "n_events": len(payload["events"]),
                        "tiers": dict(Counter(e["tier"] for e in payload["events"]))}
        out[t] = fs
    return out, summaries


# --------------------------------------------------------------------------- the decision cycle


def decide(cfg: FirmConfig, llm: LLM | None, as_of: str, prices: PriceData, factsheets: dict[str, FactSheet], *,
           current_weights: dict[str, float], nav: float, prev_nav: float | None, risk_state: RiskState,
           forecast_store: ForecastStore, lesson_store: LessonStore, ledger: Ledger, run_id: str,
           positions: dict[str, float] | None = None,
           trading_calendar: Iterable | None = None,
           runs_dir: str | None = None,
           nav_marks: Sequence[float] | None = None) -> Decision:
    """Run one decision cycle at the close of ``as_of`` (see the module docstring).

    * ``llm``: None is allowed only when no stage needs a model (see
      :func:`quant_only_config`); otherwise a ``ValueError`` is raised.
    * ``current_weights``: fraction of NAV per ticker, including the benchmark (core leg).
    * ``positions``: current shares per ticker for order sizing; if None they are derived
      from ``current_weights`` at the reference prices.
    * ``trading_calendar``: dates used to set each forecast's resolution date; defaults to
      the price index up to ``as_of`` (beyond it, business days are used).
    * ``runs_dir``: where the risk engine looks for the kill-switch file (default
      :meth:`FirmConfig.runs_path`: a relative ``runs_dir`` is relative to the config
      file, never to the working directory).
    * ``prev_nav``: the previous trading day's closing NAV. Used for the daily-loss halt
      only when no NAV path since the previous decision is known.
    * ``nav_marks``: the book's closing NAV on every trading day since the previous
      decision, oldest first: that decision's NAV, then each later close, excluding
      ``as_of`` (the backtest passes them). Each one updates the risk state's high-water
      mark and the worst one-day loss among them feeds the daily-loss halt. If None, the
      path is rebuilt from the held shares when ``risk_state`` records the previous
      decision (``last_as_of``, ``last_nav``); see ``_nav_path_since_last_decision``.

    Orders are sized at the ``as_of`` close and are meant to be filled at the next
    trading day's open (P3); executing them is the caller's job.
    """
    if llm is None and needs_llm(cfg):
        raise ValueError("this configuration has LLM stages (analysts, memo or risk review) but no LLM was given; "
                         "pass an LLM or use quant_only_config(cfg)")
    for t, w in current_weights.items():
        if not math.isfinite(float(w)):
            raise ValueError(f"current weight for {t} is not finite: {w!r}")
    as_of_s = _day(as_of)
    bench = cfg.universe.benchmark.upper()
    model = llm or _NO_LLM
    mode = _mode(llm)
    p = prices.as_of(as_of_s)
    calendar = trading_calendar if trading_calendar is not None else p.close.index
    calls_before = len(model.usage.calls)

    def log(event: str, payload: Mapping) -> None:
        ledger.append(event, {"run_id": run_id, "as_of": as_of_s, **payload})

    log("cycle_start", {"mode": mode, "config_fingerprint": cfg.fingerprint(), "config": cfg.model_dump(),
                        "nav": nav, "prev_nav": prev_nav, "current_weights": current_weights,
                        "positions": positions, "risk_state": risk_state,
                        "nav_marks": list(nav_marks) if nav_marks is not None else None})
    try:
        return _decide(cfg, llm, model, mode, as_of_s, bench, p, factsheets, calendar, calls_before, log,
                       current_weights=current_weights, nav=nav, prev_nav=prev_nav, risk_state=risk_state,
                       forecast_store=forecast_store, lesson_store=lesson_store, ledger=ledger, run_id=run_id,
                       positions=positions, runs_dir=runs_dir or str(cfg.runs_path()), nav_marks=nav_marks)
    except Exception as exc:
        log("cycle_error", {"error": _short(exc)})
        raise


def _decide(cfg: FirmConfig, llm: LLM | None, model: LLM, mode: str, as_of: str, bench: str, p: PriceData,
            factsheets: dict[str, FactSheet], calendar: Iterable, calls_before: int,
            log: Callable[[str, Mapping], None], *, current_weights: dict[str, float], nav: float,
            prev_nav: float | None, risk_state: RiskState, forecast_store: ForecastStore,
            lesson_store: LessonStore, ledger: Ledger, run_id: str, positions: dict[str, float] | None,
            runs_dir: str, nav_marks: Sequence[float] | None) -> Decision:
    warnings: list[str] = []
    tickers = [t for t in universe(cfg) if t in factsheets]
    missing = [t for t in universe(cfg) if t not in factsheets]
    if missing:
        warnings.append(f"no fact sheet (no price data) for: {', '.join(missing)}; no view this cycle")
    held_names = {t for t, w in current_weights.items() if float(w) != 0.0 and t != bench}
    overlay = cfg.process.combination == "overlay"
    if overlay and QUANT_AGENT not in cfg.analysts.enabled:
        warnings.append("process.combination is 'overlay' but the quant analyst is not enabled: every name's core is "
                        "0.5, so the LLM pool can move probabilities by at most the overlay bound")

    # 2a. the quant over the whole cross-section; the screen picks the deep-dive set (code, no LLM)
    screen_on = cfg.universe.screen.enabled
    want_quant = (QUANT_AGENT in cfg.analysts.enabled or screen_on) and bool(tickers)
    qviews = quant_views([factsheets[t] for t in tickers], cfg.process.horizon_days) if want_quant else {}
    screen: ScreenResult | None = None
    if screen_on:
        sc = cfg.universe.screen
        screen = screen_universe({t: factsheets[t] for t in tickers}, qviews, p, as_of, top_n=sc.deep_dive_top_n,
                                 min_dollar_volume=sc.min_dollar_volume, min_price=cfg.risk.min_price,
                                 held=held_names)
        committee_tickers = list(screen.deep_dive)
    else:
        committee_tickers = list(tickers)

    # 2b. news: quarantined reader for the committee's names; every other headline is withheld
    factsheets, news = _quarantine_news(cfg, llm, model, as_of, factsheets, committee_tickers, log)
    for t in tickers:
        log("factsheet", _factsheet_payload(factsheets[t]))
    if screen is not None:
        log("screen", screen.to_dict())

    # 1. resolve, learn, then read the track record point-in-time (P7, P3)
    resolved = forecast_store.resolve(p, as_of)
    if resolved:
        log("forecasts_resolved", {"n": len(resolved), "records": [
            {"id": r.id, "agent": r.agent, "outcome": r.outcome, "brier": r.brier,
             "realized_excess_return_pct": r.realized_excess_return_pct} for r in resolved]})
    unresolvable = dict(getattr(forecast_store, "unresolvable", {}) or {})
    if unresolvable:
        log("forecasts_unresolvable", {"n": len(unresolvable), "reasons": unresolvable})
        for why, n in Counter(unresolvable.values()).items():
            warnings.append(f"{n} due forecast(s) cannot be resolved yet: {why}")
    lessons_written = _write_lessons(llm, resolved, cfg, ledger, lesson_store, log) if llm is not None else []
    agents = list(dict.fromkeys(cfg.analysts.enabled))
    records = forecast_store.all()
    # Deterministic rules (offline mode, the quant) cannot have memorised history, so their
    # forecasts count toward weights whatever their date; a real LLM's pre-cutoff ones do not (P10).
    include_pre = llm is None or llm.is_offline
    believability = believability_weights(records, as_of, agents, include_pre_cutoff=include_pre,
                                          min_dates=cfg.process.believability_min_dates)
    readers = [*llm_analysts(cfg), COMMITTEE_AGENT]
    k = cfg.analysts.max_lessons_in_prompt
    lessons = {a: lesson_store.retrieve(a, as_of, k=k) for a in readers}
    log("believability", {"weights": believability, "lessons_retrieved": {a: len(v) for a, v in lessons.items()}})
    calibrators: dict[str, Calibrator] = {}
    if cfg.calibration.enabled and llm_analysts(cfg):
        calibrators = fit_calibrators(records, as_of, llm_analysts(cfg), include_pre_cutoff=include_pre,
                                      min_n=cfg.calibration.min_n, prior_strength=cfg.calibration.prior_strength,
                                      models=cfg.models.analyst_models)
    log("calibration", {"enabled": cfg.calibration.enabled,
                        "calibrators": {a: c.to_dict() for a, c in calibrators.items()}})
    bound: OverlayBound | None = None
    if overlay:
        bound = adaptive_bound(records, as_of, cfg, include_pre_cutoff=include_pre)
        log("overlay_bound", bound.to_dict())
        if bound.max_tilt < bound.configured:
            warnings.append(f"overlay bound reduced to {bound.max_tilt:.3f} (configured {bound.configured:.3f}): "
                            f"{bound.reason}")

    # 2c. committee per deep-dive ticker (sequential over tickers; analysts are concurrent inside)
    committee: dict[str, CommitteeResult] = {}
    memo_total = memo_failed = 0
    quant_seat = QUANT_AGENT in cfg.analysts.enabled
    for t in committee_tickers:
        buffer = _EventBuffer()
        try:
            committee[t] = run_committee(model, factsheets[t], cfg, quant_view=qviews.get(t) if quant_seat else None,
                                         believability=believability, lessons=lessons,
                                         held=float(current_weights.get(t, 0.0)) != 0.0, log=buffer,
                                         calibrators=calibrators, max_tilt=bound.max_tilt if bound else None)
        finally:   # the real ticker on every event (in blind mode an LLM may echo the alias)
            for event, payload in buffer.ordered():
                log(event, {**payload, "ticker": t})
        memos = [pl for ev, pl in buffer.events if ev == "memo"]
        memo_total += len(memos)
        memo_failed += sum(1 for pl in memos if "error" in pl)
    aggregates = {t: r.aggregate for t, r in committee.items()}

    llm_views = [v for r in committee.values() for v in r.views if v.agent != QUANT_AGENT]
    failed_views = sum(1 for v in llm_views if _llm_failed(v))
    outage_reasons = []
    if llm_views and failed_views / len(llm_views) >= LLM_OUTAGE_FRACTION:
        outage_reasons.append(f"{failed_views} of {len(llm_views)} analyst calls failed")
    if memo_total and memo_failed / memo_total >= LLM_OUTAGE_FRACTION:
        outage_reasons.append(f"{memo_failed} of {memo_total} required IC memos failed")
    if math.isfinite(model.usage.budget_usd) and model.usage.spent_usd >= model.usage.budget_usd:
        outage_reasons.append(f"LLM budget of ${model.usage.budget_usd:.2f} exhausted")
    held_by_outage = bool(outage_reasons)
    if not held_by_outage:
        warnings.extend(_abstention_warnings(committee, current_weights))

    # 3. size (code), review (LLM, shrink only), risk engine (code, final say)
    sectors = sector_map(cfg, factsheets)
    review: RiskReview | None = None
    if held_by_outage:
        note = ("LLM OUTAGE: " + "; ".join(outage_reasons) +
                ". Holding the current book: an outage is not information about the stocks.")
        warnings.append(note)
        proposed = _hold_target(current_weights, bench, as_of, note)
        reviewed = proposed
        log("target_proposed", {"held_by_outage": True, "reasons": outage_reasons, "target": proposed})
    else:
        # A held name whose required memo failed keeps its current weight: its pooled view was
        # never reviewed (so it must not add exposure), and an LLM failure is not a reason to sell.
        unreviewed = sorted(t for t, r in committee.items() if r.hold_current)
        if unreviewed:
            warnings.append(f"required IC memo failed on held name(s) {', '.join(unreviewed)}: kept at the current "
                            "weight this cycle (an LLM failure is not a reason to buy or to sell)")
        proposed = construct_portfolio(aggregates, {t: r.edge_multiplier for t, r in committee.items()},
                                       factsheets, current_weights, cfg, as_of, sectors, returns=p.returns(),
                                       keep_current=unreviewed)
        log("target_proposed", {"held_by_outage": False, "target": proposed, "sectors": sectors,
                                "kept_unreviewed": unreviewed})
        review = review_portfolio(model, proposed, factsheets, cfg) if cfg.process.risk_review_enabled else None
        # The current book is passed so that a halt means what the reviewer was told: no trades
        # this cycle (the book is kept), never a liquidation of the active sleeve.
        reviewed = apply_risk_review(proposed, review, current_weights=current_weights, benchmark=bench)
        if review is not None and review.halt_trading:
            warnings.append(f"RISK REVIEW HALT: no trades from the committee this cycle, the current book is kept "
                            f"({review.summary})")
        log("risk_review", {"enabled": cfg.process.risk_review_enabled, "review": review, "target": reviewed})

    # Every NAV since the previous decision updates the high-water mark, and the worst one-day
    # loss among them can halt buying (P9): a crash between rebalances must not go unseen.
    held = {t for t, w in current_weights.items() if float(w) != 0.0}
    market = market_snapshot(p, as_of, {*committee_tickers, bench, *held, *(positions or {})})
    if positions is None:
        positions = {t: float(w) * nav / market.prices[t] for t, w in current_weights.items()
                     if float(w) != 0.0 and t in market.prices}
    if nav_marks is not None:
        path = [float(x) for x in nav_marks]
    else:
        path = _nav_path_since_last_decision(risk_state, p, positions, nav, as_of)
    state = risk_state
    for mark in path:
        state = update_risk_state(state, mark)
    state = replace(update_risk_state(state, nav), last_as_of=as_of)
    worst_day = _worst_day_return([*path, nav]) if path else None
    final, risk_messages = apply_risk_limits(reviewed, state, nav, None if path else prev_nav, current_weights, cfg,
                                             factsheets, runs_dir, worst_day_return=worst_day)
    log("risk_engine", {"messages": risk_messages, "target": final, "risk_state": state,
                        "nav_path_days": len(path), "worst_day_return": worst_day})

    # 4. orders at the close, to be filled at the next open (P3)
    orders, order_messages = orders_from_target(final, positions, market.prices, nav, market.adv, cfg, bench)
    log("orders", {"orders": orders, "messages": order_messages, "reference_prices": market.prices,
                   "adv": market.adv})

    # 5. every non-abstaining view becomes a resolvable forecast (P7): the RAW stated views, the
    # final probability ("committee") and the LLM pool ("llm_pool"); with the screen, also the
    # quant's forecast for every screened name that was not deep-dived (breadth for IC tests).
    made = [rec for t in committee for rec in
            make_forecast_records(run_id, as_of, committee[t].views, committee[t].aggregate, cfg, calendar)]
    if screen is not None and quant_seat:     # the quant's own forecasts (it sits on the committee)
        for t in tickers:
            q = qviews.get(t)
            if t in committee or q is None or q.view.abstain:
                continue
            solo = AggregatedView(ticker=t, prob_outperform=q.view.prob_outperform,
                                  prob_equal_weight=q.view.prob_outperform,
                                  expected_excess_return_pct=q.view.expected_excess_return_pct, dispersion=0.0,
                                  n_views=1, n_abstained=0)
            made += [r for r in make_forecast_records(run_id, as_of, [q], solo, cfg, calendar)
                     if r.agent == QUANT_AGENT]
    forecasts = forecast_store.add(made)
    if len(forecasts) < len(made):     # a re-decision of a session already decided (run --force)
        warnings.append(f"{len(made) - len(forecasts)} forecast(s) were not stored: this session was already "
                        "forecast by an earlier run, whose forecasts stand (no question is scored twice)")
    log("forecasts_written", {"n": len(forecasts), "ids": [r.id for r in forecasts],
                              "not_stored_repeats": len(made) - len(forecasts)})

    usage = _usage_summary(model.usage.calls[calls_before:])
    data_gaps = {t: list(factsheets[t].data_gaps) for t in committee if factsheets[t].data_gaps}
    log("cycle_end", {"n_orders": len(orders), "usage": usage, "warnings": warnings,
                      "held_by_outage": held_by_outage})
    return Decision(
        as_of=as_of, run_id=run_id, mode=mode, benchmark=bench, config_fingerprint=cfg.fingerprint(), nav=nav,
        committee=committee, aggregates=aggregates, believability=believability,
        target_proposed=proposed, target_reviewed=reviewed, target_final=final, risk_review=review,
        risk_messages=risk_messages, orders=orders, order_messages=order_messages, market=market,
        forecasts=forecasts, resolved=resolved, lessons_written=lessons_written, risk_state=state,
        usage=usage, warnings=warnings, data_gaps=data_gaps, held_by_outage=held_by_outage,
        ledger_head=ledger.head, lessons_used={a: len(v) for a, v in lessons.items()}, screen=screen,
        overlay_bound=bound, calibrators=calibrators, news=news,
    )
