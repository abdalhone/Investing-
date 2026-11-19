"""Walk-forward backtest: the live decision cycle replayed day by day, with honest accounting (P10).

What makes it honest, and why each rule exists:

* **Same code as live.** Every rebalance calls :func:`aifirm.pipeline.decide`, the exact
  function ``aifirm run`` uses. There is no separate "backtest logic" that could flatter
  the strategy.
* **Decide at the close, fill at the next open (P3).** Orders are sized at the close of
  the decision date and filled at the *open of the next trading day* by the
  :class:`~aifirm.execution.broker.PaperBroker`, paying commission, spread and
  square-root market impact. Same-close fills are the most common source of fake alpha
  in published LLM-trading results (FINSABER).
* **Point-in-time information (P3).** ``decide`` cuts prices at each decision date, the
  data loaders are called with the decision date, forecasts resolve only as time
  passes, and believability weights and lessons use outcomes resolved by the decision
  date only. Changing any price after a date cannot change a decision on that date (a
  test perturbs future prices to prove it).
* **Daily mark to market** at the close, so drawdowns between rebalances are visible,
  and every daily NAV reaches the risk engine: at each decision ``decide`` receives the
  NAV path since the previous one, which raises the high-water mark and feeds the worst
  one-day loss to the daily-loss halt (P9).
* **Total-return accounting.** Books are marked at raw closes and fill at raw opens, and
  each ex-date's cash dividend (inferred from the adjusted vs raw closes,
  :func:`aifirm.pipeline.implied_dividends`) is credited to the paper broker for the
  shares held at the previous close. The benchmark baseline is built from the *same*
  raw closes and dividend amounts, so a book that holds the benchmark earns exactly the
  benchmark's return before costs, and forecasts (scored on total return) and P&L agree.
* **Baselines through the same engine.** Benchmark buy-and-hold (no costs: it is the
  investable alternative), an equal-weight universe rebalanced on the same schedule with
  the same costs, and (unless the run *is* quant-only) the deterministic quant analyst
  alone. If an LLM committee cannot beat the quant-only run, the LLMs add nothing.
* **Turnover** is reported one-way: half the traded notional over NAV per fill date, the
  same definition ``max_turnover`` caps.
* **Training contamination (P10).** An LLM may have memorised what happened in the
  period covered by its training data, so a decision made on or before the model's
  training cutoff can "predict" an outcome it has already seen. Returns earned on such
  decisions are reported separately as *contaminated* and never count as evidence of
  skill. An unknown cutoff is treated as "today" (everything contaminated).
* **Multiple testing.** Each run is appended to ``<runs_dir>/trials.jsonl``; the number of
  distinct (config fingerprint, mode, start, end) trials feeds the deflated Sharpe ratio
  (Bailey and Lopez de Prado). Trying many variants, or one variant over many windows,
  and reporting the best is exactly what DSR corrects for, so the trial ledger must be
  complete to be meaningful.

* **Exposure is visible.** The fraction of NAV invested (positions, core included) is
  recorded every day (``BacktestResult.exposure``) and each decision records its target
  active gross, core and cash, so a result driven by how much cash the book held is
  never mistaken for stock selection.
* **The screen runs point-in-time.** With ``universe.screen`` enabled every decision runs
  the same screen as live (:func:`aifirm.pipeline.decide`): prices, liquidity and price
  filters cut at the decision close. The universe *list* is not point-in-time (see below).

Known limitations (also stated in the report): a universe chosen today is biased toward
survivors (a bundled list of current index members makes this explicit); costs are modelled, not observed; no taxes, withholding or borrow fees; cash
distributions are inferred from Yahoo's adjusted closes (ordinary dividends and other
distributions alike), not read from a corporate-actions feed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import logging
import math
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TypeVar

import pandas as pd

from ..agents import CommitteeResult, llm_analysts, make_offline_llm
from ..config import FirmConfig, model_cutoff
from ..data.factsheet import FactSheet
from ..data.prices import PriceData, load_prices
from ..execution.broker import PaperBroker
from ..journal import ForecastStore, Ledger, LessonStore, scorecard
from ..llm.client import LLM
from ..universes import SURVIVORSHIP_NOTE
from ..pipeline import (PRICE_LOOKBACK_DAYS, FilingsLoader, FundamentalsLoader, MacroLoader, MarketSnapshot, decide,
                        gather_data, implied_dividends, market_snapshot, quant_only_config, sector_map, universe)
from ..portfolio import RiskState, orders_from_target
from ..schemas import Order, TargetPortfolio, TargetPosition
from .evaluation import Evaluation, cost_sensitivity
from .factors import PROXY_TICKERS, FactorAttribution, factor_attribution
from .ic import FORWARD_RETURN_DEFINITION, POOL, information_coefficients, power_analysis
from .metrics import PerformanceReport, performance_report
from .random_books import build_schedule, random_books, realised_cost_rate

MODES = ("quant", "offline", "llm")
_T = TypeVar("_T")
MIN_REPORT_DAYS = 10            # fewer daily returns and a sub-period report says nothing
TRIALS_FILE = "trials.jsonl"
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

log = logging.getLogger(__name__)

# policy(decision date, broker, nav at that close, previous close nav, marks, nav path) -> (orders, market, record)
# The nav path is the closing NAV of every day since the previous decision (that decision's NAV
# first), excluding today; empty at the first decision.
Policy = Callable[[pd.Timestamp, PaperBroker, float, float | None, dict[str, float], list[float]],
                  tuple[list[Order], MarketSnapshot, dict]]


def _day(d: object) -> str:
    return str(pd.Timestamp(d).date())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- result


@dataclass(frozen=True)
class OverlayValue:
    """What the language-model stages added over the quant-only baseline, net of what they
    cost (RESEARCH 6.6: "let measured net-of-cost improvement set the overlay's weight").

    Both books start with the same cash and run through the same engine, universe, screen,
    limits and costs; the only difference is the LLM stages. ``difference_usd`` is the
    strategy's final NAV minus the quant-only baseline's; LLM spend is not charged to the
    simulated book, so ``value_added_usd = difference_usd - llm_cost_usd``;
    ``value_added_pct`` is that in percent of the starting NAV. ``offline``: deterministic
    rules stood in for the LLM (cost 0: a plumbing check, not a measurement of an LLM)."""

    initial_nav: float
    strategy_nav: float
    quant_nav: float
    difference_usd: float
    llm_cost_usd: float
    value_added_usd: float
    value_added_pct: float
    offline: bool


@dataclass
class BacktestResult:
    """Everything a backtest produced. Return series are daily, net of costs.

    ``reports`` holds a :class:`PerformanceReport` for ``"strategy"`` and each baseline
    (``"benchmark"``, ``"equal_weight"`` and, unless ``mode == "quant"``, ``"quant_only"``).
    ``model_cutoff`` is the latest training cutoff of the models the run used (None in
    quant mode); ``contamination_end`` is that cutoff plus ``cutoff_buffer_days``.
    ``pre_cutoff_report`` / ``post_cutoff_report`` split the strategy's returns by whether
    the decision behind them was made on or before ``contamination_end``.
    ``evaluation`` answers "is there skill, and would we know?" (signal IC, power,
    composition-matched random books, cost sensitivity, factor attribution; see
    :mod:`aifirm.backtest.evaluation`); None only if it was switched off by the caller.
    """

    run_id: str
    mode: str
    start: str
    end: str
    config_fingerprint: str
    run_dir: str
    nav: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    baseline_returns: dict[str, pd.Series]
    turnover: pd.Series                 # one-way: 0.5 * traded notional / NAV, per fill date
    costs: pd.Series                    # commission + slippage / NAV, per fill date
    decisions: list[dict]
    reports: dict[str, PerformanceReport]
    pre_cutoff_report: PerformanceReport | None
    post_cutoff_report: PerformanceReport | None
    model_cutoff: str | None
    n_trials: int
    scorecard: dict[str, dict]
    usage: dict
    costs_usd: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    dividends_usd: float = 0.0          # net dividend cash credited to the strategy book
    benchmark: str = ""
    tickers: list[str] = field(default_factory=list)          # the active universe (benchmark excluded)
    models: list[str] = field(default_factory=list)           # language models the run's LLM stages used
    contamination_end: str | None = None                      # model_cutoff + cutoff_buffer_days
    cutoff_buffer_days: int = 0
    cost_model: dict[str, float] = field(default_factory=dict)  # commission_bps, slippage_bps, impact_coefficient
    trials_path: str = ""
    evaluation: Evaluation | None = None
    # Daily fraction of NAV held in positions (core included) at the close; 1 - this is cash.
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    universe_note: str = ""                                   # survivorship caveat of the universe list, if any
    # Daily NAV of each traded baseline ("equal_weight", and "quant_only" unless mode == "quant").
    baseline_nav: dict[str, pd.Series] = field(default_factory=dict)

    def overlay_value(self) -> OverlayValue | None:
        """The LLM stages' value added over the quant-only baseline net of LLM cost (see
        :class:`OverlayValue`); None in quant mode (no baseline to compare with) or when a
        NAV series is empty."""
        q = self.baseline_nav.get("quant_only")
        if q is None or q.empty or self.nav.empty:
            return None
        start, s_end, q_end = float(self.nav.iloc[0]), float(self.nav.iloc[-1]), float(q.iloc[-1])
        cost = float(self.usage.get("cost_usd", 0.0) or 0.0)
        diff = s_end - q_end
        return OverlayValue(initial_nav=start, strategy_nav=s_end, quant_nav=q_end, difference_usd=diff,
                            llm_cost_usd=cost, value_added_usd=diff - cost,
                            value_added_pct=(diff - cost) / start * 100.0 if start > 0 else math.nan,
                            offline=self.mode == "offline")

    def exposure_summary(self) -> dict[str, float]:
        """Averages over the run: ``invested`` (daily, positions incl. core / NAV), ``cash``
        (1 - invested), and from the decisions' final targets ``target_active_gross``,
        ``target_core``, ``target_cash`` and ``target_names`` (active names per decision).
        NaN where nothing was recorded."""
        def mean(xs: list[float]) -> float:
            xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
            return sum(xs) / len(xs) if xs else math.nan
        inv = float(self.exposure.mean()) if len(self.exposure) else math.nan
        return {"invested": inv, "cash": 1.0 - inv if math.isfinite(inv) else math.nan,
                "target_active_gross": mean([d.get("active_gross") for d in self.decisions]),
                "target_core": mean([d.get("core_weight") for d in self.decisions]),
                "target_cash": mean([d.get("cash_weight") for d in self.decisions]),
                "target_names": mean([d.get("n_names") for d in self.decisions])}


# --------------------------------------------------------------------------- simulation core


@dataclass
class _Sim:
    nav: pd.Series
    turnover: pd.Series
    costs: pd.Series
    decisions: list[dict]
    decision_dates: list[pd.Timestamp]
    commission_usd: float
    slippage_usd: float
    dividends_usd: float = 0.0
    invested: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))   # 1 - cash / NAV at each close


def _simulate(prices: PriceData, cfg: FirmConfig, days: pd.DatetimeIndex, rebalance_days: set[pd.Timestamp],
              policy: Policy) -> _Sim:
    """The daily loop shared by the strategy and every traded baseline.

    Each trading day: (1) cash dividends with an ex-date today are credited for the shares
    held at the previous close; (2) orders decided at the previous rebalance close fill at
    today's open; (3) the book is marked at today's close; (4) on a rebalance day the policy
    decides at today's close, seeing the NAV path since its previous decision. The last day
    never decides (there is no next open)."""
    broker = PaperBroker(cfg.execution.initial_cash, cfg.costs)
    close_ff = prices.close.ffill()           # a missing close is marked at the last known close
    opens = prices.open if prices.open is not None else prices.close
    dividends = implied_dividends(prices)
    navs: dict[pd.Timestamp, float] = {}
    invested: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    costs: dict[pd.Timestamp, float] = {}
    records: list[dict] = []
    decision_dates: list[pd.Timestamp] = []
    pending: tuple[list[Order], MarketSnapshot, float, dict] | None = None
    prev_nav: float | None = None
    since_decision: list[float] = []          # closing NAVs from the last decision through yesterday
    for i, d in enumerate(days):
        if i > 0 and broker.positions():      # the holder at the previous close receives the dividend
            row_div = dividends.loc[d]
            broker.credit_dividends({t: float(row_div[t]) for t in broker.positions()
                                     if t in row_div.index and row_div[t] > 0}, _day(d))
        if pending is not None:
            orders, market, decision_nav, rec = pending
            fill_px = {}
            for o in orders:
                px = opens.at[d, o.ticker] if o.ticker in opens.columns else float("nan")
                if math.isfinite(px) and px > 0:
                    fill_px[o.ticker] = float(px)
            fills = broker.submit(orders, fill_px, _day(d), market.adv, market.daily_vol)
            turnover[d] = 0.5 * sum(f.quantity * f.price for f in fills) / decision_nav     # one-way
            costs[d] = sum(f.commission + f.slippage_cost for f in fills) / decision_nav
            rec["fill_date"] = _day(d)
            rec["fills"] = [f.model_dump() for f in fills]
            rec["rejections"] = list(broker.last_rejections)
            pending = None
        row = close_ff.loc[d]
        marks = {t: float(v) for t, v in row.items() if math.isfinite(v) and v > 0}
        nav = broker.nav(marks)
        navs[d] = nav
        if math.isfinite(nav) and nav > 0:
            invested[d] = 1.0 - broker.cash() / nav
        if d in rebalance_days and i + 1 < len(days):
            orders, market, rec = policy(d, broker, nav, prev_nav, marks, list(since_decision))
            rec.update({"as_of": _day(d), "nav": nav,
                        "orders": [{"ticker": o.ticker, "side": o.side, "quantity": o.quantity,
                                    "notional": o.notional} for o in orders]})
            records.append(rec)
            decision_dates.append(d)
            since_decision = []
            if orders:
                pending = (orders, market, nav, rec)
        since_decision.append(nav)
        prev_nav = nav
    return _Sim(nav=pd.Series(navs, dtype=float), turnover=pd.Series(turnover, dtype=float),
                costs=pd.Series(costs, dtype=float), decisions=records, decision_dates=decision_dates,
                commission_usd=broker.total_commission, slippage_usd=broker.total_slippage,
                dividends_usd=broker.total_dividends, invested=pd.Series(invested, dtype=float))


def _total_returns(prices: PriceData, ticker: str, days: pd.DatetimeIndex) -> pd.Series:
    """Daily total returns of ``ticker`` on ``days[1:]`` from raw closes plus the same implied
    dividends the paper broker credits: ``(close_t + D_t) / close_{t-1} - 1``. A missing
    print is forward-filled (the move across a gap lands on the next print instead of
    being dropped), so a book holding ``ticker`` and this series agree exactly."""
    close = prices.close[ticker].ffill()
    div = implied_dividends(prices)[ticker]
    tr = (close + div) / close.shift(1) - 1.0
    return tr.reindex(days).iloc[1:].fillna(0.0)


def _agent_probs(committee: Mapping[str, CommitteeResult],
                 screen_quant: Mapping[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Ticker -> agent -> sealed P(outperform), the mean over that agent's non-abstaining
    samples (the analyst's independent forecast, before any challenge round). With a screen,
    ``screen_quant`` adds the quant's probability for screened names outside the committee,
    so the quant's IC is measured over the whole universe."""
    out: dict[str, dict[str, float]] = {}
    for t, r in committee.items():
        by_agent: dict[str, list[float]] = {}
        for v in r.views:
            if not v.view.abstain:
                by_agent.setdefault(v.agent, []).append(float(v.view.prob_outperform))
        out[t] = {a: sum(ps) / len(ps) for a, ps in by_agent.items()}
    for t, q in (screen_quant or {}).items():
        out.setdefault(t, {"quant": float(q)})
    return out


class _SheetCache:
    """Point-in-time fact sheets built once per decision date and shared by every policy of
    one backtest (the strategy and the quant-only baseline).

    In a backtest :func:`gather_data` runs with ``live=False`` (no headlines), so the sheets
    depend only on the decision date, the prices, the loaders and ``cfg.universe``, which
    :func:`quant_only_config` keeps: the baseline's sheets are the strategy's, byte for
    byte. Building them (SEC fundamentals for every screened name) is the dominant cost of a
    screened backtest, so building them twice doubled it. The key includes the universe
    section, so two configurations with different universes never share sheets, and each
    caller gets a deep copy, so nothing one policy does to a sheet can reach another."""

    def __init__(self, prices: PriceData, loaders: dict) -> None:
        self.prices, self.loaders = prices, loaders
        self._sheets: dict[tuple[str, str], dict[str, FactSheet]] = {}
        self.builds = 0

    def get(self, cfg: FirmConfig, as_of: str) -> dict[str, FactSheet]:
        key = (as_of, json.dumps(cfg.universe.model_dump(), sort_keys=True, default=str))
        if key not in self._sheets:
            _, sheets = gather_data(cfg, as_of, live=False, prices=self.prices, **self.loaders)
            self._sheets[key] = sheets
            self.builds += 1
        return copy.deepcopy(self._sheets[key])


class _CommitteePolicy:
    """The firm itself: gather point-in-time data and run :func:`decide` at each rebalance."""

    def __init__(self, cfg: FirmConfig, llm: LLM | None, prices: PriceData, run_dir: Path, run_id: str,
                 loaders: dict, sheets: _SheetCache | None = None) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.llm, self.prices, self.loaders = cfg, llm, prices, loaders
        self.sheets = sheets if sheets is not None else _SheetCache(prices, loaders)
        self.run_dir, self.run_id = run_dir, run_id
        self.store = ForecastStore(run_dir / "forecasts.sqlite")
        self.lessons = LessonStore(run_dir / "lessons.jsonl")
        self.ledger = Ledger(run_dir / "ledger.jsonl")
        self.state: RiskState | None = None
        self.cycles = 0
        self._warning_counts: Counter[str] = Counter()     # insertion-ordered: first seen first
        self.final_warnings: list[str] = []

    def __call__(self, d: pd.Timestamp, broker: PaperBroker, nav: float, prev_nav: float | None,
                 marks: dict[str, float], nav_path: list[float]) -> tuple[list[Order], MarketSnapshot, dict]:
        as_of = _day(d)
        factsheets = self.sheets.get(self.cfg, as_of)
        if self.state is None:
            self.state = RiskState(peak_nav=nav_path[0] if nav_path else nav)
        held = broker.positions()
        current = {t: q * marks[t] / nav for t, q in held.items() if t in marks}
        dec = decide(self.cfg, self.llm, as_of, self.prices, factsheets, current_weights=current, nav=nav,
                     prev_nav=prev_nav, risk_state=self.state, forecast_store=self.store, lesson_store=self.lessons,
                     ledger=self.ledger, run_id=self.run_id, positions=held,
                     trading_calendar=self.prices.close.index, runs_dir=str(self.run_dir), nav_marks=nav_path)
        self.state = dec.risk_state
        self.cycles += 1
        messages = [*dec.warnings, *(f"{t}: {g}" for t, gaps in dec.data_gaps.items() for g in gaps)]
        self._warning_counts.update(dict.fromkeys((_ISO_DATE.sub("<date>", w) for w in messages), 1))
        screen_quant = dec.screen.quant_probs() if dec.screen is not None else {}
        outside = {t: q for t, q in screen_quant.items() if t not in dec.aggregates}
        exposure = dec.exposure()
        rec = {
            # the firm's final probability for every name it scored (committee names: the final pooled or
            # overlay probability; screened names outside the deep-dive set: the quant's)
            "pooled_prob": dec.final_probabilities(),
            # for the evaluation (signal IC per agent and per pool, sector-matched null); never read by decide
            "prob_equal_weight": {**outside, **{t: a.prob_equal_weight for t, a in dec.aggregates.items()}},
            "n_views": {**{t: 1 for t in outside}, **{t: a.n_views for t, a in dec.aggregates.items()}},
            "agent_probs": _agent_probs(dec.committee, outside),
            "deep_dive": list(dec.committee),
            "screen": ({"n_universe": dec.screen.n_universe, "n_eligible": dec.screen.n_eligible}
                       if dec.screen is not None else None),
            "overlay": {t: {"p_quant": a.prob_quant, "p_llm": a.prob_llm, "tilt": a.overlay_tilt}
                        for t, a in dec.aggregates.items() if a.combination == "overlay"},
            "overlay_max_tilt": dec.overlay_bound.max_tilt if dec.overlay_bound is not None else None,
            "active_gross": exposure["active_gross"],
            "n_names": exposure["n_names"],
            "sectors": sector_map(self.cfg, factsheets),
            "edge_multiplier": {t: r.edge_multiplier for t, r in dec.committee.items()},
            "challenged": sorted(t for t, a in dec.aggregates.items() if a.challenged),
            "target": dec.target_final.weights(),
            "core_weight": dec.target_final.core_weight,
            "cash_weight": dec.target_final.cash_weight,
            "risk_messages": dec.risk_messages,
            "held_by_outage": dec.held_by_outage,
            "frozen": dec.risk_state.frozen,
            "n_forecasts": len(dec.forecasts),
            "n_resolved": len(dec.resolved),
            "n_lessons": len(dec.lessons_written),
            "llm_cost_usd": dec.usage.get("cost_usd", 0.0),
        }
        return dec.orders, dec.market, rec

    @property
    def warnings(self) -> list[str]:
        """Distinct warnings and data gaps (dates masked) with how many decisions they affected,
        then anything found when the run was scored at its end."""
        return [*(f"{w} ({n} of {self.cycles} decisions)" for w, n in self._warning_counts.items()),
                *self.final_warnings]

    def run(self, prices: PriceData, days: pd.DatetimeIndex, rebalance_days: set[pd.Timestamp],
            end: str) -> tuple[_Sim, dict[str, dict]]:
        """Simulate, then resolve what can be resolved by ``end`` and score the forecasts.
        The forecast store is closed even if the simulation fails."""
        try:
            sim = _simulate(prices, self.cfg, days, rebalance_days, self)
            self.store.resolve(prices, end)
            for why, n in Counter(self.store.unresolvable.values()).items():
                self.final_warnings.append(f"{n} forecast(s) due by {end} could not be resolved: {why}")
            return sim, scorecard(self.store.all(), as_of=end)
        finally:
            self.store.close()


def _equal_weight_policy(cfg: FirmConfig, prices: PriceData) -> Policy:
    """Equal weight over every universe name with a price, cash floor kept, no views."""
    bench = cfg.universe.benchmark.upper()
    names_all = universe(cfg)

    def policy(d: pd.Timestamp, broker: PaperBroker, nav: float, prev_nav: float | None,
               marks: dict[str, float], nav_path: list[float]) -> tuple[list[Order], MarketSnapshot, dict]:
        market = market_snapshot(prices, _day(d), {*names_all, bench, *broker.positions()})
        names = [t for t in names_all if t in market.prices]
        w = (1.0 - cfg.portfolio.min_cash) / len(names) if names else 0.0
        target = TargetPortfolio(as_of=_day(d), core_weight=0.0, cash_weight=1.0 - w * len(names),
                                 positions=[TargetPosition(ticker=t, weight=w, prob_outperform=0.5,
                                                           expected_excess_return_pct=0.0, rationale="equal weight")
                                            for t in names])
        orders, _ = orders_from_target(target, broker.positions(), market.prices, nav, market.adv, cfg, bench)
        return orders, market, {"target": target.weights()}

    return policy


# --------------------------------------------------------------------------- re-execution at other costs


def scaled_costs(cfg: FirmConfig, k: float) -> FirmConfig:
    """A copy of ``cfg`` with every cost component (commission, spread/slippage, impact
    coefficient) multiplied by ``k >= 0``. Nothing else changes."""
    if not (math.isfinite(k) and k >= 0):
        raise ValueError(f"cost multiplier must be finite and >= 0, got {k!r}")
    out = cfg.model_copy(deep=True)
    out.costs.commission_bps *= k
    out.costs.slippage_bps *= k
    out.costs.impact_coefficient *= k
    return out


class _ReplayPolicy:
    """Re-executes the recorded final targets of a finished run (no new decision, no LLM
    call): at each rebalance close the recorded target (active weights, core, cash) is
    turned into orders by :func:`orders_from_target` against the replayed book, exactly as
    :func:`decide` did. With the run's own configuration it reproduces the run's NAV path;
    with :func:`scaled_costs` it shows what the SAME decisions earn at other costs.
    Market snapshots do not depend on costs, so they are shared through ``markets``."""

    def __init__(self, cfg: FirmConfig, prices: PriceData, decisions: list[dict],
                 markets: dict[tuple[str, frozenset[str]], MarketSnapshot]) -> None:
        self.cfg, self.prices, self.markets = cfg, prices, markets
        self.by_date = {d["as_of"]: d for d in decisions}
        self.bench = cfg.universe.benchmark.upper()
        self.names = universe(cfg)

    def __call__(self, d: pd.Timestamp, broker: PaperBroker, nav: float, prev_nav: float | None,
                 marks: dict[str, float], nav_path: list[float]) -> tuple[list[Order], MarketSnapshot, dict]:
        as_of = _day(d)
        rec = self.by_date.get(as_of)
        if rec is None or "core_weight" not in rec:
            raise ValueError(f"no recorded target for {as_of}: cannot replay a decision that was not made")
        held = broker.positions()
        tickers = frozenset({*self.names, self.bench, *held, *rec["target"]})
        key = (as_of, tickers)
        if key not in self.markets:
            self.markets[key] = market_snapshot(self.prices, as_of, tickers)
        market = self.markets[key]
        target = TargetPortfolio(
            as_of=as_of, core_weight=float(rec["core_weight"]), cash_weight=float(rec["cash_weight"]),
            positions=[TargetPosition(ticker=t, weight=float(w), prob_outperform=0.5, expected_excess_return_pct=0.0,
                                      rationale="replay of a recorded decision") for t, w in rec["target"].items()])
        orders, _ = orders_from_target(target, held, market.prices, nav, market.adv, self.cfg, self.bench)
        return orders, market, {}


def replay_total_return(prices: PriceData, cfg: FirmConfig, days: pd.DatetimeIndex,
                        rebalance_days: set[pd.Timestamp], decisions: list[dict], k: float = 1.0,
                        markets: dict[tuple[str, frozenset[str]], MarketSnapshot] | None = None) -> float:
    """Total return in percent of ``decisions`` (a finished run's records, made with
    ``cfg``) re-executed through the same engine with costs multiplied by ``k``."""
    policy = _ReplayPolicy(scaled_costs(cfg, k), prices, decisions, markets if markets is not None else {})
    nav = _simulate(prices, policy.cfg, days, rebalance_days, policy).nav
    return float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0)


# --------------------------------------------------------------------------- evaluation helpers


def _factor_proxy_prices(cfg: FirmConfig, prices: PriceData, factor_prices: PriceData | None, downloaded: bool,
                         start: str, end: str, notes: list[str]) -> PriceData | None:
    """Where factor-proxy prices come from: ``factor_prices`` if given; else ``prices`` when
    it holds every proxy; else, when run_backtest downloaded ``prices`` itself, a download
    of the proxies; else (prices passed in, as offline tests do: never a network call)
    whatever proxies ``prices`` holds, typically just the benchmark. A failed download is
    a note, never an error."""
    if factor_prices is not None:
        return factor_prices
    have = [t for t in PROXY_TICKERS if t in prices.adj_close.columns]
    if len(have) == len(PROXY_TICKERS) or not downloaded:
        if not have:
            notes.append("factor attribution skipped: the prices passed in hold no factor proxies "
                         f"({', '.join(PROXY_TICKERS)}); pass factor_prices")
            return None
        return prices
    try:
        return load_prices(list(PROXY_TICKERS), pd.Timestamp(start) - pd.Timedelta(days=10), end)
    except Exception as exc:  # noqa: BLE001 - proxies are optional: missing ones are reported, never fatal
        notes.append(f"factor attribution skipped: proxy prices could not be loaded ({type(exc).__name__}: {exc})")
        return None


def _guarded(part: str, notes: list[str], fn: Callable[[], _T]) -> _T | None:
    """Run one evaluation part; a failure becomes None plus a note (fail closed: a part
    that could not be computed is reported as missing, never as a result)."""
    try:
        return fn()
    except Exception as exc:  # parts are independent; one failure must not hide the others
        log.warning("backtest evaluation: %s failed", part, exc_info=True)
        notes.append(f"{part} could not be computed ({type(exc).__name__}: {exc})")
        return None


def _evaluate(*, cfg: FirmConfig, strat_cfg: FirmConfig, prices: PriceData, days: pd.DatetimeIndex,
              rebalance_days: set[pd.Timestamp], end: str, sim: _Sim, q_sim: _Sim | None, returns: pd.Series,
              bench_total_pct: float, factor_prices: PriceData | None, downloaded: bool, start: str,
              contamination_end: str | None = None, clean_returns: pd.Series | None = None) -> Evaluation:
    """Compute every part of :class:`Evaluation` from a finished run (no new decisions).

    With ``contamination_end`` (the run's training cutoff plus buffer) the skill evidence
    uses only decisions made after it and ``clean_returns`` (the returns earned under those
    decisions); see :class:`Evaluation`."""
    ev = cfg.evaluation
    proc = strat_cfg.process
    bench = cfg.universe.benchmark.upper()
    names = universe(cfg)
    out = Evaluation(horizon_days=proc.horizon_days, step_days=proc.rebalance_every_days,
                     forward_return_definition=FORWARD_RETURN_DEFINITION)
    notes = out.notes
    engine_total = float((sim.nav.iloc[-1] / sim.nav.iloc[0] - 1.0) * 100.0)

    def clean(decisions: list[dict]) -> list[dict]:
        return decisions if contamination_end is None else [d for d in decisions if d["as_of"] > contamination_end]

    decisions = clean(sim.decisions)
    q_decisions = clean(q_sim.decisions) if q_sim is not None else None
    skill_returns = returns if contamination_end is None or clean_returns is None else clean_returns
    out.clean_after, out.n_decisions, out.n_clean_decisions = contamination_end, len(sim.decisions), len(decisions)
    if contamination_end is not None and len(decisions) < len(sim.decisions):
        full = _guarded("signal IC (whole window, contaminated)", notes, lambda: information_coefficients(
            sim.decisions, prices, names, bench, proc.horizon_days, proc.rebalance_every_days, end, ev.ic_min_names,
            quant_decisions=q_sim.decisions if q_sim is not None else None))
        if full is not None:
            out.contaminated_ic, out.contaminated_ic_differences = full
        if decisions:
            notes.append(f"skill evidence (IC, power, random books, factor alpha) uses the {len(decisions)} decisions "
                         f"made after {contamination_end} only; the {len(sim.decisions) - len(decisions)} earlier "
                         "ones may rest on memorised outcomes (P10)")
        else:
            notes.append(f"no decision was made after {contamination_end} (training cutoff plus buffer): IC, power, "
                         "random books and factor alpha have no clean dates; the whole-window IC includes "
                         "contaminated decisions and is not evidence of skill")

    if decisions:
        ic = _guarded("signal IC", notes, lambda: information_coefficients(
            decisions, prices, names, bench, proc.horizon_days, proc.rebalance_every_days, end, ev.ic_min_names,
            quant_decisions=q_decisions))
        if ic is not None:
            out.ic, out.ic_differences = ic
            if POOL in out.ic:
                out.power = _guarded("power analysis", notes,
                                     lambda: power_analysis(out.ic[POOL], proc.rebalance_every_days, skill_returns))

    rate = realised_cost_rate(sim.decisions)
    if ev.n_random_books > 0 and decisions:
        sectors: dict[str, str] = {}
        for rec in sim.decisions:
            sectors.update(rec.get("sectors") or {})
        # The schedule holds the clean decisions only: compare it with the engine over the same
        # period (from the first clean decision's close), not with the whole contaminated run.
        rb_engine, rb_from = engine_total, None
        if len(decisions) < len(sim.decisions):
            before = sim.nav[sim.nav.index <= pd.Timestamp(decisions[0]["as_of"])]
            if len(before):
                rb_engine, rb_from = float((sim.nav.iloc[-1] / before.iloc[-1] - 1.0) * 100.0), decisions[0]["as_of"]
        out.random_books = _guarded("random books", notes, lambda: random_books(
            build_schedule(decisions, prices, days, names, bench, sectors), prices, ev.n_random_books,
            ev.random_seed, rate, rb_engine, sector_matched=ev.random_books_sector_matched,
            engine_period_start=rb_from))

    markets: dict[tuple[str, frozenset[str]], MarketSnapshot] = {}
    q_markets: dict[tuple[str, frozenset[str]], MarketSnapshot] = {}
    q_cfg = quant_only_config(cfg)
    out.costs = _guarded("cost sensitivity", notes, lambda: cost_sensitivity(
        lambda k: replay_total_return(prices, strat_cfg, days, rebalance_days, sim.decisions, k, markets),
        bench_total_pct, list(ev.cost_multipliers), ev.breakeven_cap, rate * 1e4, engine_total,
        replay_quant=(None if q_sim is None else
                      lambda k: replay_total_return(prices, q_cfg, days, rebalance_days, q_sim.decisions, k,
                                                    q_markets))))
    if out.costs is not None and not out.costs.replay_matches_engine:
        notes.append("re-executing the recorded decisions at 1x costs did not reproduce the engine's NAV; the cost "
                     "sensitivity is approximate")
    if out.costs is not None and contamination_end is not None and len(decisions) < len(sim.decisions):
        notes.append("the cost sensitivity and breakeven re-execute every recorded decision, contaminated ones "
                     "included: they describe the whole run, not clean skill")

    if ev.factor_proxies:
        def factors() -> FactorAttribution | None:
            fp = _factor_proxy_prices(cfg, prices, factor_prices, downloaded, start, end, notes)
            result, why = factor_attribution(skill_returns, fp, invested=sim.invested)
            notes.extend(n for n in why if n not in notes)
            return result
        out.factors = _guarded("factor attribution", notes, factors)
    return out


def run_models(cfg: FirmConfig, mode: str) -> list[str]:
    """The language models the run's LLM stages call, sorted ([] in quant mode or when no
    stage uses a model). In offline mode these are the models the offline rules stand in
    for: the configuration, not the rules, decides what a real run would have seen."""
    if mode == "quant":
        return []
    models: set[str] = set()
    if llm_analysts(cfg):
        models |= set(cfg.models.analyst_models)
    if cfg.process.memo_enabled or llm_analysts(cfg):
        models.add(cfg.models.committee_model)     # the chair and the lesson writer
    if cfg.process.risk_review_enabled:
        models.add(cfg.models.risk_model)
    return sorted(models)


def training_cutoff(cfg: FirmConfig, mode: str) -> str | None:
    """The latest training cutoff among :func:`run_models` (None without a model). An
    unknown cutoff counts as today (fail closed: every past period is contaminated)."""
    models = run_models(cfg, mode)
    if not models:
        return None
    cutoffs = [model_cutoff(m) for m in models]
    if any(c is None for c in cutoffs):
        return str(date.today())
    return max(c for c in cutoffs if c is not None)


def contamination_cutoff(cfg: FirmConfig, mode: str, buffer_days: int | None = None) -> str | None:
    """The last decision date whose outcomes the run's language models may have seen.

    None in quant mode (no model). Otherwise :func:`training_cutoff` plus ``buffer_days``
    calendar days (default ``cfg.evaluation.cutoff_buffer_days``): measured cutoffs differ
    from vendor dates by months (research report 7.2), so clean evaluation starts only
    after a buffer. Decisions on or before the returned date are contaminated."""
    cutoff = training_cutoff(cfg, mode)
    if cutoff is None:
        return None
    days = cfg.evaluation.cutoff_buffer_days if buffer_days is None else int(buffer_days)
    if days < 0:
        raise ValueError(f"buffer_days must be >= 0, got {days}")
    return _day(pd.Timestamp(cutoff) + pd.Timedelta(days=days))


def _decision_in_force(return_days: pd.DatetimeIndex, decision_dates: list[pd.Timestamp]) -> pd.Series:
    """For each return day, the latest decision date strictly before it (NaT if none)."""
    dd = pd.DatetimeIndex(sorted(decision_dates))
    pos = dd.searchsorted(return_days, side="left") - 1
    vals = [dd[i] if i >= 0 else pd.NaT for i in pos]
    return pd.Series(vals, index=return_days)


def _report_or_none(r: pd.Series, bench: pd.Series, n_trials: int) -> PerformanceReport | None:
    r = r.dropna()
    if len(r) < MIN_REPORT_DAYS:
        return None
    return performance_report(r, bench.reindex(r.index), n_trials=n_trials)


# Modules whose source decides what the language models are asked and how their answers are
# used: editing a prompt is a strategy change even though it lives in code, not in the config.
PROMPT_MODULES = ("aifirm.agents.analysts", "aifirm.agents.committee", "aifirm.agents.risk_review",
                  "aifirm.agents.reflection", "aifirm.agents.news_reader", "aifirm.agents.roles",
                  "aifirm.agents.tier_gate")


def prompt_digest() -> str:
    """Short hash of the source of :data:`PROMPT_MODULES` (the prompts, the evidence rules,
    the role slices and the code that reads the answers). It is part of an LLM-stage trial's
    identity in the trial ledger, so a prompt edit counts as a new configuration for the
    deflated Sharpe even though the config fingerprint does not see it. Any edit to those
    modules counts, comments included (the conservative direction: more trials, a higher bar)."""
    h = hashlib.sha256()
    for name in PROMPT_MODULES:
        h.update(name.encode())
        h.update(inspect.getsource(importlib.import_module(name)).encode())
    return h.hexdigest()[:16]


def _trial_key(fingerprint: str, mode: str, start: str, end: str, prompts: str = "") -> tuple[str, ...]:
    return (fingerprint, mode, start, end, prompts if mode != "quant" else "")


def _count_trials(path: Path, fingerprint: str, mode: str, start: str, end: str, prompts: str = "") -> int:
    """Distinct (fingerprint, mode, start, end, prompts) trials in the trial ledger, including this one.

    The backtest window is part of the trial: re-running one configuration over many
    windows and keeping the best is the same selection DSR corrects for. Re-running the
    identical (configuration, mode, window) is not a new trial. Every ledger row carries
    its window, so older rows count correctly too. ``prompts`` (:func:`prompt_digest`)
    identifies the prompt code of an offline or llm run (ignored in quant mode, which sends
    no prompt); rows written before it was recorded count with an empty digest."""
    trials = {_trial_key(fingerprint, mode, start, end, prompts)}
    if path.exists():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                trials.add(_trial_key(str(entry["fingerprint"]), str(entry["mode"]), str(entry["start"]),
                                      str(entry["end"]), str(entry.get("prompts") or "")))
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("%s line %d is not a valid trial record; skipped", path, n)
    return len(trials)


def _append_trial(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, allow_nan=False, default=str) + "\n")


def _screen_mismatch(strategy: list[dict], quant: list[dict]) -> str:
    """A warning when the quant-only baseline did not screen the same universe as the
    strategy on the same dates (same names priced, same names eligible), else "". Both use
    the configuration's screen by construction (:func:`quant_only_config` keeps
    ``universe``); this guards that a comparison of the two never mixes universes."""
    q_by_date = {d["as_of"]: d.get("screen") for d in quant}
    bad = [d["as_of"] for d in strategy if d["as_of"] in q_by_date and d.get("screen") != q_by_date[d["as_of"]]]
    missing = sorted({d["as_of"] for d in strategy} ^ set(q_by_date))
    if not bad and not missing:
        return ""
    return (f"the quant-only baseline did not screen the same universe as the strategy on {len(bad) + len(missing)} "
            f"decision date(s) (e.g. {(bad + missing)[0]}): the baseline comparison mixes universes")


def _finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


# --------------------------------------------------------------------------- entry point


def run_backtest(cfg: FirmConfig, start: str, end: str, mode: str, llm: LLM | None = None,
                 prices: PriceData | None = None, fundamentals_loader: FundamentalsLoader | None = None,
                 macro_loader: MacroLoader | None = None, filings_loader: FilingsLoader | None = None, *,
                 run_id: str | None = None, factor_prices: PriceData | None = None) -> BacktestResult:
    """Walk-forward backtest of the firm from ``start`` to ``end`` (inclusive).

    * ``mode="quant"``: the deterministic quant analyst alone, no LLM anywhere.
    * ``mode="offline"``: the full committee answered by :func:`make_offline_llm`
      (deterministic rules, not AI: a plumbing and sanity check, never evidence of skill).
    * ``mode="llm"``: the full committee with ``llm`` (required).

    ``prices`` defaults to a download covering ``start`` minus :data:`PRICE_LOOKBACK_DAYS`
    through ``end``. The loaders default to the live SEC / FRED sources (point-in-time);
    inject them to run offline. Rebalances happen every ``cfg.process.rebalance_every_days``
    trading days starting at the first trading day on or after ``start``.
    Artifacts (ledger, forecasts, lessons) go to ``<runs_dir>/backtests/<run_id>/`` and the
    trial ledger to ``<runs_dir>/trials.jsonl``, with ``runs_dir`` resolved by
    :meth:`FirmConfig.runs_path` (relative to the config file, not the working directory).

    After the run, ``result.evaluation`` is computed from the recorded decisions without
    any new decision or LLM call, as configured by ``cfg.evaluation`` (signal IC and power,
    composition-matched random books, cost sensitivity, factor attribution). Factor-proxy
    prices come from ``factor_prices`` if given, else from ``prices`` if it holds any
    proxy, else they are downloaded, but only when ``prices`` was not passed in (so an
    offline caller never touches the network). Returns are split into contaminated and
    clean periods at :func:`contamination_cutoff` (training cutoff plus
    ``cfg.evaluation.cutoff_buffer_days``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "llm" and llm is None:
        raise ValueError("mode 'llm' needs an LLM client (for example ClaudeLLM)")
    if mode != "llm" and llm is not None:
        raise ValueError(f"mode {mode!r} does not take an llm (offline uses make_offline_llm(); quant uses none)")
    if cfg.process.rebalance_every_days <= 0:
        raise ValueError("process.rebalance_every_days must be positive")
    start_s, end_s = _day(start), _day(end)
    if start_s >= end_s:
        raise ValueError(f"start {start_s} must be before end {end_s}")
    bench = cfg.universe.benchmark.upper()
    downloaded = prices is None
    if prices is None:
        prices = load_prices([*universe(cfg), bench],
                             pd.Timestamp(start_s) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS), end_s)
    idx = prices.close.index
    days = idx[(idx >= pd.Timestamp(start_s)) & (idx <= pd.Timestamp(end_s))]
    if len(days) < 2:
        raise ValueError(f"fewer than two trading days with prices between {start_s} and {end_s}")
    rebalance_days = set(days[::cfg.process.rebalance_every_days])

    fingerprint = cfg.fingerprint()
    run_id = run_id or f"bt-{time.strftime('%Y%m%dT%H%M%S')}-{mode}-{secrets.token_hex(3)}"
    run_dir = cfg.runs_path() / "backtests" / run_id
    loaders = {"fundamentals_loader": fundamentals_loader, "macro_loader": macro_loader,
               "filings_loader": filings_loader}

    # the strategy
    strat_cfg = quant_only_config(cfg) if mode == "quant" else cfg
    strat_llm = make_offline_llm() if mode == "offline" else llm
    sheets = _SheetCache(prices, loaders)          # built once per date, shared with the quant-only baseline
    policy = _CommitteePolicy(strat_cfg, strat_llm, prices, run_dir, run_id, loaders, sheets)
    sim, card = policy.run(prices, days, rebalance_days, end_s)
    warnings = list(policy.warnings)

    # baselines through the same engine; the benchmark uses the same total-return accounting as
    # the books (raw closes + implied dividends, missing prints forward-filled)
    bench_ret = _total_returns(prices, bench, days)
    ew = _simulate(prices, cfg, days, rebalance_days, _equal_weight_policy(cfg, prices))
    baseline_returns = {"benchmark": bench_ret, "equal_weight": ew.nav.pct_change().iloc[1:]}
    baseline_sims = {"equal_weight": ew}
    q_sim: _Sim | None = None
    if mode != "quant":
        q_policy = _CommitteePolicy(quant_only_config(cfg), None, prices, run_dir / "quant_only",
                                    f"{run_id}-quant", loaders, sheets)
        q_sim, _ = q_policy.run(prices, days, rebalance_days, end_s)
        baseline_returns["quant_only"] = q_sim.nav.pct_change().iloc[1:]
        baseline_sims["quant_only"] = q_sim
        mismatch = _screen_mismatch(sim.decisions, q_sim.decisions)
        if mismatch:
            warnings.append(mismatch)

    returns = sim.nav.pct_change().iloc[1:]
    trials_path = cfg.runs_path() / TRIALS_FILE
    scr = cfg.universe.screen
    if scr.enabled and scr.preset.strip():
        universe_note = f"preset {scr.preset.strip()!r}: {SURVIVORSHIP_NOTE}"
    elif scr.enabled:
        universe_note = ("screened universe from a file or inline list: if it is today's membership of an index, "
                         + SURVIVORSHIP_NOTE)
    else:
        universe_note = ""
    if universe_note:
        warnings.insert(0, f"SURVIVORSHIP: {universe_note}")
    prompts = prompt_digest() if mode != "quant" else ""
    n_trials = _count_trials(trials_path, fingerprint, mode, start_s, end_s, prompts)
    reports = {"strategy": performance_report(returns, bench_ret, sim.turnover, sim.costs, n_trials=n_trials),
               "benchmark": performance_report(bench_ret, bench_ret)}
    for name, s in baseline_sims.items():
        reports[name] = performance_report(baseline_returns[name], bench_ret, s.turnover, s.costs)

    cutoff = training_cutoff(strat_cfg, mode)
    contamination_end = contamination_cutoff(strat_cfg, mode)
    pre = post = None
    if contamination_end is not None:
        in_force = _decision_in_force(returns.index, sim.decision_dates)
        contaminated = in_force.notna() & (in_force <= pd.Timestamp(contamination_end))
        pre = _report_or_none(returns[contaminated], bench_ret, n_trials)
        post = _report_or_none(returns[~contaminated], bench_ret, n_trials)

    clean_returns = None
    if contamination_end is not None:      # returns earned under a decision made after the contaminated period
        clean_returns = returns[in_force.notna() & (in_force > pd.Timestamp(contamination_end))]
    evaluation = _evaluate(cfg=cfg, strat_cfg=strat_cfg, prices=prices, days=days, rebalance_days=rebalance_days,
                           end=end_s, sim=sim, q_sim=q_sim, returns=returns,
                           bench_total_pct=reports["benchmark"].total_return_pct, factor_prices=factor_prices,
                           downloaded=downloaded, start=start_s, contamination_end=contamination_end,
                           clean_returns=clean_returns)

    _append_trial(trials_path, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "fingerprint": fingerprint, "mode": mode,
        "start": start_s, "end": end_s, "days": reports["strategy"].days, "universe_digest": cfg.universe_digest(),
        "prompts": prompts,
        "sharpe": _finite_or_none(reports["strategy"].sharpe),
        "total_return_pct": _finite_or_none(reports["strategy"].total_return_pct),
    })
    return BacktestResult(
        run_id=run_id, mode=mode, start=start_s, end=end_s, config_fingerprint=fingerprint, run_dir=str(run_dir),
        nav=sim.nav, returns=returns, benchmark_returns=bench_ret, baseline_returns=baseline_returns,
        turnover=sim.turnover, costs=sim.costs, decisions=sim.decisions, reports=reports,
        pre_cutoff_report=pre, post_cutoff_report=post, model_cutoff=cutoff, n_trials=n_trials, scorecard=card,
        usage=strat_llm.usage.summary() if strat_llm is not None else {},
        costs_usd={"commission": sim.commission_usd, "slippage": sim.slippage_usd}, warnings=warnings,
        dividends_usd=sim.dividends_usd, benchmark=bench, tickers=universe(cfg), models=run_models(strat_cfg, mode),
        contamination_end=contamination_end,
        cutoff_buffer_days=cfg.evaluation.cutoff_buffer_days if cutoff is not None else 0,
        cost_model={"commission_bps": cfg.costs.commission_bps, "slippage_bps": cfg.costs.slippage_bps,
                    "impact_coefficient": cfg.costs.impact_coefficient},
        trials_path=str(trials_path), evaluation=evaluation, exposure=sim.invested, universe_note=universe_note,
        baseline_nav={name: s.nav for name, s in baseline_sims.items()},
    )
