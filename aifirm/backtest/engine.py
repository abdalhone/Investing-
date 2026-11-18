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

Known limitations (also stated in the report): a universe chosen today is biased toward
survivors; costs are modelled, not observed; no taxes, withholding or borrow fees; cash
distributions are inferred from Yahoo's adjusted closes (ordinary dividends and other
distributions alike), not read from a corporate-actions feed.
"""

from __future__ import annotations

import json
import logging
import math
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from ..agents import llm_analysts, make_offline_llm
from ..config import FirmConfig, model_cutoff
from ..data.prices import PriceData, load_prices
from ..execution.broker import PaperBroker
from ..journal import ForecastStore, Ledger, LessonStore, scorecard
from ..llm.client import LLM
from ..pipeline import (PRICE_LOOKBACK_DAYS, FilingsLoader, FundamentalsLoader, MacroLoader, MarketSnapshot, decide,
                        gather_data, implied_dividends, market_snapshot, quant_only_config, universe)
from ..portfolio import RiskState, orders_from_target
from ..schemas import Order, TargetPortfolio, TargetPosition
from .metrics import PerformanceReport, performance_report

MODES = ("quant", "offline", "llm")
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


@dataclass
class BacktestResult:
    """Everything a backtest produced. Return series are daily, net of costs.

    ``reports`` holds a :class:`PerformanceReport` for ``"strategy"`` and each baseline
    (``"benchmark"``, ``"equal_weight"`` and, unless ``mode == "quant"``, ``"quant_only"``).
    ``pre_cutoff_report`` / ``post_cutoff_report`` split the strategy's returns by whether
    the decision behind them was made on or before ``model_cutoff`` (None in quant mode).
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
                dividends_usd=broker.total_dividends)


def _total_returns(prices: PriceData, ticker: str, days: pd.DatetimeIndex) -> pd.Series:
    """Daily total returns of ``ticker`` on ``days[1:]`` from raw closes plus the same implied
    dividends the paper broker credits: ``(close_t + D_t) / close_{t-1} - 1``. A missing
    print is forward-filled (the move across a gap lands on the next print instead of
    being dropped), so a book holding ``ticker`` and this series agree exactly."""
    close = prices.close[ticker].ffill()
    div = implied_dividends(prices)[ticker]
    tr = (close + div) / close.shift(1) - 1.0
    return tr.reindex(days).iloc[1:].fillna(0.0)


class _CommitteePolicy:
    """The firm itself: gather point-in-time data and run :func:`decide` at each rebalance."""

    def __init__(self, cfg: FirmConfig, llm: LLM | None, prices: PriceData, run_dir: Path, run_id: str,
                 loaders: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.llm, self.prices, self.loaders = cfg, llm, prices, loaders
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
        _, factsheets = gather_data(self.cfg, as_of, live=False, prices=self.prices, **self.loaders)
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
        rec = {
            "pooled_prob": {t: a.prob_outperform for t, a in dec.aggregates.items()},
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


# --------------------------------------------------------------------------- evaluation helpers


def contamination_cutoff(cfg: FirmConfig, mode: str) -> str | None:
    """The date up to which the run's language models may have seen the future.

    None in quant mode (no model). Otherwise the latest training cutoff among the models
    the configuration uses; an unknown cutoff counts as today, so every period is treated
    as contaminated (fail closed: unproven skill is not counted as skill)."""
    if mode == "quant":
        return None
    models: set[str] = set()
    if llm_analysts(cfg):
        models |= set(cfg.models.analyst_models)
    if cfg.process.memo_enabled or llm_analysts(cfg):
        models.add(cfg.models.committee_model)     # the chair and the lesson writer
    if cfg.process.risk_review_enabled:
        models.add(cfg.models.risk_model)
    if not models:
        return None
    cutoffs = [model_cutoff(m) for m in sorted(models)]
    if any(c is None for c in cutoffs):
        return str(date.today())
    return max(c for c in cutoffs if c is not None)


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


def _count_trials(path: Path, fingerprint: str, mode: str, start: str, end: str) -> int:
    """Distinct (fingerprint, mode, start, end) trials in the trial ledger, including this one.

    The backtest window is part of the trial: re-running one configuration over many
    windows and keeping the best is the same selection DSR corrects for. Re-running the
    identical (configuration, mode, window) is not a new trial. Every ledger row carries
    its window, so older rows count correctly too."""
    trials = {(fingerprint, mode, start, end)}
    if path.exists():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                trials.add((str(entry["fingerprint"]), str(entry["mode"]), str(entry["start"]), str(entry["end"])))
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("%s line %d is not a valid trial record; skipped", path, n)
    return len(trials)


def _append_trial(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, allow_nan=False, default=str) + "\n")


def _finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


# --------------------------------------------------------------------------- entry point


def run_backtest(cfg: FirmConfig, start: str, end: str, mode: str, llm: LLM | None = None,
                 prices: PriceData | None = None, fundamentals_loader: FundamentalsLoader | None = None,
                 macro_loader: MacroLoader | None = None, filings_loader: FilingsLoader | None = None, *,
                 run_id: str | None = None) -> BacktestResult:
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
    policy = _CommitteePolicy(strat_cfg, strat_llm, prices, run_dir, run_id, loaders)
    sim, card = policy.run(prices, days, rebalance_days, end_s)
    warnings = list(policy.warnings)

    # baselines through the same engine; the benchmark uses the same total-return accounting as
    # the books (raw closes + implied dividends, missing prints forward-filled)
    bench_ret = _total_returns(prices, bench, days)
    ew = _simulate(prices, cfg, days, rebalance_days, _equal_weight_policy(cfg, prices))
    baseline_returns = {"benchmark": bench_ret, "equal_weight": ew.nav.pct_change().iloc[1:]}
    baseline_sims = {"equal_weight": ew}
    if mode != "quant":
        q_policy = _CommitteePolicy(quant_only_config(cfg), None, prices, run_dir / "quant_only",
                                    f"{run_id}-quant", loaders)
        q_sim, _ = q_policy.run(prices, days, rebalance_days, end_s)
        baseline_returns["quant_only"] = q_sim.nav.pct_change().iloc[1:]
        baseline_sims["quant_only"] = q_sim

    returns = sim.nav.pct_change().iloc[1:]
    trials_path = cfg.runs_path() / TRIALS_FILE
    n_trials = _count_trials(trials_path, fingerprint, mode, start_s, end_s)
    reports = {"strategy": performance_report(returns, bench_ret, sim.turnover, sim.costs, n_trials=n_trials),
               "benchmark": performance_report(bench_ret, bench_ret)}
    for name, s in baseline_sims.items():
        reports[name] = performance_report(baseline_returns[name], bench_ret, s.turnover, s.costs)

    cutoff = contamination_cutoff(strat_cfg, mode)
    pre = post = None
    if cutoff is not None:
        in_force = _decision_in_force(returns.index, sim.decision_dates)
        contaminated = in_force.notna() & (in_force <= pd.Timestamp(cutoff))
        pre = _report_or_none(returns[contaminated], bench_ret, n_trials)
        post = _report_or_none(returns[~contaminated], bench_ret, n_trials)

    _append_trial(trials_path, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "fingerprint": fingerprint, "mode": mode,
        "start": start_s, "end": end_s, "days": reports["strategy"].days,
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
        dividends_usd=sim.dividends_usd,
    )
