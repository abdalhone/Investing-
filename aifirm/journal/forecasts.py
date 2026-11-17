"""Forecast store, resolution, scoring and believability weights (P7, P3, P10).

Every non-abstaining analyst view and every pooled committee view becomes a
:class:`~aifirm.schemas.ForecastRecord`: a binary, resolvable question ("does TICKER beat
the benchmark over the next ``horizon_days`` trading days?") with a probability. When
the horizon passes, the record is resolved from adjusted-close total returns and scored
with the Brier score. Those scores become each analyst's believability weight.

Why it works this way
---------------------
* **Score everything (P7).** Superforecasting research shows that track records, not
  confidence or eloquence, separate good forecasters from bad ones. Brier and log loss
  reward calibration as well as direction.
* **Strictly point-in-time (P3).** A record counts for a decision on date ``D`` only if
  its ``resolve_on <= D``. The filter is on the forecast's own ``resolve_on`` date, not on
  when :meth:`ForecastStore.resolve` happened to run, so a backtest that resolves
  everything early still cannot leak future outcomes into past weights.
* **Shrink toward equal weight.** With ``n`` resolved questions the weight is
  ``1 + n/(n+prior_n) * 4 * skill``, clipped to ``[floor, cap]``, and exactly 1.0 below
  ``min_n``. A handful of lucky calls cannot capture the committee, and nobody is
  silenced completely (the floor keeps dissent alive, see P6).
* **Training-contamination guard (P10).** Forecasts made on dates the model may have
  seen in training (``pre_cutoff``) can "predict" memorised history. They are excluded
  from weights by default and reported separately in the scorecard.
* **Sealed views are scored, not revisions.** Believability should measure each
  analyst's *independent* judgement from its own data slice (P4/P5); scoring revised
  views would reward moving toward the challenger, i.e. herding.
* **Samples are not extra evidence.** With ``samples_per_analyst > 1`` an analyst answers
  the same question several times. Believability counts distinct questions
  (``n_questions``: one per ticker and decision date), so repeated samples cannot rush an
  analyst past ``min_n`` or inflate its shrinkage factor.

Known limitation: a record whose ticker has no price on the resolution date (delisted,
halted) stays unresolved rather than being guessed. That biases scores toward survivors;
use a point-in-time universe for serious research (see :mod:`aifirm.data.prices`).
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from ..config import FirmConfig, model_cutoff
from ..data.prices import PriceData
from ..schemas import AggregatedView, ForecastRecord, ViewRecord

COMMITTEE_AGENT = "committee"
DETERMINISTIC_MODELS = frozenset({"deterministic"})   # code, not a trained model: nothing to have memorised
NO_SKILL_BRIER = 0.25          # Brier score of a constant 0.5 forecast
BELIEVABILITY_SLOPE = 4.0      # k in weight = 1 + shrink * k * skill (contract, ARCHITECTURE.md 3.12)
LOG_LOSS_EPS = 1e-6
# Calibration buckets are finer near 0.5, where honest one-month equity forecasts live.
CALIBRATION_EDGES = (0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0)

TradingCalendar = Sequence[str | date | pd.Timestamp] | pd.Index


def _day(d: str | date | pd.Timestamp) -> str:
    """Normalise any date-like value to ``YYYY-MM-DD`` (lexicographic order = date order)."""
    return str(pd.Timestamp(d).date())


# --------------------------------------------------------------------------- record creation


def _calendar(trading_calendar: TradingCalendar | None) -> pd.DatetimeIndex:
    if trading_calendar is None or len(trading_calendar) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(trading_calendar))).normalize().unique().sort_values()


def resolve_date(as_of: str | date | pd.Timestamp, horizon_days: int,
                 trading_calendar: TradingCalendar | None) -> str:
    """The date ``horizon_days`` trading days after the ``as_of`` close.

    Counts from the last calendar date on or before ``as_of`` (a weekend decision uses
    Friday's close). Where the calendar runs out (live trading: the future is not in the
    price index yet), the remaining days are counted in pandas business days, which
    ignore exchange holidays, so the approximate date can be up to a holiday or two early.
    :meth:`ForecastStore.resolve` then uses the last close on or before it.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    d = pd.Timestamp(as_of).normalize()
    cal = _calendar(trading_calendar)
    if len(cal) and cal[0] <= d <= cal[-1]:
        target = int(cal.searchsorted(d, side="right")) - 1 + horizon_days
        if target < len(cal):
            return _day(cal[target])
        return _day(cal[-1] + BDay(target - (len(cal) - 1)))
    return _day(d + BDay(horizon_days))


def is_pre_cutoff(model: str, as_of: str | date | pd.Timestamp) -> bool:
    """True if ``model`` may have seen the period after ``as_of`` in training (P10).

    Deterministic code is never contaminated. An LLM with an unknown cutoff is assumed
    contaminated (fail closed: unproven skill is not counted as skill). A forecast made
    on or before the cutoff date is contaminated because the model may know what
    happened next.
    """
    if model in DETERMINISTIC_MODELS:
        return False
    cutoff = model_cutoff(model)
    return cutoff is None or _day(as_of) <= _day(cutoff)


def make_forecast_records(run_id: str, as_of: str, views: list[ViewRecord], agg: AggregatedView,
                          cfg: FirmConfig, trading_calendar: TradingCalendar | None) -> list[ForecastRecord]:
    """One record per non-abstaining view plus one ``"committee"`` record for the pool.

    IDs are ``run_id|as_of|ticker|agent|s<sample>`` (``|committee`` for the pool), so they
    are unique across decision dates within one run and re-adding is idempotent. The
    ticker comes from ``agg`` (in blind mode an LLM may echo an alias such as "STOCK").
    Each analyst record stores its *sealed* probability (see module docstring).

    The committee record is skipped when every view abstained: a 0.5 pooled from nothing
    is not a forecast and would only dilute the committee's score. It is flagged
    ``pre_cutoff`` if any contributing member is, since contamination propagates through
    the pool.
    """
    as_of_s = _day(as_of)
    h = cfg.process.horizon_days
    common = dict(run_id=run_id, ticker=agg.ticker, benchmark=cfg.universe.benchmark, as_of=as_of_s,
                  horizon_days=h, resolve_on=resolve_date(as_of_s, h, trading_calendar))
    members = [v for v in views if not v.view.abstain]
    records = [
        ForecastRecord(id=f"{run_id}|{as_of_s}|{agg.ticker}|{v.agent}|s{v.sample}", agent=v.agent, model=v.model,
                       prob_outperform=v.view.prob_outperform,
                       expected_excess_return_pct=v.view.expected_excess_return_pct,
                       pre_cutoff=is_pre_cutoff(v.model, as_of_s), **common)
        for v in members
    ]
    if members:
        records.append(ForecastRecord(
            id=f"{run_id}|{as_of_s}|{agg.ticker}|{COMMITTEE_AGENT}", agent=COMMITTEE_AGENT,
            model="+".join(sorted({v.model for v in members})),
            prob_outperform=agg.prob_outperform, expected_excess_return_pct=agg.expected_excess_return_pct,
            pre_cutoff=any(r.pre_cutoff for r in records), **common))
    return records


# --------------------------------------------------------------------------- store


class ForecastStore:
    """SQLite-backed forecast store (one table, one row per record).

    The full record is stored as JSON next to a few indexed columns, so new optional
    fields on :class:`ForecastRecord` need no migration. ``add`` is idempotent and never
    overwrites: a forecast is sealed when made, and re-adding a record (e.g. a re-run of
    the same ``run_id``) must not reset an already resolved outcome.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock, self._con:
            self._con.execute("""CREATE TABLE IF NOT EXISTS forecasts (
                id TEXT PRIMARY KEY, agent TEXT NOT NULL, ticker TEXT NOT NULL, as_of TEXT NOT NULL,
                resolve_on TEXT NOT NULL, resolved INTEGER NOT NULL, record TEXT NOT NULL)""")
            self._con.execute("CREATE INDEX IF NOT EXISTS forecasts_pending ON forecasts (resolved, resolve_on)")

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> "ForecastStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, records: Iterable[ForecastRecord]) -> None:
        rows = [(r.id, r.agent, r.ticker, _day(r.as_of), _day(r.resolve_on), int(r.resolved), r.model_dump_json())
                for r in records]
        with self._lock, self._con:
            self._con.executemany("INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def all(self) -> list[ForecastRecord]:
        """Every record, ordered by decision date then id (deterministic)."""
        with self._lock:
            rows = self._con.execute("SELECT record FROM forecasts ORDER BY as_of, id").fetchall()
        return [ForecastRecord.model_validate_json(r[0]) for r in rows]

    def resolve(self, prices: PriceData, as_of: str) -> list[ForecastRecord]:
        """Resolve every pending record with ``resolve_on <= as_of`` whose prices exist.

        ``prices`` is first cut at ``as_of`` (no look-ahead even if the caller passes the
        full history). Outcome = 1 if the ticker's adjusted-close (total) return from the
        ``as_of`` close to the ``resolve_on`` close strictly beats the benchmark's. A
        record stays pending when the data do not yet reach ``resolve_on`` or either
        series lacks a price at the start or end date; it is never resolved by guessing.
        Returns the newly resolved records.
        """
        as_of_s = _day(as_of)
        with self._lock:
            rows = self._con.execute(
                "SELECT record FROM forecasts WHERE resolved = 0 AND resolve_on <= ? ORDER BY as_of, id",
                (as_of_s,)).fetchall()
        if not rows:
            return []
        adj = prices.as_of(as_of_s).adj_close
        resolved = []
        for (blob,) in rows:
            rec = ForecastRecord.model_validate_json(blob)
            excess = _excess_return(adj, rec.ticker, rec.benchmark, rec.as_of, rec.resolve_on)
            if excess is None:
                continue
            outcome = int(excess > 0)
            resolved.append(rec.model_copy(update=dict(
                resolved=True, outcome=outcome, realized_excess_return_pct=excess * 100.0,
                brier=(rec.prob_outperform - outcome) ** 2)))
        with self._lock, self._con:
            self._con.executemany("UPDATE forecasts SET resolved = 1, record = ? WHERE id = ? AND resolved = 0",
                                  [(r.model_dump_json(), r.id) for r in resolved])
        return resolved


def _excess_return(adj: pd.DataFrame, ticker: str, benchmark: str, start: str, end: str) -> float | None:
    """Ticker total return minus benchmark total return (fractions) from the last close on
    or before ``start`` to the last close on or before ``end``; ``None`` if not knowable."""
    if adj.empty or ticker not in adj.columns or benchmark not in adj.columns:
        return None
    idx = adj.index
    if idx[-1] < pd.Timestamp(end):
        return None                                   # the resolution date has not been reached in the data
    i0 = int(idx.searchsorted(pd.Timestamp(start), side="right")) - 1
    i1 = int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    if i0 < 0 or i1 <= i0:
        return None
    t0, t1 = adj[ticker].iloc[i0], adj[ticker].iloc[i1]
    b0, b1 = adj[benchmark].iloc[i0], adj[benchmark].iloc[i1]
    if not all(math.isfinite(x) and x > 0 for x in (t0, t1, b0, b1)):
        return None
    return float((t1 / t0) - (b1 / b0))


# --------------------------------------------------------------------------- scoring


def _scored(r: ForecastRecord, as_of: str | None) -> bool:
    return r.resolved and r.outcome in (0, 1) and (as_of is None or _day(r.resolve_on) <= as_of)


def _bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}{']' if hi == CALIBRATION_EDGES[-1] else ')'}"


def _calibration(probs: list[float], outcomes: list[int]) -> list[dict]:
    """Non-empty probability buckets with mean forecast vs observed frequency."""
    out = []
    edges = CALIBRATION_EDGES
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        sel = [(p, o) for p, o in zip(probs, outcomes) if lo <= p < hi or (last and p == hi)]
        if sel:
            out.append({"bucket": _bucket_label(lo, hi), "lo": lo, "hi": hi, "n": len(sel),
                        "mean_prob": sum(p for p, _ in sel) / len(sel),
                        "observed_rate": sum(o for _, o in sel) / len(sel)})
    return out


def _agent_stats(recs: list[ForecastRecord]) -> dict:
    probs = [float(r.prob_outperform) for r in recs]
    outcomes = [int(r.outcome) for r in recs]
    n = len(recs)
    brier = sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / n
    clipped = [min(max(p, LOG_LOSS_EPS), 1 - LOG_LOSS_EPS) for p in probs]
    log_loss = -sum(o * math.log(p) + (1 - o) * math.log(1 - p) for p, o in zip(clipped, outcomes)) / n
    directional = [(p > 0.5) == (o == 1) for p, o in zip(probs, outcomes) if p != 0.5]
    excess = [r.realized_excess_return_pct for r in recs if r.realized_excess_return_pct is not None]
    return {
        "n": n,
        "n_questions": len({(r.ticker, _day(r.as_of)) for r in recs}),
        "n_pre_cutoff": sum(1 for r in recs if r.pre_cutoff),
        "brier": brier,
        "brier_skill": (NO_SKILL_BRIER - brier) / NO_SKILL_BRIER,
        "log_loss": log_loss,
        "hit_rate": sum(directional) / len(directional) if directional else None,
        "mean_prob": sum(probs) / n,
        "base_rate": sum(outcomes) / n,
        "mean_realized_excess_pct": sum(excess) / len(excess) if excess else None,
        "calibration": _calibration(probs, outcomes),
    }


def scorecard(records: Iterable[ForecastRecord], as_of: str | None = None,
              include_pre_cutoff: bool = True) -> dict[str, dict]:
    """Per-agent track record from resolved records (only ``resolve_on <= as_of`` if given).

    Keys per agent: ``n`` (records), ``n_questions`` (distinct ticker and decision date),
    ``n_pre_cutoff``, ``brier``, ``brier_skill`` (= (0.25 - brier) / 0.25, skill over a
    constant 0.5), ``log_loss``, ``hit_rate`` (direction right, over forecasts != 0.5;
    ``None`` if none), ``mean_prob``, ``base_rate``, ``mean_realized_excess_pct`` and
    ``calibration`` (buckets with ``n``, ``mean_prob``, ``observed_rate``). Agents with no
    scored record are absent. Ordered by agent name.
    """
    cutoff = _day(as_of) if as_of is not None else None
    by_agent: dict[str, list[ForecastRecord]] = defaultdict(list)
    for r in records:
        if _scored(r, cutoff) and (include_pre_cutoff or not r.pre_cutoff):
            by_agent[r.agent].append(r)
    return {agent: _agent_stats(by_agent[agent]) for agent in sorted(by_agent)}


def believability_weights(records: Iterable[ForecastRecord], as_of: str, agents: Iterable[str],
                          min_n: int = 20, prior_n: int = 30, floor: float = 0.25, cap: float = 2.0,
                          include_pre_cutoff: bool = False) -> dict[str, float]:
    """Track-record weights for ``agents`` usable for a decision on ``as_of`` (P7).

    ``skill = (0.25 - brier) / 0.25``; ``weight = 1 + n/(n + prior_n) * 4 * skill``
    clipped to ``[floor, cap]``; ``weight = 1.0`` while ``n < min_n``. ``n`` counts
    distinct questions (see module docstring). Only records with ``resolve_on <= as_of``
    count, and ``pre_cutoff`` records only with ``include_pre_cutoff=True``.
    """
    if not 0 < floor <= 1.0 <= cap:
        raise ValueError(f"need 0 < floor <= 1 <= cap, got floor={floor}, cap={cap}")
    if min_n < 0 or prior_n < 0:
        raise ValueError("min_n and prior_n must be non-negative")
    cards = scorecard(records, as_of, include_pre_cutoff=include_pre_cutoff)
    weights = {}
    for agent in agents:
        card = cards.get(agent)
        n = card["n_questions"] if card else 0
        if n == 0 or n < min_n:
            weights[agent] = 1.0
            continue
        skill = (NO_SKILL_BRIER - card["brier"]) / NO_SKILL_BRIER
        shrink = n / (n + prior_n)
        weights[agent] = min(cap, max(floor, 1.0 + shrink * BELIEVABILITY_SLOPE * skill))
    return weights
