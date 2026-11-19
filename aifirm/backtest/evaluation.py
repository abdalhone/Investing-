"""Backtest evaluation: does the record show skill, and would we know? (P10; research
report 6.5, 7.1-7.4 and 9.3 item 6.)

:class:`Evaluation` gathers, for one backtest:

* **Signal IC** (:mod:`.ic`): per-date Spearman IC of the pooled probability, of each
  analyst, of the equal-weight pool and of the quant-only baseline, with t-stats
  (Newey-West when forecast windows overlap) and paired per-date differences.
* **Power** (:mod:`.ic`): how many decision dates (and years) the observed IC would need
  for t = 2 and for 80% power, and MinTRL in years from the strategy's Sharpe.
* **Composition-matched random books** (:mod:`.random_books`): the strategy's percentile
  and empirical p-value against books with the same weights, count, holding periods and
  cost rate but random names.
* **Cost sensitivity** (this module): the SAME recorded decisions re-executed with every
  cost component multiplied by ``k`` (no new decisions, no LLM call), and the breakeven
  multiplier at which the excess return vs the benchmark (and vs the quant-only
  baseline) reaches zero, found by bisection (:func:`breakeven_multiplier`).
* **Factor attribution** (:mod:`.factors`): alpha and loadings on ETF factor proxies.

Fail closed: each part is computed independently and a part that cannot be computed is
None with a note saying why. A missing part is never read as evidence of skill; the
report says it is missing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .factors import FactorAttribution
from .ic import ICDifference, ICStats, PowerAnalysis
from .random_books import RandomBooksResult

BREAKEVEN_TOL = 0.01            # absolute tolerance on the cost multiplier
BREAKEVEN_REL_TOL = 0.005       # ... or this fraction of it, whichever is larger
BREAKEVEN_MAX_ITER = 60


@dataclass
class Breakeven:
    """Cost multiplier ``k*`` at which an excess return changes sign (see
    :func:`breakeven_multiplier`).

    ``status``:

    * ``"found"``: ahead at zero cost, behind from ``k*`` on (``multiplier`` is ``k*``);
    * ``"above_cap"``: still ahead at ``cap`` (``multiplier`` is None: read "> cap");
    * ``"no_edge"``: not ahead at zero cost, nor at ``cap`` (``multiplier`` is 0.0);
    * ``"gains_above"``: behind at zero cost but ahead from ``k*`` on, which happens
      against a comparator that trades more (``multiplier`` is ``k*``).

    ``bps_one_way`` is ``k*`` (or ``cap`` for ``above_cap``, a lower bound) times the
    modelled cost per dollar traded at ``k = 1``; each trade is one side of a round trip."""

    status: str
    multiplier: float | None
    cap: float
    evaluations: int
    bps_one_way: float | None = None

    def describe(self, against: str) -> str:
        """One plain sentence, e.g. "breakeven vs the benchmark at 3.20x modelled costs (...)"."""
        bps = f" (~{self.bps_one_way:,.1f} bps per dollar traded, one-way)" if self.bps_one_way is not None else ""
        if self.status == "above_cap":
            bound = f" (> {self.bps_one_way:,.0f} bps per dollar traded, one-way)" if self.bps_one_way is not None else ""
            return f"still ahead of {against} at {self.cap:g}x modelled costs: breakeven > {self.cap:g}x{bound}"
        if self.status == "found" and self.multiplier is not None:
            return f"breakeven vs {against} at {self.multiplier:.2f}x modelled costs{bps}"
        if self.status == "gains_above" and self.multiplier is not None:
            return (f"behind {against} at zero cost, but ahead above {self.multiplier:.2f}x modelled costs{bps}: it "
                    f"trades less, so it gains as costs rise")
        return f"no edge over {against} at zero cost or at {self.cap:g}x modelled costs (breakeven 0x)"


def breakeven_multiplier(excess: Callable[[float], float], cap: float,
                         known: Mapping[float, float] | None = None, tol: float = BREAKEVEN_TOL,
                         rel_tol: float = BREAKEVEN_REL_TOL, max_iter: int = BREAKEVEN_MAX_ITER) -> Breakeven:
    """Bisection for the cost multiplier ``k`` in ``[0, cap]`` at which ``excess(k)`` (an
    excess return; "ahead" means ``> 0``) changes sign.

    ``known`` holds already computed ``{k: excess(k)}`` (for example the report's cost
    grid); they are reused, and the first adjacent pair of them (with 0 and, when needed,
    ``cap``) whose signs differ is the initial bracket. Bisection keeps a bracket
    ``[lo, hi]`` with ``excess(lo)`` and ``excess(hi)`` on different sides of zero until it
    is narrower than ``max(tol, rel_tol * hi)`` and returns its midpoint. Against a
    comparator without costs (the benchmark) ``excess`` falls as ``k`` rises and the
    answer is the usual breakeven (``"found"``); against one that trades more it can rise
    instead (``"gains_above"``). A non-finite excess raises ``ValueError`` (a broken
    re-simulation must not produce a number)."""
    if not (math.isfinite(cap) and cap > 0):
        raise ValueError(f"cap must be positive and finite, got {cap!r}")
    values: dict[float, float] = {float(k): float(v) for k, v in (known or {}).items() if 0 <= k <= cap}
    calls = 0

    def ahead(k: float) -> bool:
        nonlocal calls
        if k not in values:
            calls += 1
            values[k] = float(excess(k))
        if not math.isfinite(values[k]):
            raise ValueError(f"excess return at cost multiplier {k:g} is not finite")
        return values[k] > 0

    start = ahead(0.0)
    lo, hi = 0.0, None
    for k in sorted(values):
        if ahead(k) == start:
            lo = k
        else:
            hi = k
            break
    if hi is None:
        if ahead(cap) == start:
            return Breakeven("above_cap", None, cap, calls) if start else Breakeven("no_edge", 0.0, cap, calls)
        hi = cap
    for _ in range(max_iter):
        if hi - lo <= max(tol, rel_tol * hi):
            break
        mid = 0.5 * (lo + hi)
        if ahead(mid) == start:
            lo = mid
        else:
            hi = mid
    return Breakeven("found" if start else "gains_above", 0.5 * (lo + hi), cap, calls)


@dataclass
class CostSensitivity:
    """The same decisions re-executed at cost multipliers ``k`` (total returns in percent).

    ``realised_cost_bps``: the strategy's modelled cost per dollar traded at ``k = 1``
    (commission + spread + impact over traded notional), the unit of ``bps_one_way``."""

    multipliers: list[float]
    strategy_total_return_pct: dict[float, float]
    benchmark_total_return_pct: float
    quant_total_return_pct: dict[float, float] | None
    realised_cost_bps: float
    vs_benchmark: Breakeven
    vs_quant: Breakeven | None
    replay_matches_engine: bool          # k = 1 reproduced the engine's NAV (a self-check)

    def excess_vs_benchmark(self, k: float) -> float:
        return self.strategy_total_return_pct[k] - self.benchmark_total_return_pct

    def excess_vs_quant(self, k: float) -> float | None:
        if self.quant_total_return_pct is None or k not in self.quant_total_return_pct:
            return None
        return self.strategy_total_return_pct[k] - self.quant_total_return_pct[k]


def cost_sensitivity(replay_strategy: Callable[[float], float], benchmark_total_return_pct: float,
                     multipliers: list[float], cap: float, realised_cost_bps: float,
                     engine_total_return_pct: float,
                     replay_quant: Callable[[float], float] | None = None) -> CostSensitivity:
    """Re-execute at each multiplier and search the breakevens.

    ``replay_strategy(k)`` / ``replay_quant(k)`` return the total return in percent of the
    strategy's (the quant-only baseline's) recorded decisions re-executed with every cost
    component multiplied by ``k``; results are memoised, so the grid and both bisections
    share simulations. The benchmark is bought and held without costs, so its return does
    not depend on ``k``."""
    strat: dict[float, float] = {}
    quant: dict[float, float] = {}

    def s(k: float) -> float:
        if k not in strat:
            strat[k] = float(replay_strategy(k))
        return strat[k]

    def q(k: float) -> float:
        if replay_quant is None:
            raise RuntimeError("no quant-only replay was given")
        if k not in quant:
            quant[k] = float(replay_quant(k))
        return quant[k]

    grid = sorted({float(k) for k in multipliers} | {1.0})
    for k in grid:
        s(k)
        if replay_quant is not None:
            q(k)
    matches = math.isclose(strat[1.0], engine_total_return_pct, rel_tol=1e-9, abs_tol=1e-9)
    vs_b = breakeven_multiplier(lambda k: s(k) - benchmark_total_return_pct, cap,
                                known={k: strat[k] - benchmark_total_return_pct for k in grid})
    vs_q = None
    if replay_quant is not None:
        vs_q = breakeven_multiplier(lambda k: s(k) - q(k), cap, known={k: strat[k] - quant[k] for k in grid})
    for b in (vs_b, vs_q):
        if b is not None and b.status != "no_edge" and realised_cost_bps > 0:
            b.bps_one_way = (b.cap if b.status == "above_cap" else (b.multiplier or 0.0)) * realised_cost_bps
    shown = sorted({float(k) for k in multipliers})
    return CostSensitivity(
        multipliers=shown, strategy_total_return_pct={k: strat[k] for k in sorted(strat)},
        benchmark_total_return_pct=benchmark_total_return_pct,
        quant_total_return_pct={k: quant[k] for k in sorted(quant)} if replay_quant is not None else None,
        realised_cost_bps=realised_cost_bps, vs_benchmark=vs_b, vs_quant=vs_q, replay_matches_engine=matches)


@dataclass
class Evaluation:
    """Everything that answers "is there skill, and would we know?" for one backtest.
    A part is None when it could not be computed; ``notes`` says why."""

    horizon_days: int
    step_days: int                                   # rebalance_every_days
    forward_return_definition: str
    ic: dict[str, ICStats] = field(default_factory=dict)
    ic_differences: list[ICDifference] = field(default_factory=list)
    power: PowerAnalysis | None = None
    random_books: RandomBooksResult | None = None
    costs: CostSensitivity | None = None
    factors: FactorAttribution | None = None
    notes: list[str] = field(default_factory=list)
