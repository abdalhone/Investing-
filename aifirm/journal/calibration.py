"""Per-agent probability calibration and point-in-time rank IC from resolved forecasts.

Why (RESEARCH 4 "overconfidence", 6.1, 6.2): stated LLM confidence is poorly calibrated
and agents carry persistent directional biases, so the firm should size from *calibrated*
probabilities, fitted out of sample, and should know whether an agent's probabilities
rank stocks at all (IC) before trusting its overlay.

Calibration map
---------------
For each agent, Platt recalibration in log-odds space::

    logit(p_cal) = a + b * logit(p_raw)

fitted to that agent's *resolved* forecasts:

* **Point-in-time (P3).** Only records with ``resolve_on <= as_of`` count (the outcome was
  known at the decision close of ``as_of``). Forecasts flagged ``pre_cutoff`` (made on a
  date the model may have seen in training) are excluded unless
  ``include_pre_cutoff=True``, exactly as :func:`aifirm.journal.forecasts.believability_weights`
  decides: the pipeline passes ``include_pre_cutoff = llm is None or llm.is_offline``.
  Deterministic code (the quant, offline rules) is never flagged ``pre_cutoff``.
* **One question, one vote.** A question (ticker, decision date) answered by several
  samples gives each sample weight ``1/k``; a question re-forecast by another run counts
  once (:func:`aifirm.journal.forecasts.question_key`; the record that sorts first by
  decision date then id stands, the order of ``ForecastStore.all()``). ``n`` is the number
  of distinct questions.
* **Shrink toward the identity.** The fit is a MAP estimate whose prior is worth
  ``prior_strength`` forecasts: it adds pseudo-observations with total weight
  ``prior_strength`` at the agent's own stated probabilities whose outcomes equal those
  probabilities (soft labels). The identity map ``(a, b) = (0, 1)`` is the exact optimum
  of the pseudo-data, so with ``n`` real questions the estimate sits roughly
  ``n / (n + prior_strength)`` of the way from the identity to the unregularised fit:
  with 30 questions and a prior of 50, about 37.5% of the way. (Locally this is
  a Gaussian prior centred on the identity with the precision of ``prior_strength``
  perfectly calibrated forecasts.) A tiny ridge keeps the problem strictly convex.
* **Bounded.** ``b`` is constrained to ``b_bounds`` (default [0, 1.5]): calibration may
  shrink a view toward 0.5 or sharpen it mildly, never invert its order. ``a`` is
  constrained to ``a_bounds`` (default [-0.5, 0.5] in log-odds, i.e. at most about 12
  points at p = 0.5): it removes a directional bias without dominating. The bounded
  optimum is exact (the objective is convex; the box boundary is searched edge by edge).
* **Never flips or creates a direction.** :meth:`Calibrator.apply` returns 0.5 when the
  map would move a view across 0.5, and leaves a raw 0.5 at 0.5: calibration can shrink,
  neutralise or mildly sharpen a view, but an LLM's "bullish" never becomes "bearish"
  (or the reverse) by code, and no direction is created where the agent stated none.
* **Minimum sample.** With fewer than ``min_n`` questions (default 30) the map is the
  identity.
* **Deterministic.** Records are sorted (decision date, id) before use and the solver is a
  fixed Newton iteration: the same records and ``as_of`` give the same parameters.

The stored ``ForecastRecord.prob_outperform`` must be the agent's RAW stated probability:
fitting on calibrated values would calibrate the calibration (a feedback loop). Calibrate a
copy of the views used for pooling (:func:`calibrate_view`), never the stored forecasts.

Rank IC
-------
:func:`resolved_rank_ic` measures, point-in-time, whether an agent's (or the analysts'
pooled) probabilities rank stocks by realised excess return: the Spearman correlation per
decision date, averaged over dates, with a t-statistic (Newey-West when horizons overlap).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..schemas import ForecastRecord, ViewRecord
from .forecasts import COMMITTEE_AGENT, POOLED_AGENTS, question_key

PROB_CLIP = 0.01                 # inputs clipped to [0.01, 0.99] before logit (same as pooling)
OUTPUT_FLOOR = 0.02              # calibrated probabilities stay in [0.02, 0.98] (same floor as sanitize_view)
DEFAULT_MIN_N = 30
DEFAULT_PRIOR_STRENGTH = 50.0
DEFAULT_A_BOUNDS = (-0.5, 0.5)
DEFAULT_B_BOUNDS = (0.0, 1.5)
RIDGE = 1e-6                     # numerical stabiliser; negligible next to any real prior
_MAX_ITER = 100
_TOL = 1e-12


def _day(d: object) -> str:
    return str(pd.Timestamp(d).date())  # type: ignore[arg-type]


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# --------------------------------------------------------------------------- calibrator


@dataclass(frozen=True)
class Calibrator:
    """A per-agent calibration map ``logit(p_cal) = a + b*logit(p_raw)`` (module docstring).

    ``n``: distinct questions used; ``n_records``: records used (samples included);
    ``brier_raw`` / ``brier_calibrated``: in-sample, question-weighted Brier scores before
    and after the map (None without data); ``fitted``: False when the map is the identity
    because the sample was too small; ``bound``: which parameter bound was active, if any."""

    agent: str
    a: float = 0.0
    b: float = 1.0
    n: int = 0
    n_records: int = 0
    brier_raw: float | None = None
    brier_calibrated: float | None = None
    fitted: bool = False
    reason: str = "no resolved forecasts"
    as_of: str = ""
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    bound: str = ""

    @property
    def is_identity(self) -> bool:
        return self.a == 0.0 and self.b == 1.0

    def apply(self, p: float) -> float:
        """Calibrated probability for a raw probability ``p``.

        The identity map returns ``p`` unchanged. Otherwise ``p`` is clipped to
        [0.01, 0.99], mapped, and the result is clipped to [0.02, 0.98]; a raw 0.5 stays
        0.5 and a result on the other side of 0.5 from ``p`` becomes 0.5 (no flips, no new
        directions). A non-finite ``p`` returns 0.5 (no edge: fail closed)."""
        try:
            p = float(p)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(p):
            return 0.5
        if self.is_identity:
            return p
        if p == 0.5:
            return 0.5
        pc = _sigmoid(self.a + self.b * float(_logit(p)))
        if (pc - 0.5) * (p - 0.5) <= 0:
            return 0.5
        return min(1 - OUTPUT_FLOOR, max(OUTPUT_FLOOR, pc))

    def to_dict(self) -> dict:
        return {**asdict(self), "is_identity": self.is_identity}


def identity(agent: str, reason: str, **kw: object) -> Calibrator:
    return Calibrator(agent=agent, reason=reason, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- fitting


class _Objective:
    """Weighted cross-entropy against soft targets plus a tiny ridge toward the identity."""

    def __init__(self, z: np.ndarray, w: np.ndarray, t: np.ndarray, ridge: float):
        self.z, self.w, self.t, self.ridge = z, w, t, ridge

    def value(self, a: float, b: float) -> float:
        eta = a + b * self.z
        # cross-entropy: t*softplus(-eta) + (1-t)*softplus(eta)
        ce = self.t * np.logaddexp(0.0, -eta) + (1 - self.t) * np.logaddexp(0.0, eta)
        return float(np.sum(self.w * ce) + 0.5 * self.ridge * (a * a + (b - 1.0) ** 2))

    def grad_hess(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        eta = a + b * self.z
        s = 1.0 / (1.0 + np.exp(-eta))
        r = self.w * (s - self.t)
        h = self.w * s * (1 - s)
        g = np.array([np.sum(r) + self.ridge * a, np.sum(r * self.z) + self.ridge * (b - 1.0)])
        H = np.array([[np.sum(h) + self.ridge, np.sum(h * self.z)],
                      [np.sum(h * self.z), np.sum(h * self.z * self.z) + self.ridge]])
        return g, H


def _newton(obj: _Objective, a: float, b: float, free: tuple[bool, bool]) -> tuple[float, float]:
    """Damped Newton on the free coordinates (a, b); fixed ones stay where they are."""
    x = np.array([a, b], dtype=float)
    idx = [i for i in (0, 1) if free[i]]
    f = obj.value(*x)
    for _ in range(_MAX_ITER):
        g, H = obj.grad_hess(*x)
        gs, Hs = g[idx], H[np.ix_(idx, idx)]
        try:
            step = np.linalg.solve(Hs, gs)
        except np.linalg.LinAlgError:
            step = gs / max(float(np.trace(Hs)), 1e-12)
        t = 1.0
        while True:
            cand = x.copy()
            cand[idx] = x[idx] - t * step
            fc = obj.value(*cand)
            if fc <= f or t < 1e-10:
                break
            t *= 0.5
        if fc > f:
            break
        moved = float(np.max(np.abs(cand - x)))
        x, f_old, f = cand, f, fc
        if moved < _TOL or f_old - f < _TOL * max(1.0, abs(f)):
            break
    return float(x[0]), float(x[1])


def fit_platt(probs: Iterable[float], outcomes: Iterable[int], weights: Iterable[float] | None = None, *,
              prior_strength: float = DEFAULT_PRIOR_STRENGTH, a_bounds: tuple[float, float] = DEFAULT_A_BOUNDS,
              b_bounds: tuple[float, float] = DEFAULT_B_BOUNDS, ridge: float = RIDGE) -> tuple[float, float, str]:
    """MAP Platt fit shrunk toward the identity (module docstring). Returns ``(a, b, bound)``
    where ``bound`` names the active constraint ("" if the optimum is interior).

    ``weights`` default to 1 (their sum is the effective sample size the prior is compared
    with). Pseudo-observations of total weight ``prior_strength`` with soft labels equal to
    the stated probabilities implement the prior."""
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(list(weights), dtype=float)
    if not (len(p) == len(y) == len(w)) or len(p) == 0:
        raise ValueError("probs, outcomes and weights must be non-empty and of equal length")
    if np.any(~np.isfinite(p)) or np.any((y != 0) & (y != 1)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("need finite probabilities, outcomes in {0, 1} and non-negative weights with a positive sum")
    _check_bounds(a_bounds, b_bounds, prior_strength)
    pc = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    z = np.log(pc / (1 - pc))
    lam = prior_strength / float(w.sum())
    obj = _Objective(z=z, w=w * (1 + lam), t=(y + lam * pc) / (1 + lam), ridge=ridge)

    a, b = _newton(obj, 0.0, 1.0, (True, True))
    (alo, ahi), (blo, bhi) = a_bounds, b_bounds
    if alo <= a <= ahi and blo <= b <= bhi:
        return a, b, ""
    # Convex objective: the bounded optimum lies on the box boundary. On each edge the
    # restricted problem is a 1-D convex problem whose bounded optimum is the clipped one.
    candidates = []
    for name, fixed_a, fixed_b in (("a=lower", alo, None), ("a=upper", ahi, None),
                                   ("b=lower", None, blo), ("b=upper", None, bhi)):
        if fixed_a is not None:
            _, bb = _newton(obj, fixed_a, 1.0, (False, True))
            cand = (fixed_a, min(bhi, max(blo, bb)))
        else:
            aa, _ = _newton(obj, 0.0, fixed_b, (True, False))
            cand = (min(ahi, max(alo, aa)), fixed_b)
        candidates.append((obj.value(*cand), name, cand))
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, name, (a, b) = candidates[0]
    return float(a), float(b), name


def _check_bounds(a_bounds: tuple[float, float], b_bounds: tuple[float, float], prior_strength: float) -> None:
    alo, ahi = a_bounds
    blo, bhi = b_bounds
    if not (alo <= 0.0 <= ahi):
        raise ValueError(f"a_bounds must contain 0 (the identity), got {a_bounds}")
    if not (0.0 <= blo <= 1.0 <= bhi):
        raise ValueError(f"b_bounds must satisfy 0 <= lower <= 1 <= upper (no inversion; identity allowed), got {b_bounds}")
    if not (prior_strength >= 0 and math.isfinite(prior_strength)):
        raise ValueError(f"prior_strength must be finite and non-negative, got {prior_strength}")


def _eligible(records: list[ForecastRecord], as_of: str, agent: str | None, include_pre_cutoff: bool,
              models: frozenset[str] | None, need_excess: bool = False) -> list[ForecastRecord]:
    """Resolved records usable at ``as_of`` (resolve_on <= as_of), pre-cutoff rule applied,
    one record per question (first by decision date then id)."""
    out, seen = [], set()
    for r in sorted(records, key=lambda r: (_day(r.as_of), r.id)):
        if agent is not None and r.agent != agent:
            continue
        if not (r.resolved and r.outcome in (0, 1) and _day(r.resolve_on) <= as_of):
            continue
        if r.pre_cutoff and not include_pre_cutoff:
            continue
        if models is not None and r.model not in models:
            continue
        if need_excess and (r.realized_excess_return_pct is None or not math.isfinite(r.realized_excess_return_pct)):
            continue
        key = question_key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _question_weights(recs: list[ForecastRecord]) -> list[float]:
    """1/k for each of the k sample records of one question (ticker, decision date)."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in recs:
        counts[(r.ticker, _day(r.as_of))] += 1
    return [1.0 / counts[(r.ticker, _day(r.as_of))] for r in recs]


def _brier(probs: list[float], outcomes: list[int], weights: list[float]) -> float:
    return sum(w * (p - o) ** 2 for p, o, w in zip(probs, outcomes, weights, strict=True)) / sum(weights)


def fit_calibrators(records: Iterable[ForecastRecord], as_of: str, agents: Iterable[str], *,
                    include_pre_cutoff: bool = False, min_n: int = DEFAULT_MIN_N,
                    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
                    a_bounds: tuple[float, float] = DEFAULT_A_BOUNDS,
                    b_bounds: tuple[float, float] = DEFAULT_B_BOUNDS,
                    models: Iterable[str] | None = None) -> dict[str, Calibrator]:
    """One :class:`Calibrator` per agent in ``agents``, usable for a decision at the close of
    ``as_of`` (module docstring). Agents with fewer than ``min_n`` distinct resolved
    questions get the identity (``fitted=False``) with their in-sample Brier still
    reported. ``models``: if given, only records made by these models count (a model
    change is a new agent, RESEARCH 6.2)."""
    if min_n < 0:
        raise ValueError(f"min_n must be non-negative, got {min_n}")
    _check_bounds(a_bounds, b_bounds, prior_strength)
    as_of_s = _day(as_of)
    recs_all = list(records)
    model_set = frozenset(models) if models is not None else None
    out: dict[str, Calibrator] = {}
    for agent in dict.fromkeys(agents):
        recs = _eligible(recs_all, as_of_s, agent, include_pre_cutoff, model_set)
        common = dict(as_of=as_of_s, prior_strength=float(prior_strength))
        if not recs:
            out[agent] = identity(agent, "no resolved forecasts by the as-of date", **common)
            continue
        probs = [float(r.prob_outperform) for r in recs]
        outcomes = [int(r.outcome) for r in recs]  # type: ignore[arg-type]
        weights = _question_weights(recs)
        n = len({(r.ticker, _day(r.as_of)) for r in recs})
        brier_raw = _brier(probs, outcomes, weights)
        stats = dict(n=n, n_records=len(recs), brier_raw=brier_raw, **common)
        if n < min_n:
            out[agent] = identity(agent, f"{n} resolved questions < min_n={min_n}: identity",
                                  brier_calibrated=brier_raw, **stats)
            continue
        a, b, bound = fit_platt(probs, outcomes, weights, prior_strength=prior_strength,
                                a_bounds=a_bounds, b_bounds=b_bounds)
        cal = Calibrator(agent=agent, a=a, b=b, fitted=True, bound=bound,
                         reason=f"MAP Platt fit on {n} questions (prior worth {prior_strength:g})", **stats)
        out[agent] = Calibrator(**{**asdict(cal),
                                   "brier_calibrated": _brier([cal.apply(p) for p in probs], outcomes, weights)})
    return out


# --------------------------------------------------------------------------- applying to views


def calibrate_view(v: ViewRecord, cal: Calibrator | None) -> ViewRecord:
    """A copy of ``v`` with its sealed and revised probabilities calibrated by ``cal``.

    Expected excess returns (sealed and revised) are scaled by the ratio of the normal
    edges ``Phi^-1(p_cal) / Phi^-1(p_raw)`` clipped to [0, max(b_bounds)=1.5], so the two
    sizing inputs move together; a view whose probability becomes 0.5 keeps no excess of
    the same sign. Abstentions, a missing calibrator and the identity return ``v`` itself.
    Scenarios and evidence are not changed. Use the copy for pooling only: the forecast
    store must keep the raw view (module docstring)."""
    if cal is None or cal.is_identity or v.view.abstain:
        return v
    out = v.model_copy(deep=True)
    p0 = out.view.prob_outperform
    out.view.prob_outperform = cal.apply(p0)
    out.view.expected_excess_return_pct = _scale_excess(out.view.expected_excess_return_pct, p0, out.view.prob_outperform)
    if out.revised_prob is not None:
        r0 = out.revised_prob
        out.revised_prob = cal.apply(r0)
        if out.revised_excess_pct is not None:
            out.revised_excess_pct = _scale_excess(out.revised_excess_pct, r0, out.revised_prob)
    return out


_MAX_EXCESS_SCALE = DEFAULT_B_BOUNDS[1]


def _scale_excess(excess: float, p_raw: float, p_cal: float) -> float:
    if p_raw == 0.5:
        return excess
    nd = NormalDist()
    raw_edge = nd.inv_cdf(min(1 - PROB_CLIP, max(PROB_CLIP, p_raw)))
    cal_edge = nd.inv_cdf(min(1 - PROB_CLIP, max(PROB_CLIP, p_cal)))
    ratio = min(_MAX_EXCESS_SCALE, max(0.0, cal_edge / raw_edge))
    return excess * ratio


# --------------------------------------------------------------------------- rank IC


class RankIC(NamedTuple):
    mean_ic: float      # mean over decision dates of the cross-sectional Spearman IC (nan if no date qualifies)
    t_stat: float       # mean / standard error (Newey-West with nw_lags); nan with < 2 dates or no variation
    n_dates: int        # decision dates with at least min_names ranked names


def _spearman(x: list[float], y: list[float]) -> float | None:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _newey_west_t(values: list[float], lags: int) -> float:
    n = len(values)
    if n < 2:
        return math.nan
    x = np.asarray(values) - float(np.mean(values))
    lrv = float(np.dot(x, x)) / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(np.dot(x[k:], x[:-k])) / n
    if lrv <= 0:
        return math.nan
    # small-sample variance uses n - 1 when there is no autocorrelation correction
    se = math.sqrt(lrv * n / (n - 1) / n) if lags == 0 else math.sqrt(lrv / n)
    return float(np.mean(values)) / se


def _scores_by_date(records: Iterable[ForecastRecord], as_of: str, agent: str | None, include_pre_cutoff: bool,
                    models: Iterable[str] | None) -> dict[str, dict[str, tuple[float, float]]]:
    """Decision date -> ticker -> (score, realised excess %) from resolved, point-in-time
    records (see :func:`resolved_rank_ic` for how ``agent=None`` pools the analysts)."""
    model_set = frozenset(models) if models is not None else None
    recs = _eligible(list(records), as_of, agent, include_pre_cutoff, model_set, need_excess=True)
    if agent is None:
        recs = [r for r in recs if r.agent not in POOLED_AGENTS]
    # (date, ticker) -> agent -> logits
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    excess: dict[tuple[str, str], float] = {}
    for r in recs:
        key = (_day(r.as_of), r.ticker)
        cells[key][r.agent].append(float(_logit(float(r.prob_outperform))))
        excess[key] = float(r.realized_excess_return_pct)  # type: ignore[arg-type]
    by_date: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (d, t), per_agent in sorted(cells.items()):
        score = float(np.mean([np.mean(v) for _, v in sorted(per_agent.items())]))
        by_date[d][t] = (score, excess[(d, t)])
    return by_date


def _date_ic(rows: dict[str, tuple[float, float]], min_names: int) -> float | None:
    if len(rows) < max(2, min_names):
        return None
    names = sorted(rows)
    return _spearman([rows[t][0] for t in names], [rows[t][1] for t in names])


def resolved_rank_ic(records: Iterable[ForecastRecord], as_of: str, agent: str | None = COMMITTEE_AGENT, *,
                     include_pre_cutoff: bool = False, min_names: int = 3, nw_lags: int = 0,
                     models: Iterable[str] | None = None) -> RankIC:
    """Point-in-time rank IC of probabilities versus realised excess return.

    Uses resolved records with ``resolve_on <= as_of`` (pre-cutoff rule and one record per
    question as in :func:`fit_calibrators`). ``agent``: one agent's records (default the
    pooled ``"committee"`` record); ``None`` pools every analyst except the pooled records
    (:data:`~aifirm.journal.forecasts.POOLED_AGENTS`: ``"committee"``, ``"llm_pool"`` and the
    shadow book's ``"shadow"``): per (decision date, ticker) each agent's samples are
    averaged in log-odds, then agents are averaged with equal weight. For each decision
    date with at least ``min_names`` tickers and non-constant ranks, the Spearman
    correlation between probability and ``realized_excess_return_pct`` is one IC. Returns
    the mean IC, its t-statistic and the number of dates. The t-statistic assumes
    independent dates when ``nw_lags`` is 0; when forecast horizons overlap (decision
    interval shorter than the horizon), pass ``nw_lags = ceil(horizon / interval) - 1`` for
    a Newey-West (Bartlett) standard error. Probability calibration is monotone per agent,
    so it does not change a single agent's rank IC; it can change the pooled one."""
    by_date = _scores_by_date(records, _day(as_of), agent, include_pre_cutoff, models)
    ics = [ic for d in sorted(by_date) if (ic := _date_ic(by_date[d], min_names)) is not None]
    if not ics:
        return RankIC(math.nan, math.nan, 0)
    return RankIC(float(np.mean(ics)), _newey_west_t(ics, max(0, int(nw_lags))), len(ics))


class PairedIC(NamedTuple):
    mean_diff: float    # mean over dates of IC(agent_a) - IC(agent_b) on their common names (nan without dates)
    t_stat: float       # t-statistic of that mean (Newey-West with nw_lags); nan with < 2 dates or no variation
    n_dates: int        # dates on which both ICs exist
    mean_a: float       # mean IC of agent_a over those dates
    mean_b: float       # mean IC of agent_b over those dates


def paired_rank_ic(records: Iterable[ForecastRecord], as_of: str, agent_a: str, agent_b: str, *,
                   include_pre_cutoff: bool = False, min_names: int = 3, nw_lags: int = 0,
                   models: Iterable[str] | None = None) -> PairedIC:
    """Does ``agent_a`` rank stocks better than ``agent_b``? A paired, point-in-time test.

    For each decision date, both agents' resolved forecasts (``resolve_on <= as_of``,
    pre-cutoff rule as in :func:`fit_calibrators`, one record per question) are restricted
    to the tickers BOTH forecast that day, and each agent's Spearman IC against the realised
    excess return is computed on those common names (at least ``min_names``; a date where
    either ranking is constant is skipped). The per-date difference ``IC_a - IC_b`` is
    averaged over dates, with a t-statistic (Newey-West with ``nw_lags`` when horizons
    overlap). Comparing on common names matters: an agent that only forecasts a hand-picked
    subset must not be compared with another agent's IC over a different cross-section."""
    as_of_s = _day(as_of)
    recs = list(records)
    a = _scores_by_date(recs, as_of_s, agent_a, include_pre_cutoff, models)
    b = _scores_by_date(recs, as_of_s, agent_b, include_pre_cutoff, models)
    diffs, ics_a, ics_b = [], [], []
    for d in sorted(set(a) & set(b)):
        common = sorted(set(a[d]) & set(b[d]))
        ia = _date_ic({t: a[d][t] for t in common}, min_names)
        ib = _date_ic({t: b[d][t] for t in common}, min_names)
        if ia is None or ib is None:
            continue
        diffs.append(ia - ib)
        ics_a.append(ia)
        ics_b.append(ib)
    if not diffs:
        return PairedIC(math.nan, math.nan, 0, math.nan, math.nan)
    return PairedIC(float(np.mean(diffs)), _newey_west_t(diffs, max(0, int(nw_lags))), len(diffs),
                    float(np.mean(ics_a)), float(np.mean(ics_b)))
