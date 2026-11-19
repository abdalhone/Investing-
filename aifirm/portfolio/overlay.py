"""Bounded LLM overlay on the deterministic quant core (RESEARCH 6.6).

Why
---
The best-supported hybrid is a factor core with the LLM as a *bounded* residual overlay.
In Track-or-Follow, text features added no rank information at a weekly horizon (delta IC
-0.016, interval excluding zero), and bounding the text signal is what made that harmless:
text could reorder names inside the top band but never promote a low-ranked one.

The rule (a residual, not a level comparison)
---------------------------------------------
::

    p_final = clip(p_quant + clip(p_llm - 0.5, -max_tilt, +max_tilt), 0, 1)

* ``p_quant`` is the deterministic quant's probability (the core). When the quant has no
  view on a name (it abstained), the core is 0.5 ("no view"), so the LLM pool can still
  move a held name by at most ``max_tilt`` but never further.
* ``p_llm`` is the trimmed log-odds pool of the LLM analysts' calibrated, tier-gated
  probabilities. When every LLM analyst abstains, ``p_final = p_quant`` exactly.
* The tilt is the LLM pool's own signed edge, ``p_llm - 0.5``, not its distance from the
  quant. An LLM pool at 0.5 says "no information": a view the tier gate capped at 0.5, an
  analyst whose calibration map shrank it to 0.5 because it has shown no skill, or an
  analyst that states no edge. Such a pool tilts by 0 and leaves the quant core exactly
  where it is; it is never read as a vote that the stock will lag. (Comparing levels,
  ``p_llm - p_quant``, would turn every neutralised LLM pool into a cut of every quant
  pick toward 0.5, so the tier gate and calibration, which exist to stop unverified or
  unskilled views from adding exposure, would instead sell the quant core.) A bullish
  pool can only raise the core and a bearish one only lower it, each by at most
  ``max_tilt``.
* The expected excess return that sizing uses is derived from ``p_final`` on the quant's
  own scale, ``sigma_h * Phi^-1(p_final)``, so the LLM's own (unbounded) expected-excess
  numbers never reach sizing; they are logged with the views.
* Everything after the overlay can only shrink: the IC memo multiplier, the dispersion
  shrink, the LLM risk review and the risk engine. With the broad screen, names outside the
  deep-dive band never reach the committee, so the LLM can reorder and cut inside the top
  band but cannot promote a name from outside it.

The adaptive bound (:func:`adaptive_bound`)
-------------------------------------------
From resolved forecasts only (point-in-time), the per-date rank IC of the pooled LLM
probability (``llm_pool`` records) is compared with the quant's on the names both
forecast (:func:`aifirm.journal.calibration.paired_rank_ic`). With at least
``overlay_min_dates`` such dates: if the t-statistic of ``IC_llm - IC_quant`` is below
-1, the bound is halved; below -2, it is 0 (the LLM no longer moves probabilities). It is
never raised above the configured maximum. The evidence is returned for the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from scipy.stats import norm

from ..config import FirmConfig
from ..journal.calibration import paired_rank_ic
from ..journal.forecasts import LLM_POOL_AGENT
from ..schemas import AggregatedView, ForecastRecord, ViewRecord
from .aggregate import aggregate_views

QUANT_AGENT = "quant"
NEUTRAL = 0.5
PROB_CLIP = 0.01            # keeps Phi^-1(p) finite (same clip as construction)
IC_MIN_NAMES = 5            # fewest common names for one date's paired IC
HALVE_BELOW_T = -1.0        # t(IC_llm - IC_quant) below this halves the tilt
ZERO_BELOW_T = -2.0         # ... below this sets it to 0


def _finite_prob(p: object) -> float | None:
    try:
        v = float(p)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def combine(p_quant: float | None, p_llm: float | None, max_tilt: float) -> tuple[float, float]:
    """``(p_final, tilt)`` with ``p_final = clip(core + clip(p_llm - 0.5, -max_tilt, max_tilt), 0, 1)``
    and ``tilt = p_final - core`` (module docstring: the LLM pool's signed edge is the
    residual added to the quant core).

    ``core`` is ``p_quant``, or 0.5 when it is missing or not a valid probability. With no
    valid ``p_llm``, or ``p_llm == 0.5`` (no information), the tilt is 0 and ``p_final``
    is the core exactly. ``|tilt| <= max_tilt``; the tilt has the sign of ``p_llm - 0.5``
    (never against the LLM pool's own direction). ``max_tilt`` must be in [0, 0.5]."""
    if not (isinstance(max_tilt, (int, float)) and 0.0 <= float(max_tilt) <= 0.5):
        raise ValueError(f"max_tilt must be in [0, 0.5], got {max_tilt!r}")
    core = _finite_prob(p_quant)
    core = NEUTRAL if core is None else core
    llm = _finite_prob(p_llm)
    if llm is None:
        return core, 0.0
    edge = max(-float(max_tilt), min(float(max_tilt), llm - NEUTRAL))
    p_final = min(1.0, max(0.0, core + edge))
    return p_final, p_final - core


def overlay_aggregate(ticker: str, quant: ViewRecord | None, llm_views: Sequence[ViewRecord], *, trim: float,
                      max_tilt: float, sigma_h_pct: float,
                      use_revised: bool = False) -> tuple[AggregatedView, AggregatedView | None]:
    """The overlay's final :class:`AggregatedView` and the LLM pool it was built from.

    ``llm_views`` are the LLM analysts' pooling inputs (already calibrated and tier-gated,
    weights set); ``quant`` the quant's view (None or abstaining: core 0.5). ``sigma_h_pct``
    is the stock's horizon volatility in percent
    (:func:`aifirm.quant.signals.horizon_sigma_pct`). Returns ``(final, llm_pool)`` where
    ``llm_pool`` is None without any LLM view. In the final view: ``prob_outperform`` and
    ``prob_equal_weight`` are the overlay of the weighted and equal-weight LLM pools;
    ``expected_excess_return_pct = sigma_h_pct * Phi^-1(prob_outperform)``; ``dispersion``,
    ``challenged`` and ``flips`` are the LLM pool's; ``n_views`` / ``n_abstained`` count the
    LLM members and the quant."""
    llm_agg = aggregate_views(ticker, list(llm_views), trim=trim, use_revised=use_revised) if llm_views else None
    q_ok = quant is not None and not quant.view.abstain
    p_quant = float(quant.view.prob_outperform) if q_ok else None  # type: ignore[union-attr]
    has_llm = llm_agg is not None and llm_agg.n_views > 0
    p_llm = llm_agg.prob_outperform if has_llm else None  # type: ignore[union-attr]
    p_final, tilt = combine(p_quant, p_llm, max_tilt)
    p_eq, _ = combine(p_quant, llm_agg.prob_equal_weight if has_llm else None, max_tilt)  # type: ignore[union-attr]
    p_clip = min(max(p_final, PROB_CLIP), 1.0 - PROB_CLIP)
    sigma = float(sigma_h_pct) if math.isfinite(float(sigma_h_pct)) and sigma_h_pct > 0 else 0.0
    n_llm = llm_agg.n_views if llm_agg is not None else 0
    n_llm_abst = llm_agg.n_abstained if llm_agg is not None else 0
    final = AggregatedView(
        ticker=ticker, prob_outperform=p_final, prob_equal_weight=p_eq,
        expected_excess_return_pct=sigma * float(norm.ppf(p_clip)),
        dispersion=llm_agg.dispersion if has_llm else 0.0,  # type: ignore[union-attr]
        n_views=n_llm + (1 if q_ok else 0), n_abstained=n_llm_abst + (1 if quant is not None and not q_ok else 0),
        challenged=use_revised, flips=llm_agg.flips if llm_agg is not None else 0, combination="overlay",
        prob_quant=p_quant, prob_llm=p_llm, overlay_tilt=tilt,
        max_tilt=float(max_tilt))
    return final, llm_agg


# --------------------------------------------------------------------------- the adaptive bound


@dataclass(frozen=True)
class OverlayBound:
    """The tilt bound in force for one decision, with the evidence behind it (ledger-safe).

    ``max_tilt`` is what the committee uses; ``configured`` the configured maximum. ``n_dates``,
    ``mean_diff`` (mean per-date IC of the LLM pool minus the quant's on their common names),
    ``t_stat`` and ``lags`` (Newey-West) come from :func:`paired_rank_ic`; ``mean_ic_llm`` and
    ``mean_ic_quant`` are the two mean ICs over those dates. NaN statistics are reported as
    None."""

    max_tilt: float
    configured: float
    reason: str
    n_dates: int = 0
    mean_diff: float | None = None
    t_stat: float | None = None
    mean_ic_llm: float | None = None
    mean_ic_quant: float | None = None
    lags: int = 0
    min_dates: int = 0
    n_thin_dates: int = 0        # resolved dates skipped: fewer than IC_MIN_NAMES names with both forecasts

    def to_dict(self) -> dict:
        return asdict(self)


def _nan_none(x: float) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _overlap_lags(horizon_days: int, step_days: int) -> int:
    """ceil(h / r) - 1 Newey-West lags when forecast windows overlap (0 when they do not)."""
    return max(0, math.ceil(horizon_days / max(1, step_days)) - 1)


def adaptive_bound(records: Iterable[ForecastRecord], as_of: str, cfg: FirmConfig, *,
                   include_pre_cutoff: bool = False, min_names: int = IC_MIN_NAMES) -> OverlayBound:
    """The overlay bound for a decision at the close of ``as_of`` (module docstring).

    Only forecasts resolved on or before ``as_of`` count (point-in-time); pre-cutoff LLM
    forecasts are excluded unless ``include_pre_cutoff`` (the same rule as believability
    and calibration). Never returns more than ``cfg.process.overlay_max_tilt``."""
    configured = float(cfg.process.overlay_max_tilt)
    min_dates = int(cfg.process.overlay_min_dates)
    lags = _overlap_lags(cfg.process.horizon_days, cfg.process.rebalance_every_days)
    res = paired_rank_ic(records, as_of, LLM_POOL_AGENT, QUANT_AGENT, include_pre_cutoff=include_pre_cutoff,
                         min_names=min_names, nw_lags=lags)
    stats = dict(n_dates=res.n_dates, mean_diff=_nan_none(res.mean_diff), t_stat=_nan_none(res.t_stat),
                 mean_ic_llm=_nan_none(res.mean_a), mean_ic_quant=_nan_none(res.mean_b), lags=lags,
                 min_dates=min_dates, n_thin_dates=res.n_thin)
    if res.n_dates < min_dates:
        thin = (f"; {res.n_thin} resolved date(s) skipped because fewer than {min_names} names had both forecasts "
                f"(a deep-dive band narrower than {min_names} names can never measure this: widen "
                "universe.screen.deep_dive_top_n)" if res.n_thin else "")
        return OverlayBound(configured, configured, f"{res.n_dates} resolved dates with both an LLM-pool and a quant "
                                                    f"forecast on at least {min_names} common names < {min_dates}"
                                                    f"{thin}: the configured bound applies", **stats)
    t = res.t_stat
    if not math.isfinite(t):
        return OverlayBound(configured, configured, "the IC difference has no variation: the configured bound applies",
                            **stats)
    if t < ZERO_BELOW_T:
        return OverlayBound(0.0, configured, f"t(IC_llm - IC_quant) = {t:.2f} < {ZERO_BELOW_T:g}: the LLM pool ranks "
                                             "worse than the quant, overlay switched off (tilt 0)", **stats)
    if t < HALVE_BELOW_T:
        return OverlayBound(configured / 2.0, configured, f"t(IC_llm - IC_quant) = {t:.2f} < {HALVE_BELOW_T:g}: "
                                                         "bound halved", **stats)
    return OverlayBound(configured, configured, f"t(IC_llm - IC_quant) = {t:.2f}: the configured bound applies",
                        **stats)
