"""Statistical pooling of sealed analyst forecasts (P4, P7).

Why statistics instead of an LLM "consensus" (P4): independent estimates that are
averaged keep the wisdom-of-crowds error cancellation, whereas a conversation lets the
first or loudest view anchor everyone else (Lorenz et al. 2011; "Debate or Vote",
NeurIPS 2025). The pool is therefore a fixed formula that no agent can argue with.

Why log-odds: probabilities near 0 or 1 carry much more information than their linear
distance suggests, and averaging in logit space is the standard choice for combining
forecasters (Satopää et al. 2014; Good Judgment Project aggregation).

Why trim: a single overconfident or broken forecaster should not move the pool. We drop
``trim`` of the total *weight* from each tail (fractionally at the boundary, so the
result is continuous in the weights) once there are at least four members.

Why weights plus an equal-weight shadow (P7): believability weights earned from scored
track records should count, but they are themselves estimates, so the equal-weight pool
is computed alongside every time and logged. If the weighted pool stops beating the
shadow, the weighting is not earning its keep.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..schemas import AggregatedView, ViewRecord, clamp

PROB_CLIP = (0.01, 0.99)   # a single 0 or 1 would be +/- infinity in log-odds
MIN_MEMBERS_TO_TRIM = 4    # with three or fewer views, trimming would discard a real opinion


def pool_probabilities(probs: Sequence[float], weights: Sequence[float] | None = None,
                       trim: float = 0.2) -> float:
    """Weighted, trimmed mean of ``probs`` in log-odds space.

    Members with a non-finite probability or a non-positive / non-finite weight are
    ignored (a broken input should lose its vote, not poison the pool). With no usable
    member the result is 0.5, i.e. "no edge", which sizes to no position (fail closed).

    ``trim`` is the fraction of total weight removed from *each* tail, after sorting by
    log-odds, when at least four members remain. The member straddling the cut keeps the
    part of its weight that lies inside the kept band. With unit weights and ``trim=0``
    this is exactly the plain mean of the logits.
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    p = np.asarray(probs, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    if p.shape != w.shape:
        raise ValueError(f"{len(p)} probabilities but {len(w)} weights")
    usable = np.isfinite(p) & np.isfinite(w) & (w > 0)
    p, w = p[usable], w[usable]
    if p.size == 0:
        return 0.5
    p = np.clip(p, *PROB_CLIP)
    x = np.log(p / (1.0 - p))
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    total = float(w.sum())
    cut = trim * total if p.size >= MIN_MEMBERS_TO_TRIM else 0.0
    upper = np.cumsum(w)
    lower = upper - w
    # Length of each member's weight interval [lower, upper] inside the kept band [cut, total - cut].
    kept = np.clip(np.minimum(upper, total - cut) - np.maximum(lower, cut), 0.0, None)
    pooled_logit = float((kept * x).sum() / kept.sum())
    return 1.0 / (1.0 + math.exp(-pooled_logit))


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median; averages the two middle values when the cumulative weight splits
    exactly in half (so unit weights reproduce the ordinary median). Members with a
    non-positive weight are ignored; with none left the result is 0.0."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return 0.0
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    half = cum[-1] / 2.0
    i = int(np.searchsorted(cum, half, side="left"))
    i = min(i, v.size - 1)
    if i + 1 < v.size and math.isclose(cum[i], half, rel_tol=1e-12, abs_tol=1e-15):
        return float((v[i] + v[i + 1]) / 2.0)
    return float(v[i])


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    """Population (biased) weighted standard deviation; 0.0 for fewer than two members."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size < 2:
        return 0.0
    mean = float((w * v).sum() / w.sum())
    return float(math.sqrt((w * (v - mean) ** 2).sum() / w.sum()))


def _member_prob(v: ViewRecord, use_revised: bool) -> float:
    p = v.revised_prob if use_revised and v.revised_prob is not None else v.view.prob_outperform
    return clamp(p, 0.0, 1.0)  # NaN -> 0.5: a broken number becomes "no edge", never a bet


def _member_excess(v: ViewRecord, use_revised: bool) -> float:
    x = v.revised_excess_pct if use_revised and v.revised_excess_pct is not None else v.view.expected_excess_return_pct
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _member_weight(v: ViewRecord) -> float:
    try:
        w = float(v.weight)
    except (TypeError, ValueError):
        return 0.0
    return w if math.isfinite(w) and w > 0 else 0.0


def _crossed_half(v: ViewRecord) -> bool:
    """True when the revision moved the analyst to the other side of 0.5 (a strict crossing)."""
    if v.revised_prob is None:
        return False
    return (clamp(v.view.prob_outperform, 0.0, 1.0) - 0.5) * (clamp(v.revised_prob, 0.0, 1.0) - 0.5) < 0


def aggregate_views(ticker: str, views: list[ViewRecord], trim: float = 0.2,
                    use_revised: bool = False) -> AggregatedView:
    """Pool one ticker's non-abstaining views into an :class:`AggregatedView`.

    * ``prob_outperform``: believability-weighted trimmed log-odds mean (``v.weight``).
    * ``prob_equal_weight``: the same pool with unit weights, the shadow logged beside it (P7).
    * ``expected_excess_return_pct``: weighted median of member expected excess returns
      (a median, because one analyst's +40% scenario should not drive sizing).
    * ``dispersion``: weighted standard deviation of member probabilities. It triggers the
      challenge round and shrinks position size downstream.

    Abstentions are excluded and counted in ``n_abstained``. With ``use_revised`` the
    post-challenge revised values replace the originals where present, ``challenged`` is
    set and ``flips`` counts members whose revision crossed 0.5.

    If no member carries positive weight, the weighted pool has no information: prob 0.5
    and excess 0 (fail closed). Dispersion and the equal-weight shadow still describe the
    members, using unit weights.
    """
    members = [v for v in views if not v.view.abstain]
    n_abstained = len(views) - len(members)
    flips = sum(_crossed_half(v) for v in members) if use_revised else 0
    if not members:
        return AggregatedView(ticker=ticker, prob_outperform=0.5, prob_equal_weight=0.5,
                              expected_excess_return_pct=0.0, dispersion=0.0, n_views=0,
                              n_abstained=n_abstained, challenged=use_revised, flips=flips)
    probs = [_member_prob(v, use_revised) for v in members]
    excess = [_member_excess(v, use_revised) for v in members]
    weights = [_member_weight(v) for v in members]
    unit = [1.0] * len(members)
    has_weight = sum(weights) > 0
    return AggregatedView(
        ticker=ticker,
        prob_outperform=pool_probabilities(probs, weights, trim),
        prob_equal_weight=pool_probabilities(probs, None, trim),
        expected_excess_return_pct=weighted_median(excess, weights) if has_weight else 0.0,
        dispersion=weighted_std(probs, weights if has_weight else unit),
        n_views=len(members),
        n_abstained=n_abstained,
        challenged=use_revised,
        flips=flips,
    )
