"""Tests for statistical pooling of analyst views (portfolio/aggregate.py)."""

import math

import numpy as np
import pytest

from aifirm.portfolio.aggregate import aggregate_views, pool_probabilities, weighted_median, weighted_std
from aifirm.schemas import AnalystView, ViewRecord


def _logit(p):
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _view(agent, prob, excess=0.0, weight=1.0, abstain=False, revised_prob=None, revised_excess=None):
    v = AnalystView(ticker="AAA", abstain=abstain, abstain_reason="thin data" if abstain else "",
                    prob_outperform=prob, expected_excess_return_pct=excess, scenarios=[], evidence=[],
                    key_risks=[], what_would_change_my_mind="", private_insight="")
    return ViewRecord(agent=agent, model="test", view=v, weight=weight, revised_prob=revised_prob,
                      revised_excess_pct=revised_excess)


# ----------------------------------------------------------------------------- pool_probabilities


def test_trim_zero_unit_weights_is_simple_logit_mean():
    probs = [0.4, 0.55, 0.6, 0.7, 0.52]
    expected = _sigmoid(np.mean([_logit(p) for p in probs]))
    assert pool_probabilities(probs, trim=0.0) == pytest.approx(expected, abs=1e-12)


def test_trim_zero_weighted_is_weighted_logit_mean():
    probs, w = [0.4, 0.6, 0.8], [1.0, 2.0, 3.0]
    expected = _sigmoid(sum(wi * _logit(p) for p, wi in zip(probs, w, strict=True)) / sum(w))
    assert pool_probabilities(probs, w, trim=0.0) == pytest.approx(expected, abs=1e-12)


def test_weights_shift_the_pool_toward_heavier_member():
    probs = [0.4, 0.7]
    eq = pool_probabilities(probs, trim=0.0)
    toward_high = pool_probabilities(probs, [1.0, 3.0], trim=0.0)
    toward_low = pool_probabilities(probs, [3.0, 1.0], trim=0.0)
    assert toward_low < eq < toward_high


def test_trimming_drops_whole_extremes_with_unit_weights():
    probs = [0.05, 0.5, 0.55, 0.6, 0.99]   # n=5, trim 0.2 -> exactly one member off each tail
    expected = _sigmoid(np.mean([_logit(p) for p in [0.5, 0.55, 0.6]]))
    assert pool_probabilities(probs, trim=0.2) == pytest.approx(expected, abs=1e-12)


def test_fractional_trim_at_boundary_with_four_members():
    probs = [0.3, 0.5, 0.6, 0.9]   # total weight 4, cut 0.8 from each tail
    x = sorted(_logit(p) for p in probs)
    expected = _sigmoid((0.2 * x[0] + x[1] + x[2] + 0.2 * x[3]) / 2.4)
    assert pool_probabilities(probs, trim=0.2) == pytest.approx(expected, abs=1e-12)


def test_no_trimming_below_four_members():
    probs = [0.3, 0.6, 0.95]
    assert pool_probabilities(probs, trim=0.4) == pytest.approx(pool_probabilities(probs, trim=0.0))


def test_trimming_resists_a_single_outlier():
    honest = [0.55, 0.56, 0.57, 0.58]
    assert abs(pool_probabilities([*honest, 0.999], trim=0.2) - 0.565) < 0.02
    assert abs(pool_probabilities([*honest, 0.999], trim=0.0) - 0.565) > 0.05


def test_extremes_are_clipped_and_finite():
    p = pool_probabilities([0.0, 1.0, 1.0], trim=0.0)
    assert math.isfinite(p)
    assert p == pytest.approx(_sigmoid((_logit(0.01) + 2 * _logit(0.99)) / 3))


def test_pool_is_symmetric_and_within_member_range():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        probs = rng.uniform(0.02, 0.98, n)
        w = rng.uniform(0.1, 3.0, n)
        trim = float(rng.uniform(0, 0.45))
        p = pool_probabilities(probs, w, trim)
        assert probs.min() - 1e-12 <= p <= probs.max() + 1e-12
        assert pool_probabilities(1 - probs, w, trim) == pytest.approx(1 - p, abs=1e-12)


def test_empty_and_unusable_inputs_fail_closed_to_half():
    assert pool_probabilities([]) == 0.5
    assert pool_probabilities([0.8, 0.9], [0.0, 0.0]) == 0.5
    assert pool_probabilities([float("nan")]) == 0.5
    # a broken member loses its vote instead of poisoning the pool
    assert pool_probabilities([0.7, float("nan")], trim=0.0) == pytest.approx(0.7)
    assert pool_probabilities([0.7, 0.2], [1.0, float("nan")], trim=0.0) == pytest.approx(0.7)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        pool_probabilities([0.5], trim=0.5)
    with pytest.raises(ValueError):
        pool_probabilities([0.5], trim=-0.1)
    with pytest.raises(ValueError):
        pool_probabilities([0.5, 0.6], [1.0])


# ----------------------------------------------------------------------------- helpers


def test_weighted_median_and_std():
    assert weighted_median([1, 2, 3, 4], [1, 1, 1, 1]) == pytest.approx(2.5)
    assert weighted_median([1, 2, 3], [1, 1, 1]) == pytest.approx(2.0)
    assert weighted_median([1, 2, 3], [1, 1, 5]) == pytest.approx(3.0)
    assert weighted_median([], []) == 0.0
    assert weighted_std([0.4, 0.6], [1, 1]) == pytest.approx(0.1)
    assert weighted_std([0.4, 0.6], [3, 1]) == pytest.approx(math.sqrt(0.75 * 0.05 ** 2 + 0.25 * 0.15 ** 2))
    assert weighted_std([0.7], [1]) == 0.0


# ----------------------------------------------------------------------------- aggregate_views


def test_abstentions_are_excluded_and_counted():
    views = [_view("a", 0.6, 2.0), _view("b", 0.7, 4.0), _view("c", 0.5, 0.0, abstain=True)]
    agg = aggregate_views("AAA", views, trim=0.0)
    assert agg.n_views == 2 and agg.n_abstained == 1
    assert agg.prob_outperform == pytest.approx(pool_probabilities([0.6, 0.7], trim=0.0))
    assert agg.expected_excess_return_pct == pytest.approx(3.0)


def test_equal_weight_shadow_ignores_believability():
    views = [_view("a", 0.45, weight=0.25), _view("b", 0.65, weight=2.0)]
    agg = aggregate_views("AAA", views, trim=0.0)
    assert agg.prob_equal_weight == pytest.approx(pool_probabilities([0.45, 0.65], trim=0.0))
    assert agg.prob_outperform == pytest.approx(pool_probabilities([0.45, 0.65], [0.25, 2.0], trim=0.0))
    assert agg.prob_outperform > agg.prob_equal_weight


def test_dispersion_is_weighted_std_of_member_probs():
    views = [_view("a", 0.4, weight=1.0), _view("b", 0.6, weight=3.0)]
    agg = aggregate_views("AAA", views)
    mean = 0.25 * 0.4 + 0.75 * 0.6
    assert agg.dispersion == pytest.approx(math.sqrt(0.25 * (0.4 - mean) ** 2 + 0.75 * (0.6 - mean) ** 2))


def test_expected_excess_is_weighted_median():
    views = [_view("a", 0.6, 1.0), _view("b", 0.6, 2.0), _view("c", 0.6, 40.0, weight=0.5)]
    assert aggregate_views("AAA", views).expected_excess_return_pct == pytest.approx(2.0)
    heavy = [_view("a", 0.6, 1.0), _view("b", 0.6, 2.0), _view("c", 0.6, 40.0, weight=5.0)]
    assert aggregate_views("AAA", heavy).expected_excess_return_pct == pytest.approx(40.0)


def test_no_views_or_all_abstained_is_neutral():
    for views in ([], [_view("a", 0.9, 5.0, abstain=True)]):
        agg = aggregate_views("AAA", views)
        assert (agg.prob_outperform, agg.prob_equal_weight, agg.expected_excess_return_pct) == (0.5, 0.5, 0.0)
        assert agg.n_views == 0 and agg.dispersion == 0.0


def test_zero_total_weight_fails_closed_but_shadow_still_reported():
    views = [_view("a", 0.7, 3.0, weight=0.0), _view("b", 0.8, 5.0, weight=0.0)]
    agg = aggregate_views("AAA", views, trim=0.0)
    assert agg.prob_outperform == 0.5 and agg.expected_excess_return_pct == 0.0
    assert agg.prob_equal_weight == pytest.approx(pool_probabilities([0.7, 0.8], trim=0.0))
    assert agg.dispersion == pytest.approx(0.05)


def test_use_revised_repools_and_counts_flips():
    views = [_view("a", 0.62, 3.0, revised_prob=0.45, revised_excess=-1.0),
             _view("b", 0.58, 2.0, revised_prob=0.55),
             _view("c", 0.40, -2.0)]   # did not revise: original value is used
    agg = aggregate_views("AAA", views, trim=0.0, use_revised=True)
    assert agg.prob_outperform == pytest.approx(pool_probabilities([0.45, 0.55, 0.40], trim=0.0))
    assert agg.expected_excess_return_pct == pytest.approx(-1.0)   # median of [-1, 2, -2]
    assert agg.challenged and agg.flips == 1
    original = aggregate_views("AAA", views, trim=0.0)
    assert not original.challenged and original.flips == 0
    assert original.prob_outperform == pytest.approx(pool_probabilities([0.62, 0.58, 0.40], trim=0.0))
