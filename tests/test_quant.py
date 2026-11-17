"""Tests for the deterministic quant analyst (quant/signals.py). Hand-built fact sheets; no network."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from aifirm.data.factsheet import Fact, FactSheet
from aifirm.quant.signals import (FALLBACK_VOL_PCT, PROB_SPREAD, WINSOR_Z, composite_probability, factor_scores,
                                  quant_views)

AS_OF = "2026-01-30"
FACTOR_NAMES = ("momentum", "low_vol", "quality", "value")


def _fs(t, **facts):
    fs = FactSheet(ticker=t, as_of=AS_OF)
    for key, value in facts.items():
        fs.add(Fact(f"{t}.{key}", key, value, "%", AS_OF, "test"))
    return fs


def _z(values):
    x = np.asarray(values, dtype=float)
    return (x - x.mean()) / x.std()


def _universe():
    return [
        _fs("AAA", mom_12_1=10.0, vol_60d=20.0, roe=15.0, operating_income_margin=20.0, earnings_yield=4.0, fcf_yield=3.0),
        _fs("BBB", mom_12_1=20.0, vol_60d=30.0, roe=25.0, operating_income_margin=10.0, earnings_yield=6.0, fcf_yield=5.0),
        _fs("CCC", mom_12_1=30.0, vol_60d=40.0, roe=5.0, operating_income_margin=30.0, earnings_yield=2.0, fcf_yield=4.0),
        _fs("DDD", mom_12_1=40.0, vol_60d=50.0, roe=35.0, operating_income_margin=25.0, earnings_yield=8.0, fcf_yield=1.0),
    ]


# ----------------------------------------------------------------------------- z-scores


def test_single_input_factors_are_cross_sectional_z_scores():
    scores = factor_scores(_universe())
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    np.testing.assert_allclose([scores[t]["momentum"] for t in tickers], _z([10, 20, 30, 40]))
    # low volatility is the z-score of minus vol: the calmest name scores highest
    np.testing.assert_allclose([scores[t]["low_vol"] for t in tickers], _z([-20, -30, -40, -50]))
    assert scores["AAA"]["low_vol"] > scores["DDD"]["low_vol"]


def test_two_input_factor_averages_input_z_scores_then_restandardises():
    scores = factor_scores(_universe())
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    raw_quality = (_z([15, 25, 5, 35]) + _z([20, 10, 30, 25])) / 2
    raw_value = (_z([4, 6, 2, 8]) + _z([3, 5, 4, 1])) / 2
    np.testing.assert_allclose([scores[t]["quality"] for t in tickers], _z(raw_quality))
    np.testing.assert_allclose([scores[t]["value"] for t in tickers], _z(raw_value))
    for f in FACTOR_NAMES:                                    # every factor has unit cross-sectional scale
        vals = np.array([scores[t][f] for t in tickers])
        assert vals.mean() == pytest.approx(0, abs=1e-12) and vals.std() == pytest.approx(1)


def test_composite_is_mean_of_factor_z_scores():
    scores = factor_scores(_universe())
    for s in scores.values():
        assert s["composite"] == pytest.approx(np.mean([s[f] for f in FACTOR_NAMES]))


def test_winsorizes_at_three_standard_deviations():
    fss = [_fs(f"T{i:02d}", mom_12_1=float(i), roe=float(i), operating_income_margin=float(i)) for i in range(19)]
    fss.append(_fs("OUT", mom_12_1=10_000.0, roe=10_000.0, operating_income_margin=10_000.0))
    scores = factor_scores(fss)
    assert scores["OUT"]["momentum"] == WINSOR_Z
    assert scores["OUT"]["quality"] == WINSOR_Z
    assert all(-WINSOR_Z <= s[f] <= WINSOR_Z for s in scores.values() for f in FACTOR_NAMES)
    # the clip happens after standardising: unclipped names keep their plain z-score
    raw = [float(i) for i in range(19)] + [10_000.0]
    assert scores["T05"]["momentum"] == pytest.approx(_z(raw)[5])


# ----------------------------------------------------------------------------- missing data


def test_missing_data_scores_zero_and_is_left_out_of_the_composite():
    fss = _universe() + [_fs("ETF", mom_12_1=25.0, vol_60d=15.0)]           # no fundamentals
    s = factor_scores(fss)["ETF"]
    assert s["quality"] == 0.0 and s["value"] == 0.0
    assert s["composite"] == pytest.approx((s["momentum"] + s["low_vol"]) / 2)


def test_partial_inputs_use_what_is_there():
    fss = _universe()
    fss[0] = _fs("AAA", mom_12_1=10.0, vol_60d=20.0, roe=15.0, earnings_yield=4.0)   # no margin, no FCF yield
    views = quant_views(fss, 21)
    quality = next(e for e in views["AAA"].view.evidence if e.point.startswith("Quality"))
    assert quality.fact_ids == ["AAA.roe"]


def test_non_numeric_and_non_finite_facts_count_as_missing():
    fss = _universe()
    fss[0].facts.append(Fact("AAA.fcf_yield", "bad", "n/a", "text", AS_OF, "test"))
    for f in fss[1].facts:
        if f.id == "BBB.mom_12_1":
            f.value = math.inf                                # bypasses FactSheet.add's filter on purpose
    scores = factor_scores(fss)
    assert scores["BBB"]["momentum"] == 0.0
    np.testing.assert_allclose([scores[t]["momentum"] for t in ("AAA", "CCC", "DDD")], _z([10, 30, 40]))
    assert all(math.isfinite(v) for s in scores.values() for v in s.values())


def test_too_few_names_or_no_dispersion_makes_a_factor_unavailable():
    two = _universe()[:2]
    assert all(v == 0.0 for s in factor_scores(two).values() for v in s.values())
    views = quant_views(two, 21)
    assert all(v.view.abstain and v.view.prob_outperform == 0.5 for v in views.values())
    flat = [_fs(t, mom_12_1=12.0, vol_60d=v) for t, v in (("A1", 20.0), ("A2", 30.0), ("A3", 40.0))]
    s = factor_scores(flat)
    assert all(x["momentum"] == 0.0 for x in s.values())     # identical momentum carries no information
    assert s["A1"]["composite"] == pytest.approx(s["A1"]["low_vol"])


def test_name_without_any_factor_abstains():
    fss = _universe() + [_fs("NEW")]
    rec = quant_views(fss, 21)["NEW"]
    assert rec.view.abstain and rec.view.abstain_reason
    assert rec.view.prob_outperform == 0.5 and rec.view.expected_excess_return_pct == 0.0
    assert factor_scores(fss)["NEW"]["composite"] == 0.0


# ----------------------------------------------------------------------------- probability mapping


def test_probability_mapping_is_monotone_and_bounded():
    grid = np.linspace(-10, 10, 401)
    probs = [composite_probability(c) for c in grid]
    assert all(b > a for a, b in zip(probs, probs[1:]))
    assert all(0.5 - PROB_SPREAD < p < 0.5 + PROB_SPREAD for p in probs)
    assert composite_probability(0.0) == 0.5


def test_better_composite_gives_higher_probability():
    fss = _universe()
    views = quant_views(fss, 21)
    scores = factor_scores(fss)
    by_composite = sorted(scores, key=lambda t: scores[t]["composite"])
    probs = [views[t].view.prob_outperform for t in by_composite]
    assert all(b > a for a, b in zip(probs, probs[1:]))
    for t in scores:
        assert views[t].view.prob_outperform == pytest.approx(0.5 + 0.06 * math.tanh(scores[t]["composite"]))


def test_improving_one_input_raises_that_names_probability():
    base = quant_views(_universe(), 21)["AAA"].view.prob_outperform
    better = _universe()
    better[0] = _fs("AAA", mom_12_1=35.0, vol_60d=20.0, roe=15.0, operating_income_margin=20.0,
                    earnings_yield=4.0, fcf_yield=3.0)
    assert quant_views(better, 21)["AAA"].view.prob_outperform > base


def test_probabilities_stay_in_bounds_at_the_winsor_limit():
    fss = [_fs(f"T{i:02d}", mom_12_1=float(i), vol_60d=30.0 + i) for i in range(19)]
    fss.append(_fs("TOP", mom_12_1=1e6, vol_60d=1e-3))
    fss.append(_fs("BOT", mom_12_1=-1e6, vol_60d=1e6))
    views = quant_views(fss, 21)
    probs = {t: v.view.prob_outperform for t, v in views.items()}
    assert all(0.5 - PROB_SPREAD < p < 0.5 + PROB_SPREAD for p in probs.values())
    assert probs["TOP"] == max(probs.values()) and probs["BOT"] == min(probs.values())


# ----------------------------------------------------------------------------- view contents


def test_expected_excess_follows_sigma_h_times_inverse_normal():
    fss = _universe()
    for h in (5, 21, 63):
        views = quant_views(fss, h)
        for fs in fss:
            v = views[fs.ticker].view
            sigma_h = fs.get(f"{fs.ticker}.vol_60d").value * math.sqrt(h / 252)
            assert v.expected_excess_return_pct == pytest.approx(sigma_h * norm.ppf(v.prob_outperform))
            assert math.copysign(1, v.expected_excess_return_pct) == math.copysign(1, v.prob_outperform - 0.5)


def test_missing_volatility_uses_the_fallback():
    fss = _universe()[:3] + [_fs("NOVOL", mom_12_1=50.0)]
    v = quant_views(fss, 21)["NOVOL"].view
    sigma_h = FALLBACK_VOL_PCT * math.sqrt(21 / 252)
    assert v.expected_excess_return_pct == pytest.approx(sigma_h * norm.ppf(v.prob_outperform))
    assert any("volatility unavailable" in r for r in v.key_risks)


def test_scenarios_are_consistent_with_the_expected_excess():
    for rec in quant_views(_universe(), 21).values():
        v = rec.view
        names = [s.name for s in v.scenarios]
        assert names == ["bull", "base", "bear"]
        assert sum(s.probability for s in v.scenarios) == pytest.approx(1.0)
        mean = sum(s.probability * s.excess_return_pct for s in v.scenarios)
        assert mean == pytest.approx(v.expected_excess_return_pct)
        bull, base, bear = (s.excess_return_pct for s in v.scenarios)
        assert bull > base > bear


def test_view_record_metadata_and_evidence_cite_real_fact_ids():
    fss = _universe()
    views = quant_views(fss, 21)
    assert set(views) == {"AAA", "BBB", "CCC", "DDD"}
    for fs in fss:
        rec = views[fs.ticker]
        assert rec.agent == "quant" and rec.model == "deterministic" and rec.sample == 0
        assert rec.grounding_ok and rec.view.ticker == fs.ticker and not rec.view.abstain
        assert len(rec.view.evidence) == 4
        for e in rec.view.evidence:
            assert e.fact_ids and all(i in fs.ids() for i in e.fact_ids)
        assert {i for e in rec.view.evidence for i in e.fact_ids} == {
            f"{fs.ticker}.{k}" for k in ("mom_12_1", "vol_60d", "roe", "operating_income_margin",
                                         "earnings_yield", "fcf_yield")}
    assert views["DDD"].view.evidence[0].direction == "positive"      # strongest momentum
    assert views["AAA"].view.evidence[0].direction == "negative"      # weakest momentum


def test_deterministic_and_independent_of_input_order():
    fss = _universe()
    assert factor_scores(fss) == factor_scores(list(reversed(fss)))
    a = quant_views(fss, 21)
    b = quant_views(list(reversed(fss)), 21)
    assert {t: v.model_dump() for t, v in a.items()} == {t: v.model_dump() for t, v in b.items()}


def test_rejects_duplicates_and_bad_horizon():
    fss = _universe()
    with pytest.raises(ValueError):
        factor_scores(fss + [fss[0]])
    with pytest.raises(ValueError):
        quant_views(fss, 0)
