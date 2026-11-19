"""Tests for per-agent Platt calibration and point-in-time rank IC (aifirm.journal.calibration)."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pandas as pd
import pytest

from aifirm.journal import Calibrator, calibrate_view, fit_calibrators, resolved_rank_ic
from aifirm.journal.calibration import fit_platt
from aifirm.journal.ledger import canonical_json, to_jsonable
from aifirm.schemas import AnalystView, ForecastRecord, ViewRecord

AS_OF_END = "2031-01-01"


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def make_records(agent: str, n: int, truth, seed: int, start: str = "2024-01-01", per_date: int = 10,
                 pre_cutoff: bool = False, resolve_days: int = 30, samples: int = 1, run: str = "r",
                 model: str = "m") -> list[ForecastRecord]:
    """``n`` questions; the agent states p, the outcome is drawn with probability truth(p)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n // per_date + 1, freq="5B")
    out, k = [], 0
    for d in dates:
        for j in range(per_date):
            if k >= n:
                return out
            stated = float(np.clip(0.5 + rng.normal(0, 0.1), 0.05, 0.95))
            y = int(rng.random() < truth(stated))
            excess = (truth(stated) - 0.5) * 10 + rng.normal(0, 3)
            for s in range(samples):
                out.append(ForecastRecord(
                    id=f"{run}|{d.date()}|T{j}|{agent}|s{s}", run_id=run, agent=agent, model=model, ticker=f"T{j}",
                    benchmark="SPY", as_of=str(d.date()), horizon_days=21,
                    resolve_on=str((d + pd.Timedelta(days=resolve_days)).date()), prob_outperform=stated,
                    expected_excess_return_pct=0.0, pre_cutoff=pre_cutoff, resolved=True, outcome=y,
                    realized_excess_return_pct=excess, brier=(stated - y) ** 2))
            k += 1
    return out


OVERCONFIDENT = lambda p: _sig(0.4 * _logit(p))                      # noqa: E731
CALIBRATED = lambda p: p                                             # noqa: E731
OPTIMISTIC = lambda p: _sig(-0.3 + _logit(p))                        # noqa: E731  (stated too high)
ANTI = lambda p: 1 - p                                               # noqa: E731


# --------------------------------------------------------------------------- Calibrator.apply

def test_identity_returns_input_unchanged_and_nan_is_no_edge():
    c = Calibrator(agent="x")
    assert c.is_identity and c.apply(0.73) == 0.73 and c.apply(0.5) == 0.5
    assert c.apply(float("nan")) == 0.5 and c.apply(None) == 0.5   # type: ignore[arg-type]


@pytest.mark.parametrize("a,b", [(0.5, 1.5), (-0.5, 1.5), (0.5, 0.0), (-0.5, 0.3), (0.2, 1.0), (0.0, 0.0)])
def test_apply_never_flips_or_creates_a_direction_and_is_monotone(a, b):
    c = Calibrator(agent="x", a=a, b=b, fitted=True)
    grid = [i / 200 for i in range(1, 200)]
    out = [c.apply(p) for p in grid]
    for p, q in zip(grid, out):
        assert (q - 0.5) * (p - 0.5) >= 0 and 0.02 <= q <= 0.98
    assert c.apply(0.5) == 0.5
    assert all(y >= x - 1e-12 for x, y in zip(out, out[1:]))


# --------------------------------------------------------------------------- fitting

def test_overconfident_agent_is_shrunk_and_brier_improves():
    recs = make_records("over", 2000, OVERCONFIDENT, 1)
    c = fit_calibrators(recs, AS_OF_END, ["over"])["over"]
    assert c.fitted and 0.25 < c.b < 0.6 and abs(c.a) < 0.15 and c.n == 2000
    assert c.brier_calibrated < c.brier_raw
    assert abs(c.apply(0.7) - 0.5) < abs(0.7 - 0.5)


def test_calibrated_agent_stays_near_identity_and_biased_agent_gets_an_intercept():
    cals = fit_calibrators(make_records("cal", 2000, CALIBRATED, 2) + make_records("opt", 2000, OPTIMISTIC, 3),
                           AS_OF_END, ["cal", "opt"])
    assert abs(cals["cal"].a) < 0.1 and abs(cals["cal"].b - 1) < 0.2
    assert -0.4 < cals["opt"].a < -0.1


def test_anti_skilled_agent_hits_the_lower_slope_bound_and_is_never_inverted():
    c = fit_calibrators(make_records("anti", 2000, ANTI, 4), AS_OF_END, ["anti"])["anti"]
    assert c.b == 0.0 and c.bound.startswith("b=lower")
    assert all((c.apply(p) - 0.5) * (p - 0.5) >= 0 for p in (0.3, 0.45, 0.55, 0.7))


def test_slope_and_intercept_bounds_hold():
    sharp = lambda p: _sig(4 * _logit(p))                                           # noqa: E731
    c = fit_calibrators(make_records("sharp", 3000, sharp, 5), AS_OF_END, ["sharp"])["sharp"]
    assert c.b == 1.5 and -0.5 <= c.a <= 0.5
    a, b, bound = fit_platt([0.6] * 400, [0] * 400)
    assert (a, b) == (-0.5, 0.0) and bound
    a, b, _ = fit_platt([0.6] * 400, [1] * 400, a_bounds=(-0.2, 0.2), b_bounds=(0.5, 1.2))
    assert -0.2 <= a <= 0.2 and 0.5 <= b <= 1.2


def test_small_samples_are_identity_and_shrinkage_grows_with_the_prior():
    recs = make_records("over", 25, OVERCONFIDENT, 6)
    c = fit_calibrators(recs, AS_OF_END, ["over"])["over"]
    assert c.is_identity and not c.fitted and c.n == 25 and "min_n=30" in c.reason
    assert c.brier_raw is not None and c.brier_calibrated == c.brier_raw
    recs = make_records("over", 60, OVERCONFIDENT, 7)
    slopes = [fit_calibrators(recs, AS_OF_END, ["over"], prior_strength=k)["over"].b for k in (0, 50, 500)]
    assert abs(slopes[0] - 1) > abs(slopes[1] - 1) > abs(slopes[2] - 1)
    assert abs(slopes[2] - 1) < 0.15


def test_prior_worth_50_moves_a_30_question_fit_about_n_over_n_plus_prior_of_the_way():
    """Exact data: 15 forecasts at 0.35 (6 hits) and 15 at 0.65 (9 hits): the unshrunk slope is
    logit(0.6)/logit(0.65) = 0.655; a prior worth 50 moves 30/(30+50) = 37.5% of the way."""
    p = [0.35] * 15 + [0.65] * 15
    y = [1] * 6 + [0] * 9 + [1] * 9 + [0] * 6
    _, b0, _ = fit_platt(p, y, prior_strength=0)
    a, b50, _ = fit_platt(p, y, prior_strength=50)
    assert b0 == pytest.approx(math.log(1.5) / _logit(0.65), abs=1e-6)
    assert (1 - b50) / (1 - b0) == pytest.approx(30 / 80, abs=0.03) and abs(a) < 1e-9


def test_point_in_time_future_resolutions_never_influence_the_fit():
    early = make_records("a", 300, OVERCONFIDENT, 9, start="2024-01-01")
    as_of = max(r.resolve_on for r in early)
    base = fit_calibrators(early, as_of, ["a"])["a"]
    # forecasts resolving after as_of with extreme, contradicting outcomes (planted leak)
    leak = make_records("a", 300, ANTI, 10, start=as_of, run="leak")
    leak = [r.model_copy(update={"id": r.id.replace("|T", "|L"), "ticker": "L" + r.ticker}) for r in leak]
    assert min(r.resolve_on for r in leak) > as_of
    assert fit_calibrators(early + leak, as_of, ["a"])["a"] == base
    unresolved = [r.model_copy(update={"id": r.id + "u", "resolved": False, "outcome": None}) for r in early[:50]]
    assert fit_calibrators(early + unresolved, as_of, ["a"])["a"] == base


def test_pre_cutoff_records_follow_the_believability_rule():
    recs = make_records("a", 200, OVERCONFIDENT, 11, pre_cutoff=True)
    assert fit_calibrators(recs, AS_OF_END, ["a"])["a"].n == 0
    assert fit_calibrators(recs, AS_OF_END, ["a"], include_pre_cutoff=True)["a"].n == 200


def test_samples_count_once_and_repeat_runs_are_deduplicated():
    one = make_records("a", 200, OVERCONFIDENT, 12)
    three = make_records("a", 200, OVERCONFIDENT, 12, samples=3)
    c1, c3 = (fit_calibrators(r, AS_OF_END, ["a"])["a"] for r in (one, three))
    assert c3.n == c1.n == 200 and c3.n_records == 600
    assert c3.a == pytest.approx(c1.a, abs=1e-9) and c3.b == pytest.approx(c1.b, abs=1e-9)
    # A question forecast again by another run counts once; which record stands is decided
    # deterministically by (decision date, id), the order of ForecastStore.all().
    rerun = [r.model_copy(update={"id": "r2" + r.id[1:], "run_id": "r2", "outcome": 1 - r.outcome}) for r in one]
    both = fit_calibrators(one + rerun, AS_OF_END, ["a"])["a"]
    assert both.n == 200 and both.n_records == 200 and both == fit_calibrators(rerun, AS_OF_END, ["a"])["a"]


def test_fit_is_deterministic_and_order_independent():
    recs = make_records("a", 500, OVERCONFIDENT, 13) + make_records("b", 500, OPTIMISTIC, 14)
    shuffled = list(recs)
    random.Random(0).shuffle(shuffled)
    a = fit_calibrators(recs, AS_OF_END, ["a", "b"])
    b = fit_calibrators(iter(shuffled), AS_OF_END, ["b", "a"])
    assert a == b


def test_models_filter_and_unknown_agents():
    recs = make_records("a", 100, CALIBRATED, 15, model="old") + make_records("a", 100, CALIBRATED, 16, model="new",
                                                                             run="n", start="2026-01-01")
    assert fit_calibrators(recs, AS_OF_END, ["a"], models=["new"])["a"].n == 100
    missing = fit_calibrators(recs, AS_OF_END, ["nobody"])["nobody"]
    assert missing.is_identity and missing.n == 0 and missing.brier_raw is None


def test_to_dict_is_ledger_safe():
    c = fit_calibrators(make_records("a", 100, OVERCONFIDENT, 17), AS_OF_END, ["a"])["a"]
    d = c.to_dict()
    assert {"agent", "a", "b", "n", "brier_raw", "brier_calibrated", "is_identity", "fitted"} <= set(d)
    json.loads(canonical_json(to_jsonable(d)))


@pytest.mark.parametrize("kw", [dict(min_n=-1), dict(prior_strength=-1), dict(a_bounds=(0.1, 0.5)),
                                dict(b_bounds=(-0.5, 1.5)), dict(b_bounds=(0.0, 0.9))])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        fit_calibrators([], AS_OF_END, ["a"], **kw)


def test_fit_platt_rejects_bad_inputs():
    for args in (([], []), ([0.5], [2]), ([float("nan")], [1]), ([0.5, 0.6], [1])):
        with pytest.raises(ValueError):
            fit_platt(*args)


# --------------------------------------------------------------------------- calibrate_view

def _view(p: float, excess: float, abstain: bool = False) -> ViewRecord:
    v = AnalystView(ticker="AAA", abstain=abstain, abstain_reason="", prob_outperform=p,
                    expected_excess_return_pct=excess, scenarios=[], evidence=[], key_risks=[],
                    what_would_change_my_mind="", private_insight="")
    return ViewRecord(agent="a", model="m", view=v, revised_prob=p, revised_excess_pct=excess)


def test_calibrate_view_returns_a_scaled_copy_and_leaves_the_raw_view_alone():
    cal = Calibrator(agent="a", a=0.0, b=0.5, fitted=True)
    raw = _view(0.7, 4.0)
    out = calibrate_view(raw, cal)
    assert raw.view.prob_outperform == 0.7 and raw.view.expected_excess_return_pct == 4.0
    assert out.view.prob_outperform == pytest.approx(cal.apply(0.7)) and 0.5 < out.view.prob_outperform < 0.7
    assert 0 < out.view.expected_excess_return_pct < 4.0 and out.revised_prob == out.view.prob_outperform
    assert calibrate_view(raw, Calibrator(agent="a")) is raw and calibrate_view(raw, None) is raw
    ab = _view(0.5, 0.0, abstain=True)
    assert calibrate_view(ab, cal) is ab


def test_calibrate_view_neutralised_view_keeps_no_excess_and_scale_is_capped():
    out = calibrate_view(_view(0.52, 3.0), Calibrator(agent="a", a=-0.5, b=1.0, fitted=True))
    assert out.view.prob_outperform == 0.5 and out.view.expected_excess_return_pct == 0.0
    sharp = calibrate_view(_view(0.51, 1.0), Calibrator(agent="a", a=0.5, b=1.5, fitted=True))
    assert sharp.view.expected_excess_return_pct <= 1.5


# --------------------------------------------------------------------------- rank IC

def _ic_records(dates: int, names: int, sign: float, agent: str = "a", start: str = "2024-01-05") -> list[ForecastRecord]:
    out = []
    for i, d in enumerate(pd.bdate_range(start, periods=dates, freq="21B")):
        for j in range(names):
            p = 0.4 + 0.2 * j / (names - 1)
            out.append(ForecastRecord(
                id=f"r|{d.date()}|N{j}|{agent}|s0", run_id="r", agent=agent, model="m", ticker=f"N{j}", benchmark="SPY",
                as_of=str(d.date()), horizon_days=21, resolve_on=str((d + pd.Timedelta(days=30)).date()),
                prob_outperform=p, expected_excess_return_pct=0.0, resolved=True, outcome=int(sign * (p - 0.5) > 0),
                realized_excess_return_pct=sign * (j - names / 2) + 0.6 * ((i * 7 + j * 3) % 5)))
    return out


def test_rank_ic_perfect_and_reversed():
    ic = resolved_rank_ic(_ic_records(12, 6, 1.0), AS_OF_END, "a")
    assert ic.n_dates == 12 and 0.7 < ic.mean_ic < 1.0 and ic.t_stat > 10
    ic = resolved_rank_ic(_ic_records(12, 6, -1.0), AS_OF_END, "a")
    assert -1.0 < ic.mean_ic < -0.7


def test_rank_ic_is_point_in_time():
    recs = _ic_records(12, 6, 1.0)
    as_of = sorted({r.resolve_on for r in recs})[4]
    ic = resolved_rank_ic(recs, as_of, "a")
    assert ic.n_dates == 5
    assert resolved_rank_ic(recs, "2020-01-01", "a") == (pytest.approx(math.nan, nan_ok=True),
                                                         pytest.approx(math.nan, nan_ok=True), 0)


def test_rank_ic_skips_thin_or_constant_dates_and_nan_t_with_one_date():
    recs = _ic_records(3, 6, 1.0)
    one_date = [r for r in recs if r.as_of == recs[0].as_of]
    ic = resolved_rank_ic(one_date, AS_OF_END, "a")
    assert ic.n_dates == 1 and math.isnan(ic.t_stat)
    thin = [r for r in recs if r.ticker in ("N0", "N1")]
    assert resolved_rank_ic(thin, AS_OF_END, "a").n_dates == 0
    flat = [r.model_copy(update={"prob_outperform": 0.55}) for r in recs]
    assert resolved_rank_ic(flat, AS_OF_END, "a").n_dates == 0


def test_rank_ic_pooled_excludes_the_committee_and_newey_west_runs():
    good, bad = _ic_records(10, 6, 1.0, "good"), _ic_records(10, 6, -1.0, "committee")
    pooled = resolved_rank_ic(good + bad, AS_OF_END, None)
    assert pooled.mean_ic > 0.7
    assert resolved_rank_ic(good + bad, AS_OF_END).mean_ic < -0.7     # default agent: the committee
    nw = resolved_rank_ic(good, AS_OF_END, "good", nw_lags=2)
    assert nw.n_dates == 10 and math.isfinite(nw.t_stat)
