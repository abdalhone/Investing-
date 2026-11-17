"""Tests for forecast records, resolution, scoring, believability and lessons
(journal/forecasts.py, journal/lessons.py). Synthetic data only; no network."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from aifirm.config import FirmConfig, ProcessConfig
from aifirm.data.prices import from_frames
from aifirm.journal.forecasts import (COMMITTEE_AGENT, ForecastStore, believability_weights, is_pre_cutoff,
                                      make_forecast_records, resolve_date, scorecard)
from aifirm.journal.lessons import MAX_LESSON_CHARS, LessonStore
from aifirm.schemas import AggregatedView, AnalystView, ForecastRecord, Lesson, ViewRecord

LLM_MODEL = "claude-opus-5-5"       # cutoff 2026-06-30 in config.MODEL_TRAINING_CUTOFFS
DATES = pd.bdate_range("2026-01-05", periods=40)
H = 5


def _cfg(h=H):
    return FirmConfig(process=ProcessConfig(horizon_days=h))


def _prices(dates=DATES):
    n = len(dates)
    step = np.arange(n)
    adj = pd.DataFrame({"SPY": 100 * 1.001 ** step,          # +0.1%/day
                        "AAA": 50 * 1.003 ** step,           # +0.3%/day: beats SPY
                        "BBB": 80 * 0.998 ** step},          # -0.2%/day: lags SPY
                       index=dates)
    close = adj.copy()
    close["AAA"] = 50.0                                       # flat raw close: only adj_close shows the dividends
    return from_frames(close, adj_close=adj)


def _view(agent, p, model=LLM_MODEL, sample=0, abstain=False, ticker="AAA", excess=1.0):
    v = AnalystView(ticker=ticker, abstain=abstain, abstain_reason="thin" if abstain else "", prob_outperform=p,
                    expected_excess_return_pct=excess, scenarios=[], evidence=[], key_risks=[],
                    what_would_change_my_mind="", private_insight="")
    return ViewRecord(agent=agent, model=model, sample=sample, view=v)


def _agg(p=0.58, ticker="AAA", excess=0.7):
    return AggregatedView(ticker=ticker, prob_outperform=p, prob_equal_weight=p, expected_excess_return_pct=excess,
                          dispersion=0.05, n_views=3, n_abstained=1)


def _rec(agent, p, outcome, resolve_on, *, as_of=None, ticker="AAA", pre_cutoff=False, resolved=True, i=0):
    as_of = as_of or str((pd.Timestamp(resolve_on) - BDay(H)).date())
    return ForecastRecord(id=f"r|{as_of}|{ticker}|{agent}|{i}", run_id="r", agent=agent, model="m", ticker=ticker,
                          benchmark="SPY", as_of=as_of, horizon_days=H, resolve_on=resolve_on, prob_outperform=p,
                          expected_excess_return_pct=0.0, pre_cutoff=pre_cutoff, resolved=resolved,
                          outcome=outcome if resolved else None,
                          brier=(p - outcome) ** 2 if resolved else None)


def _track(agent, probs_outcomes, start="2026-02-02", **kw):
    """One record per (p, outcome), each a distinct question resolving on consecutive business days."""
    days = pd.bdate_range(start, periods=len(probs_outcomes))
    return [_rec(agent, p, o, str(d.date()), ticker=f"T{i}", i=i, **kw)
            for i, ((p, o), d) in enumerate(zip(probs_outcomes, days))]


# ----------------------------------------------------------------------------- resolve_date


def test_resolve_date_counts_trading_days_in_the_calendar():
    cal = DATES.delete(6)                                    # a holiday inside the window
    assert resolve_date(cal[2], 5, cal) == str(cal[7].date())
    assert resolve_date(str(cal[2].date()), 5, [str(d.date()) for d in cal]) == str(cal[7].date())


def test_resolve_date_weekend_counts_from_the_last_close():
    saturday = pd.Timestamp("2026-01-10")
    friday_pos = DATES.get_loc(pd.Timestamp("2026-01-09"))
    assert resolve_date(saturday, 3, DATES) == str(DATES[friday_pos + 3].date())


def test_resolve_date_falls_back_to_business_days_beyond_the_calendar():
    # calendar runs out two days after as_of: count the rest in business days
    as_of = DATES[-3]
    assert resolve_date(as_of, 5, DATES) == str((DATES[-1] + BDay(3)).date())
    # live: the calendar ends at as_of (or is missing)
    assert resolve_date(DATES[-1], 5, DATES) == str((DATES[-1] + BDay(5)).date())
    assert resolve_date("2026-03-02", 5, None) == "2026-03-09"
    assert resolve_date("2026-03-02", 5, []) == "2026-03-09"
    with pytest.raises(ValueError):
        resolve_date("2026-03-02", 0, DATES)


# ----------------------------------------------------------------------------- make_forecast_records


def _committee_views():
    return [_view("technical", 0.62, sample=0), _view("technical", 0.57, sample=1, ticker="STOCK"),
            _view("fundamental", 0.5, abstain=True), _view("quant", 0.53, model="deterministic")]


def test_one_record_per_non_abstaining_view_plus_committee():
    as_of = str(DATES[3].date())
    recs = make_forecast_records("run1", as_of, _committee_views(), _agg(), _cfg(), DATES)
    assert [(r.agent, r.prob_outperform) for r in recs] == [
        ("technical", 0.62), ("technical", 0.57), ("quant", 0.53), (COMMITTEE_AGENT, 0.58)]
    assert len({r.id for r in recs}) == 4
    assert recs[0].id == f"run1|{as_of}|AAA|technical|s0" and recs[1].id.endswith("|technical|s1")
    assert recs[-1].id == f"run1|{as_of}|AAA|committee"
    assert recs[-1].expected_excess_return_pct == 0.7 and recs[-1].model == f"{LLM_MODEL}+deterministic"
    for r in recs:
        assert r.ticker == "AAA" and r.benchmark == "SPY" and r.as_of == as_of and r.horizon_days == H
        assert r.resolve_on == str(DATES[3 + H].date()) and not r.resolved and r.outcome is None


def test_pre_cutoff_flags():
    views = _committee_views()
    before = make_forecast_records("r", "2026-01-08", views, _agg(), _cfg(), DATES)
    assert {r.agent: r.pre_cutoff for r in before} == {"technical": True, "quant": False, COMMITTEE_AGENT: True}
    after = make_forecast_records("r", "2026-08-03", views, _agg(), _cfg(), None)
    assert not any(r.pre_cutoff for r in after)
    assert is_pre_cutoff(LLM_MODEL, "2026-06-30") and not is_pre_cutoff(LLM_MODEL, "2026-07-01")
    assert is_pre_cutoff("some-unknown-model", "2030-01-01")      # unknown cutoff: assume contaminated
    assert not is_pre_cutoff("deterministic", "2001-01-01")


def test_all_abstaining_gives_no_records():
    views = [_view("technical", 0.5, abstain=True), _view("macro", 0.5, abstain=True)]
    assert make_forecast_records("r", "2026-01-08", views, _agg(p=0.5), _cfg(), DATES) == []


# ----------------------------------------------------------------------------- ForecastStore


def test_store_round_trip_persistence_and_idempotent_add(tmp_path):
    path = tmp_path / "db" / "forecasts.sqlite"
    recs = make_forecast_records("r", str(DATES[0].date()), _committee_views(), _agg(), _cfg(), DATES)
    with ForecastStore(path) as store:
        store.add(recs)
        store.add(recs)
        assert store.all() == sorted(recs, key=lambda r: (r.as_of, r.id))
    with ForecastStore(path) as reopened:
        assert len(reopened.all()) == len(recs)


def test_resolve_uses_adjusted_close_total_returns(tmp_path):
    prices = _prices()
    as_of = str(DATES[2].date())
    recs = (make_forecast_records("r", as_of, [_view("technical", 0.7)], _agg(0.7), _cfg(), DATES)
            + make_forecast_records("r", as_of, [_view("technical", 0.7, ticker="BBB")], _agg(0.7, ticker="BBB"),
                                    _cfg(), DATES))
    resolve_on = recs[0].resolve_on
    with ForecastStore(tmp_path / "f.sqlite") as store:
        store.add(recs)
        assert store.resolve(prices, str(DATES[2 + H - 1].date())) == []      # horizon not over yet
        done = store.resolve(prices, resolve_on)
        assert len(done) == 4 and all(r.resolved for r in done)
        adj = prices.adj_close
        for r in done:
            t = r.ticker
            expected = (adj[t][resolve_on] / adj[t][as_of] - adj["SPY"][resolve_on] / adj["SPY"][as_of]) * 100
            assert r.realized_excess_return_pct == pytest.approx(expected)
            assert r.outcome == (1 if t == "AAA" else 0)       # AAA's raw close is flat; adj_close beats SPY
            assert r.brier == pytest.approx((r.prob_outperform - r.outcome) ** 2)
        assert store.resolve(prices, str(DATES[-1].date())) == []             # already resolved
        assert all(r.resolved for r in store.all())


def test_resolve_is_point_in_time_and_waits_for_data(tmp_path):
    prices = _prices()
    recs = make_forecast_records("r", str(DATES[2].date()), [_view("technical", 0.6)], _agg(), _cfg(), DATES)
    resolve_on = recs[0].resolve_on
    with ForecastStore(tmp_path / "f.sqlite") as store:
        store.add(recs)
        # full history passed in, but the query date is before resolve_on: nothing may resolve
        assert store.resolve(prices, str(DATES[5].date())) == []
        # query date after resolve_on, but the price data stop before it: stay pending
        stale = prices.as_of(DATES[2 + H - 1])
        assert store.resolve(stale, str(DATES[-1].date())) == []
        assert len(store.resolve(prices, resolve_on)) == 2


def test_resolve_leaves_missing_prices_pending(tmp_path):
    prices = _prices()
    prices.adj_close.loc[DATES[2 + H], "AAA"] = np.nan                       # halted on the resolution date
    as_of = str(DATES[2].date())
    recs = (make_forecast_records("r", as_of, [_view("technical", 0.6)], _agg(), _cfg(), DATES)
            + make_forecast_records("r", as_of, [_view("technical", 0.6, ticker="ZZZ")], _agg(ticker="ZZZ"),
                                    _cfg(), DATES))
    with ForecastStore(tmp_path / "f.sqlite") as store:
        store.add(recs)
        assert store.resolve(prices, str(DATES[-1].date())) == []
        assert not any(r.resolved for r in store.all())


def test_resolution_is_not_reset_by_re_adding(tmp_path):
    prices = _prices()
    recs = make_forecast_records("r", str(DATES[2].date()), [_view("technical", 0.6)], _agg(), _cfg(), DATES)
    with ForecastStore(tmp_path / "f.sqlite") as store:
        store.add(recs)
        store.resolve(prices, str(DATES[-1].date()))
        store.add(recs)                                                      # unresolved copies
        assert all(r.resolved for r in store.all())


def test_resolve_from_a_weekend_decision_uses_the_prior_close(tmp_path):
    prices = _prices()
    saturday = "2026-01-10"
    recs = make_forecast_records("r", saturday, [_view("technical", 0.6)], _agg(), _cfg(), DATES)
    friday = "2026-01-09"
    assert recs[0].resolve_on == str(DATES[DATES.get_loc(pd.Timestamp(friday)) + H].date())
    with ForecastStore(tmp_path / "f.sqlite") as store:
        store.add(recs)
        r = store.resolve(prices, str(DATES[-1].date()))[0]
    adj = prices.adj_close
    expected = (adj["AAA"][r.resolve_on] / adj["AAA"][friday] - adj["SPY"][r.resolve_on] / adj["SPY"][friday]) * 100
    assert r.realized_excess_return_pct == pytest.approx(expected)


# ----------------------------------------------------------------------------- scorecard


def test_scorecard_metrics():
    pairs = [(0.7, 1), (0.6, 0), (0.4, 0), (0.5, 1), (0.8, 1)]
    card = scorecard(_track("tech", pairs))["tech"]
    probs = np.array([p for p, _ in pairs])
    outs = np.array([o for _, o in pairs])
    brier = float(np.mean((probs - outs) ** 2))
    assert card["n"] == 5 and card["n_questions"] == 5 and card["n_pre_cutoff"] == 0
    assert card["brier"] == pytest.approx(brier)
    assert card["brier_skill"] == pytest.approx((0.25 - brier) / 0.25)
    assert card["log_loss"] == pytest.approx(-np.mean(outs * np.log(probs) + (1 - outs) * np.log(1 - probs)))
    assert card["hit_rate"] == pytest.approx(3 / 4)                  # the 0.5 forecast has no direction
    assert card["base_rate"] == pytest.approx(0.6) and card["mean_prob"] == pytest.approx(0.6)
    buckets = {b["bucket"]: b for b in card["calibration"]}
    assert set(buckets) == {"[0.40, 0.45)", "[0.50, 0.55)", "[0.60, 0.70)", "[0.70, 1.00]"}
    assert buckets["[0.70, 1.00]"]["n"] == 2 and buckets["[0.70, 1.00]"]["observed_rate"] == 1.0
    assert buckets["[0.70, 1.00]"]["mean_prob"] == pytest.approx(0.75)
    assert sum(b["n"] for b in card["calibration"]) == 5


def test_scorecard_is_point_in_time_and_ignores_unresolved():
    recs = (_track("a", [(0.6, 1)] * 3, start="2026-02-02")
            + _track("a", [(0.6, 0)] * 2, start="2026-03-02")
            + [_rec("a", 0.9, 0, "2026-02-03", resolved=False, ticker="P")])
    assert scorecard(recs)["a"]["n"] == 5
    early = scorecard(recs, as_of="2026-02-04")["a"]                # only the first three had resolved
    assert early["n"] == 3 and early["base_rate"] == 1.0
    assert scorecard(recs, as_of="2026-01-30") == {}


def test_scorecard_can_exclude_pre_cutoff_and_sorts_agents():
    recs = _track("zeta", [(0.6, 1)] * 2, pre_cutoff=True) + _track("alpha", [(0.6, 1)])
    assert list(scorecard(recs)) == ["alpha", "zeta"]
    assert scorecard(recs)["zeta"]["n_pre_cutoff"] == 2
    assert list(scorecard(recs, include_pre_cutoff=False)) == ["alpha"]


# ----------------------------------------------------------------------------- believability


def test_weight_is_one_below_min_n():
    recs = _track("a", [(0.9, 1)] * 19)
    assert believability_weights(recs, "2027-01-01", ["a"]) == {"a": 1.0}


def test_weight_formula_with_shrinkage():
    pairs = [(0.6, 1)] * 30 + [(0.6, 0)] * 10                       # brier = 0.22, n = 40
    recs = _track("a", pairs)
    brier = (30 * 0.16 + 10 * 0.36) / 40
    skill = (0.25 - brier) / 0.25
    expected = 1 + 40 / (40 + 30) * 4 * skill
    assert believability_weights(recs, "2027-01-01", ["a"])["a"] == pytest.approx(expected)


def test_weight_is_clipped_to_floor_and_cap():
    good = _track("good", [(0.95, 1)] * 200)
    bad = _track("bad", [(0.95, 0)] * 200)
    w = believability_weights(good + bad, "2027-06-01", ["good", "bad", "nobody"])
    assert w == {"good": 2.0, "bad": 0.25, "nobody": 1.0}
    assert list(w) == ["good", "bad", "nobody"]
    custom = believability_weights(good + bad, "2027-06-01", ["good", "bad"], floor=0.5, cap=1.5)
    assert custom == {"good": 1.5, "bad": 0.5}


def test_weights_exclude_pre_cutoff_by_default():
    recs = _track("a", [(0.95, 1)] * 50, pre_cutoff=True)
    assert believability_weights(recs, "2027-01-01", ["a"]) == {"a": 1.0}
    assert believability_weights(recs, "2027-01-01", ["a"], include_pre_cutoff=True)["a"] > 1.0


def test_weights_never_use_outcomes_resolved_after_the_decision_date():
    recs = _track("a", [(0.6, 1)] * 50, start="2026-03-02")          # resolve 2026-03-02 .. 2026-05-08; below the cap
    last = max(r.resolve_on for r in recs)
    assert believability_weights(recs, "2026-03-01", ["a"]) == {"a": 1.0}
    assert believability_weights(recs, last, ["a"])["a"] > 1.0
    day_before = str((pd.Timestamp(last) - BDay(1)).date())
    n_known = sum(1 for r in recs if r.resolve_on <= day_before)
    w_before = believability_weights(recs, day_before, ["a"])["a"]
    w_last = believability_weights(recs, last, ["a"])["a"]
    assert n_known == 49 and w_before < w_last                        # one more outcome known on the last day


def test_repeated_samples_do_not_count_as_extra_questions():
    days = pd.bdate_range("2026-02-02", periods=15)
    recs = [_rec("a", 0.9, 1, str(d.date()), ticker=f"T{i}", i=s) for i, d in enumerate(days) for s in range(2)]
    assert scorecard(recs)["a"]["n"] == 30 and scorecard(recs)["a"]["n_questions"] == 15
    assert believability_weights(recs, "2027-01-01", ["a"]) == {"a": 1.0}      # 15 questions < min_n 20


def test_believability_rejects_bad_bounds():
    with pytest.raises(ValueError):
        believability_weights([], "2026-01-01", ["a"], floor=1.5, cap=2.0)
    with pytest.raises(ValueError):
        believability_weights([], "2026-01-01", ["a"], min_n=-1)


# ----------------------------------------------------------------------------- LessonStore


def _lesson(agent="technical", text="Stay near 0.50 when momentum and RSI disagree.", when="RSI above 70",
            ticker="NVDA"):
    return Lesson(agent=agent, ticker=ticker, what_happened="", was_it_skill_or_luck="luck",
                  process_lesson=text, applies_when=when)


def test_lessons_are_point_in_time_most_recent_first(tmp_path):
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.add(_lesson(text="old lesson"), "2026-01-10")
    store.add(_lesson(text="newer lesson"), "2026-02-10")
    store.add(_lesson(text="future lesson"), "2026-03-10")
    store.add(_lesson(agent="macro", text="macro lesson"), "2026-01-05")
    got = store.retrieve("technical", "2026-02-10")
    assert got == ["newer lesson (applies when: RSI above 70)", "old lesson (applies when: RSI above 70)"]
    assert store.retrieve("technical", "2026-02-09") == ["old lesson (applies when: RSI above 70)"]
    assert store.retrieve("technical", "2026-01-09") == []
    assert store.retrieve("macro", "2026-12-31") == ["macro lesson (applies when: RSI above 70)"]
    assert store.retrieve("technical", "2026-12-31", k=1) == ["future lesson (applies when: RSI above 70)"]
    assert store.retrieve("technical", "2026-12-31", k=0) == []


def test_lessons_persist_dedupe_and_mask_the_ticker(tmp_path):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    lesson = _lesson(text="After NVDA ran up, weight NVDA.rsi_14 less; ANVDA is unrelated.", ticker="NVDA")
    store.add(lesson, "2026-01-10")
    store.add(lesson, "2026-01-10")                                   # idempotent
    store.add(_lesson(text=lesson.process_lesson, ticker="NVDA"), "2026-01-11")   # same text, other date
    assert len(path.read_text().splitlines()) == 2
    got = LessonStore(path).retrieve("technical", "2026-02-01")
    assert got == ["After STOCK ran up, weight STOCK.rsi_14 less; ANVDA is unrelated. (applies when: RSI above 70)"]


def test_lessons_are_short_and_skip_damaged_lines(tmp_path):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.add(_lesson(text="x" * 1000, when=""), "2026-01-10")
    store.add(_lesson(text="   "), "2026-01-11")                      # empty rule: nothing to show
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
        fh.write(json.dumps({"resolved_on": "2026-01-12", "lesson": {"agent": "technical"}}) + "\n")
    got = LessonStore(path).retrieve("technical", "2026-02-01")
    assert len(got) == 1 and len(got[0]) == MAX_LESSON_CHARS and got[0].endswith("...")
