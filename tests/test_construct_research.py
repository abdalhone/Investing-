"""Tests for the research-aligned portfolio and risk defaults (RESEARCH 6.4, 6.5, 9.3 item 5):
the 8% target / 10% hard cap, partial rebalancing, the 20% freeze with half-recovery
re-risking, equal weights until a minimum resolved sample, and the exposure note. No network."""

import pandas as pd
import pytest
from pydantic import ValidationError

from aifirm.config import FirmConfig, PortfolioConfig
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.journal.forecasts import believability_weights
from aifirm.portfolio.construct import construct_portfolio
from aifirm.portfolio.risk_engine import RiskState, apply_risk_limits, update_risk_state
from aifirm.schemas import AggregatedView, ForecastRecord, TargetPortfolio, TargetPosition

AS_OF = "2026-01-30"


def test_defaults_follow_research_section_6_4():
    cfg = FirmConfig()
    pc, rc = cfg.portfolio, cfg.risk
    assert (pc.max_position, rc.max_position_hard, pc.max_sector) == (0.08, 0.10, 0.25)
    assert (pc.target_vol, pc.no_trade_band, pc.max_turnover, pc.rebalance_fraction) == (0.15, 0.005, 0.25, 0.5)
    assert (rc.drawdown_derisk, rc.drawdown_freeze, rc.daily_loss_halt) == (0.10, 0.20, 0.05)
    assert rc.max_adv_participation == 0.02            # stricter than the research's 5% ceiling: kept
    assert cfg.process.combination == "overlay" and cfg.process.overlay_max_tilt == 0.05
    assert cfg.process.believability_min_dates == 20 and cfg.calibration.enabled


def test_the_sizing_target_may_not_exceed_the_hard_cap_and_overlay_needs_the_quant():
    with pytest.raises(ValidationError, match="max_position_hard"):
        FirmConfig(portfolio=PortfolioConfig(max_position=0.12))
    with pytest.raises(ValidationError, match="rebalance_fraction"):
        PortfolioConfig(rebalance_fraction=0.0)
    with pytest.raises(ValidationError, match="must include"):
        FirmConfig.model_validate({"analysts": {"enabled": ["technical"]}})
    FirmConfig.model_validate({"analysts": {"enabled": ["technical"]}, "process": {"combination": "pool"}})
    with pytest.raises(ValidationError, match="combination"):
        FirmConfig.model_validate({"process": {"combination": "vote"}})
    with pytest.raises(ValidationError):
        FirmConfig.model_validate({"process": {"overlay_max_tilt": 0.7}})


def _fs(t, vol=25.0):
    fs = FactSheet(ticker=t, as_of=AS_OF, sector=f"S{t}")
    fs.add(Fact(f"{t}.vol_60d", "vol", vol, "%", AS_OF, "test"))
    fs.add(Fact(f"{t}.price", "price", 100.0, "USD", AS_OF, "test"))
    return fs


def _agg(t, p=0.60, excess=3.0):
    return AggregatedView(ticker=t, prob_outperform=p, prob_equal_weight=p, expected_excess_return_pct=excess,
                          dispersion=0.0, n_views=3, n_abstained=0)


def _build(aggs, current, cfg=None):
    cfg = cfg or FirmConfig()
    sheets = {t: _fs(t) for t in set(aggs) | set(current) if t != "SPY"}
    return construct_portfolio(aggs, {}, sheets, current, cfg, AS_OF, {})


def _w(tp):
    return {p.ticker: p.weight for p in tp.positions if p.weight}


def test_partial_rebalancing_moves_half_way_each_cycle_and_closes_sub_band_gaps():
    aggs = {"AAA": _agg("AAA")}
    first = _build(aggs, {})
    assert _w(first) == {"AAA": pytest.approx(0.04)}                    # half of the 8% target
    pos = next(p for p in first.positions if p.ticker == "AAA")
    assert "partial_rebalance" in pos.constraints_hit and "max_position" in pos.constraints_hit
    assert any(n.startswith("partial rebalancing") for n in first.notes)
    second = _build(aggs, {"AAA": 0.04})
    assert _w(second) == {"AAA": pytest.approx(0.06)}
    third = _build(aggs, {"AAA": 0.06})
    assert _w(third) == {"AAA": pytest.approx(0.07)}
    # 0.075 -> 0.08 would stop at 0.0775, a 0.25% gap the 0.5% band could never trade: completed instead
    assert _w(_build(aggs, {"AAA": 0.075})) == {"AAA": pytest.approx(0.08)}
    full = FirmConfig()
    full.portfolio.rebalance_fraction = 1.0
    assert _w(_build(aggs, {}, full)) == {"AAA": pytest.approx(0.08)}


def test_exits_shrink_in_halves_until_the_remainder_is_below_the_band():
    no_edge = {"AAA": _agg("AAA", p=0.45, excess=-1.0)}
    assert _w(_build(no_edge, {"AAA": 0.06})) == {"AAA": pytest.approx(0.03)}
    assert _w(_build(no_edge, {"AAA": 0.009})) == {}                    # 0.0045 left would be dust: exit fully


def test_forced_exits_go_all_the_way_and_unchanged_holds_are_not_traded():
    tp = _build({"BBB": _agg("BBB")}, {"GONE": 0.07})                    # held, no longer covered
    assert "GONE" not in _w(tp) and _w(tp)["BBB"] == pytest.approx(0.04)
    gone = next(p for p in tp.positions if p.ticker == "GONE")
    assert "forced_exit" in gone.constraints_hit and "partial_rebalance" not in gone.constraints_hit
    abstained = AggregatedView(ticker="CCC", prob_outperform=0.5, prob_equal_weight=0.5, expected_excess_return_pct=0.0,
                               dispersion=0.0, n_views=0, n_abstained=4)
    assert _w(_build({"CCC": abstained}, {"CCC": 0.05})) == {"CCC": pytest.approx(0.05)}


def test_drift_between_the_target_and_the_hard_cap_is_not_force_traded_but_a_hard_breach_is():
    aggs = {"AAA": _agg("AAA")}
    drifted = _build(aggs, {"AAA": 0.095})                              # above the 8% target, below the 10% cap
    assert _w(drifted) == {"AAA": pytest.approx(0.0875)}                # half way back, no forced trim
    breach = _build(aggs, {"AAA": 0.14})                                # above the hard cap
    assert _w(breach)["AAA"] <= 0.10 + 1e-9
    assert any("limit" in h for p in breach.positions for h in p.constraints_hit)


def test_turnover_cap_of_a_quarter_holds_with_partial_rebalancing():
    aggs = {t: _agg(t) for t in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")}
    current = {f"OLD{i}": 0.09 for i in range(8)}
    tp = _build({**aggs, **{t: _agg(t, p=0.40, excess=-2.0) for t in current}}, current)
    new = _w(tp)
    turnover = 0.5 * sum(abs(new.get(t, 0.0) - current.get(t, 0.0)) for t in set(new) | set(current))
    assert turnover <= 0.25 + 1e-9


def test_every_target_explains_its_exposure():
    tp = _build({"AAA": _agg("AAA"), "LOW": _agg("LOW", p=0.51)}, {})
    note = next(n for n in tp.notes if n.startswith("exposure:"))
    assert "1 name(s)" in note and "1 of 2 name(s) cleared min_prob_to_hold 0.52" in note and "cash" in note


# ----------------------------------------------------------------------------- risk engine


def _target(weights, core=0.0):
    return TargetPortfolio(as_of=AS_OF, core_weight=core, cash_weight=1 - core - sum(weights.values()),
                           positions=[TargetPosition(ticker=t, weight=w, prob_outperform=0.6,
                                                     expected_excess_return_pct=1.0, rationale="t")
                                      for t, w in weights.items()])


def _apply(tmp_path, target, state, nav, cfg=None):
    sheets = {p.ticker: _fs(p.ticker) for p in target.positions}
    out, msgs = apply_risk_limits(target, state, nav, None, {}, cfg or FirmConfig(runs_dir=str(tmp_path)), sheets,
                                  str(tmp_path))
    return {p.ticker: p.weight for p in out.positions}, msgs


def test_risk_engine_clips_at_the_hard_cap_not_at_the_sizing_target(tmp_path):
    w, msgs = _apply(tmp_path, _target({"AAA": 0.09, "BBB": 0.13}), RiskState(peak_nav=100.0), 100.0)
    assert w == {"AAA": pytest.approx(0.09), "BBB": pytest.approx(0.10)}
    assert any(m.startswith("BBB") and "position cap" in m for m in msgs)


def test_update_risk_state_tracks_the_trough_since_the_last_peak():
    s = RiskState(peak_nav=100.0)
    for nav in (95.0, 88.0, 93.0):
        s = update_risk_state(s, nav)
    assert (s.peak_nav, s.trough_nav, s.last_nav) == (100.0, 88.0, 93.0)
    s = update_risk_state(s, 101.0)                                     # a new peak restarts the trough
    assert (s.peak_nav, s.trough_nav) == (101.0, 101.0)
    legacy = RiskState(**{"peak_nav": 100.0, "frozen": False, "last_nav": 99.0, "last_as_of": "2026-01-02"})
    assert legacy.trough_nav is None and not legacy.derisked             # an old risk_state.json still loads
    assert update_risk_state(legacy, 97.0).trough_nav == 97.0


def test_derisk_holds_until_half_the_drawdown_is_recovered_then_re_risks(tmp_path):
    """Research 6.4: -10% halves gross, re-risk after recovering half the drawdown. Without the
    hysteresis the book re-risked as soon as the drawdown crept back under 10%."""
    target = _target({"AAA": 0.08})
    s = RiskState(peak_nav=100.0)
    path = []
    for nav in (95.0, 88.0):                                            # trough 88: a 12% drawdown
        s = update_risk_state(s, nav)
    w, msgs = _apply(tmp_path, target, s, 88.0)
    path.append(w["AAA"])
    assert s.derisked and any("drawdown de-risk:" in m for m in msgs)
    s = update_risk_state(s, 91.0)                                      # drawdown 9% < 10%, recovered 3 of 12
    w, msgs = _apply(tmp_path, target, s, 91.0)
    path.append(w["AAA"])
    assert s.derisked and any("still in force" in m and "94.00" in m for m in msgs)
    s = update_risk_state(s, 94.0)                                      # half of the 12-point drawdown recovered
    w, msgs = _apply(tmp_path, target, s, 94.0)
    path.append(w["AAA"])
    assert not s.derisked and any(m.startswith("re-risk") for m in msgs)
    assert path == [pytest.approx(0.04), pytest.approx(0.04), pytest.approx(0.08)]
    s = update_risk_state(s, 93.0)                                      # a dip after re-risking: no new de-risk
    w, msgs = _apply(tmp_path, target, s, 93.0)
    assert w["AAA"] == pytest.approx(0.08) and msgs == []


def test_a_human_unfreeze_also_ends_the_derisk_episode(tmp_path):
    s = update_risk_state(RiskState(peak_nav=100.0), 78.0)
    _apply(tmp_path, _target({"AAA": 0.08}), s, 78.0)
    assert s.frozen and s.derisked is False                              # froze at -22% (deeper than de-risk)
    s.derisked = True                                                    # e.g. a de-risk recorded before the freeze
    s.frozen = False                                                     # the human decision
    w, msgs = _apply(tmp_path, _target({"AAA": 0.08}), s, 78.0)
    assert w == {"AAA": pytest.approx(0.08)} and not s.derisked and s.trough_nav == 78.0
    assert any("unfrozen by a human" in m for m in msgs)


# ----------------------------------------------------------------------------- equal weights until a minimum sample


def _rec(agent, as_of, ticker, p, outcome):
    resolve = str((pd.Timestamp(as_of) + pd.Timedelta(days=30)).date())
    return ForecastRecord(id=f"r|{as_of}|{ticker}|{agent}|s0", run_id="r", agent=agent, model="m", ticker=ticker,
                          benchmark="SPY", as_of=as_of, horizon_days=21, resolve_on=resolve, prob_outperform=p,
                          expected_excess_return_pct=0.0, resolved=True, outcome=outcome, brier=(p - outcome) ** 2)


def test_believability_stays_equal_until_enough_resolved_dates_however_many_names():
    """Research 6.2 / 9.3 item 5: a hundred names resolved on one date are one market episode,
    not a track record of a hundred questions."""
    dates = [str(d.date()) for d in pd.bdate_range("2025-01-02", periods=25, freq="21B")]
    broad = [_rec("a", dates[0], f"T{i}", 0.9, 1) for i in range(100)]
    assert believability_weights(broad, "2026-06-01", ["a"]) == {"a": 1.0}
    assert believability_weights(broad, "2026-06-01", ["a"], min_dates=0)["a"] == 2.0      # the old rule tilted
    long = [_rec("a", d, f"T{i}", 0.9, 1) for d in dates[:20] for i in range(3)]
    assert believability_weights(long, "2027-01-01", ["a"])["a"] > 1.0
    assert believability_weights(long[:-3], "2027-01-01", ["a"]) == {"a": 1.0}             # 19 dates
    with pytest.raises(ValueError):
        believability_weights(long, "2027-01-01", ["a"], min_dates=-1)


def test_an_exit_completes_even_with_a_zero_no_trade_band():
    """Regression: with no_trade_band = 0 (allowed) and partial rebalancing, an exit halved for
    ever (37 cycles for an 8% position, a sell order and a deep dive every cycle). A step that
    would leave less than MIN_RESIDUAL_GAP to trade is completed."""
    from aifirm.data.factsheet import Fact, FactSheet
    from aifirm.portfolio.construct import MIN_RESIDUAL_GAP, construct_portfolio
    from aifirm.schemas import AggregatedView
    as_of = "2026-01-30"
    fs = FactSheet(ticker="AAA", as_of=as_of, sector="S")
    fs.add(Fact("AAA.vol_60d", "vol", 25.0, "%", as_of, "t"))
    fs.add(Fact("AAA.price", "price", 100.0, "USD", as_of, "t"))
    bear = AggregatedView(ticker="AAA", prob_outperform=0.45, prob_equal_weight=0.45, expected_excess_return_pct=-1.0,
                          dispersion=0.0, n_views=3, n_abstained=0)
    cfg = FirmConfig()
    cfg.portfolio.no_trade_band = 0.0
    w, path = 0.08, []
    while w and len(path) < 60:
        tp = construct_portfolio({"AAA": bear}, {}, {"AAA": fs}, {"AAA": w}, cfg, as_of, {})
        w = {p.ticker: p.weight for p in tp.positions}.get("AAA", 0.0)
        path.append(w)
    assert path == pytest.approx([0.04, 0.02, 0.01, 0.005, 0.0025, 0.0]) and MIN_RESIDUAL_GAP == 0.0025
