"""Tests for the invariant risk core (portfolio/risk_engine.py). No network."""

import math

import numpy as np
import pytest

from aifirm.config import FirmConfig, PortfolioConfig, RiskConfig
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.portfolio.risk_engine import RiskState, apply_risk_limits, orders_from_target, update_risk_state
from aifirm.schemas import TargetPortfolio, TargetPosition

BENCH = "SPY"
AS_OF = "2026-01-30"


def _cfg(tmp_path, risk=None, **portfolio) -> FirmConfig:
    return FirmConfig(runs_dir=str(tmp_path), risk=RiskConfig(**(risk or {})), portfolio=PortfolioConfig(**portfolio))


def _target(weights, core=0.0):
    positions = [TargetPosition(ticker=t, weight=w, prob_outperform=0.6, expected_excess_return_pct=2.0,
                                rationale="test") for t, w in weights.items()]
    return TargetPortfolio(as_of=AS_OF, positions=positions, core_weight=core,
                           cash_weight=1 - core - sum(weights.values()))


def _fss(prices):
    out = {}
    for t, p in prices.items():
        fs = FactSheet(ticker=t, as_of=AS_OF)
        fs.add(Fact(f"{t}.price", "Last close", float(p), "USD", AS_OF, "test"))
        out[t] = fs
    return out


def _apply(tmp_path, target, *, nav=100.0, prev_nav=None, peak=100.0, frozen=False, current=None, cfg=None,
           prices=None, state=None, worst_day_return=None):
    cfg = cfg or _cfg(tmp_path)
    state = state or RiskState(peak_nav=peak, frozen=frozen)
    prices = prices if prices is not None else {t: 50.0 for t in [p.ticker for p in target.positions]}
    out, msgs = apply_risk_limits(target, state, nav, prev_nav, current or {}, cfg, _fss(prices), str(tmp_path),
                                  worst_day_return=worst_day_return)
    return out, msgs, state


def _w(tp):
    return {p.ticker: p.weight for p in tp.positions}


# ----------------------------------------------------------------------------- RiskState


def test_update_risk_state_tracks_peak_and_last_nav():
    s0 = RiskState(peak_nav=100.0)
    s1 = update_risk_state(s0, 110.0)
    assert (s1.peak_nav, s1.last_nav, s1.frozen) == (110.0, 110.0, False)
    assert s0.peak_nav == 100.0                     # input not modified
    s2 = update_risk_state(s1, 90.0)
    assert (s2.peak_nav, s2.last_nav) == (110.0, 90.0)


def test_update_risk_state_fails_closed_and_frozen_is_sticky():
    assert update_risk_state(RiskState(peak_nav=100.0), float("nan")).frozen
    assert update_risk_state(RiskState(peak_nav=100.0), -5.0).frozen
    bad_peak = update_risk_state(RiskState(peak_nav=float("nan")), 100.0)
    assert bad_peak.frozen and bad_peak.peak_nav == 100.0
    assert update_risk_state(RiskState(peak_nav=100.0, frozen=True), 200.0).frozen


# ----------------------------------------------------------------------------- rules


def test_kill_switch_holds_current_book_and_produces_no_orders(tmp_path):
    (tmp_path / "KILL").write_text("stop")
    current = {"AAA": 0.05, "BBB": 0.02, BENCH: 0.3}
    out, msgs, _ = _apply(tmp_path, _target({"AAA": 0.10, "CCC": 0.08}, core=0.5), current=current)
    assert _w(out) == {"AAA": 0.05, "BBB": 0.02}
    assert out.core_weight == 0.3
    assert any("KILL SWITCH" in m for m in msgs)
    nav = 100_000.0
    prices = {"AAA": 50.0, "BBB": 20.0, "CCC": 10.0, BENCH: 500.0}
    positions = {t: w * nav / prices[t] for t, w in current.items()}
    orders, _ = orders_from_target(out, positions, prices, nav, {t: 1e12 for t in prices}, _cfg(tmp_path), BENCH)
    assert orders == []


def test_drawdown_freeze_zeroes_active_keeps_core_and_is_sticky(tmp_path):
    target = _target({"AAA": 0.10, "BBB": 0.05}, core=0.4)
    out, msgs, state = _apply(tmp_path, target, nav=79.0, peak=100.0)      # freeze at 20% (research 6.4)
    assert state.frozen
    assert all(w == 0 for w in _w(out).values()) and out.core_weight == pytest.approx(0.4)
    assert any("DRAWDOWN FREEZE" in m for m in msgs)
    assert all("risk:drawdown_freeze" in p.constraints_hit for p in out.positions)
    # NAV recovers, but only a human may unfreeze
    out2, msgs2, _ = _apply(tmp_path, target, nav=100.0, state=state)
    assert all(w == 0 for w in _w(out2).values()) and out2.core_weight == pytest.approx(0.4)
    assert any("frozen" in m for m in msgs2)


def test_drawdown_derisk_halves_active(tmp_path):
    out, msgs, state = _apply(tmp_path, _target({"AAA": 0.10, "BBB": 0.06}, core=0.2), nav=89.0)
    assert _w(out) == pytest.approx({"AAA": 0.05, "BBB": 0.03})
    assert out.core_weight == pytest.approx(0.2) and not state.frozen
    assert any("de-risk" in m for m in msgs)


def test_drawdown_uses_nav_if_peak_is_stale(tmp_path):
    out, msgs, _ = _apply(tmp_path, _target({"AAA": 0.10}), nav=120.0, peak=100.0)
    assert _w(out) == {"AAA": 0.10} and msgs == []


def test_daily_loss_halt_prevents_any_increase(tmp_path):
    target = _target({"UP": 0.08, "DOWN": 0.03, "NEW": 0.04}, core=0.3)
    current = {"UP": 0.05, "DOWN": 0.06, BENCH: 0.2}
    out, msgs, _ = _apply(tmp_path, target, nav=94.0, prev_nav=100.0, peak=100.0, current=current)
    assert _w(out) == pytest.approx({"UP": 0.05, "DOWN": 0.03, "NEW": 0.0})
    assert out.core_weight == pytest.approx(0.2)
    assert any("DAILY-LOSS HALT" in m for m in msgs)
    # a smaller loss does not halt
    out2, _, _ = _apply(tmp_path, target, nav=96.0, prev_nav=100.0, peak=100.0, current=current)
    assert _w(out2) == pytest.approx({"UP": 0.08, "DOWN": 0.03, "NEW": 0.04})
    # an invalid previous NAV is treated as a loss day (fail closed)
    out3, _, _ = _apply(tmp_path, target, prev_nav=float("nan"), current=current)
    assert _w(out3)["NEW"] == 0.0


def test_daily_loss_halt_sees_any_loss_day_since_the_last_cycle(tmp_path):
    """Regression: a -8% day in the middle of a cycle must halt buying at the next decision,
    even though the decision day itself was flat."""
    target = _target({"UP": 0.08, "NEW": 0.04})
    current = {"UP": 0.05}
    out, msgs, _ = _apply(tmp_path, target, nav=100.0, prev_nav=100.0, current=current,
                          worst_day_return=-0.08)
    assert _w(out) == pytest.approx({"UP": 0.05, "NEW": 0.0})
    assert any("DAILY-LOSS HALT" in m and "since the last cycle" in m for m in msgs)
    out2, msgs2, _ = _apply(tmp_path, target, nav=100.0, prev_nav=100.0, current=current, worst_day_return=-0.01)
    assert _w(out2) == pytest.approx({"UP": 0.08, "NEW": 0.04}) and msgs2 == []
    out3, _, _ = _apply(tmp_path, target, current=current, worst_day_return=float("nan"))   # fail closed
    assert _w(out3)["NEW"] == 0.0


def test_human_unfreeze_resets_the_high_water_mark_instead_of_refreezing(tmp_path):
    """Regression: the documented unfreeze (set "frozen": false in risk_state.json) must work.
    A frozen book sits in cash and cannot regain the old peak, so the next cycle has to start
    a new high-water mark rather than re-freeze at once."""
    target = _target({"AAA": 0.10})
    _, _, state = _apply(tmp_path, target, nav=79.0, peak=100.0)
    assert state.frozen and state.frozen_at_nav == pytest.approx(79.0)
    state.frozen = False                                              # the human decision
    out, msgs, state2 = _apply(tmp_path, target, nav=79.0, state=state)
    assert not state2.frozen and state2.peak_nav == pytest.approx(79.0) and state2.frozen_at_nav is None
    assert _w(out) == {"AAA": 0.10}
    assert any("unfrozen by a human" in m for m in msgs)
    # a book that was never frozen keeps measuring drawdowns from its real peak
    _, _, fresh = _apply(tmp_path, target, nav=79.0, state=RiskState(peak_nav=100.0))
    assert fresh.frozen


def test_forced_exits_are_not_skipped_by_the_no_trade_band(tmp_path):
    """Regression: a freeze, the restricted list or a name absent from the target sets the
    weight to 0; orders_from_target must sell those positions even when they are smaller
    than the no-trade band (1% of NAV by default), or no rule could ever exit them."""
    nav = 1_000_000.0
    names = [f"T{i:02d}" for i in range(12)]
    current = {"BAD": 0.008} | {t: 0.009 for t in names}
    prices = {t: 50.0 for t in current}
    cfg = _cfg(tmp_path, risk={"restricted": ["BAD"]}, no_trade_band=0.01)   # positions below a 1% band
    out, _, state = _apply(tmp_path, _target(current), nav=nav, peak=1.3 * nav, current=current, cfg=cfg,
                           prices=prices)
    assert state.frozen and all(w == 0.0 for w in _w(out).values())
    positions = {t: w * nav / prices[t] for t, w in current.items()}
    adv = {t: 1e12 for t in [*prices, BENCH]}
    orders, _ = orders_from_target(out, positions, {**prices, BENCH: 500.0}, nav, adv, cfg, BENCH)
    assert {o.ticker for o in orders} == set(current) and all(o.side == "sell" for o in orders)
    assert all(o.quantity == pytest.approx(positions[o.ticker]) for o in orders)
    # a held name missing from the target is a full exit too; a discretionary sub-band change is not traded
    keep = _target({"T00": 0.0095})
    orders2, _ = orders_from_target(keep, {"T00": positions["T00"], "GONE": 100.0}, {**prices, "GONE": 50.0}, nav,
                                    {**adv, "GONE": 1e12}, cfg, BENCH)
    assert [(o.ticker, o.side) for o in orders2] == [("GONE", "sell")]


def test_restricted_and_min_price(tmp_path):
    cfg = _cfg(tmp_path, risk={"restricted": ["bad", BENCH]})
    out, _, _ = _apply(tmp_path, _target({"BAD": 0.05, "PENNY": 0.05, "OK": 0.05}, core=0.2), cfg=cfg,
                          prices={"BAD": 50.0, "PENNY": 1.5, "OK": 50.0})
    assert _w(out) == {"BAD": 0.0, "PENNY": 0.0, "OK": 0.05}
    assert out.core_weight == 0.0
    assert {"risk:restricted", "risk:min_price"} <= {h for p in out.positions for h in p.constraints_hit}


def test_unverifiable_price_blocks_growth_but_not_holding(tmp_path):
    out, msgs, _ = _apply(tmp_path, _target({"AAA": 0.08, "BBB": 0.02}), prices={}, current={"AAA": 0.05, "BBB": 0.04})
    assert _w(out) == pytest.approx({"AAA": 0.05, "BBB": 0.02})
    assert any("price unavailable" in m for m in msgs)


def test_backstops_catch_broken_targets(tmp_path):
    target = _target({"BIG": 0.5, "SHORT": -0.05, "NAN": float("nan")})
    out, msgs, _ = _apply(tmp_path, target)
    assert _w(out) == {"BIG": 0.10, "SHORT": 0.0, "NAN": 0.0}
    assert len(msgs) == 3
    many = _target({f"T{i}": 0.1 for i in range(12)})
    out2, msgs2, _ = _apply(tmp_path, many)
    assert sum(_w(out2).values()) == pytest.approx(0.98)
    assert out2.cash_weight == pytest.approx(0.02)
    assert any("backstop" in m for m in msgs2)


def test_invalid_nav_holds_current_book(tmp_path):
    out, msgs, _ = _apply(tmp_path, _target({"AAA": 0.1}), nav=float("nan"), current={"BBB": 0.03})
    assert _w(out) == {"BBB": 0.03} and "invalid NAV" in msgs[0]


@pytest.mark.parametrize("seed", range(80))
def test_property_engine_never_increases_exposure(tmp_path, seed):
    rng = np.random.default_rng(seed)
    long_only = bool(rng.random() < 0.7)
    tickers = [f"T{i}" for i in range(int(rng.integers(1, 12)))]
    lo = 0.0 if long_only else -0.2
    target = _target({t: float(rng.uniform(lo, 0.25)) for t in tickers}, core=float(rng.uniform(0, 0.6)))
    current = {t: float(rng.uniform(lo, 0.2)) for t in tickers if rng.random() < 0.5}
    current[BENCH] = float(rng.uniform(0, 0.5))
    restricted = [t for t in tickers if rng.random() < 0.15]
    cfg = _cfg(tmp_path, risk={"restricted": restricted}, long_only=long_only,
               max_gross=1.0 if long_only else 1.6)
    prices = {t: float(rng.choice([1.0, 30.0, 100.0, np.nan])) for t in tickers}
    prices = {t: p for t, p in prices.items() if not math.isnan(p)}
    nav = float(rng.uniform(70, 130))
    prev_nav = float(rng.uniform(70, 130)) if rng.random() < 0.7 else None
    state = RiskState(peak_nav=float(rng.uniform(90, 140)), frozen=bool(rng.random() < 0.1))
    out, _, _ = _apply(tmp_path, target, nav=nav, prev_nav=prev_nav, current=current, cfg=cfg, prices=prices,
                       state=state)
    tw = _w(target)
    for t, w in _w(out).items():
        assert abs(w) <= abs(tw[t]) + 1e-12
        if long_only:
            assert w >= 0
    assert out.core_weight <= target.core_weight + 1e-12
    assert sum(abs(w) for w in _w(out).values()) + out.core_weight <= cfg.portfolio.max_gross + 1e-9
    assert out.cash_weight >= cfg.portfolio.min_cash - 1e-9
    if state.frozen:
        assert all(w == 0 for w in _w(out).values())


# ----------------------------------------------------------------------------- orders


def test_orders_size_band_and_core(tmp_path):
    cfg = _cfg(tmp_path)
    nav = 100_000.0
    target = _target({"AAA": 0.10, "BBB": 0.054, "CCC": 0.0}, core=0.3)
    positions = {"BBB": 250.0, "CCC": 40.0}                      # BBB is $5,000 = 5%; CCC $2,000
    prices = {"AAA": 50.0, "BBB": 20.0, "CCC": 50.0, BENCH: 500.0}
    orders, msgs = orders_from_target(target, positions, prices, nav, {t: 1e12 for t in prices}, cfg, BENCH)
    by = {o.ticker: o for o in orders}
    assert set(by) == {"AAA", "CCC", BENCH}                       # BBB's $400 change is inside the $1,000 band
    assert by["AAA"].side == "buy" and by["AAA"].quantity == pytest.approx(200.0)
    assert by["CCC"].side == "sell" and by["CCC"].quantity == pytest.approx(40.0)
    assert by[BENCH].side == "buy" and by[BENCH].notional == pytest.approx(30_000.0)
    assert [o.side for o in orders] == ["sell", "buy", "buy"]    # sells first
    assert msgs == []


def test_adv_cap_limits_order_notional(tmp_path):
    cfg = _cfg(tmp_path)
    orders, msgs = orders_from_target(_target({"AAA": 0.10}), {}, {"AAA": 50.0}, 1_000_000.0, {"AAA": 1_000_000.0},
                                      cfg, BENCH)
    assert len(orders) == 1 and orders[0].notional == pytest.approx(20_000.0)
    assert orders[0].quantity == pytest.approx(400.0)
    assert any("capped" in m for m in msgs)


def test_missing_price_adv_or_nav_blocks_orders(tmp_path):
    cfg = _cfg(tmp_path)
    target = _target({"NOPX": 0.1, "NOADV": 0.1})
    orders, msgs = orders_from_target(target, {}, {"NOADV": 10.0}, 100_000.0, {}, cfg, BENCH)
    assert orders == []
    assert any("NOPX" in m and "price" in m for m in msgs) and any("NOADV" in m and "ADV" in m for m in msgs)
    orders2, msgs2 = orders_from_target(target, {}, {"NOPX": 1.0}, 0.0, {"NOPX": 1e9}, cfg, BENCH)
    assert orders2 == [] and "invalid NAV" in msgs2[0]


def test_long_only_sells_never_exceed_holdings(tmp_path):
    cfg = _cfg(tmp_path)
    orders, msgs = orders_from_target(_target({"AAA": -0.2}), {"AAA": 10.0}, {"AAA": 100.0}, 100_000.0,
                                      {"AAA": 1e12}, cfg, BENCH)
    assert any("long-only" in m for m in msgs)
    assert len(orders) == 1 and orders[0].side == "sell" and orders[0].quantity == pytest.approx(10.0)


@pytest.mark.parametrize("seed", range(60))
def test_property_orders_move_toward_target_within_adv(tmp_path, seed):
    rng = np.random.default_rng(seed)
    cfg = _cfg(tmp_path, risk={"max_adv_participation": float(rng.uniform(0.001, 0.05))},
               no_trade_band=float(rng.uniform(0, 0.02)))
    tickers = [f"T{i}" for i in range(int(rng.integers(1, 10)))] + [BENCH]
    prices = {t: float(rng.uniform(5, 500)) for t in tickers}
    nav = float(rng.uniform(1e4, 1e7))
    positions = {t: float(rng.uniform(0, 0.15)) * nav / prices[t] for t in tickers if rng.random() < 0.5}
    adv = {t: float(rng.uniform(1e4, 1e8)) for t in tickers}
    target = _target({t: float(rng.uniform(0, 0.12)) for t in tickers if t != BENCH and rng.random() < 0.7},
                     core=float(rng.uniform(0, 0.4)))
    orders, _ = orders_from_target(target, positions, prices, nav, adv, cfg, BENCH)
    seen_buy = False
    tw = dict(target.weights())
    tw[BENCH] = tw.get(BENCH, 0.0) + target.core_weight
    for o in orders:
        assert o.notional <= cfg.risk.max_adv_participation * adv[o.ticker] * (1 + 1e-12)
        assert o.quantity > 0 and o.notional == pytest.approx(o.quantity * prices[o.ticker])
        seen_buy = seen_buy or o.side == "buy"
        assert not (seen_buy and o.side == "sell")                 # all sells before any buy
        before = positions.get(o.ticker, 0.0) * prices[o.ticker]
        after = before + (o.notional if o.side == "buy" else -o.notional)
        goal = tw.get(o.ticker, 0.0) * nav
        assert min(before, goal) - 1e-6 <= after <= max(before, goal) + 1e-6   # never overshoots
        assert after >= -1e-6                                     # never shorts


def test_float_rounding_above_the_position_cap_is_clipped_silently(tmp_path):
    """A construction weight of 0.10000000000100001 (cap 0.10, float rounding) used to raise the
    message "exceeds the position cap" on every clean run; a real breach is still reported."""
    out, msgs, _ = _apply(tmp_path, _target({"AAA": 0.10000000000100001, "BBB": 0.15}))
    assert _w(out) == {"AAA": 0.10, "BBB": 0.10}
    assert not any(m.startswith("AAA") for m in msgs)
    assert any(m.startswith("BBB") and "exceeds the position cap" in m for m in msgs)


# ----------------------------------------------------------------------------- de-risk is idempotent


def _construct_sheets(names):
    out = {}
    for t in names:
        fs = FactSheet(ticker=t, as_of=AS_OF, sector=f"S{t}")
        fs.add(Fact(f"{t}.vol_60d", "vol", 25.0, "%", AS_OF, "t"))
        fs.add(Fact(f"{t}.price", "price", 100.0, "USD", AS_OF, "t"))
        out[t] = fs
    return out


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_derisk_does_not_compound_with_partial_rebalancing(tmp_path, fraction):
    """Regression: the de-risk halved whatever construction proposed, and with partial
    rebalancing that target is a step from a book already halved last cycle, so the episode
    sold every cycle (0.24 -> 0.18 -> 0.165 -> ... -> 0.16, a third of the ideal). The cap
    is now half of construction's ideal weight: the book sits at 0.24 every cycle."""
    from aifirm.portfolio.construct import construct_portfolio
    from aifirm.schemas import AggregatedView
    names = [f"N{i}" for i in range(6)]
    aggs = {t: AggregatedView(ticker=t, prob_outperform=0.6, prob_equal_weight=0.6, expected_excess_return_pct=3.0,
                              dispersion=0.0, n_views=3, n_abstained=0) for t in names}
    sheets = _construct_sheets(names)
    cfg = _cfg(tmp_path, rebalance_fraction=fraction)
    cur = {t: 0.08 for t in names}
    state = RiskState(peak_nav=100.0, trough_nav=88.0)
    path = []
    for _ in range(6):
        tgt = construct_portfolio(aggs, {}, sheets, cur, cfg, AS_OF, {})
        assert all(p.ideal_weight == pytest.approx(0.08) for p in tgt.positions)
        final, _ = apply_risk_limits(tgt, state, 88.0, None, cur, cfg, sheets, str(tmp_path))
        assert all(abs(p.weight) <= abs(q.weight) + 1e-12 for p, q in zip(final.positions, tgt.positions))
        cur = {p.ticker: p.weight for p in final.positions if p.weight}
        path.append(round(sum(cur.values()), 6))
    assert path == [pytest.approx(0.24)] * 6


def test_derisk_halves_a_deliberate_hold_once_and_never_again(tmp_path):
    """Regression: during a de-risk episode an LLM-outage cycle proposes 'hold the current book'
    and the engine halved it again every cycle (0.12 -> 0.06 -> 0.03), so an LLM outage forced
    sales. A hold is halved only on the cycle the de-risk triggers."""
    from aifirm.pipeline import _hold_target
    cfg = _cfg(tmp_path)
    fss = _fss({t: 100.0 for t in "ABC"})
    state = RiskState(peak_nav=100.0)
    for nav in (95.0, 89.0):
        state = update_risk_state(state, nav)
    cur = {t: 0.08 for t in "ABC"}
    tgt = _hold_target(cur, "SPY", AS_OF, "LLM OUTAGE")          # the cycle the de-risk triggers: halved once
    out, msgs = apply_risk_limits(tgt, state, 89.0, None, cur, cfg, fss, str(tmp_path))
    assert state.derisked and _w(out) == pytest.approx({t: 0.04 for t in "ABC"})
    cur = _w(out)
    state = update_risk_state(state, 91.0)                            # still de-risked (needs 94.5)
    for _ in range(3):
        tgt = _hold_target(cur, "SPY", AS_OF, "LLM OUTAGE")
        out, msgs = apply_risk_limits(tgt, state, 91.0, None, cur, cfg, fss, str(tmp_path))
        assert _w(out) == pytest.approx(cur) and any("kept at the current weight" in m for m in msgs)
        assert any("still in force" in m for m in msgs)
    for label in ("unreviewed_hold", "no_view_hold", "risk_review_halt"):
        held = TargetPortfolio(as_of=AS_OF, core_weight=0.0, cash_weight=0.88, positions=[TargetPosition(
            ticker="A", weight=0.04, prob_outperform=0.5, expected_excess_return_pct=0.0, rationale="",
            constraints_hit=[label], ideal_weight=0.0)])
        out, _ = apply_risk_limits(held, state, 91.0, None, {"A": 0.04}, cfg, fss, str(tmp_path))
        assert _w(out) == {"A": pytest.approx(0.04)}, label


def test_derisk_without_an_ideal_weight_still_halves(tmp_path):
    out, _, _ = _apply(tmp_path, _target({"AAA": 0.10}), nav=89.0)
    assert _w(out) == pytest.approx({"AAA": 0.05})
    ideal = _target({"AAA": 0.06})
    ideal.positions[0].ideal_weight = 0.08                            # a partial step toward 0.08: cap 0.04
    out, _, _ = _apply(tmp_path, ideal, nav=89.0)
    assert _w(out) == pytest.approx({"AAA": 0.04})


# ----------------------------------------------------------------------------- the reviewer's cut is part of the plan


def _book_cycle(tmp_path, fraction, mult, nav, cycles=6, names=("AAA", "BBB", "CCC")):
    """construct -> LLM risk review (a PORTFOLIO multiplier every cycle) -> risk engine, from a
    full book at the plan (0.08 per name). Returns the gross path and the last messages."""
    from aifirm.agents.risk_review import apply_risk_review
    from aifirm.portfolio.construct import construct_portfolio
    from aifirm.schemas import AggregatedView, RiskConcern, RiskReview
    aggs = {t: AggregatedView(ticker=t, prob_outperform=0.6, prob_equal_weight=0.6, expected_excess_return_pct=3.0,
                              dispersion=0.0, n_views=3, n_abstained=0) for t in names}
    sheets = _construct_sheets(names)
    cfg = _cfg(tmp_path, rebalance_fraction=fraction)
    cur = {t: 0.08 for t in names}
    state = RiskState(peak_nav=100.0, trough_nav=min(100.0, nav))
    path, msgs = [], []
    for _ in range(cycles):
        tgt = construct_portfolio(aggs, {}, sheets, cur, cfg, AS_OF, {})
        review = RiskReview(summary="cautious", hidden_concentrations=[], halt_trading=False,
                            concerns=[RiskConcern(ticker="PORTFOLIO", concern="drawdown regime", multiplier=mult)])
        tgt = apply_risk_review(tgt, review, current_weights={**cur, "SPY": 0.5}, benchmark="SPY")
        final, msgs = apply_risk_limits(tgt, state, nav, None, cur, cfg, sheets, str(tmp_path))
        cur = {p.ticker: p.weight for p in final.positions if p.weight}
        path.append(round(sum(cur.values()), 6))
    return path, msgs, final


@pytest.mark.parametrize("fraction", [1.0, 0.5])
@pytest.mark.parametrize("mult", [0.75, 0.5, 0.3])
def test_a_standing_reviewer_cut_neither_ratchets_nor_hides_from_the_derisk(tmp_path, fraction, mult):
    """Regressions: (1) a reviewer multiplier applied to the partial step every cycle ratcheted
    the book down (x0.5 at rebalance_fraction 0.5 settled at 1/3 of the plan, 0.08 gross, not
    1/2); (2) the de-risk capped at half of the *un-reviewed* plan, so after a reviewer x0.5
    it cut nothing while the ledger said 'halved'. The multiplier now scales the plan."""
    plan = 0.24
    path, _, _ = _book_cycle(tmp_path, fraction, mult, nav=100.0)
    assert path == [pytest.approx(mult * plan)] * 6
    path, msgs, final = _book_cycle(tmp_path, fraction, mult, nav=88.0)
    assert path == [pytest.approx(0.5 * mult * plan)] * 6
    assert all(p.ideal_weight == pytest.approx(mult * 0.08) for p in final.positions)
    assert any(m.startswith("drawdown de-risk") for m in msgs)


def test_reviewer_x05_then_derisk_at_full_rebalancing_gives_a_quarter(tmp_path):
    """The reviewer's own example: plan 0.08 per name, reviewer x0.5 -> 0.04, 12% drawdown ->
    0.02 per name, with a de-risk label and a message naming the legs cut."""
    from aifirm.agents.risk_review import apply_risk_review
    from aifirm.portfolio.construct import construct_portfolio
    from aifirm.schemas import AggregatedView, RiskConcern, RiskReview
    names = ("AAA", "BBB", "CCC")
    aggs = {t: AggregatedView(ticker=t, prob_outperform=0.6, prob_equal_weight=0.6, expected_excess_return_pct=3.0,
                              dispersion=0.0, n_views=3, n_abstained=0) for t in names}
    sheets = _construct_sheets(names)
    cfg = _cfg(tmp_path, rebalance_fraction=1.0)
    cur = {t: 0.08 for t in names}
    tgt = construct_portfolio(aggs, {}, sheets, cur, cfg, AS_OF, {})
    review = RiskReview(summary="", hidden_concentrations=[], halt_trading=False,
                        concerns=[RiskConcern(ticker="PORTFOLIO", concern="regime", multiplier=0.5)])
    reviewed = apply_risk_review(tgt, review, current_weights={**cur, "SPY": 0.5}, benchmark="SPY")
    assert _w(reviewed) == pytest.approx({t: 0.04 for t in names})
    out, msgs = apply_risk_limits(reviewed, RiskState(peak_nav=100.0, trough_nav=88.0), 88.0, None, cur, cfg,
                                  sheets, str(tmp_path))
    assert _w(out) == pytest.approx({t: 0.02 for t in names})
    assert all("risk:drawdown_derisk" in p.constraints_hit for p in out.positions)
    assert any("cut: AAA, BBB, CCC" in m for m in msgs)


def test_derisk_message_does_not_claim_a_cut_that_did_not_happen(tmp_path):
    ideal = _target({"AAA": 0.02})
    ideal.positions[0].ideal_weight = 0.08                     # a step already under half the plan
    out, msgs, _ = _apply(tmp_path, ideal, nav=89.0)
    assert _w(out) == pytest.approx({"AAA": 0.02})
    assert any("nothing cut" in m for m in msgs) and not any("cut: AAA" in m for m in msgs)


def test_a_hold_finishes_a_derisk_sale_that_did_not_execute(tmp_path):
    """Regression: the hold branch fired only on the trigger cycle. When the trigger's halving
    order was blocked (no ADV, no price, a rejection), every later hold cycle of the episode
    kept the un-halved weight. The level the de-risk set is recorded and later holds are cut
    to it, never below it; drift within the no-trade band does not trade."""
    from aifirm.pipeline import _hold_target
    cfg = _cfg(tmp_path)
    fss = _fss({t: 100.0 for t in "ABC"})
    state = RiskState(peak_nav=100.0)
    state = update_risk_state(state, 88.0)
    cur = {t: 0.08 for t in "ABC"}
    out, _ = apply_risk_limits(_hold_target(cur, "SPY", AS_OF, "LLM OUTAGE"), state, 88.0, None, cur, cfg, fss,
                               str(tmp_path))
    assert _w(out) == pytest.approx({t: 0.04 for t in "ABC"}) and state.derisk_caps == pytest.approx(
        {t: 0.04 for t in "ABC"})
    cur = {"A": 0.08, "B": 0.04, "C": 0.043}                  # A's sale was blocked; C drifted up 0.3% (< band)
    for _ in range(3):
        out, msgs = apply_risk_limits(_hold_target(cur, "SPY", AS_OF, "LLM OUTAGE"), state, 88.0, None, cur, cfg,
                                      fss, str(tmp_path))
        assert _w(out) == pytest.approx({"A": 0.04, "B": cur["B"], "C": 0.043})
        assert any("A cut to the level the de-risk set earlier" in m for m in msgs)
        cur = {**cur, "B": 0.03}                              # a price fall: a hold is never sold below its level
    out, _ = apply_risk_limits(_hold_target({"B": 0.035}, "SPY", AS_OF, "x"), state, 88.0, None, {"B": 0.035}, cfg,
                               fss, str(tmp_path))
    assert _w(out) == pytest.approx({"B": 0.035})             # recovered toward 0.04: still under the level, kept
    state = update_risk_state(state, 96.0)                    # half of the drawdown recovered: re-risk
    apply_risk_limits(_hold_target(cur, "SPY", AS_OF, "x"), state, 96.0, None, cur, cfg, fss, str(tmp_path))
    assert not state.derisked and state.derisk_caps == {}


def test_risk_state_with_derisk_caps_round_trips_and_old_files_load(tmp_path):
    import json
    from dataclasses import asdict
    from aifirm.books import load_risk_state
    p = tmp_path / "risk_state.json"
    p.write_text(json.dumps({"peak_nav": 100.0, "frozen": False, "derisked": True}))       # saved before the field
    assert load_risk_state(p, 1.0).derisk_caps == {}
    s = RiskState(peak_nav=100.0, derisked=True, derisk_caps={"A": 0.04})
    p.write_text(json.dumps(asdict(s)))
    assert load_risk_state(p, 1.0) == s


def test_a_hold_whose_price_rose_after_the_derisk_sale_is_not_sold_again(tmp_path):
    """Regression (round 4): the 'finish the sale' branch compared weights, so a hold whose
    halving sale filled but whose price then rose 20% (0.04 -> 0.0476, above the band) was
    sold again during an LLM outage, with a ledger line saying the sale had not executed.
    The de-risk level is now a share count: same shares at a higher price are kept; more
    shares than recorded (a blocked sale) are cut back to the recorded count."""
    from aifirm.pipeline import _hold_target
    cfg = _cfg(tmp_path)
    state = update_risk_state(RiskState(peak_nav=100.0), 88.0)
    cur = {t: 0.08 for t in "ABC"}
    out, _ = apply_risk_limits(_hold_target(cur, "SPY", AS_OF, "LLM OUTAGE"), state, 88.0, None, cur, cfg,
                               _fss({t: 100.0 for t in "ABC"}), str(tmp_path))
    assert _w(out) == pytest.approx({t: 0.04 for t in "ABC"})
    assert state.derisk_shares == pytest.approx({t: 0.04 * 88.0 / 100.0 for t in "ABC"})
    # A's sale filled and A then rose 20%; B's sale was blocked and B also rose 20%; C is flat.
    prices = {"A": 120.0, "B": 120.0, "C": 100.0}
    cur = {"A": 0.048, "B": 0.096, "C": 0.04}
    out, msgs = apply_risk_limits(_hold_target(cur, "SPY", AS_OF, "LLM OUTAGE"), state, 88.0, None, cur, cfg,
                                  _fss(prices), str(tmp_path))
    assert _w(out) == pytest.approx({"A": 0.048, "B": 0.048, "C": 0.04})   # B back to its recorded shares
    assert any("B cut to the level the de-risk set earlier" in m for m in msgs)
    assert not any("A cut" in m or "A, B cut" in m for m in msgs)
    # Without a price the hold is kept: an outage never forces a sale on a guess.
    out, _ = apply_risk_limits(_hold_target({"B": 0.096}, "SPY", AS_OF, "x"), state, 88.0, None, {"B": 0.096},
                               cfg, {}, str(tmp_path))
    assert _w(out) == pytest.approx({"B": 0.096})
