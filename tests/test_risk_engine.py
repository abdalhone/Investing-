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
           prices=None, state=None):
    cfg = cfg or _cfg(tmp_path)
    state = state or RiskState(peak_nav=peak, frozen=frozen)
    prices = prices if prices is not None else {t: 50.0 for t in [p.ticker for p in target.positions]}
    out, msgs = apply_risk_limits(target, state, nav, prev_nav, current or {}, cfg, _fss(prices), str(tmp_path))
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
    out, msgs, state = _apply(tmp_path, target, nav=84.0, peak=100.0)
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
