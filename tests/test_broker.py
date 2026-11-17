"""Tests for execution (execution/broker.py). No network: Alpaca is exercised with a fake session."""

import json
import math

import numpy as np
import pytest
import requests

from aifirm.config import CostConfig
from aifirm.execution.broker import AlpacaBroker, ApprovalRequired, Broker, PaperBroker, execution_cost
from aifirm.schemas import Order

COSTS = CostConfig(commission_bps=0.5, slippage_bps=5.0, impact_coefficient=0.1)


def _order(t, side, qty, price=100.0):
    return Order(ticker=t, side=side, quantity=qty, reference_price=price, notional=qty * price, reason="test")


# ----------------------------------------------------------------------------- cost model


def test_execution_cost_formula():
    commission, slippage = execution_cost(1_000_000.0, 100.0, adv=1e8, daily_vol=0.02, cfg=COSTS)
    assert commission == pytest.approx(50.0)                     # 0.5 bps
    assert slippage == pytest.approx(700.0)                      # 5 bps + 0.1*1e4*0.02*sqrt(0.01) = 7 bps
    assert execution_cost(-1_000_000.0, 100.0, 1e8, 0.02, COSTS) == (commission, slippage)
    assert execution_cost(1_000_000.0, 100.0, None, 0.02, COSTS)[1] == pytest.approx(500.0)
    assert execution_cost(1_000_000.0, 100.0, 1e8, None, COSTS)[1] == pytest.approx(500.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            execution_cost(1000.0, bad, None, None, COSTS)


# ----------------------------------------------------------------------------- paper broker


def test_round_trip_pays_costs_on_both_sides():
    b = PaperBroker(100_000.0, COSTS)
    assert isinstance(b, Broker)
    buy = b.submit([_order("X", "buy", 100)], {"X": 100.0}, "2026-01-02")[0]
    assert buy.price > 100.0 and buy.commission > 0 and buy.slippage_cost > 0
    assert b.cash() == pytest.approx(100_000.0 - 100 * buy.price - buy.commission)
    assert b.nav({"X": 100.0}) == pytest.approx(b.cash() + 100 * 100.0)
    sell = b.submit([_order("X", "sell", 100)], {"X": 100.0}, "2026-01-03")[0]
    assert sell.price < 100.0 and sell.commission > 0 and sell.slippage_cost > 0
    total_costs = buy.commission + buy.slippage_cost + sell.commission + sell.slippage_cost
    assert b.positions() == {}
    assert b.cash() == pytest.approx(100_000.0 - total_costs)
    assert total_costs == pytest.approx(2 * 10_000.0 * 5.5 / 1e4)
    assert (b.total_commission, b.total_slippage) == pytest.approx((1.0, 10.0))


def test_buy_is_scaled_to_available_cash():
    b = PaperBroker(1_000.0, COSTS)
    fill = b.submit([_order("X", "buy", 100)], {"X": 100.0}, "d")[0]
    assert b.cash() >= 0 and b.cash() == pytest.approx(0.0, abs=1e-6)
    assert fill.quantity == pytest.approx(1_000.0 / (100.0 * (1 + 5.5e-4)))
    assert "scaled" in b.last_rejections[0]
    b.submit([_order("Y", "buy", 1)], {"Y": 10.0}, "d")
    assert b.cash() >= 0


def test_sells_fund_buys_regardless_of_order_list_order():
    b = PaperBroker(0.0, COSTS)
    b._positions = {"X": 10.0}
    fills = b.submit([_order("Y", "buy", 10), _order("X", "sell", 10)], {"X": 100.0, "Y": 100.0}, "d")
    assert [f.side for f in fills] == ["sell", "buy"]
    # the sale nets 10 * 100 * (1 - 5.5 bps); that buys slightly fewer than 10 shares at 100 * (1 + 5.5 bps)
    assert b.positions()["Y"] == pytest.approx(10 * (1 - 5.5e-4) / (1 + 5.5e-4), rel=1e-9)
    assert 0 <= b.cash() < 1e-6


def test_no_shorting_unless_allowed():
    b = PaperBroker(0.0, COSTS)
    b._positions = {"X": 10.0}
    fills = b.submit([_order("X", "sell", 20)], {"X": 100.0}, "d")
    assert fills[0].quantity == pytest.approx(10.0) and b.positions() == {}
    assert "no shorting" in b.last_rejections[0]
    s = PaperBroker(0.0, COSTS, allow_short=True)
    s.submit([_order("X", "sell", 10)], {"X": 100.0}, "d")
    assert s.positions() == {"X": -10.0} and s.cash() > 0
    assert s.nav({"X": 100.0}) == pytest.approx(s.cash() - 1000.0)


def test_missing_fill_price_or_bad_quantity_does_not_fill():
    b = PaperBroker(10_000.0, COSTS)
    assert b.submit([_order("X", "buy", 1)], {}, "d") == []
    assert b.submit([_order("X", "buy", 1)], {"X": float("nan")}, "d") == []
    assert b.submit([_order("X", "buy", 0)], {"X": 10.0}, "d") == []
    assert b.cash() == 10_000.0 and len(b.last_rejections) == 1


def test_impact_uses_adv_and_daily_vol():
    plain = PaperBroker(1e7, COSTS).submit([_order("X", "buy", 10_000)], {"X": 100.0}, "d")[0]
    impact = PaperBroker(1e7, COSTS).submit([_order("X", "buy", 10_000)], {"X": 100.0}, "d",
                                            adv={"X": 1e8}, daily_vol={"X": 0.02})[0]
    assert impact.slippage_cost == pytest.approx(plain.slippage_cost * 7 / 5)


def test_nav_uses_stale_mark_when_price_missing():
    b = PaperBroker(10_000.0, COSTS)
    b.submit([_order("X", "buy", 10)], {"X": 100.0}, "d")
    assert b.nav({}) == pytest.approx(b.cash() + 1000.0)
    assert b.nav({"X": 120.0}) == pytest.approx(b.cash() + 1200.0)
    assert b.nav({"X": float("nan")}) == pytest.approx(b.cash() + 1200.0)
    fresh = PaperBroker(0.0, COSTS)
    fresh._positions = {"Z": 1.0}
    with pytest.raises(ValueError):
        fresh.nav({})


@pytest.mark.parametrize("seed", range(50))
def test_property_cash_never_negative_and_nav_identity(seed):
    rng = np.random.default_rng(seed)
    tickers = ["A", "B", "C", "D"]
    b = PaperBroker(float(rng.uniform(1e3, 1e6)), COSTS, allow_short=bool(rng.random() < 0.2))
    for day in range(25):
        prices = {t: float(rng.uniform(5, 300)) for t in tickers}
        orders = [_order(t, str(rng.choice(["buy", "sell"])), float(rng.uniform(0.1, 5000)), prices[t])
                  for t in rng.choice(tickers, size=int(rng.integers(0, 6)))]
        adv = {t: float(rng.uniform(1e5, 1e9)) for t in tickers}
        dv = {t: float(rng.uniform(0.005, 0.05)) for t in tickers}
        nav_before = b.nav(prices)
        fills = b.submit(orders, prices, f"day{day}", adv, dv)
        assert b.cash() >= 0
        nav_after = b.nav(prices)
        assert nav_after == pytest.approx(b.cash() + sum(q * prices[t] for t, q in b.positions().items()))
        # trading at the reference price changes NAV by exactly the costs paid
        costs = sum(f.commission + f.slippage_cost for f in fills)
        assert nav_before - nav_after == pytest.approx(costs, rel=1e-9, abs=1e-6)
        for f in fills:
            assert f.price > prices[f.ticker] if f.side == "buy" else f.price < prices[f.ticker]
        if not b.allow_short:
            assert all(q > 0 for q in b.positions().values())


def test_save_load_round_trip_and_autosave(tmp_path):
    path = tmp_path / "state" / "broker.json"
    b = PaperBroker(50_000.0, COSTS, state_path=str(path))
    assert b.load() is False
    b.submit([_order("X", "buy", 10)], {"X": 100.0}, "d")         # autosaves
    assert path.exists()
    restored = PaperBroker(0.0, COSTS, state_path=str(path))
    assert restored.load() is True
    assert restored.cash() == b.cash() and restored.positions() == b.positions()
    assert restored.nav({}) == pytest.approx(b.nav({}))           # marks survive
    assert restored.total_commission == b.total_commission
    path.write_text(json.dumps({"cash": -5.0, "positions": {}}))
    with pytest.raises(ValueError):
        restored.load()


# ----------------------------------------------------------------------------- Alpaca (fake session)


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json, "timeout": timeout})
        handler = self.routes[(method, url.split(".markets", 1)[1])]
        return handler(json) if callable(handler) else handler


@pytest.fixture
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)


def test_alpaca_refuses_without_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="refuses"):
        AlpacaBroker(session=_FakeSession({}))


def test_alpaca_live_endpoint_needs_explicit_opt_in(alpaca_env, monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(PermissionError):
        AlpacaBroker(session=_FakeSession({}))
    assert AlpacaBroker(session=_FakeSession({}), allow_live=True).base_url == "https://api.alpaca.markets"
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2/")
    assert AlpacaBroker(session=_FakeSession({})).base_url == "https://paper-api.alpaca.markets"


def test_alpaca_requires_approval(alpaca_env):
    session = _FakeSession({})
    b = AlpacaBroker(session=session)
    with pytest.raises(ApprovalRequired):
        b.submit([_order("X", "buy", 1)], {"X": 100.0}, "2026-01-02")
    assert session.calls == []                                     # nothing was sent


def test_alpaca_submit_formats_market_orders(alpaca_env):
    def place(body):
        if body["symbol"] == "AAA":
            return _Resp({"id": "1", "status": "filled", "filled_qty": body["qty"], "filled_avg_price": "99.5",
                          "symbol": "AAA", "side": "sell"})
        return _Resp({"id": "2", "status": "accepted", "filled_qty": "0", "filled_avg_price": None})

    session = _FakeSession({("POST", "/v2/orders"): place})
    b = AlpacaBroker(session=session)
    fills = b.submit([_order("BBB", "buy", 1.23456789), _order("AAA", "sell", 2.0)],
                     {"AAA": 100.0, "BBB": 50.0}, "2026-01-02", approve=True)
    bodies = [c["json"] for c in session.calls]
    assert [x["symbol"] for x in bodies] == ["AAA", "BBB"]          # sells first
    assert bodies[1] | {"client_order_id": None} == {"symbol": "BBB", "qty": "1.234567", "side": "buy",
                                                     "type": "market", "time_in_force": "day",
                                                     "client_order_id": None}
    assert bodies[0]["client_order_id"].startswith("aifirm-")
    assert session.calls[0]["headers"] == {"APCA-API-KEY-ID": "key-id", "APCA-API-SECRET-KEY": "secret"}
    assert session.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/orders"
    assert len(fills) == 1 and fills[0].ticker == "AAA" and fills[0].price == 99.5
    assert fills[0].slippage_cost == pytest.approx(1.0)              # sold 0.5 below the reference, 2 shares
    assert len(b.last_orders) == 2
    # the same plan resubmitted carries the same client_order_id, so Alpaca rejects the duplicate
    b.submit([_order("AAA", "sell", 2.0)], {"AAA": 100.0}, "2026-01-02", approve=True)
    assert session.calls[-1]["json"]["client_order_id"] == bodies[0]["client_order_id"]


def test_alpaca_http_error_stops_the_batch(alpaca_env):
    session = _FakeSession({("POST", "/v2/orders"): _Resp({"message": "forbidden"}, status=403)})
    b = AlpacaBroker(session=session, require_approval=False)
    with pytest.raises(requests.HTTPError):
        b.submit([_order("A", "sell", 1), _order("B", "buy", 1)], {}, "d")
    assert len(session.calls) == 1


def test_alpaca_account_and_positions(alpaca_env):
    session = _FakeSession({
        ("GET", "/v2/account"): _Resp({"cash": "1000.50", "equity": "3000"}),
        ("GET", "/v2/positions"): _Resp([
            {"symbol": "AAA", "qty": "10", "side": "long", "current_price": "100"},
            {"symbol": "BBB", "qty": "-2", "side": "short", "current_price": "50"},
        ]),
    })
    b = AlpacaBroker(session=session)
    assert b.cash() == 1000.5
    assert b.positions() == {"AAA": 10.0, "BBB": -2.0}
    assert b.nav({"AAA": 110.0}) == pytest.approx(1000.5 + 10 * 110.0 - 2 * 50.0)
    assert not math.isnan(b.nav({}))


# ----------------------------------------------------------------------------- end to end


def test_end_to_end_cycles_keep_accounting_and_limits(tmp_path):
    """Synthetic market -> fact sheets -> construct -> risk engine -> orders -> paper broker,
    deciding at the close and filling at the next open (P3), for a dozen rebalances."""
    import pandas as pd

    from aifirm.config import FirmConfig, PortfolioConfig
    from aifirm.data.factsheet import build_factsheet
    from aifirm.data.prices import from_frames
    from aifirm.portfolio import (RiskState, apply_risk_limits, construct_portfolio, orders_from_target,
                                  update_risk_state)
    from aifirm.schemas import AggregatedView

    rng = np.random.default_rng(7)
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    n = 520
    idx = pd.bdate_range("2024-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    close = pd.DataFrame({t: 50 * np.exp(np.cumsum(mkt + rng.normal(0, 0.015, n))) for t in names}
                         | {"SPY": 400 * np.exp(np.cumsum(mkt))}, index=idx)
    open_ = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.002, close.shape))
    volume = pd.DataFrame(5e6, index=idx, columns=close.columns)
    prices = from_frames(close, volume=volume, open_=open_)
    cfg = FirmConfig(portfolio=PortfolioConfig(core_weight=0.3), runs_dir=str(tmp_path))
    broker = PaperBroker(cfg.execution.initial_cash, cfg.costs)
    state = RiskState(peak_nav=broker.cash())
    for i in range(260, n - 1, 21):
        d, nxt = idx[i], idx[i + 1]
        as_of = str(d.date())
        p = prices.as_of(as_of)
        fss = {t: build_factsheet(t, as_of, p, sector=f"S{k % 3}") for k, t in enumerate(names)}
        last = {t: float(close.loc[d, t]) for t in close.columns}
        nav = broker.nav(last)
        prev_nav = broker.nav({t: float(close.loc[idx[i - 1], t]) for t in close.columns})
        state = update_risk_state(state, nav)
        current = {t: q * last[t] / nav for t, q in broker.positions().items()}
        aggs = {}
        for t in names:
            mom = fss[t].get(f"{t}.mom_12_1").value
            prob = 0.5 + 0.1 * math.tanh(mom / 10)
            aggs[t] = AggregatedView(ticker=t, prob_outperform=prob, prob_equal_weight=prob,
                                     expected_excess_return_pct=mom / 5, dispersion=0.03, n_views=4, n_abstained=0)
        target = construct_portfolio(aggs, {}, fss, current, cfg, as_of, {}, p.returns())
        target, _ = apply_risk_limits(target, state, nav, prev_nav, current, cfg, fss, str(tmp_path))
        adv = {t: float((close[t] * volume[t]).loc[:d].tail(20).mean()) for t in close.columns}
        daily_vol = {t: float(p.returns()[t].tail(20).std()) for t in close.columns}
        orders, msgs = orders_from_target(target, broker.positions(), last, nav, adv, cfg, "SPY")
        assert not any("blocked" in m for m in msgs)
        fill_px = {t: float(open_.loc[nxt, t]) for t in close.columns}
        broker.submit(orders, fill_px, str(nxt.date()), adv, daily_vol)
        assert broker.cash() >= 0
        assert all(q > 0 for q in broker.positions().values())
        post_nav = broker.nav(fill_px)
        assert post_nav == pytest.approx(broker.cash() + sum(q * fill_px[t] for t, q in broker.positions().items()))
        weights = {t: q * fill_px[t] / post_nav for t, q in broker.positions().items()}
        # limits hold up to close-to-open drift and costs
        assert all(w <= cfg.portfolio.max_position * 1.05 for t, w in weights.items() if t != "SPY")
        assert broker.cash() / post_nav >= cfg.portfolio.min_cash * 0.5
    assert broker.positions().get("SPY", 0) > 0                 # the core was built
    assert broker.total_commission > 0 and broker.total_slippage > 0
