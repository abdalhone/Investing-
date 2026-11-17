"""Tests for deterministic portfolio construction (portfolio/construct.py). No network."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from aifirm.config import FirmConfig, PortfolioConfig
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.portfolio.construct import construct_portfolio
from aifirm.schemas import AggregatedView

AS_OF = "2026-01-30"
BENCH = "SPY"


def _cfg(**portfolio) -> FirmConfig:
    base = dict(max_turnover=10.0)   # most tests isolate one rule; the turnover cap is tested separately
    base.update(portfolio)
    return FirmConfig(portfolio=PortfolioConfig(**base))


def _fs(t, vol=30.0, price=100.0, sector=""):
    fs = FactSheet(ticker=t, as_of=AS_OF, sector=sector)
    if vol is not None:
        fs.add(Fact(f"{t}.vol_60d", "Realized volatility, 60 days", float(vol), "%", AS_OF, "test"))
    fs.add(Fact(f"{t}.price", "Last close", float(price), "USD", AS_OF, "test"))
    return fs


def _agg(t, p, excess=0.0, dispersion=0.0):
    return AggregatedView(ticker=t, prob_outperform=p, prob_equal_weight=p, expected_excess_return_pct=excess,
                          dispersion=dispersion, n_views=4, n_abstained=0)


def _kelly(p, excess, vol=0.30, h=21, kelly=0.25, mult=1.0, disp_factor=1.0):
    sh = vol * math.sqrt(h / 252)
    mu = 0.5 * (sh * norm.ppf(p) + excess / 100) * mult * disp_factor
    return kelly * mu / sh ** 2


def _run(aggs, cfg=None, current=None, mult=None, factsheets=None, sectors=None, returns=None):
    cfg = cfg or _cfg()
    factsheets = factsheets if factsheets is not None else {t: _fs(t) for t in aggs}
    return construct_portfolio(aggs, mult or {}, factsheets, current or {}, cfg, AS_OF, sectors or {}, returns)


def _pos(tp, t):
    return next(p for p in tp.positions if p.ticker == t)


# ----------------------------------------------------------------------------- sizing


def test_fractional_kelly_weight_matches_formula():
    tp = _run({"AAA": _agg("AAA", 0.52, 0.0)})
    expected = _kelly(0.52, 0.0)
    assert 0 < expected < 0.10
    assert _pos(tp, "AAA").weight == pytest.approx(expected, rel=1e-12)
    assert tp.cash_weight == pytest.approx(1 - expected)
    assert tp.core_weight == 0.0


def test_position_cap_binds_and_is_recorded():
    tp = _run({"AAA": _agg("AAA", 0.62, 3.0)})
    assert _pos(tp, "AAA").weight == pytest.approx(0.10)
    assert "max_position" in _pos(tp, "AAA").constraints_hit


def test_hold_threshold_and_edge_sign_gate_positions():
    tp = _run({"LOW": _agg("LOW", 0.51, 5.0), "NEG": _agg("NEG", 0.55, -10.0)})
    assert _pos(tp, "LOW").weight == 0 and "min_prob_to_hold" in _pos(tp, "LOW").constraints_hit
    assert _pos(tp, "NEG").weight == 0 and "no_edge" in _pos(tp, "NEG").constraints_hit


def test_edge_multiplier_scales_and_defaults_to_one_when_missing():
    aggs = {"AAA": _agg("AAA", 0.52), "BBB": _agg("BBB", 0.52), "CCC": _agg("CCC", 0.52)}
    tp = _run(aggs, mult={"AAA": 0.5, "BBB": 0.0})
    assert _pos(tp, "AAA").weight == pytest.approx(_kelly(0.52, 0, mult=0.5))
    assert _pos(tp, "BBB").weight == 0
    assert _pos(tp, "CCC").weight == pytest.approx(_kelly(0.52, 0))
    # the chair can only shrink: a multiplier above 1 is clamped, a NaN means no trade
    tp2 = _run({"AAA": _agg("AAA", 0.52)}, mult={"AAA": 3.0})
    assert _pos(tp2, "AAA").weight == pytest.approx(_kelly(0.52, 0))
    tp3 = _run({"AAA": _agg("AAA", 0.52)}, mult={"AAA": float("nan")})
    assert _pos(tp3, "AAA").weight == 0


def test_dispersion_shrinks_size():
    half = _run({"AAA": _agg("AAA", 0.52, dispersion=0.25)})
    assert _pos(half, "AAA").weight == pytest.approx(_kelly(0.52, 0, disp_factor=0.5))
    gone = _run({"AAA": _agg("AAA", 0.60, 2.0, dispersion=0.6)})
    assert _pos(gone, "AAA").weight == 0


def test_missing_vol_uses_30pct_fallback_and_tiny_vol_is_floored():
    tp = _run({"AAA": _agg("AAA", 0.52)}, factsheets={"AAA": _fs("AAA", vol=None)})
    assert _pos(tp, "AAA").weight == pytest.approx(_kelly(0.52, 0, vol=0.30))
    tp2 = _run({"AAA": _agg("AAA", 0.501, 0.0)}, cfg=_cfg(min_prob_to_hold=0.5),
               factsheets={"AAA": _fs("AAA", vol=0.5)})
    assert _pos(tp2, "AAA").weight == pytest.approx(_kelly(0.501, 0, vol=0.05))
    assert _kelly(0.501, 0, vol=0.05) < 0.10 < _kelly(0.501, 0, vol=0.005)


# ----------------------------------------------------------------------------- caps and targets


def test_sector_cap_scales_names_proportionally():
    names = [f"T{i}" for i in range(6)]
    aggs = {t: _agg(t, 0.62, 3.0) for t in names}
    tp = _run(aggs, cfg=_cfg(target_vol=5.0), sectors={t: "Tech" for t in names})
    for t in names:
        assert _pos(tp, t).weight == pytest.approx(0.35 / 6)
        assert "sector_cap" in _pos(tp, t).constraints_hit
    assert any("sector cap: Tech" in n for n in tp.notes)


def test_sector_falls_back_to_factsheet_and_unclassified_names_are_noted():
    aggs = {t: _agg(t, 0.62, 3.0) for t in ("A", "B", "C", "D", "E")}
    fss = {t: _fs(t, sector="Energy") for t in ("A", "B", "C", "D")} | {"E": _fs("E", sector="")}
    tp = _run(aggs, cfg=_cfg(target_vol=5.0, max_sector=0.2), factsheets=fss)
    assert sum(_pos(tp, t).weight for t in "ABCD") == pytest.approx(0.2)
    assert _pos(tp, "E").weight == pytest.approx(0.10)
    assert any("no sector label" in n and "E" in n for n in tp.notes)


def _correlated_returns(tickers, n=300, daily=0.035, seed=0, end=AS_OF):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    common = rng.normal(0, daily, n)
    data = {t: common + rng.normal(0, daily * 0.2, n) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_vol_target_uses_sample_covariance():
    names = ["A", "B", "C", "D", "E"]
    rets = _correlated_returns(names)
    cfg = _cfg(max_sector=1.0, no_trade_band=0.0)
    tp = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=cfg, returns=rets)
    w = np.array([_pos(tp, t).weight for t in names])
    cov = rets[names].tail(126).cov().to_numpy() * 252
    assert math.sqrt(w @ cov @ w) == pytest.approx(cfg.portfolio.target_vol, rel=1e-9)
    assert all("vol_target" in _pos(tp, t).constraints_hit for t in names)
    assert any("sample covariance of 126" in n for n in tp.notes)


def test_vol_target_fallback_constant_correlation_model():
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cfg = _cfg(max_sector=1.0, target_vol=0.12, no_trade_band=0.0)
    tp = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=cfg,
              factsheets={t: _fs(t, vol=60.0) for t in names})
    w = np.array([_pos(tp, t).weight for t in names])
    s = np.full(len(names), 0.60)
    cov = 0.3 * np.outer(s, s)
    np.fill_diagonal(cov, s ** 2)
    assert math.sqrt(w @ cov @ w) == pytest.approx(0.12, rel=1e-9)
    assert any("constant-correlation" in n for n in tp.notes)


def test_returns_after_as_of_are_ignored():
    names = ["A", "B", "C"]
    rets = _correlated_returns(names)
    future = pd.DataFrame(0.5, index=pd.bdate_range(start="2026-02-02", periods=30), columns=names)
    cfg = _cfg(max_sector=1.0)
    a = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=cfg, returns=rets)
    b = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=cfg, returns=pd.concat([rets, future]))
    assert a.model_dump() == b.model_dump()


def test_short_or_sparse_returns_fall_back_to_model():
    names = ["A", "B"]
    rets = _correlated_returns(names, n=30)
    tp = _run({t: _agg(t, 0.62, 3.0) for t in names}, returns=rets, cfg=_cfg(target_vol=0.01))
    assert any("constant-correlation" in n and "only 30" in n for n in tp.notes)


def test_core_satellite_and_cash_floor():
    names = [f"T{i:02d}" for i in range(15)]
    aggs = {t: _agg(t, 0.62, 3.0) for t in names}
    sectors = {t: f"S{i}" for i, t in enumerate(names)}
    cfg = _cfg(target_vol=5.0)
    tp = _run(aggs, cfg=cfg, sectors=sectors)
    assert sum(p.weight for p in tp.positions) == pytest.approx(0.98)
    assert tp.cash_weight == pytest.approx(0.02)
    assert all("min_cash" in p.constraints_hit for p in tp.positions)

    core_cfg = _cfg(target_vol=5.0, core_weight=0.4)
    tp2 = _run(aggs, cfg=core_cfg, sectors=sectors)
    assert tp2.core_weight == pytest.approx(0.4)
    assert tp2.cash_weight == pytest.approx(0.02)   # the floor is real cash, not a fraction of the sleeve
    assert sum(p.weight for p in tp2.positions) == pytest.approx(0.58)

    small = _run({"AAA": _agg("AAA", 0.62, 3.0)}, cfg=_cfg(core_weight=0.4))
    assert _pos(small, "AAA").weight == pytest.approx(0.10 * 0.6)


def test_max_gross_counts_the_core():
    names = [f"T{i:02d}" for i in range(15)]
    tp = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=_cfg(target_vol=5.0, max_gross=0.7, core_weight=0.2),
              sectors={t: t for t in names})
    assert tp.core_weight + sum(abs(p.weight) for p in tp.positions) == pytest.approx(0.7)
    assert all("max_gross" in p.constraints_hit for p in tp.positions)


# ----------------------------------------------------------------------------- trade path


def test_no_trade_band_keeps_current_weight():
    tp = _run({"AAA": _agg("AAA", 0.62, 3.0)}, current={"AAA": 0.095})
    assert _pos(tp, "AAA").weight == pytest.approx(0.095)
    assert "no_trade_band" in _pos(tp, "AAA").constraints_hit


def test_turnover_cap_scales_all_trades():
    names = [f"T{i:02d}" for i in range(12)]
    cfg = _cfg(target_vol=5.0, max_turnover=0.5)
    tp = _run({t: _agg(t, 0.62, 3.0) for t in names}, cfg=cfg, sectors={t: t for t in names})
    assert sum(abs(p.weight) for p in tp.positions) == pytest.approx(0.5)
    assert all("turnover_cap" in p.constraints_hit for p in tp.positions)


def test_held_names_without_views_are_sold_unless_turnover_prevents_it():
    tp = _run({"AAA": _agg("AAA", 0.52)}, current={"OLD": 0.08})
    assert _pos(tp, "OLD").weight == 0
    assert "no committee view" in _pos(tp, "OLD").rationale
    slow = _run({"AAA": _agg("AAA", 0.62, 3.0)}, current={"OLD": 0.08}, cfg=_cfg(max_turnover=0.09))
    total_trade = abs(_pos(slow, "OLD").weight - 0.08) + _pos(slow, "AAA").weight
    assert total_trade == pytest.approx(0.09)
    assert 0 < _pos(slow, "OLD").weight < 0.08


def test_benchmark_is_the_core_not_an_active_name():
    tp = _run({"AAA": _agg("AAA", 0.52), BENCH: _agg(BENCH, 0.9, 10.0)}, cfg=_cfg(core_weight=0.3),
              current={BENCH: 0.3}, factsheets={"AAA": _fs("AAA")})
    assert all(p.ticker != BENCH for p in tp.positions)
    assert tp.core_weight == pytest.approx(0.3)
    assert any("benchmark" in n for n in tp.notes)
    # a held benchmark with core_weight 0 is sold like any other leg
    tp2 = _run({"AAA": _agg("AAA", 0.52)}, current={BENCH: 0.3})
    assert tp2.core_weight == 0.0


def test_long_only_never_shorts_and_shorts_when_allowed():
    aggs = {"BAD": _agg("BAD", 0.38, -4.0)}
    tp = _run(aggs)
    assert _pos(tp, "BAD").weight == 0
    tp2 = _run(aggs, cfg=_cfg(long_only=False))
    assert _pos(tp2, "BAD").weight == pytest.approx(-0.10)


def test_band_induced_sector_breach_is_repaired_by_trimming_buys():
    names = ["A", "B", "C", "D"]
    sectors = {t: "Tech" for t in names}
    current = {"A": 0.095, "B": 0.095, "C": 0.095, "D": 0.05}         # Tech = 0.335, within the cap
    aggs = {t: _agg(t, 0.62, 3.0) for t in names}
    tp = _run(aggs, cfg=_cfg(target_vol=5.0, max_sector=0.35), current=current, sectors=sectors)
    # ideal is 0.0875 each. A-C are within the band, so they stay at 0.095; D's buy to 0.0875
    # would put Tech at 0.3725, so the buy is trimmed to what the cap allows: 0.35 - 0.285.
    assert sum(_pos(tp, t).weight for t in names) == pytest.approx(0.35)
    assert _pos(tp, "A").weight == pytest.approx(0.095)
    assert _pos(tp, "D").weight == pytest.approx(0.065)
    assert "limit_trim" in _pos(tp, "D").constraints_hit


def test_out_of_bounds_current_book_is_brought_into_compliance():
    current = {"A": 0.30, "B": 0.30, "C": 0.39}                     # drifted: cash 1%, positions over cap
    aggs = {t: _agg(t, 0.62, 3.0) for t in current}
    tp = _run(aggs, cfg=_cfg(max_turnover=0.05, target_vol=5.0), current=current,
              sectors={"A": "x", "B": "y", "C": "z"})
    assert all(p.weight <= 0.10 + 1e-9 for p in tp.positions)
    assert tp.cash_weight >= 0.02 - 1e-9
    assert any("risk limits override the turnover cap" in n for n in tp.notes)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        _run({"AAA": _agg("AAA", 0.6)}, current={"AAA": float("nan")})
    with pytest.raises(ValueError):
        _run({"AAA": _agg("AAA", 0.6)}, cfg=_cfg(min_cash=1.2))
    with pytest.raises(ValueError):
        _run({"AAA": _agg("AAA", 0.6)}, cfg=_cfg(target_vol=0.0))


def test_deterministic():
    names = [f"T{i}" for i in range(8)]
    aggs = {t: _agg(t, 0.5 + 0.02 * i, i - 3.0, 0.05) for i, t in enumerate(names)}
    rets = _correlated_returns(names, seed=3)
    a = _run(aggs, returns=rets, current={"T1": 0.04, "T7": 0.02})
    b = _run(aggs, returns=rets, current={"T1": 0.04, "T7": 0.02})
    assert a.model_dump() == b.model_dump()


# ----------------------------------------------------------------------------- property tests


def _random_setup(rng):
    n = int(rng.integers(3, 16))
    tickers = [f"T{i:02d}" for i in range(n)]
    labels = ["Tech", "Energy", "Health", ""]
    sectors = {t: labels[int(rng.integers(0, 4))] for t in tickers}
    long_only = bool(rng.random() < 0.8)
    cfg = _cfg(
        core_weight=float(rng.choice([0.0, 0.2, 0.5])),
        long_only=long_only,
        max_position=float(rng.uniform(0.03, 0.2)),
        max_sector=float(rng.uniform(0.1, 0.5)),
        min_cash=float(rng.uniform(0.0, 0.1)),
        max_gross=float(rng.uniform(0.5, 1.0)) if long_only else float(rng.uniform(0.8, 1.6)),
        target_vol=float(rng.uniform(0.05, 0.3)),
        no_trade_band=float(rng.uniform(0.0, 0.03)),
        max_turnover=float(rng.uniform(0.05, 1.5)),
        kelly_fraction=float(rng.uniform(0.1, 1.0)),
    )
    vols = {t: float(rng.uniform(10, 80)) for t in tickers}
    fss = {t: _fs(t, vol=vols[t]) for t in tickers}
    return tickers, sectors, cfg, vols, fss


def _random_aggs(rng, tickers):
    chosen = [t for t in tickers if rng.random() < 0.85]
    return {t: _agg(t, float(rng.uniform(0.3, 0.7)), float(rng.normal(0, 4)), float(rng.uniform(0, 0.2)))
            for t in chosen}


def _check_hard_limits(tp, cfg, sectors, vols):
    pc = cfg.portfolio
    w = {p.ticker: p.weight for p in tp.positions}
    assert all(abs(x) <= pc.max_position + 1e-9 for x in w.values())
    if pc.long_only:
        assert all(x >= -1e-12 for x in w.values()) and tp.core_weight >= -1e-12
    by_sector = {}
    for t, x in w.items():
        if sectors.get(t):
            by_sector[sectors[t]] = by_sector.get(sectors[t], 0.0) + abs(x)
    assert all(g <= pc.max_sector + 1e-9 for g in by_sector.values())
    assert tp.cash_weight == pytest.approx(1 - tp.core_weight - sum(w.values()), abs=1e-12)
    assert tp.cash_weight >= pc.min_cash - 1e-9
    assert tp.core_weight + sum(abs(x) for x in w.values()) <= pc.max_gross + 1e-9
    names = sorted(w)
    s = np.array([vols[t] / 100 for t in names])
    cov = 0.3 * np.outer(s, s)
    np.fill_diagonal(cov, s ** 2)
    x = np.array([w[t] for t in names])
    core = min(pc.core_weight, 1 - pc.min_cash, pc.max_gross)
    assert math.sqrt(max(0.0, x @ cov @ x)) <= pc.target_vol * (1 - core) + 1e-9


@pytest.mark.parametrize("seed", range(60))
def test_property_limits_and_turnover_over_chained_cycles(seed):
    """Feed each cycle's output back as the next cycle's current book: every hard limit holds
    every cycle, and because the current book is always compliant, so does the turnover cap."""
    rng = np.random.default_rng(seed)
    tickers, sectors, cfg, vols, fss = _random_setup(rng)
    current: dict[str, float] = {}
    for _ in range(4):
        tp = construct_portfolio(_random_aggs(rng, tickers), {}, fss, current, cfg, AS_OF, sectors)
        _check_hard_limits(tp, cfg, sectors, vols)
        new = {p.ticker: p.weight for p in tp.positions} | {BENCH: tp.core_weight}
        turnover = sum(abs(new.get(t, 0.0) - current.get(t, 0.0)) for t in set(new) | set(current))
        assert turnover <= cfg.portfolio.max_turnover + 1e-9
        assert not any("override the turnover cap" in n for n in tp.notes)
        current = {t: x for t, x in new.items() if x != 0}


@pytest.mark.parametrize("seed", range(40))
def test_property_limits_hold_even_from_an_out_of_bounds_book(seed):
    rng = np.random.default_rng(1000 + seed)
    tickers, sectors, cfg, vols, fss = _random_setup(rng)
    current = {t: float(rng.uniform(0, 0.3)) for t in tickers if rng.random() < 0.6}
    if not cfg.portfolio.long_only:
        current = {t: x * float(rng.choice([-1, 1])) for t, x in current.items()}
    tp = construct_portfolio(_random_aggs(rng, tickers), {}, fss, current, cfg, AS_OF, sectors)
    _check_hard_limits(tp, cfg, sectors, vols)


@pytest.mark.parametrize("seed", range(20))
def test_property_vol_target_with_sample_covariance(seed):
    rng = np.random.default_rng(2000 + seed)
    names = [f"T{i}" for i in range(int(rng.integers(2, 10)))]
    rets = _correlated_returns(names, daily=float(rng.uniform(0.01, 0.05)), seed=seed)
    cfg = _cfg(max_sector=1.0, target_vol=float(rng.uniform(0.05, 0.25)), max_turnover=float(rng.uniform(0.1, 2)))
    aggs = {t: _agg(t, float(rng.uniform(0.5, 0.7)), float(rng.normal(1, 3))) for t in names}
    tp = _run(aggs, cfg=cfg, returns=rets)
    w = np.array([_pos(tp, t).weight for t in names])
    cov = rets[names].tail(126).cov().to_numpy() * 252
    assert math.sqrt(w @ cov @ w) <= cfg.portfolio.target_vol + 1e-9
