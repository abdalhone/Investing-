"""Tests for the backtest evaluation modules (signal IC and power, composition-matched random
books, breakeven bisection, factor attribution). Synthetic data only, no network."""

import math

import numpy as np
import pandas as pd
import pytest
from synthetic_market import make_prices, perturb_after

from aifirm.backtest.evaluation import breakeven_multiplier, cost_sensitivity
from aifirm.backtest.factors import FACTORS, factor_attribution
from aifirm.backtest.ic import (
    POOL,
    POOL_EQUAL,
    QUANT_ONLY,
    forward_excess_returns,
    ic_stats,
    information_coefficients,
    overlap_lags,
    power_analysis,
    probability_tables,
    rank_ic,
)
from aifirm.backtest.random_books import (
    build_schedule,
    random_books,
    realised_cost_rate,
    value_paths,
)
from aifirm.data.prices import from_frames

BENCH = "SPY"


# ----------------------------------------------------------------------------- a market with a planted signal


def planted_market(n_names=100, n_dates=120, step=21, beta=0.0024, seed=11):
    """Prices in which each decision date's signal ``s`` (one draw per name) shifts the next
    ``step`` daily returns by ``beta * s / step``: over a 21-day window with 1% daily noise that
    is a rank IC of about 0.05. Opens equal the previous close, so the execution-open entry
    is the decision close. Returns (prices, decisions, names); each decision record carries
    the planted signal as the pool, a pure-noise analyst and a signal-following analyst."""
    rng = np.random.default_rng(seed)
    t_len = n_dates * step + step + 2
    idx = pd.bdate_range("2015-01-01", periods=t_len)
    names = [f"N{i:03d}" for i in range(n_names)]
    daily = rng.normal(0.0, 0.01, (t_len, n_names))
    decisions = []
    for k in range(n_dates):
        d = k * step
        s = rng.normal(size=n_names)
        daily[d + 1:d + step + 1] += beta * s / step
        noise = rng.normal(size=n_names)
        decisions.append({
            "as_of": str(idx[d].date()),
            "pooled_prob": {t: 0.5 + 0.05 * math.tanh(x) for t, x in zip(names, s, strict=True)},
            "prob_equal_weight": {t: 0.5 + 0.04 * math.tanh(x) for t, x in zip(names, s, strict=True)},
            "n_views": dict.fromkeys(names, 2),
            "agent_probs": {t: {"signal": 0.5 + 0.05 * math.tanh(x), "noise": 0.5 + 0.05 * math.tanh(z)}
                            for t, x, z in zip(names, s, noise, strict=True)},
        })
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(daily, axis=0)), index=idx, columns=names)
    close[BENCH] = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, t_len)))
    prices = from_frames(close, open_=close.shift(1).bfill())
    return prices, decisions, names


@pytest.fixture(scope="module")
def planted():
    return planted_market()


def test_a_planted_ic_of_about_005_is_recovered_with_a_positive_t_stat(planted):
    prices, decisions, names = planted
    end = str(prices.close.index[-1].date())
    stats, diffs = information_coefficients(decisions, prices, names, BENCH, 21, 21, end, min_names=5)
    pool = stats[POOL]
    assert pool.n_dates == 120 and pool.lags == 0 and pool.method == "iid"
    assert 0.03 < pool.mean < 0.08                                      # planted IC ~ 0.05
    assert pool.t_stat > 3.0
    assert pool.t_stat == pytest.approx(pool.mean / pool.std * math.sqrt(pool.n_dates))
    assert pool.avg_names == pytest.approx(100)
    assert stats[POOL_EQUAL].mean == pytest.approx(pool.mean)            # a monotone transform: same ranks
    assert stats["agent:signal"].mean == pytest.approx(pool.mean)
    noise = stats["agent:noise"]
    assert abs(noise.mean) < 0.025 and abs(noise.t_stat) < 2.5          # a random signal shows no skill
    weighted_vs_equal = next(d for d in diffs if d.b == POOL_EQUAL)
    assert weighted_vs_equal.mean_diff == pytest.approx(0.0) and math.isnan(weighted_vs_equal.t_stat)


def test_paired_ic_difference_tells_a_signal_from_noise(planted):
    prices, decisions, names = planted
    quant = [{"as_of": d["as_of"], "pooled_prob": {t: v["noise"] for t, v in d["agent_probs"].items()}}
             for d in decisions]
    end = str(prices.close.index[-1].date())
    stats, diffs = information_coefficients(decisions, prices, names, BENCH, 21, 21, end, 5, quant_decisions=quant)
    vs_quant = next(d for d in diffs if (d.a, d.b) == (POOL, QUANT_ONLY))
    assert vs_quant.n_dates == 120
    assert vs_quant.mean_diff == pytest.approx(stats[POOL].mean - stats[QUANT_ONLY].mean)
    assert vs_quant.t_stat > 2.0


def test_power_analysis_matches_the_closed_form(planted):
    prices, decisions, names = planted
    end = str(prices.close.index[-1].date())
    pool = information_coefficients(decisions, prices, names, BENCH, 21, 21, end, 5)[0][POOL]
    rets = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 756))      # Sharpe ~1.6
    p = power_analysis(pool, 21, rets)
    assert p.dates_for_t2 == pytest.approx((2 * pool.std / pool.mean) ** 2)
    assert p.dates_for_80pct == pytest.approx((2.8016 * pool.std / pool.mean) ** 2, rel=1e-3)
    assert p.years_for_80pct == pytest.approx(p.dates_for_80pct * 21 / 252)
    assert p.anecdotal == (p.n_dates < p.dates_for_80pct)
    assert "decision dates" in p.ic_sentence() and "years" in p.ic_sentence()
    assert "breadth (~100 names per date)" in p.ic_sentence()
    assert p.min_trl_years > 0 and "MinTRL" in p.sharpe_sentence()


def test_power_analysis_with_no_positive_ic_says_no_amount_of_data_helps():
    stats = ic_stats("pool", pd.DataFrame({"A": [0.6, 0.4, 0.55], "B": [0.4, 0.6, 0.45], "C": [0.5, 0.5, 0.5]},
                                          index=pd.bdate_range("2024-01-01", periods=3)),
                     pd.DataFrame({"A": [-0.01, 0.02, -0.03], "B": [0.01, -0.02, 0.03], "C": [0.0, 0.0, 0.0]},
                                  index=pd.bdate_range("2024-01-01", periods=3)), min_names=3, lags=0)
    assert stats.mean < 0
    p = power_analysis(stats, 21, pd.Series(np.random.default_rng(2).normal(-0.001, 0.01, 300)))
    assert p.dates_for_t2 == math.inf and p.dates_for_80pct == math.inf and p.anecdotal
    assert "not positive" in p.ic_sentence()
    assert p.min_trl_years == math.inf and "no track record" in p.sharpe_sentence()


def test_overlapping_forecast_windows_use_newey_west(planted):
    prices, decisions, names = planted
    assert overlap_lags(21, 21) == 0 and overlap_lags(21, 5) == 4 and overlap_lags(5, 21) == 0
    end = str(prices.close.index[-1].date())
    stats, _ = information_coefficients(decisions, prices, names, BENCH, 63, 21, end, 5)
    pool = stats[POOL]
    assert pool.lags == 2 and pool.method == "Newey-West (2 lags)"
    assert pool.t_stat != pytest.approx(pool.mean / pool.std * math.sqrt(pool.n_dates))


def test_forward_excess_return_is_execution_open_to_resolution_close_on_adjusted_prices():
    """Hand-computed: entry at the next day's open on the adjusted basis (dividends count),
    exit at the close ``horizon`` sessions after the decision; nothing after ``end`` is used."""
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15, 16, 17], BENCH: [100] * 8}, index=idx, dtype=float)
    open_ = pd.DataFrame({"A": [10, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], BENCH: [100] * 8}, index=idx,
                         dtype=float)
    adj = close.copy()
    adj.loc[:idx[2], "A"] = close.loc[:idx[2], "A"] * 0.9          # a dividend went ex on idx[3]
    p = from_frames(close, adj_close=adj, open_=open_)
    fwd = forward_excess_returns(p, [idx[1], idx[4]], ["A"], BENCH, horizon_days=3, end=str(idx[6].date()))
    entry = 11.5 * 0.9                                              # idx[2] open, adjusted with idx[2]'s factor
    assert fwd.loc[idx[1], "A"] == pytest.approx(14.0 / entry - 1.0)   # exit: idx[4] adjusted close
    assert idx[4] not in fwd.index                                   # its window ends at idx[7], after end
    # point-in-time: scrambling every price after end changes nothing
    later = perturb_after(p, str(idx[6].date()))
    pd.testing.assert_frame_equal(fwd, forward_excess_returns(later, [idx[1], idx[4]], ["A"], BENCH, 3,
                                                              str(idx[6].date())))


def test_abstentions_and_thin_dates_are_not_forecasts():
    recs = [{"as_of": "2024-01-02", "pooled_prob": {"A": 0.5, "B": 0.6}, "n_views": {"A": 0, "B": 3},
             "prob_equal_weight": {"A": 0.5, "B": 0.58}}]
    tables = probability_tables(recs)
    assert list(tables[POOL].columns) == ["B"]                      # an all-abstained 0.5 is not a forecast
    probs = pd.Series({"A": 0.6, "B": 0.4, "C": 0.5})
    fwd = pd.Series({"A": 0.01, "B": -0.02, "C": 0.0})
    assert rank_ic(probs, fwd, min_names=3)[0] == pytest.approx(1.0)
    assert math.isnan(rank_ic(probs, fwd, min_names=4)[0])           # too few names
    assert math.isnan(rank_ic(pd.Series({"A": 0.5, "B": 0.5, "C": 0.5}), fwd, 3)[0])   # no variation


# ----------------------------------------------------------------------------- composition-matched random books


def _random_market(n_names=30, n_days=300, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    names = [f"R{i:02d}" for i in range(n_names)]
    close = pd.DataFrame(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, (n_days, n_names)), axis=0)), index=idx,
                         columns=names)
    close[BENCH] = 400.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n_days)))
    return from_frames(close, open_=close.shift(1).bfill()), names


def _decisions(prices, names, pick, step=21, n_hold=5, core=0.2):
    """Decision records that hold ``pick(date_index) -> names`` with equal weights."""
    idx = prices.close.index
    recs = []
    for d in range(0, len(idx) - 1, step):
        chosen = pick(d)
        w = (1.0 - core - 0.05) / len(chosen)
        recs.append({"as_of": str(idx[d].date()), "orders": [{"ticker": "x"}], "target": dict.fromkeys(chosen, w),
                     "core_weight": core, "cash_weight": 0.05})
    return recs


def test_a_random_selection_sits_near_the_median_of_random_books():
    """The null is calibrated: strategies that pick names at random land anywhere in the
    random-book distribution (uniformly), on average near the 50th percentile, and reach
    p < 0.05 about 5% of the time. 100 strategies: the mean percentile has a standard error
    of ~2.9, so the bounds below are ~2.7 standard errors wide."""
    prices, names = _random_market()
    days = prices.close.index
    pcts, pvals = [], []
    for k in range(100):
        rng = np.random.default_rng(100 + k)
        recs = _decisions(prices, names, lambda d, rng=rng: list(rng.choice(names, 5, replace=False)))
        res = random_books(build_schedule(recs, prices, days, names, BENCH), prices, 200, seed=k, cost_rate=0.0005,
                           engine_total_return_pct=0.0)
        pcts.append(res.percentile_return)
        pvals.append(res.p_value_return)
    assert 42 < np.mean(pcts) < 58
    assert np.mean(np.array(pvals) < 0.05) <= 0.12
    assert np.mean(np.array(pcts) < 25) > 0.12 and np.mean(np.array(pcts) > 75) > 0.12   # spread over the range


def test_perfect_foresight_beats_almost_every_random_book():
    prices, names = _random_market()
    days = prices.close.index
    fwd = prices.adj_close[names].shift(-21) / prices.adj_close[names] - 1

    def best(d):
        return list(fwd.iloc[d].nlargest(5).index)

    res = random_books(build_schedule(_decisions(prices, names, best), prices, days, names, BENCH), prices, 500,
                       seed=1, cost_rate=0.0005, engine_total_return_pct=0.0)
    assert res.percentile_return >= 99 and res.p_value_return <= 0.01 and res.distinguishable
    assert "beat random picks" in res.verdict()


def test_random_books_copy_the_composition_and_only_draw_priced_names():
    """Each random book holds exactly the strategy's weights at every rebalance; a name with no
    price yet at a decision date (listed later) is never drawn (point-in-time universe)."""
    prices, names = _random_market(n_names=12, n_days=120)
    close = prices.close.copy()
    late, gone = names[0], names[11]
    close.loc[close.index[:70], late] = np.nan                      # listed on day 70
    close.loc[close.index[50:], gone] = np.nan                      # delisted after day 49
    prices = from_frames(close, open_=close.shift(1).bfill())
    recs = _decisions(prices, names, lambda d: names[3:7] if d < 60 else names[5:9], n_hold=4)
    sched = build_schedule(recs, prices, prices.close.index, names, BENCH)
    assert not sched.eligible[0, 0] and sched.eligible[-1, 0]
    assert sched.eligible[2, 11] and not sched.eligible[3, 11]
    from aifirm.backtest.random_books import _random_weights
    mats = list(_random_weights(sched, 200, np.random.default_rng(0), sector_matched=False))
    assert (mats[2][:, 11] != 0).any()                              # some books held the name before it went
    for i, mat in enumerate(mats):
        strat = np.sort(sched.weights[i][:-1][sched.weights[i][:-1] != 0])
        for row in mat:
            assert np.allclose(np.sort(row[:-1][row[:-1] != 0]), strat)
            assert row[-1] == pytest.approx(sched.weights[i][-1])   # the core stays on the benchmark
        for j in (0, 11):
            if not sched.eligible[i, j]:                             # unpriced names are never held
                assert (mat[:, j] == 0).all()
    # holding periods are matched: names the strategy keeps keep their stand-in
    kept = [j for j in range(len(names)) if sched.weights[0][j] and sched.weights[1][j]]
    assert kept
    held0, held1 = mats[0][:, :-1] != 0, mats[1][:, :-1] != 0
    assert (held0 & held1).sum(axis=1).min() >= len(kept)


def test_sector_matched_books_replace_names_within_their_sector():
    prices, names = _random_market(n_names=12, n_days=80)
    sectors = {t: ("Tech" if i < 6 else "Energy") for i, t in enumerate(names)}
    recs = _decisions(prices, names, lambda d: [names[0], names[1], names[7]], n_hold=3)
    sched = build_schedule(recs, prices, prices.close.index, names, BENCH, sectors)
    from aifirm.backtest.random_books import _random_weights
    for mat in _random_weights(sched, 300, np.random.default_rng(3), sector_matched=True):
        tech = mat[:, :6].astype(bool).sum(axis=1)
        energy = mat[:, 6:12].astype(bool).sum(axis=1)
        assert (tech == 2).all() and (energy == 1).all()


def test_value_paths_match_a_hand_computed_book():
    """One name at 50%, 50% cash, cost 10 bps per dollar traded, rebalanced back to 50% once."""
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 100.0, 110.0, 121.0], BENCH: [10.0] * 4}, index=idx)
    prices = from_frames(close, open_=close.shift(1).bfill())
    recs = [{"as_of": str(idx[0].date()), "orders": [1], "target": {"A": 0.5}, "core_weight": 0.0},
            {"as_of": str(idx[2].date()), "orders": [1], "target": {"A": 0.5}, "core_weight": 0.0}]
    sched = build_schedule(recs, prices, idx, ["A"], BENCH)
    ac = prices.adj_close[["A", BENCH]].to_numpy()
    ao = prices.open[["A", BENCH]].to_numpy()
    v = value_paths(sched, ac, ao, iter(w[None, :] for w in sched.weights), 1, 0.001)[0]
    v1 = 1 - 0.001 * 0.5                                            # buy 50% at idx[1]'s open (100)
    assert v[0] == 1.0 and v[1] == pytest.approx(v1) and v[2] == pytest.approx(v1 * 1.05)
    drift = 0.55 / 1.05                                             # at idx[3]'s open (110) before rebalancing
    v_post = v1 * 1.05 * (1 - 0.001 * abs(0.5 - drift))
    assert v[3] == pytest.approx(v_post * (0.5 * 1.1 + 0.5))


def test_random_books_are_reproducible_and_refuse_an_empty_schedule():
    prices, names = _random_market()
    recs = _decisions(prices, names, lambda d: names[:5])
    sched = build_schedule(recs, prices, prices.close.index, names, BENCH)
    a = random_books(sched, prices, 100, seed=7, cost_rate=0.001, engine_total_return_pct=0.0)
    b = random_books(sched, prices, 100, seed=7, cost_rate=0.001, engine_total_return_pct=0.0)
    c = random_books(sched, prices, 100, seed=8, cost_rate=0.001, engine_total_return_pct=0.0)
    assert a.return_quantiles_pct == b.return_quantiles_pct and a.return_quantiles_pct != c.return_quantiles_pct
    assert a.gross_percentile_return >= 0 and a.cost_rate_bps == pytest.approx(10.0)
    no_trades = [{**r, "orders": []} for r in recs]
    with pytest.raises(ValueError, match="never traded"):
        random_books(build_schedule(no_trades, prices, prices.close.index, names, BENCH), prices, 10, 0, 0.0, 0.0)
    assert realised_cost_rate([{"fills": [{"quantity": 10, "price": 100.0, "commission": 0.05,
                                           "slippage_cost": 0.45}]}]) == pytest.approx(0.0005)


# ----------------------------------------------------------------------------- breakeven bisection


def test_breakeven_bisection_matches_a_hand_computed_case():
    """excess(k) = 12% - 3% * k crosses zero at k = 4 exactly."""
    b = breakeven_multiplier(lambda k: 0.12 - 0.03 * k, cap=50.0)
    assert b.status == "found" and b.multiplier == pytest.approx(4.0, abs=0.02)
    calls: list[float] = []
    seeded = breakeven_multiplier(lambda k: calls.append(k) or 0.12 - 0.03 * k, cap=50.0,
                                  known={0.0: 0.12, 1.0: 0.09, 2.0: 0.06, 5.0: -0.03})
    assert seeded.multiplier == pytest.approx(4.0, abs=0.02)
    assert all(2.0 <= k <= 5.0 for k in calls)                      # the known grid narrowed the bracket


def test_breakeven_edge_cases():
    assert breakeven_multiplier(lambda k: 0.05, 50.0).status == "above_cap"
    no = breakeven_multiplier(lambda k: -0.01 - 0.01 * k, 50.0)
    assert no.status == "no_edge" and no.multiplier == 0.0
    gains = breakeven_multiplier(lambda k: -0.02 + 0.01 * k, 50.0)      # the comparator trades more
    assert gains.status == "gains_above" and gains.multiplier == pytest.approx(2.0, abs=0.02)
    assert "ahead above 2.00x" in gains.describe("the quant-only baseline")
    with pytest.raises(ValueError, match="not finite"):
        breakeven_multiplier(lambda k: float("nan") if k > 0 else 0.1, 50.0)


def test_cost_sensitivity_reuses_simulations_and_expresses_the_breakeven_in_bps():
    calls: list[float] = []

    def strat(k):
        calls.append(k)
        return 10.0 - 2.5 * k                                       # percent

    cs = cost_sensitivity(strat, benchmark_total_return_pct=0.0, multipliers=[0.0, 1.0, 2.0], cap=50.0,
                          realised_cost_bps=6.0, engine_total_return_pct=7.5)
    assert cs.replay_matches_engine and cs.vs_quant is None
    assert cs.vs_benchmark.multiplier == pytest.approx(4.0, abs=0.02)
    assert cs.vs_benchmark.bps_one_way == pytest.approx(24.0, abs=0.2)  # 4x of 6 bps per dollar traded
    assert len(calls) == len(set(calls))                              # memoised: no k simulated twice
    assert [cs.excess_vs_benchmark(k) for k in cs.multipliers] == [10.0, 7.5, 5.0]


# ----------------------------------------------------------------------------- factor attribution


def _proxy_prices(n=600, seed=9, mtum_from=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    rets = {t: rng.normal(0.0003, 0.01, n) for t in ("SPY", "IWM", "IWD", "IWF", "MTUM", "QUAL", "USMV")}
    rets["BIL"] = np.full(n, 0.0001)
    adj = pd.DataFrame({t: 100 * np.cumprod(1 + r) for t, r in rets.items()}, index=idx)
    if mtum_from is not None:
        adj.loc[adj.index[:mtum_from], "MTUM"] = np.nan
    return adj


def test_factor_regression_recovers_planted_loadings_and_alpha():
    adj = _proxy_prices()
    r = adj.pct_change(fill_method=None)
    rng = np.random.default_rng(1)
    strat = (r["BIL"] + 0.0004 + 0.9 * (r["SPY"] - r["BIL"]) + 0.4 * (r["IWM"] - r["SPY"])
             - 0.3 * (r["MTUM"] - r["SPY"]) + rng.normal(0, 0.002, len(r))).iloc[1:]
    fa, notes = factor_attribution(strat, adj)
    assert fa is not None and fa.factors == [name for name, _, _ in FACTORS] and fa.cash_proxy == "BIL"
    assert fa.loadings["market"] == pytest.approx(0.9, abs=0.05)
    assert fa.loadings["size"] == pytest.approx(0.4, abs=0.05)
    assert fa.loadings["momentum"] == pytest.approx(-0.3, abs=0.05)
    assert fa.loadings["value"] == pytest.approx(0.0, abs=0.05)
    assert fa.alpha_annual_pct == pytest.approx(0.0004 * 252 * 100, abs=3.0) and fa.alpha_t > 2
    assert fa.proxies["size"] == "IWM - SPY" and fa.proxies["market"] == "SPY - BIL" and not notes


def test_factor_attribution_skips_missing_proxies_with_a_note():
    adj = _proxy_prices(mtum_from=300)                              # momentum ETF listed half way through
    strat = adj["SPY"].pct_change(fill_method=None).iloc[1:] * 0.8
    fa, notes = factor_attribution(strat, adj.drop(columns=["BIL"]))
    assert fa is not None and "momentum" not in fa.factors and "market" in fa.factors
    assert any("momentum factor skipped" in n and "cover only" in n for n in notes)
    assert any("cash return is taken as 0" in n for n in notes) and fa.cash_proxy is None
    none, why = factor_attribution(strat, None)
    assert none is None and why
    no_market, why = factor_attribution(strat, adj.drop(columns=["SPY"]))
    assert no_market is None and any("market factor" in n for n in why)


def test_synthetic_market_prices_work_for_the_market_factor_only():
    p = make_prices()
    strat = p.adj_close["AAA"].pct_change(fill_method=None).iloc[300:]
    fa, notes = factor_attribution(strat, p)
    assert fa is not None and fa.factors == ["market"] and any("size factor skipped" in n for n in notes)


def test_percentiles_read_as_ordinals():
    from aifirm.backtest.random_books import ordinal
    assert [ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 62.4, 99.6, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "62nd", "100th", "100th"]


def test_a_book_holding_the_whole_universe_is_flagged_as_a_weak_null():
    prices, names = _random_market(n_names=6, n_days=80)
    recs = _decisions(prices, names, lambda d: names[:4] if d == 0 else list(names), n_hold=6)
    res = random_books(build_schedule(recs, prices, prices.close.index, names, BENCH), prices, 50, seed=0,
                       cost_rate=0.0, engine_total_return_pct=0.0)
    assert res.min_distinct_books == 1
    assert any("held 6 of the 6 eligible names, which leaves only 1 distinct book:" in n for n in res.notes)


@pytest.mark.parametrize("invested", [0.2, 0.7, 1.0])
def test_cash_that_earns_nothing_in_the_simulator_is_not_negative_alpha(invested):
    """Regression: y = r - BIL charged the T-bill rate on uninvested cash that earns 0 in the
    paper broker, so a book holding x of NAV in SPY and nothing else showed alpha
    -(1 - x) * BIL with a huge t-stat. With the invested fraction the intercept is ~0 and the
    cash drag is reported separately."""
    adj = _proxy_prices()
    adj["BIL"] = 100 * np.cumprod(np.full(len(adj), 1 + 0.04 / 252))     # a 4% T-bill rate
    r = adj.pct_change(fill_method=None)
    strat = (invested * r["SPY"]).iloc[1:]                                # cash earns 0
    exposure = pd.Series(invested, index=adj.index)
    naive, _ = factor_attribution(strat, adj)
    fa, _ = factor_attribution(strat, adj, invested=exposure)
    assert naive.alpha_annual_pct == pytest.approx(-(1 - invested) * 4.0, abs=0.05)   # the artifact
    assert fa.alpha_annual_pct == pytest.approx(0.0, abs=1e-6) and fa.loadings["market"] == pytest.approx(invested)
    assert fa.cash_drag_annual_pct == pytest.approx((1 - invested) * 4.0, abs=0.05)
    assert fa.mean_invested == pytest.approx(invested)
    assert naive.cash_drag_annual_pct is None


def test_invested_fraction_of_the_previous_close_earns_the_day():
    adj = _proxy_prices(n=300)
    r = adj.pct_change(fill_method=None)
    x = pd.Series(np.where(np.arange(len(adj)) < 150, 0.0, 1.0), index=adj.index)   # buys at the close of day 149
    strat = (x.shift(1) * r["SPY"]).iloc[1:]
    fa, _ = factor_attribution(strat, adj, invested=x)
    days_in_cash = 150                                                  # return days 1..150 were earned by cash
    assert fa.cash_drag_annual_pct == pytest.approx(0.0001 * days_in_cash / fa.n_obs * 252 * 100, rel=1e-9)
    assert fa.mean_invested == pytest.approx(1 - days_in_cash / fa.n_obs)
