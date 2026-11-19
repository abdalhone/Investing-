import math

import numpy as np
import pandas as pd
import pytest

from aifirm.backtest.metrics import (
    brier_score,
    deflated_sharpe,
    long_run_variance,
    max_drawdown,
    mean_t_stat,
    min_track_record_length,
    newey_west_lags,
    ols_hac,
    performance_report,
    probabilistic_sharpe,
    sharpe,
)


def _series(mu=0.0005, sd=0.01, n=756, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2022-01-03", periods=n))


def test_sharpe_annualization():
    r = _series()
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_max_drawdown():
    eq = pd.Series([1.0, 1.2, 0.9, 1.3, 0.65])
    assert max_drawdown(eq) == pytest.approx(0.65 / 1.3 - 1)


def test_max_drawdown_counts_losses_from_the_starting_capital():
    """Regression: the equity curve built from returns must start at the initial 1.0, or a
    loss on the first day(s) is never measured against the starting capital."""
    idx = pd.bdate_range("2024-01-01", periods=12)
    two_losses = pd.Series([-0.05, -0.05] + [0.0] * 10, index=idx)
    assert performance_report(two_losses).max_drawdown_pct == pytest.approx((0.95 * 0.95 - 1) * 100)
    first_day_loss = pd.Series([-0.10] + [0.01] * 11, index=idx)
    rep = performance_report(first_day_loss)
    assert rep.max_drawdown_pct == pytest.approx(-10.0)
    assert not math.isnan(rep.calmar)                         # a real drawdown gives a real Calmar ratio
    assert max_drawdown(pd.Series([0.9, 1.0]), initial=1.0) == pytest.approx(-0.1)


def test_psr_bounds_and_monotonic():
    good, bad = _series(mu=0.001), _series(mu=-0.001)
    assert 0.5 < probabilistic_sharpe(good) <= 1.0
    assert 0.0 <= probabilistic_sharpe(bad) < 0.5


def test_deflated_sharpe_penalizes_many_trials():
    r = _series(mu=0.0006)
    assert deflated_sharpe(r, n_trials=100) < deflated_sharpe(r, n_trials=2) <= probabilistic_sharpe(r)


def test_min_track_record_infinite_without_edge():
    assert min_track_record_length(_series(mu=-0.001)) == float("inf")


def test_performance_report_with_benchmark():
    r, b = _series(mu=0.0006, seed=2), _series(mu=0.0004, seed=3)
    rep = performance_report(r, b, turnover=pd.Series([0.1, 0.2]), costs=pd.Series([0.001, 0.002]), n_trials=5)
    assert rep.days == len(r)
    assert rep.excess_return_pct == pytest.approx(rep.total_return_pct - rep.benchmark_total_return_pct)
    assert rep.avg_turnover_pct == pytest.approx(15.0)
    assert rep.total_costs_pct == pytest.approx(0.3)
    assert rep.deflated_sharpe is not None


def test_brier():
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


# ----------------------------------------------------------------------------- Newey-West and HAC regression


def test_long_run_variance_without_lags_is_the_sample_variance():
    x = _series(mu=0.001, sd=0.02, n=200)
    assert long_run_variance(x, 0) == pytest.approx(x.var(ddof=1))
    assert mean_t_stat(x, 0) == pytest.approx(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
    assert math.isnan(mean_t_stat(pd.Series([0.1])))                       # one value: no t-stat
    assert math.isnan(mean_t_stat(pd.Series([0.1, 0.1, 0.1])))             # no variation: no t-stat


def test_newey_west_t_stat_discounts_overlapping_observations():
    """Regression guard: overlapping forward windows (e.g. 21-day forecasts made every 5 days)
    are positively autocorrelated; an iid t-stat would overstate significance."""
    rng = np.random.default_rng(3)
    shocks = rng.normal(0.01, 0.05, 400)
    overlapped = pd.Series(np.convolve(shocks, np.ones(4) / 4, mode="valid"))   # MA(3): 4 overlapping draws
    assert long_run_variance(overlapped, 3) > long_run_variance(overlapped, 0)
    assert abs(mean_t_stat(overlapped, 3)) < abs(mean_t_stat(overlapped, 0))
    assert newey_west_lags(100) == 4 and newey_west_lags(1000) == 6


def test_ols_hac_recovers_known_coefficients_and_matches_hc1_without_lags():
    rng = np.random.default_rng(5)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.3 + 1.5 * x1 - 0.7 * x2 + rng.normal(0, 0.5, n)
    x = np.column_stack([np.ones(n), x1, x2])
    fit = ols_hac(y, x, lags=0)
    assert fit.coef == pytest.approx([0.3, 1.5, -0.7], abs=0.08)
    # lags = 0 is White's heteroskedasticity-robust covariance with the n/(n-k) correction (HC1)
    e = y - x @ fit.coef
    inv = np.linalg.inv(x.T @ x)
    hc1 = inv @ (x.T * e ** 2) @ x @ inv * n / (n - 3)
    assert fit.se == pytest.approx(np.sqrt(np.diag(hc1)))
    assert fit.t == pytest.approx(fit.coef / fit.se)
    assert 0.8 < fit.r2 < 1.0 and fit.n == n


def test_ols_hac_refuses_degenerate_regressions():
    x = np.column_stack([np.ones(10), np.ones(10)])                        # a constant "factor"
    with pytest.raises(ValueError, match="collinear"):
        ols_hac(np.arange(10.0), x, 2)
    with pytest.raises(ValueError, match="more observations"):
        ols_hac(np.arange(2.0), np.ones((2, 2)), 0)
