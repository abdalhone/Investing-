import math

import numpy as np
import pandas as pd
import pytest

from aifirm.backtest.metrics import (
    brier_score,
    deflated_sharpe,
    max_drawdown,
    min_track_record_length,
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
