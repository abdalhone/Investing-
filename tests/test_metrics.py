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
