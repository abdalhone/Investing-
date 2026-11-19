"""Performance and statistical-significance metrics.

Includes the corrections from Bailey & Lopez de Prado for judging a backtest honestly:
the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe Ratio (DSR, which accounts for
how many strategy variants were tried), and the Minimum Track Record Length (MinTRL).
A strategy whose DSR is below ~0.95 has not shown skill distinguishable from luck.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


@dataclass
class PerformanceReport:
    start: str
    end: str
    days: int
    total_return_pct: float
    cagr_pct: float
    vol_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    hit_rate_pct: float
    psr: float                 # P(true Sharpe > 0)
    benchmark_total_return_pct: float | None = None
    excess_return_pct: float | None = None
    information_ratio: float | None = None
    beta: float | None = None
    alpha_annual_pct: float | None = None
    avg_turnover_pct: float | None = None
    total_costs_pct: float | None = None
    deflated_sharpe: float | None = None
    min_track_record_days: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = returns.dropna() - rf_daily
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = returns.dropna() - rf_daily
    downside = r[r < 0]
    if len(r) < 2 or len(downside) < 2:
        return 0.0
    dd = math.sqrt((downside ** 2).sum() / len(r))
    return float(r.mean() / dd * math.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def max_drawdown(equity: pd.Series, initial: float | None = None) -> float:
    """Worst peak-to-trough fall of an equity curve (a negative fraction, or 0).

    ``initial`` is the equity before the first point (e.g. 1.0 for a curve built from
    returns): it is the first peak, so a loss on the very first day counts. Without it a
    curve that starts with a loss would measure its drawdown from the already-lower first
    point and understate it."""
    values = equity.dropna().to_numpy(dtype=float)
    if initial is not None:
        values = np.concatenate([[float(initial)], values])
    if values.size == 0:
        return 0.0
    return float((values / np.maximum.accumulate(values) - 1.0).min())


def probabilistic_sharpe(returns: pd.Series, sr_benchmark_annual: float = 0.0) -> float:
    """PSR: probability that the true (annualized) Sharpe exceeds ``sr_benchmark_annual``,
    adjusting for sample length, skewness and kurtosis (Bailey & Lopez de Prado 2012)."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)                     # per-period Sharpe
    sr_b = sr_benchmark_annual / math.sqrt(TRADING_DAYS)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0                      # pandas returns excess kurtosis
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_b) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sharpe_trials: float) -> float:
    """Expected maximum (per-period) Sharpe among ``n_trials`` unskilled strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sharpe_trials) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sharpe_trials: float | None = None) -> float:
    """DSR: PSR against the Sharpe you would expect from the best of ``n_trials`` lucky tries.

    ``var_sharpe_trials`` is the variance of the (per-period) Sharpe ratios across the
    variants you tried; if unknown we use the asymptotic variance of a single estimate.
    Record every configuration/prompt variant you test - honesty about n_trials is the
    whole point.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return float("nan")
    if var_sharpe_trials is None:
        var_sharpe_trials = 1.0 / max(1, n - 1)
    sr0 = expected_max_sharpe(max(1, n_trials), var_sharpe_trials)
    return probabilistic_sharpe(r, sr0 * math.sqrt(TRADING_DAYS))


def min_track_record_length(returns: pd.Series, sr_benchmark_annual: float = 0.0, confidence: float = 0.95) -> float:
    """Number of observations needed for PSR to reach ``confidence`` given current stats."""
    r = returns.dropna()
    if len(r) < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    sr_b = sr_benchmark_annual / math.sqrt(TRADING_DAYS)
    if sr <= sr_b:
        return float("inf")
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    z = norm.ppf(confidence)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / (sr - sr_b)) ** 2)


def performance_report(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    n_trials: int = 1,
) -> PerformanceReport:
    """Summarize a daily *net* return series (after costs)."""
    r = returns.dropna()
    if r.empty:
        raise ValueError("empty return series")
    equity = (1 + r).cumprod()
    days = len(r)
    years = days / TRADING_DAYS
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if days > 1 else 0.0
    mdd = max_drawdown(equity, initial=1.0)     # the starting capital is the first high-water mark
    rep = PerformanceReport(
        start=str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        end=str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        days=days,
        total_return_pct=total * 100,
        cagr_pct=cagr * 100,
        vol_pct=vol * 100,
        sharpe=sharpe(r),
        sortino=sortino(r),
        max_drawdown_pct=mdd * 100,
        calmar=float(cagr / abs(mdd)) if mdd < 0 and not math.isnan(cagr) else float("nan"),
        hit_rate_pct=float((r > 0).mean() * 100),
        psr=probabilistic_sharpe(r),
        deflated_sharpe=deflated_sharpe(r, n_trials) if n_trials > 1 else None,
        min_track_record_days=min_track_record_length(r),
    )
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).fillna(0.0)
        rep.benchmark_total_return_pct = float(((1 + b).prod() - 1) * 100)
        rep.excess_return_pct = rep.total_return_pct - rep.benchmark_total_return_pct
        active = r - b
        if active.std(ddof=1) > 0:
            rep.information_ratio = float(active.mean() / active.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if b.var() > 0:
            beta = float(r.cov(b) / b.var())
            rep.beta = beta
            rep.alpha_annual_pct = float((r.mean() - beta * b.mean()) * TRADING_DAYS * 100)
    if turnover is not None and len(turnover):
        rep.avg_turnover_pct = float(turnover.mean() * 100)
    if costs is not None and len(costs):
        rep.total_costs_pct = float(costs.sum() * 100)
    return rep


def long_run_variance(x: np.ndarray | pd.Series, lags: int = 0) -> float:
    """Newey-West (Bartlett kernel) long-run variance of a series about its mean.

    ``Gamma_l = sum_{t>l} (x_t - mean)(x_{t-l} - mean) / (n - 1)`` and
    ``LRV = Gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * Gamma_l``. With ``lags = 0`` this is
    exactly the sample variance (ddof=1), so a t-stat built on it reduces to the plain
    ``mean / std * sqrt(n)``. Bartlett weights keep the estimate non-negative. NaNs are
    dropped first; fewer than two values give NaN. ``lags`` is capped at ``n - 1``."""
    v = np.asarray(pd.Series(x, dtype=float).dropna(), dtype=float)
    n = v.size
    if n < 2:
        return float("nan")
    if np.ptp(v) == 0:
        return 0.0                         # identical values: exactly no variation (no rounding residue)
    d = v - v.mean()
    lags = max(0, min(int(lags), n - 1))
    lrv = float(d @ d) / (n - 1)
    for lag in range(1, lags + 1):
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * float(d[lag:] @ d[:-lag]) / (n - 1)
    return max(lrv, 0.0)


def mean_t_stat(x: np.ndarray | pd.Series, lags: int = 0) -> float:
    """t-statistic of the mean of ``x``: ``mean / sqrt(LRV / n)`` with the Newey-West
    long-run variance of :func:`long_run_variance` (``lags = 0``: the iid t-stat,
    ``mean / std * sqrt(n)``). Use ``lags > 0`` when consecutive values overlap (for
    example forecasts whose horizons are longer than the spacing between them). NaN when
    there are fewer than two values or no variation."""
    v = pd.Series(x, dtype=float).dropna()
    lrv = long_run_variance(v, lags)
    if not math.isfinite(lrv) or lrv <= 0:
        return float("nan")
    return float(v.mean() / math.sqrt(lrv / len(v)))


def newey_west_lags(n: int) -> int:
    """The usual automatic lag choice for daily data, ``floor(4 * (n / 100) ** (2/9))``."""
    return math.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))


@dataclass
class OLSResult:
    """Ordinary least squares with heteroskedasticity- and autocorrelation-consistent
    (Newey-West) standard errors. ``coef[0]`` is the intercept when the caller added one."""

    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    r2: float
    n: int
    lags: int


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int) -> OLSResult:
    """OLS of ``y`` (n,) on ``x`` (n, k; include a column of ones for an intercept) with
    Newey-West HAC covariance ``(X'X)^-1 S (X'X)^-1``, ``S = sum_l w_l (G_l + G_l')`` with
    Bartlett weights and ``G_l = sum_t e_t e_{t-l} x_t x_{t-l}'``, scaled by ``n / (n - k)``.
    Raises ``ValueError`` when there are not more observations than regressors or ``X'X`` is
    singular (a constant or duplicated factor), so a degenerate regression is never reported."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    if n <= k or y.shape != (n,):
        raise ValueError(f"need more observations than regressors (n={n}, k={k})")
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < k:
        raise ValueError("regressors are collinear (a constant or duplicated factor)")
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ x.T @ y
    e = y - x @ beta
    xe = x * e[:, None]
    lags = max(0, min(int(lags), n - 1))
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        s += (1.0 - lag / (lags + 1.0)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv * n / (n - k)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / np.where(se > 0, se, 1.0), np.nan)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else float("nan")
    return OLSResult(coef=beta, se=se, t=t, r2=r2, n=n, lags=lags)


def brier_score(probs: list[float] | np.ndarray, outcomes: list[int] | np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - o) ** 2))
