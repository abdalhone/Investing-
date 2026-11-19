"""Factor attribution with ETF proxies (research report 7.4).

A strategy that "beats the market" by holding small, cheap, trending or low-volatility
stocks has found a known factor premium, not skill: those premia are cheap to buy with
an ETF. Regressing the strategy's daily excess returns on factor returns separates the
two. What is left, the intercept (alpha), is the part no listed factor explains.

Contract
--------
* Factors are long-short return differences of liquid ETFs (daily total returns from
  adjusted closes):

  ============== ================= ==========================================
  factor          proxy             meaning
  ============== ================= ==========================================
  market          SPY - cash        equity market premium
  size            IWM - SPY         small caps vs large caps
  value           IWD - IWF         Russell 1000 value vs growth
  momentum        MTUM - SPY        momentum tilt (MTUM listed April 2013)
  quality         QUAL - SPY        quality tilt (QUAL listed July 2013)
  low_volatility  USMV - SPY        minimum-volatility tilt (USMV listed October 2011)
  ============== ================= ==========================================

  Cash is BIL (1-3 month T-bills). Without BIL the cash return is taken as 0 and a note
  says so.
* **The dependent variable matches the simulator.** The paper broker pays no interest on
  cash, so a book that is ``x`` invested earns ``x * R_assets + (1 - x) * 0``. Subtracting
  the full T-bill return (``r - rf``) would charge ``(1 - x) * rf`` of cash drag to the
  intercept: a book holding 20% SPY and nothing else would show an alpha of about
  ``-0.8 * rf`` per year with a huge t-statistic, which is not selection. With
  ``invested`` (the daily fraction of NAV in positions, ``BacktestResult.exposure``),
  the dependent variable is ``r_t - x_{t-1} * rf_t``, where ``x_{t-1}`` is the fraction
  invested at the previous close (the book that earned day t's return); the cash leg is
  compared with the 0 it actually earned. ``cash_drag_annual_pct`` reports the T-bill
  return the uninvested cash forwent, ``mean((1 - x_{t-1}) * rf_t) * 252``, separately: a
  cost of holding cash, never alpha. Without ``invested`` the book is treated as fully
  invested (``y = r - rf``; ``cash_drag_annual_pct`` and ``mean_invested`` are None). The
  backtest always passes it.
* A factor is used only if both its legs have returns on at least
  :data:`MIN_COVERAGE` of the strategy's return days; otherwise it is dropped with a note
  (for example momentum before 2013). Without the market factor nothing is reported.
  Days on which any used factor is missing are left out of the regression.
* OLS with Newey-West (HAC) standard errors, lags ``floor(4 (n/100)^(2/9))``. Alpha is
  annualised (x 252) and reported in percent with its HAC t-statistic.
* Ex post only: factor returns are realised returns over the backtest window; they are
  never shown to an agent and never reach a decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.prices import PriceData
from .metrics import TRADING_DAYS, newey_west_lags, ols_hac

CASH_PROXY = "BIL"
# (factor, long leg, short leg); a short leg of None means "minus cash"
FACTORS: tuple[tuple[str, str, str | None], ...] = (
    ("market", "SPY", None),
    ("size", "IWM", "SPY"),
    ("value", "IWD", "IWF"),
    ("momentum", "MTUM", "SPY"),
    ("quality", "QUAL", "SPY"),
    ("low_volatility", "USMV", "SPY"),
)
PROXY_TICKERS: tuple[str, ...] = tuple(dict.fromkeys([CASH_PROXY, *(t for _, a, b in FACTORS for t in (a, b) if t)]))
MIN_COVERAGE = 0.95
MIN_OBS = 60


@dataclass
class FactorAttribution:
    """Result of the factor regression (see module docstring)."""

    factors: list[str]                  # used, in regression order
    proxies: dict[str, str]             # factor -> "IWM - SPY"
    loadings: dict[str, float]
    loading_t: dict[str, float]
    alpha_annual_pct: float
    alpha_t: float
    r2: float
    n_obs: int
    hac_lags: int
    cash_proxy: str | None              # None: cash return taken as 0
    notes: list[str] = field(default_factory=list)
    # T-bill return forgone on uninvested cash (annualised %, the simulator pays none); None
    # when the invested fraction was not given
    cash_drag_annual_pct: float | None = None
    mean_invested: float | None = None


def _proxy_returns(proxies: PriceData | pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    adj = proxies.adj_close if isinstance(proxies, PriceData) else proxies
    return adj.pct_change(fill_method=None).reindex(days)


def factor_attribution(returns: pd.Series, proxies: PriceData | pd.DataFrame | None,
                       factors: Sequence[tuple[str, str, str | None]] = FACTORS, *,
                       invested: pd.Series | None = None) -> tuple[FactorAttribution | None, list[str]]:
    """Regress daily strategy ``returns`` minus the cash the book actually held at the
    T-bill rate on the available ETF factor proxies (module docstring).

    ``proxies``: prices (or a frame of adjusted closes) holding some of
    :data:`PROXY_TICKERS`. ``invested``: the daily fraction of NAV in positions at each
    close (``BacktestResult.exposure``); day t's return is charged ``x_{t-1} * rf_t``, with
    gaps forward-filled and values clipped to [0, 1]; None treats the book as fully
    invested. Returns ``(result, notes)``; ``result`` is None, with notes saying why, when
    proxies are missing, the market factor is unavailable, or there are too few days for a
    regression. Never raises for missing data."""
    notes: list[str] = []
    r = returns.dropna()
    if proxies is None:
        return None, ["no factor proxy prices were available (pass factor_prices, or let run_backtest download them)"]
    if r.empty:
        return None, ["no strategy returns to attribute"]
    rets = _proxy_returns(proxies, pd.DatetimeIndex(r.index))
    cash_name: str | None = CASH_PROXY
    if CASH_PROXY in rets.columns and rets[CASH_PROXY].notna().mean() >= MIN_COVERAGE:
        cash = rets[CASH_PROXY].fillna(0.0)
    else:
        cash, cash_name = pd.Series(0.0, index=r.index), None
        notes.append(f"{CASH_PROXY} (cash) prices unavailable: the cash return is taken as 0")
    cols: dict[str, pd.Series] = {}
    shown: dict[str, str] = {}
    for name, long_leg, short_leg in factors:
        legs = [long_leg] + ([short_leg] if short_leg else [])
        missing = [t for t in legs if t not in rets.columns]
        if missing:
            notes.append(f"{name} factor skipped: no prices for {', '.join(missing)}")
            continue
        series = rets[long_leg] - (rets[short_leg] if short_leg else cash)
        coverage = float(series.notna().mean())
        if coverage < MIN_COVERAGE:
            first = rets[legs].dropna().index.min()
            since = f" (data from {first.date()})" if pd.notna(first) else ""
            notes.append(f"{name} factor skipped: {'/'.join(legs)} cover only {coverage:.0%} of the backtest "
                         f"days{since}")
            continue
        cols[name] = series
        shown[name] = f"{long_leg} - {short_leg or ('cash' if cash_name is None else cash_name)}"
    if "market" not in cols:
        return None, [*notes, "factor attribution skipped: the market factor (SPY) is unavailable"]
    frame = pd.DataFrame(cols)
    if invested is not None and len(invested.dropna()):
        prev = invested.astype(float).sort_index().shift(1).reindex(r.index.union(invested.index)).ffill()
        x_prev = prev.reindex(r.index).fillna(1.0).clip(0.0, 1.0)
    else:
        x_prev = None          # treated as fully invested (the documented default)
    frame["y"] = r - (cash if x_prev is None else x_prev * cash)
    frame["_drag"] = 0.0 if x_prev is None else (1.0 - x_prev) * cash
    frame["_x"] = 1.0 if x_prev is None else x_prev
    frame = frame.dropna()
    drag, xs = frame.pop("_drag"), frame.pop("_x")
    names = [c for c in frame.columns if c != "y"]
    if len(frame) < max(MIN_OBS, len(names) + 10):
        return None, [*notes, f"factor attribution skipped: only {len(frame)} days with every factor"]
    x = np.column_stack([np.ones(len(frame)), frame[names].to_numpy()])
    lags = newey_west_lags(len(frame))
    try:
        fit = ols_hac(frame["y"].to_numpy(), x, lags)
    except ValueError as exc:
        return None, [*notes, f"factor attribution skipped: {exc}"]
    return FactorAttribution(
        factors=names, proxies={n: shown[n] for n in names},
        loadings={n: float(fit.coef[i + 1]) for i, n in enumerate(names)},
        loading_t={n: float(fit.t[i + 1]) for i, n in enumerate(names)},
        alpha_annual_pct=float(fit.coef[0] * TRADING_DAYS * 100), alpha_t=float(fit.t[0]), r2=fit.r2,
        n_obs=fit.n, hac_lags=fit.lags, cash_proxy=cash_name, notes=notes,
        cash_drag_annual_pct=None if x_prev is None else float(drag.mean() * TRADING_DAYS * 100),
        mean_invested=None if x_prev is None else float(xs.mean())), notes
