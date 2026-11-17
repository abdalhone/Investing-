"""Synthetic market data for the offline integration tests (no network, deterministic).

Not a test module (pytest collects ``test_*.py`` only); imported by test_pipeline_e2e.py
and test_backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aifirm.config import FirmConfig
from aifirm.data.fundamentals import FactValue, Fundamentals
from aifirm.data.macro import MacroValue
from aifirm.data.prices import PriceData, from_frames

NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy", "EEE": "Health", "FFF": "Health"}


def make_prices(n: int = 560, seed: int = 7, start: str = "2023-01-02") -> PriceData:
    """Daily prices for NAMES plus SPY. Names share a market factor and differ in drift
    (AAA worst, FFF best), so momentum and trend signals rank them clearly. Opens differ from
    the prior close (so next-open fills are distinguishable from close fills)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    mkt = rng.normal(0.0004, 0.009, n)
    drifts = np.linspace(-0.0015, 0.0015, len(NAMES))
    close = pd.DataFrame({t: 40.0 * (1 + k) * np.exp(np.cumsum(mkt + drifts[k] + rng.normal(0, 0.012, n)))
                          for k, t in enumerate(NAMES)} | {"SPY": 400.0 * np.exp(np.cumsum(mkt))}, index=idx)
    gap = pd.DataFrame(rng.normal(0.0, 0.004, close.shape), index=idx, columns=close.columns)
    open_ = close.shift(1).bfill() * (1.0 + gap)
    volume = pd.DataFrame(rng.uniform(3e6, 6e6, close.shape), index=idx, columns=close.columns)
    return from_frames(close, volume=volume, open_=open_)


def perturb_after(prices: PriceData, after: str, seed: int = 99) -> PriceData:
    """A copy whose prices and volumes strictly after ``after`` are scrambled."""
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp(after)

    def scramble(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > ts
        out.loc[mask] = out.loc[mask].to_numpy() * rng.uniform(lo, hi, out.loc[mask].shape)
        return out

    return PriceData(close=scramble(prices.close, 0.5, 1.5), adj_close=scramble(prices.adj_close, 0.5, 1.5),
                     volume=scramble(prices.volume, 0.2, 3.0), open=scramble(prices.open, 0.5, 1.5))


def fake_fundamentals(ticker: str, as_of: str) -> Fundamentals:
    """Plausible TTM fundamentals that improve from AAA to FFF (quality and value agree with
    the price trend, so the analysts' slices point the same way)."""
    h = (NAMES.index(ticker) + 0.5) / len(NAMES) if ticker in NAMES else 0.5
    revenue = 5e9 * (1 + 2 * h)
    net_income = revenue * (0.02 + 0.2 * h)
    equity = revenue * 0.8
    period = str((pd.Timestamp(as_of) - pd.Timedelta(days=45)).date())

    def fv(v: float) -> FactValue:
        return FactValue(v, period, period, "10-Q", "synthetic", "TTM (4 quarters)")

    values = {"revenue": fv(revenue), "net_income": fv(net_income), "operating_income": fv(net_income * 1.3),
              "op_cash_flow": fv(net_income * 1.2), "capex": fv(net_income * 0.3), "equity": fv(equity)}
    prior = {"revenue": fv(revenue / (0.95 + 0.3 * h)), "net_income": fv(net_income / (0.9 + 0.4 * h))}
    return Fundamentals(ticker=ticker, cik="0", entity_name=f"{ticker} Corp", as_of=as_of,
                        sic_description=SECTORS.get(ticker, ""), values=values, prior_year=prior,
                        shares_outstanding=FactValue(2e8, period, period, "10-Q", "synthetic", "instant"))


def fake_macro(as_of: str) -> list[MacroValue]:
    return [MacroValue("VIXCLS", "VIX implied volatility index", "index", 16.0, as_of, -1.0),
            MacroValue("DGS10", "10-year Treasury yield", "%", 4.2, as_of, 0.1),
            MacroValue("DGS2", "2-year Treasury yield", "%", 4.0, as_of, -0.1)]


def fake_filings(ticker: str, as_of: str) -> list[dict]:
    filed = str((pd.Timestamp(as_of) - pd.Timedelta(days=20)).date())
    return [{"form": "8-K", "filed": filed, "items": "2.02,9.01", "description": ""},
            {"form": "10-Q", "filed": filed, "items": "", "description": ""}]


LOADERS = {"fundamentals_loader": fake_fundamentals, "macro_loader": fake_macro, "filings_loader": fake_filings}


def make_config(runs_dir: str, **process: object) -> FirmConfig:
    cfg = FirmConfig(runs_dir=runs_dir)
    cfg.universe.tickers = list(NAMES)
    cfg.universe.sectors = dict(SECTORS)
    for k, v in process.items():
        setattr(cfg.process, k, v)
    return cfg
