"""Daily price history with point-in-time slicing.

Two price series are kept on purpose:

* ``close``     - the raw (split-adjusted only, as Yahoo reports it) closing price. Used for
                  *levels* shown to agents (price, market cap) and for trade fills.
* ``adj_close`` - dividend- and split-adjusted close. Used only for *returns*.

Dividend back-adjustment rewrites historical price levels using dividends paid *after*
the as-of date. TradingAgents shows agents ``auto_adjust=True`` levels, which leaks a
little future information. Returns computed from ``adj_close`` are fine; levels are not.

Survivorship caveat: Yahoo does not serve most delisted tickers, so a universe chosen
today and back-tested is biased toward survivors. Use a point-in-time universe (e.g.
index membership as of each date) for serious research.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .cache import DiskCache, default_cache


@dataclass
class PriceData:
    close: pd.DataFrame       # index: DatetimeIndex, columns: tickers
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    open: pd.DataFrame | None = None   # raw open; used for next-open execution in backtests

    def as_of(self, as_of: date | str | pd.Timestamp) -> "PriceData":
        """Return only rows dated on or before ``as_of`` (no look-ahead)."""
        ts = pd.Timestamp(as_of)
        return PriceData(
            close=self.close.loc[:ts],
            adj_close=self.adj_close.loc[:ts],
            volume=self.volume.loc[:ts],
            open=self.open.loc[:ts] if self.open is not None else None,
        )

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]

    def returns(self) -> pd.DataFrame:
        return self.adj_close.pct_change(fill_method=None)


def _download_yf(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers, start=start, end=end, auto_adjust=False, actions=False,
        progress=False, group_by="column", threads=True, multi_level_index=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {tickers} {start}..{end}")
    out = {}
    for field in ("Open", "Close", "Adj Close", "Volume"):
        frame = raw[field]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(tickers[0])
        frame = frame.reindex(columns=tickers)
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        out[field] = frame.astype(float)
    return out


def load_prices(
    tickers: list[str],
    start: date | str,
    end: date | str,
    cache: DiskCache | None = None,
) -> PriceData:
    """Load daily OHLCV-derived frames for ``tickers`` between ``start`` and ``end`` inclusive.

    Historical windows that end more than 3 days ago are cached permanently; windows that
    touch the present are cached for 6 hours.
    """
    cache = cache or default_cache()
    tickers = sorted({t.upper() for t in tickers})
    start_s = str(pd.Timestamp(start).date())
    # yfinance's ``end`` is exclusive
    end_s = str((pd.Timestamp(end) + timedelta(days=1)).date())
    recent = pd.Timestamp(end) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    key = f"{','.join(tickers)}|{start_s}|{end_s}"
    frames = cache.get_or_set(
        "prices", key, lambda: _download_yf(tickers, start_s, end_s),
        ttl_seconds=6 * 3600 if recent else None,
    )
    return PriceData(close=frames["Close"], adj_close=frames["Adj Close"], volume=frames["Volume"],
                     open=frames.get("Open"))


def from_frames(close: pd.DataFrame, adj_close: pd.DataFrame | None = None,
                volume: pd.DataFrame | None = None, open_: pd.DataFrame | None = None) -> PriceData:
    """Build PriceData from in-memory frames (used by tests and custom data sources)."""
    adj = adj_close if adj_close is not None else close.copy()
    vol = volume if volume is not None else pd.DataFrame(1e6, index=close.index, columns=close.columns)
    op = open_ if open_ is not None else close.copy()
    return PriceData(close=close.astype(float), adj_close=adj.astype(float), volume=vol.astype(float),
                     open=op.astype(float))
