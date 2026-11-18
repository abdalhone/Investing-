"""Daily price history with point-in-time slicing.

Price series kept on purpose:

* ``close``     - Yahoo's close (``auto_adjust=False``). Yahoo back-adjusts this series for
                  every stock split up to the *download* date, so for dates before a later
                  split it is NOT the price that printed that day (NVDA's 2024-06-07 close shows
                  as ~120.9, while the stock actually closed near 1,209 before its 10:1 split).
                  Because the whole series is on one consistent basis it is the right series for
                  fills, share accounting, dollar volume and the backtest's mark-to-market.
* ``raw_close`` - the **as-traded** close: what a person looking at the tape on that day saw.
                  Rebuilt from ``close`` by un-applying every split whose ex-date is after the
                  row's date (``raw_t = close_t x product of split ratios with ex-date > t``).
                  Each row only depends on that day's actual print, so slicing at ``as_of`` keeps
                  it point-in-time. Used for *levels* shown to agents (the price fact, market cap
                  and therefore every valuation multiple) and for the minimum-price rule, so
                  that a price level is consistent with the as-filed SEC share count.
* ``adj_close`` - dividend- and split-adjusted close. Used only for *returns*.
* ``splits``    - split ratios by ex-date (``10.0`` = 10-for-1, ``0.1`` = 1-for-10 reverse),
                  used to bring an as-filed share count (dated before a split) onto the
                  as-of share basis. ``as_of`` keeps only splits with ex-date <= as_of.

Dividend back-adjustment rewrites historical price levels using dividends paid *after*
the as-of date. TradingAgents shows agents ``auto_adjust=True`` levels, which leaks a
little future information. Returns computed from ``adj_close`` are fine; levels are not.

Completed sessions only: Yahoo serves today's bar while the market is open, and its
"close" is then just the latest intraday print. :func:`load_prices` drops every row after
:func:`last_completed_session` (the latest weekday whose close is final in New York) and
never caches a partial bar, for every caller, not only the CLI (P3).

Survivorship caveat: Yahoo does not serve most delisted tickers, so a universe chosen
today and back-tested is biased toward survivors. Use a point-in-time universe (e.g.
index membership as of each date) for serious research.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from .cache import DiskCache, default_cache

log = logging.getLogger(__name__)

# Split events are downloaded from this many days before the requested window start, so a
# share count dated up to ~15 months before an as-of date can still be put on the as-of
# share basis (fundamentals rejects share counts older than that).
SPLIT_HISTORY_DAYS = 550
# A download in which some ticker came back empty (Yahoo throttling is common) is only
# cached this long, so a transient failure cannot poison later runs of the same window.
INCOMPLETE_TTL_SECONDS = 30 * 60
RECENT_TTL_SECONDS = 6 * 3600
_CACHE_VERSION = "v2"   # v2 adds as-traded closes and split events; older entries are not reused


def _market_tz() -> tzinfo:
    """New York time; without a time-zone database (e.g. Windows without tzdata) fixed EST,
    which reads the clock an hour early in summer and so errs toward the previous session."""
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=-5), "EST")


MARKET_TZ = _market_tz()
SESSION_CLOSE_SETTLED = (16, 30)    # after 16:30 New York time the day's close is final


def last_completed_session(now: datetime | None = None) -> str:
    """The latest weekday whose close is final in New York (a holiday resolves to the last
    trading day through the price index). Deciding on a partial intraday bar would be
    deciding on a price that is not a close (P3)."""
    t = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    day = t.date()
    if t.weekday() < 5 and (t.hour, t.minute) < SESSION_CLOSE_SETTLED:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return str(day)


@dataclass
class PriceData:
    close: pd.DataFrame       # index: DatetimeIndex, columns: tickers (split-adjusted basis, see module doc)
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    open: pd.DataFrame | None = None        # same basis as close; used for next-open execution in backtests
    raw_close: pd.DataFrame | None = None   # as-traded close; None = ``close`` is already as-traded
    splits: pd.DataFrame | None = None      # index: ex-date, columns: tickers, values: split ratio (1.0 = none)

    def as_of(self, as_of: date | str | pd.Timestamp) -> PriceData:
        """Return only rows (and split events) dated on or before ``as_of`` (no look-ahead)."""
        ts = pd.Timestamp(as_of)
        return PriceData(
            close=self.close.loc[:ts],
            adj_close=self.adj_close.loc[:ts],
            volume=self.volume.loc[:ts],
            open=self.open.loc[:ts] if self.open is not None else None,
            raw_close=self.raw_close.loc[:ts] if self.raw_close is not None else None,
            splits=self.splits.loc[:ts] if self.splits is not None else None,
        )

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]

    def returns(self) -> pd.DataFrame:
        return self.adj_close.pct_change(fill_method=None)

    def price_level(self, ticker: str) -> pd.Series:
        """As-traded closing levels for ``ticker`` (falls back to ``close`` when this data
        has no separate as-traded series, i.e. ``close`` is already as-traded)."""
        if self.raw_close is not None and ticker in self.raw_close.columns:
            return self.raw_close[ticker]
        return self.close[ticker]

    def split_ratio(self, ticker: str, after: date | str | pd.Timestamp,
                    through: date | str | pd.Timestamp | None = None) -> float:
        """Product of the split ratios of ``ticker`` with ``after`` < ex-date <= ``through``
        (default: every split in this data, i.e. through the as-of date after slicing).
        Multiply a share count dated ``after`` by this to put it on the ``through`` basis."""
        if self.splits is None or ticker not in self.splits.columns or self.splits.empty:
            return 1.0
        s = self.splits[ticker]
        s = s[s.index > pd.Timestamp(after)]
        if through is not None:
            s = s[s.index <= pd.Timestamp(through)]
        out = 1.0
        for r in s:
            if r is not None and math.isfinite(r) and r > 0:
                out *= float(r)
        return out


# --------------------------------------------------------------------------- Yahoo download


def _field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame | None:
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            return None
        frame = raw[field]
    else:
        if field not in raw.columns:
            return None
        frame = raw[[field]].rename(columns={field: tickers[0]})
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(tickers[0])
    frame = frame.reindex(columns=tickers)
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame.astype(float)


def _yf_frames(tickers: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers, start=start, end=end, auto_adjust=False, actions=True,
        progress=False, group_by="column", threads=True, multi_level_index=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for field in ("Open", "Close", "Adj Close", "Volume", "Stock Splits"):
        frame = _field(raw, field, tickers)
        if frame is not None:
            out[field] = frame
    return out


def _complete(frames: dict[str, pd.DataFrame], t: str, start: str, end: str) -> bool:
    """True if ``t`` has closes and adjusted closes inside the requested window."""
    for field in ("Close", "Adj Close"):
        f = frames.get(field)
        if f is None or t not in f.columns:
            return False
        if f[t].loc[pd.Timestamp(start):pd.Timestamp(end) - pd.Timedelta(days=1)].dropna().empty:
            return False
    return True


def _as_traded(close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Undo Yahoo's split back-adjustment: multiply each row by the ratios of every split
    with a later ex-date (all splits Yahoo had applied when the data was downloaded)."""
    raw = close.copy()
    for t in splits.columns:
        if t not in raw.columns:
            continue
        for ex, r in splits[t].items():
            if r is not None and math.isfinite(r) and r > 0 and r != 1.0:
                raw.loc[raw.index < ex, t] = raw.loc[raw.index < ex, t] * float(r)
    return raw


def _download_yf(tickers: list[str], start: str, end: str) -> dict:
    """Download daily bars for [start, end) plus split events.

    The download runs from ``start - SPLIT_HISTORY_DAYS`` to *today*: Yahoo's closes are
    split-adjusted through the download date, so the split events must come from the same
    response (through today) to rebuild the as-traded level; the extra history before
    ``start`` carries splits that happened after an older SEC share count was dated.
    Tickers that come back empty are retried one by one; any still empty are listed under
    ``"missing"`` so the caller caches the result only briefly.
    """
    dl_start = str((pd.Timestamp(start) - pd.Timedelta(days=SPLIT_HISTORY_DAYS)).date())
    frames = _yf_frames(tickers, dl_start, None)
    for t in [t for t in tickers if not _complete(frames, t, start, end)]:
        try:   # a throttled ticker in a batch often succeeds on its own
            one = _yf_frames([t], dl_start, None)
        except Exception as exc:  # noqa: BLE001 - recorded as missing, never cached for long
            log.warning("price download retry for %s failed: %s", t, exc)
            continue
        if _complete(one, t, start, end):
            for field, f in one.items():
                base = frames.get(field)
                if base is None:
                    base = pd.DataFrame(index=f.index, columns=tickers, dtype=float)
                idx = base.index.union(f.index)
                base = base.reindex(idx)
                base[t] = f[t].reindex(idx)
                frames[field] = base
    missing = [t for t in tickers if not _complete(frames, t, start, end)]
    if len(missing) == len(tickers):
        raise RuntimeError(f"No price data returned for {tickers} {start}..{end}")

    # Never keep (or cache) a bar whose close is not final yet: today's while the market is open.
    settled = pd.Timestamp(last_completed_session())
    close = frames["Close"]
    splits_daily = frames.get("Stock Splits")
    if splits_daily is None:
        splits_daily = pd.DataFrame(0.0, index=close.index, columns=tickers)
    splits_daily = splits_daily.reindex(columns=tickers).fillna(0.0)
    events = splits_daily[(splits_daily > 0).any(axis=1)].replace(0.0, 1.0)
    raw_close = _as_traded(close, events)

    lo, hi = pd.Timestamp(start), min(pd.Timestamp(end) - pd.Timedelta(days=1), settled)
    out: dict = {}
    for field, frame in (("Open", frames.get("Open")), ("Close", close), ("Adj Close", frames["Adj Close"]),
                         ("Volume", frames.get("Volume")), ("Raw Close", raw_close)):
        if frame is None:
            frame = pd.DataFrame(index=close.index, columns=tickers, dtype=float)
        out[field] = frame.reindex(columns=tickers).loc[lo:hi].astype(float)
    # split events known by the end of the window (later ones are already folded into Raw Close)
    out["Splits"] = events.loc[:hi].astype(float)
    out["missing"] = missing
    out["complete_through"] = str(settled.date())
    return out


def load_prices(
    tickers: list[str],
    start: date | str,
    end: date | str,
    cache: DiskCache | None = None,
) -> PriceData:
    """Load daily OHLCV-derived frames for ``tickers`` between ``start`` and ``end`` inclusive.

    Complete historical windows that end more than 3 days ago are cached permanently;
    windows that touch the present are cached for 6 hours, and not past the completion of a
    later session. A download in which any ticker came back empty is cached for only
    :data:`INCOMPLETE_TTL_SECONDS` (and logged), so a throttled request is retried instead
    of silently dropping the ticker for good.

    Rows after :func:`last_completed_session` are never returned or cached: an ``end`` of
    today (before the close settles) or in the future yields data through the last
    completed session, never an intraday bar presented as a close (P3).
    """
    cache = cache or default_cache()
    tickers = sorted({t.upper() for t in tickers})
    start_s = str(pd.Timestamp(start).date())
    # yfinance's ``end`` is exclusive
    end_s = str((pd.Timestamp(end) + timedelta(days=1)).date())
    recent = pd.Timestamp(end) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    key = f"{_CACHE_VERSION}|{','.join(tickers)}|{start_s}|{end_s}"
    settled = last_completed_session()
    wanted = min(str(pd.Timestamp(end).date()), settled)

    def ttl_for(value: dict) -> float | None:
        if value.get("missing"):
            return INCOMPLETE_TTL_SECONDS
        through = value.get("complete_through")
        if through is not None and str(through) < wanted:
            return 0.0                    # a session within the window has completed since: refetch
        return RECENT_TTL_SECONDS if recent else None

    frames = cache.get_or_set("prices", key, lambda: _download_yf(tickers, start_s, end_s), ttl_for=ttl_for)
    if frames.get("missing"):
        log.warning("no Yahoo prices for %s in %s..%s (cached briefly, will be retried)",
                    frames["missing"], start_s, end_s)
    cut = pd.Timestamp(settled)     # also guards entries cached before this rule existed

    def done(df: pd.DataFrame | None) -> pd.DataFrame | None:
        return df.loc[:cut] if df is not None else None

    return PriceData(close=done(frames["Close"]), adj_close=done(frames["Adj Close"]), volume=done(frames["Volume"]),
                     open=done(frames.get("Open")), raw_close=done(frames.get("Raw Close")),
                     splits=done(frames.get("Splits")))


def from_frames(close: pd.DataFrame, adj_close: pd.DataFrame | None = None,
                volume: pd.DataFrame | None = None, open_: pd.DataFrame | None = None,
                raw_close: pd.DataFrame | None = None, splits: pd.DataFrame | None = None) -> PriceData:
    """Build PriceData from in-memory frames (used by tests and custom data sources).

    ``close`` is taken as the as-traded level unless ``raw_close`` is given (pass both, plus
    ``splits``, when ``close`` is back-adjusted for splits)."""
    adj = adj_close if adj_close is not None else close.copy()
    vol = volume if volume is not None else pd.DataFrame(1e6, index=close.index, columns=close.columns)
    op = open_ if open_ is not None else close.copy()
    raw = raw_close if raw_close is not None else close.copy()
    return PriceData(close=close.astype(float), adj_close=adj.astype(float), volume=vol.astype(float),
                     open=op.astype(float), raw_close=raw.astype(float),
                     splits=splits.astype(float) if splits is not None else None)
