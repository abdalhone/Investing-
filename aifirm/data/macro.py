"""Macro / market-regime data from FRED (no API key needed for the CSV endpoint).

Point-in-time handling: FRED's CSV serves the *latest revised* values. Daily market series
(yields, VIX, credit spreads) are never revised, so we only apply a 1-day availability lag.
Monthly economic series are revised and published with a delay. Exact release vintages
(ALFRED) are not available here, so an observation becomes visible only when BOTH hold:

* ``as_of >= observation date + a conservative lag`` chosen to be on or after the regular
  release date in every month (CPI is released around the 10th-15th of the next month, at
  most ~45 days after the 1st of the reference month; the jobs report by ~the 10th, at most
  ~40 days). The lag errs *late*: a print can be hidden for up to ~10 days after it was
  public, never shown before it was; and
* ``as_of >`` any release date listed in :data:`DELAYED_RELEASES`. Government shutdowns pushed
  some releases weeks past the usual date (October 2013, October-November 2025); those are
  listed explicitly because no fixed lag is long enough for them without hiding every
  normal month for weeks. Add an entry when a release slips past the lag.

Remaining limitation: the value shown is the latest revision, not the first print.

Year-over-year changes are computed on calendar months (value / value 12 months earlier),
not 12 rows back: FRED leaves out months that were never published (October 2025 CPI), and a
positional shift would silently turn every later "year over year" into a 13-month change.
A month whose base month is missing has no YoY value.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .cache import DiskCache, default_cache
from .http import get

# series id -> (label, unit, availability lag in days, transform)
SERIES: dict[str, tuple[str, str, int, str]] = {
    "DGS10": ("10-year Treasury yield", "%", 1, "level"),
    "DGS2": ("2-year Treasury yield", "%", 1, "level"),
    "DGS3MO": ("3-month Treasury bill yield", "%", 1, "level"),
    "VIXCLS": ("VIX implied volatility index", "index", 1, "level"),
    "BAMLH0A0HYM2": ("High-yield credit spread (ICE BofA OAS)", "%", 1, "level"),
    "DFF": ("Effective federal funds rate", "%", 1, "level"),
    "CPIAUCSL": ("CPI inflation, year over year", "%", 50, "yoy"),
    "UNRATE": ("Unemployment rate", "%", 46, "level"),
}

# (series id, observation date) -> date the value was actually published, for releases that
# came out later than SERIES' lag allows. Sources: BLS release schedules / press releases.
DELAYED_RELEASES: dict[tuple[str, str], str] = {
    # October 2013 federal shutdown
    ("UNRATE", "2013-09-01"): "2013-10-22",
    ("CPIAUCSL", "2013-09-01"): "2013-10-30",
    # October-November 2025 federal shutdown (October 2025 CPI and UNRATE were never published)
    ("CPIAUCSL", "2025-09-01"): "2025-10-24",
    ("UNRATE", "2025-09-01"): "2025-11-20",
}

# A "3-month change" must compare with a value within this many days of 3 months earlier.
_CHANGE_TOLERANCE_DAYS = 10


@dataclass
class MacroValue:
    series_id: str
    label: str
    unit: str
    value: float
    observation_date: str
    change_3m: float | None


def _download(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = get(url, host_key="fred", min_interval=0.3).text
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def load_series(series_id: str, cache: DiskCache | None = None) -> pd.Series:
    cache = cache or default_cache()
    return cache.get_or_set("fred", series_id, lambda: _download(series_id), ttl_seconds=12 * 3600)


def yoy_percent(s: pd.Series) -> pd.Series:
    """Year-over-year % change of a monthly series, matched on calendar months.

    Each observation is divided by the observation of the same month one year earlier; when
    that month is missing (never published) the result for that month is left out."""
    if s.empty:
        return s
    per = pd.Series(s.to_numpy(dtype=float), index=s.index.to_period("M"))
    per = per[~per.index.duplicated(keep="last")]
    base = pd.Series(per.to_numpy(), index=per.index + 12)   # value 12 months earlier, keyed by month
    out = (per / base.reindex(per.index) - 1.0) * 100.0
    out.index = out.index.to_timestamp()
    return out.dropna()


def visible_series(sid: str, s: pd.Series, as_of: date | str | pd.Timestamp) -> pd.Series:
    """Observations of ``sid`` that had been published by the close of ``as_of``."""
    as_of_ts = pd.Timestamp(as_of)
    lag = SERIES[sid][2] if sid in SERIES else 0
    visible = s[s.index <= as_of_ts - pd.Timedelta(days=lag)]
    if visible.empty:
        return visible
    hidden = [pd.Timestamp(obs) for (series_id, obs), released in DELAYED_RELEASES.items()
              if series_id == sid and pd.Timestamp(released) > as_of_ts]
    if hidden:
        visible = visible[~visible.index.isin(hidden)]
    return visible


def _change_3m(visible: pd.Series) -> float | None:
    """Change versus the value ~3 calendar months before the latest one (None if there is
    no observation close enough to that date)."""
    cur_date = visible.index[-1]
    target = cur_date - pd.DateOffset(months=3)
    prior = visible[visible.index <= target]
    if prior.empty or prior.index[-1] < target - pd.Timedelta(days=_CHANGE_TOLERANCE_DAYS):
        return None
    return float(visible.iloc[-1] - prior.iloc[-1])


def macro_snapshot(as_of: date | str, cache: DiskCache | None = None,
                   series: list[str] | None = None) -> list[MacroValue]:
    as_of_ts = pd.Timestamp(as_of)
    out: list[MacroValue] = []
    for sid in series or list(SERIES):
        label, unit, _lag, transform = SERIES[sid]
        try:
            s = load_series(sid, cache)
        except Exception:
            continue
        # Visibility is decided on the raw observations (a YoY value needs its own month
        # published); the YoY base month is older still, so it was public too.
        s = visible_series(sid, s, as_of_ts)
        if transform == "yoy":
            s = yoy_percent(s)
        if s.empty:
            continue
        cur_date = s.index[-1]
        out.append(MacroValue(sid, label, unit, float(s.iloc[-1]), str(cur_date.date()), _change_3m(s)))
    return out
