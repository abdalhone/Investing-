"""Macro / market-regime data from FRED (no API key needed for the CSV endpoint).

Point-in-time handling: FRED's CSV serves the *latest revised* values. Daily market series
(yields, VIX, credit spreads) are never revised, so we only apply a 1-day availability lag.
Monthly economic series are revised and published with a delay, so we apply a conservative
publication lag. For exact vintages use ALFRED (https://alfred.stlouisfed.org).
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
    "CPIAUCSL": ("CPI inflation, year over year", "%", 45, "yoy"),
    "UNRATE": ("Unemployment rate", "%", 38, "level"),
}


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


def macro_snapshot(as_of: date | str, cache: DiskCache | None = None,
                   series: list[str] | None = None) -> list[MacroValue]:
    as_of_ts = pd.Timestamp(as_of)
    out: list[MacroValue] = []
    for sid in series or list(SERIES):
        label, unit, lag, transform = SERIES[sid]
        try:
            s = load_series(sid, cache)
        except Exception:
            continue
        if transform == "yoy":
            s = (s / s.shift(12) - 1.0) * 100.0
            s = s.dropna()
        visible = s[s.index <= as_of_ts - pd.Timedelta(days=lag)]
        if visible.empty:
            continue
        cur_date = visible.index[-1]
        prior = visible[visible.index <= cur_date - pd.Timedelta(days=91)]
        change = float(visible.iloc[-1] - prior.iloc[-1]) if not prior.empty else None
        out.append(MacroValue(sid, label, unit, float(visible.iloc[-1]), str(cur_date.date()), change))
    return out
