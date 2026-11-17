"""Recent headlines (live mode only).

Free headline feeds (Yahoo via yfinance) only expose the latest ~10-20 stories, so they
cannot be replayed point-in-time. In historical/backtest mode the news desk is disabled
and agents are told explicitly that no news is available. This mirrors TradingAgents'
choice to refuse live-only data in historical runs instead of leaking the present.
For historical news use a timestamped archive (e.g. GDELT, a paid vendor) and filter by
publish time <= as_of.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd


@dataclass
class Headline:
    title: str
    publisher: str
    published: str  # ISO timestamp (UTC)
    summary: str = ""


def _parse_item(item: dict) -> Headline | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title")
    if not title:
        return None
    provider = content.get("provider") or {}
    publisher = provider.get("displayName") if isinstance(provider, dict) else None
    publisher = publisher or content.get("publisher") or ""
    ts = content.get("pubDate") or content.get("displayTime")
    if not ts and content.get("providerPublishTime"):
        ts = datetime.fromtimestamp(int(content["providerPublishTime"]), tz=timezone.utc).isoformat()
    summary = content.get("summary") or ""
    return Headline(title=title.strip(), publisher=publisher, published=str(ts or ""), summary=summary[:400])


def recent_headlines(ticker: str, as_of: date | str, max_items: int = 12, max_age_days: int = 14) -> list[Headline]:
    """Return headlines published on/before ``as_of`` within ``max_age_days``. Live use only."""
    import yfinance as yf

    as_of_ts = pd.Timestamp(as_of).tz_localize("UTC") + pd.Timedelta(days=1)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out: list[Headline] = []
    for item in raw:
        h = _parse_item(item)
        if not h or not h.published:
            continue
        try:
            ts = pd.Timestamp(h.published)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            continue
        if ts > as_of_ts or ts < as_of_ts - pd.Timedelta(days=max_age_days + 1):
            continue
        out.append(h)
    return out[:max_items]
