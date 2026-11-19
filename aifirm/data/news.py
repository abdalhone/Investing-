"""Headlines (live mode only), the point-in-time news cutoff and the source-tier table.

Free headline feeds (Yahoo via yfinance) only expose the latest ~10-20 stories, so they
cannot be replayed point-in-time. In historical/backtest mode the news desk is disabled
and agents are told explicitly that no news is available. This mirrors TradingAgents'
choice to refuse live-only data in historical runs instead of leaking the present.
For historical news use a timestamped archive (e.g. GDELT, a paid vendor) and filter it
with :func:`headlines_as_of`.

Point-in-time rule (P3)
-----------------------
A decision is made at the close of ``as_of``: 16:00 America/New_York on that date, or
13:00 on the exchange's scheduled half days (:func:`aifirm.data.calendar.session_close_time`:
the day after Thanksgiving, and Christmas Eve and 3 July on Monday to Thursday), see
:func:`decision_close_utc` (daylight saving handled by the time-zone database). Only a
headline whose parsed publish time is at or before that instant may be read: a 14:30
headline on a 13:00 half day came after the close the forecast is scored from. A headline
with a missing or unparseable time is dropped (fail closed): its age cannot be proven. A
caller may still pass an explicit ``close_time``.

Source tiers (RESEARCH 5.5, 9.3 item 3)
---------------------------------------
The tier of a headline is assigned by *code* from the publisher name and the link host,
using the maintained table :data:`NEWS_SOURCES`; an LLM never sets or changes it.

* **Tier A** - primary sources: SEC filings (sec.gov), the issuer's own investor-relations
  site (only for domains the caller passes in ``issuer_domains``) and the official
  newswires that carry company releases (Business Wire, PR Newswire, GlobeNewswire).
* **Tier B** - established news organisations: Reuters, Bloomberg, Associated Press, The
  Wall Street Journal, Financial Times, CNBC, Barron's, MarketWatch, The New York Times.
* **Tier C** - everything else: aggregators, blogs, opinion sites, unknown publishers,
  a missing or non-http link, and any attribution the link does not confirm.

Matching (:func:`source_tier`) trusts the link host over the publisher string, because
the publisher field is free text that anyone can set while the host is where the story
actually lives:

1. A link whose host is (a subdomain of) a table entry's domain gets that entry's tier,
   unless the publisher string names a *different* table entry (contradictory
   attribution: tier C).
2. A link on a syndication host (:data:`SYNDICATION_HOSTS`, e.g. finance.yahoo.com, which
   republishes Reuters and AP stories under the provider's name and sets that name
   itself) takes the tier of the publisher named, matched exactly (after normalising case
   and punctuation) against the table's aliases; an unknown publisher there is tier C.
3. A link on any other host is tier C, whatever the publisher string claims: "Reuters"
   on ``reuters-markets.biz`` is a spoof.
4. No usable http(s) link: tier C (an unverifiable attribution).

This simplifies the research's Tier B rule (two upstream-independent outlets): a single
established outlet is Tier B here. Rumor events are forced to tier C later, by the reader
(:mod:`aifirm.agents.news_reader`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

import pandas as pd

from .calendar import session_close_time

DECISION_CLOSE_TZ = "America/New_York"
DECISION_CLOSE_TIME = "16:00"


@dataclass
class Headline:
    title: str
    publisher: str
    published: str  # ISO timestamp (UTC)
    summary: str = ""
    link: str = ""  # URL of the story (Yahoo: canonical URL); its host feeds the source tier


# --------------------------------------------------------------------------- point-in-time cutoff


def decision_close_utc(as_of: date | str | pd.Timestamp, close_time: str | None = None) -> pd.Timestamp:
    """The decision instant for ``as_of``: ``close_time`` New York time on that calendar
    date, in UTC. ``close_time=None`` (the default) is the session's own close: 13:00 on a
    scheduled half day, else 16:00 (:func:`aifirm.data.calendar.session_close_time`)."""
    day = pd.Timestamp(as_of).date()
    close_time = close_time or session_close_time(day)
    return pd.Timestamp(f"{day.isoformat()} {close_time}").tz_localize(DECISION_CLOSE_TZ).tz_convert("UTC")


def published_utc(h: Headline) -> pd.Timestamp | None:
    """The headline's publish time in UTC, or None if missing or unparseable. A naive
    timestamp is taken as UTC (the feed's convention)."""
    if not h.published:
        return None
    try:
        ts = pd.Timestamp(h.published)
    except (ValueError, TypeError, OverflowError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def headlines_as_of(headlines: list[Headline], as_of: date | str | pd.Timestamp,
                    max_age_days: int | None = None,
                    close_time: str | None = None) -> tuple[list[Headline], int]:
    """Headlines readable at the close of ``as_of`` (point-in-time, see the module docstring).

    Keeps a headline only if its publish time parses and is at or before
    :func:`decision_close_utc` (``as_of``, ``close_time``) and, when ``max_age_days`` is
    given, no more than that many days before it. Returns ``(kept, n_dropped)``; ``kept`` is
    newest first, ties broken by title, so the order is deterministic whatever the feed's
    order."""
    close = decision_close_utc(as_of, close_time)
    oldest = close - pd.Timedelta(days=max_age_days) if max_age_days is not None else None
    kept: list[tuple[pd.Timestamp, Headline]] = []
    for h in headlines:
        ts = published_utc(h)
        if ts is None or ts > close or (oldest is not None and ts < oldest):
            continue
        kept.append((ts, h))
    kept.sort(key=lambda x: (-x[0].value, x[1].title, x[1].publisher))
    return [h for _, h in kept], len(headlines) - len(kept)


# --------------------------------------------------------------------------- feed


def _url(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("url")
    return value.strip() if isinstance(value, str) else ""


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
    link = _url(content.get("canonicalUrl")) or _url(content.get("clickThroughUrl")) or _url(content.get("link"))
    return Headline(title=title.strip(), publisher=publisher, published=str(ts or ""), summary=summary[:400], link=link)


def recent_headlines(ticker: str, as_of: date | str, max_items: int = 12, max_age_days: int = 14,
                     close_time: str | None = None) -> list[Headline]:
    """Headlines published at or before the New York close of ``as_of`` (the session's own
    close unless ``close_time`` is given: 13:00 on a half day, else 16:00) and at most
    ``max_age_days`` before it, newest first. Live use only (the feed cannot be replayed)."""
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    parsed = [h for h in (_parse_item(item) for item in raw if isinstance(item, dict)) if h is not None]
    kept, _ = headlines_as_of(parsed, as_of, max_age_days=max_age_days, close_time=close_time)
    return kept[:max_items]


# --------------------------------------------------------------------------- source tiers


@dataclass(frozen=True)
class NewsSource:
    """One row of the source table: canonical name, tier, publisher aliases (normalised with
    :func:`normalize_publisher`) and the registrable domains the source publishes on."""

    name: str
    tier: str
    aliases: tuple[str, ...]
    domains: tuple[str, ...]


NEWS_SOURCES: tuple[NewsSource, ...] = (
    # Tier A: primary sources
    NewsSource("SEC EDGAR", "A", ("sec", "sec edgar", "edgar", "securities and exchange commission",
                                  "us securities and exchange commission", "u s securities and exchange commission"),
               ("sec.gov",)),
    NewsSource("Business Wire", "A", ("business wire", "businesswire"), ("businesswire.com",)),
    NewsSource("PR Newswire", "A", ("pr newswire", "prnewswire", "cision pr newswire"), ("prnewswire.com",)),
    NewsSource("GlobeNewswire", "A", ("globenewswire", "globe newswire", "globenewswire inc"), ("globenewswire.com",)),
    # Tier B: established news organisations
    NewsSource("Reuters", "B", ("reuters", "thomson reuters"), ("reuters.com",)),
    NewsSource("Bloomberg", "B", ("bloomberg", "bloomberg news"), ("bloomberg.com",)),
    NewsSource("Associated Press", "B", ("associated press", "the associated press", "ap", "ap news"), ("apnews.com",)),
    NewsSource("The Wall Street Journal", "B", ("the wall street journal", "wall street journal", "wsj"), ("wsj.com",)),
    NewsSource("Financial Times", "B", ("financial times", "the financial times", "ft"), ("ft.com",)),
    NewsSource("CNBC", "B", ("cnbc",), ("cnbc.com",)),
    NewsSource("Barron's", "B", ("barrons",), ("barrons.com",)),
    NewsSource("MarketWatch", "B", ("marketwatch",), ("marketwatch.com",)),
    NewsSource("The New York Times", "B", ("the new york times", "new york times", "nyt"), ("nytimes.com",)),
)

# Hosts that republish other outlets' stories and set the provider attribution themselves.
SYNDICATION_HOSTS: tuple[str, ...] = ("finance.yahoo.com",)

TIER_DESCRIPTIONS = {
    "A": "primary source",
    "B": "established news organisation",
    "C": "unverified",
}
UNVERIFIED_SOURCE = "unverified source"


@dataclass(frozen=True)
class TierAssignment:
    tier: str            # "A", "B" or "C"
    source_name: str     # canonical name for tier A/B, UNVERIFIED_SOURCE for tier C
    link_host: str       # normalised host of the link ("" if none)
    reason: str          # why (for the ledger; never shown to analysts)


def normalize_publisher(name: str) -> str:
    """Case-, accent- and punctuation-insensitive form of a publisher name: "Barron's" ->
    "barrons", "The Wall Street Journal" -> "the wall street journal", "U.S. SEC" -> "u s sec"."""
    text = unicodedata.normalize("NFKC", name or "").casefold()
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return text.strip()


_HOST_CHARS = re.compile(r"[a-z0-9.-]+")


def link_host(link: str) -> str:
    """Lower-case host of an http(s) link without "www." or a trailing dot; "" for anything
    else (no link, another scheme, unparseable, or a host that is not a plain DNS name).

    The host must be the one a browser would load, or a story could borrow a trusted
    outlet's tier. Parsing follows the WHATWG URL rules where they differ from
    :func:`urllib.parse.urlsplit`: a backslash in an http(s) URL is a path separator, so
    ``https://evil.example\\@www.reuters.com/x`` is on ``evil.example`` (urlsplit alone reads
    ``reuters.com``) and ``https://evil.io\\.reuters.com/x`` is on ``evil.io``; userinfo is
    not the host (``https://reuters.com@evil.io/x`` -> "evil.io"). After IDNA encoding the
    host may contain only ``[a-z0-9.-]``; anything else returns "" (tier C: unverifiable)."""
    if not link or not isinstance(link, str):
        return ""
    text = link.strip().replace("\\", "/")
    try:
        parts = urlsplit(text)
        host = parts.hostname or ""
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https"):
        return ""
    host = host.strip().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii").lower() if host else ""
    except (UnicodeError, ValueError):
        return ""
    if not host or not _HOST_CHARS.fullmatch(host) or ".." in host or host.startswith((".", "-")):
        return ""
    return host[4:] if host.startswith("www.") else host


def _on_domain(host: str, domain: str) -> bool:
    domain = domain.lower().strip(".")
    return bool(host) and (host == domain or host.endswith("." + domain))


def _by_alias(publisher: str, sources: tuple[NewsSource, ...]) -> NewsSource | None:
    key = normalize_publisher(publisher)
    if not key:
        return None
    for s in sources:
        if key in s.aliases:
            return s
    return None


def _by_host(host: str, sources: tuple[NewsSource, ...]) -> NewsSource | None:
    for s in sources:
        if any(_on_domain(host, d) for d in s.domains):
            return s
    return None


def source_tier(publisher: str, link: str, issuer_domains: tuple[str, ...] | list[str] = (),
                sources: tuple[NewsSource, ...] = NEWS_SOURCES,
                syndication_hosts: tuple[str, ...] = SYNDICATION_HOSTS) -> TierAssignment:
    """Assign a headline's source tier from its publisher string and link (module docstring).

    ``issuer_domains``: the company's own web domains (e.g. ``("nvidia.com",)``); a link on
    one of them is the issuer's investor-relations release (tier A). Without it, an issuer
    site is not recognised (tier C), which errs on the side of not raising exposure."""
    host = link_host(link)
    claimed = _by_alias(publisher, sources)
    if not host:
        return TierAssignment("C", UNVERIFIED_SOURCE, "", "no usable http(s) link to confirm the source")
    if any(_on_domain(host, d) for d in issuer_domains):
        return TierAssignment("A", "company release", host, f"link on the issuer's own domain {host}")
    on_host = _by_host(host, sources)
    if on_host is not None:
        if claimed is not None and claimed is not on_host:
            return TierAssignment("C", UNVERIFIED_SOURCE, host,
                                  f"publisher claims {claimed.name} but the link is on {on_host.name}'s domain")
        return TierAssignment(on_host.tier, on_host.name, host, f"link host {host} belongs to {on_host.name}")
    if any(_on_domain(host, d) for d in syndication_hosts):
        if claimed is not None:
            return TierAssignment(claimed.tier, claimed.name, host,
                                  f"{claimed.name} story syndicated on {host} (provider set by the platform)")
        return TierAssignment("C", UNVERIFIED_SOURCE, host, f"publisher not in the source table (syndicated on {host})")
    if claimed is not None:
        return TierAssignment("C", UNVERIFIED_SOURCE, host,
                              f"publisher claims {claimed.name} but the link host {host} is not one of its domains")
    return TierAssignment("C", UNVERIFIED_SOURCE, host, f"link host {host} is not in the source table")
