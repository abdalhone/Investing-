"""Point-in-time fundamentals from SEC EDGAR XBRL "companyfacts".

Why SEC instead of a vendor snapshot: every XBRL fact carries the date it was *filed*.
Filtering on ``filed <= as_of`` gives what an investor could actually have known on the
as-of date, including the effect of later restatements being invisible until filed.
Vendor "current fundamentals" endpoints (e.g. yfinance ``.info``) are live-only and leak
the future when used in backtests.

Flow items (revenue, net income, cash flow) are rebuilt into a quarterly series
(10-Q 3-month values plus Q4 derived as FY minus 9-month YTD), and TTM = sum of the last
four quarters. Balance-sheet items are taken from the latest filed instant.

**When is a filing known?** Decisions are taken at the close (16:00 ET) of the as-of date.
SEC ``filed`` dates carry no time, and most earnings 10-Qs and 8-Ks are accepted just after
the close (EDGAR still dates a filing accepted up to 17:30 ET that day). So a filing counts
as known at the as-of close only if its EDGAR ``acceptanceDateTime`` (from the submissions
JSON, joined to XBRL facts by accession number) is before 16:00 ET on the as-of date. When
the acceptance time is unknown, a filing dated D is treated as known only after the close
of D, i.e. usable from the next day's decision.

**Which filer?** Today's ``company_tickers.json`` maps a ticker to its *current* CIK. For
historical dates that can be the wrong entity (XOM moved to a new holding-company CIK on
2026-07-01). :data:`TICKER_CIK_HISTORY` lists known predecessors, and a CIK can be pinned
explicitly (``cik=`` argument, :func:`set_cik_overrides` or the ``AIFIRM_CIK_OVERRIDES``
environment variable, e.g. ``"XOM=34088,FOO=1234"``). A filer with no facts filed by the
as-of date is reported as a data gap, never as silently empty data.

**Share counts** older than ~15 months or not positive are rejected (data gap). Foreign
private issuers (20-F/40-F filers, IFRS reporters) report *ordinary* shares, while the US
listing is often an ADR representing several of them, so they are flagged and the fact
sheet does not compute market-cap-based multiples for them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .cache import DiskCache, default_cache
from .http import sec_get_json

FLOW_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",  # banks / brokers
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "op_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
}

INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}


@dataclass
class FactValue:
    """One fundamental number with provenance."""

    value: float
    period_end: str
    filed: str
    form: str
    concept: str
    basis: str = ""  # e.g. "TTM (4 quarters)", "instant"


@dataclass
class Fundamentals:
    ticker: str
    cik: str
    entity_name: str
    as_of: str
    sic_description: str = ""
    values: dict[str, FactValue] = field(default_factory=dict)       # latest TTM / instant values
    prior_year: dict[str, FactValue] = field(default_factory=dict)   # TTM one year earlier
    shares_outstanding: FactValue | None = None
    shares_outstanding_prior_year: FactValue | None = None
    quarters: dict[str, pd.Series] = field(default_factory=dict)     # concept -> quarterly series
    foreign_filer: bool = False       # 20-F/40-F filer or IFRS reporter: share count may not match the listed ADR
    gaps: list[str] = field(default_factory=list)                    # data gaps to tell the agents


# --------------------------------------------------------------------------- filer (CIK) resolution

# Earlier SEC filers that traded under a ticker before today's filer took it over:
# ticker -> [(cik, name, until)], ``until`` = first date the *next* filer used the ticker.
TICKER_CIK_HISTORY: dict[str, list[tuple[str, str, str]]] = {
    # Holding-company reorganization: ExxonMobil Holdings Corp (CIK 2115436) became the XOM
    # issuer with its 8-K12B on 2026-07-01; the operating company filed everything before.
    "XOM": [("0000034088", "EXXON MOBIL CORP", "2026-07-01")],
}

_CIK_OVERRIDES: dict[str, str] = {}


def set_cik_overrides(mapping: dict[str, str | int] | None) -> None:
    """Pin tickers to explicit SEC CIKs (e.g. from configuration) for every date. Replaces
    any previous pins; ``None`` or ``{}`` clears them."""
    _CIK_OVERRIDES.clear()
    for t, c in (mapping or {}).items():
        _CIK_OVERRIDES[t.upper().replace(".", "-")] = str(c).strip().zfill(10)


def _overrides() -> dict[str, str]:
    out = {}
    for part in os.environ.get("AIFIRM_CIK_OVERRIDES", "").split(","):
        if "=" in part:
            t, c = part.split("=", 1)
            if t.strip() and c.strip().isdigit():
                out[t.strip().upper().replace(".", "-")] = c.strip().zfill(10)
    out.update(_CIK_OVERRIDES)
    return out


def _filer_segments(ticker: str, cik: str | None, cache: DiskCache | None) -> list[tuple[str, str, pd.Timestamp | None, pd.Timestamp | None]]:
    """[(cik, name, from, until)] filers of ``ticker`` over time (``None`` = open-ended), oldest first."""
    t = ticker.upper().replace(".", "-")
    if cik:
        return [(str(cik).strip().zfill(10), "", None, None)]
    pinned = _overrides().get(t)
    if pinned:
        return [(pinned, "", None, None)]
    segs: list[tuple[str, str, pd.Timestamp | None, pd.Timestamp | None]] = []
    lo: pd.Timestamp | None = None
    for c, name, until in sorted(TICKER_CIK_HISTORY.get(t, []), key=lambda x: x[2]):
        segs.append((c, name, lo, pd.Timestamp(until)))
        lo = pd.Timestamp(until)
    try:
        current, title = ticker_to_cik(ticker, cache)
    except KeyError:
        if not segs:   # unknown ticker and no known history
            raise
        return segs    # only historical filers (the ticker is no longer listed with the SEC)
    segs.append((current, title, lo, None))
    return segs


def resolve_cik(ticker: str, as_of: date | str, cache: DiskCache | None = None,
                cik: str | None = None) -> tuple[str, str]:
    """The CIK (and name, if known) of the filer that used ``ticker`` on ``as_of``."""
    ts = pd.Timestamp(as_of)
    for c, name, lo, until in _filer_segments(ticker, cik, cache):
        if (lo is None or ts >= lo) and (until is None or ts < until):
            return c, name
    raise KeyError(f"No SEC filer known for {ticker} on {ts.date()}")


# --------------------------------------------------------------------------- helpers

def ticker_to_cik(ticker: str, cache: DiskCache | None = None) -> tuple[str, str]:
    """Today's CIK for ``ticker`` (from SEC's current ticker table; see :func:`resolve_cik`)."""
    cache = cache or default_cache()
    table = cache.get_or_set(
        "sec", "company_tickers", lambda: sec_get_json("https://www.sec.gov/files/company_tickers.json"),
        ttl_seconds=7 * 86400, kind="json",
    )
    t = ticker.upper().replace(".", "-")
    for row in table.values():
        if row["ticker"].upper() == t:
            return str(row["cik_str"]).zfill(10), row["title"]
    raise KeyError(f"Ticker {ticker} not found in SEC company_tickers.json (ETFs and non-US issuers are not covered)")


def company_facts(cik: str, cache: DiskCache | None = None) -> dict:
    cache = cache or default_cache()
    return cache.get_or_set(
        "sec", f"companyfacts:{cik}",
        lambda: sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"),
        ttl_seconds=86400, kind="json",
    )


def submissions(cik: str, cache: DiskCache | None = None) -> dict:
    cache = cache or default_cache()
    return cache.get_or_set(
        "sec", f"submissions:{cik}",
        lambda: sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json"),
        ttl_seconds=6 * 3600, kind="json",
    )


def _submissions_page(page: dict, cache: DiskCache | None = None) -> dict:
    """One older page of a filer's submission history (``filings.files[]``). The cache key
    includes the page's range and count, so a re-paginated history is fetched again."""
    cache = cache or default_cache()
    name = page["name"]
    key = f"submissions-page:{name}:{page.get('filingFrom')}:{page.get('filingTo')}:{page.get('filingCount')}"
    return cache.get_or_set("sec", key, lambda: sec_get_json(f"https://data.sec.gov/submissions/{name}"),
                            ttl_seconds=None, kind="json")


DECISION_CUTOFF = pd.Timedelta(hours=16)   # decisions are taken at the 16:00 ET close


def decision_cutoff(as_of: date | str | pd.Timestamp) -> pd.Timestamp:
    """The moment (naive US/Eastern) at which a decision dated ``as_of`` is taken."""
    return pd.Timestamp(as_of).normalize() + DECISION_CUTOFF


def _accepted_et(values: list) -> pd.Series:
    """EDGAR acceptanceDateTime strings (UTC) -> naive US/Eastern timestamps (NaT if absent)."""
    ts = pd.to_datetime(pd.Series(values, dtype="object"), utc=True, errors="coerce", format="ISO8601")
    return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)


def _block_frame(block: dict) -> pd.DataFrame:
    n = len(block.get("form", []))

    def col(name: str) -> list:
        v = block.get(name) or []
        return list(v) + [""] * (n - len(v))

    df = pd.DataFrame({
        "accession": col("accessionNumber"),
        "filed": pd.to_datetime(pd.Series(col("filingDate"), dtype="object"), errors="coerce"),
        "form": col("form"),
        "items": col("items"),
        "description": col("primaryDocDescription"),
    })
    df["accepted"] = _accepted_et(col("acceptanceDateTime")).to_numpy() if n else pd.Series(dtype="datetime64[ns]")
    return df.dropna(subset=["filed"])


def known_at(filed: pd.Series, accepted: pd.Series | None = None) -> pd.Series:
    """When each filing became public: its acceptance time if known, else the end of the
    filing day (so a filing dated D is first usable for the decision of D + 1)."""
    fallback = pd.to_datetime(filed) + pd.Timedelta(days=1)
    if accepted is None:
        return fallback
    return pd.to_datetime(accepted).where(pd.notna(accepted), fallback)


def filing_index(cik: str, start: date | str | pd.Timestamp, end: date | str | pd.Timestamp,
                 cache: DiskCache | None = None, sub: dict | None = None) -> pd.DataFrame:
    """Every filing of ``cik`` dated in [start, end]: the submissions 'recent' block plus any
    older pages (``filings.files``) whose date range overlaps the window.

    Columns: accession, filed, form, items, description, accepted (naive ET, may be NaT),
    known_at. Raises if a page the window needs cannot be loaded, so a caller never
    mistakes a fetch failure for "no filings"."""
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    sub = sub if sub is not None else submissions(cik, cache)
    filings = sub.get("filings", {})
    frames = [_block_frame(filings.get("recent", {}))]
    pad = pd.Timedelta(days=3)   # page ranges in the index can be off by a day or two
    for page in filings.get("files", []) or []:
        p_from, p_to = pd.Timestamp(page.get("filingFrom")), pd.Timestamp(page.get("filingTo"))
        if p_to + pad < lo or p_from - pad > hi:
            continue
        frames.append(_block_frame(_submissions_page(page, cache)))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["filed"] >= lo) & (df["filed"] <= hi)]
    df = df.drop_duplicates(subset=["accession"], keep="first").sort_values("filed", ascending=False)
    df["known_at"] = known_at(df["filed"], df["accepted"])
    return df.reset_index(drop=True)


def _facts_frame(facts: dict, taxonomy: str, concept: str, unit_pref: tuple[str, ...] = ("USD",),
                 accepted: dict[str, pd.Timestamp] | None = None) -> pd.DataFrame | None:
    node = facts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return None
    units = node.get("units", {})
    rows = None
    for u in unit_pref:
        if u in units:
            rows = units[u]
            break
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    if df.empty or "filed" not in df:
        return None
    df["concept"] = concept
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"])
    acc = None
    if accepted and "accn" in df:
        acc = pd.to_datetime(df["accn"].map(accepted))
    df["known_at"] = known_at(df["filed"], acc)
    return df


def _known(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows public before the as-of decision cutoff (16:00 ET; see module docstring)."""
    known = df["known_at"] if "known_at" in df.columns else known_at(df["filed"])
    return df[known < decision_cutoff(as_of)]


def _pit(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Keep facts known at the as-of close; for each period keep the latest filing known then."""
    df = _known(df, as_of)
    if df.empty:
        return df
    keys = ["start", "end"] if "start" in df.columns else ["end"]
    df = df.sort_values("filed").drop_duplicates(subset=keys, keep="last")
    return df


STALE_AFTER_DAYS = 550  # ignore tags a company stopped using long ago


def _choose_concept(facts: dict, candidates: list[str], as_of: pd.Timestamp, unit_pref=("USD",),
                    quarterly: bool = False, accepted: dict[str, pd.Timestamp] | None = None) -> pd.DataFrame | None:
    """Companies switch XBRL tags over time; use the candidate with the most recent period.

    With ``quarterly=True`` recency is judged on rebuilt quarterly periods, so a tag that
    only appears as a comparative in recent 10-Ks does not win over the live tag.
    """
    best, best_end = None, None
    for concept in candidates:
        df = _facts_frame(facts, "us-gaap", concept, unit_pref, accepted)
        if df is None:
            continue
        df = _pit(df, as_of)
        if df.empty:
            continue
        if quarterly:
            q = quarterly_series(df)
            if q.empty:
                continue
            end = q.index.max()
        else:
            end = df["end"].max()
        if end < as_of - pd.Timedelta(days=STALE_AFTER_DAYS):
            continue
        if best_end is None or end > best_end:
            best, best_end = df, end
    return best


def quarterly_series(df: pd.DataFrame) -> pd.Series:
    """Rebuild 3-month values (indexed by period end) from 10-Q/10-K duration facts."""
    if df is None or df.empty or "start" not in df.columns:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["start"]).copy()
    df["days"] = (df["end"] - df["start"]).dt.days
    q = {row.end: float(row.val) for row in df[(df.days >= 80) & (df.days <= 100)].itertuples()}
    ytd = df[(df.days > 100)]
    # derive missing quarters from YTD differences (most importantly Q4 = FY - 9M)
    for row in ytd.itertuples():
        if row.end in q:
            continue
        shorter = ytd[(abs((ytd.start - row.start).dt.days) <= 3) & (ytd.end < row.end)]
        shorter = shorter[(row.days - shorter.days).between(80, 100)]
        if not shorter.empty:
            s = shorter.sort_values("end").iloc[-1]
            q[row.end] = float(row.val) - float(s.val)
            continue
        if 80 <= row.days - 91 <= 100:  # 6M YTD with only a 3M first quarter
            first = df[(abs((df.start - row.start).dt.days) <= 3) & df.days.between(80, 100)]
            if not first.empty:
                q[row.end] = float(row.val) - float(first.sort_values("end").iloc[0].val)
    s = pd.Series(q, dtype=float).sort_index()
    return s


def ttm_from_quarters(q: pd.Series, offset: int = 0) -> tuple[float, pd.Timestamp] | None:
    """Sum of 4 consecutive quarters ending ``offset`` quarters before the latest one."""
    if len(q) < 4 + offset:
        return None
    window = q.iloc[len(q) - 4 - offset: len(q) - offset]
    ends = list(window.index)
    gaps = [(b - a).days for a, b in zip(ends, ends[1:])]
    if not all(75 <= g <= 105 for g in gaps):
        return None
    return float(window.sum()), ends[-1]


def _latest_annual(df: pd.DataFrame) -> FactValue | None:
    if df is None or df.empty or "start" not in df.columns:
        return None
    d = df.dropna(subset=["start"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    a = d[d.days.between(350, 380)].sort_values("end")
    if a.empty:
        return None
    r = a.iloc[-1]
    return FactValue(float(r.val), str(r.end.date()), str(r.filed.date()), str(r.get("form", "")), r.concept, "fiscal year")


# --------------------------------------------------------------------------- share counts and filer type

SHARES_STALE_AFTER_DAYS = 460   # ~15 months: an older share count is not used for valuation
SHARE_CONCEPTS = (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"))
FOREIGN_FORMS = frozenset({"20-F", "20-F/A", "40-F", "40-F/A", "6-K", "6-K/A"})
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})


def _share_value(rows: pd.DataFrame, end: pd.Timestamp) -> FactValue:
    """The count reported for ``end`` by the latest filing known (summing the rows of that
    one filing, in case it reports several classes without dimensions)."""
    same = rows[rows["end"] == end].sort_values("filed")
    if "accn" in same.columns:
        same = same[same["accn"] == same["accn"].iloc[-1]]
    last = same.iloc[-1]
    return FactValue(float(same["val"].sum()), str(end.date()), str(last.filed.date()), str(last.get("form", "")),
                     last.concept, "instant")


def _shares(facts: dict, as_of: pd.Timestamp, accepted: dict[str, pd.Timestamp] | None,
            gaps: list[str]) -> tuple[FactValue | None, FactValue | None]:
    """(latest share count, count ~1 year before it) known at the as-of close.

    Counts that are not positive are ignored; the freshest candidate concept wins; a latest
    count older than :data:`SHARES_STALE_AFTER_DAYS` is rejected with a data gap."""
    best: tuple[pd.Timestamp, pd.DataFrame] | None = None
    stale_end: pd.Timestamp | None = None
    for taxonomy, concept in SHARE_CONCEPTS:
        df = _facts_frame(facts, taxonomy, concept, ("shares",), accepted)
        if df is None:
            continue
        df = _known(df, as_of)
        df = df[pd.to_numeric(df["val"], errors="coerce") > 0]
        if df.empty:
            continue
        end = df["end"].max()
        if end < as_of - pd.Timedelta(days=SHARES_STALE_AFTER_DAYS):
            stale_end = end if stale_end is None or end > stale_end else stale_end
            continue
        if best is None or end > best[0]:
            best = (end, df)
    if best is None:
        if stale_end is not None:
            gaps.append(f"Latest SEC share count is dated {stale_end.date()} (more than 15 months before "
                        f"{as_of.date()}) - shares outstanding, market cap and valuation multiples not computed")
        return None, None
    end, df = best
    current = _share_value(df, end)
    prior_rows = df[df["end"] <= end - pd.Timedelta(days=330)]
    prior = _share_value(prior_rows, prior_rows["end"].max()) if not prior_rows.empty else None
    return current, prior


def _is_foreign_filer(facts: dict, as_of: pd.Timestamp, sub: dict, shares: FactValue | None) -> bool:
    """True for a foreign private issuer as of ``as_of``: its latest annual report known then
    is a 20-F/40-F, or (when the filing history does not reach back that far) its share
    count comes from a 20-F/40-F/6-K, or it reports in IFRS."""
    try:
        recent = _block_frame(sub.get("filings", {}).get("recent", {}))
        recent = recent[known_at(recent["filed"], recent["accepted"]) < decision_cutoff(as_of)]
        annual = recent[recent["form"].isin(_ANNUAL_FORMS)].sort_values("filed")
        if not annual.empty:
            return annual["form"].iloc[-1] in FOREIGN_FORMS
    except Exception:  # noqa: BLE001, S110 - malformed metadata: fall through to the XBRL-based signals
        pass
    if shares is not None and shares.form in FOREIGN_FORMS:
        return True
    return bool(facts.get("facts", {}).get("ifrs-full"))


def _acceptance_map(cik: str, as_of: pd.Timestamp, sub: dict, cache: DiskCache | None) -> dict[str, pd.Timestamp]:
    """accession number -> acceptance time (naive ET) for filings around ``as_of``. Only
    filings dated on the as-of day (or just before, for after-hours acceptances) need it;
    without it the conservative date rule applies, so a failure here is not an error."""
    try:
        idx = filing_index(cik, as_of - pd.Timedelta(days=4), as_of, cache, sub=sub)
    except Exception:  # noqa: BLE001
        try:
            idx = _block_frame(sub.get("filings", {}).get("recent", {}))
        except Exception:  # noqa: BLE001
            return {}
    idx = idx.dropna(subset=["accepted"])
    return dict(zip(idx["accession"], idx["accepted"]))


_PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-QT", "20-F", "20-F/A", "40-F", "40-F/A",
                             "6-K", "6-K/A"})


def _has_periodic_facts(facts: dict, as_of: pd.Timestamp) -> bool:
    """True if the filer had filed XBRL from a periodic report (10-K/10-Q/20-F/40-F/6-K)
    before the as-of day. Fee exhibits on registration statements do not count."""
    for taxonomy in facts.get("facts", {}).values():
        for node in taxonomy.values():
            for rows in node.get("units", {}).values():
                for r in rows:
                    filed = r.get("filed")
                    if filed and r.get("form") in _PERIODIC_FORMS and pd.Timestamp(filed) < as_of:
                        return True
    return False


# --------------------------------------------------------------------------- main API

def load_fundamentals(ticker: str, as_of: date | str, cache: DiskCache | None = None,
                      cik: str | None = None) -> Fundamentals:
    """Point-in-time fundamentals for ``ticker`` as known at the close of ``as_of``.

    ``cik`` pins the SEC filer; otherwise the filer that used the ticker on ``as_of`` is
    used (:func:`resolve_cik`), falling back to a predecessor filer when the current one had
    filed nothing yet. Missing or unusable data is listed in ``Fundamentals.gaps``."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    segments = _filer_segments(ticker, cik, cache)
    # the filer in use on as_of first, then its predecessors (most recent first)
    active = [s_ for s_ in segments if (s_[2] is None or as_of_ts >= s_[2]) and (s_[3] is None or as_of_ts < s_[3])]
    earlier = [s_ for s_ in reversed(segments) if s_[3] is not None and as_of_ts >= s_[3]]
    candidates = active[:1] + earlier
    if not candidates:
        raise KeyError(f"No SEC filer known for {ticker} on {as_of_ts.date()}")
    gaps: list[str] = []
    chosen = None
    for i, (c, title, _lo, _until) in enumerate(candidates):
        facts = company_facts(c, cache)
        if _has_periodic_facts(facts, as_of_ts):
            chosen = (c, title, facts)
            if i > 0:
                gaps.append(f"SEC data for {ticker.upper()} on {as_of_ts.date()} comes from predecessor filer "
                            f"{facts.get('entityName', title) or title} (CIK {c})")
            break
    if chosen is None:   # nobody had filed yet: say so instead of returning silently empty data
        c, title, _, _ = candidates[0]
        facts = company_facts(c, cache)
        chosen = (c, title, facts)
        gaps.append(f"SEC filer {facts.get('entityName', title) or title} (CIK {c}) had filed no periodic-report XBRL data before "
                    f"{as_of_ts.date()}; the ticker may have belonged to another filer then "
                    "(pin the historical CIK with a CIK override)")
    cik_, title, facts = chosen
    sub: dict = {}
    try:
        sub = submissions(cik_, cache)
    except Exception:  # noqa: BLE001, S110 - only metadata; acceptance times fall back to the date rule
        pass
    accepted = _acceptance_map(cik_, as_of_ts, sub, cache) if sub else {}
    out = Fundamentals(ticker=ticker.upper(), cik=cik_, entity_name=facts.get("entityName", title),
                       as_of=str(as_of_ts.date()), sic_description=sub.get("sicDescription", "") or "")
    out.gaps = gaps

    for name, candidates_ in FLOW_CONCEPTS.items():
        df = _choose_concept(facts, candidates_, as_of_ts, quarterly=True, accepted=accepted)
        if df is None:
            df = _choose_concept(facts, candidates_, as_of_ts, accepted=accepted)
        if df is None:
            continue
        q = quarterly_series(df)
        out.quarters[name] = q
        cur = ttm_from_quarters(q)
        concept = df["concept"].iloc[0]
        filed_latest = df[df["end"] == df["end"].max()]["filed"].max()
        if cur:
            out.values[name] = FactValue(cur[0], str(cur[1].date()), str(filed_latest.date()), "10-Q/10-K", concept, "TTM (4 quarters)")
            prev = ttm_from_quarters(q, offset=4)
            if prev:
                out.prior_year[name] = FactValue(prev[0], str(prev[1].date()), "", "10-Q/10-K", concept, "TTM (4 quarters)")
        else:
            fy = _latest_annual(df)
            if fy:
                out.values[name] = fy

    for name, candidates_ in INSTANT_CONCEPTS.items():
        df = _choose_concept(facts, candidates_, as_of_ts, accepted=accepted)
        if df is None or df.empty:
            continue
        r = df.sort_values(["end", "filed"]).iloc[-1]
        out.values[name] = FactValue(float(r.val), str(r.end.date()), str(r.filed.date()), str(r.get("form", "")), r.concept, "instant")

    out.shares_outstanding, out.shares_outstanding_prior_year = _shares(facts, as_of_ts, accepted, out.gaps)
    out.foreign_filer = _is_foreign_filer(facts, as_of_ts, sub, out.shares_outstanding)
    return out


def recent_filings(ticker: str, as_of: date | str, lookback_days: int = 120,
                   cache: DiskCache | None = None, cik: str | None = None) -> list[dict]:
    """SEC filings (8-K items, 10-Q/10-K, Form 4 counts) in the lookback window that were
    public by the as-of close (acceptance before 16:00 ET, or dated before the as-of day).

    Older submission pages are fetched when the window needs them, and every filer that
    used the ticker during the window is included. Raises (instead of returning ``[]``)
    when the history cannot be loaded, so the caller records a data gap."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    start = as_of_ts - pd.Timedelta(days=lookback_days)
    cutoff = decision_cutoff(as_of_ts)
    frames = []
    for c, _name, lo, until in _filer_segments(ticker, cik, cache):
        if (until is not None and until <= start) or (lo is not None and lo > as_of_ts):
            continue
        df = filing_index(c, start, as_of_ts, cache)
        if until is not None:
            df = df[df["filed"] < until]
        frames.append(df)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["accession"])
    df = df[df["known_at"] < cutoff].sort_values("filed", ascending=False)
    return [{
        "form": r.form,
        "filed": str(r.filed.date()),
        "accepted": "" if pd.isna(r.accepted) else r.accepted.isoformat(),
        "items": r["items"] or "",
        "description": r.description or "",
    } for _, r in df.iterrows()]


EIGHT_K_ITEMS = {
    "1.01": "entry into a material agreement", "1.02": "termination of a material agreement",
    "1.03": "bankruptcy or receivership", "2.01": "completion of acquisition or disposition",
    "2.02": "results of operations (earnings release)", "2.03": "new material financial obligation",
    "2.05": "exit or restructuring costs", "2.06": "material impairment",
    "3.01": "delisting notice / listing rule failure", "3.02": "unregistered equity sale",
    "4.01": "change of auditor", "4.02": "non-reliance on prior financials (restatement)",
    "5.02": "director/officer departure or appointment", "5.07": "shareholder vote results",
    "7.01": "Reg FD disclosure", "8.01": "other events", "9.01": "financial statements and exhibits",
}
