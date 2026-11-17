"""Point-in-time fundamentals from SEC EDGAR XBRL "companyfacts".

Why SEC instead of a vendor snapshot: every XBRL fact carries the date it was *filed*.
Filtering on ``filed <= as_of`` gives what an investor could actually have known on the
as-of date, including the effect of later restatements being invisible until filed.
Vendor "current fundamentals" endpoints (e.g. yfinance ``.info``) are live-only and leak
the future when used in backtests.

Flow items (revenue, net income, cash flow) are rebuilt into a quarterly series
(10-Q 3-month values plus Q4 derived as FY minus 9-month YTD), and TTM = sum of the last
four quarters. Balance-sheet items are taken from the latest filed instant.
"""

from __future__ import annotations

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


# --------------------------------------------------------------------------- helpers

def ticker_to_cik(ticker: str, cache: DiskCache | None = None) -> tuple[str, str]:
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


def _facts_frame(facts: dict, taxonomy: str, concept: str, unit_pref: tuple[str, ...] = ("USD",)) -> pd.DataFrame | None:
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
    return df


def _pit(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Keep facts filed on/before as_of; for each period keep the latest filing known then."""
    df = df[df["filed"] <= as_of]
    if df.empty:
        return df
    keys = ["start", "end"] if "start" in df.columns else ["end"]
    df = df.sort_values("filed").drop_duplicates(subset=keys, keep="last")
    return df


STALE_AFTER_DAYS = 550  # ignore tags a company stopped using long ago


def _choose_concept(facts: dict, candidates: list[str], as_of: pd.Timestamp, unit_pref=("USD",),
                    quarterly: bool = False) -> pd.DataFrame | None:
    """Companies switch XBRL tags over time; use the candidate with the most recent period.

    With ``quarterly=True`` recency is judged on rebuilt quarterly periods, so a tag that
    only appears as a comparative in recent 10-Ks does not win over the live tag.
    """
    best, best_end = None, None
    for concept in candidates:
        df = _facts_frame(facts, "us-gaap", concept, unit_pref)
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


# --------------------------------------------------------------------------- main API

def load_fundamentals(ticker: str, as_of: date | str, cache: DiskCache | None = None) -> Fundamentals:
    as_of_ts = pd.Timestamp(as_of)
    cik, title = ticker_to_cik(ticker, cache)
    facts = company_facts(cik, cache)
    sub = {}
    try:
        sub = submissions(cik, cache)
    except Exception:
        pass
    out = Fundamentals(ticker=ticker.upper(), cik=cik, entity_name=facts.get("entityName", title),
                       as_of=str(as_of_ts.date()), sic_description=sub.get("sicDescription", "") or "")

    for name, candidates in FLOW_CONCEPTS.items():
        df = _choose_concept(facts, candidates, as_of_ts, quarterly=True)
        if df is None:
            df = _choose_concept(facts, candidates, as_of_ts)
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

    for name, candidates in INSTANT_CONCEPTS.items():
        df = _choose_concept(facts, candidates, as_of_ts)
        if df is None or df.empty:
            continue
        r = df.sort_values(["end", "filed"]).iloc[-1]
        out.values[name] = FactValue(float(r.val), str(r.end.date()), str(r.filed.date()), str(r.get("form", "")), r.concept, "instant")

    sh = _facts_frame(facts, "dei", "EntityCommonStockSharesOutstanding", ("shares",))
    if sh is None:
        sh = _facts_frame(facts, "us-gaap", "CommonStockSharesOutstanding", ("shares",))
    if sh is not None:
        sh = _pit(sh, as_of_ts)
        if not sh.empty:
            # dual-class issuers report one row per class for the same date: sum them
            latest_end = sh["end"].max()
            same = sh[sh["end"] == latest_end]
            out.shares_outstanding = FactValue(float(same["val"].sum()), str(latest_end.date()),
                                               str(same["filed"].max().date()), str(same["form"].iloc[-1]),
                                               same["concept"].iloc[0], "instant")
            prior = sh[sh["end"] <= latest_end - pd.Timedelta(days=330)]
            if not prior.empty:
                pend = prior["end"].max()
                p = prior[prior["end"] == pend]
                out.shares_outstanding_prior_year = FactValue(float(p["val"].sum()), str(pend.date()),
                                                              str(p["filed"].max().date()), str(p["form"].iloc[-1]),
                                                              p["concept"].iloc[0], "instant")
    return out


def recent_filings(ticker: str, as_of: date | str, lookback_days: int = 120,
                   cache: DiskCache | None = None) -> list[dict]:
    """SEC filings (8-K items, 10-Q/10-K, Form 4 counts) filed in the lookback window. Point-in-time safe."""
    as_of_ts = pd.Timestamp(as_of)
    cik, _ = ticker_to_cik(ticker, cache)
    sub = submissions(cik, cache)
    r = sub.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(r.get("form", [])):
        filed = pd.Timestamp(r["filingDate"][i])
        if filed > as_of_ts or filed < as_of_ts - pd.Timedelta(days=lookback_days):
            continue
        out.append({
            "form": form,
            "filed": str(filed.date()),
            "items": r.get("items", [""] * (i + 1))[i],
            "description": r.get("primaryDocDescription", [""] * (i + 1))[i],
        })
    return out


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
