"""Regression tests for the data adapters (Yahoo prices, SEC EDGAR, FRED) with fake transports.

No network: ``yfinance.download`` and the SEC/FRED fetchers are replaced by in-memory fakes,
so each test pins one point-in-time or robustness rule of the Data Desk.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from aifirm.data import fundamentals as fmod
from aifirm.data import macro as mmod
from aifirm.data import prices as pmod
from aifirm.data.cache import DiskCache
from aifirm.data.factsheet import build_factsheet
from aifirm.data.fundamentals import (
    FactValue,
    Fundamentals,
    load_fundamentals,
    recent_filings,
)

# =========================================================================== Yahoo fake

FAKE_TODAY = "2024-12-31"
SPLIT_EX = pd.Timestamp("2024-06-10")   # AAA does a 10-for-1 split on this ex-date


def _yahoo_history() -> dict[str, pd.DataFrame]:
    """Synthetic Yahoo history as served 'today': AAA traded at 1000 before its split and
    100 after it, so Yahoo's split-adjusted Close is ~100 throughout."""
    idx = pd.bdate_range("2022-01-03", FAKE_TODAY)
    rng = np.random.default_rng(3)
    traded = pd.Series(1000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    traded[idx >= SPLIT_EX] = traded[idx >= SPLIT_EX] / 10.0       # what printed on the tape
    adjusted = traded.where(idx >= SPLIT_EX, traded / 10.0)        # Yahoo back-adjusts history
    spy = pd.Series(400.0 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    msft = pd.Series(300.0, index=idx)
    close = pd.DataFrame({"AAA": adjusted, "SPY": spy, "MSFT": msft})
    splits = pd.DataFrame(0.0, index=idx, columns=close.columns)
    splits.loc[SPLIT_EX, "AAA"] = 10.0
    return {"Open": close.copy(), "High": close.copy(), "Low": close.copy(), "Close": close,
            "Adj Close": close.copy(), "Volume": pd.DataFrame(1e6, index=idx, columns=close.columns),
            "Dividends": pd.DataFrame(0.0, index=idx, columns=close.columns), "Stock Splits": splits}


class FakeYahoo:
    def __init__(self) -> None:
        self.hist = _yahoo_history()
        self.throttled: set[str] = set()   # tickers that come back all-NaN (HTTP 429)
        self.calls: list[list[str]] = []

    def download(self, tickers, start=None, end=None, **kw):
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self.calls.append(tickers)
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) - pd.Timedelta(days=1) if end is not None else pd.Timestamp(FAKE_TODAY)
        fields = {}
        for field, frame in self.hist.items():
            f = frame.loc[lo:hi].reindex(columns=tickers).copy()
            for t in tickers:
                if t in self.throttled:
                    f[t] = np.nan
            fields[field] = f
        return pd.concat(fields, axis=1, names=["Price", "Ticker"])


@pytest.fixture
def yahoo(monkeypatch):
    import yfinance

    fake = FakeYahoo()
    monkeypatch.setattr(yfinance, "download", fake.download)
    return fake


def _shares_fundamentals(ticker: str, as_of: str, shares: float, shares_date: str,
                         prior: float | None = None, prior_date: str | None = None, **kw) -> Fundamentals:
    f = Fundamentals(ticker=ticker, cik="0", entity_name=f"{ticker} Corp", as_of=as_of, **kw)
    f.values = {"net_income": FactValue(5e7, "2024-03-31", "2024-04-25", "10-Q", "NetIncomeLoss", "TTM (4 quarters)")}
    f.shares_outstanding = FactValue(shares, shares_date, shares_date, "10-Q", "dei", "instant")
    if prior is not None:
        f.shares_outstanding_prior_year = FactValue(prior, prior_date, prior_date, "10-Q", "dei", "instant")
    return f


# --------------------------------------------------------------------------- split-adjusted levels


def test_price_level_and_market_cap_use_as_traded_close_before_a_later_split(yahoo, tmp_path):
    """Yahoo's Close for a date before a later split is back-adjusted. The price fact and
    market cap must use the level that printed that day, consistent with the as-filed
    (pre-split) SEC share count."""
    p = pmod.load_prices(["AAA", "SPY"], "2023-06-01", "2024-05-31", cache=DiskCache(tmp_path))
    day = pd.Timestamp("2024-05-31")
    f = _shares_fundamentals("AAA", "2024-05-31", 1e6, "2024-04-20")
    fs = build_factsheet("AAA", "2024-05-31", p, fundamentals=f)
    price = fs.get("AAA.price").value
    assert price > 900                                             # ~1000 as traded, not ~100
    assert fs.get("AAA.market_cap").value == pytest.approx(1e6 * price)
    assert price == pytest.approx(p.raw_close.loc[day, "AAA"])
    assert p.close.loc[day, "AAA"] == pytest.approx(price / 10.0)  # fills keep the adjusted basis
    assert p.as_of(day).split_ratio("AAA", "2024-01-01") == 1.0    # the split is still in the future
    assert fs.get("AAA.pe_ttm").value == pytest.approx(1e6 * price / 5e7)


def test_share_count_dated_before_a_past_split_is_put_on_the_as_of_basis(yahoo, tmp_path):
    p = pmod.load_prices(["AAA", "SPY"], "2023-06-01", "2024-09-30", cache=DiskCache(tmp_path))
    f = _shares_fundamentals("AAA", "2024-06-12", 1e6, "2024-04-20", prior=1e6, prior_date="2023-04-20")
    fs = build_factsheet("AAA", "2024-06-12", p, fundamentals=f)
    price = fs.get("AAA.price").value
    assert 90 < price < 110                                        # post-split level
    assert fs.get("AAA.shares_out").value == pytest.approx(1e7)    # 1e6 pre-split shares x 10
    assert fs.get("AAA.market_cap").value == pytest.approx(1e7 * price)
    assert fs.get("AAA.share_count_change_yoy").value == pytest.approx(0.0)   # not +900% "dilution"


# --------------------------------------------------------------------------- partial downloads


def test_partial_yahoo_download_is_not_cached_permanently(yahoo, tmp_path):
    cache = DiskCache(tmp_path)
    yahoo.throttled = {"MSFT"}
    p1 = pmod.load_prices(["MSFT", "SPY"], "2023-01-03", "2023-12-29", cache=cache)
    assert p1.close["MSFT"].dropna().empty and not p1.close["SPY"].dropna().empty
    yahoo.throttled = set()                                         # Yahoo is healthy again ...
    for f in (tmp_path / "prices").iterdir():                       # ... later than the short TTL
        old = time.time() - 2 * 3600                               # (the incomplete-result TTL is shorter)
        os.utime(f, (old, old))
    p2 = pmod.load_prices(["MSFT", "SPY"], "2023-01-03", "2023-12-29", cache=cache)
    assert len(p2.close["MSFT"].dropna()) > 200


def test_throttled_ticker_is_retried_alone_and_complete_result_cached(yahoo, tmp_path, monkeypatch):
    cache = DiskCache(tmp_path)
    real = yahoo.download

    def flaky(tickers, *a, **kw):   # MSFT fails inside the batch but succeeds on its own
        yahoo.throttled = {"MSFT"} if len(tickers) > 1 else set()
        return real(tickers, *a, **kw)

    import yfinance
    monkeypatch.setattr(yfinance, "download", flaky)
    p = pmod.load_prices(["MSFT", "SPY"], "2023-01-03", "2023-12-29", cache=cache)
    assert len(p.close["MSFT"].dropna()) > 200
    n = len(yahoo.calls)
    pmod.load_prices(["MSFT", "SPY"], "2023-01-03", "2023-12-29", cache=cache)
    assert len(yahoo.calls) == n                                    # complete -> served from cache


# =========================================================================== SEC fake


def _rows(arrays: list[tuple]) -> dict:
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocDescription"]
    return {k: [a[i] for a in arrays] for i, k in enumerate(keys)}


def _quarter(start: str, end: str, val: float, accn: str, filed: str, form: str = "10-Q") -> dict:
    return {"start": start, "end": end, "val": val, "accn": accn, "form": form, "filed": filed}


class FakeSEC:
    """Serves company_tickers, companyfacts, submissions and older submission pages."""

    def __init__(self) -> None:
        self.tickers: dict = {}
        self.facts: dict[str, dict] = {}
        self.subs: dict[str, dict] = {}
        self.pages: dict[str, dict] = {}
        self.fail_pages = False
        self.urls: list[str] = []

    def get_json(self, url: str) -> dict:
        self.urls.append(url)
        if url.endswith("company_tickers.json"):
            return self.tickers
        if "/companyfacts/CIK" in url:
            return self.facts[url.rsplit("CIK", 1)[1].removesuffix(".json")]
        name = url.rsplit("/", 1)[1]
        if name in self.pages:
            if self.fail_pages:
                raise RuntimeError("503 from SEC")
            return self.pages[name]
        return self.subs[name.removeprefix("CIK").removesuffix(".json")]


AAA_CIK = "0000001234"


def _aaa_sec() -> FakeSEC:
    sec = FakeSEC()
    sec.tickers = {"0": {"cik_str": 1234, "ticker": "AAA", "title": "AAA Corp"}}
    revenue = [
        _quarter("2023-01-01", "2023-03-31", 100.0, "a-23q1", "2023-04-27"),
        _quarter("2023-04-01", "2023-06-30", 100.0, "a-23q2", "2023-07-27"),
        _quarter("2023-07-01", "2023-09-30", 100.0, "a-23q3", "2023-10-26"),
        _quarter("2023-10-01", "2023-12-31", 100.0, "a-23q4", "2024-01-30"),
        _quarter("2024-01-01", "2024-03-31", 200.0, "a-24q1", "2024-04-25"),   # accepted after the close
    ]
    shares = [{"end": "2024-01-25", "val": 1000.0, "accn": "a-23q4", "form": "10-Q", "filed": "2024-01-30"},
              {"end": "2024-04-22", "val": 1100.0, "accn": "a-24q1", "form": "10-Q", "filed": "2024-04-25"}]
    sec.facts[AAA_CIK] = {"entityName": "AAA Corp", "facts": {
        "us-gaap": {"Revenues": {"units": {"USD": revenue}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}},
    }}
    recent = _rows([
        ("a-24q1", "2024-04-25", "2024-04-25T20:06:24.000Z", "10-Q", "", "10-Q"),            # 16:06 ET
        ("a-8k-pm", "2024-04-25", "2024-04-25T20:03:19.000Z", "8-K", "2.02,9.01", "8-K"),    # 16:03 ET
        ("a-8k-am", "2024-04-25", "2024-04-25T12:30:00.000Z", "8-K", "5.02", "8-K"),         # 08:30 ET
        ("a-8k-nt", "2024-04-24", "", "8-K", "8.01", "8-K"),                                 # time unknown
        ("a-23q4", "2024-01-30", "2024-01-30T21:05:46.000Z", "10-Q", "", "10-Q"),
        ("a-10k", "2024-01-30", "2024-01-30T21:06:00.000Z", "10-K", "", "10-K"),
    ])
    sec.subs[AAA_CIK] = {"cik": AAA_CIK, "name": "AAA Corp", "sicDescription": "Software",
                         "filings": {"recent": recent, "files": []}}
    return sec


@pytest.fixture
def sec(monkeypatch):
    fake = _aaa_sec()
    monkeypatch.setattr(fmod, "sec_get_json", fake.get_json)
    monkeypatch.delenv("AIFIRM_CIK_OVERRIDES", raising=False)
    fmod.set_cik_overrides(None)
    yield fake
    fmod.set_cik_overrides(None)


NOCACHE = DiskCache(enabled=False)


def test_filing_accepted_after_the_close_is_not_known_that_day(sec):
    """A 10-Q accepted at 16:06 ET on D is public only after the D close: decisions dated D
    must not see it; decisions dated D+1 do."""
    before = load_fundamentals("AAA", "2024-04-25", cache=NOCACHE)
    assert before.values["revenue"].value == pytest.approx(400.0)            # 2023 quarters only
    assert before.values["revenue"].period_end == "2023-12-31"
    assert before.shares_outstanding.value == pytest.approx(1000.0)
    after = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE)
    assert after.values["revenue"].value == pytest.approx(500.0)
    assert after.shares_outstanding.value == pytest.approx(1100.0)


def test_events_window_uses_acceptance_time_and_conservative_date_rule(sec):
    on_day = {(x["filed"], x["items"]) for x in recent_filings("AAA", "2024-04-25", 120, cache=NOCACHE)}
    assert ("2024-04-25", "5.02") in on_day                      # 08:30 ET: before the close
    assert ("2024-04-25", "2.02,9.01") not in on_day             # 16:03 ET: after the close
    assert ("2024-04-24", "8.01") in on_day                      # dated the day before
    assert not any(f == "2024-04-25" and i == "" for f, i in on_day)   # the 16:06 10-Q
    unknown_time = {(x["filed"], x["items"]) for x in recent_filings("AAA", "2024-04-24", 120, cache=NOCACHE)}
    assert ("2024-04-24", "8.01") not in unknown_time            # no time: known only after day D
    next_day = {(x["filed"], x["items"]) for x in recent_filings("AAA", "2024-04-26", 120, cache=NOCACHE)}
    assert ("2024-04-25", "2.02,9.01") in next_day


def test_older_submission_pages_are_fetched_when_the_window_needs_them(sec):
    sec.subs[AAA_CIK]["filings"]["files"] = [
        {"name": "CIK0000001234-submissions-001.json", "filingCount": 2, "filingFrom": "2022-01-01", "filingTo": "2023-12-31"}]
    sec.pages["CIK0000001234-submissions-001.json"] = _rows([
        ("a-old8k", "2023-03-01", "2023-03-01T14:00:00.000Z", "8-K", "4.02", "8-K"),
        ("a-old10q", "2022-11-01", "2022-11-01T14:00:00.000Z", "10-Q", "", "10-Q"),
    ])
    got = recent_filings("AAA", "2023-03-15", 120, cache=NOCACHE)
    assert [(x["form"], x["items"]) for x in got] == [("8-K", "4.02")]
    sec.fail_pages = True                       # a failed page is an error (-> data gap), not "no filings"
    with pytest.raises(RuntimeError):
        recent_filings("AAA", "2023-03-15", 120, cache=NOCACHE)
    sec.urls.clear()
    recent_filings("AAA", "2024-04-25", 30, cache=NOCACHE)   # window inside 'recent': no page fetch
    assert not any("submissions-001" in u for u in sec.urls)


def test_historical_ticker_uses_the_filer_of_that_date(sec, monkeypatch):
    """Today's ticker table can point at a successor with no history; the predecessor's
    filings are used for earlier dates, and an explicit CIK pin always wins."""
    new_cik = "0000009999"
    sec.tickers = {"0": {"cik_str": 9999, "ticker": "AAA", "title": "AAA Holdings"}}
    sec.facts[new_cik] = {"entityName": "AAA Holdings", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        _quarter("2026-04-01", "2026-06-30", 999.0, "h-26q2", "2026-08-03")]}}}}}
    sec.subs[new_cik] = {"cik": new_cik, "name": "AAA Holdings", "filings": {"recent": _rows([]), "files": []}}
    monkeypatch.setitem(fmod.TICKER_CIK_HISTORY, "AAA", [(AAA_CIK, "AAA Corp", "2026-07-01")])
    old = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE)
    assert old.cik == AAA_CIK and old.values["revenue"].value == pytest.approx(500.0)
    assert recent_filings("AAA", "2024-04-26", 30, cache=NOCACHE)          # predecessor's events
    monkeypatch.setitem(fmod.TICKER_CIK_HISTORY, "AAA", [])
    silent = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE)         # no history known:
    assert silent.cik == new_cik and not silent.values
    assert any("had filed no periodic-report" in g for g in silent.gaps)  # a data gap, not silence
    pinned = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE, cik="1234")
    assert pinned.cik == AAA_CIK and pinned.values["revenue"].value == pytest.approx(500.0)
    fmod.set_cik_overrides({"AAA": 1234})
    assert load_fundamentals("AAA", "2024-04-26", cache=NOCACHE).cik == AAA_CIK


def test_stale_or_zero_share_counts_are_rejected(sec):
    sec.facts[AAA_CIK]["facts"]["dei"] = {}
    sec.facts[AAA_CIK]["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {"units": {"shares": [
        {"end": "2015-12-31", "val": 0, "accn": "x1", "form": "20-F", "filed": "2016-02-17"},
        {"end": "2014-12-31", "val": 500.0, "accn": "x0", "form": "20-F", "filed": "2015-02-17"}]}}
    f = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE)
    assert f.shares_outstanding is None
    assert any("2014-12-31" in g and "15 months" in g for g in f.gaps)
    idx = pd.bdate_range("2023-01-02", "2024-04-26")
    p = pmod.from_frames(pd.DataFrame({"AAA": 50.0, "SPY": 400.0}, index=idx))
    fs = build_factsheet("AAA", "2024-04-26", p, fundamentals=f)
    for key in ("shares_out", "market_cap", "pe_ttm", "ps_ttm", "share_count_change_yoy"):
        assert fs.get(f"AAA.{key}") is None
    assert any("share count" in g for g in fs.data_gaps)


def test_adr_filer_gets_no_market_cap_based_multiples(sec):
    sec.facts[AAA_CIK]["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"] = [
        {"end": "2023-12-31", "val": 2.5e10, "accn": "t-20f", "form": "20-F", "filed": "2024-04-16"}]
    sec.subs[AAA_CIK]["filings"]["recent"] = _rows([
        ("t-20f", "2024-04-16", "2024-04-16T10:00:00.000Z", "20-F", "", "20-F")])
    f = load_fundamentals("AAA", "2024-04-26", cache=NOCACHE)
    assert f.foreign_filer
    idx = pd.bdate_range("2023-01-02", "2024-04-26")
    p = pmod.from_frames(pd.DataFrame({"AAA": 400.0, "SPY": 400.0}, index=idx))
    fs = build_factsheet("AAA", "2024-04-26", p, fundamentals=f)
    assert fs.get("AAA.market_cap") is None and fs.get("AAA.ps_ttm") is None
    assert any("ADR" in g for g in fs.data_gaps)


# =========================================================================== FRED fake


def _monthly(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)), dtype=float)


@pytest.fixture
def fred(monkeypatch):
    series: dict[str, pd.Series] = {}
    monkeypatch.setattr(mmod, "load_series", lambda sid, cache=None: series[sid])
    return series


def test_cpi_yoy_matches_calendar_months_when_a_month_is_missing(fred):
    months = pd.date_range("2024-01-01", "2026-10-01", freq="MS")
    level = pd.Series(100.0 * 1.003 ** np.arange(len(months)), index=months)
    level = level.drop(pd.Timestamp("2025-10-01"))          # never published (shutdown)
    fred["CPIAUCSL"] = level
    snap = {m.series_id: m for m in mmod.macro_snapshot("2026-01-10", series=["CPIAUCSL"])}
    cpi = snap["CPIAUCSL"]
    assert cpi.observation_date == "2025-11-01"
    true_yoy = (level[pd.Timestamp("2025-11-01")] / level[pd.Timestamp("2024-11-01")] - 1) * 100
    assert cpi.value == pytest.approx(true_yoy)               # 12 months, not 13
    yoy = mmod.yoy_percent(level)
    assert pd.Timestamp("2025-10-01") not in yoy.index and pd.Timestamp("2026-10-01") not in yoy.index
    for d, v in yoy.items():                                  # every value is a calendar 12-month change
        assert v == pytest.approx((level[d] / level[d - pd.DateOffset(months=12)] - 1) * 100)


def test_monthly_prints_are_hidden_until_actually_released(fred):
    fred["UNRATE"] = _monthly({"2025-06-01": 4.1, "2025-07-01": 4.2, "2025-08-01": 4.3, "2025-09-01": 4.4,
                               "2025-11-01": 4.5, "2025-12-01": 4.4})
    fred["CPIAUCSL"] = _monthly({f"2024-{m:02d}-01": 300.0 + m for m in range(1, 13)}
                                | {f"2025-{m:02d}-01": 310.0 + m for m in range(1, 10)}
                                | {"2025-11-01": 322.0, "2025-12-01": 323.0})

    def obs(sid, as_of):
        return {m.series_id: m.observation_date for m in mmod.macro_snapshot(as_of, series=[sid])}.get(sid)

    assert obs("UNRATE", "2025-11-19") == "2025-08-01"        # September report came out 2025-11-20
    assert obs("UNRATE", "2025-11-20") == "2025-09-01"
    assert obs("UNRATE", "2025-12-09") == "2025-09-01"        # November report: 2025-12-16
    assert obs("UNRATE", "2026-01-08") == "2025-11-01"        # December report: 2026-01-09
    assert obs("CPIAUCSL", "2025-10-23") == "2025-08-01"      # September CPI: 2025-10-24
    assert obs("CPIAUCSL", "2025-10-24") == "2025-09-01"
    assert obs("CPIAUCSL", "2025-12-16") == "2025-09-01"      # November CPI: 2025-12-18


def test_library_price_loads_never_return_or_cache_a_partial_bar(yahoo, tmp_path, monkeypatch):
    """Open request (cli -> data/prices.py): only the CLI refused unfinished sessions. A library
    caller asking for ``end`` = today before the close got (and cached for 6 hours) today's
    intraday bar as if it were a close. Rows after the last completed session are now dropped,
    and once that session completes the cached window is fetched again."""
    clock = {"settled": "2024-12-27"}                       # a Friday; Yahoo already serves Monday's bar
    monkeypatch.setattr(pmod, "last_completed_session", lambda now=None: clock["settled"])
    cache = DiskCache(tmp_path)
    p = pmod.load_prices(["MSFT", "SPY"], "2024-06-03", "2024-12-31", cache=cache)
    assert p.close.index[-1] == pd.Timestamp("2024-12-27")
    n = len(yahoo.calls)
    pmod.load_prices(["MSFT", "SPY"], "2024-06-03", "2024-12-31", cache=cache)
    assert len(yahoo.calls) == n                             # same session: served from the cache
    clock["settled"] = "2024-12-30"                          # Monday's close is now final
    p2 = pmod.load_prices(["MSFT", "SPY"], "2024-06-03", "2024-12-31", cache=cache)
    assert len(yahoo.calls) > n and p2.close.index[-1] == pd.Timestamp("2024-12-30")
