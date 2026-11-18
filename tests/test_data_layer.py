"""Tests for the deterministic data desk (no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aifirm.data.factsheet import build_factsheet, filing_events, fundamental_facts, price_facts
from aifirm.data.fundamentals import FactValue, Fundamentals, _pit, quarterly_series, ttm_from_quarters
from aifirm.data.prices import from_frames


def _prices(n=320, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    aaa = 50 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, n)))
    close = pd.DataFrame({"AAA": aaa, "SPY": spy}, index=idx)
    return from_frames(close, volume=pd.DataFrame(2e6, index=idx, columns=close.columns))


def test_price_facts_match_hand_computation():
    p = _prices()
    facts = {f.id: f for f in price_facts("AAA", p, "SPY")}
    adj = p.adj_close["AAA"]
    assert facts["AAA.price"].value == pytest.approx(adj.iloc[-1])
    assert facts["AAA.ret_3m"].value == pytest.approx((adj.iloc[-1] / adj.iloc[-64] - 1) * 100)
    assert facts["AAA.mom_12_1"].value == pytest.approx((adj.iloc[-22] / adj.iloc[-253] - 1) * 100)
    vol = adj.pct_change().dropna().iloc[-60:].std() * np.sqrt(252) * 100
    assert facts["AAA.vol_60d"].value == pytest.approx(vol)
    assert 0 <= facts["AAA.rsi_14"].value <= 100
    assert facts["AAA.max_dd_1y"].value <= 0
    assert "AAA.beta_1y" in facts


def test_factsheet_is_point_in_time():
    """Changing prices after the as-of date must not change any fact."""
    p = _prices()
    asof = str(p.close.index[250].date())
    fs1 = build_factsheet("AAA", asof, p)
    close2 = p.close.copy()
    close2.iloc[251:] *= 3.0
    p2 = from_frames(close2, volume=p.volume)
    fs2 = build_factsheet("AAA", asof, p2)
    assert [(f.id, f.value) for f in fs1.facts] == [(f.id, f.value) for f in fs2.facts]
    assert fs1.as_of == asof


def test_factsheet_render_contains_ids_and_gaps():
    p = _prices()
    fs = build_factsheet("AAA", str(p.close.index[-1].date()), p)
    text = fs.render()
    assert "[AAA.ret_3m]" in text
    assert "No SEC fundamentals" in text  # missing fundamentals are declared, not silently empty
    assert "no headlines available" in text


def _flow_rows():
    # fiscal year starting 2024-01-01; 10-Q quarters Q1-Q3 + FY from 10-K (Q4 must be derived)
    rows = [
        ("2024-01-01", "2024-03-31", 100.0, "2024-05-01"),
        ("2024-04-01", "2024-06-30", 110.0, "2024-08-01"),
        ("2024-01-01", "2024-06-30", 210.0, "2024-08-01"),
        ("2024-07-01", "2024-09-30", 120.0, "2024-11-01"),
        ("2024-01-01", "2024-09-30", 330.0, "2024-11-01"),
        ("2024-01-01", "2024-12-31", 460.0, "2025-02-15"),
        ("2025-01-01", "2025-03-31", 140.0, "2025-05-01"),
    ]
    df = pd.DataFrame(rows, columns=["start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["concept"] = "Revenues"
    df["form"] = "10-Q"
    return df


def test_quarterly_series_derives_q4_and_ttm():
    q = quarterly_series(_flow_rows())
    assert q[pd.Timestamp("2024-12-31")] == pytest.approx(130.0)  # 460 - 330
    total, end = ttm_from_quarters(q)
    assert end == pd.Timestamp("2025-03-31")
    assert total == pytest.approx(110 + 120 + 130 + 140)


def test_pit_hides_unfiled_and_uses_latest_known_restatement():
    df = _flow_rows()
    # restatement of Q1 2024 filed later
    extra = pd.DataFrame([{"start": pd.Timestamp("2024-01-01"), "end": pd.Timestamp("2024-03-31"), "val": 90.0,
                           "filed": pd.Timestamp("2025-06-01"), "concept": "Revenues", "form": "10-K/A"}])
    df = pd.concat([df, extra], ignore_index=True)
    before = _pit(df, pd.Timestamp("2025-04-01"))
    assert pd.Timestamp("2025-03-31") not in set(before["end"])  # Q1 2025 not filed yet
    assert before[before["end"] == pd.Timestamp("2024-03-31")]["val"].iloc[0] == 100.0
    after = _pit(df, pd.Timestamp("2025-07-01"))
    assert after[after["end"] == pd.Timestamp("2024-03-31")]["val"].iloc[0] == 90.0


def test_fundamental_facts_valuation():
    f = Fundamentals(ticker="AAA", cik="0", entity_name="AAA Corp", as_of="2025-06-30")
    f.values = {
        "revenue": FactValue(1000.0, "2025-03-31", "2025-05-01", "10-Q", "Revenues", "TTM (4 quarters)"),
        "net_income": FactValue(100.0, "2025-03-31", "2025-05-01", "10-Q", "NetIncomeLoss", "TTM (4 quarters)"),
        "op_cash_flow": FactValue(150.0, "2025-03-31", "2025-05-01", "10-Q", "X", "TTM (4 quarters)"),
        "capex": FactValue(50.0, "2025-03-31", "2025-05-01", "10-Q", "X", "TTM (4 quarters)"),
        "equity": FactValue(500.0, "2025-03-31", "2025-05-01", "10-Q", "X", "instant"),
    }
    f.prior_year = {"revenue": FactValue(800.0, "2024-03-31", "", "10-Q", "Revenues", "TTM (4 quarters)")}
    f.shares_outstanding = FactValue(10.0, "2025-04-20", "2025-05-01", "10-Q", "dei", "instant")
    facts, gaps = fundamental_facts(f, last_price=200.0)
    d = {x.id: x.value for x in facts}
    assert d["AAA.market_cap"] == pytest.approx(2000.0)
    assert d["AAA.pe_ttm"] == pytest.approx(20.0)
    assert d["AAA.fcf"] == pytest.approx(100.0)
    assert d["AAA.fcf_yield"] == pytest.approx(5.0)
    assert d["AAA.revenue_growth_yoy"] == pytest.approx(25.0)
    assert d["AAA.net_income_margin"] == pytest.approx(10.0)
    assert d["AAA.roe"] == pytest.approx(20.0)


def test_filing_events_describe_8k_items():
    ev = filing_events([
        {"form": "8-K", "filed": "2025-05-01", "items": "2.02,9.01"},
        {"form": "4", "filed": "2025-05-02", "items": ""},
        {"form": "10-Q", "filed": "2025-05-03", "items": ""},
    ])
    assert any("results of operations" in e for e in ev)
    assert any("Form 4" in e for e in ev)
    assert any("10-Q" in e for e in ev)


def test_pit_treats_a_filing_dated_the_as_of_day_as_unknown_without_acceptance_time():
    """SEC dates carry no time and most results are filed after the 16:00 close: a filing
    dated D is usable from the decision of D + 1 unless its acceptance time says otherwise."""
    df = _flow_rows()
    assert pd.Timestamp("2025-03-31") not in set(_pit(df, pd.Timestamp("2025-05-01"))["end"])  # filed that day
    assert pd.Timestamp("2025-03-31") in set(_pit(df, pd.Timestamp("2025-05-02"))["end"])
    early = df.copy()
    early["known_at"] = early["filed"] + pd.Timedelta(hours=9)          # accepted 09:00 ET
    assert pd.Timestamp("2025-03-31") in set(_pit(early, pd.Timestamp("2025-05-01"))["end"])


def test_short_history_does_not_claim_one_year_statistics():
    p = _prices(n=18)
    facts = {f.id for f in price_facts("AAA", p, "SPY")}
    for key in ("max_dd_1y", "dist_52w_high", "ret_12m", "mom_12_1", "beta_1y", "adv_20d"):
        assert f"AAA.{key}" not in facts
    assert "AAA.price" in facts
    fs = build_factsheet("AAA", str(p.close.index[-1].date()), p)
    assert any("Only 18 trading days" in g for g in fs.data_gaps)
    full = build_factsheet("AAA", str(_prices().close.index[-1].date()), _prices())
    assert full.get("AAA.max_dd_1y") is not None and not any("trading days of price history" in g for g in full.data_gaps)


def _valuation_fundamentals(shares: float, **kw) -> Fundamentals:
    f = Fundamentals(ticker="AAA", cik="0", entity_name="AAA Corp", as_of="2025-06-30", **kw)
    f.values = {"revenue": FactValue(1000.0, "2025-03-31", "2025-05-01", "10-Q", "Revenues", "TTM (4 quarters)"),
                "net_income": FactValue(100.0, "2025-03-31", "2025-05-01", "10-Q", "NetIncomeLoss", "TTM (4 quarters)")}
    f.shares_outstanding = FactValue(shares, "2025-04-20", "2025-05-01", "10-Q", "dei", "instant")
    return f


def test_zero_share_count_produces_no_valuation_facts():
    facts, gaps = fundamental_facts(_valuation_fundamentals(0.0), last_price=200.0)
    ids = {x.id for x in facts}
    assert not ids & {"AAA.shares_out", "AAA.market_cap", "AAA.pe_ttm", "AAA.ps_ttm"}
    assert any("Shares outstanding unavailable" in g for g in gaps)


def test_foreign_filer_gets_no_market_cap_or_multiples():
    facts, gaps = fundamental_facts(_valuation_fundamentals(10.0, foreign_filer=True), last_price=200.0)
    ids = {x.id for x in facts}
    assert "AAA.market_cap" not in ids and "AAA.pe_ttm" not in ids and "AAA.ps_ttm" not in ids
    assert any("ADR" in g for g in gaps)


def test_market_cap_uses_as_traded_price_and_split_adjusted_share_count():
    """Back-adjusted ``close`` must not be multiplied by an as-filed share count: the fact
    sheet uses the as-traded level, and puts a pre-split count on the as-of share basis."""
    idx = pd.bdate_range("2024-01-01", periods=300)
    ex = idx[200]
    traded = pd.Series(np.where(idx < ex, 1000.0, 100.0), index=idx)   # 10-for-1 split at ``ex``
    adjusted = pd.Series(100.0, index=idx)                            # what Yahoo serves today
    splits = pd.DataFrame({"AAA": [10.0], "SPY": [1.0]}, index=[ex])
    p = from_frames(pd.DataFrame({"AAA": adjusted, "SPY": 400.0}),
                    raw_close=pd.DataFrame({"AAA": traded, "SPY": 400.0}), splits=splits)
    f = _valuation_fundamentals(1e6)
    f.shares_outstanding = FactValue(1e6, str(idx[150].date()), str(idx[155].date()), "10-Q", "dei", "instant")
    f.shares_outstanding_prior_year = FactValue(1e6, str(idx[0].date()), str(idx[5].date()), "10-Q", "dei", "instant")
    before = build_factsheet("AAA", str(idx[190].date()), p, fundamentals=f)
    assert before.get("AAA.price").value == pytest.approx(1000.0)
    assert before.get("AAA.market_cap").value == pytest.approx(1e9)
    after = build_factsheet("AAA", str(idx[250].date()), p, fundamentals=f)
    assert after.get("AAA.price").value == pytest.approx(100.0)
    assert after.get("AAA.shares_out").value == pytest.approx(1e7)
    assert after.get("AAA.market_cap").value == pytest.approx(1e9)
    assert after.get("AAA.share_count_change_yoy").value == pytest.approx(0.0)


def test_default_cache_dir_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """Open request (cli -> data/cache.py): the default cache was ``<cwd>/.cache/aifirm``, so a
    scheduled run from another directory started with a cold cache (refetching data and paying
    again for cached LLM calls)."""
    from aifirm.data.cache import DiskCache, default_dir

    monkeypatch.delenv("AIFIRM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    first = default_dir()
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path / "other")
    assert default_dir() == first == tmp_path / "xdg" / "aifirm"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_dir() == Path.home() / ".cache" / "aifirm"
    monkeypatch.setenv("AIFIRM_CACHE_DIR", str(tmp_path / "pinned"))
    assert DiskCache().root == tmp_path / "pinned"
