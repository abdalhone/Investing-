"""Tests for the deterministic data desk (no network)."""

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
