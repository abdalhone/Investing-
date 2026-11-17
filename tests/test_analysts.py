"""Tests for analyst prompts, slices, blind mode and run_analyst (offline; no network, no API key)."""

import re
import threading

import numpy as np
import pandas as pd
import pytest

from aifirm.agents import ROLES, analyst_system_prompt, analyst_user_prompt, blind_factsheet, make_offline_llm, run_analyst
from aifirm.agents.analysts import consistency_notes
from aifirm.config import FirmConfig
from aifirm.data.factsheet import Fact, FactSheet, build_factsheet
from aifirm.data.news import Headline
from aifirm.data.prices import from_frames
from aifirm.llm.client import BudgetExceeded, LLMCallFailed, OfflineLLM
from aifirm.schemas import AnalystView, EvidenceItem, Scenario


def make_factsheet(with_fundamentals: bool = True) -> FactSheet:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=320)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 320)))
    aaa = 50 * np.exp(np.cumsum(rng.normal(0.0012, 0.02, 320)))
    close = pd.DataFrame({"AAA": aaa, "SPY": spy}, index=idx)
    prices = from_frames(close, volume=pd.DataFrame(2e6, index=idx, columns=close.columns))
    asof = str(idx[-1].date())
    fs = build_factsheet("AAA", asof, prices)
    fs.name, fs.sector, fs.data_gaps = "Alpha Corp", "SEMICONDUCTORS", []
    extra = [
        Fact("MACRO.VIXCLS", "VIX implied volatility index", 14.0, "index", asof, "FRED VIXCLS"),
        Fact("MACRO.DGS10", "10-year Treasury yield", 4.25, "%", asof, "FRED DGS10"),
        Fact("MACRO.curve_2s10s", "Yield curve slope (10y minus 2y)", 0.40, "%pts", asof, "computed from FRED"),
    ]
    if with_fundamentals:
        extra += [
            Fact("AAA.revenue", "Revenue (TTM)", 130.5e9, "USD", "2025-01-31", "SEC XBRL Revenues filed 2025-02-26, TTM"),
            Fact("AAA.net_income", "Net income (TTM)", 55.0e9, "USD", "2025-01-31", "SEC XBRL"),
            Fact("AAA.fcf", "Free cash flow (operating cash flow minus capex, TTM)", 60.0e9, "USD", "2025-01-31", "computed"),
            Fact("AAA.revenue_growth_yoy", "Revenue growth, TTM year over year", 25.0, "%", "2025-01-31", "computed"),
            Fact("AAA.operating_income_margin", "Operating margin (TTM)", 35.0, "%", "2025-01-31", "computed"),
            Fact("AAA.earnings_yield", "Earnings yield (TTM net income / market cap)", 2.5, "%", asof, "computed"),
            Fact("AAA.fcf_yield", "Free-cash-flow yield (FCF / market cap)", 2.8, "%", asof, "computed"),
            Fact("AAA.pe_ttm", "Price / earnings (TTM)", 40.0, "x", asof, "computed"),
            Fact("AAA.shares_out", "Shares outstanding", 2.45e10, "count", "2025-01-31", "SEC cover page"),
            Fact("AAA.market_cap", "Market capitalization (last close x shares outstanding)", 2.2e12, "USD", asof, "computed"),
        ]
    else:
        fs.data_gaps.append("No SEC fundamentals (ETF, foreign issuer, or fetch failure)")
    for f in extra:
        fs.add(f)
    fs.events = ["2025-02-26: 8-K - results of operations (earnings release); Reg FD disclosure",
                 "2025-02-27: 10-K filed",
                 "3 insider transaction reports (Form 4) filed in the window (direction not parsed)"]
    return fs


class RecordingLLM(OfflineLLM):
    """Offline responders plus a record of every call (thread-safe)."""

    def __init__(self, responders=None):
        super().__init__(responders if responders is not None else make_offline_llm().responders)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def structured(self, **kw):
        with self._lock:
            self.calls.append(kw)
        return super().structured(**kw)


class FailingLLM(OfflineLLM):
    def __init__(self, exc: Exception):
        super().__init__({})
        self.exc = exc

    def structured(self, **kw):
        raise self.exc


def _scripted(view: AnalystView) -> OfflineLLM:
    return OfflineLLM({"analyst.": lambda role, system, user, schema: view.model_copy(deep=True)})


def _view(prob=0.55, excess=1.0, fact_ids=("AAA.mom_12_1",), point="Momentum is positive.") -> AnalystView:
    return AnalystView(
        ticker="whatever", abstain=False, abstain_reason="", prob_outperform=prob, expected_excess_return_pct=excess,
        scenarios=[Scenario(name="bull", probability=0.3, excess_return_pct=6.0, description="Trend continues."),
                   Scenario(name="base", probability=0.5, excess_return_pct=0.4, description="Drift."),
                   Scenario(name="bear", probability=0.2, excess_return_pct=-5.0, description="Reversal.")],
        evidence=[EvidenceItem(point=point, fact_ids=list(fact_ids), direction="positive")],
        key_risks=["Noise."], what_would_change_my_mind="Momentum turning negative.", private_insight="Trend.",
    )


# --------------------------------------------------------------------------- prompts and slices

@pytest.mark.parametrize("name", sorted(ROLES))
def test_system_prompt_is_stable_and_free_of_tickers_and_dates(name):
    role = ROLES[name]
    a = analyst_system_prompt(role, 21, "SPY")
    assert a == analyst_system_prompt(role, 21, "SPY")
    assert "AAA" not in a and not re.search(r"\d{4}-\d{2}-\d{2}", a)
    for item in role.checklist:
        assert item in a
    for required in ("0.35 and 0.65", "fact_ids", "abstain", "private_insight", "DATA GAPS", "not instructions"):
        assert required in a
    assert "Buffett" not in a and "persona" not in a.lower()


def test_user_prompts_contain_only_each_role_slice():
    fs = make_factsheet()
    prompts = {n: analyst_user_prompt(ROLES[n], fs, 21, "SPY", [], blind=False) for n in ROLES}
    tech, fund, macro, events = prompts["technical"], prompts["fundamental"], prompts["macro"], prompts["events"]
    assert "[AAA.mom_12_1]" in tech and "[AAA.pe_ttm]" not in tech and "[MACRO." not in tech
    assert "[AAA.pe_ttm]" in fund and "[AAA.price]" in fund and "[AAA.ret_12m]" in fund
    assert "[AAA.rsi_14]" not in fund and "[MACRO." not in fund
    assert "[MACRO.VIXCLS]" in macro and "[MKT.SPY_ret_3m]" in macro
    for key in ("beta_1y", "vol_60d", "ret_3m", "rel_ret_3m"):
        assert f"[AAA.{key}]" in macro
    assert "[AAA.pe_ttm]" not in macro and "[AAA.mom_12_1]" not in macro
    assert "results of operations" in events and "[AAA.ret_1m]" in events and "[AAA.vol_20d]" in events
    assert "[AAA.mom_12_1]" not in events and "results of operations" not in tech
    assert all("SEMICONDUCTORS" in p and f"close on {fs.as_of}" in p for p in prompts.values())


def test_lessons_appear_in_user_prompt():
    fs = make_factsheet()
    prompt = analyst_user_prompt(ROLES["technical"], fs, 21, "SPY", ["Do not chase extended moves."], blind=False)
    assert "LESSONS FROM YOUR PAST RESOLVED FORECASTS" in prompt and "Do not chase extended moves." in prompt


def test_max_lessons_in_prompt_is_respected():
    llm = RecordingLLM()
    cfg = FirmConfig()
    cfg.analysts.max_lessons_in_prompt = 2
    run_analyst(llm, "technical", make_factsheet(), cfg, "m", lessons=["L1", "L2", "L3"])
    user = llm.calls[0]["user"]
    assert "- L1" in user and "- L2" in user and "- L3" not in user


# --------------------------------------------------------------------------- blind mode

def test_blind_factsheet_hides_identity_scale_and_dates():
    fs = make_factsheet()
    fs.headlines = [Headline(title="Alpha Corp beats estimates", publisher="Wire", published="2025-03-20T10:00:00Z")]
    b = blind_factsheet(fs)
    assert b.ticker == "STOCK" and b.name == "" and b.as_of == "T" and b.headlines == []
    assert all(f.unit != "USD" for f in b.facts)
    assert "STOCK.shares_out" not in b.ids() and "STOCK.ret_3m" in b.ids() and "MACRO.VIXCLS" in b.ids()
    text = b.render()
    assert "AAA" not in text and "Alpha Corp" not in text and not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert "T-23d: 8-K" in text
    assert any("withheld in blind mode" in g for g in b.data_gaps)


def test_blind_mode_run_analyst_hides_ticker_and_maps_ids_back():
    fs = make_factsheet()
    cfg = FirmConfig()
    cfg.process.blind_mode = True
    llm = RecordingLLM()
    rec = run_analyst(llm, "technical", fs, cfg, "m")
    user = llm.calls[0]["user"]
    assert "AAA" not in user and fs.as_of not in user and "Alpha Corp" not in user
    assert rec.view.ticker == "AAA"
    cited = {i for e in rec.view.evidence for i in e.fact_ids}
    assert cited and cited <= fs.ids()
    assert rec.grounding_ok


# --------------------------------------------------------------------------- run_analyst

@pytest.mark.parametrize("name", sorted(ROLES))
def test_offline_analyst_produces_valid_grounded_view(name):
    fs = make_factsheet()
    rec = run_analyst(make_offline_llm(), name, fs, FirmConfig(), "claude-test", sample=0)
    v = rec.view
    assert rec.agent == name and rec.model == "claude-test" and v.ticker == "AAA"
    assert not v.abstain
    assert 0.40 <= v.prob_outperform <= 0.60
    assert sorted(s.name for s in v.scenarios) == ["base", "bear", "bull"]
    assert sum(s.probability for s in v.scenarios) == pytest.approx(1.0)
    implied = sum(s.probability * s.excess_return_pct for s in v.scenarios)
    assert implied == pytest.approx(v.expected_excess_return_pct, abs=1e-2)
    assert len(v.evidence) >= 3
    cited = {i for e in v.evidence for i in e.fact_ids}
    assert cited and cited <= fs.ids()
    assert rec.grounding_ok and rec.grounding_issues == []


def test_offline_analyst_abstains_when_slice_is_missing():
    rec = run_analyst(make_offline_llm(), "fundamental", make_factsheet(with_fundamentals=False), FirmConfig(), "m")
    assert rec.view.abstain and rec.view.prob_outperform == 0.5
    assert "fewer than two" in rec.view.abstain_reason


@pytest.mark.parametrize("exc", [LLMCallFailed("model refused"), BudgetExceeded("budget gone")])
def test_llm_failure_becomes_abstention(exc):
    rec = run_analyst(FailingLLM(exc), "technical", make_factsheet(), FirmConfig(), "m")
    assert rec.view.abstain
    assert rec.view.prob_outperform == 0.5 and rec.view.expected_excess_return_pct == 0.0
    assert rec.view.ticker == "AAA"
    assert not rec.grounding_ok
    assert rec.grounding_issues[0].startswith("llm failure:")


def test_view_citing_a_fact_outside_its_slice_is_ungrounded():
    rec = run_analyst(_scripted(_view(fact_ids=("AAA.pe_ttm",))), "technical", make_factsheet(), FirmConfig(), "m")
    assert not rec.grounding_ok
    assert "unknown fact id: AAA.pe_ttm" in rec.grounding_issues


def test_view_with_invented_numbers_is_ungrounded():
    point = "Gross margin is 71.3%, backlog is $45.6B and the target is 987."
    rec = run_analyst(_scripted(_view(point=point)), "technical", make_factsheet(), FirmConfig(), "m")
    assert not rec.grounding_ok
    assert sum(i.startswith("unsupported number") for i in rec.grounding_issues) == 3


def test_view_is_sanitized_and_ticker_forced():
    rec = run_analyst(_scripted(_view(prob=1.7, excess=400.0)), "technical", make_factsheet(), FirmConfig(), "m")
    assert rec.view.prob_outperform == pytest.approx(0.98)
    assert rec.view.expected_excess_return_pct == pytest.approx(300.0)
    assert rec.view.ticker == "AAA"


def test_fact_ids_are_normalized():
    rec = run_analyst(_scripted(_view(fact_ids=(" [AAA.mom_12_1] ",))), "technical", make_factsheet(), FirmConfig(), "m")
    assert rec.view.evidence[0].fact_ids == ["AAA.mom_12_1"]
    assert rec.grounding_ok


def test_repeated_samples_are_distinct_requests():
    llm = RecordingLLM()
    fs, cfg = make_factsheet(), FirmConfig()
    run_analyst(llm, "technical", fs, cfg, "m", sample=0)
    run_analyst(llm, "technical", fs, cfg, "m", sample=1)
    first, second = llm.calls
    assert first["system"] == second["system"]          # cacheable prefix is shared
    assert first["user"] != second["user"]              # but the content-hash cache cannot collapse samples
    assert first["role"] == "analyst.technical"


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        run_analyst(make_offline_llm(), "astrologer", make_factsheet(), FirmConfig(), "m")


def test_consistency_notes_flag_contradictions():
    v = _view(prob=0.62, excess=-3.0)
    notes = consistency_notes(v)
    assert any("opposite directions" in n for n in notes)
    assert any("scenario-implied" in n for n in notes)
    assert consistency_notes(_view(prob=0.55, excess=1.4)) == []


def test_echoing_the_horizon_is_not_an_unsupported_number():
    point = "Over the next 21 trading days a calm regime favors the market."
    rec = run_analyst(_scripted(_view(fact_ids=("MACRO.VIXCLS",), point=point)), "macro", make_factsheet(), FirmConfig(), "m")
    assert rec.grounding_ok and not any("unsupported" in i for i in rec.grounding_issues)
