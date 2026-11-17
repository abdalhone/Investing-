"""Tests for the committee flow, risk review and reflection (offline; no network, no API key)."""

import re
import threading
import time

import numpy as np
import pandas as pd
import pytest

from aifirm.agents import apply_risk_review, make_offline_llm, review_portfolio, run_committee, write_lesson
from aifirm.agents.committee import anonymous_order, edge_multiplier_for, select_challenger
from aifirm.config import FirmConfig
from aifirm.data.factsheet import Fact, FactSheet, build_factsheet
from aifirm.data.prices import from_frames
from aifirm.llm.client import LLMCallFailed, OfflineLLM
from aifirm.portfolio.aggregate import pool_probabilities
from aifirm.schemas import (AnalystView, Challenge, ChecklistItem, EvidenceItem, ForecastRecord, ICMemo, Revision,
                            RiskConcern, RiskReview, Scenario, TargetPortfolio, TargetPosition, ViewRecord)

ROLE_FACT = {"fundamental": "AAA.pe_ttm", "technical": "AAA.mom_12_1", "macro": "MACRO.VIXCLS", "events": "AAA.ret_1m"}
NAMES = ["fundamental", "technical", "macro", "events"]


def make_factsheet(ticker: str = "AAA") -> FactSheet:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=320)
    close = pd.DataFrame({ticker: 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 320))),
                          "SPY": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 320)))}, index=idx)
    prices = from_frames(close, volume=pd.DataFrame(2e6, index=idx, columns=close.columns))
    asof = str(idx[-1].date())
    fs = build_factsheet(ticker, asof, prices)
    fs.name, fs.sector, fs.data_gaps = "Alpha Corp", "SEMICONDUCTORS", []
    for f in [Fact(f"{ticker}.pe_ttm", "Price / earnings (TTM)", 40.0, "x", asof, "computed"),
              Fact(f"{ticker}.earnings_yield", "Earnings yield (TTM net income / market cap)", 2.5, "%", asof, "computed"),
              Fact(f"{ticker}.revenue_growth_yoy", "Revenue growth, TTM year over year", 25.0, "%", asof, "computed"),
              Fact("MACRO.VIXCLS", "VIX implied volatility index", 14.0, "index", asof, "FRED VIXCLS"),
              Fact("MACRO.DGS10", "10-year Treasury yield", 4.25, "%", asof, "FRED DGS10")]:
        fs.add(f)
    fs.events = ["2025-02-26: 8-K - results of operations (earnings release)", "2025-02-27: 10-K filed"]
    return fs


def view(agent: str, p: float, excess: float | None = None, ticker: str = "AAA") -> AnalystView:
    x = (p - 0.5) * 20 if excess is None else excess
    return AnalystView(
        ticker=ticker, abstain=False, abstain_reason="", prob_outperform=p, expected_excess_return_pct=x,
        scenarios=[Scenario(name="bull", probability=0.25, excess_return_pct=x + 4, description="Upside."),
                   Scenario(name="base", probability=0.5, excess_return_pct=x, description="Drift."),
                   Scenario(name="bear", probability=0.25, excess_return_pct=x - 4, description="Downside.")],
        evidence=[EvidenceItem(point="A signal from this desk's slice.", fact_ids=[ROLE_FACT[agent].replace("AAA", ticker)],
                               direction="positive" if p >= 0.5 else "negative")],
        key_risks=["Noise."], what_would_change_my_mind="New facts.", private_insight="Slice-specific signal.",
    )


def challenge_obj() -> Challenge:
    return Challenge(ticker="AAA", strongest_counterargument="Valuation is stretched relative to growth.",
                     evidence=[EvidenceItem(point="P/E is high.", fact_ids=["AAA.pe_ttm"], direction="negative")],
                     overlooked_risks_or_opportunities=["Multiple compression."],
                     questions_for_the_committee=["Is the growth already priced?"])


def memo_obj(rec: str = "buy", mult: float = 0.8) -> ICMemo:
    return ICMemo(ticker="AAA", recommendation=rec, thesis_one_sentence="t", variant_perception="none",
                  base_rate_note="about half", catalysts=[], premortem=["noise"], kill_criteria=[],
                  checklist=[ChecklistItem(item="Evidence grounded", passed=True, note="")],
                  conviction_multiplier=mult, multiplier_reason="r")


class ScriptedLLM(OfflineLLM):
    """Scripted committee: fixed analyst probabilities, optional revisions, memo and failures.
    Records every call; optional per-role delays scramble thread completion order."""

    def __init__(self, probs: dict[str, float], revisions: dict[str, Revision] | None = None,
                 memo: ICMemo | None = None, fail: tuple[str, ...] = (), delays: dict[str, float] | None = None):
        super().__init__({
            "analyst.": lambda role, s, u, schema: view(role.split(".")[1], probs[role.split(".")[1]]),
            "challenge": lambda role, s, u, schema: challenge_obj(),
            "revision.": lambda role, s, u, schema: (revisions or {}).get(role.split(".")[1]) or Revision(
                ticker="AAA", prob_outperform=0.5, expected_excess_return_pct=0.0, changed=False, reason="",
                new_evidence_fact_ids=[]),
            "memo": lambda role, s, u, schema: memo or memo_obj(),
        })
        self.fail, self.delays = fail, delays or {}
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def structured(self, **kw):
        with self._lock:
            self.calls.append(kw)
        time.sleep(self.delays.get(kw["role"], 0.0))
        if any(kw["role"].startswith(f) for f in self.fail):
            raise LLMCallFailed(f"{kw['role']} failed")
        return super().structured(**kw)

    def roles(self) -> list[str]:
        return [c["role"] for c in self.calls]

    def user(self, role: str) -> str:
        return next(c["user"] for c in self.calls if c["role"] == role)


def committee(llm, cfg=None, held=False, quant=None, believability=None, log=None, ticker="AAA"):
    return run_committee(llm, make_factsheet(ticker), cfg or FirmConfig(), quant_view=quant,
                         believability=believability or {}, lessons={}, held=held,
                         log=log or (lambda e, p: None))


AGREE = {"fundamental": 0.56, "technical": 0.57, "macro": 0.55, "events": 0.56}
SPLIT = {"fundamental": 0.85, "technical": 0.62, "macro": 0.60, "events": 0.42}


# --------------------------------------------------------------------------- sealed views and weights

def test_views_come_back_in_config_order_despite_thread_timing():
    cfg = FirmConfig()
    cfg.models.analyst_models = ["m0", "m1"]
    cfg.models.samples_per_analyst = 2
    delays = {"analyst.fundamental": 0.05, "analyst.technical": 0.02}
    res = committee(ScriptedLLM(AGREE, delays=delays), cfg)
    assert [(v.agent, v.sample, v.model) for v in res.views] == [
        (n, s, f"m{s}") for n in NAMES for s in (0, 1)]          # models rotate across samples
    assert all(v.weight == pytest.approx(0.5) for v in res.views)  # believability split across samples


def test_weights_use_believability_and_halve_ungrounded_views():
    quant = ViewRecord(agent="quant", model="deterministic", view=view("technical", 0.53), grounding_ok=False)
    res = committee(ScriptedLLM(AGREE), quant=quant, believability={"technical": 2.0, "quant": 0.8})
    w = {v.agent: v.weight for v in res.views}
    assert w == {"fundamental": 1.0, "technical": 2.0, "macro": 1.0, "events": 1.0, "quant": pytest.approx(0.4)}
    assert quant.weight == 1.0          # the caller's quant view is not mutated


def test_failed_analyst_abstains_and_is_excluded_from_the_pool():
    res = committee(ScriptedLLM(AGREE, fail=("analyst.macro",)))
    macro = next(v for v in res.views if v.agent == "macro")
    assert macro.view.abstain and macro.grounding_issues[0].startswith("llm failure")
    assert res.aggregate.n_views == 3 and res.aggregate.n_abstained == 1


def test_unknown_analyst_in_config_fails_loudly():
    cfg = FirmConfig()
    cfg.analysts.enabled = ["technical", "fundamentals"]
    with pytest.raises(ValueError, match="fundamentals"):
        committee(ScriptedLLM(AGREE), cfg)


# --------------------------------------------------------------------------- challenge

def test_no_challenge_when_views_agree():
    llm = ScriptedLLM(AGREE)
    res = committee(llm)
    assert res.challenge is None and res.challenger is None and not res.aggregate.challenged
    assert not any(r.startswith(("challenge", "revision")) for r in llm.roles())


def test_challenge_disabled_by_config():
    cfg = FirmConfig()
    cfg.process.challenge_enabled = False
    llm = ScriptedLLM(SPLIT)
    assert committee(llm, cfg).challenge is None
    assert "challenge" not in llm.roles()


def test_challenger_is_most_opposed_on_the_other_side_of_half():
    llm = ScriptedLLM(SPLIT)
    res = committee(llm)
    # fundamental (0.85) is farthest from the pool, but events (0.42) is the farthest on the opposite side of 0.5
    assert res.aggregate.dispersion > FirmConfig().process.challenge_dispersion_threshold
    assert res.challenger == "events" and res.challenge is not None
    assert llm.roles().count("challenge") == 1
    challenge_call = next(c for c in llm.calls if c["role"] == "challenge")
    assert challenge_call["model"] == next(v.model for v in res.views if v.agent == "events")
    prompt = challenge_call["user"]
    assert "[AAA.pe_ttm]" in prompt and "[MACRO.VIXCLS]" in prompt          # full fact sheet
    assert "0.420" in prompt and "0.850" in prompt                         # own view and anonymous member list
    assert "fundamental" not in prompt and "technical" not in prompt       # no agent names


def test_select_challenger_falls_back_to_farthest_and_skips_quant():
    recs = [ViewRecord(agent="technical", model="m", view=view("technical", 0.55)),
            ViewRecord(agent="macro", model="m", view=view("macro", 0.70)),
            ViewRecord(agent="quant", model="deterministic", view=view("technical", 0.30))]
    assert select_challenger(recs, 0.60).agent == "macro"     # quant is opposite but cannot write a challenge


def test_revisions_are_private_and_limited_to_own_slice():
    llm = ScriptedLLM(SPLIT)
    committee(llm)
    revision_roles = sorted(r for r in llm.roles() if r.startswith("revision."))
    assert revision_roles == ["revision.fundamental", "revision.macro", "revision.technical"]
    tech = llm.user("revision.technical")
    assert "0.620" in tech                                                  # its own sealed view
    assert "Valuation is stretched relative to growth." in tech            # the challenge
    assert "[AAA.mom_12_1]" in tech and "[AAA.pe_ttm]" not in tech.split("THE CHALLENGE")[0]
    assert "[MACRO." not in tech
    for other in ("0.850", "0.600", "0.420"):                              # nobody else's estimate
        assert other not in tech
    assert "events" not in tech and "fundamental" not in tech


def test_revisions_repool_count_flips_and_reject_unsupported_changes():
    revisions = {
        "technical": Revision(ticker="AAA", prob_outperform=0.45, expected_excess_return_pct=-1.0, changed=True,
                              reason="P/E is high.", new_evidence_fact_ids=["AAA.pe_ttm"]),          # from the challenge
        "macro": Revision(ticker="AAA", prob_outperform=0.40, expected_excess_return_pct=-2.0, changed=True,
                          reason="Everyone else disagrees.", new_evidence_fact_ids=[]),            # social pressure
    }
    events = []
    res = committee(ScriptedLLM(SPLIT, revisions=revisions), log=lambda e, p: events.append((e, p)))
    by = {v.agent: v for v in res.views}
    assert by["technical"].revised_prob == pytest.approx(0.45)
    assert by["macro"].revised_prob == pytest.approx(0.60)           # rejected: sealed view stands
    assert by["fundamental"].revised_prob == pytest.approx(0.85)     # unchanged
    assert by["events"].revised_prob == pytest.approx(0.42)          # the challenger keeps its view
    assert res.aggregate.challenged and res.aggregate.flips == 1
    expected = pool_probabilities([0.85, 0.45, 0.60, 0.42], [1.0] * 4, FirmConfig().process.trim_fraction)
    assert res.aggregate.prob_outperform == pytest.approx(expected)
    statuses = {p["agent"]: p["status"] for e, p in events if e == "revision"}
    assert statuses["technical"] == "accepted" and statuses["macro"].startswith("rejected")
    assert statuses["fundamental"] == "unchanged"


def test_failed_revision_keeps_sealed_view_and_failed_challenge_skips_round():
    res = committee(ScriptedLLM(SPLIT, fail=("revision.technical",)))
    assert next(v for v in res.views if v.agent == "technical").revised_prob == pytest.approx(0.62)
    assert res.aggregate.challenged
    res = committee(ScriptedLLM(SPLIT, fail=("challenge",)))
    assert res.challenge is None and res.challenger is None and not res.aggregate.challenged


# --------------------------------------------------------------------------- memo

@pytest.mark.parametrize("rec, mult, long_only, expected", [
    ("buy", 0.8, True, 0.8), ("buy", 1.7, True, 1.0), ("hold", -0.3, True, 0.0),
    ("review", 0.9, True, 0.0), ("avoid", 0.9, True, 0.0), ("sell", 0.6, True, 0.0), ("avoid", 0.6, False, 0.6),
])
def test_edge_multiplier_rules(rec, mult, long_only, expected):
    assert edge_multiplier_for(memo_obj(rec, mult), long_only) == pytest.approx(expected)


def test_no_memo_means_multiplier_one():
    assert edge_multiplier_for(None, True) == 1.0


def test_memo_only_with_enough_edge_or_when_held():
    flat = {n: 0.51 for n in NAMES}
    llm = ScriptedLLM(flat)
    res = committee(llm)
    assert res.memo is None and res.edge_multiplier == 1.0 and "memo" not in llm.roles()
    llm = ScriptedLLM(flat)
    res = committee(llm, held=True)
    assert res.memo is not None and res.edge_multiplier == pytest.approx(0.8)
    call = next(c for c in llm.calls if c["role"] == "memo")
    assert call["model"] == FirmConfig().models.committee_model and call["effort"] == FirmConfig().models.committee_effort


def test_memo_review_forces_zero_and_failure_fails_closed():
    assert committee(ScriptedLLM(AGREE, memo=memo_obj("review", 1.0))).edge_multiplier == 0.0
    res = committee(ScriptedLLM(AGREE, fail=("memo",)))
    assert res.memo is None and res.edge_multiplier == 0.0


def test_memo_prompt_is_anonymized_and_deterministic():
    prompts = []
    for _ in range(2):
        llm = ScriptedLLM(SPLIT)
        committee(llm)
        prompts.append(llm.user("memo"))
    assert prompts[0] == prompts[1]
    block = prompts[0].split("ANALYST VIEWS")[1].split("FULL FACT SHEET")[0]
    for name in NAMES:
        assert name not in block
    labels = re.findall(r"^Analyst ([A-Z])$", block, re.M)
    assert labels == ["A", "B", "C", "D"]
    order = anonymous_order("AAA", 4)
    sealed = re.findall(r"Sealed: P\(outperform\) ([0-9.]+)", block)
    assert [float(p) for p in sealed] == [SPLIT[NAMES[i]] for i in order]
    assert "Valuation is stretched relative to growth." in prompts[0]      # the challenge is shown to the chair


def test_log_events_cover_every_stage():
    events = []
    committee(ScriptedLLM(SPLIT), log=lambda e, p: events.append(e))
    assert events[:4] == ["view"] * 4
    assert events[4:6] == ["aggregate", "challenge"]
    assert events.count("revision") == 3
    assert events[-2:] == ["aggregate", "memo"]


# --------------------------------------------------------------------------- offline end to end

def test_offline_committee_runs_end_to_end_including_challenge():
    cfg = FirmConfig()
    cfg.process.challenge_dispersion_threshold = 0.0      # force the challenge path
    events = []
    res = committee(make_offline_llm(), cfg, held=True, log=lambda e, p: events.append(e))
    assert len(res.views) == 4 and all(0.40 <= v.view.prob_outperform <= 0.60 for v in res.views if not v.view.abstain)
    assert all(v.grounding_ok for v in res.views)
    assert res.aggregate.challenged and res.challenge is not None
    assert res.memo is not None and 0.0 <= res.edge_multiplier <= 1.0
    assert {"view", "aggregate", "challenge", "revision", "memo"} <= set(events)


def test_blind_committee_never_shows_the_ticker():
    cfg = FirmConfig()
    cfg.process.blind_mode = True
    llm = ScriptedLLM(SPLIT)
    res = committee(llm, cfg)
    for call in llm.calls:
        assert "AAA" not in call["user"] and "Alpha Corp" not in call["user"]
    assert res.memo is not None and res.memo.ticker == "AAA"


# --------------------------------------------------------------------------- risk review

def target() -> TargetPortfolio:
    return TargetPortfolio(as_of="2025-03-21", core_weight=0.0, cash_weight=0.70, positions=[
        TargetPosition(ticker="AAA", weight=0.10, prob_outperform=0.56, expected_excess_return_pct=1.0, rationale="r"),
        TargetPosition(ticker="BBB", weight=0.20, prob_outperform=0.58, expected_excess_return_pct=1.5, rationale="r"),
    ])


def test_apply_risk_review_only_scales_down():
    review = RiskReview(summary="s", hidden_concentrations=[], halt_trading=False, concerns=[
        RiskConcern(ticker="AAA", concern="thin liquidity", multiplier=0.5),
        RiskConcern(ticker="aaa", concern="another", multiplier=0.8),     # duplicates: smallest wins
        RiskConcern(ticker="BBB", concern="cannot increase", multiplier=1.7),
        RiskConcern(ticker="ZZZ", concern="not held", multiplier=0.0),
    ])
    t = target()
    out = apply_risk_review(t, review)
    w = out.weights()
    assert w["AAA"] == pytest.approx(0.05) and w["BBB"] == pytest.approx(0.20)
    assert out.cash_weight == pytest.approx(0.75)
    assert "risk_review" in out.positions[0].constraints_hit
    assert t.weights()["AAA"] == pytest.approx(0.10)          # input not mutated


def test_apply_risk_review_portfolio_level_and_halt():
    review = RiskReview(summary="s", hidden_concentrations=[], halt_trading=False, concerns=[
        RiskConcern(ticker="PORTFOLIO", concern="all momentum", multiplier=0.5),
        RiskConcern(ticker="AAA", concern="x", multiplier=-3.0)])
    out = apply_risk_review(target(), review)
    assert out.weights() == {"BBB": pytest.approx(0.10)}
    halted = apply_risk_review(target(), review.model_copy(update={"halt_trading": True}))
    assert halted.weights() == {} and halted.cash_weight == pytest.approx(1.0)
    assert apply_risk_review(target(), None) == target()


def test_review_portfolio_offline_disabled_and_failing():
    fs = {"AAA": make_factsheet("AAA"), "BBB": make_factsheet("BBB")}
    cfg = FirmConfig()
    review = review_portfolio(make_offline_llm(), target(), fs, cfg)
    assert isinstance(review, RiskReview) and not review.halt_trading
    assert all(0.0 <= c.multiplier <= 1.0 for c in review.concerns)
    assert review_portfolio(ScriptedLLM(AGREE, fail=("risk_review",)), target(), fs, cfg) is None
    cfg.process.risk_review_enabled = False
    assert review_portfolio(make_offline_llm(), target(), fs, cfg) is None


def test_review_portfolio_blind_mode_aliases_tickers_and_maps_back():
    fs = {"AAA": make_factsheet("AAA"), "BBB": make_factsheet("BBB")}
    cfg = FirmConfig()
    cfg.process.blind_mode = True
    seen = {}

    def responder(role, system, user, schema):
        seen["user"] = user
        return RiskReview(summary="S02 is crowded", hidden_concentrations=["S01 and S02 share a theme"],
                          concerns=[RiskConcern(ticker="S02", concern="crowded", multiplier=0.5)], halt_trading=False)

    review = review_portfolio(OfflineLLM({"risk_review": responder}), target(), fs, cfg)
    assert "AAA" not in seen["user"] and "BBB" not in seen["user"]
    assert review.concerns[0].ticker == "BBB"
    assert review.hidden_concentrations == ["AAA and BBB share a theme"]


# --------------------------------------------------------------------------- reflection

def forecast(resolved: bool = True) -> ForecastRecord:
    return ForecastRecord(id="f1", run_id="r1", agent="technical", model="m", ticker="AAA", benchmark="SPY",
                          as_of="2025-01-02", horizon_days=21, resolve_on="2025-02-03", prob_outperform=0.68,
                          expected_excess_return_pct=2.0, resolved=resolved, outcome=0 if resolved else None,
                          realized_excess_return_pct=-3.1 if resolved else None, brier=0.4624 if resolved else None)


def test_write_lesson_offline_unresolved_and_failure():
    lesson = write_lesson(make_offline_llm(), forecast(), "Momentum was strong.", FirmConfig())
    assert lesson is not None and lesson.agent == "technical" and lesson.ticker == "AAA"
    assert "several independent facts" in lesson.process_lesson
    assert write_lesson(make_offline_llm(), forecast(resolved=False), "", FirmConfig()) is None
    assert write_lesson(ScriptedLLM(AGREE, fail=("lesson",)), forecast(), "", FirmConfig()) is None


def test_write_lesson_blind_prompt_hides_ticker_and_dates():
    cfg = FirmConfig()
    cfg.process.blind_mode = True
    seen = {}

    def responder(role, system, user, schema):
        seen["user"] = user
        return make_offline_llm().responders["lesson"](role, system, user, schema)

    lesson = write_lesson(OfflineLLM({"lesson": responder}), forecast(), "Cited AAA.mom_12_1.", cfg)
    assert "AAA" not in seen["user"] and "2025-01-02" not in seen["user"]
    assert lesson.ticker == "AAA"
