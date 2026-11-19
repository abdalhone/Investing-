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
    # the LLM analysts are re-pooled with their revised values; in the default overlay mode that
    # pool tilts the quant core (0.5 here: no quant view) by at most the overlay bound
    assert res.llm_aggregate.prob_outperform == pytest.approx(expected)
    tilt = FirmConfig().process.overlay_max_tilt
    assert res.aggregate.prob_outperform == pytest.approx(0.5 + max(-tilt, min(tilt, expected - 0.5)))
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

@pytest.mark.parametrize("rec, mult, long_only, held, expected", [
    ("buy", 0.8, True, False, 0.8), ("buy", 1.7, True, False, 1.0), ("hold", -0.3, True, True, 0.0),
    ("review", 0.9, True, True, 0.0), ("avoid", 0.9, True, True, 0.0), ("sell", 0.6, True, True, 0.0),
    ("avoid", 0.6, False, False, 0.6),
    ("hold", 0.8, True, True, 0.8),      # keep an existing position (the chair may still shrink it)
    ("hold", 1.0, True, False, 0.0),     # "hold" on a name the book does not own: do not buy
    ("hold", 1.0, False, False, 0.0),
])
def test_edge_multiplier_rules(rec, mult, long_only, held, expected):
    assert edge_multiplier_for(memo_obj(rec, mult), long_only, held=held) == pytest.approx(expected)
    if not held:
        assert edge_multiplier_for(memo_obj(rec, mult), long_only) == pytest.approx(expected)   # default: not held


def test_no_memo_means_multiplier_one():
    assert edge_multiplier_for(None, True) == 1.0


def test_memo_only_with_enough_edge_or_when_held():
    flat = {n: 0.51 for n in NAMES}               # below min_prob_to_hold and memo_min_edge: no memo needed
    llm = ScriptedLLM(flat)
    res = committee(llm)
    # not reviewed and not held: the edge multiplier is 0, so an unreviewed edge can never open a position
    assert res.memo is None and res.edge_multiplier == 0.0 and "memo" not in llm.roles()
    cfg = FirmConfig()
    cfg.process.memo_enabled = False              # chair switched off (e.g. quant mode): the pooled view is sized as is
    assert committee(ScriptedLLM(flat), cfg).edge_multiplier == 1.0
    llm = ScriptedLLM(flat)
    res = committee(llm, held=True)
    assert res.memo is not None and res.edge_multiplier == pytest.approx(0.8)
    call = next(c for c in llm.calls if c["role"] == "memo")
    assert call["model"] == FirmConfig().models.committee_model and call["effort"] == FirmConfig().models.committee_effort


def test_memo_review_forces_zero_and_failure_fails_closed():
    assert committee(ScriptedLLM(AGREE, memo=memo_obj("review", 1.0))).edge_multiplier == 0.0
    res = committee(ScriptedLLM(AGREE, fail=("memo",)))
    assert res.memo is None and res.edge_multiplier == 0.0 and not res.hold_current


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
    assert events[4:7] == ["pool_inputs", "aggregate", "challenge"]   # raw vs pooled inputs, then the pool
    assert events.count("revision") == 3
    assert events[-3:] == ["pool_inputs", "aggregate", "memo"]


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
    with pytest.raises(ValueError, match="current_weights"):    # never a guessed book (zeroing would liquidate)
        apply_risk_review(target(), review.model_copy(update={"halt_trading": True}))
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


# --------------------------------------------------------------------------- memo policy with the shipped defaults

def _construct(res, current=None, cfg=None):
    from aifirm.portfolio.construct import construct_portfolio
    fs = make_factsheet()
    return construct_portfolio({"AAA": res.aggregate}, {"AAA": res.edge_multiplier}, {"AAA": fs}, current or {},
                               cfg or FirmConfig(), fs.as_of, {"AAA": fs.sector})


def test_default_config_reviews_every_name_that_could_be_held():
    """FirmConfig() defaults: min_prob_to_hold 0.52 sits below 0.5 + memo_min_edge (0.53). A name pooled at
    0.525 can be sized, so the chair must review it, and its veto must be able to stop the trade."""
    cfg = FirmConfig()
    assert 0.5 + cfg.process.memo_min_edge > 0.525 >= cfg.portfolio.min_prob_to_hold   # the gap being tested
    band = {n: 0.525 for n in NAMES}
    llm = ScriptedLLM(band, memo=memo_obj("avoid", 0.9))
    res = committee(llm, cfg)
    assert res.aggregate.prob_outperform == pytest.approx(0.525)
    assert "memo" in llm.roles() and res.memo is not None and res.edge_multiplier == 0.0
    assert _construct(res).weights().get("AAA", 0.0) == 0.0          # the veto stops the new position
    approved = committee(ScriptedLLM(band, memo=memo_obj("buy", 1.0)), cfg)
    assert approved.edge_multiplier == 1.0 and _construct(approved).weights().get("AAA", 0.0) > 0.0


def test_short_side_names_that_could_be_held_are_reviewed_too():
    cfg = FirmConfig()
    cfg.portfolio.long_only = False
    llm = ScriptedLLM({n: 0.475 for n in NAMES})
    committee(llm, cfg)
    assert "memo" in llm.roles()


def test_unreviewed_names_get_no_edge_whatever_the_thresholds():
    cfg = FirmConfig()
    cfg.process.memo_min_edge = 0.2                 # a lax memo threshold cannot let an edge through unreviewed
    cfg.portfolio.min_prob_to_hold = 0.6
    res = committee(ScriptedLLM({n: 0.58 for n in NAMES}), cfg)
    assert res.memo is None and res.edge_multiplier == 0.0
    assert _construct(res, cfg=cfg).weights().get("AAA", 0.0) == 0.0


def test_failed_memo_on_a_held_name_holds_the_position_instead_of_selling():
    events = []
    res = committee(ScriptedLLM(AGREE, fail=("memo",)), held=True, log=lambda e, p: events.append((e, p)))
    assert res.memo is None and res.hold_current
    assert res.edge_multiplier == 1.0                # never 0: an LLM failure must not size the position to zero
    memo_event = next(p for e, p in events if e == "memo")
    assert "error" in memo_event and memo_event["hold_current"] is True    # still counted as a failed memo
    # Ignoring the flag would not sell the name; honouring it (as the pipeline does) keeps its weight.
    assert _construct(res, current={"AAA": 0.08}).weights().get("AAA", 0.0) > 0.0
    from aifirm.portfolio.construct import construct_portfolio
    fs = make_factsheet()
    pinned = construct_portfolio({"AAA": res.aggregate}, {"AAA": res.edge_multiplier}, {"AAA": fs}, {"AAA": 0.08},
                                 FirmConfig(), fs.as_of, {"AAA": fs.sector}, keep_current=["AAA"])
    assert pinned.weights()["AAA"] == pytest.approx(0.08)


def test_chair_hold_on_an_unheld_name_opens_nothing():
    res = committee(ScriptedLLM(AGREE, memo=memo_obj("hold", 1.0)))
    assert res.memo is not None and res.edge_multiplier == 0.0
    assert _construct(res).weights().get("AAA", 0.0) == 0.0
    held = committee(ScriptedLLM(AGREE, memo=memo_obj("hold", 1.0)), held=True)
    assert held.edge_multiplier == 1.0


# --------------------------------------------------------------------------- anonymity

ROLE_LEAKS = re.compile(r"(?i)\b(?:analyst|revision)\.[a-z]+|\b(?:fundamental|technical|macro|events?|quant)"
                        r"(?:\s+and\s+(?:market-regime|filings))?\s+(?:analyst|desk)s?\b")


class LeakyLLM(ScriptedLLM):
    """Analysts and the challenger describe themselves; the technical analyst's call fails with the
    LLM client's role-prefixed error message."""

    def __init__(self):
        super().__init__(SPLIT, fail=("analyst.technical",))

        def leaky_view(role, s, u, schema):
            name = role.split(".")[1]
            v = view(name, SPLIT[name])
            v.private_insight = f"As the {name} analyst, my desk sees what others miss."
            v.key_risks = [f"The {ROLES_TITLES[name]} may be wrong."]
            v.evidence[0].point = f"Speaking for the {name} desk: the signal is clear."
            return v

        def leaky_challenge(role, s, u, schema):
            c = challenge_obj()
            c.strongest_counterargument = "As the events analyst, I think analyst.events data is decisive."
            return c

        self.register("analyst.", leaky_view)
        self.register("challenge", leaky_challenge)

    def structured(self, **kw):
        if kw["role"] == "analyst.technical":
            with self._lock:
                self.calls.append(kw)
            raise LLMCallFailed("analyst.technical: model refused")
        return super().structured(**kw)


ROLES_TITLES = {"fundamental": "Fundamental analyst", "technical": "Technical analyst",
                "macro": "Macro and market-regime analyst", "events": "Events and filings analyst"}


def test_anonymised_memo_and_revision_prompts_do_not_leak_identities():
    llm = LeakyLLM()
    res = committee(llm)
    assert res.challenger == "events" and res.memo is not None
    memo = llm.user("memo")
    block = memo.split("ANALYST VIEWS")[1].split("FULL FACT SHEET")[0]       # views, challenge and lessons
    assert not ROLE_LEAKS.search(block), ROLE_LEAKS.search(block)
    assert "Abstained: LLM call failed: model refused" in block
    assert "a committee member" in block                                       # self-descriptions replaced
    for role in ("revision.fundamental", "revision.macro"):
        challenge_part = llm.user(role).split("THE CHALLENGE")[1]
        assert not ROLE_LEAKS.search(challenge_part)
        assert "decisive" in challenge_part
    consensus = llm.user("challenge").split("THE COMMITTEE'S POOLED VIEW")[1].split("THE FULL FACT SHEET")[0]
    assert not ROLE_LEAKS.search(consensus)
    from aifirm.agents.committee import CHALLENGE_SYSTEM
    assert "Do not state your role" in CHALLENGE_SYSTEM


# --------------------------------------------------------------------------- blind mode, every prompt

def _leaky_sheet(ticker: str = "CCC") -> FactSheet:
    fs = make_factsheet(ticker)
    fs.name = "Gamma Holdings Inc"
    fs.data_gaps = [
        f"SEC fundamentals could not be loaded (KeyError: 'Ticker {ticker} not found in SEC company_tickers.json')",
        "SEC filing events could not be loaded (HTTPError: 404 for https://data.sec.gov/submissions/CIK0001045810.json)",
        f"SEC data for {ticker} on 2025-01-02 comes from predecessor filer GAMMA HOLDINGS INC (CIK 1045810)",
    ]
    fs.events = fs.events + ["2025-02-28: 8-K - Gamma Holdings announces a material agreement"]
    return fs


BLIND_LEAKS = (re.compile(r"(?i)(?<![A-Za-z0-9])CCC(?![A-Za-z0-9])"), re.compile(r"(?i)gamma"),
               re.compile("1045810"), re.compile(r"(?i)https?://|sec\.gov"), re.compile(r"\d{4}-\d{2}-\d{2}"))


def _assert_blind(text: str, where: str) -> None:
    for pat in BLIND_LEAKS:
        assert not pat.search(text), (where, pat.pattern, pat.search(text))


def test_every_prompt_in_blind_mode_hides_the_ticker_and_name():
    from aifirm.agents import analyst_system_prompt, analyst_user_prompt
    from aifirm.agents.reflection import LESSON_SYSTEM
    from aifirm.agents.risk_review import RISK_REVIEW_SYSTEM
    from aifirm.agents.roles import ROLES
    cfg = FirmConfig()
    cfg.process.blind_mode = True
    cfg.process.challenge_dispersion_threshold = 0.0      # render the challenge and revision prompts too
    fs = _leaky_sheet()
    leaky_lessons = ["CCC (Gamma Holdings) fell after 2024-12-01; do not chase."]
    quant = ViewRecord(agent="quant", model="deterministic", view=view("technical", 0.55, ticker="CCC"))
    quant.view.private_insight = "CCC ranks first in the universe; Gamma Holdings Inc momentum is strong."
    llm = ScriptedLLM(SPLIT)
    run_committee(llm, fs, cfg, quant_view=quant, believability={}, held=True,
                  lessons={"committee": leaky_lessons, **{n: leaky_lessons for n in NAMES}})
    roles = llm.roles()
    assert {"challenge", "memo"} <= set(roles) and any(r.startswith("revision.") for r in roles)
    for c in llm.calls:
        _assert_blind(c["user"], c["role"])
        _assert_blind(c["system"], c["role"])
    for role in ROLES.values():
        _assert_blind(analyst_user_prompt(role, fs, 21, "SPY", leaky_lessons, blind=True), role.name)
        _assert_blind(analyst_system_prompt(role, 21, "SPY"), role.name)
    seen = {}

    def responder(role, system, user, schema):
        seen["risk"] = (system, user)
        return RiskReview(summary="s", hidden_concentrations=[], concerns=[], halt_trading=False)

    tgt = TargetPortfolio(as_of=fs.as_of, core_weight=0.0, cash_weight=0.9, positions=[
        TargetPosition(ticker="CCC", weight=0.10, prob_outperform=0.56, expected_excess_return_pct=1.0,
                       rationale="CCC sized by Kelly")])
    review_portfolio(OfflineLLM({"risk_review": responder}), tgt, {"CCC": fs}, cfg)
    _assert_blind(seen["risk"][1], "risk_review")
    _assert_blind(RISK_REVIEW_SYSTEM, "risk_review system")

    def lesson_responder(role, system, user, schema):
        seen["lesson"] = user
        return make_offline_llm().responders["lesson"](role, system, user, schema)

    rec = forecast().model_copy(update={"ticker": "CCC"})
    write_lesson(OfflineLLM({"lesson": lesson_responder}), rec,
                 "CCC.mom_12_1 was strong; CCC beat (see https://data.sec.gov/CIK0001045810).", cfg)
    _assert_blind(seen["lesson"], "lesson")
    _assert_blind(LESSON_SYSTEM, "lesson system")


# --------------------------------------------------------------------------- served model and citations

class ServedByLLM(ScriptedLLM):
    """Looks like a live LLM whose calls are answered by a fallback model."""

    @property
    def is_offline(self) -> bool:
        return False

    def structured(self, **kw):
        obj, rec = super().structured(**kw)
        rec.model, rec.served_by = kw["model"], "claude-fable-5-2"
        return obj, rec


def test_committee_logs_the_model_that_served_each_call():
    events = []
    res = committee(ServedByLLM(SPLIT), log=lambda e, p: events.append((e, p)))
    assert all(v.model == "claude-fable-5-2" for v in res.views)
    for kind in ("challenge", "revision", "memo"):
        payloads = [p for e, p in events if e == kind]
        assert payloads and all(p["served_by"] == "claude-fable-5-2" for p in payloads), kind


def test_revision_citing_a_bare_key_is_accepted():
    revisions = {"technical": Revision(ticker="AAA", prob_outperform=0.50, expected_excess_return_pct=0.0, changed=True,
                                       reason="Momentum is weaker than I thought.", new_evidence_fact_ids=["mom_12_1"])}
    events = []
    res = committee(ScriptedLLM(SPLIT, revisions=revisions), log=lambda e, p: events.append((e, p)))
    tech = next(p for e, p in events if e == "revision" and p["agent"] == "technical")
    assert tech["status"] == "accepted" and tech["new_evidence_fact_ids"] == ["AAA.mom_12_1"]
    assert next(v for v in res.views if v.agent == "technical").revised_prob == pytest.approx(0.50)


# --------------------------------------------------------------------------- risk review halt

def test_halt_keeps_the_current_book_instead_of_liquidating():
    from aifirm.agents.risk_review import RISK_REVIEW_SYSTEM
    review = RiskReview(summary="data look broken", hidden_concentrations=[], halt_trading=True, concerns=[])
    current = {"AAA": 0.06, "CCC": 0.04, "SPY": 0.30}          # BBB is a proposed new buy; CCC a proposed sale
    out = apply_risk_review(target(), review, current_weights=current, benchmark="SPY")
    assert out.weights() == {"AAA": pytest.approx(0.06), "CCC": pytest.approx(0.04)}
    assert out.core_weight == pytest.approx(0.30) and out.cash_weight == pytest.approx(0.60)
    assert any("no trades this cycle" in n for n in out.notes)
    # Without ``benchmark``, current legs that are not target positions are the core (construction lists
    # every held active name among its positions).
    inferred = apply_risk_review(target(), review, current_weights={"AAA": 0.06, "SPY": 0.30})
    assert inferred.weights() == {"AAA": pytest.approx(0.06)} and inferred.core_weight == pytest.approx(0.30)
    # The model is told the same thing the code does.
    desc = RiskReview.model_fields["halt_trading"].description
    assert "current weight" in desc and "does not liquidate" in desc
    assert "stays at its current weight" in RISK_REVIEW_SYSTEM and "does not liquidate" in RISK_REVIEW_SYSTEM
