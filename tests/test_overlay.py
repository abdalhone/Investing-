"""Tests for the bounded LLM overlay on the quant core and its adaptive bound
(aifirm/portfolio/overlay.py, aifirm/journal/calibration.paired_rank_ic) and for how the
committee wires the overlay, the tier gate and calibration together. No network."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from test_committee import ScriptedLLM, make_factsheet, view

from aifirm.agents import run_committee
from aifirm.config import FirmConfig
from aifirm.data.factsheet import FactSheet
from aifirm.journal import LLM_POOL_AGENT, paired_rank_ic
from aifirm.journal.calibration import Calibrator
from aifirm.journal.forecasts import make_forecast_records
from aifirm.portfolio.overlay import adaptive_bound, combine, overlay_aggregate
from aifirm.quant.signals import horizon_sigma_pct
from aifirm.schemas import AnalystView, EvidenceItem, ForecastRecord, NewsEvent, Scenario, ViewRecord

# ----------------------------------------------------------------------------- combine


@pytest.mark.parametrize("p_quant,p_llm,tilt,expected", [
    (0.53, 0.70, 0.05, 0.58),          # bullish LLM: capped at +tilt
    (0.53, 0.30, 0.05, 0.48),          # bearish LLM: capped at -tilt
    (0.53, 0.55, 0.05, 0.58),          # the LLM pool's own edge (+0.05) is added to the core
    (0.53, 0.52, 0.05, 0.55),          # a mildly bullish pool raises the core, never pulls it toward 0.5
    (0.53, 0.50, 0.05, 0.53),          # a neutral pool (gated, calibrated to no skill) is no information
    (0.53, 0.48, 0.05, 0.51),          # a mildly bearish pool lowers the core by its edge only
    (0.53, None, 0.05, 0.53),          # every LLM analyst abstained: the quant exactly
    (None, 0.70, 0.05, 0.55),          # no quant view: the core is 0.5
    (0.53, 0.90, 0.0, 0.53),           # bound switched off
    (0.53, float("nan"), 0.05, 0.53),  # a broken LLM number is no LLM number
])
def test_combine_bounds_the_llm_tilt_around_the_quant(p_quant, p_llm, tilt, expected):
    p, t = combine(p_quant, p_llm, tilt)
    assert p == pytest.approx(expected)
    core = 0.5 if p_quant is None else p_quant
    assert t == pytest.approx(p - core) and abs(t) <= tilt + 1e-12


def test_combine_neutral_llm_pool_never_cuts_the_quant_core():
    """Regression: the overlay used ``p_llm - p_quant``, so an LLM pool the tier gate capped at
    0.5, or that calibration shrank toward 0.5 (no measured skill), pulled every quant pick
    toward 0.5 and below ``min_prob_to_hold``: the live book sold what the quant-only shadow
    held. The tilt is now the LLM pool's own edge, so 'no information' tilts by exactly 0."""
    for p_quant in (0.505, 0.53, 0.545, 0.56, 0.45):
        assert combine(p_quant, 0.5, 0.05) == (p_quant, 0.0)
    p, t = combine(0.545, 0.5112, 0.05)                 # bullish but calibrated as no-skill (b = 0.10)
    assert p == pytest.approx(0.5562) and t == pytest.approx(0.0112)
    for p_llm in (0.51, 0.6, 0.9):                       # bullish never lowers, bearish never raises
        assert combine(0.545, p_llm, 0.05)[1] > 0 and combine(0.545, 1 - p_llm, 0.05)[1] < 0


def test_combine_rejects_a_bound_outside_zero_to_one_half():
    for bad in (-0.01, 0.6, float("nan")):
        with pytest.raises(ValueError):
            combine(0.5, 0.6, bad)


def _vr(agent, p, excess=0.0, abstain=False, weight=1.0, model="m"):
    v = AnalystView(ticker="AAA", abstain=abstain, abstain_reason="x" if abstain else "", prob_outperform=p,
                    expected_excess_return_pct=excess, scenarios=[], evidence=[], key_risks=[],
                    what_would_change_my_mind="", private_insight="")
    return ViewRecord(agent=agent, model="deterministic" if agent == "quant" else model, view=v, weight=weight)


def test_overlay_aggregate_sizes_from_the_final_probability_on_the_quant_scale():
    """The LLM's own expected-excess numbers (here +40%) never reach sizing: the final excess
    is sigma_h * Phi^-1(p_final), and p_final is within the bound of the quant."""
    q = _vr("quant", 0.53)
    llm = [_vr("technical", 0.80, excess=40.0), _vr("macro", 0.75, excess=35.0)]
    final, pool = overlay_aggregate("AAA", q, llm, trim=0.2, max_tilt=0.05, sigma_h_pct=6.0)
    assert final.combination == "overlay" and final.prob_outperform == pytest.approx(0.58)
    assert final.prob_quant == pytest.approx(0.53) and final.prob_llm == pytest.approx(pool.prob_outperform)
    assert final.overlay_tilt == pytest.approx(0.05) and final.max_tilt == pytest.approx(0.05)
    assert final.expected_excess_return_pct == pytest.approx(6.0 * norm.ppf(0.58))
    assert final.n_views == 3 and final.dispersion == pytest.approx(pool.dispersion)


def test_overlay_aggregate_is_the_quant_exactly_when_every_llm_analyst_abstains():
    q = _vr("quant", 0.537)
    final, pool = overlay_aggregate("AAA", q, [_vr("technical", 0.5, abstain=True)], trim=0.2, max_tilt=0.05,
                                    sigma_h_pct=5.0)
    assert final.prob_outperform == 0.537 and final.prob_llm is None and final.overlay_tilt == 0.0
    assert final.n_views == 1 and final.n_abstained == 1 and pool.n_views == 0
    nothing, _ = overlay_aggregate("AAA", None, [], trim=0.2, max_tilt=0.05, sigma_h_pct=5.0)
    assert nothing.prob_outperform == 0.5 and nothing.n_views == 0          # no view at all: no edge


# ----------------------------------------------------------------------------- the committee in overlay mode


QUANT = ViewRecord(agent="quant", model="deterministic", view=view("technical", 0.53))
BULL = {"fundamental": 0.80, "technical": 0.78, "macro": 0.82, "events": 0.79}


def _cfg(**process):
    cfg = FirmConfig()
    cfg.process.challenge_enabled = False
    for k, v in process.items():
        setattr(cfg.process, k, v)
    return cfg


def _run(llm, cfg, quant=QUANT, fs=None, **kw):
    return run_committee(llm, fs or make_factsheet(), cfg, quant_view=quant, believability={}, lessons={},
                         held=False, **kw)


def test_the_llm_can_only_tilt_the_quant_core_within_the_bound():
    res = _run(ScriptedLLM(BULL), _cfg())
    a = res.aggregate
    assert a.combination == "overlay" and a.prob_quant == pytest.approx(0.53)
    assert res.llm_aggregate.prob_outperform > 0.75                   # the LLM pool is very bullish ...
    assert a.prob_outperform == pytest.approx(0.58)                   # ... but moves the quant by 0.05 only
    adaptive = _run(ScriptedLLM(BULL), _cfg(), max_tilt=0.0)          # the adaptive bound switched it off
    assert adaptive.aggregate.prob_outperform == pytest.approx(0.53)
    bear = _run(ScriptedLLM({k: 0.2 for k in BULL}), _cfg())
    assert bear.aggregate.prob_outperform == pytest.approx(0.48)


def test_the_chair_still_only_shrinks_after_the_overlay():
    from test_committee import memo_obj
    res = _run(ScriptedLLM(BULL, memo=memo_obj("buy", 0.4)), _cfg())
    assert res.edge_multiplier == pytest.approx(0.4) and res.aggregate.prob_outperform == pytest.approx(0.58)


def test_pool_mode_keeps_the_previous_behaviour_and_records_both_inputs():
    res = _run(ScriptedLLM(BULL), _cfg(combination="pool"))
    a = res.aggregate
    assert a.combination == "pool" and a.prob_outperform > 0.7       # the quant is one member among five
    assert a.prob_quant == pytest.approx(0.53) and a.prob_llm == pytest.approx(res.llm_aggregate.prob_outperform)


def test_max_tilt_outside_the_range_is_refused():
    with pytest.raises(ValueError):
        _run(ScriptedLLM(BULL), _cfg(), max_tilt=0.7)


def test_calibration_shrinks_the_pooled_copy_but_views_keep_the_raw_probability():
    """Regression guard: forecasts must store what the analyst said (calibrators are refitted
    on raw data); only the pooling copies are calibrated. The quant is never calibrated."""
    cals = {a: Calibrator(agent=a, a=0.0, b=0.25, n=100, fitted=True) for a in BULL}
    cals["quant"] = Calibrator(agent="quant", a=0.0, b=0.0, n=100, fitted=True)   # must be ignored
    events = []
    res = _run(ScriptedLLM(BULL), _cfg(combination="pool"), calibrators=cals,
               log=lambda e, p: events.append((e, p)))
    raw = {v.agent: v.view.prob_outperform for v in res.views}
    pooled = {v.agent: v.view.prob_outperform for v in res.pool_views}
    assert raw == {**BULL, "quant": pytest.approx(0.53)}
    for agent, p in BULL.items():
        assert 0.5 < pooled[agent] < p and pooled[agent] == pytest.approx(cals[agent].apply(p))
    assert pooled["quant"] == pytest.approx(0.53)
    inputs = next(p for e, p in events if e == "pool_inputs")
    row = next(r for r in inputs["views"] if r["agent"] == "technical")
    assert row["raw_prob"] == 0.78 and row["pool_prob"] == pytest.approx(cals["technical"].apply(0.78))
    assert row["calibrator"] == {"a": 0.0, "b": 0.25, "n": 100}
    # the forecast records written from the result carry the raw probabilities
    recs = make_forecast_records("r", "2025-03-03", res.views, res.aggregate, _cfg(), None)
    assert {r.agent: r.prob_outperform for r in recs if r.agent in BULL} == BULL
    llm_rec = next(r for r in recs if r.agent == LLM_POOL_AGENT)
    assert llm_rec.prob_outperform == pytest.approx(res.aggregate.prob_llm)


def _news_sheet(tier):
    fs = make_factsheet()
    ev = NewsEvent(ticker="AAA", headline_index=1, event_type="product", direction="positive", materiality="high",
                   summary="a product launch", tier=tier, tier_reason="test", source_name="src", publisher="p",
                   link_host="example.com", published="2025-03-03T12:00:00+00:00", headline_sha256="0" * 64)
    fs.attach_news_events([ev])
    return fs


def _news_only_view(agent, p):
    """A bullish view whose only positive evidence is the news event."""
    return AnalystView(
        ticker="AAA", abstain=False, abstain_reason="", prob_outperform=p, expected_excess_return_pct=3.0,
        scenarios=[Scenario(name=n, probability=1 / 3, excess_return_pct=3.0, description="d")
                   for n in ("bull", "base", "bear")],
        evidence=[EvidenceItem(point="A launch was reported.", fact_ids=["AAA.news_1"], direction="positive")],
        key_risks=["r"], what_would_change_my_mind="w", private_insight="i")


def test_tier_gate_caps_llm_views_resting_on_unverified_news_but_never_the_quant():
    llm = ScriptedLLM(BULL)
    llm.register("analyst.", lambda role, s, u, schema: _news_only_view(role.split(".")[1], 0.8))
    quiet_quant = ViewRecord(agent="quant", model="deterministic", view=AnalystView(
        ticker="AAA", abstain=False, abstain_reason="", prob_outperform=0.53, expected_excess_return_pct=1.0,
        scenarios=[], evidence=[EvidenceItem(point="Neutral factors.", fact_ids=["AAA.vol_60d"],
                                             direction="neutral")],
        key_risks=[], what_would_change_my_mind="", private_insight=""))
    res = _run(llm, _cfg(), quant=quiet_quant, fs=_news_sheet("C"))
    assert all(v.view.prob_outperform == 0.8 for v in res.views if v.agent in BULL)      # raw views untouched
    assert all(v.view.prob_outperform == 0.5 for v in res.pool_views if v.agent in BULL)  # capped for pooling
    assert next(v for v in res.pool_views if v.agent == "quant").view.prob_outperform == 0.53   # not gated
    # the gated LLM pool is "no information": the quant core stands exactly (it used to be cut to 0.5,
    # below min_prob_to_hold, so unverified bullish news sold what the quant alone would hold)
    assert len(res.tier_gate_notes) == 4 and res.aggregate.prob_outperform == pytest.approx(0.53)
    assert res.aggregate.overlay_tilt == pytest.approx(0.0)
    trusted = _run(llm, _cfg(), quant=quiet_quant, fs=_news_sheet("A"))                  # tier A supports it
    assert not trusted.tier_gate_notes and trusted.aggregate.prob_outperform == pytest.approx(0.58)
    off = _run(llm, _cfg(tier_gate_enabled=False), quant=quiet_quant, fs=_news_sheet("C"))
    assert not off.tier_gate_notes and off.aggregate.prob_outperform == pytest.approx(0.58)


# ----------------------------------------------------------------------------- the adaptive bound


def _records(n_dates, ic_llm, ic_quant, *, names=8, start="2024-01-02", resolve_lag=30, pre_cutoff=False, seed=0):
    """Resolved quant and llm_pool records per date whose rank ICs are about ic_quant and ic_llm."""
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.bdate_range(start, periods=n_dates, freq="21B"):
        x = rng.normal(size=names)
        for agent, ic in (("quant", ic_quant), (LLM_POOL_AGENT, ic_llm)):
            noise = rng.normal(size=names)
            score = ic * x + math.sqrt(max(0.0, 1 - ic * ic)) * noise
            for i in range(names):
                as_of = str(d.date())
                out.append(ForecastRecord(
                    id=f"r|{as_of}|T{i}|{agent}", run_id="r", agent=agent, model="m", ticker=f"T{i}", benchmark="SPY",
                    as_of=as_of, horizon_days=21, resolve_on=str((d + pd.Timedelta(days=resolve_lag)).date()),
                    prob_outperform=float(0.5 + 0.05 * np.tanh(score[i])), expected_excess_return_pct=0.0,
                    pre_cutoff=pre_cutoff and agent == LLM_POOL_AGENT, resolved=True, outcome=int(x[i] > 0),
                    realized_excess_return_pct=float(x[i]), brier=0.25))
    return out


def test_paired_ic_compares_the_two_agents_on_their_common_names_only():
    recs = _records(6, 0.9, 0.9, names=6)
    extra = [r.model_copy(update={"id": r.id + "x", "ticker": "ONLYQ" + r.ticker, "realized_excess_return_pct": -9.0})
             for r in recs if r.agent == "quant"]
    res = paired_rank_ic(recs + extra, "2026-01-01", LLM_POOL_AGENT, "quant", min_names=3)
    base = paired_rank_ic(recs, "2026-01-01", LLM_POOL_AGENT, "quant", min_names=3)
    assert res == base and res.n_dates == 6


def test_bound_needs_enough_resolved_dates_then_halves_and_switches_off():
    cfg = FirmConfig()
    few = adaptive_bound(_records(5, -0.9, 0.9), "2026-01-01", cfg)
    assert few.max_tilt == 0.05 and few.n_dates == 5 and "< 12" in few.reason
    worse = adaptive_bound(_records(20, -0.6, 0.6), "2026-01-01", cfg)
    assert worse.max_tilt == 0.0 and worse.t_stat < -2 and "switched off" in worse.reason
    better = adaptive_bound(_records(20, 0.6, -0.6), "2026-01-01", cfg)
    assert better.max_tilt == 0.05 and better.t_stat > 2                # never raised above the configured max
    # a mildly worse LLM pool: find seeds whose t lands in (-2, -1) and check the halving
    halved = [b for b in (adaptive_bound(_records(14, 0.0, 0.25, seed=s), "2026-01-01", cfg) for s in range(40))
              if -2 <= (b.t_stat or 0) < -1]
    assert halved and all(b.max_tilt == pytest.approx(0.025) and "halved" in b.reason for b in halved)
    assert set(worse.to_dict()) >= {"max_tilt", "configured", "reason", "n_dates", "t_stat", "lags"}


def test_bound_is_point_in_time_and_excludes_pre_cutoff_llm_forecasts():
    """Regression guard (P3): outcomes resolving after the decision close cannot shrink (or
    keep) the bound; pre-cutoff LLM forecasts count only when explicitly included."""
    cfg = FirmConfig()
    good = _records(14, 0.3, 0.3, start="2023-01-02")
    future_bad = _records(20, -0.9, 0.9, start="2025-06-02")
    as_of = "2025-06-01"
    assert adaptive_bound(good + future_bad, as_of, cfg).to_dict() == adaptive_bound(good, as_of, cfg).to_dict()
    tainted = _records(20, -0.9, 0.9, pre_cutoff=True)
    assert adaptive_bound(tainted, "2026-01-01", cfg).n_dates == 0
    assert adaptive_bound(tainted, "2026-01-01", cfg, include_pre_cutoff=True).max_tilt == 0.0


def test_horizon_sigma_matches_the_quants_scale():
    fs = FactSheet(ticker="AAA", as_of="2025-01-02")
    assert horizon_sigma_pct(fs, 21) == pytest.approx(30.0 * math.sqrt(21 / 252))     # the 30% fallback
    with pytest.raises(ValueError):
        horizon_sigma_pct(fs, 0)


def test_committee_caps_a_revision_raised_on_tier_c_news_from_the_challenge():
    """Regression (revision hole): the sealed technical view is supported by a price fact, the
    challenger cites a tier-C news event and the reviser raises to 0.9 citing that event. The
    accepted revision's new evidence is stored on the record and the gate caps the raise."""
    from test_committee import challenge_obj
    from aifirm.schemas import Revision
    rumor_challenge = challenge_obj()
    rumor_challenge.evidence = [EvidenceItem(point="A launch was reported.", fact_ids=["AAA.news_1"],
                                             direction="positive")]
    revisions = {"technical": Revision(ticker="AAA", prob_outperform=0.9, expected_excess_return_pct=8.0,
                                       changed=True, reason="The launch.", new_evidence_fact_ids=["AAA.news_1"])}
    probs = {"fundamental": 0.85, "technical": 0.62, "macro": 0.60, "events": 0.42}
    llm = ScriptedLLM(probs, revisions=revisions)
    llm.register("challenge", lambda role, s, u, schema: rumor_challenge)
    cfg = _cfg(challenge_enabled=True, challenge_dispersion_threshold=0.0)
    for tier, capped in (("C", True), ("A", False)):
        res = _run(llm, cfg, fs=_news_sheet(tier))
        tech = next(v for v in res.views if v.agent == "technical")
        assert tech.revised_prob == pytest.approx(0.9) and tech.revision_fact_ids == ["AAA.news_1"]
        pooled = next(v for v in res.pool_views if v.agent == "technical")
        assert pooled.revised_prob == pytest.approx(0.62 if capped else 0.9)
        assert any(n.startswith("tier gate: technical") and "revision capped" in n
                   for n in res.tier_gate_notes) is capped


def test_a_deep_dive_narrower_than_the_ic_minimum_is_reported_not_awaited():
    """Regression: with 2-3 deep-dive names per date every date was skipped and the bound said
    '0 resolved dates < 12' for ever, as if waiting would help. The skipped dates are now
    counted and the reason names the cause."""
    cfg = FirmConfig()
    thin = adaptive_bound(_records(20, -0.9, 0.9, names=3), "2026-01-01", cfg)
    assert thin.n_dates == 0 and thin.n_thin_dates == 20 and thin.max_tilt == 0.05
    assert "20 resolved date(s) skipped because fewer than 5 names" in thin.reason
    assert "deep_dive_top_n" in thin.reason and thin.to_dict()["n_thin_dates"] == 20
    wide = adaptive_bound(_records(20, -0.6, 0.6, names=8), "2026-01-01", cfg)
    assert wide.n_thin_dates == 0 and wide.max_tilt == 0.0
    res = paired_rank_ic(_records(4, 0.5, 0.5, names=3), "2026-01-01", LLM_POOL_AGENT, "quant", min_names=5)
    assert res.n_dates == 0 and res.n_thin == 4
