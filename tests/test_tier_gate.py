"""Tests for the deterministic tier gate: tier-C news can never raise a probability above 0.5."""

from __future__ import annotations

import pytest

from aifirm.agents import apply_tier_gate, tier_gate
from aifirm.agents.tier_gate import supporting_fact_ids
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.data.news import Headline
from aifirm.schemas import AnalystView, EvidenceItem, NewsEvent, Scenario, ViewRecord

AS_OF = "2026-03-20"


def news_event(idx: int, tier: str, etype: str = "earnings", direction: str = "positive") -> NewsEvent:
    return NewsEvent(ticker="AAA", headline_index=idx, event_type=etype, direction=direction, materiality="high",
                     summary="Quarterly results beat expectations", tier=tier, tier_reason="test",
                     source_name="Reuters" if tier != "C" else "unverified source", publisher="x", link_host="x.com",
                     published=f"{AS_OF}T14:00:00+00:00", headline_sha256="0" * 64)


def sheet(tiers: tuple[str, ...] = ("A", "B", "C"), events: tuple[str, ...] = ("2026-03-02: 8-K - results of operations",)) -> FactSheet:
    fs = FactSheet(ticker="AAA", as_of=AS_OF)
    fs.add(Fact("AAA.ret_3m", "Return, 3 months", 5.0, "%", AS_OF, "prices"))
    fs.add(Fact("MACRO.VIXCLS", "VIX", 14.0, "index", AS_OF, "FRED"))
    fs.add(Fact("MACRO.DGS10", "10-year Treasury yield", 4.2, "%", AS_OF, "FRED"))
    fs.events = list(events)
    fs.headlines = [Headline("t", "p", f"{AS_OF}T14:00:00Z")]
    fs.attach_news_events([news_event(i + 1, t) for i, t in enumerate(tiers)])
    return fs                      # AAA.news_1 = tier A, news_2 = B, news_3 = C; AAA.filing_1


def view(p: float, *items: tuple[str, list[str]], excess: float = 2.0, abstain: bool = False) -> AnalystView:
    return AnalystView(
        ticker="AAA", abstain=abstain, abstain_reason="", prob_outperform=p, expected_excess_return_pct=excess,
        scenarios=[Scenario(name="bull", probability=0.3, excess_return_pct=6.0, description=""),
                   Scenario(name="base", probability=0.5, excess_return_pct=2.0, description=""),
                   Scenario(name="bear", probability=0.2, excess_return_pct=-4.0, description="")],
        evidence=[EvidenceItem(point="p", fact_ids=ids, direction=d) for d, ids in items],
        key_risks=[], what_would_change_my_mind="", private_insight="")


def test_bullish_view_resting_only_on_tier_c_news_is_capped():
    fs = sheet()
    v = view(0.8, ("positive", ["AAA.news_3"]))
    out, note = tier_gate(v, fs)
    assert out.prob_outperform == 0.5 and out.expected_excess_return_pct == 0.0 and "capped" in note
    assert v.prob_outperform == 0.8 and v.expected_excess_return_pct == 2.0          # input untouched
    assert out.evidence == v.evidence and out.scenarios == v.scenarios


@pytest.mark.parametrize("cited", [["AAA.news_1"], ["AAA.news_2"], ["AAA.filing_1"], ["AAA.ret_3m"], ["MACRO.DGS10"],
                                   ["news_2"], ["[AAA.ret_3m]"]])
def test_admissible_support_lets_a_bullish_view_through(cited):
    fs = sheet()
    v = view(0.7, ("positive", cited), ("negative", ["AAA.news_3"]))     # a rumor cited only as a risk
    out, note = tier_gate(v, fs)
    assert out is v and note == "" and supporting_fact_ids(v, fs)


@pytest.mark.parametrize("rumor_direction", ["positive", "neutral"])
@pytest.mark.parametrize("cited", [["AAA.news_1"], ["AAA.news_2"], ["AAA.filing_1"], ["AAA.ret_3m"], ["MACRO.DGS10"]])
def test_a_rumor_in_a_separate_point_caps_the_whole_view(cited, rumor_direction):
    """Regression (gate bypass, round 3): taint was judged item by item, so a view could put
    the tier-C rumor and an admissible fact in two separate positive points and pass. A
    bullish view that leans on unverified news anywhere (a positive or neutral point) is now
    capped, whatever else it cites."""
    fs = sheet()
    v = view(0.92, (rumor_direction, ["AAA.news_3"]), ("positive", cited), excess=12.0)
    out, note = tier_gate(v, fs)
    assert out.prob_outperform == 0.5 and out.expected_excess_return_pct == 0.0 and "capped" in note
    assert supporting_fact_ids(v, fs) == []
    rec = ViewRecord(agent="events", model="m", view=v)
    assert tier_gate(rec, fs)[0].view.prob_outperform == 0.5


@pytest.mark.parametrize("filing", [
    "2026-03-02: 10-Q filed", "2026-03-02: 10-K filed", "2026-03-02: DEF 14A filed", "2026-03-02: SC 13G filed",
    "2026-03-02: S-3 filed", "3 insider transaction reports (Form 4) filed in the window (direction not parsed)",
    "2026-03-02: 8-K - director/officer departure or appointment", "2026-03-02: 8-K - shareholder vote results",
    "2026-03-02: 8-K - other events", "2026-03-02: 8-K", "2026-03-02: 4 - statement of changes in beneficial ownership",
    "2026-03-02: SC 13G/A - beneficial ownership report",
])
def test_a_routine_filing_cannot_carry_a_bullish_events_view(filing):
    """Regression (gate bypass, round 3): the events analyst could pass the gate by citing the
    routine 10-Q (or the Form 4 count) that every sheet carries. A filing that only says
    something was filed is not directional support."""
    fs = sheet(tiers=("C",), events=(filing,))
    for v in (view(0.9, ("positive", ["AAA.filing_1"]), excess=8.0),
              view(0.9, ("positive", ["AAA.filing_1"]), ("negative", ["AAA.news_1"]), excess=8.0)):
        out, note = tier_gate(ViewRecord(agent="events", model="m", view=v), fs)
        assert out.view.prob_outperform == 0.5 and note, filing


@pytest.mark.parametrize("filing", [
    "2026-03-02: 8-K - results of operations (earnings release)", "2026-03-02: 8-K - results of operations",
    "2026-03-02: 8-K - entry into a material agreement", "2026-03-02: 8-K - completion of acquisition or disposition",
    "2026-03-02: 8-K - director/officer departure or appointment; results of operations (earnings release)",
    "2026-03-02: 8-K - item 2.02", "2026-03-02: 8-K/A - results of operations (earnings release)",
    "2026-03-02: SC 13D filed",
])
def test_an_informative_filing_can_carry_a_bullish_events_view(filing):
    from aifirm.agents.tier_gate import informative_filing
    assert informative_filing(filing)
    fs = sheet(tiers=("C",), events=(filing,))
    rec = ViewRecord(agent="events", model="m", view=view(0.7, ("positive", ["AAA.filing_1"])))
    assert tier_gate(rec, fs) == (rec, "")


@pytest.mark.parametrize("cited", [["AAA.NEWS_9", "AAA.ret_3m"], ["aaa.News_9"], ["*", "AAA.ret_3m"],
                                   ["AAA.*", "AAA.ret_3m"], ["n*_3", "AAA.ret_3m"], ["AAA.<id>", "AAA.ret_3m"]])
def test_forged_ids_in_any_case_and_wildcards_matching_a_rumor_taint(cited):
    """Regression: an upper-case forged news ID not on the sheet, and a wildcard that also
    names the tier-C event, used to pass beside an admissible fact."""
    fs = sheet()
    out, _ = tier_gate(view(0.8, ("positive", cited), ("positive", ["AAA.ret_3m"])), fs)
    assert out.prob_outperform == 0.5
    clean = sheet(tiers=("A", "B"))                 # no rumor on the sheet: a wildcard names nothing unverified
    v = view(0.8, ("positive", ["*", "AAA.ret_3m"]))
    assert tier_gate(v, clean) == (v, "")


@pytest.mark.parametrize("cited", [["AAA.news_1"], ["AAA.filing_1"], ["AAA.ret_3m"], ["MACRO.DGS10"]])
def test_an_item_mixing_tier_c_news_with_an_admissible_fact_does_not_count(cited):
    """Regression (gate bypass): an item citing [AAA.news_3 (tier C), <anything admissible>] rests
    partly on the rumor; adding a fact beside it used to launder the rumor into support."""
    fs = sheet()
    out, note = tier_gate(view(0.7, ("positive", ["AAA.news_3", *cited])), fs)
    assert out.prob_outperform == 0.5 and out.expected_excess_return_pct == 0.0 and "capped" in note


@pytest.mark.parametrize("cited", [["AAA.price"], ["AAA.vol_20d"], ["AAA.vol_60d"], ["AAA.adv_20d"],
                                   ["MACRO.VIXCLS"], ["MKT.SPY_vol_20d"], ["AAA.price", "AAA.vol_20d"]])
def test_context_levels_cannot_carry_a_bullish_view(cited):
    """Regression (gate bypass): a price or volatility level says nothing about direction, so a
    positive item citing only such facts is not support."""
    fs = sheet()
    for fid in ("AAA.price", "AAA.vol_20d", "AAA.vol_60d", "AAA.adv_20d", "MKT.SPY_vol_20d"):
        fs.add(Fact(fid, fid, 10.0, "x", AS_OF, "prices"))
    out, _ = tier_gate(view(0.92, ("positive", ["AAA.news_3"]), ("positive", cited)), fs)
    assert out.prob_outperform == 0.5


def test_events_analyst_needs_a_filing_or_a_trusted_news_event():
    """Regression (gate bypass): the events analyst always sees price, ret_1m and vol_20d, so a
    rumor-based bullish view could cite them; its support must be an admissible event."""
    fs = sheet()
    fs.add(Fact("AAA.ret_1m", "Return, 1 month", 4.0, "%", AS_OF, "prices"))
    for cited, passes in ((["AAA.ret_1m"], False), (["AAA.ret_3m"], False), (["AAA.filing_1"], True),
                          (["AAA.news_2"], True), (["AAA.ret_1m", "AAA.filing_1"], True)):
        rec = ViewRecord(agent="events", model="m", view=view(0.9, ("negative", ["AAA.news_3"]), ("positive", cited)))
        out, note = tier_gate(rec, fs)
        assert (out is rec and note == "") is passes, cited
        assert tier_gate(rec.view, fs, role="events")[0].prob_outperform == (0.9 if passes else 0.5)
    tech = ViewRecord(agent="technical", model="m", view=view(0.6, ("positive", ["AAA.ret_3m"])))
    assert tier_gate(tech, fs)[0] is tech                              # other roles: directional facts count


def test_reviewer_scenario_tier_c_takeover_rumor_plus_vol_level_is_capped():
    """The reported failure: p=0.92 on a tier-C takeover blog, one positive item citing
    [news (tier C), vol_20d]. It passed before; now it is capped at 0.5."""
    fs = sheet(tiers=("C",))
    fs.add(Fact("AAA.vol_20d", "Realized volatility, 20 days", 30.0, "%", AS_OF, "prices"))
    rec = ViewRecord(agent="events", model="m", view=view(0.92, ("positive", ["AAA.news_1", "AAA.vol_20d"])))
    out, note = tier_gate(rec, fs)
    assert out.view.prob_outperform == 0.5 and out.view.expected_excess_return_pct == 0.0 and note


def test_a_forged_news_id_not_on_the_sheet_taints_its_item():
    fs = sheet()
    out, _ = tier_gate(view(0.7, ("positive", ["AAA.news_42", "AAA.ret_3m"])), fs)
    assert out.prob_outperform == 0.5
    out, _ = tier_gate(view(0.7, ("positive", ["AAA.not_a_fact", "AAA.ret_3m"])), fs)   # unknown non-event ID
    assert out.prob_outperform == 0.7


def test_positive_excess_at_one_half_is_bullish_too():
    """Regression: a p=0.50 view with +15% excess resting on tier-C news fed the pooled excess
    (and the Kelly size in pool mode) untouched; its excess is now capped at 0."""
    fs = sheet()
    out, note = tier_gate(view(0.5, ("positive", ["AAA.news_3"]), excess=15.0), fs)
    assert out.prob_outperform == 0.5 and out.expected_excess_return_pct == 0.0 and note
    ok = view(0.5, ("positive", ["AAA.ret_3m"]), excess=15.0)
    assert tier_gate(ok, fs)[0] is ok
    rec = ViewRecord(agent="macro", model="m", view=view(0.45, ("negative", ["AAA.ret_3m"]), excess=-1.0),
                     revised_prob=0.5, revised_excess_pct=12.0)
    out, _ = tier_gate(rec, fs)
    assert out.revised_excess_pct == 0.0 and out.revised_prob == 0.5


def test_pool_mode_excess_is_not_raised_by_gated_views():
    """Gate + pool: two supported views (+1%) and two p=0.50 / +15% views on tier-C news; the
    pooled excess must not jump to +8% on the unverified ones."""
    from aifirm.portfolio.aggregate import aggregate_views
    fs = sheet()
    good = [ViewRecord(agent=a, model="m", view=view(0.55, ("positive", ["AAA.ret_3m"]), excess=1.0))
            for a in ("technical", "fundamental")]
    rumor = [ViewRecord(agent=a, model="m", view=view(0.5, ("positive", ["AAA.news_3"]), excess=15.0))
             for a in ("events", "macro")]
    gated, notes = apply_tier_gate([*good, *rumor], fs)
    agg = aggregate_views("AAA", gated, trim=0.2)
    assert len(notes) == 2 and agg.expected_excess_return_pct <= 1.0


def test_a_raised_revision_needs_admissible_new_evidence():
    """Regression (revision hole): once the sealed view had admissible support, the revised
    probability could rise on tier-C content spread by the challenge round."""
    fs = sheet()
    sealed = view(0.6, ("positive", ["AAA.ret_3m"]), excess=1.0)
    rumor = ViewRecord(agent="technical", model="m", view=sealed, revised_prob=0.8, revised_excess_pct=6.0,
                       revision_fact_ids=["AAA.news_3"])
    out, note = tier_gate(rumor, fs)
    assert (out.view.prob_outperform, out.revised_prob, out.revised_excess_pct) == (0.6, 0.6, 1.0)
    assert "revision capped" in note
    mixed = rumor.model_copy(update={"revision_fact_ids": ["AAA.news_3", "AAA.ret_3m"]})
    assert tier_gate(mixed, fs)[0].revised_prob == 0.6
    none = rumor.model_copy(update={"revision_fact_ids": []})
    assert tier_gate(none, fs)[0].revised_prob == 0.6
    for ids in (["AAA.ret_3m"], ["AAA.news_2"], ["AAA.filing_1"]):
        ok = rumor.model_copy(update={"revision_fact_ids": ids})
        assert tier_gate(ok, fs) == (ok, "")
    lower = rumor.model_copy(update={"revised_prob": 0.55, "revised_excess_pct": 0.5, "revision_fact_ids": []})
    assert tier_gate(lower, fs) == (lower, "")                          # a revision down is never gated


@pytest.mark.parametrize("challenge_ids, capped", [(["AAA.news_3"], True), (["news_3", "AAA.ret_3m"], True),
                                                   (["AAA.NEWS_7"], True), (["AAA.news_1"], False),
                                                   (["AAA.ret_3m"], False), ([], False)])
def test_a_raise_answering_a_challenge_that_leans_on_unverified_news_is_capped(challenge_ids, capped):
    """Regression (revision hole, round 3): the challenger cites a tier-C takeover report and
    the reviser raises to 0.9 citing only a fact from its own slice. Whatever the reviser
    cites, a raise answering a challenge that leaned on unverified news is capped."""
    fs = sheet()
    rec = ViewRecord(agent="technical", model="m", view=view(0.53, ("positive", ["AAA.ret_3m"]), excess=1.0),
                     revised_prob=0.9, revised_excess_pct=8.0, revision_fact_ids=["AAA.ret_3m"],
                     challenge_fact_ids=challenge_ids)
    out, note = tier_gate(rec, fs)
    assert (out.revised_prob, out.revised_excess_pct) == ((0.53, 1.0) if capped else (0.9, 8.0))
    assert ("the challenge it answered cites unverified news" in note) is capped
    down = rec.model_copy(update={"revised_prob": 0.4, "revised_excess_pct": -1.0})
    assert tier_gate(down, fs) == (down, "")                             # a revision down is never gated


@pytest.mark.parametrize("rumor_direction, capped", [("positive", True), ("neutral", True), ("negative", False)])
def test_a_reviser_whose_own_sealed_view_leaned_on_a_rumor_cannot_raise(rumor_direction, capped):
    """Regression (revision hole, round 4): the events analyst's sealed view cited a tier-C
    rumor and was capped, but after a clean challenge a token revision citing only the
    earnings 8-K from its own slice restored the bullish view (p_final 0.478 -> 0.525,
    weight 0 -> 2.5%). A reviser that has already leaned on unverified news cannot raise."""
    fs = sheet()
    sealed = view(0.92, (rumor_direction, ["AAA.news_3"]), ("positive", ["AAA.filing_1"]), excess=12.0)
    rec = ViewRecord(agent="events", model="m", view=sealed, revised_prob=0.9, revised_excess_pct=10.0,
                     revision_fact_ids=["AAA.filing_1"], challenge_fact_ids=["AAA.ret_3m"])
    out, note = tier_gate(rec, fs)
    if capped:
        assert (out.view.prob_outperform, out.revised_prob, out.revised_excess_pct) == (0.5, 0.5, 0.0)
        assert "own sealed view cites unverified news" in note
    else:                                    # the rumor was cited only as a risk: the revision stands
        assert (out.revised_prob, out.revised_excess_pct) == (0.9, 10.0) and note == ""


@pytest.mark.parametrize("items", [
    (("negative", ["AAA.ret_3m"]), ("positive", ["AAA.news_3"])),     # admissible fact only on the other side
    (("neutral", ["AAA.ret_3m"]), ("positive", ["AAA.news_3"])),      # context does not carry a bullish view
    (("positive", ["AAA.unknown"]),),                                 # an ID not on the sheet never counts
    (("positive", []),),                                              # nothing cited
])
def test_support_must_be_positive_and_on_the_sheet(items):
    out, note = tier_gate(view(0.65, *items), sheet())
    assert out.prob_outperform == 0.5 and note


def test_news_fact_without_metadata_counts_as_tier_c():
    fs = sheet()
    fs.add(Fact("AAA.news_9", "News event (tier A, earnings, positive, high materiality)", "forged", "event", AS_OF, "?"))
    assert fs.news_tier("AAA.news_9") == "C"
    out, _ = tier_gate(view(0.7, ("positive", ["AAA.news_9"])), fs)
    assert out.prob_outperform == 0.5


def test_bearish_neutral_and_abstaining_views_are_never_changed():
    fs = sheet()
    for v in (view(0.4, ("positive", ["AAA.news_3"]), excess=-1.0), view(0.5, excess=0.0),
              view(0.5, abstain=True), view(0.3, excess=-2.0)):
        out, note = tier_gate(v, fs)
        assert out is v and note == ""


def test_gate_never_raises_a_probability():
    fs = sheet()
    for p in (0.02, 0.2, 0.45, 0.5, 0.55, 0.9, 0.98):
        for items in ((("positive", ["AAA.news_3"]),), (("positive", ["AAA.ret_3m"]),), ()):
            out, _ = tier_gate(view(p, *items), fs)
            assert out.prob_outperform <= p


def test_sheet_without_news_content_is_not_gated():
    fs = FactSheet(ticker="AAA", as_of=AS_OF, events=["2026-03-02: 8-K - results of operations"])
    fs.add(Fact("AAA.ret_3m", "Return, 3 months", 5.0, "%", AS_OF, "prices"))
    v = view(0.7, ("positive", []))                         # a qualitative filing point (no IDs to cite)
    assert tier_gate(v, fs) == (v, "")


def test_raw_unquarantined_headlines_activate_the_gate():
    fs = FactSheet(ticker="AAA", as_of=AS_OF, headlines=[Headline("Alpha to triple", "Blog", f"{AS_OF}T14:00:00Z")])
    fs.add(Fact("AAA.ret_3m", "Return, 3 months", 5.0, "%", AS_OF, "prices"))
    assert tier_gate(view(0.7, ("positive", [])), fs)[0].prob_outperform == 0.5
    assert tier_gate(view(0.7, ("positive", ["AAA.ret_3m"])), fs)[1] == ""


def test_view_record_caps_sealed_and_revised_values():
    fs = sheet()
    rec = ViewRecord(agent="events", model="m", view=view(0.75, ("positive", ["AAA.news_3"])),
                     revised_prob=0.7, revised_excess_pct=1.5)
    out, note = tier_gate(rec, fs)
    assert (out.view.prob_outperform, out.revised_prob, out.revised_excess_pct) == (0.5, 0.5, 0.0)
    assert note.startswith("tier gate: events view on AAA") and rec.view.prob_outperform == 0.75
    flipped = ViewRecord(agent="events", model="m", view=view(0.45, ("negative", ["AAA.ret_3m"]), excess=-1.0),
                         revised_prob=0.6, revised_excess_pct=0.5)
    out, _ = tier_gate(flipped, fs)     # a revision above 0.5 needs admissible support in its new evidence
    assert out.revised_prob == 0.5 and out.revised_excess_pct == 0.0 and out.view.prob_outperform == 0.45


def test_rumor_from_a_reputable_outlet_cannot_support():
    """The reader forces rumors to tier C; the gate then treats them as unverified."""
    fs = sheet(tiers=("C",))
    out, _ = tier_gate(view(0.7, ("positive", ["AAA.news_1"])), fs)
    assert out.prob_outperform == 0.5


def test_apply_tier_gate_keeps_order_and_reports_capped_views():
    fs = sheet()
    views = [ViewRecord(agent=a, model="m", view=view(p, ("positive", ids)))
             for a, p, ids in (("events", 0.8, ["AAA.news_3"]), ("technical", 0.6, ["AAA.ret_3m"]),
                               ("macro", 0.4, ["AAA.news_3"]))]
    views[2].view.expected_excess_return_pct = -1.0                       # bearish on both inputs
    out, notes = apply_tier_gate(views, fs)
    assert [v.agent for v in out] == ["events", "technical", "macro"]
    assert [v.view.prob_outperform for v in out] == [0.5, 0.6, 0.4] and len(notes) == 1
