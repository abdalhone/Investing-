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


@pytest.mark.parametrize("cited", [["AAA.news_1"], ["AAA.news_2"], ["AAA.filing_1"], ["AAA.ret_3m"], ["MACRO.VIXCLS"],
                                   ["news_2"], ["[AAA.ret_3m]"]])
def test_admissible_support_lets_a_bullish_view_through(cited):
    fs = sheet()
    v = view(0.7, ("positive", ["AAA.news_3", *cited]))
    out, note = tier_gate(v, fs)
    assert out is v and note == "" and supporting_fact_ids(v, fs)


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
    for v in (view(0.4, ("positive", ["AAA.news_3"]), excess=-1.0), view(0.5), view(0.5, abstain=True),
              view(0.3)):
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
    out, _ = tier_gate(flipped, fs)     # a revision above 0.5 needs the same admissible support
    assert out.revised_prob == 0.5 and out.view.prob_outperform == 0.45


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
    out, notes = apply_tier_gate(views, fs)
    assert [v.agent for v in out] == ["events", "technical", "macro"]
    assert [v.view.prob_outperform for v in out] == [0.5, 0.6, 0.4] and len(notes) == 1
