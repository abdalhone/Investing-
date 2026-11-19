"""Tier gate: unverified news can never raise exposure (RESEARCH 5.5 item 3, 9.3 item 3).

Deterministic code, applied to each analyst view before pooling. It caps a *bullish* view
at 0.5 unless the view rests on at least one piece of admissible evidence. It can only
lower a probability, never raise one, so a bearish or neutral view passes unchanged
(an unverified *negative* report may still make the firm more cautious; that is the
asymmetric, fail-closed choice).

What counts as support
----------------------
A bullish view (``prob_outperform > 0.5``, or a revised probability above 0.5) passes if
at least one of its evidence items

1. has direction ``"positive"`` (an item arguing against outperformance, or neutral
   context such as a volatility level, cannot carry a bullish view), and
2. cites a fact ID that resolves (:func:`aifirm.agents.grounding.resolve_fact_id`) to a
   fact on the given fact sheet (unknown IDs never count), which is
3. one of:

   * a **non-event fact**: price, trend and risk, fundamentals and valuation, macro or
     market facts (all computed by the data desk from point-in-time data);
   * a **filing event** ``TICKER.filing_N`` (an SEC EDGAR filing: tier A); or
   * a **news event** ``TICKER.news_N`` whose code-assigned tier is A or B. A news fact
     whose metadata is missing counts as tier C (unproven provenance).

When the gate applies
---------------------
Only on a fact sheet that carries news content: raw headlines, or reader-extracted news
facts (:meth:`FactSheet.has_news_content`). On a sheet without any news there is no
unverified text a view could rest on, and qualitative points about filing events (which
have no fact IDs unless news was attached) must not be penalised; the gate is then a
no-op. Once news is attached with ``attach_news_events(cite_filings=True)`` filing events
are citable, so a filing-based view can show its support.

When the gate caps a view, ``prob_outperform`` becomes 0.5 and a positive expected excess
return becomes 0 (both inputs of the sizing engine), and the same is done to the revised
values from a challenge round, if any. The view's evidence and scenarios are kept for the
record. The input object is never modified: a capped *copy* is returned with a note.
"""

from __future__ import annotations

from typing import TypeVar

from ..data.factsheet import FactSheet, is_event_fact_id, is_filing_fact_id, is_news_fact_id
from ..schemas import AnalystView, ViewRecord
from .grounding import resolve_fact_id

ADMISSIBLE_NEWS_TIERS = frozenset({"A", "B"})
NEUTRAL = 0.5

V = TypeVar("V", AnalystView, ViewRecord)


def admissible_support(fact_id: str, fs: FactSheet) -> bool:
    """True if a cited ``fact_id`` may support a bullish view on ``fs`` (module docstring, rule 3)."""
    fid = resolve_fact_id(fact_id, fs.ids(), fs.ticker)
    if not fid or fs.get(fid) is None:
        return False
    if not is_event_fact_id(fid):
        return True
    if is_filing_fact_id(fid):
        return True
    return is_news_fact_id(fid) and fs.news_tier(fid) in ADMISSIBLE_NEWS_TIERS


def supporting_fact_ids(view: AnalystView, fs: FactSheet) -> list[str]:
    """The admissible fact IDs cited by the view's positive evidence items (resolved, deduplicated)."""
    out: list[str] = []
    known = fs.ids()
    for item in view.evidence:
        if item.direction != "positive":
            continue
        for fid in item.fact_ids:
            rid = resolve_fact_id(fid, known, fs.ticker)
            if rid and rid not in out and admissible_support(rid, fs):
                out.append(rid)
    return out


def tier_gate(view: V, fs: FactSheet) -> tuple[V, str]:
    """Cap an unsupported bullish view at 0.5 (module docstring). Returns ``(view, note)``:
    the original object and ``""`` when nothing changes, else a capped deep copy and a note
    for the ledger. Accepts an :class:`AnalystView` or a :class:`ViewRecord` (whose sealed
    view and revised values are both checked)."""
    av = view.view if isinstance(view, ViewRecord) else view
    revised = view.revised_prob if isinstance(view, ViewRecord) else None
    bullish = (not av.abstain and av.prob_outperform > NEUTRAL) or (revised is not None and revised > NEUTRAL)
    if not bullish or not fs.has_news_content():
        return view, ""
    if supporting_fact_ids(av, fs):
        return view, ""
    out = view.model_copy(deep=True)
    target = out.view if isinstance(out, ViewRecord) else out
    before = target.prob_outperform
    target.prob_outperform = min(target.prob_outperform, NEUTRAL)
    target.expected_excess_return_pct = min(target.expected_excess_return_pct, 0.0)
    if isinstance(out, ViewRecord):
        if out.revised_prob is not None:
            out.revised_prob = min(out.revised_prob, NEUTRAL)
        if out.revised_excess_pct is not None:
            out.revised_excess_pct = min(out.revised_excess_pct, 0.0)
    who = f"{out.agent} " if isinstance(out, ViewRecord) else ""
    note = (f"tier gate: {who}view on {fs.ticker} capped from {before:.3f} to {target.prob_outperform:.3f}: no positive "
            "evidence item cites a non-news fact, a filing event or a tier A/B news event")
    return out, note


def apply_tier_gate(views: list[ViewRecord], fs: FactSheet) -> tuple[list[ViewRecord], list[str]]:
    """:func:`tier_gate` over a committee's views. Returns the (possibly capped) views in the
    same order and the notes of the views that were capped."""
    out, notes = [], []
    for v in views:
        gated, note = tier_gate(v, fs)
        out.append(gated)
        if note:
            notes.append(note)
    return out, notes
