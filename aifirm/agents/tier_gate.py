"""Tier gate: unverified news can never raise exposure (RESEARCH 5.5 item 3, 9.3 item 3).

Deterministic code, applied to each analyst view before pooling. It caps a *bullish* view
at 0.5 (and its expected excess at 0) unless the view rests on admissible evidence that is
not tied to unverified news. It can only lower a probability or an excess, never raise
one, so a bearish or neutral view passes unchanged (an unverified *negative* report may
still make the firm more cautious; that is the asymmetric, fail-closed choice).

Bullish
-------
A sealed view is bullish when it does not abstain and either ``prob_outperform > 0.5`` or
``expected_excess_return_pct > 0``: a view at 0.5 with a large positive excess (a skewed
rumor bull case) feeds the pooled excess, and therefore position size, just as a high
probability does. A revision is bullish when ``revised_prob > 0.5`` or
``revised_excess_pct > 0``.

What counts as support for a sealed view
----------------------------------------
The question is what the view would be *without* the unverified news. A bullish view
passes if at least one of its evidence items

1. has direction ``"positive"`` (an item arguing against outperformance, or neutral
   context, cannot carry a bullish view);
2. is **untainted**: it cites no inadmissible event, i.e. no tier-C news fact, no news
   fact without metadata, and no event-like ID that is not on the sheet. A mixed item
   such as ``[AAA.news_3 (tier C), AAA.ret_3m]`` rests partly on the rumor, so it does not
   count; adding an unrelated fact to a rumor-based point never launders it; and
3. cites at least one **admissible** fact (:func:`admissible_support`) that resolves
   (:func:`aifirm.agents.grounding.resolve_fact_id`) to a fact on the sheet and is

   * a **filing event** ``TICKER.filing_N`` (an SEC EDGAR filing: tier A);
   * a **news event** ``TICKER.news_N`` whose code-assigned tier is A or B; or
   * for every role except ``events``, a **directional non-event fact**: price, trend,
     risk, fundamental, valuation, macro or market facts computed by the data desk, but
     not a pure level or scale (:data:`CONTEXT_FACT_KEYS` and :data:`CONTEXT_FACT_IDS`:
     the share price, volatility levels, VIX, liquidity, share count, market cap). A
     volatility level says how wide the outcomes are, not which way they lean, so it
     cannot carry a bullish view on its own.

   The ``events`` analyst (:data:`EVENT_ROLES`) judges events: its only non-event facts
   are the price, the 1-month return and the 20-day volatility, which any rumor-based
   view could cite. Its bullish view must therefore rest on a filing event or a tier A/B
   news event.

What counts as support for a revision
-------------------------------------
After a challenge round the revised values can rise on content the challenge spread
(including tier-C news cited by the challenger), even when the sealed view was well
supported. A bullish revision that raises the probability or the excess above the gated
sealed values passes only if the IDs the reviser cited as new evidence
(``ViewRecord.revision_fact_ids``, stored by the committee when it accepts a revision)
contain admissible support and no inadmissible event, under the same rules as an evidence
item (with the reviser's role). Otherwise the revised probability is capped at
``max(gated sealed probability, 0.5)`` and the revised excess at ``max(gated sealed
excess, 0)``: the revision may stay where the supported sealed view was, never above it.

When the gate applies
---------------------
Only on a fact sheet that carries news content: raw headlines, or reader-extracted news
facts (:meth:`FactSheet.has_news_content`). On a sheet without any news there is no
unverified text a view could rest on, and qualitative points about filing events (which
have no fact IDs unless news was attached) must not be penalised; the gate is then a
no-op. Once news is attached with ``attach_news_events(cite_filings=True)`` filing events
are citable, so a filing-based view can show its support.

A capped view keeps its evidence and scenarios for the record. The input object is never
modified: a capped *copy* is returned with a note. Downstream, a capped view at 0.5 is "no
information": in the bounded overlay it tilts the quant core by 0
(:func:`aifirm.portfolio.overlay.combine`).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from typing import TypeVar

from ..data.factsheet import FactSheet, fact_key, is_event_fact_id, is_filing_fact_id, is_news_fact_id
from ..schemas import AnalystView, ViewRecord
from .grounding import normalize_fact_id, resolve_fact_id

ADMISSIBLE_NEWS_TIERS = frozenset({"A", "B"})
NEUTRAL = 0.5
# Ticker facts that state a level or scale, not a direction: they cannot carry a bullish view.
CONTEXT_FACT_KEYS = frozenset({"price", "vol_20d", "vol_60d", "adv_20d", "shares_out", "market_cap"})
# Market-wide level facts (fnmatch patterns): the VIX level and the benchmark's realised volatility.
CONTEXT_FACT_IDS = ("MACRO.VIXCLS", "MKT.*_vol_20d")
# Roles whose bullish view must rest on an admissible *event* (a filing or tier A/B news).
EVENT_ROLES = frozenset({"events"})

V = TypeVar("V", AnalystView, ViewRecord)


def is_context_fact_id(fid: str) -> bool:
    """True for a resolved fact ID that states a level or scale (module docstring, rule 3)."""
    if fid.startswith(("MACRO.", "MKT.")):
        return any(fnmatch.fnmatchcase(fid, pat) for pat in CONTEXT_FACT_IDS)
    return fact_key(fid) in CONTEXT_FACT_KEYS


def _admissible_event(fid: str, fs: FactSheet) -> bool:
    return is_filing_fact_id(fid) or (is_news_fact_id(fid) and fs.news_tier(fid) in ADMISSIBLE_NEWS_TIERS)


def admissible_support(fact_id: str, fs: FactSheet, role: str | None = None) -> bool:
    """True if a cited ``fact_id`` may support a bullish view on ``fs`` (module docstring,
    rule 3): it resolves to a fact on the sheet and is a filing event, a tier A/B news
    event, or (unless ``role`` is in :data:`EVENT_ROLES`) a non-event fact that is not a
    context level (:func:`is_context_fact_id`)."""
    fid = resolve_fact_id(fact_id, fs.ids(), fs.ticker)
    if not fid or fs.get(fid) is None:
        return False
    if is_event_fact_id(fid):
        return _admissible_event(fid, fs)
    if role in EVENT_ROLES:
        return False
    return not is_context_fact_id(fid)


def taints_support(fact_id: str, fs: FactSheet) -> bool:
    """True if citing ``fact_id`` makes an evidence item (or a revision) rest partly on
    unverified news: it is an event ID (news or filing, after resolution) that is not on
    the sheet, or a news event whose tier is not A/B (module docstring, rule 2)."""
    nid = normalize_fact_id(fact_id)
    fid = resolve_fact_id(fact_id, fs.ids(), fs.ticker)
    if fid and fs.get(fid) is not None:
        return is_event_fact_id(fid) and not _admissible_event(fid, fs)
    # not on the sheet: an event-like citation (a forged or hallucinated news ID) is tainting
    probe = nid if "." in nid else f"{fs.ticker}.{nid}"
    return is_event_fact_id(probe)


def _supported_by(ids: Iterable[str], fs: FactSheet, role: str | None) -> list[str]:
    """The admissible IDs among ``ids`` (resolved, de-duplicated), or [] when any of them taints."""
    ids = list(ids)
    if any(taints_support(fid, fs) for fid in ids):
        return []
    known = fs.ids()
    out: list[str] = []
    for fid in ids:
        rid = resolve_fact_id(fid, known, fs.ticker)
        if rid and rid not in out and admissible_support(rid, fs, role):
            out.append(rid)
    return out


def supporting_fact_ids(view: AnalystView, fs: FactSheet, role: str | None = None) -> list[str]:
    """The admissible fact IDs cited by the view's positive, untainted evidence items
    (resolved, deduplicated; module docstring). ``role`` is the analyst's role name (the
    ``events`` role needs an admissible event)."""
    out: list[str] = []
    for item in view.evidence:
        if item.direction != "positive":
            continue
        for rid in _supported_by(item.fact_ids, fs, role):
            if rid not in out:
                out.append(rid)
    return out


def _is_bullish(p: float | None, x: float | None) -> bool:
    return (p is not None and p > NEUTRAL) or (x is not None and x > 0.0)


def tier_gate(view: V, fs: FactSheet, role: str | None = None) -> tuple[V, str]:
    """Cap an unsupported bullish view (module docstring). Returns ``(view, note)``: the
    original object and ``""`` when nothing changes, else a capped deep copy and a note for
    the ledger. Accepts an :class:`AnalystView` or a :class:`ViewRecord` (whose sealed view
    and revised values are both checked; the record's ``agent`` is its role unless ``role``
    is given)."""
    is_rec = isinstance(view, ViewRecord)
    av = view.view if is_rec else view
    role = role if role is not None else (view.agent if is_rec else None)
    if not fs.has_news_content():
        return view, ""
    sealed_bullish = not av.abstain and _is_bullish(av.prob_outperform, av.expected_excess_return_pct)
    cap_sealed = sealed_bullish and not supporting_fact_ids(av, fs, role)
    gated_p = min(av.prob_outperform, NEUTRAL) if cap_sealed else av.prob_outperform
    gated_x = min(av.expected_excess_return_pct, 0.0) if cap_sealed else av.expected_excess_return_pct

    cap_revision = False
    if is_rec and (view.revised_prob is not None or view.revised_excess_pct is not None):
        rp, rx = view.revised_prob, view.revised_excess_pct
        raises = (rp is not None and rp > gated_p) or (rx is not None and rx > gated_x)
        if _is_bullish(rp, rx) and raises and not _supported_by(view.revision_fact_ids, fs, role):
            cap_revision = True
    if not cap_sealed and not cap_revision:
        return view, ""

    out = view.model_copy(deep=True)
    target = out.view if isinstance(out, ViewRecord) else out
    before = target.prob_outperform
    target.prob_outperform, target.expected_excess_return_pct = gated_p, gated_x
    notes = []
    if cap_sealed:
        notes.append(f"view capped from {before:.3f} to {gated_p:.3f}: no positive evidence item rests only on "
                     + ("a filing event or a tier A/B news event" if role in EVENT_ROLES else
                        "directional facts, filing events or tier A/B news events") + " (tier-C news, context "
                     "levels and items mixing them do not count)")
    if cap_revision and isinstance(out, ViewRecord):
        p_cap, x_cap = max(gated_p, NEUTRAL), max(gated_x, 0.0)
        rb = out.revised_prob
        if out.revised_prob is not None:
            out.revised_prob = min(out.revised_prob, p_cap)
        if out.revised_excess_pct is not None:
            out.revised_excess_pct = min(out.revised_excess_pct, x_cap)
        notes.append(f"revision capped from {rb if rb is not None else float('nan'):.3f} to "
                     f"{out.revised_prob if out.revised_prob is not None else float('nan'):.3f}: its new evidence "
                     "has no admissible support")
    who = f"{out.agent} " if isinstance(out, ViewRecord) else ""
    return out, f"tier gate: {who}view on {fs.ticker}: " + "; ".join(notes)


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
