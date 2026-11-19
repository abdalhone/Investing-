"""Tier gate: unverified news can never raise exposure (RESEARCH 5.5 item 3, 9.3 item 3).

Deterministic code, applied to each analyst view before pooling. It caps a *bullish* view
at 0.5 (and its expected excess at 0) unless the view rests on admissible evidence and not
on unverified news. It can only lower a probability or an excess, never raise one, so a
bearish or neutral view passes unchanged (an unverified *negative* report may still make
the firm more cautious; that is the asymmetric, fail-closed choice).

Bullish
-------
A sealed view is bullish when it does not abstain and either ``prob_outperform > 0.5`` or
``expected_excess_return_pct > 0``: a view at 0.5 with a large positive excess (a skewed
rumor bull case) feeds the pooled excess, and therefore position size, just as a high
probability does. A revision is bullish when ``revised_prob > 0.5`` or
``revised_excess_pct > 0``.

What counts as support for a sealed view
----------------------------------------
The question is what the view would be *without* the unverified news. Taint is judged on
the **whole view**, not item by item: a view that leans on a rumor anywhere cannot show
that its bullishness comes from somewhere else, because the analyst can always put the
rumor and an unrelated fact in two separate points. A bullish view passes only if

1. **no** evidence item whose direction is ``"positive"`` or ``"neutral"`` cites an
   inadmissible event (:func:`taints_support`): a tier-C news fact, a news fact without
   metadata, an event-like ID that is not on the sheet (whatever its letter case), or a
   wildcard citation that matches a tier-C news fact. Only an item arguing *against*
   outperformance (direction ``"negative"``) may cite unverified news without capping a
   bullish view: the view is bullish despite that report, not because of it; and
2. at least one ``"positive"`` evidence item cites an **admissible** fact
   (:func:`admissible_support`) that resolves
   (:func:`aifirm.agents.grounding.resolve_fact_id`) to a fact on the sheet and is

   * an **informative filing event** ``TICKER.filing_N`` (:func:`informative_filing`): an
     8-K that reports results (item 2.02), a material agreement (1.01) or a completed
     acquisition or disposition (2.01), or an activist stake (SC 13D). A routine filing
     (a 10-Q or 10-K, a proxy, a Form 4 count, an officer change, a passive 13G) says that
     something was filed, not which way the stock should move, so it cannot carry a
     bullish view (an SEC filing never taints either: it is tier A);
   * a **news event** ``TICKER.news_N`` whose code-assigned tier is A or B; or
   * for every role except ``events``, a **directional non-event fact**: price, trend,
     risk, fundamental, valuation, macro or market facts computed by the data desk, but
     not a pure level or scale (:data:`CONTEXT_FACT_KEYS` and :data:`CONTEXT_FACT_IDS`:
     the share price, volatility levels, VIX, liquidity, share count, market cap). A
     volatility level says how wide the outcomes are, not which way they lean, so it
     cannot carry a bullish view on its own.

   The ``events`` analyst (:data:`EVENT_ROLES`) judges events: its only non-event facts
   are the price, the 1-month return and the 20-day volatility, which any rumor-based
   view could cite. Its bullish view must therefore rest on an informative filing event
   or a tier A/B news event.

What counts as support for a revision
-------------------------------------
After a challenge round the revised values can rise on content the challenge spread
(including tier-C news cited by the challenger), even when the sealed view was well
supported, and even when the reviser cites only a fact from its own slice. A bullish
revision that raises the probability or the excess above the gated sealed values passes
only if

* the reviser's own sealed view carried no unverified news: no positive or neutral item of
  the sealed view taints (the reviser has already read the rumor, so a token revision that
  cites some other admissible fact from its slice could still rest on it); and
* the challenge it answered carried no unverified news: none of the IDs cited by the
  challenge's positive or neutral evidence items (``ViewRecord.challenge_fact_ids``,
  stored by the committee for every reviser) taints; and
* the IDs the reviser cited as new evidence (``ViewRecord.revision_fact_ids``, stored by
  the committee when it accepts a revision) contain admissible support and no inadmissible
  event, under the same rules as a sealed view (with the reviser's role).

Otherwise the revised probability is capped at ``max(gated sealed probability, 0.5)`` and
the revised excess at ``max(gated sealed excess, 0)``: the revision may stay where the
supported sealed view was, never above it.

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
import re
from collections.abc import Iterable
from typing import TypeVar

from ..data.factsheet import (FILING_FACT_PREFIX, NEWS_FACT_PREFIX, FactSheet, fact_key, is_event_fact_id,
                              is_filing_fact_id, is_news_fact_id)
from ..data.fundamentals import EIGHT_K_ITEMS
from ..schemas import AnalystView, EvidenceItem, ViewRecord
from .grounding import normalize_fact_id, resolve_fact_id

ADMISSIBLE_NEWS_TIERS = frozenset({"A", "B"})
NEUTRAL = 0.5
# Ticker facts that state a level or scale, not a direction: they cannot carry a bullish view.
CONTEXT_FACT_KEYS = frozenset({"price", "vol_20d", "vol_60d", "adv_20d", "shares_out", "market_cap"})
# Market-wide level facts (fnmatch patterns): the VIX level and the benchmark's realised volatility.
CONTEXT_FACT_IDS = ("MACRO.VIXCLS", "MKT.*_vol_20d")
# Roles whose bullish view must rest on an admissible *event* (an informative filing or tier A/B news).
EVENT_ROLES = frozenset({"events"})
# Filing events that carry new information a bullish view may rest on (module docstring, rule 2):
# 8-K items for results, a material agreement and a completed acquisition or disposition, and
# an activist (13D) stake. Everything else on the sheet (periodic reports, proxies, Form 4
# counts, officer changes, passive 13G stakes, registration statements) is routine or negative.
INFORMATIVE_8K_ITEMS = frozenset({"1.01", "2.01", "2.02"})
INFORMATIVE_FILING_FORMS = frozenset({"SC 13D", "SC 13D/A", "13D", "13D/A"})
# Evidence directions whose tainted citations cap a bullish view (only "negative" may cite a rumor).
TAINTING_DIRECTIONS = frozenset({"positive", "neutral"})

V = TypeVar("V", AnalystView, ViewRecord)

_DATE_PREFIX = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*:\s*")
_ITEM_NUMBER = re.compile(r"^item\s+(\d{1,2}\.\d{2})\b")


def _desc_key(text: str) -> str:
    """An 8-K item description without its parenthetical gloss, lower-case."""
    return text.split("(", 1)[0].strip().lower()


_INFORMATIVE_8K_DESCRIPTIONS = frozenset(_desc_key(EIGHT_K_ITEMS[i]) for i in INFORMATIVE_8K_ITEMS)


def informative_filing(text: str) -> bool:
    """True if a filing event, as the sheet shows it (``filing_events`` text such as
    "2026-05-28: 8-K - results of operations (earnings release); director/officer departure
    or appointment" or "2026-05-28: SC 13D filed"), is an informative filing (module
    docstring, rule 2): an 8-K with at least one item in :data:`INFORMATIVE_8K_ITEMS`, or a
    form in :data:`INFORMATIVE_FILING_FORMS`. Anything unparsed is not informative."""
    rest = _DATE_PREFIX.sub("", str(text), count=1).strip()
    m = re.match(r"^8-K(?:/A)?\s+-\s+(.*)$", rest)
    if m:
        for part in m.group(1).split(";"):
            key = _desc_key(part)
            num = _ITEM_NUMBER.match(key)
            if key in _INFORMATIVE_8K_DESCRIPTIONS or (num and num.group(1) in INFORMATIVE_8K_ITEMS):
                return True
        return False
    if rest.endswith(" filed"):
        return rest[: -len(" filed")].strip().upper() in INFORMATIVE_FILING_FORMS
    return False


def is_context_fact_id(fid: str) -> bool:
    """True for a resolved fact ID that states a level or scale (module docstring, rule 2)."""
    if fid.startswith(("MACRO.", "MKT.")):
        return any(fnmatch.fnmatchcase(fid, pat) for pat in CONTEXT_FACT_IDS)
    return fact_key(fid) in CONTEXT_FACT_KEYS


def _unverified_news(fid: str, fs: FactSheet) -> bool:
    """A news event on the sheet whose tier is not A/B (no metadata counts as tier C)."""
    return is_news_fact_id(fid) and fs.news_tier(fid) not in ADMISSIBLE_NEWS_TIERS


def _supporting_event(fid: str, fs: FactSheet) -> bool:
    if is_filing_fact_id(fid):
        fact = fs.get(fid)
        return fact is not None and informative_filing(str(fact.value))
    return is_news_fact_id(fid) and fs.news_tier(fid) in ADMISSIBLE_NEWS_TIERS


def admissible_support(fact_id: str, fs: FactSheet, role: str | None = None) -> bool:
    """True if a cited ``fact_id`` may support a bullish view on ``fs`` (module docstring,
    rule 2): it resolves to a fact on the sheet and is an informative filing event, a tier
    A/B news event, or (unless ``role`` is in :data:`EVENT_ROLES`) a non-event fact that is
    not a context level (:func:`is_context_fact_id`)."""
    fid = resolve_fact_id(fact_id, fs.ids(), fs.ticker)
    if not fid or fs.get(fid) is None:
        return False
    if is_event_fact_id(fid):
        return _supporting_event(fid, fs)
    if role in EVENT_ROLES:
        return False
    return not is_context_fact_id(fid)


def _event_like(fact_id: str) -> bool:
    """An ID shaped like a news or filing event, in any letter case ("AAA.NEWS_9", "news_*")."""
    if "." not in fact_id or fact_id.upper().startswith(("MACRO.", "MKT.")):
        return False
    return fact_key(fact_id).lower().startswith((NEWS_FACT_PREFIX, FILING_FACT_PREFIX))


def taints_support(fact_id: str, fs: FactSheet) -> bool:
    """True if citing ``fact_id`` makes a view (or a revision) rest partly on unverified
    news (module docstring, rule 1): it resolves to a news event on the sheet whose tier is
    not A/B; or it is not on the sheet and is either event-like in any letter case (a forged
    or hallucinated news ID) or a wildcard pattern that matches an unverified news event."""
    nid = normalize_fact_id(fact_id)
    fid = resolve_fact_id(fact_id, fs.ids(), fs.ticker)
    if fid and fs.get(fid) is not None:
        return _unverified_news(fid, fs)
    if not nid:
        return False
    probe = nid if "." in nid else f"{fs.ticker}.{nid}"
    if _event_like(probe):
        return True
    if any(ch in nid for ch in "*?[<{"):        # a pattern: tainting if it could name a rumor
        pattern = re.sub(r"<[^>]*>|\{[^}]*\}", "*", nid).lower()
        pats = {pattern} if "." in pattern else {pattern, f"{fs.ticker.lower()}.{pattern}"}
        return any(_unverified_news(k, fs) and any(fnmatch.fnmatchcase(k.lower(), p) for p in pats)
                   for k in fs.ids())
    return False


def _tainted(items: Iterable[EvidenceItem], fs: FactSheet) -> bool:
    """True if a positive or neutral item among ``items`` cites unverified news (rule 1)."""
    return any(taints_support(fid, fs) for it in items if it.direction in TAINTING_DIRECTIONS
               for fid in it.fact_ids)


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
    """The admissible fact IDs cited by the view's positive evidence items (resolved,
    deduplicated), or [] when any positive or neutral item cites unverified news (module
    docstring). ``role`` is the analyst's role name (the ``events`` role needs an
    admissible event)."""
    if _tainted(view.evidence, fs):
        return []
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

    revision_why = ""
    if is_rec and (view.revised_prob is not None or view.revised_excess_pct is not None):
        rp, rx = view.revised_prob, view.revised_excess_pct
        raises = (rp is not None and rp > gated_p) or (rx is not None and rx > gated_x)
        if _is_bullish(rp, rx) and raises:
            if _tainted(av.evidence, fs):
                revision_why = "the reviser's own sealed view cites unverified news as supporting evidence"
            elif any(taints_support(fid, fs) for fid in view.challenge_fact_ids):
                revision_why = "the challenge it answered cites unverified news as supporting evidence"
            elif not _supported_by(view.revision_fact_ids, fs, role):
                revision_why = "its new evidence has no admissible support"
    if not cap_sealed and not revision_why:
        return view, ""

    out = view.model_copy(deep=True)
    target = out.view if isinstance(out, ViewRecord) else out
    before = target.prob_outperform
    target.prob_outperform, target.expected_excess_return_pct = gated_p, gated_x
    notes = []
    if cap_sealed:
        notes.append(f"view capped from {before:.3f} to {gated_p:.3f}: it cites unverified news as supporting or "
                     "neutral evidence, or no positive evidence item rests on "
                     + ("an informative filing event or a tier A/B news event" if role in EVENT_ROLES else
                        "directional facts, informative filing events or tier A/B news events"))
    if revision_why and isinstance(out, ViewRecord):
        p_cap, x_cap = max(gated_p, NEUTRAL), max(gated_x, 0.0)
        rb = out.revised_prob
        if out.revised_prob is not None:
            out.revised_prob = min(out.revised_prob, p_cap)
        if out.revised_excess_pct is not None:
            out.revised_excess_pct = min(out.revised_excess_pct, x_cap)
        notes.append(f"revision capped from {rb if rb is not None else float('nan'):.3f} to "
                     f"{out.revised_prob if out.revised_prob is not None else float('nan'):.3f}: {revision_why}")
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
