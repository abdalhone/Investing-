"""The investment committee for one ticker: sealed views -> pool -> challenge -> revisions -> memo.

Why the process looks like this (see docs/ARCHITECTURE.md):

* **Independent first, aggregate statistically (P4).** Analysts write sealed views in
  parallel and are pooled by a believability-weighted, trimmed log-odds mean. Free-form
  LLM debate tends to converge on the most confident voice; voting or pooling independent
  estimates is at least as accurate and much cheaper ("Debate or Vote", NeurIPS 2025).
* **Debate only on disagreement.** The challenge round runs only when the dispersion of
  views exceeds a threshold. Agreement among independent slices is informative as it is.
* **Authentic dissent (P6).** The challenger is the analyst whose own sealed view was most
  opposed to the pool, arguing what it actually believes. Assigned devil's advocates are
  weaker than genuine dissenters (Nemeth 2001).
* **Private, evidence-gated revisions.** Each other analyst re-estimates privately from
  its own slice, its own sealed view and the challenge, without seeing who said what, so
  there is nobody to defer to. A change that cites no fact from its slice or from the
  challenge is discarded: social pressure is not evidence.
* **The chair can only shrink the edge.** The IC memo produces a multiplier in [0, 1]; it
  cannot raise or reverse the pooled view (P1). Views are shown to the chair anonymised
  and in a deterministic random order, so the memo responds to arguments, not to names
  or to a fixed speaking order. Anonymous means anonymous: role titles, role strings
  ("analyst.technical") and self-descriptions ("as the macro analyst") are removed from
  every peer-written text shown to the chair, the challenger and the revisers.
* **Every name that could be held is reviewed.** A memo is required whenever the pooled
  probability clears the portfolio's holding threshold (``min_prob_to_hold``, or its
  mirror for shorts), whenever ``|p - 0.5| >= memo_min_edge``, and for every held name.
  With memos enabled, a name that was not reviewed gets an edge multiplier of 0: an
  unreviewed edge never opens a position. A chair "hold" on a name the book does not own
  means "do not buy", so it also gets 0.
* **Weights from track record (P7).** Each view is weighted by its agent's believability;
  ungrounded views count half (P2).
* **What reaches the pool is code-checked (RESEARCH 6.2, 9.3 item 3).** Each LLM analyst's
  sealed view is first passed through the tier gate (a bullish view that rests only on
  unverified news is capped at 0.5, :mod:`aifirm.agents.tier_gate`) and then through its
  agent's point-in-time calibration map (:mod:`aifirm.journal.calibration`). Pooling,
  challenger selection and the overlay use those copies; ``CommitteeResult.views`` keep
  the RAW stated probabilities, which are what the forecast store records (calibrators are
  refitted on raw data). The prompts shown to agents also quote raw views: an analyst is
  never told it said something it did not. The quant is neither gated nor calibrated.
* **Quant core plus a bounded LLM overlay (RESEARCH 6.6).** With
  ``process.combination = "overlay"`` (the default) the final probability is the quant's,
  tilted by the pooled LLM analysts' own edge (pooled P minus 0.5) by at most ``max_tilt``
  (:mod:`aifirm.portfolio.overlay`); with ``"pool"`` every analyst, quant included, is one
  member of the trimmed log-odds pool. Either way the challenge round is triggered by the
  dispersion of all members and the challenger is the most opposed LLM analyst.
* **Fail closed.** A failed analyst call is an abstention; a failed challenge leaves the
  sealed pool in place; a failed revision keeps the sealed view. A failed memo on a name
  the book does not own sets the edge multiplier to 0, so an unreviewed edge is never
  traded. A failed memo on a *held* name must not sell it either: an LLM outage is not
  information about the stock. The result is flagged ``hold_current=True``: the pipeline
  passes such names to ``construct_portfolio(keep_current=...)``, which keeps them at
  their current weight (the unreviewed view is neither sized up nor used to sell; only a
  hard limit can cut them) and warns. The edge multiplier stays 1.0, never 0, so a caller
  that ignores the flag does not sell on an LLM failure; such a caller re-sizes from an
  unreviewed view, which is why the flag must be honoured.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

from ..config import FirmConfig
from ..data.factsheet import FactSheet
from ..journal.calibration import Calibrator, calibrate_view
from ..llm.client import LLM, BudgetExceeded, LLMCallFailed
from ..portfolio.aggregate import aggregate_views
from ..portfolio.overlay import overlay_aggregate
from ..quant.signals import horizon_sigma_pct
from ..schemas import AggregatedView, Challenge, EvidenceItem, ICMemo, Revision, ViewRecord, clamp
from .analysts import (BLIND_ALIAS, EVIDENCE_RULES, NEUTRALITY, blind_factsheet, blind_ids_in_text, blind_text,
                       horizon_phrase, render_slice, run_analyst, served_model, shown_sheet, unblind_id)
from .grounding import resolve_fact_id
from .roles import ROLES
from .tier_gate import TAINTING_DIRECTIONS, tier_gate

QUANT_AGENT = "quant"
PROB_FLOOR = 0.02                       # same floor as schemas.sanitize_view
_MAX_CONSENSUS_POINTS = 10

# Prompt labels. The offline responders parse these, so they are defined once here.
POOLED_PROB_LABEL = "Pooled probability (believability-weighted)"
DISPERSION_LABEL = "Dispersion (std-dev of member probabilities)"
SEALED_PROB_LABEL = "Your sealed probability of outperforming"
SEALED_EXCESS_LABEL = "Your sealed expected excess return"
CHALLENGE_HEADER = "THE CHALLENGE"
HELD_LABEL = "Currently held"
MEMBER_SEALED_LABEL = "Sealed: P(outperform)"
FLAGGED_LABEL = "Grounding: FLAGGED"
ABSTAINED_LABEL = "Abstained:"
CHALLENGE_ROUND_LABEL = "Challenge round"

LogFn = Callable[[str, dict], None]
_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass
class CommitteeResult:
    ticker: str
    views: list[ViewRecord]
    aggregate: AggregatedView
    challenge: Challenge | None
    challenger: str | None
    memo: ICMemo | None
    # memo.conviction_multiplier clamped to [0, 1]; 0 if the memo says review (or avoid/sell in a
    # long-only book), or "hold" on a name the book does not own; 1.0 if memos are disabled; 0 for
    # an unreviewed name the book does not own; 1.0 (with hold_current) if a held name's memo failed.
    edge_multiplier: float
    # True when the chair was required but could not review a HELD name (the memo call failed):
    # keep the current weight this cycle instead of re-sizing or selling on an LLM failure.
    hold_current: bool = False
    # The pool of the LLM analysts alone (calibrated, tier-gated; revised after a challenge),
    # None when no LLM analyst sits on the committee. In overlay mode it is what tilts the quant.
    llm_aggregate: AggregatedView | None = None
    # The views as pooled: tier-gated and calibrated copies of ``views`` (same order). ``views``
    # keep the raw stated probabilities, which are what the forecast store records.
    pool_views: list[ViewRecord] = field(default_factory=list)
    tier_gate_notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- prompts

CHALLENGE_SYSTEM = f"""\
You are the dissenting analyst on an equity investment committee. Your own sealed, independent forecast disagreed most with the committee's pooled view, so you have been asked to make the case you actually hold. You were chosen for your genuine disagreement, not to play devil's advocate: argue your real view as strongly and as honestly as the facts allow.

WHY THIS MATTERS
After reading your challenge, every other analyst privately re-estimates their forecast. Your challenge improves the committee only if it brings facts or inferences the consensus missed or got wrong. Every forecast, including yours, is scored against the realised outcome.

WHAT TO WRITE
- strongest_counterargument: the single strongest argument against the pooled view, grounded in specific facts. Prefer evidence the consensus arguments do not address, and explain why it matters over the forecast horizon.
- evidence: 2 to 6 items, each citing fact IDs, with direction relative to the stock beating the benchmark.
- overlooked_risks_or_opportunities: what the consensus may be under-weighting.
- questions_for_the_committee: questions whose answers would settle the disagreement, ideally naming the fact that would answer each.

You now see the full fact sheet, not just your original slice. If the full sheet convinces you that your sealed view was mistaken, say so plainly in strongest_counterargument; honesty is worth more than winning.

RULES
- Argue against the inference, never against people. You do not know who the other analysts are; do not speculate about them.
- Your challenge is shown to the committee anonymously. Do not state your role, title or data desk, or which analyst you are (for example, never write "as the technical analyst"); let the facts speak.
- Never mention position sizes, weights or trades.
{EVIDENCE_RULES}

NEUTRALITY
{NEUTRALITY}"""

REVISION_SYSTEM = f"""\
You are an analyst on an equity investment committee. You wrote a sealed forecast from your own slice of the fact sheet. Another committee member, whose identity is withheld, has written a challenge to the committee's consensus. Re-estimate your forecast privately.

HOW TO REVISE
- Nobody will see whether you changed your mind. There is no reward for agreeing and no penalty for holding your view. Your revised forecast is scored against the outcome exactly like your sealed one.
- Change your estimate only if the challenge raises a fact or an inference that your sealed analysis missed or got wrong. The existence of disagreement, the confidence of the challenger and rhetoric are not reasons.
- If you change, list the fact IDs that justify the change in new_evidence_fact_ids. They must be facts from your slice or facts cited in the challenge. A change that cites no such fact is discarded and your sealed view stands.
- Keeping your view (changed=false, same numbers) is often correct. Most good revisions are small; a large move needs strong new evidence.
- Keep the probability and the expected excess return consistent with each other.
- Never mention position sizes, weights or trades.

EVIDENCE RULES
{EVIDENCE_RULES}"""

MEMO_SYSTEM = f"""\
You are the chair of an equity investment committee. Analysts with different data slices have written sealed, independent forecasts, and code has pooled them statistically. Do not re-forecast. Your job is to audit the decision and write a machine-checkable investment-committee memo that is frozen before execution and graded later.

YOUR AUTHORITY
- You can only shrink the committee's edge. conviction_multiplier (0 to 1) multiplies the pooled edge: 1 means the analysis holds up; lower means you found problems. You cannot increase or reverse the edge.
- Position sizes are set by code. Never mention weights, share counts or dollar amounts to trade.

WORK THROUGH, IN THIS ORDER
1. Outside view (base_rate_note): name the reference class. Roughly half of individual stocks beat a broad benchmark over a one-month horizon. How far does the pooled probability move from that base rate, and does the evidence justify the distance?
2. Evidence audit: are the analysts' key claims backed by cited facts on the sheet? Are any views flagged by the grounding checker? Do different data slices agree for independent reasons, or is one fact (for example a recent return) being counted several times? Did anyone abstain, and is anything listed under DATA GAPS?
3. Variant perception: what does the price already imply (recent returns, valuation)? Where exactly does the committee differ from that, and why would the market be wrong? If there is no real variant view, write "none" and cut the multiplier.
4. Disagreement: if a challenge round ran, did the consensus answer the strongest counterargument with facts? How many analysts crossed 0.50 after it?
5. Catalysts: only events visible on the fact sheet within or near the horizon. Never invent dates.
6. Pre-mortem: assume the position lost money over the horizon. List the most likely reasons.
7. Kill criteria: 2 to 4 conditions that would force an exit or review during the horizon. metric must be a fact ID from the sheet; condition is a comparator and a number (for example "< -10" or "> 30").
8. Checklist: evaluate each item below (passed true or false, with a short note):
   - Evidence grounded: the key claims cite facts on the sheet and no influential view is flagged.
   - Independent confirmation: at least two different data slices point the same way for different reasons.
   - Base rate respected: the pooled probability is plausible for the horizon.
   - Data complete: no data gap undermines the thesis.
   - Dissent addressed: the strongest counterargument has a factual answer, or no challenge was needed.
   - Not already priced: a specific variant perception exists.
   - Risks specified: the pre-mortem and kill criteria are concrete.

MULTIPLIER GUIDANCE
Start at 1.0. About 0.75 for one serious checklist failure, about 0.5 for two, 0.25 or less for three or more. If the data or the analysis looks broken, set 0 and recommend "review".

RECOMMENDATION
buy = a positive edge worth acting on (opens or adds to a position); hold = keep an existing position, or the edge is small or unclear (for a name the book does not currently hold, hold means do not buy: no position is opened); avoid = do not own; sell = exit an existing position; review = the data or analysis looks wrong and nobody should trade until a human checks. The book may be long-only, in which case a negative pooled edge means avoid (or sell if held).

RULES
- The analysts are shown under random letters in a random order. Weigh their arguments, not their identity or their position in the list.
{EVIDENCE_RULES}

NEUTRALITY
{NEUTRALITY}"""


# --------------------------------------------------------------------------- helpers

def llm_analysts(cfg: FirmConfig) -> list[str]:
    """Enabled LLM analyst names, in config order. Raises on an unknown name, so a typo in
    the config fails loudly instead of silently dropping an analyst."""
    names = [n for n in cfg.analysts.enabled if n != QUANT_AGENT]
    unknown = [n for n in names if n not in ROLES]
    if unknown:
        raise ValueError(f"unknown analysts in config: {unknown}; known: {sorted(ROLES)} and '{QUANT_AGENT}'")
    return names


def _parallel_map(fn: Callable[[_T], _R], items: list[_T], max_workers: int) -> list[_R]:
    """Run ``fn`` over ``items`` on a thread pool. Results come back in input order whatever
    the thread timing, so the committee is deterministic."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(fn, items))


def anonymous_order(ticker: str, n: int) -> list[int]:
    """A permutation of ``range(n)`` seeded by the ticker: random-looking (no fixed speaking
    order the chair could learn) but reproducible across runs."""
    seed = int.from_bytes(hashlib.sha256(ticker.encode()).digest()[:8], "big")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def _letter(i: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[i % 26] + (str(i // 26 + 1) if i >= 26 else "")


def _evidence_lines(items: Iterable[EvidenceItem], indent: str = "",
                    tx: Callable[[str], str] = lambda x: x) -> list[str]:
    return [f"{indent}- [{e.direction}] {tx(e.point)} (facts: {', '.join(e.fact_ids) or 'none'})" for e in items]


# Who wrote a view must not leak into the anonymised prompts: role strings from error messages
# ("analyst.technical: model refused") and self-descriptions ("as the macro analyst", the role
# titles) are replaced. Fact IDs stay: they are the evidence the chair audits.
_ROLE_STRING = re.compile(r"\b(?:analyst|revision)\.[A-Za-z_]+\b(?::\s*)?", re.I)
_ROLE_WORDS = (r"fundamentals?|technical|macro(?:[- ]?economic)?(?:\s+and\s+market[- ]regime)?|market[- ]regime|"
               r"events?(?:\s+and\s+filings)?|filings|quant(?:itative)?")
_ROLE_PHRASE = re.compile(rf"\b(?:(?:the|a|an|our|my|this)\s+)?(?:{_ROLE_WORDS})\s+(?:analyst|desk|team)s?\b", re.I)


def anonymize_roles(text: str) -> str:
    """Remove analyst identities from peer-written text shown under an anonymous label."""
    if not text:
        return text
    out = _ROLE_STRING.sub("", text)
    return _ROLE_PHRASE.sub("a committee member", out)


def _peer_text(fs: FactSheet, blind: bool) -> Callable[[str], str]:
    """Transform for text another committee member wrote: identities removed, and in blind
    mode the company's identity scrubbed and dates made relative."""
    if blind:
        return lambda text: anonymize_roles(blind_text(text, fs))
    return anonymize_roles


def _own_text(fs: FactSheet, blind: bool) -> Callable[[str], str]:
    """Transform for an agent's own earlier text (blind-mode scrubbing only)."""
    if blind:
        return lambda text: blind_text(text, fs)
    return lambda text: text


def select_challenger(views: list[ViewRecord], pooled_prob: float) -> ViewRecord | None:
    """The LLM analyst whose sealed probability is farthest from the pool on the opposite
    side of 0.5; if nobody is on the opposite side, the farthest overall (P6). The quant
    view cannot write a challenge, so it is never chosen. Ties go to the earlier view."""
    candidates = [v for v in views if not v.view.abstain and v.agent in ROLES]
    if not candidates:
        return None
    side = pooled_prob - 0.5
    opposed = [v for v in candidates if (v.view.prob_outperform - 0.5) * side < 0]
    return max(opposed or candidates, key=lambda v: abs(v.view.prob_outperform - pooled_prob))


def edge_multiplier_for(memo: ICMemo | None, long_only: bool, held: bool = False) -> float:
    """1.0 without a memo; 0 for "review", for "avoid"/"sell" in a long-only book, and for
    "hold" on a name the book does not own (hold means keep an existing position, so for
    an unowned name it means do not buy); otherwise the chair's conviction multiplier
    clamped to [0, 1] (it can only shrink). ``held`` defaults to False, the fail-closed
    reading: a "hold" never opens a position unless the name is known to be held."""
    if memo is None:
        return 1.0
    if memo.recommendation == "review" or (long_only and memo.recommendation in ("avoid", "sell")):
        return 0.0
    if memo.recommendation == "hold" and not held:
        return 0.0
    return clamp(memo.conviction_multiplier, 0.0, 1.0)


def memo_required(agg: AggregatedView, cfg: FirmConfig, held: bool) -> bool:
    """Whether the chair must review this name: it is held, its pooled probability clears the
    portfolio's holding threshold (so construction could size it), or its edge is at least
    ``memo_min_edge``. The holding-threshold rule makes sure no name can be sized without a
    memo, whatever the two thresholds are set to."""
    p, pc = agg.prob_outperform, cfg.portfolio
    could_hold = p >= pc.min_prob_to_hold or (not pc.long_only and p <= 1.0 - pc.min_prob_to_hold)
    return held or could_hold or abs(p - 0.5) >= cfg.process.memo_min_edge


def _set_weights(views: list[ViewRecord], believability: dict[str, float]) -> None:
    """weight = believability, split evenly across an agent's samples so that repeated
    samples measure instability without multiplying that agent's vote; halved if ungrounded."""
    samples = Counter(v.agent for v in views)
    for v in views:
        w = max(0.0, float(believability.get(v.agent, 1.0))) / samples[v.agent]
        v.weight = w * 0.5 if not v.grounding_ok else w


def _header(fs: FactSheet, cfg: FirmConfig, blind: bool) -> str:
    h, bench = cfg.process.horizon_days, cfg.universe.benchmark
    if blind:
        return f"TICKER: {BLIND_ALIAS} (identity withheld) | as of day T | horizon: next {h} trading days ({horizon_phrase(h)}) vs {bench}"
    name = f" ({fs.name})" if fs.name else ""
    return f"TICKER: {fs.ticker}{name} | as of {fs.as_of} | horizon: next {h} trading days ({horizon_phrase(h)}) vs {bench}"


def _display_ticker(fs: FactSheet, blind: bool) -> str:
    return BLIND_ALIAS if blind else fs.ticker


def _full_sheet(fs: FactSheet, blind: bool) -> str:
    return (blind_factsheet(fs) if blind else fs).render()


def _finish_prompt(text: str, fs: FactSheet, blind: bool) -> str:
    """In blind mode, re-blind any real fact IDs carried over from stored views."""
    return blind_ids_in_text(text, fs.ticker) if blind else text


def _restore_ids(ids: list[str], ticker: str, blind: bool, shown_ids: set[str], shown_ticker: str) -> list[str]:
    """Citations as the model wrote them -> real fact IDs. Each is first resolved against the
    IDs the model was shown (a bare key "ret_3m" or a pattern "MKT.*_vol_20d" -> the ID on the
    sheet, see ``grounding.resolve_fact_id``), then mapped back from the blind alias."""
    out: list[str] = []
    for i in ids:
        rid = resolve_fact_id(i, shown_ids, shown_ticker)
        if not rid:
            continue
        rid = unblind_id(rid, ticker) if blind else rid
        if rid not in out:
            out.append(rid)
    return out


def _shown_full(fs: FactSheet, blind: bool) -> FactSheet:
    return blind_factsheet(fs) if blind else fs


# --------------------------------------------------------------------------- stages

def _sealed_views(llm: LLM, fs: FactSheet, cfg: FirmConfig, lessons: dict[str, list[str]]) -> list[ViewRecord]:
    models = cfg.models.analyst_models
    if not models:
        raise ValueError("cfg.models.analyst_models is empty")
    tasks = [(name, s) for name in llm_analysts(cfg) for s in range(max(1, cfg.models.samples_per_analyst))]

    def run(task: tuple[str, int]) -> ViewRecord:
        name, sample = task
        return run_analyst(llm, name, fs, cfg, models[sample % len(models)], sample=sample,
                           lessons=lessons.get(name, []))

    return _parallel_map(run, tasks, cfg.models.max_parallel_calls)


def challenge_user_prompt(fs: FactSheet, cfg: FirmConfig, challenger: ViewRecord, views: list[ViewRecord],
                          agg: AggregatedView) -> str:
    """Challenger's prompt: its role, its own sealed view, an anonymised summary of the pool
    and the full fact sheet. No other agent is named."""
    blind = cfg.process.blind_mode
    role = ROLES[challenger.agent]
    pooled = agg.prob_outperform
    active = [v for v in views if not v.view.abstain]
    members = ", ".join(f"{p:.3f}" for p in sorted(v.view.prob_outperform for v in active))
    consensus = [v for v in active if v is not challenger and (v.view.prob_outperform - 0.5) * (pooled - 0.5) >= 0]
    points: list[EvidenceItem] = []
    for i in anonymous_order(fs.ticker, len(consensus)):
        for e in consensus[i].view.evidence:
            if e.point not in {p.point for p in points}:
                points.append(e)
    own = challenger.view
    lines = [
        _header(fs, cfg, blind), "",
        "YOUR ROLE ON THE COMMITTEE (for your eyes only: do not state it in your challenge)",
        f"{role.title}: {role.mandate}", "",
        "YOUR SEALED VIEW (written from your own data slice before seeing anything else)",
        f"{SEALED_PROB_LABEL}: {own.prob_outperform:.3f}",
        f"{SEALED_EXCESS_LABEL}: {own.expected_excess_return_pct:+.2f}%",
        "Your evidence:", *_evidence_lines(own.evidence, tx=_own_text(fs, blind)), "",
        "THE COMMITTEE'S POOLED VIEW (anonymised)",
        f"{POOLED_PROB_LABEL}: {pooled:.3f}",
        f"Pooled expected excess return: {agg.expected_excess_return_pct:+.2f}%",
        f"{DISPERSION_LABEL}: {agg.dispersion:.3f}",
        f"Member probabilities, sorted (yours included, no names): {members}",
        "Main arguments on the consensus side:",
        *(_evidence_lines(points[:_MAX_CONSENSUS_POINTS], tx=_peer_text(fs, blind)) or ["- (none stated)"]), "",
        "THE FULL FACT SHEET (every desk's data, not only your slice)",
        "<fact_sheet>", _full_sheet(fs, blind), "</fact_sheet>", "",
        f'Write your challenge. Use ticker "{_display_ticker(fs, blind)}".',
    ]
    return _finish_prompt("\n".join(lines), fs, blind)


def revision_user_prompt(fs: FactSheet, cfg: FirmConfig, view: ViewRecord, challenge: Challenge) -> str:
    """Reviser's prompt: only its own slice, its own sealed view and the anonymous challenge."""
    blind = cfg.process.blind_mode
    role = ROLES[view.agent]
    own = view.view
    mine, peer = _own_text(fs, blind), _peer_text(fs, blind)
    lines = [
        _header(fs, cfg, blind), "",
        "YOUR ROLE ON THE COMMITTEE", f"{role.title}: {role.mandate}", "",
        "YOUR DATA SLICE", "<fact_sheet>", render_slice(role, shown_sheet(role, fs, blind)), "</fact_sheet>", "",
        "YOUR SEALED VIEW",
        f"{SEALED_PROB_LABEL}: {own.prob_outperform:.3f}",
        f"{SEALED_EXCESS_LABEL}: {own.expected_excess_return_pct:+.2f}%",
        "Your evidence:", *_evidence_lines(own.evidence, tx=mine),
        "Your key risks:", *[f"- {mine(r)}" for r in own.key_risks], "",
        f"{CHALLENGE_HEADER} (written by a committee member who disagrees with the consensus; identity withheld)",
        *_challenge_lines(challenge, peer), "",
        f'Re-estimate privately. Use ticker "{_display_ticker(fs, blind)}".',
    ]
    return _finish_prompt("\n".join(lines), fs, blind)


def memo_user_prompt(fs: FactSheet, cfg: FirmConfig, views: list[ViewRecord], agg: AggregatedView,
                     challenge: Challenge | None, held: bool, lessons: list[str]) -> str:
    """Chair's prompt: the pool, every view anonymised (letters, ticker-seeded order), the
    challenge, the committee's past lessons and the full fact sheet."""
    blind = cfg.process.blind_mode
    lines = [
        _header(fs, cfg, blind),
        f"{HELD_LABEL}: {'yes' if held else 'no'}",
        f"Book: {'long-only' if cfg.portfolio.long_only else 'long-short'}", "",
        "POOLED FORECAST",
        f"{POOLED_PROB_LABEL}: {agg.prob_outperform:.3f}",
        f"Equal-weight pooled probability: {agg.prob_equal_weight:.3f}",
        f"Pooled expected excess return: {agg.expected_excess_return_pct:+.2f}%",
        f"{DISPERSION_LABEL}: {agg.dispersion:.3f}",
        f"Views pooled: {agg.n_views}; abstained: {agg.n_abstained}",
        f"{CHALLENGE_ROUND_LABEL}: " + (f"yes (analysts who crossed 0.50 afterwards: {agg.flips})" if agg.challenged else "no"),
    ]
    if agg.combination == "overlay":
        core = f"{agg.prob_quant:.3f}" if agg.prob_quant is not None else "none (the quant abstained; core 0.500)"
        llm_p = f"{agg.prob_llm:.3f}" if agg.prob_llm is not None else "none (every language-model analyst abstained)"
        lines += ["How the pooled probability was formed: the deterministic factor model is the core; the "
                  "language-model analysts' pooled edge (their pooled probability minus 0.50) is added to it, "
                  f"limited to {agg.max_tilt or 0.0:.3f} either way.",
                  f"Factor-model core: {core}; analysts' pool (calibrated): {llm_p}; tilt applied: "
                  f"{agg.overlay_tilt or 0.0:+.3f}"]
    lines += ["", "ANALYST VIEWS (anonymised; letters are random labels in a random order, not ranks; probabilities "
                  "as each analyst stated them)"]
    peer = _peer_text(fs, blind)
    for pos, i in enumerate(anonymous_order(fs.ticker, len(views))):
        v, view = views[i], views[i].view
        lines.append(f"Analyst {_letter(pos)}")
        if view.abstain:
            lines.append(f"  {ABSTAINED_LABEL} {peer(view.abstain_reason)}")
            continue
        sealed = f"  {MEMBER_SEALED_LABEL} {view.prob_outperform:.3f}, expected excess {view.expected_excess_return_pct:+.2f}%"
        if v.revised_prob is not None and (v.revised_prob, v.revised_excess_pct) != (view.prob_outperform, view.expected_excess_return_pct):
            sealed += f"; after the challenge: {v.revised_prob:.3f}, {v.revised_excess_pct:+.2f}%"
        lines.append(sealed)
        issues = "; ".join(peer(x) for x in v.grounding_issues)
        lines.append("  Grounding: OK" if v.grounding_ok else f"  {FLAGGED_LABEL} ({issues or 'unspecified'})")
        if v.grounding_ok and v.grounding_issues:
            lines.append(f"  Notes: {issues}")
        lines.append("  Scenarios: " + " | ".join(
            f"{s.name} {s.probability:.0%} ({s.excess_return_pct:+.2f}%): {peer(s.description)}" for s in view.scenarios))
        lines += ["  Evidence:", *_evidence_lines(view.evidence, "  ", tx=peer)]
        lines.append(f"  Key risks: {'; '.join(peer(r) for r in view.key_risks) or 'none stated'}")
        lines.append(f"  Private insight: {peer(view.private_insight) or 'none stated'}")
    lines.append("")
    if challenge is None:
        lines.append("CHALLENGE: none (dispersion below the threshold, too few views, or the round was skipped).")
    else:
        lines += [f"{CHALLENGE_HEADER} (from the most opposed analyst; identity withheld)", *_challenge_lines(challenge, peer)]
    lines.append("")
    if lessons:
        own = _own_text(fs, blind)
        lines += ["PAST PROCESS LESSONS FOR THE COMMITTEE (from forecasts resolved before the as-of date)",
                  *[f"- {own(x)}" for x in lessons], ""]
    lines += ["FULL FACT SHEET", "<fact_sheet>", _full_sheet(fs, blind), "</fact_sheet>", "",
              f'Write the IC memo. Use ticker "{_display_ticker(fs, blind)}".']
    return _finish_prompt("\n".join(lines), fs, blind)


def _challenge_lines(challenge: Challenge, tx: Callable[[str], str]) -> list[str]:
    return [f"Strongest counterargument: {tx(challenge.strongest_counterargument)}",
            "Evidence:", *_evidence_lines(challenge.evidence, tx=tx),
            "Overlooked risks or opportunities:", *[f"- {tx(x)}" for x in challenge.overlooked_risks_or_opportunities],
            "Questions for the committee:", *[f"- {tx(q)}" for q in challenge.questions_for_the_committee]]


@dataclass
class _CallResult:
    obj: object | None
    error: str | None
    served_by: str


def _call(llm: LLM, schema: type[_T], **kwargs) -> _CallResult:
    """One structured call; LLM failures come back as an error string instead of raising.
    ``served_by`` is the model that actually answered (the requested one if none did)."""
    requested = kwargs.get("model", "")
    try:
        obj, rec = llm.structured(schema=schema, **kwargs)
        return _CallResult(obj, None, served_model(llm, requested, getattr(rec, "served_by", "")))
    except (LLMCallFailed, BudgetExceeded) as exc:
        return _CallResult(None, str(exc), requested)


def _run_challenge(llm: LLM, fs: FactSheet, cfg: FirmConfig, challenger: ViewRecord, views: list[ViewRecord],
                   agg: AggregatedView, log: LogFn) -> Challenge | None:
    blind = cfg.process.blind_mode
    res = _call(llm, Challenge, role="challenge", model=challenger.model, system=CHALLENGE_SYSTEM,
                user=challenge_user_prompt(fs, cfg, challenger, views, agg),
                effort=cfg.models.analyst_effort, max_tokens=cfg.models.max_tokens)
    base = {"ticker": fs.ticker, "challenger": challenger.agent, "challenger_sample": challenger.sample,
            "challenger_prob": challenger.view.prob_outperform, "pooled_prob": agg.prob_outperform,
            "model": challenger.model, "served_by": res.served_by}
    challenge = res.obj
    if challenge is None:
        log("challenge", {**base, "error": res.error})
        return None
    challenge.ticker = fs.ticker
    shown = _shown_full(fs, blind)
    for e in challenge.evidence:
        e.fact_ids = _restore_ids(e.fact_ids, fs.ticker, blind, shown.ids(), shown.ticker)
    log("challenge", {**base, **challenge.model_dump()})
    return challenge


def _apply_revision(v: ViewRecord, rev: Revision, valid_ids: set[str], ticker: str, blind: bool,
                    shown_ids: set[str] | None = None, shown_ticker: str | None = None) -> dict:
    """Store an accepted revision on ``v`` and return the ledger payload. A claimed change
    is accepted only if it cites at least one fact the analyst could legitimately use.
    ``shown_ids`` are those facts as the reviser saw them (aliased in blind mode), used to
    resolve bare keys and patterns; they default to ``valid_ids``."""
    before_p, before_x = v.view.prob_outperform, v.view.expected_excess_return_pct
    if shown_ids is None:
        shown_ids = {blind_ids_in_text(i, ticker) for i in valid_ids} if blind else set(valid_ids)
    cited = _restore_ids(rev.new_evidence_fact_ids, ticker, blind, shown_ids,
                         shown_ticker or (BLIND_ALIAS if blind else ticker))
    grounded = [i for i in cited if i in valid_ids]
    if not rev.changed:
        status = "unchanged"
    elif not grounded:
        status = "rejected: the change cites no fact from the analyst's slice or the challenge"
    else:
        status = "accepted"
        v.revised_prob = clamp(rev.prob_outperform, PROB_FLOOR, 1 - PROB_FLOOR)
        v.revised_excess_pct = clamp(rev.expected_excess_return_pct, -100.0, 300.0)
        v.revision_fact_ids = list(grounded)     # what the tier gate checks a raised revision against
    return {"ticker": ticker, "agent": v.agent, "sample": v.sample, "model": v.model, "status": status,
            "prob_before": before_p, "prob_after": v.revised_prob, "excess_before": before_x,
            "excess_after": v.revised_excess_pct, "claimed_change": rev.changed, "reason": rev.reason,
            "new_evidence_fact_ids": cited}


def _run_revisions(llm: LLM, fs: FactSheet, cfg: FirmConfig, views: list[ViewRecord], challenger: ViewRecord,
                   challenge: Challenge, log: LogFn) -> None:
    """Private revisions by every other LLM analyst, stored as ``revised_prob`` /
    ``revised_excess_pct`` on the views (the aggregator counts the flips across 0.5)."""
    blind = cfg.process.blind_mode
    active = [v for v in views if not v.view.abstain]
    for v in active:                      # default: everyone keeps the sealed view (challenger and quant included)
        v.revised_prob, v.revised_excess_pct = v.view.prob_outperform, v.view.expected_excess_return_pct
        v.revision_fact_ids = []
        v.challenge_fact_ids = []
    revisers = [v for v in active if v.agent in ROLES and v.agent != challenger.agent]
    challenge_ids = {i for e in challenge.evidence for i in e.fact_ids}
    # what the challenge leaned on: the tier gate caps a raised revision when any of these is unverified news
    leaned_on = list(dict.fromkeys(i for e in challenge.evidence if e.direction in TAINTING_DIRECTIONS
                                   for i in e.fact_ids))
    for v in revisers:
        v.challenge_fact_ids = list(leaned_on)

    def run(v: ViewRecord) -> _CallResult:
        return _call(llm, Revision, role=f"revision.{v.agent}", model=v.model, system=REVISION_SYSTEM,
                     user=revision_user_prompt(fs, cfg, v, challenge), effort=cfg.models.analyst_effort,
                     max_tokens=cfg.models.max_tokens)

    for v, res in zip(revisers, _parallel_map(run, revisers, cfg.models.max_parallel_calls)):
        if res.obj is None:
            log("revision", {"ticker": fs.ticker, "agent": v.agent, "sample": v.sample, "model": v.model,
                             "served_by": res.served_by, "status": "failed: sealed view kept", "error": res.error})
            continue
        shown = shown_sheet(ROLES[v.agent], fs, blind)
        slice_ids = {unblind_id(i, fs.ticker) for i in shown.ids()}
        shown_ids = shown.ids() | ({blind_ids_in_text(i, fs.ticker) for i in challenge_ids} if blind else challenge_ids)
        payload = _apply_revision(v, res.obj, slice_ids | challenge_ids, fs.ticker, blind, shown_ids, shown.ticker)
        log("revision", {**payload, "served_by": res.served_by})


# Edge multiplier for a held name whose required memo failed: the pooled view stands without
# the chair's shrink, and ``hold_current`` asks the pipeline to keep the current weight.
FAILED_MEMO_HELD_MULTIPLIER = 1.0


def _run_memo(llm: LLM, fs: FactSheet, cfg: FirmConfig, views: list[ViewRecord], agg: AggregatedView,
              challenge: Challenge | None, held: bool, lessons: list[str],
              log: LogFn) -> tuple[ICMemo | None, float, bool]:
    """The chair's memo. Returns (memo, edge multiplier, hold_current)."""
    model = cfg.models.committee_model
    res = _call(llm, ICMemo, role="memo", model=model, system=MEMO_SYSTEM,
                user=memo_user_prompt(fs, cfg, views, agg, challenge, held, lessons),
                effort=cfg.models.committee_effort, max_tokens=cfg.models.max_tokens)
    memo = res.obj
    if memo is None:
        if held:   # an LLM failure is not information about the stock: keep the position, do not sell it
            log("memo", {"ticker": fs.ticker, "model": model, "served_by": res.served_by, "error": res.error,
                         "held": True, "hold_current": True, "edge_multiplier": FAILED_MEMO_HELD_MULTIPLIER,
                         "note": "required memo failed on a held name: keep the current weight this cycle"})
            return None, FAILED_MEMO_HELD_MULTIPLIER, True
        log("memo", {"ticker": fs.ticker, "model": model, "served_by": res.served_by, "error": res.error,
                     "held": False, "hold_current": False, "edge_multiplier": 0.0})
        return None, 0.0, False
    memo.ticker = fs.ticker
    shown = _shown_full(fs, cfg.process.blind_mode)
    for k in memo.kill_criteria:
        metric = resolve_fact_id(k.metric, shown.ids(), shown.ticker) or k.metric
        k.metric = unblind_id(metric, fs.ticker) if cfg.process.blind_mode else metric
    multiplier = edge_multiplier_for(memo, cfg.portfolio.long_only, held=held)
    log("memo", {**memo.model_dump(), "model": model, "served_by": res.served_by, "held": held,
                 "hold_current": False, "edge_multiplier": multiplier})
    return memo, multiplier, False


# --------------------------------------------------------------------------- pooling inputs and the final aggregate


def pool_inputs(views: list[ViewRecord], fs: FactSheet, cfg: FirmConfig,
                calibrators: dict[str, Calibrator] | None) -> tuple[list[ViewRecord], list[str]]:
    """The views as they enter the pool, in the same order: each LLM analyst's view through
    the tier gate (if ``cfg.process.tier_gate_enabled``) and then its agent's calibrator (if
    given). The quant and any agent without an LLM role pass unchanged. Inputs are never
    modified; changed views are copies that keep their weights. Returns the views and the
    tier-gate notes."""
    out, notes = [], []
    for v in views:
        if v.agent not in ROLES:
            out.append(v)
            continue
        if cfg.process.tier_gate_enabled:
            v, note = tier_gate(v, fs)
            if note:
                notes.append(note)
        if calibrators:
            v = calibrate_view(v, calibrators.get(v.agent))
        out.append(v)
    return out, notes


def _pool_log(fs: FactSheet, stage: str, raw: list[ViewRecord], pooled: list[ViewRecord], notes: list[str],
              calibrators: dict[str, Calibrator] | None) -> dict:
    rows = []
    for r, v in zip(raw, pooled, strict=True):
        if r.view.abstain:
            continue
        cal = (calibrators or {}).get(r.agent)
        rows.append({"agent": r.agent, "sample": r.sample, "raw_prob": r.view.prob_outperform,
                     "pool_prob": v.view.prob_outperform, "raw_excess_pct": r.view.expected_excess_return_pct,
                     "pool_excess_pct": v.view.expected_excess_return_pct, "raw_revised_prob": r.revised_prob,
                     "pool_revised_prob": v.revised_prob, "weight": v.weight,
                     "calibrator": None if cal is None or cal.is_identity else {"a": cal.a, "b": cal.b, "n": cal.n}})
    return {"ticker": fs.ticker, "stage": stage, "tier_gate_notes": list(notes), "views": rows}


def _final_aggregate(fs: FactSheet, cfg: FirmConfig, pool: list[ViewRecord], max_tilt: float,
                     use_revised: bool) -> tuple[AggregatedView, AggregatedView, AggregatedView | None]:
    """``(final, members, llm_pool)``: the aggregate the firm sizes from (pool or overlay), the
    pool of every member (the challenge trigger) and the LLM analysts' pool (None without
    an LLM analyst view)."""
    trim = cfg.process.trim_fraction
    members = aggregate_views(fs.ticker, pool, trim=trim, use_revised=use_revised)
    quant = next((v for v in pool if v.agent == QUANT_AGENT), None)
    llm_views = [v for v in pool if v.agent in ROLES]
    if cfg.process.combination == "overlay":
        final, llm_agg = overlay_aggregate(fs.ticker, quant, llm_views, trim=trim, max_tilt=max_tilt,
                                           sigma_h_pct=horizon_sigma_pct(fs, cfg.process.horizon_days),
                                           use_revised=use_revised)
        return final, members, llm_agg
    llm_agg = aggregate_views(fs.ticker, llm_views, trim=trim, use_revised=use_revised) if llm_views else None
    q_ok = quant is not None and not quant.view.abstain
    final = members.model_copy(update={
        "combination": "pool",
        "prob_quant": float(quant.view.prob_outperform) if q_ok else None,  # type: ignore[union-attr]
        "prob_llm": llm_agg.prob_outperform if llm_agg is not None and llm_agg.n_views > 0 else None})
    return final, members, llm_agg


# --------------------------------------------------------------------------- entry point

def run_committee(llm: LLM, fs: FactSheet, cfg: FirmConfig, *, quant_view: ViewRecord | None,
                  believability: dict[str, float], lessons: dict[str, list[str]], held: bool,
                  log: LogFn = lambda e, p: None, calibrators: dict[str, Calibrator] | None = None,
                  max_tilt: float | None = None) -> CommitteeResult:
    """Run the full committee process for one ticker (docs/ARCHITECTURE.md section 3.5).

    ``lessons`` maps agent name to retrieved process lessons; the chair receives
    ``lessons["committee"]``. ``log(event, payload)`` receives "view" (raw), "pool_inputs"
    (raw vs pooled probabilities, tier-gate notes), "aggregate" (sealed and, after a
    challenge, revised), "challenge", "revision" and "memo" events for the ledger;
    challenge, revision and memo payloads carry ``served_by``, the model that actually
    answered.

    ``calibrators``: per-agent calibration maps fitted point-in-time (None: no calibration).
    ``max_tilt``: the overlay bound in force (None: ``cfg.process.overlay_max_tilt``; the
    pipeline passes the adaptive bound). Only used when ``cfg.process.combination`` is
    "overlay".

    With memos enabled, the chair reviews every name :func:`memo_required` selects; any
    other name gets an edge multiplier of 0. ``CommitteeResult.hold_current`` is True when
    a held name's required memo failed: the caller should keep that name at its current
    weight this cycle (an LLM failure is not a reason to trade it)."""
    tilt = cfg.process.overlay_max_tilt if max_tilt is None else float(max_tilt)
    if not 0.0 <= tilt <= 0.5:
        raise ValueError(f"max_tilt must be in [0, 0.5], got {max_tilt!r}")
    views = _sealed_views(llm, fs, cfg, lessons)
    if quant_view is not None:
        views.append(quant_view.model_copy(deep=True))
    _set_weights(views, believability)
    for v in views:
        log("view", {"ticker": fs.ticker, **v.model_dump()})

    pool, notes = pool_inputs(views, fs, cfg, calibrators)
    gate_notes = list(notes)
    log("pool_inputs", _pool_log(fs, "sealed", views, pool, notes, calibrators))
    agg, members, llm_agg = _final_aggregate(fs, cfg, pool, tilt, use_revised=False)
    log("aggregate", {"stage": "sealed", **agg.model_dump()})

    challenge: Challenge | None = None
    challenger: ViewRecord | None = None
    n_active = sum(1 for v in views if not v.view.abstain)
    if (cfg.process.challenge_enabled and members.dispersion > cfg.process.challenge_dispersion_threshold
            and n_active >= 2):
        chosen = select_challenger(pool, members.prob_outperform)
        if chosen is not None:      # the raw view behind the pooled copy: prompts quote what it stated
            challenger = views[next(i for i, v in enumerate(pool) if v is chosen)]
            challenge = _run_challenge(llm, fs, cfg, challenger, views, members, log)
        if challenge is not None and challenger is not None:
            _run_revisions(llm, fs, cfg, views, challenger, challenge, log)
            pool, notes = pool_inputs(views, fs, cfg, calibrators)
            gate_notes += [n for n in notes if n not in gate_notes]
            log("pool_inputs", _pool_log(fs, "revised", views, pool, notes, calibrators))
            agg, members, llm_agg = _final_aggregate(fs, cfg, pool, tilt, use_revised=True)   # challenged, flips
            log("aggregate", {"stage": "revised", **agg.model_dump()})

    memo: ICMemo | None = None
    edge_multiplier = 1.0                 # memos disabled: the pooled view is sized as it is
    hold_current = False
    if cfg.process.memo_enabled:
        if memo_required(agg, cfg, held):
            memo, edge_multiplier, hold_current = _run_memo(llm, fs, cfg, views, agg, challenge, held,
                                                            lessons.get("committee", []), log)
        else:
            edge_multiplier = 0.0         # not reviewed (and not held): an unreviewed edge opens nothing

    challenger_name = challenger.agent if challenge is not None and challenger is not None else None
    return CommitteeResult(ticker=fs.ticker, views=views, aggregate=agg, challenge=challenge,
                           challenger=challenger_name, memo=memo, edge_multiplier=edge_multiplier,
                           hold_current=hold_current, llm_aggregate=llm_agg, pool_views=pool,
                           tier_gate_notes=gate_notes)
