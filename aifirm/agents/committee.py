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
  or to a fixed speaking order.
* **Weights from track record (P7).** Each view is weighted by its agent's believability;
  ungrounded views count half (P2).
* **Fail closed.** A failed analyst call is an abstention; a failed challenge leaves the
  sealed pool in place; a failed revision keeps the sealed view; a failed memo that was
  required sets the edge multiplier to 0, so an unreviewed edge is never traded.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from ..config import FirmConfig
from ..data.factsheet import FactSheet
from ..llm.client import LLM, BudgetExceeded, LLMCallFailed
from ..portfolio.aggregate import aggregate_views
from ..schemas import AggregatedView, Challenge, EvidenceItem, ICMemo, Revision, ViewRecord, clamp
from .analysts import (BLIND_ALIAS, EVIDENCE_RULES, NEUTRALITY, blind_factsheet, blind_ids_in_text, horizon_phrase,
                       render_slice, run_analyst, shown_sheet, unblind_id)
from .grounding import normalize_fact_id
from .roles import ROLES

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
    edge_multiplier: float          # memo.conviction_multiplier clamped to [0,1]; 1.0 if no memo; 0 if memo says review/avoid


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
buy = a positive edge worth acting on; hold = keep an existing position, or the edge is small or unclear; avoid = do not own; sell = exit an existing position; review = the data or analysis looks wrong and nobody should trade until a human checks. The book may be long-only, in which case a negative pooled edge means avoid (or sell if held).

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


def _evidence_lines(items: Iterable[EvidenceItem], indent: str = "") -> list[str]:
    return [f"{indent}- [{e.direction}] {e.point} (facts: {', '.join(e.fact_ids) or 'none'})" for e in items]


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


def edge_multiplier_for(memo: ICMemo | None, long_only: bool) -> float:
    """1.0 without a memo; 0 for "review", and for "avoid"/"sell" in a long-only book;
    otherwise the chair's conviction multiplier clamped to [0, 1] (it can only shrink)."""
    if memo is None:
        return 1.0
    if memo.recommendation == "review" or (long_only and memo.recommendation in ("avoid", "sell")):
        return 0.0
    return clamp(memo.conviction_multiplier, 0.0, 1.0)


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


def _restore_ids(ids: list[str], ticker: str, blind: bool) -> list[str]:
    return [unblind_id(i, ticker) if blind else normalize_fact_id(i) for i in ids if normalize_fact_id(i)]


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
        "YOUR ROLE ON THE COMMITTEE", f"{role.title}: {role.mandate}", "",
        "YOUR SEALED VIEW (written from your own data slice before seeing anything else)",
        f"{SEALED_PROB_LABEL}: {own.prob_outperform:.3f}",
        f"{SEALED_EXCESS_LABEL}: {own.expected_excess_return_pct:+.2f}%",
        "Your evidence:", *_evidence_lines(own.evidence), "",
        "THE COMMITTEE'S POOLED VIEW (anonymised)",
        f"{POOLED_PROB_LABEL}: {pooled:.3f}",
        f"Pooled expected excess return: {agg.expected_excess_return_pct:+.2f}%",
        f"{DISPERSION_LABEL}: {agg.dispersion:.3f}",
        f"Member probabilities, sorted (yours included, no names): {members}",
        "Main arguments on the consensus side:",
        *(_evidence_lines(points[:_MAX_CONSENSUS_POINTS]) or ["- (none stated)"]), "",
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
    lines = [
        _header(fs, cfg, blind), "",
        "YOUR ROLE ON THE COMMITTEE", f"{role.title}: {role.mandate}", "",
        "YOUR DATA SLICE", "<fact_sheet>", render_slice(role, shown_sheet(role, fs, blind)), "</fact_sheet>", "",
        "YOUR SEALED VIEW",
        f"{SEALED_PROB_LABEL}: {own.prob_outperform:.3f}",
        f"{SEALED_EXCESS_LABEL}: {own.expected_excess_return_pct:+.2f}%",
        "Your evidence:", *_evidence_lines(own.evidence),
        "Your key risks:", *[f"- {r}" for r in own.key_risks], "",
        f"{CHALLENGE_HEADER} (written by a committee member who disagrees with the consensus; identity withheld)",
        f"Strongest counterargument: {challenge.strongest_counterargument}",
        "Evidence:", *_evidence_lines(challenge.evidence),
        "Overlooked risks or opportunities:", *[f"- {x}" for x in challenge.overlooked_risks_or_opportunities],
        "Questions for the committee:", *[f"- {q}" for q in challenge.questions_for_the_committee], "",
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
        "", "ANALYST VIEWS (anonymised; letters are random labels in a random order, not ranks)",
    ]
    for pos, i in enumerate(anonymous_order(fs.ticker, len(views))):
        v, view = views[i], views[i].view
        lines.append(f"Analyst {_letter(pos)}")
        if view.abstain:
            lines.append(f"  {ABSTAINED_LABEL} {view.abstain_reason}")
            continue
        sealed = f"  {MEMBER_SEALED_LABEL} {view.prob_outperform:.3f}, expected excess {view.expected_excess_return_pct:+.2f}%"
        if v.revised_prob is not None and (v.revised_prob, v.revised_excess_pct) != (view.prob_outperform, view.expected_excess_return_pct):
            sealed += f"; after the challenge: {v.revised_prob:.3f}, {v.revised_excess_pct:+.2f}%"
        lines.append(sealed)
        lines.append("  Grounding: OK" if v.grounding_ok else f"  {FLAGGED_LABEL} ({'; '.join(v.grounding_issues) or 'unspecified'})")
        if v.grounding_ok and v.grounding_issues:
            lines.append(f"  Notes: {'; '.join(v.grounding_issues)}")
        lines.append("  Scenarios: " + " | ".join(
            f"{s.name} {s.probability:.0%} ({s.excess_return_pct:+.2f}%): {s.description}" for s in view.scenarios))
        lines += ["  Evidence:", *_evidence_lines(view.evidence, "  ")]
        lines.append(f"  Key risks: {'; '.join(view.key_risks) or 'none stated'}")
        lines.append(f"  Private insight: {view.private_insight or 'none stated'}")
    lines.append("")
    if challenge is None:
        lines.append("CHALLENGE: none (dispersion below the threshold, too few views, or the round was skipped).")
    else:
        lines += [f"{CHALLENGE_HEADER} (from the most opposed analyst; identity withheld)",
                  f"Strongest counterargument: {challenge.strongest_counterargument}",
                  "Evidence:", *_evidence_lines(challenge.evidence),
                  "Overlooked risks or opportunities:", *[f"- {x}" for x in challenge.overlooked_risks_or_opportunities],
                  "Questions for the committee:", *[f"- {q}" for q in challenge.questions_for_the_committee]]
    lines.append("")
    if lessons:
        lines += ["PAST PROCESS LESSONS FOR THE COMMITTEE (from forecasts resolved before the as-of date)",
                  *[f"- {x}" for x in lessons], ""]
    lines += ["FULL FACT SHEET", "<fact_sheet>", _full_sheet(fs, blind), "</fact_sheet>", "",
              f'Write the IC memo. Use ticker "{_display_ticker(fs, blind)}".']
    return _finish_prompt("\n".join(lines), fs, blind)


def _call(llm: LLM, schema: type[_T], **kwargs) -> tuple[_T | None, str | None]:
    """One structured call; LLM failures come back as an error string instead of raising."""
    try:
        obj, _ = llm.structured(schema=schema, **kwargs)
        return obj, None
    except (LLMCallFailed, BudgetExceeded) as exc:
        return None, str(exc)


def _run_challenge(llm: LLM, fs: FactSheet, cfg: FirmConfig, challenger: ViewRecord, views: list[ViewRecord],
                   agg: AggregatedView, log: LogFn) -> Challenge | None:
    challenge, err = _call(llm, Challenge, role="challenge", model=challenger.model, system=CHALLENGE_SYSTEM,
                           user=challenge_user_prompt(fs, cfg, challenger, views, agg),
                           effort=cfg.models.analyst_effort, max_tokens=cfg.models.max_tokens)
    base = {"ticker": fs.ticker, "challenger": challenger.agent, "challenger_sample": challenger.sample,
            "challenger_prob": challenger.view.prob_outperform, "pooled_prob": agg.prob_outperform}
    if challenge is None:
        log("challenge", {**base, "error": err})
        return None
    challenge.ticker = fs.ticker
    for e in challenge.evidence:
        e.fact_ids = _restore_ids(e.fact_ids, fs.ticker, cfg.process.blind_mode)
    log("challenge", {**base, **challenge.model_dump()})
    return challenge


def _apply_revision(v: ViewRecord, rev: Revision, valid_ids: set[str], ticker: str, blind: bool) -> dict:
    """Store an accepted revision on ``v`` and return the ledger payload. A claimed change
    is accepted only if it cites at least one fact the analyst could legitimately use."""
    before_p, before_x = v.view.prob_outperform, v.view.expected_excess_return_pct
    cited = _restore_ids(rev.new_evidence_fact_ids, ticker, blind)
    grounded = [i for i in cited if i in valid_ids]
    if not rev.changed:
        status = "unchanged"
    elif not grounded:
        status = "rejected: the change cites no fact from the analyst's slice or the challenge"
    else:
        status = "accepted"
        v.revised_prob = clamp(rev.prob_outperform, PROB_FLOOR, 1 - PROB_FLOOR)
        v.revised_excess_pct = clamp(rev.expected_excess_return_pct, -100.0, 300.0)
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
    revisers = [v for v in active if v.agent in ROLES and v.agent != challenger.agent]
    challenge_ids = {i for e in challenge.evidence for i in e.fact_ids}

    def run(v: ViewRecord) -> tuple[Revision | None, str | None]:
        return _call(llm, Revision, role=f"revision.{v.agent}", model=v.model, system=REVISION_SYSTEM,
                     user=revision_user_prompt(fs, cfg, v, challenge), effort=cfg.models.analyst_effort,
                     max_tokens=cfg.models.max_tokens)

    for v, (rev, err) in zip(revisers, _parallel_map(run, revisers, cfg.models.max_parallel_calls)):
        if rev is None:
            log("revision", {"ticker": fs.ticker, "agent": v.agent, "sample": v.sample, "model": v.model,
                             "status": "failed: sealed view kept", "error": err})
            continue
        slice_ids = {unblind_id(i, fs.ticker) for i in shown_sheet(ROLES[v.agent], fs, blind).ids()}
        log("revision", _apply_revision(v, rev, slice_ids | challenge_ids, fs.ticker, blind))


def _run_memo(llm: LLM, fs: FactSheet, cfg: FirmConfig, views: list[ViewRecord], agg: AggregatedView,
              challenge: Challenge | None, held: bool, lessons: list[str], log: LogFn) -> tuple[ICMemo | None, float]:
    memo, err = _call(llm, ICMemo, role="memo", model=cfg.models.committee_model, system=MEMO_SYSTEM,
                      user=memo_user_prompt(fs, cfg, views, agg, challenge, held, lessons),
                      effort=cfg.models.committee_effort, max_tokens=cfg.models.max_tokens)
    if memo is None:
        log("memo", {"ticker": fs.ticker, "error": err, "edge_multiplier": 0.0})
        return None, 0.0
    memo.ticker = fs.ticker
    if cfg.process.blind_mode:
        for k in memo.kill_criteria:
            k.metric = unblind_id(k.metric, fs.ticker)
    multiplier = edge_multiplier_for(memo, cfg.portfolio.long_only)
    log("memo", {**memo.model_dump(), "edge_multiplier": multiplier})
    return memo, multiplier


# --------------------------------------------------------------------------- entry point

def run_committee(llm: LLM, fs: FactSheet, cfg: FirmConfig, *, quant_view: ViewRecord | None,
                  believability: dict[str, float], lessons: dict[str, list[str]], held: bool,
                  log: LogFn = lambda e, p: None) -> CommitteeResult:
    """Run the full committee process for one ticker (docs/ARCHITECTURE.md section 3.5).

    ``lessons`` maps agent name to retrieved process lessons; the chair receives
    ``lessons["committee"]``. ``log(event, payload)`` receives "view", "aggregate",
    "challenge", "revision" and "memo" events for the ledger."""
    views = _sealed_views(llm, fs, cfg, lessons)
    if quant_view is not None:
        views.append(quant_view.model_copy(deep=True))
    _set_weights(views, believability)
    for v in views:
        log("view", {"ticker": fs.ticker, **v.model_dump()})

    trim = cfg.process.trim_fraction
    agg = aggregate_views(fs.ticker, views, trim=trim)
    log("aggregate", {"stage": "sealed", **agg.model_dump()})

    challenge: Challenge | None = None
    challenger: ViewRecord | None = None
    n_active = sum(1 for v in views if not v.view.abstain)
    if (cfg.process.challenge_enabled and agg.dispersion > cfg.process.challenge_dispersion_threshold
            and n_active >= 2):
        challenger = select_challenger(views, agg.prob_outperform)
        if challenger is not None:
            challenge = _run_challenge(llm, fs, cfg, challenger, views, agg, log)
        if challenge is not None:
            _run_revisions(llm, fs, cfg, views, challenger, challenge, log)
            agg = aggregate_views(fs.ticker, views, trim=trim, use_revised=True)   # sets challenged and flips
            log("aggregate", {"stage": "revised", **agg.model_dump()})

    memo: ICMemo | None = None
    edge_multiplier = 1.0
    if cfg.process.memo_enabled and (abs(agg.prob_outperform - 0.5) >= cfg.process.memo_min_edge or held):
        memo, edge_multiplier = _run_memo(llm, fs, cfg, views, agg, challenge, held,
                                          lessons.get("committee", []), log)

    challenger_name = challenger.agent if challenge is not None and challenger is not None else None
    return CommitteeResult(ticker=fs.ticker, views=views, aggregate=agg, challenge=challenge,
                           challenger=challenger_name, memo=memo, edge_multiplier=edge_multiplier)
