"""LLM portfolio risk review: looks for risks the sizing rules miss, and may only scale down.

Why an LLM here at all: position and sector caps cannot see *themes* (several names that
all depend on the same customer, all long-duration growth, all rate-sensitive, all
riding the same momentum). Reading the fact sheets side by side is a language task.

Why it can only reduce (P1, P9): the reviewer outputs per-name multipliers in [0, 1] and
an optional halt; :func:`apply_risk_review` clamps them, so no wording can raise or
reverse a position. The deterministic risk engine runs afterwards regardless.

What a halt means (the schema, the prompt and the code say the same thing): **no trades
this cycle**. Every position and the passive core stay at their current weights; nothing
is bought or sold because of the review. A halt is the brake for broken data or a
systemic problem, where any trade could be wrong, so it must not itself trigger a wave of
sales. To cut risk, the reviewer uses multipliers (0 removes a position).

Failure handling: a failed review returns ``None`` and leaves the target unchanged. The
review is an extra layer that can only reduce risk; an API outage should not by itself
liquidate the book, and the invariant risk engine still applies its hard limits.
"""

from __future__ import annotations

import math
import re
from typing import Mapping

from ..config import FirmConfig
from ..data.factsheet import FactSheet
from ..llm.client import LLM, BudgetExceeded, LLMCallFailed
from ..schemas import RiskReview, TargetPortfolio, TargetPosition, clamp
from ..portfolio.risk_engine import HOLD_LABELS
from .analysts import EVIDENCE_RULES, blind_factsheet, fact_section

PORTFOLIO_TICKER = "PORTFOLIO"
POSITIONS_HEADER = "POSITIONS"

RISK_REVIEW_SYSTEM = f"""\
You are the portfolio risk reviewer of an equity fund. Code built the target portfolio from the investment committee's forecasts (fractional Kelly sizing, position and sector caps, a volatility target, a cash floor). After you, a deterministic risk engine enforces hard limits (drawdown de-risking, loss halts, liquidity caps). Your job is to find risks those rules miss and, where justified, scale positions down.

YOUR AUTHORITY
- You can only reduce risk. For each position you want to change, give a multiplier between 0 and 1: 1 = no change, about 0.75 = modest trim, about 0.5 = serious concern, 0 = remove. You cannot increase a position, add a name or change a direction.
- Use ticker "{PORTFOLIO_TICKER}" for a book-level concern; its multiplier scales every active position.
- halt_trading=true halts ALL trading this cycle: every position, and the passive core, stays at its current weight; nothing is bought or sold. It does not liquidate anything. Use it only for obviously broken data or a systemic problem that makes any trade unsafe, never to express a market view. To cut or remove positions, use multipliers instead.
- Never propose weights, share counts or dollar amounts.

WHAT TO LOOK FOR
1. Hidden concentrations: shared themes or factor bets across names that per-name and sector caps miss (for example several names with high beta, extreme momentum, high valuation, the same sector under different labels, or the same kind of event risk). List them in hidden_concentrations even when no trim is needed.
2. Position-level red flags on the fact sheets: recent adverse filings (restatement or non-reliance, auditor change, delisting notice, impairment, bankruptcy), data gaps that leave a position's thesis unverifiable, very thin liquidity (adv_20d), or an extreme recent move.
3. Data problems: values that look impossible or inconsistent. Broken data is a reason to trim to 0 or to halt.

DISCIPLINE
- Volatility and the committee's edge are already in the sizing; do not trim a name merely because it is volatile or because you would have forecast differently. Trim when you identify a specific risk the rules did not price.
- Every concern must name the specific facts behind it (fact IDs or the dated filing event). List only positions you want to change; an empty concerns list is a valid answer.

EVIDENCE RULES
{EVIDENCE_RULES}"""


def _alias_map(tickers: list[str], blind: bool) -> dict[str, str]:
    """Real ticker -> name shown to the reviewer (identity in normal mode, S01.. in blind mode)."""
    return {t: (f"S{i + 1:02d}" if blind else t) for i, t in enumerate(tickers)}


def _swap_names(text: str, mapping: dict[str, str]) -> str:
    for src, dst in mapping.items():
        if src != dst:
            text = re.sub(rf"(?<![\w.]){re.escape(src)}(?![\w])", dst, text)
    return text


def risk_review_user_prompt(target: TargetPortfolio, factsheets: dict[str, FactSheet], cfg: FirmConfig) -> str:
    """Positions table, construction notes, each position's fact sheet and the shared macro facts."""
    blind = cfg.process.blind_mode
    active = [p for p in target.positions if p.weight != 0]
    names = _alias_map([p.ticker for p in active], blind)
    lines = [
        f"PORTFOLIO UNDER REVIEW (as of {'day T' if blind else target.as_of})",
        f"Book: {'long-only' if cfg.portfolio.long_only else 'long-short'} | passive core (benchmark "
        f"{cfg.universe.benchmark}): {target.core_weight:.4f} | cash: {target.cash_weight:.4f} | active positions: {len(active)}",
        "",
        f"{POSITIONS_HEADER} (weights were set by code; you may only scale them down)",
    ]
    for p in active:
        fs = factsheets.get(p.ticker)
        sector = (fs.sector if fs else "") or "n/a"
        lines.append(f"- {names[p.ticker]} | sector: {sector} | weight {p.weight:.4f} | pooled P(outperform) "
                     f"{p.prob_outperform:.3f} | expected excess {p.expected_excess_return_pct:+.2f}% | "
                     f"constraints: {', '.join(p.constraints_hit) or 'none'} | rationale: {_swap_names(p.rationale, names)}")
    if target.notes:
        lines += ["", "NOTES FROM THE CONSTRUCTION ENGINE", *[f"- {_swap_names(n, names)}" for n in target.notes]]
    lines += ["", "FACT SHEETS OF THE POSITIONS"]
    macro_lines: list[str] = []
    for p in active:
        fs = factsheets.get(p.ticker)
        if fs is None:
            lines.append(f'<fact_sheet ticker="{names[p.ticker]}">no fact sheet available - treat as a data gap</fact_sheet>')
            continue
        shown = blind_factsheet(fs, alias=names[p.ticker]) if blind else fs
        lines += [f'<fact_sheet ticker="{names[p.ticker]}">', shown.render(("price", "fund", "events", "news")), "</fact_sheet>"]
        if not macro_lines:
            macro_lines = [f.render() for f in shown.facts if fact_section(f.id, shown.ticker) == "macro"]
    if macro_lines:
        lines += ["", "MARKET AND MACRO CONTEXT (shared by all positions)", *macro_lines]
    lines += ["", "Write the risk review."]
    return "\n".join(lines)


def review_portfolio(llm: LLM, target: TargetPortfolio, factsheets: dict[str, FactSheet],
                     cfg: FirmConfig) -> RiskReview | None:
    """Ask the risk model to review ``target``. ``None`` means no review was performed: the
    review is disabled, there are no active positions, or the call failed."""
    if not cfg.process.risk_review_enabled or not any(p.weight != 0 for p in target.positions):
        return None
    try:
        review, _ = llm.structured(role="risk_review", model=cfg.models.risk_model, system=RISK_REVIEW_SYSTEM,
                                   user=risk_review_user_prompt(target, factsheets, cfg), schema=RiskReview,
                                   effort=cfg.models.risk_effort, max_tokens=cfg.models.max_tokens)
    except (LLMCallFailed, BudgetExceeded):
        return None
    active = [p.ticker for p in target.positions if p.weight != 0]
    back = {alias: t for t, alias in _alias_map(active, cfg.process.blind_mode).items()}
    for c in review.concerns:
        shown = c.ticker.strip().upper()
        c.ticker = back.get(shown, shown)
        c.multiplier = clamp(c.multiplier, 0.0, 1.0)
    review.hidden_concentrations = [_swap_names(h, back) for h in review.hidden_concentrations]
    review.summary = _swap_names(review.summary, back)
    return review


def _halt_target(target: TargetPortfolio, review: RiskReview, current_weights: Mapping[str, float],
                 benchmark: str | None) -> TargetPortfolio:
    """The current book, unchanged: a halt means no trades this cycle. ``benchmark`` names the
    passive-core leg of ``current_weights``; if it is not given, current legs that are not
    active positions of ``target`` are taken to be the core (the construction engine lists
    every held active name among the target's positions)."""
    current = {t: float(w) for t, w in current_weights.items() if math.isfinite(float(w))}
    out = target.model_copy(deep=True)
    listed = {p.ticker for p in out.positions}
    core_legs = ({benchmark} if benchmark else set(current) - listed) - listed
    for t in sorted(set(current) - listed - core_legs):
        if current[t] != 0.0:
            out.positions.append(TargetPosition(ticker=t, weight=current[t], prob_outperform=0.5,
                                                expected_excess_return_pct=0.0, rationale=""))
    for p in out.positions:
        new_weight = current.get(p.ticker, 0.0)
        if new_weight != p.weight:
            p.constraints_hit.append("risk_review_halt")
        p.weight = new_weight
    out.core_weight = sum(current.get(t, 0.0) for t in core_legs)
    out.cash_weight = 1.0 - out.core_weight - sum(p.weight for p in out.positions)
    out.notes.append(f"risk review halt: no trades this cycle, every weight kept at its current value ({review.summary})")
    return out


def apply_risk_review(target: TargetPortfolio, review: RiskReview | None,
                      current_weights: Mapping[str, float] | None = None,
                      benchmark: str | None = None) -> TargetPortfolio:
    """Scale positions down by the review's multipliers (clamped to [0, 1]). Several
    concerns on one name use the smallest multiplier; a PORTFOLIO concern multiplies every
    name. A multiplier scales the leg's plan (:func:`reviewed_weight`; ``ideal_weight`` is
    scaled too, so the drawdown de-risk halves the reviewed plan), so a concern repeated
    every cycle does not ratchet the book down under partial rebalancing. Freed weight goes
    to cash. Returns a copy.

    A halt means **no trades this cycle**: with ``current_weights`` (fractions of NAV per
    ticker, including the ``benchmark`` core leg) the result is the current book, so the
    review causes no sale and no purchase. A halt without ``current_weights`` raises
    ``ValueError``: the current book is unknown, and any guess would be a trade (zeroing the
    sleeve would liquidate it, which the reviewer is told a halt never does). In
    :func:`aifirm.pipeline.decide` an error stops the cycle with no orders."""
    out = target.model_copy(deep=True)
    if review is None:
        return out
    if review.halt_trading:
        if current_weights is None:
            raise ValueError("a risk-review halt keeps the current book: pass current_weights to apply_risk_review")
        return _halt_target(target, review, current_weights, benchmark)
    book = min((clamp(c.multiplier, 0.0, 1.0) for c in review.concerns
                if c.ticker.strip().upper() == PORTFOLIO_TICKER), default=1.0)
    per_name: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for c in review.concerns:
        t = c.ticker.strip().upper()
        if t != PORTFOLIO_TICKER:
            per_name[t] = min(per_name.get(t, 1.0), clamp(c.multiplier, 0.0, 1.0))
            reasons.setdefault(t, []).append(c.concern)
    mults = {p.ticker: book * per_name.get(p.ticker.upper(), 1.0) for p in out.positions}
    if book < 1.0:
        out.notes.append(f"risk review: book-level multiplier {book:.2f} on the plan")
    freed = 0.0
    for p in out.positions:
        m = mults[p.ticker]
        if p.weight == 0 or m >= 1.0:
            continue
        before = p.weight
        new_weight = reviewed_weight(p, m)
        if p.ideal_weight is not None and math.isfinite(float(p.ideal_weight)):
            p.ideal_weight = float(p.ideal_weight) * m          # the plan the drawdown de-risk halves
        freed += p.weight - new_weight
        p.weight = new_weight
        p.constraints_hit.append("risk_review")
        if p.ticker.upper() in reasons:
            out.notes.append(f"risk review: {p.ticker} x{per_name[p.ticker.upper()]:.2f} "
                             f"({'; '.join(reasons[p.ticker.upper()])}): {before:.2%} -> {new_weight:.2%}")
    out.cash_weight += freed
    return out


def reviewed_weight(p: TargetPosition, m: float) -> float:
    """The weight of target leg ``p`` after a reviewer multiplier ``m`` in [0, 1).

    The multiplier scales the **plan**, not the partial step. A leg built by construction
    carries its ``ideal_weight``; with partial rebalancing its weight is a step from the
    current book toward that plan, and multiplying the step every cycle would compound
    (a book cut last cycle steps up again and is cut again: ``rebalance_fraction`` 0.5 and a
    standing x0.5 settle at a third of the plan, not a half). The leg is therefore held to
    ``m * max(|ideal|, |step|)``: at or above its plan (a reduction or an exit) that is ``m``
    times the step, as before; while it is still building toward the plan it is capped at
    ``m`` times the plan, so a concern repeated every cycle converges to ``m * ideal``.
    A deliberate hold (no plan: kept at the current weight) and a hand-built leg without an
    ideal weight are multiplied directly. Never raises the magnitude or flips the sign."""
    w = float(p.weight)
    ideal = p.ideal_weight
    if (HOLD_LABELS.intersection(p.constraints_hit) or ideal is None or not math.isfinite(float(ideal))):
        return w * m
    return math.copysign(min(abs(w), m * max(abs(w), abs(float(ideal)))), w)
