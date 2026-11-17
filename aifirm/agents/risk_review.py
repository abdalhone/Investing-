"""LLM portfolio risk review: looks for risks the sizing rules miss, and may only scale down.

Why an LLM here at all: position and sector caps cannot see *themes* (several names that
all depend on the same customer, all long-duration growth, all rate-sensitive, all
riding the same momentum). Reading the fact sheets side by side is a language task.

Why it can only reduce (P1, P9): the reviewer outputs per-name multipliers in [0, 1] and
an optional halt; :func:`apply_risk_review` clamps them, so no wording can raise or
reverse a position. The deterministic risk engine runs afterwards regardless.

Failure handling: a failed review returns ``None`` and leaves the target unchanged. The
review is an extra layer that can only reduce risk; an API outage should not by itself
liquidate the book, and the invariant risk engine still applies its hard limits.
"""

from __future__ import annotations

import re

from ..config import FirmConfig
from ..data.factsheet import FactSheet
from ..llm.client import LLM, BudgetExceeded, LLMCallFailed
from ..schemas import RiskReview, TargetPortfolio, clamp
from .analysts import EVIDENCE_RULES, blind_factsheet, fact_section

PORTFOLIO_TICKER = "PORTFOLIO"
POSITIONS_HEADER = "POSITIONS"

RISK_REVIEW_SYSTEM = f"""\
You are the portfolio risk reviewer of an equity fund. Code built the target portfolio from the investment committee's forecasts (fractional Kelly sizing, position and sector caps, a volatility target, a cash floor). After you, a deterministic risk engine enforces hard limits (drawdown de-risking, loss halts, liquidity caps). Your job is to find risks those rules miss and, where justified, scale positions down.

YOUR AUTHORITY
- You can only reduce risk. For each position you want to change, give a multiplier between 0 and 1: 1 = no change, about 0.75 = modest trim, about 0.5 = serious concern, 0 = remove. You cannot increase a position, add a name or change a direction.
- Use ticker "{PORTFOLIO_TICKER}" for a book-level concern; its multiplier scales every active position.
- halt_trading=true sets every active position to zero this cycle (the passive core is unaffected). Use it only for obviously broken data or a systemic problem, never to express a market view.
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


def apply_risk_review(target: TargetPortfolio, review: RiskReview | None) -> TargetPortfolio:
    """Scale positions down by the review's multipliers (clamped to [0, 1]); a halt sets all
    active weights to 0. Several concerns on one name use the smallest multiplier; a
    PORTFOLIO concern multiplies every name. Freed weight goes to cash. Returns a copy."""
    out = target.model_copy(deep=True)
    if review is None:
        return out
    if review.halt_trading:
        mults = {p.ticker: 0.0 for p in out.positions}
        out.notes.append(f"risk review halt: all active weights set to 0 ({review.summary})")
    else:
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
            out.notes.append(f"risk review: book-level multiplier {book:.2f}")
        for p in out.positions:
            if p.weight != 0 and mults[p.ticker] < 1.0 and p.ticker.upper() in reasons:
                out.notes.append(f"risk review: {p.ticker} x{per_name[p.ticker.upper()]:.2f} "
                                 f"({'; '.join(reasons[p.ticker.upper()])})")
    freed = 0.0
    for p in out.positions:
        m = mults[p.ticker]
        if p.weight != 0 and m < 1.0:
            new_weight = p.weight * m
            freed += p.weight - new_weight
            p.weight = new_weight
            p.constraints_hit.append("risk_review")
    out.cash_weight += freed
    return out
