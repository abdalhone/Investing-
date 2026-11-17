"""Typed contracts between the firm's components.

Two kinds of models live here:

1. **LLM-facing output schemas** (``AnalystView``, ``Challenge``, ``Revision``, ``ICMemo``,
   ``RiskReview``, ``Lesson``). These are passed to Claude as structured-output formats,
   so the model's answer is guaranteed to parse. They intentionally avoid numeric bound
   constraints: values are clamped in code (``sanitize``) instead, so a slightly
   out-of-range number never discards an otherwise valid decision ("coerce, don't drop").

2. **Internal records** (``AggregatedView``, ``TargetPortfolio``, ``Order``, ``Fill``,
   ``ForecastRecord``...). These are produced by deterministic code.

Hard rule: no LLM-facing schema contains share quantities, dollar amounts to trade,
or portfolio weights. LLMs express *views*; code decides *positions*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- LLM-facing


class EvidenceItem(BaseModel):
    point: str = Field(description="One specific observation, in plain words.")
    fact_ids: list[str] = Field(description="IDs of the facts (from the fact sheet) that support this point, e.g. ['AAPL.ret_3m']. Must be non-empty unless the point is purely qualitative.")
    direction: Literal["positive", "negative", "neutral"] = Field(description="Whether this point argues for (positive) or against (negative) the stock beating the benchmark.")


class Scenario(BaseModel):
    name: Literal["bull", "base", "bear"]
    probability: float = Field(description="Probability of this scenario, 0 to 1. The three scenarios must sum to 1.")
    excess_return_pct: float = Field(description="Stock return MINUS benchmark return over the horizon in this scenario, in percent (e.g. 4.0 means +4%).")
    description: str


class AnalystView(BaseModel):
    """One analyst's sealed, independent forecast for one ticker."""

    ticker: str
    abstain: bool = Field(description="True if your data slice is insufficient to form a view. Abstaining is better than guessing.")
    abstain_reason: str = Field(description="Why you abstained; empty string if not abstaining.")
    prob_outperform: float = Field(description="Probability (0 to 1) that the stock's total return beats the benchmark's over the horizon. 0.5 = no edge.")
    expected_excess_return_pct: float = Field(description="Probability-weighted expected excess return vs the benchmark over the horizon, in percent. Should be consistent with your scenarios.")
    scenarios: list[Scenario] = Field(description="Exactly three scenarios: bull, base, bear.")
    evidence: list[EvidenceItem] = Field(description="3 to 8 evidence points, each citing fact IDs.")
    key_risks: list[str] = Field(description="The main ways this view could be wrong.")
    what_would_change_my_mind: str = Field(description="Specific, observable developments that would flip this view.")
    private_insight: str = Field(description="What your data slice shows that analysts looking at other data probably do not see.")


class Challenge(BaseModel):
    """Authentic dissent from the analyst whose independent view most disagreed with the aggregate."""

    ticker: str
    strongest_counterargument: str = Field(description="The single strongest argument against the current consensus, grounded in facts.")
    evidence: list[EvidenceItem]
    overlooked_risks_or_opportunities: list[str]
    questions_for_the_committee: list[str] = Field(description="Questions whose answers would resolve the disagreement.")


class Revision(BaseModel):
    """An analyst's private re-estimate after reading the challenge (no peer identities shown)."""

    ticker: str
    prob_outperform: float
    expected_excess_return_pct: float
    changed: bool
    reason: str = Field(description="If you changed your estimate, name the NEW evidence (fact IDs) that justifies it. Social pressure is not a reason.")
    new_evidence_fact_ids: list[str]


class KillCriterion(BaseModel):
    metric: str = Field(description="A fact ID or observable metric to monitor, e.g. 'AAPL.price' or 'AAPL.revenue_growth_yoy'.")
    condition: str = Field(description="Machine-readable condition, e.g. '< 150' or 'falls below 5'.")
    rationale: str


class ChecklistItem(BaseModel):
    item: str
    passed: bool
    note: str


class ICMemo(BaseModel):
    """Investment-committee memo written by the chair. Frozen before execution, graded later."""

    ticker: str
    recommendation: Literal["buy", "hold", "avoid", "sell", "review"] = Field(description="'review' = something looks wrong with the data or the analysis; no trade until a human looks.")
    thesis_one_sentence: str
    variant_perception: str = Field(description="Where and why we disagree with what the price already implies. Say 'none' if there is no real variant view.")
    base_rate_note: str = Field(description="The outside view: how often do stocks in this situation (reference class) beat the market over this horizon?")
    catalysts: list[str]
    premortem: list[str] = Field(description="Assume the position lost money over the horizon. The most likely reasons why.")
    kill_criteria: list[KillCriterion] = Field(description="Numeric conditions that would force an exit or review.")
    checklist: list[ChecklistItem]
    conviction_multiplier: float = Field(description="0 to 1. Multiplier applied to the committee's aggregated edge. 1 = accept it; lower = you found problems (weak evidence, data gaps, crowded thesis). You cannot increase the edge.")
    multiplier_reason: str


class RiskConcern(BaseModel):
    ticker: str = Field(description="Ticker affected, or 'PORTFOLIO' for book-level concerns.")
    concern: str
    multiplier: float = Field(description="0 to 1 scaling to apply to this position (1 = no change). Risk review can only reduce risk.")


class RiskReview(BaseModel):
    summary: str
    hidden_concentrations: list[str] = Field(description="Shared themes, factor bets or correlated exposures across names that position limits might miss.")
    concerns: list[RiskConcern]
    halt_trading: bool = Field(description="True only for a serious problem (e.g. obviously broken data). Halts all new trades this cycle.")


class Lesson(BaseModel):
    """A process lesson written after a forecast resolves."""

    agent: str
    ticker: str
    what_happened: str
    was_it_skill_or_luck: str
    process_lesson: str = Field(description="A general, reusable lesson about the analysis PROCESS (not about this stock's future).")
    applies_when: str


# --------------------------------------------------------------------------- internal records


class ViewRecord(BaseModel):
    """An AnalystView plus bookkeeping (who produced it, grounding check results, weight)."""

    agent: str                    # e.g. "fundamental", "technical", "macro", "events", "quant"
    model: str                    # e.g. "claude-opus-5-5" or "deterministic"
    sample: int = 0
    view: AnalystView
    grounding_ok: bool = True
    grounding_issues: list[str] = []
    weight: float = 1.0           # believability weight used in aggregation
    revised_prob: float | None = None
    revised_excess_pct: float | None = None


class AggregatedView(BaseModel):
    ticker: str
    prob_outperform: float        # pooled (believability-weighted, trimmed log-odds mean)
    prob_equal_weight: float      # pooled with equal weights, logged side by side
    expected_excess_return_pct: float
    dispersion: float             # std-dev of member probabilities
    n_views: int
    n_abstained: int
    challenged: bool = False
    flips: int = 0                # how many analysts crossed 0.5 after the challenge


class TargetPosition(BaseModel):
    ticker: str
    weight: float
    prob_outperform: float
    expected_excess_return_pct: float
    rationale: str
    constraints_hit: list[str] = []


class TargetPortfolio(BaseModel):
    as_of: str
    positions: list[TargetPosition]
    core_weight: float            # passive benchmark core (core-satellite)
    cash_weight: float
    notes: list[str] = []

    def weights(self) -> dict[str, float]:
        return {p.ticker: p.weight for p in self.positions if p.weight != 0}


class Order(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float
    reference_price: float
    notional: float
    reason: str


class Fill(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    date: str


class ForecastRecord(BaseModel):
    """A resolvable forecast. Every analyst view and every committee view becomes one."""

    id: str
    run_id: str
    agent: str
    model: str
    ticker: str
    benchmark: str
    as_of: str
    horizon_days: int
    resolve_on: str               # first date the outcome is known
    prob_outperform: float
    expected_excess_return_pct: float
    pre_cutoff: bool = False      # made on a date the model may have seen in training
    resolved: bool = False
    outcome: int | None = None    # 1 if the stock beat the benchmark
    realized_excess_return_pct: float | None = None
    brier: float | None = None


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    if x != x:  # NaN
        return (lo + hi) / 2
    return max(lo, min(hi, x))


def sanitize_view(v: AnalystView, prob_floor: float = 0.02) -> AnalystView:
    """Clamp an LLM view into valid ranges and renormalize scenario probabilities."""
    v.prob_outperform = clamp(v.prob_outperform, prob_floor, 1 - prob_floor)
    v.expected_excess_return_pct = clamp(v.expected_excess_return_pct, -100.0, 300.0)
    total = sum(max(0.0, s.probability) for s in v.scenarios)
    for s in v.scenarios:
        s.probability = (max(0.0, s.probability) / total) if total > 0 else 1.0 / max(1, len(v.scenarios))
        s.excess_return_pct = clamp(s.excess_return_pct, -100.0, 300.0)
    if v.abstain:
        v.prob_outperform = 0.5
        v.expected_excess_return_pct = 0.0
    return v
