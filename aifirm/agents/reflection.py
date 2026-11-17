"""Lessons from resolved forecasts: one reusable *process* lesson per forecast.

Why (P7): scoring tells an analyst *how well* it forecast; a lesson records *what to do
differently*. Two guards keep lessons from doing harm:

* **Process, not predictions.** A lesson says how to weigh a kind of evidence, never what a
  stock or the market will do, and never names the company. Lessons are retrieved for
  unrelated future tickers, so a stock-specific lesson would be noise at best and
  hindsight leakage at worst.
* **Skill versus luck.** One binary outcome is weak evidence; a 0.60 forecast should be
  wrong 40% of the time. The writer is told to judge the reasoning, not the result, so
  analysts do not learn to overreact to noise.

Point-in-time use is enforced by ``journal.lessons.LessonStore``, which only returns
lessons whose forecast had resolved by the query date (P3).
"""

from __future__ import annotations

from ..config import FirmConfig
from ..llm.client import LLM, BudgetExceeded, LLMCallFailed
from ..schemas import ForecastRecord, Lesson
from .analysts import BLIND_ALIAS, blind_ids_in_text

FORECAST_PROB_LABEL = "Forecast probability that the stock beats the benchmark"
OUTCOME_LABEL = "Outcome"
OUTCOME_BEAT = "the stock BEAT the benchmark"
OUTCOME_MISSED = "the stock did NOT beat the benchmark"

LESSON_SYSTEM = """\
You review one resolved forecast made by an analyst on an equity investment committee and write one reusable process lesson for that analyst.

You are given the forecast (the probability that the stock beats the benchmark over the horizon and the expected excess return), the analyst's reasoning at the time, and the realised outcome.

HOW TO JUDGE
- A single binary outcome is weak evidence. A 0.60 forecast is supposed to be wrong 40% of the time. Judge the process: was the probability justified by the cited evidence, was it too far from (or too close to) 0.50, did the scenarios cover what happened, was the decisive risk named, were data gaps or unsupported numbers ignored?
- was_it_skill_or_luck: say which and why, in one or two sentences. If the reasoning was sound and the outcome fell inside the stated scenarios, the result was mostly luck, good or bad.
- what_happened: the forecast, the outcome and the realised excess return, in one or two sentences.
- process_lesson: one general, actionable rule about how to analyse, for example how much weight a type of evidence deserves or when to stay near 0.50. If no change is warranted, write "Keep the process:" followed by the reason. Do not overreact to one outcome.
- applies_when: the observable conditions under which the lesson applies (for example "a stock with RSI above 70 after a strong month"), so it can be matched to future cases.

RULES
- The lesson will be shown to the analyst when it forecasts other, unrelated stocks. Do not name the company, its ticker or any date, and do not predict what this stock or the market will do.
- Use only the information given; do not add numbers from memory.
- Never mention position sizes, weights or trades."""


def lesson_user_prompt(forecast: ForecastRecord, view_summary: str, blind: bool) -> str:
    outcome = OUTCOME_BEAT if forecast.outcome == 1 else OUTCOME_MISSED
    brier = forecast.brier if forecast.brier is not None else (forecast.prob_outperform - float(forecast.outcome or 0)) ** 2
    realized = (f" (realised excess return {forecast.realized_excess_return_pct:+.2f}%)"
                if forecast.realized_excess_return_pct is not None else "")
    stock = f"{BLIND_ALIAS} (identity withheld)" if blind else forecast.ticker
    when = "" if blind else f" | made on {forecast.as_of}, resolved on {forecast.resolve_on}"
    summary = blind_ids_in_text(view_summary, forecast.ticker) if blind else view_summary
    return "\n".join([
        "RESOLVED FORECAST",
        f"Forecaster: {forecast.agent}",
        f"Stock: {stock} | benchmark: {forecast.benchmark} | horizon: {forecast.horizon_days} trading days{when}",
        f"{FORECAST_PROB_LABEL}: {forecast.prob_outperform:.3f}",
        f"Expected excess return: {forecast.expected_excess_return_pct:+.2f}%",
        f"{OUTCOME_LABEL}: {outcome}{realized}",
        f"Brier score: {brier:.3f} (a constant 0.50 forecast always scores 0.250; lower is better)",
        "",
        "THE FORECASTER'S VIEW AT THE TIME",
        summary or "(no summary recorded)",
        "",
        "Write the lesson.",
    ])


def write_lesson(llm: LLM, forecast: ForecastRecord, view_summary: str, cfg: FirmConfig) -> Lesson | None:
    """Write a process lesson for a resolved forecast. Returns ``None`` for an unresolved
    forecast or when the call fails (a missing lesson is harmless; a fabricated one is not)."""
    if not forecast.resolved or forecast.outcome is None:
        return None
    try:
        lesson, _ = llm.structured(role="lesson", model=cfg.models.committee_model, system=LESSON_SYSTEM,
                                   user=lesson_user_prompt(forecast, view_summary, cfg.process.blind_mode),
                                   schema=Lesson, effort=cfg.models.analyst_effort, max_tokens=cfg.models.max_tokens)
    except (LLMCallFailed, BudgetExceeded):
        return None
    lesson.agent, lesson.ticker = forecast.agent, forecast.ticker
    return lesson
