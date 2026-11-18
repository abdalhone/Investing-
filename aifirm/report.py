"""Markdown reports for humans: one per decision cycle and one per backtest.

The reports are written for the person who has to approve, question or reject what the
firm did, so they lead with what matters and say plainly what the numbers do *not* show:

* The decision report puts the pooled probability next to the equal-weight shadow
  (P7), shows disagreement and whether a challenge ran (P4, P6), the chair's
  recommendation and edge multiplier, how each risk layer changed the book (P1, P9),
  the orders, LLM cost, and every data gap the agents were told about (P2).
* The backtest report compares the strategy with its baselines through the same engine,
  states the deflated Sharpe ratio for the number of configurations tried, separates
  results the model may have memorised (pre training cutoff) from clean ones, and ends
  with an "honest interpretation" written from the numbers, not from hope (P10).

Pure functions: they read a :class:`~aifirm.pipeline.Decision` or a
:class:`~aifirm.backtest.engine.BacktestResult` and return a string.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeGuard

from .backtest.engine import BacktestResult
from .backtest.metrics import TRADING_DAYS, PerformanceReport
from .pipeline import Decision
from .schemas import ICMemo, TargetPortfolio

DSR_BAR = 0.95            # Bailey & Lopez de Prado: below this, skill is not distinguishable from luck
SHORT_SAMPLE_YEARS = 3.0
LOW_EXPOSURE = 0.5        # below this average invested fraction, "beating the benchmark" is mostly cash timing
DISCLAIMER = ("Research software, not investment advice. Forecasts are probabilities, most of them close to a coin "
              "flip; past or simulated results do not predict future returns.")


# --------------------------------------------------------------------------- formatting helpers


def _finite(x: object) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _num(x: object, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}" if _finite(x) else "n/a"


def _pct(fraction: object, digits: int = 1) -> str:
    """A fraction shown as a percentage (0.123 -> 12.3%)."""
    return f"{float(fraction) * 100:.{digits}f}%" if _finite(fraction) else "n/a"


def _pp(percent: object, digits: int = 1, sign: bool = False) -> str:
    """A number that is already in percent (12.3 -> 12.3%)."""
    if not _finite(percent):
        return "n/a"
    return f"{float(percent):+.{digits}f}%" if sign else f"{float(percent):.{digits}f}%"


def _usd(x: object) -> str:
    return f"${float(x):,.2f}" if _finite(x) else "n/a"


def _cell(text: object) -> str:
    """Make arbitrary text safe inside a markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def _bullets(items: Iterable[object], empty: str = "none") -> list[str]:
    lines = [f"- {_cell(x)}" for x in items]
    return lines or [f"- {empty}"]


# --------------------------------------------------------------------------- decision report


def _memo_section(ticker: str, memo: ICMemo, multiplier: float, challenger: str | None,
                  challenge_text: str) -> list[str]:
    out = [f"### {ticker}: {memo.recommendation} (edge multiplier {multiplier:.2f})", "",
           f"**Thesis.** {memo.thesis_one_sentence}", "",
           f"**Variant perception.** {memo.variant_perception}", "",
           f"**Base rate.** {memo.base_rate_note}", ""]
    if challenger:
        out += [f"**Challenge** (by the {challenger} analyst). {challenge_text}", ""]
    out += ["**Catalysts**", "", *_bullets(memo.catalysts), "", "**Pre-mortem**", "", *_bullets(memo.premortem), ""]
    if memo.kill_criteria:
        out += ["**Kill criteria**", "", *_table(["Metric", "Condition", "Rationale"],
                                             [(k.metric, k.condition, k.rationale) for k in memo.kill_criteria]), ""]
    if memo.checklist:
        out += ["**Checklist**", "", *_table(["Item", "Passed", "Note"],
                                         [(c.item, "yes" if c.passed else "NO", c.note) for c in memo.checklist]), ""]
    out += [f"**Multiplier reason.** {memo.multiplier_reason}", ""]
    return out


def _weights_rows(stages: Sequence[TargetPortfolio], bench: str) -> list[list[str]]:
    tickers = sorted({p.ticker for s in stages for p in s.positions if p.weight != 0}
                     | {p.ticker for p in stages[-1].positions})
    final = {p.ticker: p for p in stages[-1].positions}
    rows = []
    for t in tickers:
        ws = [s.weights().get(t, 0.0) for s in stages]
        if all(w == 0 for w in ws):
            continue
        hits = ", ".join(final[t].constraints_hit) if t in final else ""
        rows.append([t, *(_pct(w, 2) for w in ws), hits or "-"])
    rows.append([f"{bench} (core)", *(_pct(s.core_weight, 2) for s in stages), "-"])
    rows.append(["cash", *(_pct(s.cash_weight, 2) for s in stages), "-"])
    return rows


def render_decision_report(decision: Decision) -> str:
    """Markdown report of one decision cycle (see the module docstring)."""
    d = decision
    bench = d.benchmark
    final_w = d.target_final.weights()
    lines = [f"# Decision report: {d.as_of}", "",
             f"Run `{d.run_id}` | mode **{d.mode}** | config `{d.config_fingerprint}` | "
             f"ledger head `{d.ledger_head}`", ""]
    if d.mode == "offline":
        lines += ["> **OFFLINE MODE.** The analysts, chair and risk reviewer were deterministic rules, not AI. "
                  "This report tests the plumbing; do not act on it.", ""]
    lines += [f"> {DISCLAIMER}", ""]

    gross = sum(abs(w) for w in final_w.values())
    lines += ["## Summary", "",
              f"- NAV at the close: {_usd(d.nav)}",
              f"- Final book: active gross {_pct(gross)}, core {_pct(d.target_final.core_weight)}, "
              f"cash {_pct(d.target_final.cash_weight)}",
              f"- Orders: {len(d.orders)}" + (" (to be filled at the next trading day's open)" if d.orders else ""),
              f"- Risk-engine messages: {len(d.risk_messages)}; warnings: {len(d.warnings)}",
              f"- LLM cost this cycle: {_usd(d.usage.get('cost_usd', 0.0))} over {d.usage.get('calls', 0)} calls",
              f"- Forecasts written: {len(d.forecasts)}; resolved this cycle: {len(d.resolved)}; "
              f"lessons written: {len(d.lessons_written)}"]
    if d.held_by_outage:
        lines.append("- **The book was held unchanged because of an LLM outage** (see warnings).")
    if d.risk_state.frozen:
        lines.append("- **The book is FROZEN by the drawdown rule. Only a human can reset it.**")
    lines.append("")

    lines += ["## Committee", "",
              "Pooled P is the believability-weighted, trimmed log-odds mean; equal-weight P is the same pool with "
              "unit weights, shown so you can see what the weights changed. Dispersion is the standard deviation of "
              "member probabilities (a challenge round runs above the threshold).", ""]
    rows = []
    for t, r in d.committee.items():
        a = r.aggregate
        rec = r.memo.recommendation if r.memo else "-"
        rows.append([t, _num(a.prob_outperform, 3), _num(a.prob_equal_weight, 3), _num(a.dispersion, 3),
                     f"{a.n_views} ({a.n_abstained})", "yes" if a.challenged else "no", a.flips, rec,
                     _num(r.edge_multiplier, 2), _pct(final_w.get(t, 0.0), 2)])
    lines += [*_table(["Ticker", "Pooled P", "Equal-wt P", "Dispersion", "Views (abstained)", "Challenged",
                       "Flips", "Memo", "Multiplier", "Final weight"], rows), ""]

    lines += ["### Believability weights", "",
              *_table(["Agent", "Weight", "Lessons in prompt"],
                      [(a, _num(w, 2), d.lessons_used.get(a, "-")) for a, w in d.believability.items()]), ""]

    lines += ["### Views", ""]
    for t, r in d.committee.items():
        lines += [f"**{t}**", ""]
        vrows = []
        issues = []
        for v in r.views:
            revised = _num(v.revised_prob, 3) if v.revised_prob is not None else "-"
            status = "abstained" if v.view.abstain else ("ok" if v.grounding_ok else "FLAGGED")
            vrows.append([v.agent, v.model, v.sample, _num(v.view.prob_outperform, 3), revised,
                          _pp(v.view.expected_excess_return_pct, 2, sign=True), _num(v.weight, 2), status])
            issues += [f"{v.agent}: {i}" for i in v.grounding_issues]
            if v.view.abstain:
                issues.append(f"{v.agent} abstained: {v.view.abstain_reason}")
        lines += [*_table(["Agent", "Model", "Sample", "Sealed P", "Revised P", "E[excess]", "Weight", "Grounding"],
                          vrows), ""]
        if issues:
            lines += [*_bullets(issues), ""]

    memos = [(t, r, r.memo) for t, r in d.committee.items() if r.memo is not None]
    lines += ["## IC memos", ""]
    if not memos:
        lines += ["No memo was required (no edge reached the memo threshold and no name was held), or memos are "
                  "disabled.", ""]
    for t, r, memo in memos:
        text = r.challenge.strongest_counterargument if r.challenge else ""
        lines += _memo_section(t, memo, r.edge_multiplier, r.challenger, text)

    lines += ["## Portfolio", "",
              "Code sizes every position (fractional Kelly, caps, volatility target). The LLM risk review can only "
              "shrink weights, and the deterministic risk engine has the final say.", "",
              *_table(["Ticker", "Proposed", "After risk review", "Final", "Constraints hit"],
                      _weights_rows([d.target_proposed, d.target_reviewed, d.target_final], bench)), ""]
    notes = list(dict.fromkeys(d.target_final.notes))
    if notes:
        lines += ["Notes:", *_bullets(notes), ""]

    lines += ["## LLM risk review", ""]
    if d.risk_review is None:
        lines += ["No review was performed (disabled, nothing to review, held book, or the call failed; the "
                  "deterministic risk engine still applied).", ""]
    else:
        rv = d.risk_review
        lines += [f"{rv.summary}", "", f"Halt trading: **{'YES' if rv.halt_trading else 'no'}**", "",
                  "Hidden concentrations:", *_bullets(rv.hidden_concentrations), ""]
        if rv.concerns:
            lines += [*_table(["Ticker", "Multiplier", "Concern"],
                              [(c.ticker, _num(c.multiplier, 2), c.concern) for c in rv.concerns]), ""]

    lines += ["## Risk engine", "", *_bullets(d.risk_messages, empty="no rule triggered"), ""]

    lines += ["## Orders", ""]
    if d.orders:
        lines += [*_table(["Side", "Ticker", "Quantity", "Reference price", "Notional", "Reason"],
                        [(o.side, o.ticker, _num(o.quantity, 4), _usd(o.reference_price), _usd(o.notional), o.reason)
                         for o in d.orders]), ""]
    else:
        lines += ["No orders.", ""]
    if d.order_messages:
        lines += ["Order messages:", *_bullets(d.order_messages), ""]

    lines += ["## Forecast track record", ""]
    if d.resolved:
        by_agent: dict[str, list[float]] = {}
        for fr in d.resolved:
            if fr.brier is not None:
                by_agent.setdefault(fr.agent, []).append(fr.brier)
        lines += ["Resolved this cycle (Brier: 0 is perfect, 0.25 is a constant 0.5 forecast):", "",
                  *_table(["Agent", "Resolved", "Mean Brier"],
                          [(a, len(b), _num(sum(b) / len(b), 4)) for a, b in sorted(by_agent.items())]), ""]
    else:
        lines += ["No forecast resolved this cycle.", ""]
    if d.lessons_written:
        lines += ["Lessons written:", *_bullets(f"{x.agent}: {x.process_lesson}" for x in d.lessons_written), ""]

    u = d.usage
    lines += ["## LLM usage", "",
              *_table(["Calls", "Live", "Cached", "Input tokens", "Output tokens", "Cost", "Failures"],
                      [(u.get("calls", 0), u.get("live_calls", 0), u.get("cached_calls", 0), u.get("input_tokens", 0),
                        u.get("output_tokens", 0), _usd(u.get("cost_usd", 0.0)), u.get("failures", 0))]), ""]

    lines += ["## Warnings and data gaps", "", *_bullets(d.warnings, empty="no warnings"), ""]
    for t, gaps in d.data_gaps.items():
        lines += [f"**{t}**", *_bullets(gaps), ""]
    lines += [f"Record the ledger head `{d.ledger_head}` somewhere else; `aifirm verify-ledger --anchor <head>` "
              "later proves no entry up to this point was edited or removed.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- backtest report

_METRICS: list[tuple[str, str, str]] = [
    ("Total return", "total_return_pct", "pp"),
    ("CAGR", "cagr_pct", "pp"),
    ("Volatility (annual)", "vol_pct", "pp"),
    ("Sharpe", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Max drawdown", "max_drawdown_pct", "pp"),
    ("Calmar", "calmar", "num"),
    ("Up days", "hit_rate_pct", "pp"),
    ("PSR (P[true Sharpe > 0])", "psr", "num"),
    ("Excess return vs benchmark", "excess_return_pct", "pp+"),
    ("Information ratio", "information_ratio", "num"),
    ("Beta", "beta", "num"),
    ("Alpha (annual)", "alpha_annual_pct", "pp+"),
    ("Avg turnover per rebalance", "avg_turnover_pct", "pp"),
    ("Total costs (% of NAV)", "total_costs_pct", "pp2"),
    ("Deflated Sharpe (DSR)", "deflated_sharpe", "num"),
    ("Min track record (days)", "min_track_record_days", "int"),
]

_NAMES = {"strategy": "Strategy", "benchmark": "Benchmark buy-and-hold", "equal_weight": "Equal weight",
          "quant_only": "Quant only"}


def _fmt_metric(rep: PerformanceReport, attr: str, kind: str) -> str:
    x = getattr(rep, attr)
    if x is None:
        return "-"
    if kind == "pp":
        return _pp(x)
    if kind == "pp2":
        return _pp(x, 3)
    if kind == "pp+":
        return _pp(x, 1, sign=True)
    if kind == "int":
        return "inf" if x == math.inf else (f"{x:,.0f}" if _finite(x) else "n/a")
    return _num(x)


def _metrics_table(reports: dict[str, PerformanceReport]) -> list[str]:
    names = [n for n in ("strategy", "benchmark", "equal_weight", "quant_only") if n in reports]
    rows = [[label, *(_fmt_metric(reports[n], attr, kind) for n in names)] for label, attr, kind in _METRICS]
    return _table(["Metric", *(_NAMES[n] for n in names)], rows)


def _compare(name: str, s: PerformanceReport, b: PerformanceReport) -> str:
    diff = s.total_return_pct - b.total_return_pct
    verb = "beat" if diff > 0 else "trailed"
    return (f"The strategy {verb} the {name} baseline by {abs(diff):.1f} percentage points of total return "
            f"(Sharpe {_num(s.sharpe)} vs {_num(b.sharpe)}; max drawdown {_pp(s.max_drawdown_pct)} vs "
            f"{_pp(b.max_drawdown_pct)}).")


def _interpretation(r: BacktestResult) -> list[str]:
    s = r.reports["strategy"]
    out: list[str] = []
    if r.mode == "offline":
        out.append("This run used the offline rules, not an AI model. Its numbers test the plumbing (data, sizing, "
                   "risk, accounting), not whether language models can pick stocks.")
    if s.deflated_sharpe is not None and _finite(s.deflated_sharpe):
        if s.deflated_sharpe < DSR_BAR:
            out.append(f"Deflated Sharpe {s.deflated_sharpe:.2f} is below {DSR_BAR} after {r.n_trials} recorded "
                       "configurations: the result is not distinguishable from luck.")
        else:
            out.append(f"Deflated Sharpe {s.deflated_sharpe:.2f} clears {DSR_BAR} after {r.n_trials} recorded "
                       "configurations. That is necessary, not sufficient: it assumes every variant you tried is in "
                       "the trial ledger.")
    else:
        out.append(f"Only {r.n_trials} configuration is in the trial ledger, so the deflated Sharpe is not computed; "
                   f"PSR = {_num(s.psr)} is the probability that the true Sharpe is above zero for this single try. "
                   "Every variant you test raises the bar; keep them all in runs/trials.jsonl.")
    for key, label in (("benchmark", "benchmark buy-and-hold"), ("equal_weight", "equal-weight"),
                       ("quant_only", "quant-only")):
        if key in r.reports:
            out.append(_compare(label, s, r.reports[key]))
    invested = [1.0 - float(x["cash_weight"]) for x in r.decisions if _finite(x.get("cash_weight"))]
    if invested and sum(invested) / len(invested) < LOW_EXPOSURE:
        avg = sum(invested) / len(invested)
        out.append(f"The strategy was on average only {avg:.0%} invested (beta {_num(s.beta)}). Its return relative "
                   "to the benchmark mostly reflects how much cash it held, not stock selection; the information "
                   "ratio and the comparison with the equal-weight baseline say more.")
    q = r.reports.get("quant_only")
    if q is not None and s.total_return_pct <= q.total_return_pct and s.sharpe <= q.sharpe:
        out.append("The full committee did not beat the deterministic quant analyst on return or Sharpe: in this "
                   "test the language-model analysts added no measurable value.")
    mtrl = s.min_track_record_days
    if mtrl is not None and (mtrl == math.inf or (_finite(mtrl) and mtrl > s.days)):
        need = "an unbounded number of" if mtrl == math.inf else f"about {mtrl:,.0f}"
        out.append(f"At this Sharpe, {need} trading days would be needed for 95% confidence that the true Sharpe is "
                   f"positive; the backtest has {s.days}.")
    years = s.days / TRADING_DAYS
    if years < SHORT_SAMPLE_YEARS:
        out.append(f"The sample is short ({years:.1f} years) and covers few market regimes.")
    if r.model_cutoff is not None:
        if r.post_cutoff_report is None:
            out.append(f"Every decision was made on or before the model's training cutoff ({r.model_cutoff}). The "
                       "model may have known these outcomes, so this backtest is NOT evidence of skill. Only "
                       "decisions after the cutoff (forward paper trading) can be.")
        else:
            p = r.post_cutoff_report
            out.append(f"Only the post-cutoff period ({p.days} days, total return {_pp(p.total_return_pct)}, "
                       f"Sharpe {_num(p.sharpe)}) is free of training contamination; judge the strategy on that.")
    committee = r.scorecard.get("committee")
    if committee:
        skill = committee.get("brier_skill")
        if _finite(skill) and skill <= 0:
            out.append(f"The committee's forecasts had no skill over a constant 0.5 (Brier skill {skill:+.3f} over "
                       f"{committee['n']} forecasts).")
        elif _finite(skill):
            out.append(f"The committee's Brier skill was {skill:+.3f} over {committee['n']} forecasts; one-month "
                       "equity forecasts with any positive skill are rare, so check the calibration table.")
    out += ["The universe was chosen today, so it is biased toward survivors; delisted losers are missing.",
            "Costs are modelled (commission, spread, square-root impact), not observed; taxes and borrow fees are "
            "ignored.",
            "Before real money: months of forward paper trading, a human review of every memo, and legal advice."]
    return out


def render_backtest_report(result: BacktestResult) -> str:
    """Markdown report of a backtest (see the module docstring)."""
    r = result
    s = r.reports["strategy"]
    lines = [f"# Backtest report: {r.start} to {r.end} ({r.mode} mode)", "",
             f"Run `{r.run_id}` | config `{r.config_fingerprint}` | trials recorded: {r.n_trials} | "
             f"artifacts: `{r.run_dir}`", "", f"> {DISCLAIMER}", ""]
    if r.model_cutoff is not None and r.pre_cutoff_report is not None:
        lines += [f"> **CONTAMINATION WARNING.** {r.pre_cutoff_report.days} of {s.days} trading days rest on "
                  f"decisions made on or before the model's training cutoff ({r.model_cutoff}). The model may have "
                  "seen what happened next. Those returns are not evidence of skill.", ""]
    if r.mode == "offline":
        lines += ["> **OFFLINE MODE.** Deterministic rules stood in for every LLM. Treat this as a test of the "
                  "machinery only.", ""]

    lines += ["## Performance vs baselines", "",
              "Daily returns net of costs. All traded baselines run through the same engine (next-open fills, same "
              "costs, same rebalance dates); the benchmark is bought and held without costs.", "",
              *_metrics_table(r.reports), ""]

    lines += ["## Statistical significance", "",
              f"- Probabilistic Sharpe ratio: {_num(s.psr)} (probability the true Sharpe is above 0, adjusting for "
              "sample length, skew and fat tails)",
              (f"- Deflated Sharpe ratio: {_num(s.deflated_sharpe)} for {r.n_trials} distinct configurations tried "
               f"(bar: {DSR_BAR})" if s.deflated_sharpe is not None else
               "- Deflated Sharpe ratio: not computed, only one configuration is in the trial ledger"),
              f"- Minimum track record for 95% confidence: {_fmt_metric(s, 'min_track_record_days', 'int')} days "
              f"(have {s.days})", ""]

    lines += ["## Pre / post model-cutoff split", ""]
    if r.model_cutoff is None:
        lines += ["Not applicable: no language model was used (quant mode).", ""]
    else:
        lines += [f"Model training cutoff: **{r.model_cutoff}**. Returns are assigned to the decision in force; a "
                  "decision made on or before the cutoff is contaminated (the model may have memorised the outcome).",
                  ""]
        split = {k: v for k, v in (("Pre-cutoff (contaminated)", r.pre_cutoff_report),
                                   ("Post-cutoff (clean)", r.post_cutoff_report)) if v is not None}
        if split:
            lines += [*_table(["Metric", *split],
                              [[label, *(_fmt_metric(v, attr, kind) for v in split.values())]
                               for label, attr, kind in _METRICS[:10]]), ""]
        if r.post_cutoff_report is None:
            lines += ["There is no clean (post-cutoff) period long enough to report.", ""]

    n_rebal = len(r.decisions)
    lines += ["## Trading activity and costs", "",
              f"- Decision cycles: {n_rebal}; cycles with orders: {sum(1 for x in r.decisions if x.get('orders'))}",
              f"- Average turnover per filled rebalance: {_pct(r.turnover.mean()) if len(r.turnover) else 'n/a'}",
              f"- Commission paid: {_usd(r.costs_usd.get('commission'))}; slippage and impact: "
              f"{_usd(r.costs_usd.get('slippage'))}",
              f"- Total costs: {_pp(s.total_costs_pct, 3)} of NAV",
              f"- Dividends credited to the strategy book: {_usd(r.dividends_usd)} (books are marked at raw "
              "closes; dividend cash makes their returns total returns, like the benchmark's)",
              f"- Cycles held by an LLM outage: {sum(1 for x in r.decisions if x.get('held_by_outage'))}; "
              f"book frozen at the end: {'YES' if r.decisions and r.decisions[-1].get('frozen') else 'no'}", ""]

    lines += ["## Forecast scorecard", ""]
    if r.scorecard:
        lines += [*_table(["Agent", "Forecasts", "Pre-cutoff", "Brier", "Brier skill", "Hit rate", "Mean P",
                           "Base rate"],
                          [(a, c["n"], c["n_pre_cutoff"], _num(c["brier"], 4), _num(c["brier_skill"], 3),
                            _pct(c["hit_rate"]) if c["hit_rate"] is not None else "-", _num(c["mean_prob"], 3),
                            _pct(c["base_rate"])) for a, c in r.scorecard.items()]), ""]
        cal = (r.scorecard.get("committee") or {}).get("calibration") or []
        if cal:
            lines += ["Committee calibration (a calibrated forecaster's observed rate matches its mean P):", "",
                      *_table(["Bucket", "n", "Mean P", "Observed rate"],
                              [(b["bucket"], b["n"], _num(b["mean_prob"], 3), _pct(b["observed_rate"])) for b in cal]),
                      ""]
    else:
        lines += ["No forecast resolved within the backtest window.", ""]

    if r.usage:
        u = r.usage
        lines += ["## LLM usage", "", f"- Calls: {u.get('calls', 0)} (live {u.get('live_calls', 0)}, cached "
                  f"{u.get('cached_calls', 0)}, failures {u.get('failures', 0)}); cost {_usd(u.get('cost_usd', 0.0))}",
                  ""]

    lines += ["## Honest interpretation", "", *_bullets(_interpretation(r)), ""]
    if r.warnings:
        shown = r.warnings[:30]
        lines += ["## Warnings and data gaps", "", *_bullets(shown)]
        if len(r.warnings) > len(shown):
            lines.append(f"- ... and {len(r.warnings) - len(shown)} more")
        lines.append("")
    return "\n".join(lines)
