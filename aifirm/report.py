"""Markdown reports for humans: one per decision cycle and one per backtest.

The reports are written for the person who has to approve, question or reject what the
firm did, so they lead with what matters and say plainly what the numbers do *not* show:

* The decision report puts the pooled probability next to the equal-weight shadow
  (P7), shows the quant core, the LLM pool and the bounded tilt in overlay mode, the
  screen (universe, eligible names, deep-dive set and the top of the ranking),
  disagreement and whether a challenge ran (P4, P6), the chair's recommendation and edge
  multiplier, how each risk layer changed the book (P1, P9), the exposure (names, active
  gross, core, cash), the orders, LLM cost, every data gap the agents were told
  about (P2) and, when the run kept one, the quant-only shadow book: live vs shadow NAV
  and return, cumulative LLM spend, the AI value added net of that spend, and the paired
  per-date IC with a plain verdict (RESEARCH 6.6, 7.4 S6).
* The backtest report opens with the "six lines" (window and assets; model, cutoff and
  whether the test lies after it; cost model and breakeven; configurations tried; the
  deflated Sharpe at that count; benchmark and composition-matched null), compares the
  strategy with its baselines through the same engine, answers "is there skill, and
  would we know?" (signal IC with t-stats and paired tests, power in years, random-book
  percentile, cost sensitivity, factor alpha), separates results the model may have
  memorised (before its training cutoff plus a buffer) from clean ones, and ends with an
  "honest interpretation" written from the numbers, not from hope (P10). Outside quant
  mode it states the language models' value added over the quant-only baseline (same
  screen, limits and costs) net of LLM spend. It states the
  average invested fraction (cash is never mistaken for stock selection) and, when the
  universe comes from a list of current index members, the survivorship bias up front.

Pure functions: they read a :class:`~aifirm.pipeline.Decision` or a
:class:`~aifirm.backtest.engine.BacktestResult` and return a string.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeGuard

from typing import TYPE_CHECKING

from .backtest.engine import BacktestResult
from .backtest.metrics import TRADING_DAYS, PerformanceReport
from .pipeline import Decision
from .schemas import ICMemo, TargetPortfolio

if TYPE_CHECKING:
    from .shadow import BookComparison, ShadowOutcome

DSR_BAR = 0.95            # Bailey & Lopez de Prado: below this, skill is not distinguishable from luck
SHORT_SAMPLE_YEARS = 3.0
LOW_EXPOSURE = 0.5        # below this average invested fraction, "beating the benchmark" is mostly cash timing
BROAD_UNIVERSE = 50       # names per date from which a universe counts as broad for the power reading
# Fewest resolved dates behind an IC (or a paired IC difference) before the report reads a
# direction into it. With 2-3 dates a t-stat has 1-2 degrees of freedom and |t| >= 2 is
# routine noise; the same minimum gates the adaptive overlay bound (process.overlay_min_dates).
MIN_VERDICT_DATES = 12
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


SCREEN_ROWS = 20          # ranking rows shown in a decision report (the ledger holds the full ranking)


def _screen_section(d: Decision) -> list[str]:
    sc = d.screen
    if sc is None:
        return []
    out = ["## Screen", "",
           f"The deterministic quant ranked {sc.n_universe} names; {sc.n_eligible} passed the point-in-time filters "
           f"(price >= {sc.min_price:g}, 20-day dollar volume >= {_usd(sc.min_dollar_volume)}). The top "
           f"{sc.top_n} by quant probability (above 0.5) plus every held name went to the committee: "
           f"{', '.join(sc.deep_dive) or 'none'}. Other names get no new position.", ""]
    rows = [(e.rank if e.rank is not None else "-", e.ticker, _num(e.quant_prob, 3) if e.quant_prob is not None else "-",
             _usd(e.adv_usd) if e.adv_usd is not None else "-", e.deep_dive_reason or "-")
            for e in sc.entries[:SCREEN_ROWS]]
    out += [*_table(["Rank", "Ticker", "Quant P", "Dollar volume (20d)", "Deep dive"], rows), ""]
    excluded = [e for e in sc.entries if not e.eligible]
    if excluded:
        reasons: dict[str, int] = {}
        for e in excluded:
            key = e.reason.split(":")[0].split(" below")[0]
            reasons[key] = reasons.get(key, 0) + 1
        out += [f"Not eligible: {len(excluded)} name(s) (" + "; ".join(f"{k}: {n}" for k, n in sorted(reasons.items()))
                + ")", ""]
    return out


def _shadow_section(shadow: ShadowOutcome | None, comparison: BookComparison | None) -> list[str]:
    """The live book against its quant-only shadow (RESEARCH 6.6; :mod:`aifirm.shadow`)."""
    if shadow is None and comparison is None:
        return []
    out = ["## Quant-only shadow book", "",
           "The same configuration with every language-model stage removed (the deterministic quant alone), decided "
           "at the same close on the same data, with the same limits, costs and next-open fills, in its own paper "
           "book. The difference between the two books is what the language models did.", ""]
    if shadow is not None:
        if shadow.status == "failed":
            out += [f"> **The shadow book failed this cycle** ({_cell(shadow.reason)}). The live decision above is "
                    "unaffected; the comparison below has a gap for this session.", ""]
        elif shadow.status == "skipped":
            out += [f"Shadow not decided this cycle: {_cell(shadow.reason)}.", ""]
        else:
            ex = shadow.decision.exposure() if shadow.decision is not None else {}
            born = " It started this cycle from the live book's state." if shadow.inception else ""
            out += [f"Shadow decision `{shadow.run_id}`: NAV {_usd(shadow.nav)}, {shadow.n_orders} paper order(s), "
                    f"{int(ex.get('n_names', 0))} name(s), active gross {_pct(ex.get('active_gross'))}, cash "
                    f"{_pct(ex.get('cash'))}; {shadow.mirrored} forecast(s) recorded as agent `shadow` in this "
                    f"book's forecast store.{born} Book: `{shadow.book}`.", ""]
    c = comparison
    if c is None:
        return out
    if c.status in ("no_shadow", "no_common_decision"):
        return [*out, *(_cell(x) for x in c.lines()), ""]
    out += _table(["", "Live book", "Quant-only shadow", "Live minus shadow"],
                  [("NAV at the close of " + str(c.nav_date), _usd(c.live_nav), _usd(c.shadow_nav),
                    _usd_signed(c.difference_usd)),
                   (f"Total return since {c.inception_as_of}", _pp(c.live_return_pct, 2, sign=True),
                    _pp(c.shadow_return_pct, 2, sign=True), f"{_pp(c.difference_pp, 2, sign=True)} points"),
                   ("Cumulative LLM spend", _usd(c.llm_spend_usd), "$0.00", ""),
                   ("**AI value added net of LLM cost**", "", "", f"**{_usd_signed(c.value_added_usd)}**")])
    ic_t = _num(c.ic_t_stat, 2)
    out += ["", f"- {c.value_sentence()}",
            f"- Ranking: paired per-date rank IC of the live final probability against the shadow's, on the names "
            f"both forecast, from forecasts resolved by {c.as_of or c.nav_date}: {c.ic_dates} resolved date(s), mean "
            f"IC live {_num(c.ic_live, 3)} vs shadow {_num(c.ic_shadow, 3)}, difference {_num(c.ic_mean_diff, 3)} "
            f"(t = {ic_t}; " + (f"Newey-West, {c.ic_lags} lags" if c.ic_lags else "independent dates") + ").",
            f"- **Verdict: {c.verdict}**"]
    out += [f"- {_cell(n)}" for n in c.notes]
    return [*out, ""]


def _usd_signed(x: object) -> str:
    if not _finite(x):
        return "n/a"
    v = float(x)
    return f"{'-' if v < 0 else '+'}${abs(v):,.2f}"


def render_decision_report(decision: Decision, *, shadow: ShadowOutcome | None = None,
                           comparison: BookComparison | None = None) -> str:
    """Markdown report of one decision cycle (see the module docstring). With ``shadow``
    and/or ``comparison`` (from :mod:`aifirm.shadow`), a section compares the live book with
    its quant-only shadow book."""
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
              f"- Final book: {len(final_w)} name(s), active gross {_pct(gross)}, core "
              f"{_pct(d.target_final.core_weight)}, cash {_pct(d.target_final.cash_weight)}",
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

    lines += _screen_section(d)
    overlay = any(r.aggregate.combination == "overlay" for r in d.committee.values())
    lines += ["## Committee", ""]
    if overlay:
        b = d.overlay_bound
        bound = (f"The bound in force is {b.max_tilt:.3f} (configured {b.configured:.3f}): {b.reason}." if b else "")
        lines += ["Overlay mode (research 6.6): the final P is the quant core plus the LLM pool's own edge "
                  "(LLM P minus 0.5), clipped to at most the bound either way, so an LLM P of 0.5 (no information) "
                  "leaves the quant unchanged; LLM P pools the analysts' calibrated, tier-gated probabilities "
                  "(believability-weighted, trimmed log-odds mean). " + bound, ""]
    else:
        lines += ["Pooled P is the believability-weighted, trimmed log-odds mean; equal-weight P is the same pool with "
                  "unit weights, shown so you can see what the weights changed.", ""]
    lines += ["Dispersion is the standard deviation of member probabilities (a challenge round runs above the "
              "threshold).", ""]
    rows = []
    for t, r in d.committee.items():
        a = r.aggregate
        rec = r.memo.recommendation if r.memo else "-"
        lead = [t, _num(a.prob_outperform, 3)]
        if overlay:
            lead += [_num(a.prob_quant, 3) if a.prob_quant is not None else "-",
                     _num(a.prob_llm, 3) if a.prob_llm is not None else "-",
                     f"{a.overlay_tilt:+.3f}" if _finite(a.overlay_tilt) else "-"]
        rows.append([*lead, _num(a.prob_equal_weight, 3), _num(a.dispersion, 3),
                     f"{a.n_views} ({a.n_abstained})", "yes" if a.challenged else "no", a.flips, rec,
                     _num(r.edge_multiplier, 2), _pct(final_w.get(t, 0.0), 2)])
    head = ["Ticker", "Final P", "Quant P", "LLM P", "Tilt"] if overlay else ["Ticker", "Pooled P"]
    lines += [*_table([*head, "Equal-wt P", "Dispersion", "Views (abstained)", "Challenged",
                       "Flips", "Memo", "Multiplier", "Final weight"], rows), ""]
    gate = [n for r in d.committee.values() for n in r.tier_gate_notes]
    if gate:
        lines += ["Tier gate (unverified news cannot raise exposure):", *_bullets(gate), ""]
    fitted = {a: c for a, c in d.calibrators.items() if c.fitted and not c.is_identity}
    if fitted:
        lines += ["Calibration maps in force (logit p_cal = a + b logit p_raw, fitted on resolved forecasts only):", "",
                  *_table(["Agent", "a", "b", "Questions", "Brier raw", "Brier calibrated"],
                          [(a, _num(c.a, 3), _num(c.b, 3), c.n, _num(c.brier_raw, 4), _num(c.brier_calibrated, 4))
                           for a, c in fitted.items()]), ""]

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
        lines += [*_table(["Agent", "Model", "Sample", "Sealed P (raw)", "Revised P", "E[excess]", "Weight",
                           "Grounding"], vrows), ""]
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

    lines += _shadow_section(shadow, comparison)

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


# --------------------------------------------------------------------------- evaluation: six lines and skill


def _contamination_position(r: BacktestResult) -> str:
    """Where the test window lies relative to the end of the contaminated period."""
    end = r.contamination_end
    if end is None:
        return ""
    n_pre = sum(1 for d in r.decisions if str(d.get("as_of", "")) <= end)
    if r.decisions and n_pre == len(r.decisions):
        return "the whole test lies before it (contaminated: not evidence of skill)"
    if n_pre == 0:
        return "the whole test lies after it (clean)"
    return (f"the test straddles it: {n_pre} of {len(r.decisions)} decisions are contaminated; judge only the "
            "post-cutoff part")


def _model_line(r: BacktestResult) -> str:
    if r.model_cutoff is None:
        return ("**Model and cutoff:** none. The deterministic quant analyst decided alone, so there is no "
                "training-data contamination.")
    stand_in = " (deterministic offline rules stood in for it)" if r.mode == "offline" else ""
    models = ", ".join(r.models) or "unknown"
    return (f"**Model and cutoff:** {models}{stand_in}; training cutoff {r.model_cutoff} plus a "
            f"{r.cutoff_buffer_days}-day buffer, so decisions after {r.contamination_end} are clean; "
            f"{_contamination_position(r)}.")


def _cost_line(r: BacktestResult) -> str:
    c = r.cost_model
    model = (f"commission {c.get('commission_bps', float('nan')):g} bps + spread/slippage "
             f"{c.get('slippage_bps', float('nan')):g} bps + square-root impact "
             f"({c.get('impact_coefficient', float('nan')):g} x daily vol x sqrt(order / ADV))") if c else "n/a"
    costs = r.evaluation.costs if r.evaluation is not None else None
    if costs is None:
        return f"**Cost model and breakeven:** {model}; the breakeven could not be computed."
    return (f"**Cost model and breakeven:** {model}; realised {costs.realised_cost_bps:.1f} bps per dollar traded "
            f"(one-way); {costs.vs_benchmark.describe('the benchmark')}.")


def _null_line(r: BacktestResult) -> str:
    s = r.reports["strategy"]
    bench = f"{_pp(s.excess_return_pct, 1, sign=True)} total return vs {r.benchmark or 'the benchmark'} buy-and-hold"
    rb = r.evaluation.random_books if r.evaluation is not None else None
    if rb is None:
        why = ("switched off (evaluation.n_random_books = 0)" if r.evaluation is not None and
               not any("random books" in n for n in r.evaluation.notes) else "not computed (see evaluation notes)")
        return f"**Benchmark and null:** {bench}; composition-matched random books {why}."
    return f"**Benchmark and null:** {bench}; {rb.verdict()}."


def six_lines(r: BacktestResult) -> list[str]:
    """The six lines every backtest report states (research report 7.4, after 2611.18934):
    window and assets, model and cutoff, cost model and breakeven, configurations tried,
    the deflated statistic at that count, and a benchmark plus composition-matched null."""
    s = r.reports["strategy"]
    dsr = s.deflated_sharpe
    if dsr is not None and _finite(dsr):
        verdict = "clears" if dsr >= DSR_BAR else "is below"
        deflated = (f"**Deflated Sharpe at {r.n_trials} trials:** {dsr:.2f}, which {verdict} the {DSR_BAR} bar "
                    f"(Sharpe {_num(s.sharpe)}).")
    else:
        deflated = (f"**Deflated Sharpe at {r.n_trials} trial(s):** not computed for a single configuration; "
                    f"PSR (P[true Sharpe > 0]) = {_num(s.psr)} with Sharpe {_num(s.sharpe)}.")
    survivors = " SURVIVORSHIP-BIASED universe list (current members; see the warning above)." if r.universe_note \
        else ""
    return [
        f"**Window and assets:** {r.start} to {r.end} ({s.days} trading days, {s.days / TRADING_DAYS:.1f} years); "
        f"{len(r.tickers)} stocks plus the benchmark {r.benchmark or '?'}.{survivors}",
        _model_line(r),
        _cost_line(r),
        f"**Configurations tried:** {r.n_trials} distinct (configuration, mode, window) trial(s) in "
        f"`{r.trials_path or 'trials.jsonl'}`.",
        deflated,
        _null_line(r),
    ]


def _t_bar(r: BacktestResult) -> float:
    """The t-stat a reading needs: 3 once more than 20 trials are recorded (research 7.4 S4), else 2."""
    return 3.0 if r.n_trials > 20 else 2.0


def _ic_reading(t: object, mean: object, n_dates: object = None, bar: float = 2.0) -> str:
    """Table reading of an IC or a paired IC difference. No direction is read into fewer
    than :data:`MIN_VERDICT_DATES` dates ("anecdotal"), and a direction needs ``|t| >= bar``."""
    if not _finite(mean):
        return "n/a"
    if mean == 0 and not _finite(t):
        return "identical"
    if not _finite(t):
        return "n/a"
    if isinstance(n_dates, int) and n_dates < MIN_VERDICT_DATES:
        return f"anecdotal: {n_dates} date(s) < {MIN_VERDICT_DATES}"
    if t >= bar:
        return f"positive (t >= {bar:g})"
    if t <= -bar:
        return f"negative (t <= -{bar:g})"
    return "not distinguishable from zero"


def _skill_section(r: BacktestResult) -> list[str]:
    ev = r.evaluation
    out = ["## Is there skill, and would we know?", ""]
    if ev is None:
        return [*out, "The evaluation was not run.", ""]
    from .backtest.ic import POOL, label_name, overlap_lags, t_method
    from .backtest.random_books import ordinal

    lags = overlap_lags(ev.horizon_days, ev.step_days)
    lags_note = (f"Decision dates are {ev.step_days} trading days apart and forecasts run {ev.horizon_days} days, so "
                 f"consecutive windows overlap and t-stats use {t_method(lags)} standard errors." if lags else
                 "Forecast windows do not overlap, so t-stats are the plain mean / std x sqrt(dates).")
    out += ["### Signal: rank information coefficient (IC)", "",
            f"On each decision date, the Spearman rank correlation between a signal's P(beat the benchmark) and the "
            f"realised {ev.forward_return_definition}. {lags_note} The date is the unit of replication.", ""]
    if ev.clean_after is not None:
        out += [f"**Clean decisions only.** The IC, the paired differences, the power, the random books and the "
                f"factor alpha below use the {ev.n_clean_decisions} of {ev.n_decisions} decisions made after "
                f"{ev.clean_after} (training cutoff plus buffer). Earlier decisions may rest on memorised outcomes "
                "and are never evidence of skill (P10)" + ("." if ev.n_clean_decisions else
                                                           ": there are none, so these parts are empty."), ""]
    if ev.ic:
        out += [*_table(["Signal", "Dates", "Mean IC", "IC std", "t-stat", "Hit rate", "Names per date", "Reading"],
                        [(label_name(k), s.n_dates, _num(s.mean, 3), _num(s.std, 3), _num(s.t_stat, 2),
                          _pct(s.hit_rate, 0), _num(s.avg_names, 1),
                          _ic_reading(s.t_stat, s.mean, s.n_dates, _t_bar(r)))
                         for k, s in ev.ic.items()]), ""]
    else:
        out += ["No IC could be computed (no decision date had a forecast window that ended within the backtest).",
                ""]
    if ev.ic_differences:
        out += ["Paired per-date IC differences (the test for \"does A rank stocks better than B\"; research 7.4 "
                f"asks for t >= {_t_bar(r):g} with a consistent sign before preferring the more complex design, and "
                f"no direction is read into fewer than {MIN_VERDICT_DATES} dates):", "",
                *_table(["A minus B", "Dates", "Mean difference", "t-stat", "Reading"],
                        [(f"{label_name(d.a)} minus {label_name(d.b)}", d.n_dates, _num(d.mean_diff, 3),
                          _num(d.t_stat, 2), _ic_reading(d.t_stat, d.mean_diff, d.n_dates, _t_bar(r)))
                         for d in ev.ic_differences]), ""]

    if ev.contaminated_ic:
        out += ["Whole window, contaminated decisions included (for reference only: NOT evidence of skill):", "",
                *_table(["Signal", "Dates", "Mean IC", "t-stat"],
                        [(label_name(k), s.n_dates, _num(s.mean, 3), _num(s.t_stat, 2))
                         for k, s in ev.contaminated_ic.items()]),
                *(["", *_table(["A minus B (contaminated)", "Dates", "Mean difference", "t-stat"],
                               [(f"{label_name(d.a)} minus {label_name(d.b)}", d.n_dates, _num(d.mean_diff, 3),
                                 _num(d.t_stat, 2)) for d in ev.contaminated_ic_differences])]
                  if ev.contaminated_ic_differences else []), ""]

    out += ["### Power: how long until we would know", ""]
    p = ev.power
    if p is None:
        out += ["Not computed (no IC for the committee pool).", ""]
    else:
        out += [f"- {p.ic_sentence()}", f"- {p.sharpe_sentence()}"]
        if p.anecdotal:
            need = (f"fewer than {math.ceil(p.dates_for_80pct):,} dates" if _finite(p.dates_for_80pct)
                    else "the mean IC is not positive, so no number of dates at this IC would show skill")
            pool = (ev.ic or {}).get(POOL)
            breadth = pool.avg_names if pool is not None and _finite(pool.avg_names) else 0.0
            why = ("at monthly decisions on a narrow universe that takes decades (research 7.3); a broad screened "
                   "universe (universe.screen) is what buys power" if breadth < BROAD_UNIVERSE else
                   f"breadth ({breadth:.0f} names per date) helps only when the mean IC is positive (research 7.3)")
            out.append(f"- **Anecdotal: {need}.** This backtest cannot confirm skill; {why}.")
        out.append("")

    out += ["### Composition-matched random books", ""]
    rb = ev.random_books
    if rb is None:
        out += ["Not computed (switched off with evaluation.n_random_books = 0, or see the evaluation notes).", ""]
    else:
        matched = "same sector, " if rb.sector_matched else ""
        out += [f"Each of {rb.n_books:,} random books (seed {rb.seed}) copies the strategy's weights, number of names, "
                f"core, cash and holding periods at each of its {rb.n_rebalances} rebalances, but holds names drawn "
                f"at random ({matched}about {rb.avg_names_held:.1f} of the ~{rb.avg_eligible:.0f} names priced on "
                f"that date); every book "
                f"pays the strategy's realised {rb.cost_rate_bps:.1f} bps per dollar traded. The strategy is valued "
                f"the same way ({_pp(rb.strategy_total_return_pct)} here vs {_pp(rb.engine_total_return_pct)} in the "
                "engine).", "",
                *_table(["", "Strategy", "Random 5th pct", "Random median", "Random 95th pct", "Strategy percentile",
                         "p (one-sided)"],
                        [("Total return", _pp(rb.strategy_total_return_pct),
                          _pp(rb.return_quantiles_pct["p5"]), _pp(rb.return_quantiles_pct["p50"]),
                          _pp(rb.return_quantiles_pct["p95"]), f"{rb.percentile_return:.1f}",
                          _num(rb.p_value_return, 3)),
                         ("Sharpe", _num(rb.strategy_sharpe), _num(rb.sharpe_quantiles["p5"]),
                          _num(rb.sharpe_quantiles["p50"]), _num(rb.sharpe_quantiles["p95"]),
                          f"{rb.percentile_sharpe:.1f}", _num(rb.p_value_sharpe, 3))]), "",
                f"Reading: {rb.verdict()}. Gross of costs the strategy is at the "
                f"{ordinal(rb.gross_percentile_return)} percentile.", ""]
        out += [*(f"> {n}" for n in rb.notes), ""] if rb.notes else []

    out += ["### Cost sensitivity and breakeven", ""]
    c = ev.costs
    if c is None:
        out += ["Not computed (see the evaluation notes).", ""]
    else:
        has_q = c.quant_total_return_pct is not None
        rows = []
        for k in c.multipliers:
            vq = c.excess_vs_quant(k)
            rows.append([f"{k:g}x", _pp(c.strategy_total_return_pct[k]), _pp(c.excess_vs_benchmark(k), 1, sign=True),
                         *([_pp(vq, 1, sign=True)] if has_q else [])])
        out += ["The SAME recorded decisions re-executed through the engine with every cost component (commission, "
                "spread, impact) multiplied by k: no new decision and no LLM call. The benchmark is held without "
                "costs." + (" The quant-only baseline is re-executed at the same k." if has_q else ""), "",
                *_table(["Costs", "Strategy total return", "Excess vs benchmark",
                         *(["Excess vs quant-only"] if has_q else [])], rows), "",
                f"- Realised modelled cost: {c.realised_cost_bps:.1f} bps per dollar traded (one-way).",
                f"- Benchmark: {c.vs_benchmark.describe('the benchmark')}."]
        if c.vs_quant is not None:
            out.append(f"- Quant-only baseline: {c.vs_quant.describe('the quant-only baseline')}.")
        if not c.replay_matches_engine:
            out.append("- Warning: re-executing at 1x did not reproduce the engine exactly; treat these as estimates.")
        out.append("")

    out += ["### Factor attribution (ETF proxies)", ""]
    f = ev.factors
    if f is None:
        out += ["Not computed: no usable factor proxy prices (see the notes below).", ""]
    else:
        if f.mean_invested is not None:
            dep = (f"Daily strategy return minus the T-bill return ({f.cash_proxy or 'taken as 0'}) on the invested "
                   f"fraction only (on average {_pct(f.mean_invested, 0)} of NAV; the simulator pays no interest "
                   f"on cash, so the cash leg is compared with the 0 it earned)")
        else:
            dep = f"Daily strategy return minus cash ({f.cash_proxy or 'taken as 0'})"
        out += [f"{dep} regressed on factor returns, "
                f"{f.n_obs} days, Newey-West standard errors ({f.hac_lags} lags), R-squared {_num(f.r2, 2)}.", "",
                *_table(["Term", "Proxy", "Estimate", "t-stat (HAC)"],
                        [("Alpha (annual)", "-", _pp(f.alpha_annual_pct, 2, sign=True), _num(f.alpha_t, 2)),
                         *((name, f.proxies[name], _num(f.loadings[name], 3), _num(f.loading_t[name], 2))
                           for name in f.factors)]), ""]
        if f.cash_drag_annual_pct is not None:
            out += [f"Cash drag (not alpha): uninvested cash forwent {_pp(f.cash_drag_annual_pct, 2)} a year of "
                    "T-bill return. It is a cost of holding cash, reported here and not charged to the alpha above.",
                    ""]
    if ev.notes:
        out += ["Evaluation notes:", *_bullets(ev.notes), ""]
    return out


def _overlay_value_section(r: BacktestResult) -> list[str]:
    """The language models' value added over the quant-only baseline, net of LLM cost
    (RESEARCH 6.6, 9.3 item 6). Empty in quant mode (there is no LLM to judge)."""
    ov = r.overlay_value()
    if ov is None:
        return []
    from .backtest.ic import POOL, QUANT_ONLY

    out = ["## Value added by the language models (vs the quant-only baseline, net of LLM cost)", "",
           "The quant-only baseline is the same configuration with every language-model stage removed: the same "
           "universe and screen, limits, costs, rebalance dates and next-open fills. LLM spend is not charged to the "
           "simulated book, so it is subtracted here.", "",
           *_table(["", "Value"],
                   [("Starting NAV (both books)", _usd(ov.initial_nav)),
                    ("Final NAV, strategy", _usd(ov.strategy_nav)),
                    ("Final NAV, quant-only baseline", _usd(ov.quant_nav)),
                    ("Strategy minus baseline", _usd_signed(ov.difference_usd)),
                    ("LLM spend" + (" (offline rules: no spend)" if ov.offline else ""), _usd(ov.llm_cost_usd)),
                    ("**Value added net of LLM cost**",
                     f"**{_usd_signed(ov.value_added_usd)} ({_pp(ov.value_added_pct, 2, sign=True)} of the "
                     "starting NAV)**")]), ""]
    diff = next((d for d in (r.evaluation.ic_differences if r.evaluation is not None else [])
                 if d.a == POOL and d.b == QUANT_ONLY), None)
    ev = r.evaluation
    if diff is None and ev is not None and ev.clean_after is not None:
        out.append(f"- Ranking: no paired IC on clean dates (decisions after {ev.clean_after}); the whole-window "
                   "difference includes contaminated decisions and is not a test.")
    if diff is not None:
        out.append(f"- Ranking{' (clean dates only)' if ev is not None and ev.clean_after else ''}: the committee "
                   f"pool's per-date IC minus the quant-only baseline's is "
                   f"{_num(diff.mean_diff, 3)} on average over {diff.n_dates} dates (t = {_num(diff.t_stat, 2)}, "
                   f"{diff.method}): {_ic_reading(diff.t_stat, diff.mean_diff, diff.n_dates, _t_bar(r))}.")
    who = "the offline rules standing in for the language models" if ov.offline else "the language models"
    verdict = (f"{who} added value net of their cost in this backtest" if ov.value_added_usd > 0 else
               f"{who} did not pay for themselves in this backtest")
    caveat = (" Offline rules stood in for the model, so this is a plumbing check, not a measurement of an LLM."
              if ov.offline else
              " A NAV difference from one path is noisy and pre-cutoff decisions may be contaminated; the paired IC "
              "on clean dates and the post-cutoff split are the tests.")
    out += [f"- Reading: {verdict}.{caveat}", ""]
    return out


def _evaluation_interpretation(r: BacktestResult) -> list[str]:
    """Plain-language reading of the evaluation (skill, power, null, costs, factors)."""
    ev = r.evaluation
    if ev is None:
        return []
    from .backtest.ic import POOL, QUANT_ONLY

    out: list[str] = []
    pool = ev.ic.get(POOL)
    bar = _t_bar(r)                             # research 7.4 S4: t >= 3 for a pick from more than 20 trials
    clean = "clean " if ev.clean_after is not None else ""
    if ev.clean_after is not None and ev.n_clean_decisions == 0:
        out.append(f"No decision was made after {ev.clean_after} (the models' training cutoff plus buffer), so no "
                   "clean evidence of ranking skill exists in this backtest; the whole-window IC, random books and "
                   "factor alpha would include memorised outcomes and are not used as evidence.")
    elif pool is None or pool.n_dates == 0:
        out.append("No signal IC could be computed" + (f" on the clean decisions after {ev.clean_after}"
                                                       if ev.clean_after is not None else "")
                   + ", so the ranking skill of the forecasts is unmeasured.")
    elif pool.n_dates < MIN_VERDICT_DATES:
        out.append(f"The committee's ranking had a mean IC of {_num(pool.mean, 3)} over only {pool.n_dates} "
                   f"{clean}date(s) (t = {_num(pool.t_stat, 2)}): anecdotal. No verdict on ranking skill is read into "
                   f"fewer than {MIN_VERDICT_DATES} dates.")
    elif _finite(pool.t_stat) and pool.t_stat >= bar:
        out.append(f"The committee's ranking had a mean IC of {pool.mean:+.3f} over {pool.n_dates} "
                   f"{clean}dates (t = "
                   f"{pool.t_stat:.2f}, {pool.method}), above the t >= {bar:g} bar for {r.n_trials} recorded "
                   "configuration(s).")
    else:
        out.append(f"The committee's ranking had a mean IC of {_num(pool.mean, 3)} over {pool.n_dates} "
                   f"{clean}dates (t = "
                   f"{_num(pool.t_stat, 2)}, {pool.method}): not distinguishable from no skill (bar: t >= {bar:g}).")
    for d in ev.ic_differences:
        if d.a == POOL and d.b == QUANT_ONLY:
            if d.n_dates < MIN_VERDICT_DATES:
                out.append(f"Anecdotal: the LLM committee's ranking was compared with the quant-only model's on only "
                           f"{d.n_dates} {clean}date(s) (mean IC difference {_num(d.mean_diff, 3)}, t = "
                           f"{_num(d.t_stat, 2)}); no verdict is read into fewer than {MIN_VERDICT_DATES} dates, so by "
                           "research 7.4 default to the simpler design.")
            elif _finite(d.t_stat) and abs(d.t_stat) >= bar:
                better = "better" if d.mean_diff > 0 else "worse"
                out.append(f"The LLM committee ranked stocks {better} than the quant-only model (mean IC difference "
                           f"{d.mean_diff:+.3f}, t = {d.t_stat:.2f} over {d.n_dates} {clean}dates; bar t >= "
                           f"{bar:g}).")
            else:
                out.append(f"The LLM committee's ranking is not distinguishable from the quant-only model's (mean IC "
                           f"difference {_num(d.mean_diff, 3)}, t = {_num(d.t_stat, 2)}, bar t >= {bar:g}); by "
                           "research 7.4 default to the simpler design.")
    if ev.power is not None and ev.power.anecdotal:
        pw = ev.power
        if _finite(pw.dates_for_80pct):
            out.append(f"Anecdotal: fewer than {math.ceil(pw.dates_for_80pct):,} decision dates (the backtest has "
                       f"{pw.n_dates}). At this IC and breadth you would need ~{pw.years_for_80pct:,.0f} years of "
                       "decisions for 80% power, so the result cannot tell skill from luck.")
        else:
            out.append(f"Anecdotal: the mean IC over {pw.n_dates} dates is not positive, so no number of dates at "
                       "this IC would show skill.")
    if ev.random_books is not None:
        out.append(ev.random_books.verdict()[0].upper() + ev.random_books.verdict()[1:] + ".")
    c = ev.costs
    if c is not None:
        doubled = [k for k in c.multipliers if k >= 2.0]
        if c.vs_benchmark.status == "no_edge":
            out.append("Even with zero trading costs the strategy did not beat the benchmark, so costs are not what "
                       "held it back.")
        elif doubled and c.excess_vs_benchmark(doubled[0]) <= 0:
            out.append(f"The result does not survive {doubled[0]:g}x costs (excess vs the benchmark "
                       f"{_pp(c.excess_vs_benchmark(doubled[0]), 1, sign=True)}); research 7.4 asks that it does.")
    f = ev.factors
    if f is not None:
        if _finite(f.alpha_t) and f.alpha_t >= 2:
            out.append(f"Factor-adjusted alpha {_pp(f.alpha_annual_pct, 1, sign=True)} a year with t = "
                       f"{f.alpha_t:.2f} on {', '.join(f.factors)}: some return is not explained by the listed "
                       "factors (necessary, not sufficient).")
        elif _finite(f.alpha_t) and f.alpha_t <= -2:
            out.append(f"Factor-adjusted alpha is negative ({_pp(f.alpha_annual_pct, 1, sign=True)} a year, t = "
                       f"{f.alpha_t:.2f}): after its {', '.join(f.factors)} exposures the strategy lost to what the "
                       "same exposures bought through ETFs would have earned.")
        else:
            out.append(f"No factor-adjusted alpha at t >= 2 (alpha {_pp(f.alpha_annual_pct, 1, sign=True)} a year, "
                       f"t = {_num(f.alpha_t, 2)}): the return is explained by {', '.join(f.factors)} exposure or "
                       "noise.")
    return out


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
    ex = r.exposure_summary()
    avg = ex["invested"]
    if not _finite(avg):
        invested = [1.0 - float(x["cash_weight"]) for x in r.decisions if _finite(x.get("cash_weight"))]
        avg = sum(invested) / len(invested) if invested else math.nan
    if _finite(avg) and avg < LOW_EXPOSURE:
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
            out.append(f"Every decision was made on or before {r.contamination_end} (the model's training cutoff "
                       f"{r.model_cutoff} plus a {r.cutoff_buffer_days}-day buffer). The model may have known these "
                       "outcomes, so this backtest is NOT evidence of skill. Only decisions after that date (forward "
                       "paper trading) can be.")
        else:
            p = r.post_cutoff_report
            out.append(f"Only the post-cutoff period ({p.days} days after {r.contamination_end}, total return "
                       f"{_pp(p.total_return_pct)}, Sharpe {_num(p.sharpe)}) is free of training contamination; "
                       "judge the strategy on that.")
    out += _evaluation_interpretation(r)
    committee = r.scorecard.get("committee")
    if committee:
        skill = committee.get("brier_skill")
        if _finite(skill) and skill <= 0:
            out.append(f"The committee's forecasts had no skill over a constant 0.5 (Brier skill {skill:+.3f} over "
                       f"{committee['n']} forecasts).")
        elif _finite(skill):
            out.append(f"The committee's Brier skill was {skill:+.3f} over {committee['n']} forecasts; one-month "
                       "equity forecasts with any positive skill are rare, so check the calibration table.")
    if r.universe_note:
        out.append(f"Survivorship: {r.universe_note}.")
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
                  f"decisions made on or before {r.contamination_end}: the model's training cutoff ({r.model_cutoff}) "
                  f"plus a {r.cutoff_buffer_days}-day buffer. The model may have seen what happened next. Those "
                  "returns are not evidence of skill.", ""]
    if r.mode == "offline":
        lines += ["> **OFFLINE MODE.** Deterministic rules stood in for every LLM. Treat this as a test of the "
                  "machinery only.", ""]
    if r.universe_note:
        lines += [f"> **SURVIVORSHIP BIAS.** The {r.universe_note}. Treat absolute returns as an upper bound.", ""]
    lines += ["## The six lines", "",
              "What every backtest report must state before any other number (research report 7.4):", "",
              *(f"{i}. {line}" for i, line in enumerate(six_lines(r), start=1)), ""]

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
    lines += _skill_section(r)
    lines += _overlay_value_section(r)

    lines += ["## Pre / post model-cutoff split", ""]
    if r.model_cutoff is None:
        lines += ["Not applicable: no language model was used (quant mode).", ""]
    else:
        lines += [f"Model training cutoff: **{r.model_cutoff}**, plus a {r.cutoff_buffer_days}-day buffer (vendor "
                  f"cutoffs are unreliable): decisions made on or before **{r.contamination_end}** are contaminated "
                  "(the model may have memorised the outcome); clean evaluation starts after it. Returns are "
                  "assigned to the decision in force.", ""]
        split = {k: v for k, v in (("Pre-cutoff (contaminated)", r.pre_cutoff_report),
                                   ("Post-cutoff (clean)", r.post_cutoff_report)) if v is not None}
        if split:
            lines += [*_table(["Metric", *split],
                              [[label, *(_fmt_metric(v, attr, kind) for v in split.values())]
                               for label, attr, kind in _METRICS[:10]]), ""]
        if r.post_cutoff_report is None:
            lines += ["There is no clean (post-cutoff) period long enough to report.", ""]

    n_rebal = len(r.decisions)
    ex = r.exposure_summary()
    lines += ["## Trading activity and costs", "",
              f"- Exposure: on average {_pct(ex['invested'])} of NAV invested (positions and core) and "
              f"{_pct(ex['cash'])} in cash at the daily closes; decisions targeted on average "
              f"{_num(ex['target_names'], 1)} name(s), active gross {_pct(ex['target_active_gross'])}, core "
              f"{_pct(ex['target_core'])}, cash {_pct(ex['target_cash'])} (see the construct docstring: capital no "
              "name earns stays in cash)",
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
