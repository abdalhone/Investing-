"""Deterministic offline responders for every LLM role, so the pipeline runs without an API key.

These are simple, transparent rules, **not** analysis, and must never drive real money.
They exist so the whole system (prompts, grounding, pooling, challenge, memo, sizing,
risk, ledger, backtest) can be tested end to end, offline and reproducibly.

Each responder reads only its prompt: it parses the rendered fact lines
(``[ID] label: value (as of ...; source)``), the filing events and the labelled committee
numbers, and returns a schema-valid object whose evidence cites only fact IDs present in
that prompt and quotes values exactly as rendered, so the grounding check passes.
Analyst probabilities stay within [0.40, 0.60]: the rules are deliberately modest, in
line with the calibration guidance the real analysts receive.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

from pydantic import BaseModel

from ..llm.client import OfflineLLM
from ..schemas import (AnalystView, Challenge, ChecklistItem, EvidenceItem, ICMemo, KillCriterion, Lesson,
                       NewsReading, ReaderEvent, ReidGuess, RiskConcern, RiskReview, Revision, Scenario)
from .committee import (ABSTAINED_LABEL, CHALLENGE_HEADER, CHALLENGE_ROUND_LABEL, DISPERSION_LABEL,
                        FLAGGED_LABEL, HELD_LABEL, MEMBER_SEALED_LABEL, POOLED_PROB_LABEL, SEALED_EXCESS_LABEL,
                        SEALED_PROB_LABEL)
from .reflection import FORECAST_PROB_LABEL, OUTCOME_BEAT
from .reid import CANDIDATES_LABEL, MONTH_WINDOW_LABEL, months_between
from .risk_review import POSITIONS_HEADER

MAX_EDGE = 0.10              # analyst probabilities stay within 0.5 +/- MAX_EDGE
_TRADING_DAYS = 252
_DEFAULT_VOL_PCT = 30.0

_FACT_LINE = re.compile(r"^\[(?P<id>[^\]]+)\] (?P<label>.*): (?P<value>[^:]*?) \(as of ", re.M)
_HEADER = re.compile(r"FACT SHEET: (\S+) ")
_HORIZON = re.compile(r"next (\d+) trading days")
_EVIDENCE_LINE = re.compile(r"^\s*- \[(positive|negative|neutral)\] .*\(facts: (?P<ids>[^)]*)\)\s*$", re.M)

NEGATIVE_EVENTS = ("non-reliance", "change of auditor", "delisting", "bankruptcy", "material impairment",
                   "exit or restructuring", "termination of a material agreement")
DILUTION_EVENTS = ("unregistered equity sale", "S-1 filed", "S-3 filed")


# --------------------------------------------------------------------------- parsing

@dataclass(frozen=True)
class ParsedFact:
    id: str
    label: str
    text: str             # value exactly as rendered, e.g. "35.20%" or "$1.23B"
    value: float | None


def parse_value(text: str) -> float | None:
    """Invert ``Fact.render`` formatting: "$1.23B" -> 1.23e9, "35.20%" -> 35.2, "1,234.00" -> 1234."""
    t = text.strip().replace(",", "").replace("$", "")
    scale = 1.0
    if t and t[-1] in "TBM" and t[:-1].replace(".", "", 1).lstrip("-").isdigit():
        scale = {"T": 1e12, "B": 1e9, "M": 1e6}[t[-1]]
        t = t[:-1]
    t = t.rstrip("%")
    try:
        return float(t) * scale
    except ValueError:
        return None


def parse_facts(text: str) -> dict[str, ParsedFact]:
    """All fact lines in ``text``, keyed by ID (first occurrence wins)."""
    out: dict[str, ParsedFact] = {}
    for m in _FACT_LINE.finditer(text):
        out.setdefault(m["id"], ParsedFact(m["id"], m["label"], m["value"], parse_value(m["value"])))
    return out


def _section_lines(text: str, header: str) -> list[str]:
    """Bullet lines under a '## HEADER' block of a rendered fact sheet."""
    m = re.search(rf"^## {re.escape(header)}[^\n]*\n((?:- [^\n]*\n?)*)", text, re.M)
    return [ln[2:].strip() for ln in m.group(1).splitlines()] if m else []


def _labelled_number(text: str, label: str, default: float) -> float:
    m = re.search(rf"{re.escape(label)}[^:\n]*:\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else default


class _Sheet:
    """Facts parsed from a prompt, addressable by ticker key ("mom_12_1") or full ID."""

    def __init__(self, text: str):
        self.text = text
        self.facts = parse_facts(text)
        m = _HEADER.search(text)
        self.ticker = m.group(1) if m else ""

    def get(self, key: str) -> ParsedFact | None:
        f = self.facts.get(f"{self.ticker}.{key}") or self.facts.get(key)
        return f if f is not None and f.value is not None else None

    def first_with_prefix(self, prefix: str, suffix: str) -> ParsedFact | None:
        for fid, f in self.facts.items():
            if fid.startswith(prefix) and fid.endswith(suffix) and f.value is not None:
                return f
        return None


@dataclass(frozen=True)
class Signal:
    delta: float                   # contribution to P(outperform) - 0.5
    fact_ids: tuple[str, ...]
    point: str
    direction_override: str = ""   # for context facts whose direction is not their own delta

    @property
    def direction(self) -> str:
        if self.direction_override:
            return self.direction_override
        return "positive" if self.delta > 0 else "negative" if self.delta < 0 else "neutral"


def _quote(f: ParsedFact) -> str:
    return f"{f.label} is {f.text}"


# --------------------------------------------------------------------------- analyst rules

def technical_signals(s: _Sheet) -> tuple[list[Signal], str]:
    """12-1 momentum (persists), trend, RSI and 1-month reversal, relative strength, vol regime."""
    out: list[Signal] = []
    mom, trend = s.get("mom_12_1"), s.get("sma50_sma200")
    if mom is None and trend is None:
        return [], "no momentum or trend facts in the slice"
    if mom is not None:
        out.append(Signal(0.03 * math.tanh(mom.value / 20), (mom.id,),
                          f"{_quote(mom)}; intermediate-term momentum tends to persist."))
    if trend is not None:
        up = trend.value > 1
        out.append(Signal(0.015 if up else -0.015, (trend.id,),
                          f"{_quote(trend)}, so the medium-term trend is {'up' if up else 'down'}."))
    if (rsi := s.get("rsi_14")) is not None:
        d = -0.015 if rsi.value > 70 else 0.015 if rsi.value < 30 else 0.0
        state = "overbought" if d < 0 else "oversold" if d > 0 else "neither overbought nor oversold"
        out.append(Signal(d, (rsi.id,), f"{_quote(rsi)}: {state}."))
    if (r1 := s.get("ret_1m")) is not None and abs(r1.value) > 15:
        out.append(Signal(-0.01 if r1.value > 0 else 0.01, (r1.id,),
                          f"{_quote(r1)}; an extreme latest month tends to partially reverse."))
    if (rel := s.get("rel_ret_3m")) is not None:
        out.append(Signal(0.01 * math.tanh(rel.value / 10), (rel.id,),
                          f"{_quote(rel)}, a measure of relative strength."))
    v20, v60 = s.get("vol_20d"), s.get("vol_60d")
    if v20 is not None and v60 is not None:
        rising = v20.value > v60.value
        out.append(Signal(0.0, (v20.id, v60.id),
                          f"Short-term volatility ({v20.text}) is {'above' if rising else 'below'} the 60-day level "
                          f"({v60.text}); {'a rising' if rising else 'a calm'} volatility regime."))
    return out, ""


def fundamental_signals(s: _Sheet) -> tuple[list[Signal], str]:
    """Earnings and FCF yield, growth, margins, cash conversion, dilution, leverage."""
    core = [s.get(k) for k in ("earnings_yield", "fcf_yield", "revenue_growth_yoy", "operating_income_margin")]
    if sum(f is not None for f in core) < 2:
        return [], "fewer than two of earnings yield, FCF yield, revenue growth and operating margin are available"
    ey, fy, growth, margin = core
    out: list[Signal] = []
    if ey is not None:
        d = 0.015 if ey.value > 5 else -0.015 if ey.value < 2 else 0.0
        out.append(Signal(d, (ey.id,), f"{_quote(ey)}: {'cheap' if d > 0 else 'expensive' if d < 0 else 'moderately priced'} on earnings."))
    if fy is not None:
        d = 0.015 if fy.value > 4 else -0.015 if fy.value < 1 else 0.0
        out.append(Signal(d, (fy.id,), f"{_quote(fy)}: {'high' if d > 0 else 'low' if d < 0 else 'middling'} cash yield."))
    if growth is not None:
        d = 0.015 if growth.value > 10 else -0.015 if growth.value < 0 else 0.0
        out.append(Signal(d, (growth.id,), f"{_quote(growth)}: {'strong' if d > 0 else 'shrinking' if d < 0 else 'modest'} growth."))
    if margin is not None:
        d = 0.01 if margin.value > 20 else -0.01 if margin.value < 5 else 0.0
        out.append(Signal(d, (margin.id,), f"{_quote(margin)}: {'high' if d > 0 else 'thin' if d < 0 else 'moderate'} profitability."))
    fcf, ni = s.get("fcf"), s.get("net_income")
    if fcf is not None and ni is not None and ni.value > 0:
        if fcf.value >= ni.value:
            out.append(Signal(0.01, (fcf.id, ni.id), f"Free cash flow ({fcf.text}) exceeds net income ({ni.text}): earnings are backed by cash."))
        elif fcf.value < 0.5 * ni.value:
            out.append(Signal(-0.01, (fcf.id, ni.id), f"Free cash flow ({fcf.text}) is well below net income ({ni.text}): weak cash conversion."))
    if (dil := s.get("share_count_change_yoy")) is not None:
        d = -0.01 if dil.value > 2 else 0.005 if dil.value < -1 else 0.0
        out.append(Signal(d, (dil.id,), f"{_quote(dil)}: {'dilution' if d < 0 else 'buybacks' if d > 0 else 'a stable share count'}."))
    if (lev := s.get("debt_to_equity")) is not None and lev.value > 2:
        out.append(Signal(-0.01, (lev.id,), f"{_quote(lev)}: high leverage."))
    return out, ""


def macro_signals(s: _Sheet) -> tuple[list[Signal], str]:
    """Beta times regime: risk-on favors high beta relative to the benchmark, risk-off low beta."""
    beta = s.get("beta_1y")
    if beta is None:
        return [], "the stock's beta is not available"
    pieces: list[tuple[float, ParsedFact, str]] = []
    if (vix := s.get("MACRO.VIXCLS")) is not None:
        r = 0.5 if vix.value < 16 else -0.5 if vix.value > 25 else 0.0
        pieces.append((r, vix, "calm" if r > 0 else "stressed" if r < 0 else "normal"))
    if (hy := s.get("MACRO.BAMLH0A0HYM2_chg_3m")) is not None:
        r = -0.5 if hy.value > 0.5 else 0.25 if hy.value < -0.25 else 0.0
        pieces.append((r, hy, "widening credit spreads" if r < 0 else "tightening credit spreads" if r > 0 else "stable credit"))
    if (trend := s.first_with_prefix("MKT.", "_vs_sma200")) is not None:
        r = 0.25 if trend.value > 0 else -0.25
        pieces.append((r, trend, "a market above its long-term trend" if r > 0 else "a market below its long-term trend"))
    if (curve := s.get("MACRO.curve_2s10s")) is not None and curve.value < 0:
        pieces.append((-0.25, curve, "an inverted yield curve"))
    if not pieces:
        return [], "no macro or market-regime facts in the slice"
    regime = max(-1.0, min(1.0, sum(r for r, _, _ in pieces)))
    tilt = max(-1.0, min(1.0, beta.value - 1.0))
    out = [Signal(0.03 * regime * tilt, (beta.id,) + tuple(f.id for _, f, _ in pieces),
                  f"{_quote(beta)}; in a {'risk-on' if regime > 0 else 'risk-off' if regime < 0 else 'neutral'} regime a "
                  f"{'higher' if tilt > 0 else 'lower' if tilt < 0 else 'market-like'} beta "
                  f"{'helps' if regime * tilt > 0 else 'hurts' if regime * tilt < 0 else 'does not matter'} relative performance.")]
    for r, f, desc in pieces:
        d = "positive" if r * tilt > 0 else "negative" if r * tilt < 0 else "neutral"
        out.append(Signal(0.0, (f.id,), f"{_quote(f)}, consistent with {desc}.", direction_override=d))
    return out, ""


_ID_PREFIX = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?P<text>.*)$")
_NEWS_LABEL = re.compile(r"tier (?P<tier>[ABC]), (?P<type>\w+), (?P<direction>\w+), (?P<materiality>\w+) materiality")


def _filing_lines(text: str) -> list[tuple[str, str]]:
    """(fact ID or "", event text) for each filing-event line; IDs appear once news is attached."""
    out = []
    for line in _section_lines(text, "RECENT SEC FILING EVENTS"):
        if line == "none in window":
            continue
        m = _ID_PREFIX.match(line)
        out.append((m["id"], m["text"]) if m else ("", line))
    return out


def news_signals(s: _Sheet) -> list[Signal]:
    """Reader-extracted news events: tier A/B events with a clear direction move the
    probability (+/-0.01, doubled for high materiality); tier C events are context only."""
    out: list[Signal] = []
    for fid, f in s.facts.items():
        if ".news_" not in fid or (m := _NEWS_LABEL.search(f.label)) is None:
            continue
        tier, etype, direction, materiality = m["tier"], m["type"], m["direction"], m["materiality"]
        kind = etype.replace("_", " ")
        if tier in ("A", "B") and direction in ("positive", "negative") and etype != "analyst_action":
            size = 0.02 if materiality == "high" else 0.01
            out.append(Signal(size if direction == "positive" else -size, (fid,),
                              f"Tier {tier} {kind} news ({materiality} materiality), {direction}: {f.text}."))
        else:
            out.append(Signal(0.0, (fid,), f"Tier {tier} {kind} news, {direction}: {f.text}; context only "
                                           "(unverified or opinion).", direction_override="neutral"))
    return out


def events_signals(s: _Sheet) -> tuple[list[Signal], str]:
    """Count adverse 8-K items and dilution signals; earnings drift via the 1-month return;
    tier A/B news events (tier C news is context only)."""
    events = _filing_lines(s.text)
    headlines = [h for h in _section_lines(s.text, "RECENT HEADLINES") if not h.startswith("no headlines")]
    news = news_signals(s)
    if not events and not headlines and not news:
        return [], "no filing events, news events or headlines in the window"
    out: list[Signal] = []
    for fid, e in events:
        ids = (fid,) if fid else ()
        if any(k in e for k in NEGATIVE_EVENTS):
            out.append(Signal(-0.02, ids, f"Adverse filing event: {e}."))
        elif any(k in e for k in DILUTION_EVENTS):
            out.append(Signal(-0.01, ids, f"Possible dilution: {e}."))
        elif "insider transaction" in e:
            out.append(Signal(0.0, ids, f"{e}; a count says nothing about buying versus selling."))
    r1 = s.get("ret_1m")
    earnings = [fid for fid, e in events if "results of operations" in e]
    if r1 is not None and earnings:
        d = 0.01 if r1.value > 5 else -0.01 if r1.value < -5 else 0.0
        out.append(Signal(d, (r1.id, *[i for i in earnings[:1] if i]),
                          f"An earnings release was filed and {r1.label.lower()} is {r1.text}; "
                          "post-earnings drift tends to follow the initial reaction."))
    out += news
    if not out:
        latest = events[0][1] if events else "headlines only"
        out.append(Signal(0.0, (events[0][0],) if events and events[0][0] else (),
                          f"Only routine filings in the window (latest: {latest})."))
    if (v20 := s.get("vol_20d")) is not None:
        out.append(Signal(0.0, (v20.id,), f"{_quote(v20)}; it sets the width of the outcome range."))
    return out, ""


ANALYST_RULES: dict[str, Callable[[_Sheet], tuple[list[Signal], str]]] = {
    "technical": technical_signals,
    "fundamental": fundamental_signals,
    "macro": macro_signals,
    "events": events_signals,
}


def _horizon(text: str) -> int:
    m = _HORIZON.search(text)
    return int(m.group(1)) if m else 21


def _scenarios(p: float, excess: float, sigma_pct: float) -> list[Scenario]:
    """Bull/bear at +/- one horizon sigma; base solves for the stated expected excess."""
    edge = p - 0.5
    bull_p, bear_p, base_p = 0.25 + edge, 0.25 - edge, 0.5
    base_x = (excess - bull_p * sigma_pct + bear_p * sigma_pct) / base_p
    return [
        Scenario(name="bull", probability=round(bull_p, 4), excess_return_pct=round(sigma_pct, 3),
                 description="The positive signals play out and the stock leads the benchmark."),
        Scenario(name="base", probability=base_p, excess_return_pct=round(base_x, 3),
                 description="Mostly noise around a small edge."),
        Scenario(name="bear", probability=round(bear_p, 4), excess_return_pct=round(-sigma_pct, 3),
                 description="The negative signals dominate and the stock lags the benchmark."),
    ]


def _abstain(ticker: str, reason: str) -> AnalystView:
    return AnalystView(ticker=ticker, abstain=True, abstain_reason=f"Offline rules: {reason}.", prob_outperform=0.5,
                       expected_excess_return_pct=0.0, scenarios=[], evidence=[], key_risks=[],
                       what_would_change_my_mind="", private_insight="")


def analyst_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> AnalystView:
    """p = 0.5 + sum of rule deltas (clipped to +/- MAX_EDGE); expected excess = sigma_h * inverse-normal(p)."""
    name = role.split(".", 1)[1] if "." in role else role
    sheet = _Sheet(user)
    rules = ANALYST_RULES.get(name)
    if rules is None:
        return _abstain(sheet.ticker, f"no rules for role '{name}'")
    signals, reason = rules(sheet)
    if reason:
        return _abstain(sheet.ticker, reason)
    edge = max(-MAX_EDGE, min(MAX_EDGE, sum(sig.delta for sig in signals)))
    p = round(0.5 + edge, 4)
    vol = sheet.get("vol_60d") or sheet.get("vol_20d")
    sigma_pct = (vol.value if vol else _DEFAULT_VOL_PCT) * math.sqrt(_horizon(user) / _TRADING_DAYS)
    excess = round(sigma_pct * NormalDist().inv_cdf(p), 3)
    ranked = sorted(signals, key=lambda sig: -abs(sig.delta))[:8]
    cited = [i for sig in ranked for i in sig.fact_ids]
    return AnalystView(
        ticker=sheet.ticker, abstain=False, abstain_reason="", prob_outperform=p,
        expected_excess_return_pct=excess, scenarios=_scenarios(p, excess, sigma_pct),
        evidence=[EvidenceItem(point=sig.point, fact_ids=list(sig.fact_ids), direction=sig.direction) for sig in ranked],
        key_risks=["Short-horizon returns are dominated by noise.",
                   "These are fixed offline rules, not analysis of the business."],
        what_would_change_my_mind=("A reversal in the facts cited above: " + ", ".join(dict.fromkeys(cited)) + ".")
        if cited else "New filing events or headlines.",
        private_insight=f"The {name} rules net to {'a positive' if edge > 0 else 'a negative' if edge < 0 else 'no'} "
                        "tilt; each evidence item states the rule that fired.",
    )


# --------------------------------------------------------------------------- committee roles

def _all_signals(sheet: _Sheet) -> list[Signal]:
    out: list[Signal] = []
    for rules in ANALYST_RULES.values():
        out += rules(sheet)[0]
    return out


def challenge_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> Challenge:
    """Cite the directional facts on the full sheet that oppose the pooled view's side."""
    sheet = _Sheet(user)
    pooled = _labelled_number(user, POOLED_PROB_LABEL, 0.5)
    against = "negative" if pooled >= 0.5 else "positive"
    opposing = sorted((sig for sig in _all_signals(sheet) if sig.direction == against and sig.fact_ids),
                      key=lambda sig: -abs(sig.delta))[:4]
    lean = "toward outperformance" if pooled >= 0.5 else "toward underperformance"
    if opposing:
        argument = f"The pooled view leans {lean}, but these facts point the other way. Strongest: {opposing[0].point}"
    else:
        argument = f"The pooled view leans {lean}, yet no single fact on the sheet strongly supports a move away from the base rate."
    return Challenge(
        ticker=sheet.ticker, strongest_counterargument=argument,
        evidence=[EvidenceItem(point=sig.point, fact_ids=list(sig.fact_ids), direction=sig.direction) for sig in opposing],
        overlooked_risks_or_opportunities=["Over one month, noise dominates; agreement among analysts may reflect shared evidence."],
        questions_for_the_committee=["Which of the cited facts does the consensus believe is already reflected in the price?"],
    )


def revision_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> Revision:
    """Shrink 25% toward 0.5 only if the challenge's evidence leans against the sealed view
    and cites facts; otherwise keep the sealed view."""
    own_p = _labelled_number(user, SEALED_PROB_LABEL, 0.5)
    own_x = _labelled_number(user, SEALED_EXCESS_LABEL, 0.0)
    challenge = user.split(CHALLENGE_HEADER, 1)[1] if CHALLENGE_HEADER in user else ""
    lines = list(_EVIDENCE_LINE.finditer(challenge))
    lean = sum(1 if m.group(1) == "positive" else -1 if m.group(1) == "negative" else 0 for m in lines)
    ids = list(dict.fromkeys(i.strip() for m in lines for i in m.group("ids").split(",")
                             if i.strip() and i.strip() != "none"))
    m = re.search(r'Use ticker "([^"]+)"', user)
    ticker = m.group(1) if m else ""
    if ids and lean and (own_p - 0.5) * lean < 0:
        return Revision(ticker=ticker, prob_outperform=round(0.5 + 0.75 * (own_p - 0.5), 4),
                        expected_excess_return_pct=round(0.75 * own_x, 3), changed=True,
                        reason="The challenge cites facts that argue against my view; I moved a quarter of the way toward 0.5.",
                        new_evidence_fact_ids=ids[:3])
    return Revision(ticker=ticker, prob_outperform=own_p, expected_excess_return_pct=own_x, changed=False,
                    reason="The challenge does not add evidence against my view.", new_evidence_fact_ids=[])


def memo_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> ICMemo:
    """Checklist failures (flagged views, no independent confirmation, data gaps or abstentions,
    high dispersion) cut conviction by 0.25 each; the recommendation follows the pooled side."""
    sheet = _Sheet(user)
    pooled = _labelled_number(user, POOLED_PROB_LABEL, 0.5)
    dispersion = _labelled_number(user, DISPERSION_LABEL, 0.0)
    held = re.search(rf"{re.escape(HELD_LABEL)}: yes", user) is not None
    challenged = re.search(rf"{re.escape(CHALLENGE_ROUND_LABEL)}: yes", user) is not None
    flagged = user.count(FLAGGED_LABEL)
    members = [float(x) for x in re.findall(rf"{re.escape(MEMBER_SEALED_LABEL)} ([0-9.]+)", user)]
    side = pooled - 0.5
    confirming = sum(1 for p in members if (p - 0.5) * side > 0)
    gaps = "## DATA GAPS" in user
    abstained = user.count(ABSTAINED_LABEL)
    checklist = [
        ChecklistItem(item="Evidence grounded", passed=flagged == 0, note=f"{flagged} view(s) flagged by the grounding check."),
        ChecklistItem(item="Independent confirmation", passed=confirming >= 2, note=f"{confirming} view(s) on the pooled side."),
        ChecklistItem(item="Data complete", passed=not gaps and abstained == 0,
                      note=f"Data gaps listed: {'yes' if gaps else 'no'}; abstentions: {abstained}."),
        ChecklistItem(item="Dissent addressed", passed=dispersion <= 0.10,
                      note="A challenge round ran." if challenged else "Views were close; no challenge needed."),
    ]
    failures = sum(1 for c in checklist if not c.passed)
    if side >= 0.02:
        rec = "buy"
    elif side <= -0.02:
        rec = "sell" if held else "avoid"
    else:
        rec = "hold" if held else "avoid"
    kills = []
    if (rel := sheet.get("rel_ret_3m")) is not None:
        kills.append(KillCriterion(metric=rel.id, condition="< -10", rationale="Sustained relative weakness contradicts the thesis."))
    if (vix := sheet.get("MACRO.VIXCLS")) is not None:
        kills.append(KillCriterion(metric=vix.id, condition="> 30", rationale="A volatility shock changes the regime."))
    return ICMemo(
        ticker=sheet.ticker, recommendation=rec,
        thesis_one_sentence=f"Offline rules: the pooled view leans {'positive' if side > 0 else 'negative' if side < 0 else 'neutral'}.",
        variant_perception="none",
        base_rate_note="About half of individual stocks beat the benchmark over a month; the pooled edge is small.",
        catalysts=[], premortem=["Short-horizon noise overwhelms a small edge.", "The cited signals were already priced."],
        kill_criteria=kills, checklist=checklist, conviction_multiplier=max(0.0, 1.0 - 0.25 * failures),
        multiplier_reason=f"{failures} checklist failure(s), 0.25 each.",
    )


def risk_review_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> RiskReview:
    """Trim very volatile names (vol_60d above 60%) to 0.75 and names with adverse filings to 0.5;
    flag a sector holding more than half of the active gross weight."""
    block = user.split(POSITIONS_HEADER, 1)[1] if POSITIONS_HEADER in user else ""
    positions = re.findall(r"^- (\S+) \| sector: (.*?) \| weight ([-0-9.]+)", block, re.M)
    sheets = dict(re.findall(r'<fact_sheet ticker="([^"]+)">\n(.*?)\n</fact_sheet>', user, re.S))
    concerns: list[RiskConcern] = []
    for ticker, _, _ in positions:
        sheet = _Sheet(sheets.get(ticker, ""))
        adverse = [e for e in _section_lines(sheet.text, "RECENT SEC FILING EVENTS") if any(k in e for k in NEGATIVE_EVENTS)]
        vol = sheet.get("vol_60d")
        if adverse:
            concerns.append(RiskConcern(ticker=ticker, concern=f"Adverse filing event: {adverse[0]}", multiplier=0.5))
        elif vol is not None and vol.value > 60:
            concerns.append(RiskConcern(ticker=ticker, concern=f"{_quote(vol)}, unusually high.", multiplier=0.75))
    gross: dict[str, float] = {}
    for _, sector, w in positions:
        gross[sector] = gross.get(sector, 0.0) + abs(float(w))
    total = sum(gross.values())
    hidden = [f"Sector '{sec}' holds most of the active gross weight." for sec, g in sorted(gross.items())
              if total > 0 and g / total > 0.5 and len(positions) > 1]
    return RiskReview(summary=f"Offline rules reviewed {len(positions)} position(s); {len(concerns)} concern(s).",
                      hidden_concentrations=hidden, concerns=concerns, halt_trading=False)


def lesson_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> Lesson:
    """Distance from 0.5 versus the outcome decides the lesson; one outcome is treated as weak evidence."""
    p = _labelled_number(user, FORECAST_PROB_LABEL, 0.5)
    beat = OUTCOME_BEAT in user
    agent = re.search(r"Forecaster: (\S+)", user)
    stock = re.search(r"Stock: (\S+)", user)
    right = (p > 0.5) == beat if p != 0.5 else False
    confident = abs(p - 0.5) >= 0.1
    if confident and not right:
        lesson = "Move far from 0.50 only when several independent facts agree; one signal is not enough."
    elif not right:
        lesson = "Keep the process: a small edge is expected to be wrong often."
    else:
        lesson = "Keep the process: a single correct call is weak evidence of skill."
    return Lesson(agent=agent.group(1) if agent else "", ticker=stock.group(1) if stock else "",
                  what_happened=f"The forecast was {'right' if right else 'wrong'} on direction.",
                  was_it_skill_or_luck="Mostly luck: one binary outcome says little about calibration.",
                  process_lesson=lesson,
                  applies_when="Forecasts far from 0.50" if confident else "Forecasts close to 0.50")


# --------------------------------------------------------------------------- news reader and re-identification probe

_BLOCK = re.compile(r'<headline index="(?P<idx>\d+)">\ntitle: (?P<title>[^\n]*)\n(?:feed summary: (?P<summary>[^\n]*)\n)?</headline>')

# (event type, keywords) in priority order: the first type with a keyword in the text wins.
NEWS_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rumor", ("reportedly", "in talks", "considering", "rumor", "rumour", "sources say", "people familiar",
               "speculation", "said to be", "could be")),
    ("m_and_a", ("acquire", "acquisition", "merger", "merge", "takeover", "buyout", "to buy ")),
    ("guidance", ("guidance", "outlook", "forecast")),
    ("earnings", ("earnings", "results", "quarterly", "revenue", "profit", "eps")),
    ("legal_regulatory", ("lawsuit", "sues", "sued", "probe", "investigation", "antitrust", "fined", "regulator",
                          "settlement", "subpoena", "recall")),
    ("management", ("ceo", "cfo", "chief executive", "resigns", "steps down", "appoints", "appointed", "names new")),
    ("capital_return", ("dividend", "buyback", "repurchase")),
    ("analyst_action", ("upgrade", "downgrade", "price target", "rating", "initiates coverage", "analyst")),
    ("product", ("launch", "unveil", "introduces", "new product", "approval", "approved", "fda")),
    ("macro_sector", ("fed ", "rates", "inflation", "tariff", "sector", "industry", "economy")),
)
NEWS_POSITIVE = ("beat", "record", "raise", "surge", "jump", "soar", "gain", "rise", "upgrade", "approv", "wins",
                 "strong", "growth", "higher", "boost", "expand", "increase", "hike")
NEWS_NEGATIVE = ("miss", "cut", "lower", "fall", "plunge", "drop", "slump", "downgrade", "lawsuit", "probe", "recall",
                 "weak", "loss", "resign", "bankrupt", "fraud", "delay", "warn", "investigation", "decline", "sue")
_HIGH_TYPES = ("earnings", "guidance", "m_and_a", "legal_regulatory")
_LOW_TYPES = ("analyst_action", "macro_sector", "other", "rumor")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _has_word(text: str, word: str) -> bool:
    """``word`` starts a word in ``text`` ("eps" matches "EPS beat" but not "steps")."""
    return re.search(r"\b" + re.escape(word), text) is not None


def classify_headline(text: str) -> tuple[str, str, str]:
    """Keyword rules: (event_type, direction, materiality) for one headline's text."""
    t = f" {text.lower()} "
    etype = next((name for name, words in NEWS_TYPE_RULES if any(_has_word(t, w) for w in words)), "other")
    score = sum(_has_word(t, w) for w in NEWS_POSITIVE) - sum(_has_word(t, w) for w in NEWS_NEGATIVE)
    direction = "positive" if score > 0 else "negative" if score < 0 else "unclear"
    if etype in _LOW_TYPES:
        materiality = "low"
    elif etype in _HIGH_TYPES and direction != "unclear":
        materiality = "high"
    else:
        materiality = "medium"
    return etype, direction, materiality


def news_reader_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> NewsReading:
    """One event per headline block from keyword rules. The summary is a template built
    from the typed fields; no headline text is ever copied."""
    events = []
    for m in _BLOCK.finditer(user):
        etype, direction, materiality = classify_headline(f"{m['title']} {m['summary'] or ''}")
        events.append(ReaderEvent(headline_index=int(m["idx"]), event_type=etype, direction=direction,
                                  materiality=materiality,
                                  summary=f"Offline rules read {_article(direction)} {direction} "
                                          f"{etype.replace('_', ' ')} report about the company"))
    return NewsReading(events=events)


def reid_responder(role: str, system: str, user: str, schema: type[BaseModel]) -> ReidGuess:
    """A uniformly random guess from the candidate tickers and the month window, seeded by the
    prompt's SHA-256 (deterministic). Its hit rates should match the chance baseline."""
    rng = random.Random(int(hashlib.sha256(user.encode()).hexdigest()[:16], 16))
    tickers = re.search(rf"^{re.escape(CANDIDATES_LABEL)}: (.*)$", user, re.M)
    window = re.search(rf"^{re.escape(MONTH_WINDOW_LABEL)}: (\d{{4}}-\d{{2}}) to (\d{{4}}-\d{{2}})", user, re.M)
    cands = [t.strip() for t in tickers.group(1).split(",") if t.strip()] if tickers else ["UNKNOWN"]
    months = months_between(window.group(1), window.group(2)) if window else ["2000-01"]
    return ReidGuess(ticker_guess=rng.choice(cands), month_guess=rng.choice(months),
                     confidence=1.0 / (len(cands) * len(months)), clues="Offline random guess; no analysis.")


def make_offline_llm() -> OfflineLLM:
    """An ``OfflineLLM`` with deterministic responders for every role used by the firm."""
    return OfflineLLM({
        "analyst.": analyst_responder,
        "challenge": challenge_responder,
        "revision.": revision_responder,
        "memo": memo_responder,
        "risk_review": risk_review_responder,
        "lesson": lesson_responder,
        "news_reader": news_reader_responder,
        "reid_probe": reid_responder,
    })
