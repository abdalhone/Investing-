"""Re-identification probe: can the model tell which stock and which month a blind fact sheet is?

Why (RESEARCH 4 "look-ahead from the model's own memory", 7.2, 7.4 stage S2, 9.3 item 2):
blind mode hides the ticker, the name, dollar amounts and absolute dates so that an LLM
cannot recall what happened next. Whether it *works* is an empirical question: frontier
models date masked macro snapshots to the right year almost always, while masked
price/factor payloads were re-identified rarely. This harness measures it on the exact
payload the firm sends, before any pre-cutoff backtest relies on blind mode.

Protocol
--------
* Input: the real (un-blinded) fact sheets of N (ticker, date) probes. Each is blinded with
  :func:`aifirm.agents.analysts.blind_factsheet` and rendered exactly as the firm renders
  it: the full blind sheet (the chair's view; a superset of every analyst slice, so the
  conservative choice) or, with ``role=``, one analyst's slice.
* One structured call per probe (role ``reid_probe``) asks for the ticker and the calendar
  month of day T. The probe is *closed-world*: the model is shown the candidate tickers
  (default: the distinct tickers of the probe set) and the month window (default: the
  first to the last probe month). That is a stronger attacker than an open guess, and it
  makes chance exact: ``1/len(candidates)`` for the ticker, ``1/len(months)`` for the
  month and their product for both.
* Hit rates get Wilson 95% intervals and are compared with chance. A failed call is
  excluded from the denominator (never counted as a miss, which would flatter blind mode)
  and reported; the gate needs enough *answered* probes.
* Gate (RESEARCH 7.2, a derived threshold): at least ``min_probes`` answered probes (400),
  joint hit rate at most ``max_excess`` (1 percentage point) above joint chance, and the
  joint Wilson 95% upper bound at most ``max_upper`` (2 points) above joint chance. The
  research states the bound for an open guess, where chance is about 0; with a closed
  candidate list chance can itself approach 1%, so both thresholds apply to the *excess
  over chance* (otherwise a pure random guesser would fail). A payload that literally
  contains the true ticker, as-of date or company name fails the gate outright (a
  blinding bug, not memory).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..data.factsheet import FactSheet
from ..llm.client import LLM
from ..schemas import ReidGuess
from .analysts import blind_factsheet, render_slice, shown_sheet
from .roles import ROLES

REID_ROLE = "reid_probe"
DEFAULT_REID_MODEL = "claude-opus-5-5"
CANDIDATES_LABEL = "CANDIDATE TICKERS"
MONTH_WINDOW_LABEL = "MONTH WINDOW"
Z_95 = 1.959963984540054
GATE_MIN_PROBES = 400
GATE_MAX_EXCESS = 0.01
GATE_MAX_UPPER = 0.02

REID_SYSTEM = """\
You are auditing an anonymisation procedure. You will see an anonymised fact sheet for one stock on one day (day T): the ticker is replaced by an alias, the company name and dollar amounts are removed, and dates are shown relative to day T.
Using everything you know, make your single best guess of (1) the stock's real ticker, chosen from the candidate list, and (2) the calendar month of day T, within the window given. Guess even if you are unsure: a guess is required. This is a measurement of how much the sheet reveals, not a forecast."""


def wilson_interval(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when ``n`` is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if hits <= 0 else max(0.0, centre - half)        # exact at the boundaries (no rounding residue)
    hi = 1.0 if hits >= n else min(1.0, centre + half)
    return lo, hi


def months_between(first: str, last: str) -> list[str]:
    """Calendar months ``YYYY-MM`` from ``first`` to ``last`` inclusive."""
    start, end = pd.Period(first[:7], "M"), pd.Period(last[:7], "M")
    if end < start:
        start, end = end, start
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def normalize_ticker(t: str) -> str:
    """Upper case, no "$" or spaces, class separator unified ("BRK.B" -> "BRK-B")."""
    return re.sub(r"\s+", "", (t or "").upper().replace("$", "")).replace(".", "-").replace("/", "-")


def normalize_month(m: str) -> str:
    """A guess as ``YYYY-MM`` ("2024-03", "2024-03-15", "March 2024"), or "" if unreadable."""
    text = (m or "").strip()
    hit = re.search(r"\b(\d{4})-(\d{2})\b", text)
    if hit:
        return f"{hit.group(1)}-{hit.group(2)}"
    if not text:
        return ""
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError, OverflowError):
        return ""
    return "" if pd.isna(ts) else str(pd.Period(ts, "M"))


@dataclass(frozen=True)
class HitRate:
    hits: int
    n: int
    rate: float
    lo: float              # Wilson 95% lower bound
    hi: float              # Wilson 95% upper bound
    chance: float          # hit rate of uniform random guessing over the candidates
    excess: float          # rate - chance

    @classmethod
    def of(cls, hits: int, n: int, chance: float) -> "HitRate":
        lo, hi = wilson_interval(hits, n)
        rate = hits / n if n else 0.0
        return cls(hits=hits, n=n, rate=rate, lo=lo, hi=hi, chance=chance, excess=rate - chance)


@dataclass(frozen=True)
class ProbeResult:
    ticker: str
    as_of: str
    true_month: str
    ticker_guess: str = ""
    month_guess: str = ""
    ticker_hit: bool = False
    month_hit: bool = False
    joint_hit: bool = False
    payload_leak: str = ""     # what of the identity the payload contained verbatim ("" = nothing)
    error: str = ""            # non-empty when the call failed (excluded from the rates)


@dataclass
class ReidReport:
    n_probes: int
    n_answered: int
    n_failed: int
    ticker: HitRate
    month: HitRate
    joint: HitRate
    candidates: list[str]
    month_window: list[str]
    gate_passed: bool
    gate_notes: list[str]
    model: str
    payload: str               # "full blind sheet" or "<role> slice"
    probes: list[ProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["month_window"] = [self.month_window[0], self.month_window[-1]] if self.month_window else []
        return d

    def summary(self) -> str:
        def line(name: str, h: HitRate) -> str:
            return (f"{name}: {h.hits}/{h.n} = {h.rate:.2%} (95% CI {h.lo:.2%}-{h.hi:.2%}), chance {h.chance:.2%}, "
                    f"excess {h.excess:+.2%}")
        return "\n".join([f"Re-identification probe ({self.payload}, model {self.model}): {self.n_answered} answered, "
                          f"{self.n_failed} failed of {self.n_probes}",
                          line("ticker", self.ticker), line("month", self.month), line("ticker+month", self.joint),
                          f"gate: {'PASSED' if self.gate_passed else 'FAILED'}" +
                          (f" ({'; '.join(self.gate_notes)})" if self.gate_notes else "")])


def probe_payload(fs: FactSheet, role: str | None = None) -> str:
    """The exact text the firm would send about ``fs`` in blind mode: the full blind sheet,
    or the blind slice of analyst ``role``."""
    if role is None:
        return blind_factsheet(fs).render()
    if role not in ROLES:
        raise ValueError(f"unknown analyst role '{role}'; known: {sorted(ROLES)}")
    return render_slice(ROLES[role], shown_sheet(ROLES[role], fs, blind=True))


def reid_user_prompt(payload: str, candidates: Sequence[str], window: Sequence[str]) -> str:
    return "\n".join([
        f"{CANDIDATES_LABEL}: {', '.join(candidates)}",
        f"{MONTH_WINDOW_LABEL}: {window[0]} to {window[-1]}",
        "", "<fact_sheet>", payload, "</fact_sheet>", "",
        "Guess the ticker (from the candidates) and the month of day T (YYYY-MM, within the window).",
    ])


def _leak(payload: str, fs: FactSheet) -> str:
    found = []
    if len(fs.ticker) >= 2 and re.search(rf"(?<![A-Za-z0-9]){re.escape(fs.ticker)}(?![A-Za-z0-9])", payload):
        found.append("ticker")
    if fs.as_of and fs.as_of in payload:
        found.append("as-of date")
    if fs.name and len(fs.name) >= 3 and fs.name.lower() in payload.lower():
        found.append("company name")
    return ", ".join(found)


def run_reid_probe(llm: LLM, factsheets: Sequence[FactSheet], *, model: str = DEFAULT_REID_MODEL,
                   effort: str = "low", max_tokens: int = 2000, role: str | None = None,
                   candidate_tickers: Sequence[str] | None = None, month_window: tuple[str, str] | None = None,
                   min_probes: int = GATE_MIN_PROBES, max_excess: float = GATE_MAX_EXCESS,
                   max_upper: float = GATE_MAX_UPPER, max_parallel: int = 1) -> ReidReport:
    """Run the probe over ``factsheets`` (real, un-blinded sheets; see the module docstring).

    ``candidate_tickers`` defaults to the sorted distinct tickers of the probes (every true
    ticker is always added); ``month_window`` to (first probe month, last probe month).
    Calls run in order (or on ``max_parallel`` threads); results keep the input order and
    are deterministic for a deterministic ``llm``."""
    if not factsheets:
        raise ValueError("run_reid_probe needs at least one fact sheet")
    truths = [(fs.ticker.upper(), fs.as_of[:7]) for fs in factsheets]
    cands = sorted({normalize_ticker(t) for t in (candidate_tickers or [])} | {normalize_ticker(t) for t, _ in truths})
    first, last = month_window or (min(m for _, m in truths), max(m for _, m in truths))
    window = months_between(first, last)
    outside = sorted({m for _, m in truths if m not in window})
    if outside:
        raise ValueError(f"probe months {outside} fall outside the month window {window[0]}..{window[-1]}")

    def one(fs: FactSheet) -> ProbeResult:
        payload = probe_payload(fs, role)
        truth_t, truth_m = normalize_ticker(fs.ticker), fs.as_of[:7]
        leak = _leak(payload, fs)
        try:
            guess, _ = llm.structured(role=REID_ROLE, model=model, system=REID_SYSTEM,
                                      user=reid_user_prompt(payload, cands, window), schema=ReidGuess,
                                      effort=effort, max_tokens=max_tokens)
        except Exception as exc:     # a failed probe is excluded, never scored as a miss
            return ProbeResult(ticker=fs.ticker, as_of=fs.as_of, true_month=truth_m, payload_leak=leak,
                               error=f"{type(exc).__name__}: {str(exc)[:120]}")
        gt, gm = normalize_ticker(guess.ticker_guess), normalize_month(guess.month_guess)
        return ProbeResult(ticker=fs.ticker, as_of=fs.as_of, true_month=truth_m, ticker_guess=gt, month_guess=gm,
                           ticker_hit=gt == truth_t, month_hit=gm == truth_m,
                           joint_hit=gt == truth_t and gm == truth_m, payload_leak=leak)

    if max_parallel > 1:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            probes = list(pool.map(one, factsheets))
    else:
        probes = [one(fs) for fs in factsheets]

    answered = [p for p in probes if not p.error]
    n = len(answered)
    t_chance, m_chance = 1.0 / len(cands), 1.0 / len(window)
    ticker = HitRate.of(sum(p.ticker_hit for p in answered), n, t_chance)
    month = HitRate.of(sum(p.month_hit for p in answered), n, m_chance)
    joint = HitRate.of(sum(p.joint_hit for p in answered), n, t_chance * m_chance)
    notes = []
    leaks = [p for p in probes if p.payload_leak]
    if leaks:
        notes.append(f"{len(leaks)} payload(s) contain the identity verbatim (e.g. {leaks[0].ticker} "
                     f"{leaks[0].as_of}: {leaks[0].payload_leak}): a blinding bug")
    if n < min_probes:
        notes.append(f"only {n} answered probes (need {min_probes})")
    if joint.excess > max_excess:
        notes.append(f"joint hit rate {joint.rate:.2%} is {joint.excess:.2%} above chance (max {max_excess:.2%})")
    if joint.hi - joint.chance > max_upper:
        notes.append(f"joint 95% upper bound {joint.hi:.2%} is {joint.hi - joint.chance:.2%} above chance "
                     f"(max {max_upper:.2%})")
    return ReidReport(n_probes=len(probes), n_answered=n, n_failed=len(probes) - n, ticker=ticker, month=month,
                      joint=joint, candidates=cands, month_window=window, gate_passed=not notes, gate_notes=notes,
                      model=model, payload=f"{role} slice" if role else "full blind sheet", probes=probes)
