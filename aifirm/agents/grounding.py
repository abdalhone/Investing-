"""Numeric grounding checker: is every number an analyst wrote actually on the fact sheet?

Why (P2): financial QA benchmarks show LLMs confidently state wrong numbers, and multi-agent
trading systems have repeatedly shipped fixes for hallucinated figures. The Data Desk gives
every number an ID, analysts must cite IDs, and this module audits the free text of each
view. The check is deliberately lenient about *presentation* (rounding, "%", "$130.5B",
"45 bps") and strict about *provenance*: a number must match some fact value, or appear
verbatim in fact-sheet text the analyst was shown (labels such as "60 days", filing
events, headlines), or be the view's own forecast.

An ungrounded view is not discarded. It is flagged and its pooling weight is halved
(see :mod:`aifirm.agents.committee`), because a single stray number is weak evidence of a
bad forecast, while a pattern of them is strong evidence of fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.factsheet import FactSheet
from ..schemas import AnalystView

# Fact-shaped and other non-quantity tokens are masked before numbers are extracted, so
# digits inside identifiers ("NVDA.ret_3m", "MACRO.DGS10"), dates, SEC form names
# ("8-K", "10-Q", "Form 4"), 8-K item numbers ("Item 2.02") and index names ("S&P 500")
# are never mistaken for claimed quantities.
_MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z][A-Z0-9\-]*\.[A-Za-z][A-Za-z0-9_]*\b"),                # fact IDs
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.+\-]+Z?)?\b"),                    # ISO dates
    re.compile(r"\bT[-+]\d+d\b"),                                               # blind-mode relative dates
    re.compile(r"\b(?:8-K|10-K|10-Q|20-F|6-K|S-1|S-3|S-4|S-8|DEF\s*14A|SC\s*13[DG]|13[DGF]|Form\s+[345])(?:/A)?\b", re.I),
    re.compile(r"\bitems?\s+\d{1,2}\.\d{2}(?:\s*(?:,|and|&)\s*\d{1,2}\.\d{2})*", re.I),
    re.compile(r"(?:S&P|Russell|Nasdaq|Dow)\s*\d+", re.I),
    re.compile(r"\b\d+s\d+s\b"),                                                # "2s10s" curve
)

_NUMBER = re.compile(r"(?<![\w.])(?P<dollar>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)")
_SCALE_WORDS = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
                "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12, "t": 1e12}
_SCALE = re.compile(r"\s?(thousand|million|billion|trillion|mn|mm|bn|tn|[KkMmBbTt])\b")
_TIME_UNIT = re.compile(
    r"\s?-?\s?(?:trading\s+)?(?:days?|d|weeks?|wks?|w|months?|mos?|m|quarters?|qtrs?|q|years?|yrs?|y|sessions?|hours?|hrs?|h)\b",
    re.I,
)
_BPS = re.compile(r"\s?(?:bps?|basis\s+points?)\b", re.I)
_PERCENT = re.compile(r"\s?(?:%|percent\b|pct\b|percentage\s+points?\b|pp\b)", re.I)

# Base-rate references the prompts themselves make ("about 0.5", "50%"); quoting them is fine.
_BASE_RATE_VALUES = (0.5, 50.0)
_MAX_UNSUPPORTED = 2


@dataclass(frozen=True)
class NumberToken:
    """One quantity found in free text."""

    text: str          # as written, e.g. "$130.5B" or "12.3%"
    value: float       # magnitude as written, without scale (130.5)
    decimals: int      # digits after the decimal point, used for rounding tolerance
    scale: float       # 1e9 for "B"/"billion", 1.0 otherwise
    percent: bool
    bps: bool
    time_unit: bool    # followed by a time unit ("3 months", "12-month", "60d")
    dollar: bool

    @property
    def is_year(self) -> bool:
        return (self.decimals == 0 and "," not in self.text and 1900 <= self.value <= 2100
                and not (self.dollar or self.percent or self.bps or self.scale != 1.0))


@dataclass
class GroundingReport:
    ok: bool
    unknown_fact_ids: list[str]
    unsupported_numbers: list[str]
    checked_numbers: int

    def issues(self) -> list[str]:
        """Human-readable issues for ``ViewRecord.grounding_issues`` and the ledger."""
        out = [f"unknown fact id: {i}" for i in self.unknown_fact_ids]
        out += [f"unsupported number: {n}" for n in self.unsupported_numbers]
        return out


def extract_numbers(text: str) -> list[NumberToken]:
    """Find quantities in ``text`` with their presentation (%, $, scale, bps, time unit)."""
    masked = text
    for pat in _MASKS:
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)
    out: list[NumberToken] = []
    for m in _NUMBER.finditer(masked):
        raw = m.group("num")
        dollar = m.group("dollar") is not None
        tail = masked[m.end(): m.end() + 32]
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        scale, percent, bps, time_unit = 1.0, False, False, False
        suffix = ""
        if (pm := _PERCENT.match(tail)) is not None:
            percent, suffix = True, pm.group(0)
        elif (bm := _BPS.match(tail)) is not None:
            bps, suffix = True, bm.group(0)
        elif (sm := _SCALE.match(tail)) is not None and _is_scale(sm.group(1), dollar):
            scale, suffix = _SCALE_WORDS[sm.group(1).lower()], sm.group(0)
        elif (tm := _TIME_UNIT.match(tail)) is not None:
            time_unit, suffix = True, tm.group(0)
        out.append(NumberToken(text=("$" if dollar else "") + raw + suffix.rstrip(), value=value,
                               decimals=decimals, scale=scale, percent=percent, bps=bps,
                               time_unit=time_unit, dollar=dollar))
    return out


def _is_scale(word: str, dollar: bool) -> bool:
    """A lowercase single "m"/"b"/"t"/"k" after a bare number is ambiguous ("3m" is usually
    three months in this domain); treat it as a scale only after a dollar sign. Uppercase
    letters and spelled-out words are always scales."""
    if len(word) == 1 and word.islower():
        return dollar
    return True


def _close(n: float, decimals: int, c: float, rel_tol: float) -> bool:
    """True if a number written with ``decimals`` decimals is a fair rendering of fact value
    ``c``: within ``rel_tol`` relative, 0.05 absolute, or equal after rounding to the written
    precision. Magnitudes are compared because sign is usually carried by words ("fell 12%")."""
    tol = max(rel_tol * abs(c), 0.05, 0.5 * 10.0 ** (-decimals))
    return abs(n - abs(c)) <= tol + 1e-12


def _rounds_to(n: float, decimals: int, c: float) -> bool:
    """True if ``c`` rounded to the written precision is ``n`` (strict: no relative slack)."""
    return abs(n - abs(c)) <= 0.5 * 10.0 ** (-decimals) + 1e-12


def _fact_variants(fs: FactSheet) -> tuple[list[float], list[float]]:
    """(all display variants, absolute-scale values) of the numeric facts on ``fs``."""
    variants: list[float] = []
    absolute: list[float] = []
    for f in fs.facts:
        if isinstance(f.value, bool) or not isinstance(f.value, (int, float)):
            continue
        v = abs(float(f.value))
        variants.append(v)
        if f.unit in ("ratio", "x"):
            variants.append(v * 100.0)          # a ratio quoted as a percentage
        if f.unit in ("USD", "count"):
            absolute.append(v)
            variants += [v / 1e3, v / 1e6, v / 1e9, v / 1e12]
    return variants, absolute


def _sheet_text_numbers(fs: FactSheet) -> list[float]:
    """Numbers printed in fact-sheet *text* the analyst saw: fact labels ("60 days",
    "52-week"), filing events, headlines and data-gap notes."""
    texts = [f.label for f in fs.facts] + list(fs.events) + list(fs.data_gaps)
    texts += [f"{h.title} {h.summary}" for h in fs.headlines]
    return [t.value * t.scale for text in texts for t in extract_numbers(text)]


def _own_values(view: AnalystView) -> list[float]:
    vals = [view.prob_outperform, view.prob_outperform * 100.0, view.expected_excess_return_pct]
    for s in view.scenarios:
        vals += [s.probability, s.probability * 100.0, s.excess_return_pct]
    return vals


def view_texts(view: AnalystView) -> list[str]:
    """The free-text fields that may contain claimed numbers."""
    texts = [e.point for e in view.evidence] + [s.description for s in view.scenarios]
    texts += [view.private_insight] + list(view.key_risks)
    return [t for t in texts if t]


def normalize_fact_id(fid: str) -> str:
    """Strip the decoration models sometimes add around IDs ("[NVDA.ret_3m]", " NVDA.pe_ttm ")."""
    return fid.strip().strip("[]`'\"").strip()


def check_view(view: AnalystView, fs: FactSheet, rel_tol: float = 0.03) -> GroundingReport:
    """Audit ``view`` against ``fs``.

    ``fs`` should be the fact sheet the analyst was actually shown (its slice), so that a
    cited ID the analyst could not have seen counts as unknown. ``ok`` is False if any
    cited fact ID is unknown or more than two distinct numbers are unsupported.
    """
    known = fs.ids()
    unknown: list[str] = []
    for item in view.evidence:
        for fid in item.fact_ids:
            nid = normalize_fact_id(fid)
            if nid and nid not in known and nid not in unknown:
                unknown.append(nid)

    variants, absolute = _fact_variants(fs)
    text_numbers = _sheet_text_numbers(fs)
    own = _own_values(view)
    unsupported: list[str] = []
    checked = 0
    for text in view_texts(view):
        for tok in extract_numbers(text):
            if _ignored(tok, own):
                continue
            checked += 1
            if not _is_supported(tok, variants, absolute, text_numbers, rel_tol) and tok.text not in unsupported:
                unsupported.append(tok.text)

    ok = not unknown and len(unsupported) <= _MAX_UNSUPPORTED
    return GroundingReport(ok=ok, unknown_fact_ids=unknown, unsupported_numbers=unsupported,
                           checked_numbers=checked)


def _ignored(tok: NumberToken, own: list[float]) -> bool:
    """Numbers that are not claims about the world: years, short time windows ("3 months",
    "12-month"), the view's own forecast numbers, and the 0.5 / 50% base rate."""
    if tok.is_year or (tok.time_unit and tok.value <= 12):
        return True
    if tok.scale != 1.0 or tok.bps:
        return False
    return tok.value in _BASE_RATE_VALUES or any(_rounds_to(tok.value, tok.decimals, o) for o in own)


def _is_supported(tok: NumberToken, variants: list[float], absolute: list[float],
                  text_numbers: list[float], rel_tol: float) -> bool:
    if tok.scale != 1.0:
        # "$130.5B" must match an absolute USD/count fact at that scale, not any value ~130.5
        return any(_close(tok.value, tok.decimals, a / tok.scale, rel_tol) for a in absolute)
    if tok.bps:
        # "45 bps" is 0.45 percentage points
        return any(_close(tok.value / 100.0, tok.decimals + 2, v, rel_tol) for v in variants)
    if any(_close(tok.value, tok.decimals, v, rel_tol) for v in variants):
        return True
    return any(_rounds_to(tok.value, tok.decimals, t) for t in text_numbers)
