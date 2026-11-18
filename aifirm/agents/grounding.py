"""Numeric grounding checker: is every number an analyst wrote actually on the fact sheet?

Why (P2): financial QA benchmarks show LLMs confidently state wrong numbers, and multi-agent
trading systems have repeatedly shipped fixes for hallucinated figures. The Data Desk gives
every number an ID, analysts must cite IDs, and this module audits the free text of each
view. The check is deliberately lenient about *presentation* (rounding, "%", "$130.5B",
"45 bps") and strict about *provenance*: a number must match some fact value, or appear
verbatim in fact-sheet text the analyst was shown (labels such as "60 days", filing
events, headlines), or be the view's own forecast.

Matching is also *unit-aware*: a percentage ("7%") must match a percent fact (or a ratio
shown as a percent), and a dollar amount must match a dollar fact; neither can be "supported"
by a P/E multiple, an index level or a window length in a label ("60 days") that happens to
round to the same integer. Headline summaries are not shown to analysts (only titles are),
so they do not count as support.

Matching is *scoped by precision*, so an invented round number cannot pass by coincidence.
A slice holds a dozen or more percent facts, and an integer such as "7%" is a fair rounding
of any of them within +-0.5, so about one invented integer in five used to match some fact
by chance ("Short interest is 7% of float" passed because a price-vs-200-day fact was 6.57%).
Therefore:

* a number is checked leniently (rounding, 3% relative, 0.05 absolute) against the facts
  the view **cites** (anywhere in the view: a number in a scenario or a risk may quote a
  fact cited in the evidence);
* against the rest of the slice it must be written with informative precision (at least
  three significant digits, or one decimal and two significant digits: "12.3%", "$130.5B",
  "45.67") and must equal the fact rounded to that precision; a low-precision number
  ("7%", "$60B", "45 bps") that matches only an uncited fact is unsupported. Cite the fact
  you quote.

Citations: a non-abstaining view must cite the facts it uses. A view that writes numbers but
cites no fact ID, or that cites nothing although its slice holds only numeric facts (no
filing events or headlines to describe qualitatively), is flagged: that is what an answer
from memory looks like. Citations are resolved leniently first (:func:`resolve_fact_id`):
a bare key ("ret_3m") or a checklist-style pattern ("MKT.*_vol_20d") that names exactly
one fact on the sheet counts as that fact.

An ungrounded view is not discarded. It is flagged and its pooling weight is halved
(see :mod:`aifirm.agents.committee`), because a single stray number is weak evidence of a
bad forecast, while a pattern of them is strong evidence of fabrication.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from ..data.factsheet import Fact, FactSheet
from ..schemas import AnalystView

# Fact-shaped and other non-quantity tokens are masked before numbers are extracted, so
# digits inside identifiers ("NVDA.ret_3m", "MACRO.DGS10"), dates, SEC form names
# ("8-K", "10-Q", "Form 4"), 8-K item numbers ("Item 2.02"), index names ("S&P 500") and
# fiscal-period labels ("3Q24", "1H25"; "Q3" and "FY25" never match as numbers) are never
# mistaken for claimed quantities.
_MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z][A-Z0-9\-]*\.[A-Za-z][A-Za-z0-9_]*\b"),                # fact IDs
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.+\-]+Z?)?\b"),                    # ISO dates
    re.compile(r"\bT[-+]\d+d\b"),                                               # blind-mode relative dates
    re.compile(r"\b(?:8-K|10-K|10-Q|20-F|6-K|S-1|S-3|S-4|S-8|DEF\s*14A|SC\s*13[DG]|13[DGF]|Form\s+[345])(?:/A)?\b", re.I),
    re.compile(r"\bitems?\s+\d{1,2}\.\d{2}(?:\s*(?:,|and|&)\s*\d{1,2}\.\d{2})*", re.I),
    re.compile(r"(?:S&P|Russell|Nasdaq|Dow)\s*\d+", re.I),
    re.compile(r"\b\d+s\d+s\b"),                                                # "2s10s" curve
    re.compile(r"\b50\s*[/-]\s*50\b"),                                          # "a 50/50 call": odds, not a quantity
    re.compile(r"\b(?:[1-4]Q|[12]H)(?:\s?FY)?'?\d{2}(?:\d{2})?\b", re.I),             # period labels: "3Q24", "1H2025"
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

# Base-rate references the prompts themselves make ("about 0.5", "50%"); quoting them as a
# probability is fine, but "$50" or "50 days" is a claim like any other.
_BASE_RATE_PROB = 0.5
_BASE_RATE_PCT = 50.0
_MAX_UNSUPPORTED = 2
_PERCENT_UNITS = ("%", "%pts")       # facts a percentage may quote as they are
_RATIO_UNITS = ("ratio",)            # facts a percentage may quote x100 (0.42 -> "42%")


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

    @property
    def significant_digits(self) -> int:
        """Significant digits as written: "12.30" -> 4, "0.45" -> 2, "130" -> 2, "7" -> 1."""
        digits = f"{self.value:.{self.decimals}f}".replace(".", "").lstrip("0")
        if self.decimals == 0:
            digits = digits.rstrip("0")       # trailing zeros of an integer are not significant
        return max(1, len(digits))

    @property
    def precise(self) -> bool:
        """Written precisely enough that matching an uncited fact is unlikely to be chance."""
        sig = self.significant_digits
        return sig >= 3 or (self.decimals >= 1 and sig >= 2)


@dataclass
class GroundingReport:
    ok: bool
    unknown_fact_ids: list[str]
    unsupported_numbers: list[str]
    checked_numbers: int
    cited_fact_ids: int = 0         # citations (with repeats) that name a fact on the sheet
    citation_issue: str = ""        # set when a non-abstaining view cites no fact although it must

    def issues(self) -> list[str]:
        """Human-readable issues for ``ViewRecord.grounding_issues`` and the ledger."""
        out = [f"unknown fact id: {i}" for i in self.unknown_fact_ids]
        out += [f"unsupported number: {n}" for n in self.unsupported_numbers]
        if self.citation_issue:
            out.append(f"no citations: {self.citation_issue}")
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


@dataclass(frozen=True)
class _Variants:
    """Display variants of the numeric facts on a sheet, by how a number may quote them."""

    any: list[float]        # a bare number may quote any fact in any display form
    percent: list[float]    # "x%": percent facts as they are, ratios x100
    usd: list[float]        # "$x": dollar facts, raw or in thousands/millions/billions/trillions
    absolute: list[float]   # "$x B" / "x million": absolute USD and count facts


def _fact_variants(facts: Iterable[Fact]) -> _Variants:
    out = _Variants([], [], [], [])
    for f in facts:
        if isinstance(f.value, bool) or not isinstance(f.value, (int, float)):
            continue
        v = abs(float(f.value))
        out.any.append(v)
        if f.unit in _PERCENT_UNITS:
            out.percent.append(v)
        if f.unit in ("ratio", "x"):
            out.any.append(v * 100.0)           # a ratio quoted as a percentage
        if f.unit in _RATIO_UNITS:
            out.percent.append(v * 100.0)
        if f.unit in ("USD", "count"):
            out.absolute.append(v)
            scaled = [v / 1e3, v / 1e6, v / 1e9, v / 1e12]
            out.any.extend(scaled)
            if f.unit == "USD":
                out.usd.extend([v, *scaled])
    return out


_TEXT_PERCENT, _TEXT_TIME, _TEXT_OTHER, _TEXT_ANY = "percent", "time", "other", "any"


def _kind(tok: NumberToken) -> str:
    if tok.percent or tok.bps:
        return _TEXT_PERCENT
    return _TEXT_TIME if tok.time_unit else _TEXT_OTHER


def _sheet_text_numbers(fs: FactSheet) -> list[tuple[float, str]]:
    """Numbers printed in fact-sheet *text* the analyst saw, with their presentation: fact
    labels ("60 days", "52-week"), filing events, headline titles (summaries are not
    rendered) and data-gap notes."""
    texts = [f.label for f in fs.facts] + list(fs.events) + list(fs.data_gaps)
    texts += [h.title for h in fs.headlines]
    return [(t.value * t.scale, _kind(t)) for text in texts for t in extract_numbers(text)]


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


_PLACEHOLDER = re.compile(r"<[^>]*>|\{[^}]*\}")


def resolve_fact_id(fid: str, known: Iterable[str], ticker: str = "") -> str:
    """Map a citation to the fact ID it unambiguously names on the sheet.

    * decoration is stripped ("[NVDA.ret_3m]" -> "NVDA.ret_3m");
    * a bare key resolves to the ticker's fact ("ret_3m" -> "NVDA.ret_3m"), or to the only
      fact with that key ("DGS10" -> "MACRO.DGS10");
    * a checklist-style pattern resolves to the only fact it matches
      ("MKT.*_vol_20d" or "MKT.<benchmark>_vol_20d" -> "MKT.SPY_vol_20d");
    * a difference in letter case only is forgiven when one fact matches.

    Anything else comes back normalized and unchanged (and is then an unknown ID)."""
    nid = normalize_fact_id(fid)
    known = known if isinstance(known, (set, frozenset, dict)) else set(known)
    if not nid or nid in known:
        return nid
    if "." not in nid:
        if ticker and f"{ticker}.{nid}" in known:
            return f"{ticker}.{nid}"
        bare = [k for k in known if k.split(".", 1)[-1] == nid]
        if len(bare) == 1:
            return bare[0]
    if "*" in nid or _PLACEHOLDER.search(nid):
        pattern = _PLACEHOLDER.sub("*", nid).replace("?", "[?]")
        matches = [k for k in known if fnmatch.fnmatchcase(k, pattern)]
        if len(matches) == 1:
            return matches[0]
    folded = [k for k in known if k.lower() == nid.lower()]
    return folded[0] if len(folded) == 1 else nid


def resolve_fact_ids(ids: Iterable[str], known: Iterable[str], ticker: str = "") -> list[str]:
    """:func:`resolve_fact_id` over a list, dropping empty entries and repeats (order kept)."""
    known = known if isinstance(known, (set, frozenset, dict)) else set(known)
    out: list[str] = []
    for fid in ids:
        rid = resolve_fact_id(fid, known, ticker)
        if rid and rid not in out:
            out.append(rid)
    return out


def check_view(view: AnalystView, fs: FactSheet, rel_tol: float = 0.03,
               prompt_numbers: tuple[float, ...] = ()) -> GroundingReport:
    """Audit ``view`` against ``fs``.

    ``fs`` should be the fact sheet the analyst was actually shown (its slice), so that a
    cited ID the analyst could not have seen counts as unknown. ``prompt_numbers`` are
    numbers the prompt itself states (e.g. the horizon in trading days), which the analyst
    may repeat. ``ok`` is False if any cited fact ID is unknown or more than two distinct
    numbers are unsupported, or if the view must cite facts and cites none (see the module
    docstring). Citations are resolved with :func:`resolve_fact_id` first. A number is
    matched leniently against the cited facts and, only when written with informative
    precision, strictly against the rest of the slice (see the module docstring).
    """
    known = fs.ids()
    unknown: list[str] = []
    cited = 0
    cited_ids: set[str] = set()
    for item in view.evidence:
        for fid in item.fact_ids:
            nid = resolve_fact_id(fid, known, fs.ticker)
            if not nid:
                continue
            if nid in known:
                cited += 1
                cited_ids.add(nid)
            elif nid not in unknown:
                unknown.append(nid)

    variants = _Scoped(cited=_fact_variants(f for f in fs.facts if f.id in cited_ids),
                       sheet=_fact_variants(fs.facts))
    text_numbers = _sheet_text_numbers(fs) + [(float(n), _TEXT_ANY) for n in prompt_numbers]
    own = _own_values(view)
    unsupported: list[str] = []
    checked = 0
    for text in view_texts(view):
        for tok in extract_numbers(text):
            if _ignored(tok, own):
                continue
            checked += 1
            if not _is_supported(tok, variants, text_numbers, rel_tol) and tok.text not in unsupported:
                unsupported.append(tok.text)

    citation_issue = ""
    if not view.abstain and cited == 0 and not unknown:
        if checked:
            citation_issue = f"the view states {checked} number(s) but cites no fact ID"
        elif not (fs.events or fs.headlines):
            citation_issue = "the view cites no fact ID, and its slice has only numeric facts"
    ok = not unknown and len(unsupported) <= _MAX_UNSUPPORTED and not citation_issue
    return GroundingReport(ok=ok, unknown_fact_ids=unknown, unsupported_numbers=unsupported,
                           checked_numbers=checked, cited_fact_ids=cited, citation_issue=citation_issue)


def _ignored(tok: NumberToken, own: list[float]) -> bool:
    """Numbers that are not claims about the world: years, short time windows ("3 months",
    "12-month"), the view's own forecast numbers, and the base rate written as a probability
    ("0.5" or "50%"; "$50" or "50 days" are checked like any other number)."""
    if tok.is_year or (tok.time_unit and tok.value <= 12):
        return True
    if tok.scale != 1.0 or tok.bps or tok.dollar:
        return False
    if (tok.value == _BASE_RATE_PROB and not tok.percent and not tok.time_unit) or \
            (tok.value == _BASE_RATE_PCT and tok.percent):
        return True
    return any(_rounds_to(tok.value, tok.decimals, o) for o in own)


@dataclass(frozen=True)
class _Scoped:
    cited: _Variants        # facts the view cites: lenient matching
    sheet: _Variants        # every fact on the slice: strict matching of precise numbers only


def _matches(tok: NumberToken, variants: _Variants, close: Callable[[float, int, float], bool]) -> bool:
    """Does ``tok`` quote one of ``variants`` in a unit-compatible form, judged by ``close``?"""
    if tok.scale != 1.0:
        # "$130.5B" must match an absolute USD/count fact at that scale, not any value ~130.5
        return any(close(tok.value, tok.decimals, a / tok.scale) for a in variants.absolute)
    if tok.bps:
        # "45 bps" is 0.45 percentage points
        return any(close(tok.value / 100.0, tok.decimals + 2, v) for v in variants.any)
    candidates = variants.percent if tok.percent else variants.usd if tok.dollar else variants.any
    return any(close(tok.value, tok.decimals, v) for v in candidates)


def _is_supported(tok: NumberToken, variants: _Scoped, text_numbers: list[tuple[float, str]],
                  rel_tol: float) -> bool:
    if _matches(tok, variants.cited, lambda n, d, c: _close(n, d, c, rel_tol)):
        return True
    if tok.precise and _matches(tok, variants.sheet, _rounds_to):
        return True
    if tok.scale != 1.0 or tok.bps:
        return False
    if tok.percent:
        kinds: tuple[str, ...] = (_TEXT_PERCENT, _TEXT_ANY)
    elif tok.dollar:
        kinds = (_TEXT_OTHER, _TEXT_ANY)
    else:
        kinds = (_TEXT_PERCENT, _TEXT_TIME, _TEXT_OTHER, _TEXT_ANY)
    return any(_rounds_to(tok.value, tok.decimals, t) for t, kind in text_numbers if kind in kinds)
