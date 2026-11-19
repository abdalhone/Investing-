"""Firm configuration. Load from a TOML file (see ``examples/firm.toml``) or build in code.

Every default below is a deliberate, research-backed choice; see docs/ARCHITECTURE.md
for the reasoning and docs/research/RESEARCH_REPORT.md for the evidence.

Validation (fail loudly, never fall back silently): every section forbids unknown keys,
so a misspelled setting or section (``max_positon``, ``[portfollio]``) is an error instead
of quietly leaving the default in force; non-finite numbers are rejected; and numeric
settings are range-checked when a config is built or loaded. Fractions are written as
fractions (``max_position = 0.10`` for 10%); a value such as ``15`` where a fraction is
expected is reported with the fraction it probably meant. Assigning to an attribute
after construction is *not* re-validated (tests and code use that to build variants).

``runs_dir``: a relative path is relative to the directory of the TOML file it came from
(:meth:`FirmConfig.load`), not to the shell's working directory, so a scheduled run from
another directory still finds the same book and the same kill switch. Use
:meth:`FirmConfig.runs_path` for the absolute directory; a config built in code has no
file, so its relative ``runs_dir`` is relative to the working directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, ValidationInfo, field_validator, model_validator

from .universes import file_tickers, preset_tickers

# Calendar days after a model's training cutoff that still count as contaminated (research
# 7.2: vendor cutoffs are unreliable). One rule for the backtest's contaminated period and the
# forecast store's ``pre_cutoff`` flag (evaluation.cutoff_buffer_days overrides it per config).
DEFAULT_CUTOFF_BUFFER_DAYS = 31

# Training-data cutoffs per model. Forecasts made on dates BEFORE a model's cutoff may be
# contaminated by memorized outcomes, so backtests report those periods separately and
# never count them as evidence of skill. If a model is missing here, the cutoff is
# assumed to be "today" (i.e. every historical date is treated as contaminated).
# Verify these against Anthropic's models overview page when you change models.
MODEL_TRAINING_CUTOFFS: dict[str, str] = {
    "claude-opus-5-5": "2026-06-30",
}

BROKERS = ("paper", "alpaca")


# --------------------------------------------------------------------------- validation helpers


@dataclass(frozen=True)
class _Range:
    """Allowed interval for one numeric setting. ``fraction``: the value is a share of
    something (NAV, a probability, a drawdown), so ``15`` was probably meant as 15%."""

    lo: float | None = None
    hi: float | None = None
    lo_open: bool = False      # True: value must be > lo (else >= lo)
    hi_open: bool = False      # True: value must be < hi (else <= hi)
    fraction: bool = False

    def contains(self, v: float) -> bool:
        if self.lo is not None and (v <= self.lo if self.lo_open else v < self.lo):
            return False
        if self.hi is not None and (v >= self.hi if self.hi_open else v > self.hi):
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.lo is not None:
            parts.append(f"{'>' if self.lo_open else '>='} {self.lo:g}")
        if self.hi is not None:
            parts.append(f"{'<' if self.hi_open else '<='} {self.hi:g}")
        return " and ".join(parts)

    def problem(self, value: object) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None                      # type errors are pydantic's job
        v = float(value)
        if not math.isfinite(v):
            return f"{value!r} must be a finite number"
        if self.contains(v):
            return None
        msg = f"{value!r} is out of range: it must be {self.describe()}"
        if self.fraction and 1.0 < v <= 100.0 and self.contains(v / 100.0):
            frac = v / 100.0
            text = f"{frac:.2f}" if round(frac, 2) == frac else f"{frac:g}"
            msg += f". It looks like a percentage: use {text} for {v:g}% (fractions, not percent)"
        return msg


def _fraction(lo: float = 0.0, hi: float = 1.0, *, lo_open: bool = False, hi_open: bool = False) -> _Range:
    return _Range(lo, hi, lo_open, hi_open, fraction=True)


class _Section(BaseModel):
    """Base of every config model: unknown keys are errors (a typo must not silently leave a
    default in force), NaN/inf are rejected, each field named in ``_RANGES`` is range-checked
    (per field, so it is reported together with any unknown key in the same section), and
    :meth:`_cross_checks` (rules between fields) runs once every field is valid."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    _RANGES: ClassVar[dict[str, _Range]] = {}

    def _cross_checks(self) -> list[str]:
        return []

    @field_validator("*", mode="after")
    @classmethod
    def _validate_range(cls, value: object, info: ValidationInfo) -> object:
        rng = cls._RANGES.get(info.field_name or "")
        problem = rng.problem(value) if rng is not None else None
        if problem:
            raise ValueError(problem)
        return value

    @model_validator(mode="after")
    def _validate_cross_checks(self):
        problems = self._cross_checks()
        if problems:      # "; " separates problems (describe_config_error prints one per line)
            raise ValueError("; ".join(problems))
        return self


def describe_config_error(exc: ValidationError) -> str:
    """One readable line per problem, with the TOML path (``risk.drawdown_freeze``)."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(top level)"
        if err.get("type") == "extra_forbidden":
            lines.append(f"{loc}: unknown setting (a typo? examples/firm.toml lists every valid key)")
        else:
            msg = str(err.get("msg", "")).removeprefix("Value error, ")
            lines.extend(f"{loc}: {part}" for part in msg.split("; "))
    return "\n  ".join(lines)


# --------------------------------------------------------------------------- sections


class ModelConfig(_Section):
    analyst_models: list[str] = Field(default_factory=lambda: ["claude-opus-5-5"],
                                      description="Models used by LLM analysts. With several models, samples rotate through them (model diversity reduces correlated errors).")
    samples_per_analyst: int = Field(1, description="Independent samples per analyst per ticker. >1 measures instability; dispersion shrinks position size.")
    analyst_effort: str = "medium"
    committee_model: str = "claude-opus-5-5"
    committee_effort: str = "high"
    risk_model: str = "claude-opus-5-5"
    risk_effort: str = "medium"
    use_server_fallbacks: bool = Field(True, description="Ask the API to retry a refused request on a fallback model (server-side fallbacks beta).")
    news_reader_model: str = Field("claude-opus-5-5", description="Model of the quarantined news reader (the only call that sees raw headline text).")
    news_reader_effort: str = "low"
    max_tokens: int = 16000
    max_parallel_calls: int = 4

    _RANGES: ClassVar[dict[str, _Range]] = {
        "samples_per_analyst": _Range(1),
        "max_tokens": _Range(1),
        "max_parallel_calls": _Range(1),
    }

    def _cross_checks(self) -> list[str]:
        if not any(m.strip() for m in self.analyst_models):
            return ["analyst_models must name at least one model"]
        return []


COMBINATIONS = ("overlay", "pool")


class ScreenConfig(_Section):
    """Broad-universe quant screen with an LLM deep dive on the top band only
    (RESEARCH 7.3, 9.2 "cadence and universe", 9.3 item 1: breadth is what gives statistical power).

    When ``enabled``, the investable universe is the union of the bundled ``preset``, the
    tickers listed in ``file`` and the inline ``tickers`` (plus ``universe.tickers``, which are
    always covered). Every decision then: (1) builds fact sheets for the whole universe
    (prices, SEC fundamentals, no LLM; in a live run the headline feed is fetched for every
    name too, but only the deep-dive names' headlines are read by the news reader and every
    other name's are withheld unread), (2) scores the deterministic quant over the
    whole cross-section, (3) keeps names that pass the point-in-time filters (last close at
    least ``risk.min_price``, 20-day average dollar volume at least ``min_dollar_volume``, a
    recent price), and (4) sends the top ``deep_dive_top_n`` of them by quant probability
    (only names above 0.5) plus every held name to the LLM committee. Names outside that
    deep-dive set get no new position. Held names are always deep-dived (not configurable):
    a held position must be reviewed, and dropping it from review would force a sale.

    Survivorship: a list of today's members (such as the bundled ``sp100``) omits names that
    were dropped or delisted, so backtests on it are biased upward; reports say so."""

    enabled: bool = Field(False, description="Screen a broad universe and deep-dive only the top band.")
    preset: str = Field("", description="A bundled universe list, e.g. \"sp100\" (current members: survivorship-biased).")
    file: str = Field("", description="A text file with one ticker per line ('#' comments). A relative path is relative "
                                      "to the config file's directory.")
    tickers: list[str] = Field(default_factory=list, description="Inline tickers added to the screened universe.")
    deep_dive_top_n: int = Field(12, description="Names (ranked by quant probability, only those above 0.5) sent to the "
                                                 "LLM committee each cycle, in addition to every held name.")
    min_dollar_volume: float = Field(10_000_000.0, description="Minimum 20-day average daily dollar volume (close x "
                                                               "volume, data up to the decision close) to be eligible.")

    _RANGES: ClassVar[dict[str, _Range]] = {
        "deep_dive_top_n": _Range(1, 1000),
        "min_dollar_volume": _Range(0.0),
    }

    def _cross_checks(self) -> list[str]:
        problems = []
        if self.enabled and not (self.preset.strip() or self.file.strip() or any(t.strip() for t in self.tickers)):
            problems.append("screen.enabled needs a universe source: preset, file or tickers")
        if self.preset.strip():
            from .universes import PRESETS        # local import: the universes package needs no config
            if self.preset.strip().lower() not in PRESETS:
                problems.append(f"screen.preset = {self.preset!r} is unknown; bundled presets: {', '.join(PRESETS)}")
        return problems


class UniverseConfig(_Section):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "PG"])
    benchmark: str = "SPY"
    sectors: dict[str, str] = Field(default_factory=dict, description="Optional manual sector labels (ticker -> sector) used for sector limits. Defaults to SEC SIC descriptions.")
    ciks: dict[str, str | int] = Field(default_factory=dict, description="Optional SEC filer pins (ticker -> CIK) used for that ticker's fundamentals and filing events on every date. Use when SEC's current ticker table maps a ticker to the wrong filer for your dates (e.g. after a holding-company reorganization).")
    issuer_domains: dict[str, list[str]] = Field(default_factory=dict, description="Optional investor-relations domains per ticker (e.g. NVDA = [\"nvidia.com\"]): headlines linking there count as tier A (primary source) for the news reader.")
    screen: ScreenConfig = Field(default_factory=ScreenConfig)

    def _cross_checks(self) -> list[str]:
        bench = self.benchmark.strip().upper()
        if not bench:
            return ["benchmark must not be empty"]
        screened = self.screen.enabled and (self.screen.preset.strip() or self.screen.file.strip()
                                            or any(t.strip().upper() != bench for t in self.screen.tickers if t.strip()))
        if not screened and not any(t.strip() and t.strip().upper() != bench for t in self.tickers):
            return ["tickers must list at least one stock other than the benchmark"]
        bad = [f"{t} = {c!r}" for t, c in self.ciks.items()
               if isinstance(c, bool) or not str(c).strip().isdigit() or len(str(c).strip()) > 10]
        if bad:
            return [f"ciks must map tickers to SEC CIK numbers (up to 10 digits): {', '.join(bad)}"]
        return []

    def cik_pins(self) -> dict[str, str]:
        """``ciks`` normalized: upper-case tickers -> 10-digit CIK strings."""
        return {t.strip().upper(): str(c).strip().zfill(10) for t, c in self.ciks.items()}

    def issuer_domain_map(self) -> dict[str, tuple[str, ...]]:
        """``issuer_domains`` normalized: upper-case tickers -> lower-case domains."""
        return {t.strip().upper(): tuple(d.strip().lower() for d in ds if d.strip())
                for t, ds in self.issuer_domains.items()}


class AnalystsConfig(_Section):
    enabled: list[str] = Field(default_factory=lambda: ["fundamental", "technical", "macro", "events", "quant"],
                               description="Which analysts sit on the committee. 'quant' is deterministic (no LLM).")
    max_lessons_in_prompt: int = 5

    _RANGES: ClassVar[dict[str, _Range]] = {"max_lessons_in_prompt": _Range(0)}

    def _cross_checks(self) -> list[str]:
        return [] if self.enabled else ["enabled must list at least one analyst (for example [\"quant\"])"]


class ProcessConfig(_Section):
    horizon_days: int = Field(21, description="Forecast horizon in trading days (21 ~ 1 month).")
    rebalance_every_days: int = Field(21, description="Trading days between decision cycles in backtests.")
    challenge_enabled: bool = True
    challenge_dispersion_threshold: float = Field(0.12, description="Std-dev of analyst probabilities above which a red-team challenge round runs.")
    trim_fraction: float = Field(0.2, description="Fraction trimmed from each tail when pooling probabilities in log-odds space.")
    memo_enabled: bool = True
    memo_min_edge: float = Field(0.03, description="The chair writes an IC memo when |pooled prob - 0.5| >= this, when the pooled prob clears portfolio.min_prob_to_hold (so no name is sized unreviewed), or when the name is held. With memos enabled, an unreviewed name that is not held gets no position.")
    risk_review_enabled: bool = True
    blind_mode: bool = Field(False, description="Hide ticker, company name and absolute dates from LLM analysts (for historical research: reduces memorization leakage).")
    news_enabled: bool = True
    max_llm_cost_usd_per_run: float = 10.0
    combination: str = Field("overlay", description="How the quant core and the LLM analysts combine (RESEARCH 6.6). "
                                                    "'overlay': p_final = p_quant + clip(p_llm - 0.5, +/- max tilt), "
                                                    "the LLM pool's own edge tilts the quant inside a bound (an LLM pool "
                                                    "at 0.5 leaves the quant unchanged); 'pool': every "
                                                    "analyst, quant included, is one member of the trimmed log-odds pool.")
    overlay_max_tilt: float = Field(0.05, description="Largest move (in probability) the LLM pool may apply to the quant "
                                                      "core. Halved, then set to 0, when resolved forecasts show the LLM "
                                                      "pool ranking worse than the quant (never raised above this).")
    overlay_min_dates: int = Field(12, description="Resolved decision dates with both an LLM-pool and a quant forecast "
                                                   "needed before the adaptive bound may shrink the tilt.")
    believability_min_dates: int = Field(20, description="Distinct resolved decision dates an agent needs before its "
                                                         "track record tilts its pooling weight (equal weights until then).")
    news_reader_enabled: bool = Field(True, description="Live headlines are read by the quarantined news reader into "
                                                        "typed, tier-labelled events; analysts never see raw headline text.")
    news_max_headlines: int = Field(20, description="Most recent headlines read per ticker and cycle.")
    news_isolate_untrusted: bool = Field(True, description="Read tier A/B and tier C headlines in separate reader calls.")
    tier_gate_enabled: bool = Field(True, description="Cap a bullish LLM view (P > 0.5 or excess > 0) at 0.5 and excess "
                                                      "0 unless a positive evidence item free of tier-C news cites a "
                                                      "directional fact, a filing or a tier A/B news event (the events "
                                                      "analyst needs a filing or tier A/B news); a raised revision "
                                                      "needs the same support in its new evidence.")

    _RANGES: ClassVar[dict[str, _Range]] = {
        "horizon_days": _Range(1, 252),                 # trading days: one day to one year
        "rebalance_every_days": _Range(1, 252),
        "challenge_dispersion_threshold": _fraction(0.0, 0.5),   # std-dev of probabilities is at most 0.5
        "trim_fraction": _fraction(0.0, 0.5, hi_open=True),
        "memo_min_edge": _fraction(0.0, 0.5),
        "max_llm_cost_usd_per_run": _Range(0.0, lo_open=True),
        "overlay_max_tilt": _fraction(0.0, 0.5),
        "overlay_min_dates": _Range(2, 10_000),
        "believability_min_dates": _Range(0, 10_000),
        "news_max_headlines": _Range(1, 200),
    }

    def _cross_checks(self) -> list[str]:
        if self.combination not in COMBINATIONS:
            return [f"combination = {self.combination!r} is unknown; use one of {', '.join(COMBINATIONS)}"]
        return []


class CalibrationConfig(_Section):
    """Per-agent Platt calibration of LLM analyst probabilities before pooling
    (:mod:`aifirm.journal.calibration`). Fitted point-in-time from resolved forecasts; the
    identity until an agent has ``min_n`` resolved questions from at least ``min_dates``
    distinct decision dates; shrunk toward the identity by
    a prior worth ``prior_strength`` forecasts. Forecast records keep the RAW probability.
    The deterministic quant is never calibrated."""

    enabled: bool = True
    min_n: int = Field(30, description="Resolved questions an agent needs before its map moves from the identity.")
    min_dates: int = Field(20, description="Distinct resolved decision dates an agent needs before its map moves from "
                                           "the identity (one date's questions share one market episode).")
    prior_strength: float = Field(50.0, description="Pseudo-forecasts at the identity that the fit is shrunk toward.")

    _RANGES: ClassVar[dict[str, _Range]] = {
        "min_n": _Range(0, 1_000_000),
        "min_dates": _Range(0, 10_000),
        "prior_strength": _Range(0.0, 1e9),
    }


class PortfolioConfig(_Section):
    core_weight: float = Field(0.0, description="Fraction of capital held passively in the benchmark (core-satellite). The agents manage the rest.")
    long_only: bool = True
    max_position: float = Field(0.08, description="Target cap per name used by sizing (research 6.4: 8% target; the hard "
                                                  "cap is risk.max_position_hard).")
    max_sector: float = Field(0.25, description="Cap on the gross weight of one sector (research 6.4).")
    min_cash: float = 0.02
    max_gross: float = 1.0
    min_prob_to_hold: float = Field(0.52, description="Minimum pooled P(outperform) to hold a long.")
    kelly_fraction: float = Field(0.25, description="Fraction of the Kelly-optimal weight to use (full Kelly is far too aggressive with estimated edges).")
    target_vol: float = Field(0.15, description="Annualized volatility target for the active sleeve (research 6.4: "
                                                "12-15%; the excess stays in cash, no leverage).")
    no_trade_band: float = Field(0.005, description="Skip trades that change a weight by less than this (0.5% of NAV).")
    max_turnover: float = Field(0.25, description="Max one-way turnover per rebalance (fraction of NAV).")
    rebalance_fraction: float = Field(0.5, description="Each rebalance moves this fraction of the way from the current "
                                                       "to the target weight (research 6.4/6.5); forced exits and "
                                                       "hard-limit repairs go all the way.")
    dispersion_shrink: float = Field(2.0, description="Edge is multiplied by max(0, 1 - dispersion_shrink * dispersion).")

    # Upper bounds well above any sensible value still catch percent-style typos: a value that
    # never binds (max_turnover above 2, target_vol of several hundred percent) is a way to
    # switch a rule off, but 50 or 18 is a percentage typed as a number.
    _RANGES: ClassVar[dict[str, _Range]] = {
        "core_weight": _fraction(),
        "max_position": _fraction(),
        "max_sector": _fraction(),
        "min_cash": _fraction(hi_open=True),
        "max_gross": _fraction(0.0, 3.0, lo_open=True),
        "min_prob_to_hold": _fraction(0.5, 1.0),       # below 0.5 would hold names expected to lag
        "kelly_fraction": _fraction(),                  # more than full Kelly is never sensible
        "target_vol": _fraction(0.0, 5.0, lo_open=True),
        "no_trade_band": _fraction(0.0, 0.5),
        "max_turnover": _fraction(0.0, 10.0),
        "rebalance_fraction": _fraction(0.0, 1.0, lo_open=True),
        "dispersion_shrink": _Range(0.0),
    }


class RiskConfig(_Section):
    drawdown_derisk: float = Field(0.10, description="At this drawdown from peak NAV, gross exposure of the active sleeve "
                                                     "is halved; full gross returns once half of the drawdown (peak to "
                                                     "trough) has been recovered (research 6.4).")
    drawdown_freeze: float = Field(0.20, description="At this drawdown, the active sleeve goes to cash/core until a human resets it.")
    max_position_hard: float = Field(0.10, description="Hard cap per name (research 6.4). Construction never leaves a "
                                                       "name above it, drift included, and the risk engine clips any "
                                                       "target above it. Sizing targets portfolio.max_position.")
    daily_loss_halt: float = Field(0.05, description="A one-day NAV loss this large halts new buys for the next cycle.")
    max_adv_participation: float = Field(0.02, description="Max order size as a fraction of 20-day average dollar volume.")
    restricted: list[str] = Field(default_factory=list)
    kill_switch_file: str = Field("KILL", description="If this file exists in the runs directory, no orders are generated.")
    min_price: float = 2.0

    _RANGES: ClassVar[dict[str, _Range]] = {
        "drawdown_derisk": _fraction(0.0, 1.0, lo_open=True, hi_open=True),
        "drawdown_freeze": _fraction(0.0, 1.0, lo_open=True, hi_open=True),
        "daily_loss_halt": _fraction(0.0, 1.0, lo_open=True, hi_open=True),
        "max_adv_participation": _fraction(0.0, 1.0, lo_open=True),
        "min_price": _Range(0.0),
        "max_position_hard": _fraction(),
    }

    def _cross_checks(self) -> list[str]:
        problems = []
        if self.drawdown_derisk > self.drawdown_freeze:
            problems.append(f"drawdown_derisk = {self.drawdown_derisk:g} must not exceed drawdown_freeze = "
                            f"{self.drawdown_freeze:g} (de-risk first, freeze at the deeper drawdown)")
        name = self.kill_switch_file
        if not name.strip() or name in (".", "..") or "/" in name or "\\" in name:
            problems.append(f"kill_switch_file = {name!r} must be a plain file name inside runs_dir (e.g. \"KILL\")")
        return problems


class CostConfig(_Section):
    commission_bps: float = 0.5
    slippage_bps: float = 5.0
    impact_coefficient: float = Field(0.1, description="Square-root market impact: cost_bps += coef * 1e4 * sqrt(order / ADV) * daily_vol.")

    _RANGES: ClassVar[dict[str, _Range]] = {
        "commission_bps": _Range(0.0, 10_000.0, hi_open=True),    # basis points of notional
        "slippage_bps": _Range(0.0, 10_000.0, hi_open=True),
        "impact_coefficient": _Range(0.0),
    }


class ExecutionConfig(_Section):
    broker: str = Field("paper", description="'paper' (simulated) or 'alpaca' (Alpaca paper/live API).")
    require_approval: bool = Field(True, description="Ask a human to approve orders before sending them to a real broker.")
    initial_cash: float = 100_000.0

    _RANGES: ClassVar[dict[str, _Range]] = {"initial_cash": _Range(0.0, lo_open=True)}

    def _cross_checks(self) -> list[str]:
        if self.broker not in BROKERS:
            return [f"broker = {self.broker!r} is unknown; use one of {', '.join(BROKERS)}"]
        return []


class EvaluationConfig(_Section):
    """How a backtest is judged (P10): "is there skill, and would we know?", and whether the
    live book keeps a quant-only shadow book to be judged against (RESEARCH 6.6, 7.4 S6).

    None of these settings changes a decision, an order or a fill of the strategy; they only
    change what is measured. They are therefore left out of :meth:`FirmConfig.fingerprint`
    (changing them is not a new trial for the deflated Sharpe ratio, and adding this
    section did not change any existing fingerprint)."""

    n_random_books: int = Field(1000, description="Composition-matched random books for the null distribution "
                                                  "(0 switches them off).")
    random_seed: int = Field(20240917, description="Seed for the random books (results are reproducible).")
    random_books_sector_matched: bool = Field(False, description="Draw each random replacement from the same sector "
                                                                 "as the name it replaces (when sector labels exist).")
    cost_multipliers: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0],
                                          description="Every cost component (commission, spread, impact) is "
                                                      "multiplied by each k and the SAME decisions are re-executed.")
    breakeven_cap: float = Field(50.0, description="Largest cost multiplier searched for the breakeven; beyond it the "
                                                   "breakeven is reported as '> cap'.")
    cutoff_buffer_days: int = Field(DEFAULT_CUTOFF_BUFFER_DAYS, description="Calendar days after a model's training cutoff that still count as "
                                                    "contaminated (vendor cutoffs are unreliable; research 7.2).")
    ic_min_names: int = Field(5, description="Fewest names with a forecast and a realised return for a date's "
                                             "rank IC to count.")
    factor_proxies: bool = Field(True, description="Regress strategy returns on ETF factor proxies (SPY, IWM, IWD/IWF, "
                                                   "MTUM, QUAL, USMV; BIL as cash) when their prices are available.")
    shadow_book: bool = Field(True, description="Every `aifirm run` also decides a quant-only shadow book (same "
                                                "universe, screen, limits and costs; paper only) under "
                                                "<book>/shadow, so the LLM's value added net of its cost is measured "
                                                "forward. Never changes the live decision.")

    _RANGES: ClassVar[dict[str, _Range]] = {
        "n_random_books": _Range(0, 100_000),
        "random_seed": _Range(0),
        "breakeven_cap": _Range(1.0, 1000.0, lo_open=True),
        "cutoff_buffer_days": _Range(0, 3650),
        "ic_min_names": _Range(3, 100_000),
    }

    def _cross_checks(self) -> list[str]:
        problems = []
        if not self.cost_multipliers:
            problems.append("cost_multipliers must list at least one multiplier (for example [0.0, 1.0, 2.0])")
        bad = [k for k in self.cost_multipliers if not math.isfinite(k) or k < 0 or k > self.breakeven_cap]
        if bad:
            problems.append(f"cost_multipliers {bad} must be finite, >= 0 and <= breakeven_cap "
                            f"({self.breakeven_cap:g})")
        return problems


class FirmConfig(_Section):
    name: str = "AI Investment Firm"
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    analysts: AnalystsConfig = Field(default_factory=AnalystsConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runs_dir: str = Field("runs", description="Book and backtest directory. Relative paths are relative to the "
                                              "config file's directory (see FirmConfig.runs_path).")

    # Directory of the TOML file this config was loaded from (None when built in code).
    # A private attribute: not part of model_dump(), so it never changes the fingerprint.
    _source_dir: Path | None = PrivateAttr(default=None)

    def _cross_checks(self) -> list[str]:
        problems = [] if self.runs_dir.strip() else ["runs_dir must not be empty"]
        if self.portfolio.max_position > self.risk.max_position_hard:
            problems.append(f"portfolio.max_position = {self.portfolio.max_position:g} must not exceed "
                            f"risk.max_position_hard = {self.risk.max_position_hard:g} (the sizing target sits "
                            "under the hard cap)")
        if self.process.combination == "overlay" and "quant" not in self.analysts.enabled:
            problems.append("process.combination = \"overlay\" tilts the deterministic quant core, so analysts.enabled "
                            "must include \"quant\" (or use combination = \"pool\")")
        return problems

    @classmethod
    def load(cls, path: str | Path) -> "FirmConfig":
        """Load and validate a TOML file. Raises ``FileNotFoundError``, ``tomllib.TOMLDecodeError``
        (a ``ValueError``) or ``pydantic.ValidationError`` (unknown keys, values out of range)."""
        path = Path(path).expanduser()
        with open(path, "rb") as fh:
            cfg = cls.model_validate(tomllib.load(fh))
        cfg._source_dir = path.resolve().parent
        scr = cfg.universe.screen
        if scr.enabled and scr.file.strip() and not cfg.screen_file_path().is_file():   # fail at load, not mid-run
            raise FileNotFoundError(f"universe.screen.file {scr.file!r} not found (resolved to {cfg.screen_file_path()})")
        return cfg

    def revalidated(self, data: dict) -> "FirmConfig":
        """``data`` (a complete ``model_dump()``, typically this config's with some values
        changed) validated as a new config that keeps this config's source directory, so a
        relative ``runs_dir`` or screen file resolves exactly as before. Unlike attribute
        assignment, every range and cross-check runs again. Raises
        ``pydantic.ValidationError`` like :meth:`load`, and ``FileNotFoundError`` when an
        enabled screen names a file that does not exist."""
        cfg = type(self).model_validate(data)
        cfg._source_dir = self._source_dir
        scr = cfg.universe.screen
        if scr.enabled and scr.file.strip() and not cfg.screen_file_path().is_file():
            raise FileNotFoundError(f"universe.screen.file {scr.file!r} not found (resolved to {cfg.screen_file_path()})")
        return cfg

    def with_frozen_universe(self) -> "FirmConfig":
        """A copy whose screened universe has been read once and is fixed from then on.

        With ``universe.screen`` enabled, the screen's sources (``preset``, ``file``, inline
        ``tickers``) are replaced by the resolved list (:meth:`investable_tickers`), so every
        later read sees the same names even if the universe file is edited while a long
        backtest runs. :meth:`fingerprint` and :meth:`universe_digest` are unchanged (both
        hash the resolved list). Without the screen, or when nothing resolves (the run then
        fails loudly on its own), returns ``self``. Raises like :meth:`investable_tickers`."""
        if not self.universe.screen.enabled:
            return self
        resolved = self.investable_tickers()
        if not resolved:
            return self
        data = self.model_dump()
        data["universe"]["screen"].update(preset="", file="", tickers=resolved)
        return self.revalidated(data)

    def screen_file_path(self) -> Path:
        """Absolute path of ``universe.screen.file``: a relative path is anchored at the
        config file's directory (like ``runs_dir``), else at the working directory."""
        p = Path(self.universe.screen.file).expanduser()
        if not p.is_absolute():
            p = (self._source_dir if self._source_dir is not None else Path.cwd()) / p
        return p.resolve()

    def runs_path(self) -> Path:
        """Absolute ``runs_dir``. A relative value is anchored at the directory of the config
        file it was loaded from, so the book, the trial ledger and the kill switch do not
        depend on the directory a command runs from; without a file (a config built in
        code), at the current working directory."""
        p = Path(self.runs_dir).expanduser()
        if not p.is_absolute():
            p = (self._source_dir if self._source_dir is not None else Path.cwd()) / p
        return p.resolve()

    def investable_tickers(self) -> list[str]:
        """Investable tickers in config order: upper-cased, de-duplicated, benchmark excluded
        (the benchmark is the passive core, never an active name).

        With ``universe.screen`` enabled the list is ``universe.tickers`` followed by the
        screen's sources: the bundled ``preset``, the tickers in ``file`` (relative to the
        config file, :meth:`screen_file_path`) and the inline ``tickers``. An unknown preset
        raises ``KeyError`` and a missing file ``FileNotFoundError`` (fail loudly: a silently
        empty universe would never trade and never say why)."""
        bench = self.universe.benchmark.upper()
        sources = list(self.universe.tickers)
        scr = self.universe.screen
        if scr.enabled:
            if scr.preset.strip():
                sources += preset_tickers(scr.preset)
            if scr.file.strip():
                sources += file_tickers(self.screen_file_path())
            sources += list(scr.tickers)
        out: list[str] = []
        for t in sources:
            u = t.strip().upper()
            if u and u != bench and u not in out:
                out.append(u)
        return out

    def universe_digest(self) -> str:
        """Short hash of :meth:`investable_tickers` (what the firm can trade), for the trial
        ledger; ``"unreadable"`` when the screen's universe file or preset cannot be read."""
        try:
            names = self.investable_tickers()
        except (OSError, KeyError, ValueError):
            return "unreadable"
        return hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]

    def fingerprint(self) -> str:
        """Stable hash of the configuration (used to count distinct trials for the deflated Sharpe).

        ``runs_dir`` is left out: where results are written is not a strategy choice, and
        the same config must not count as a new trial because its path was resolved or moved.
        ``evaluation`` is left out too: it changes what a backtest report measures, so
        re-scoring the same strategy with another null or cost grid is not a new trial (and
        configs written before the section existed keep their fingerprint). The one
        exception is ``evaluation.cutoff_buffer_days`` when it differs from
        :data:`DEFAULT_CUTOFF_BUFFER_DAYS`: it also sets which real-LLM forecasts are
        flagged ``pre_cutoff`` and therefore which track record believability, calibration
        and the overlay bound may use, so it can change a decision.

        With ``universe.screen`` enabled the screen's *sources* (``preset``, ``file``,
        ``tickers``) are replaced by the resolved universe (:meth:`investable_tickers`): two
        runs over different tickers listed in one file are different trials, and the same
        tickers reached through another path, a relative or an absolute file name, or a
        preset are the same trial. If the universe cannot be read, the raw sources are
        hashed instead (the run itself then fails loudly). Prompt text is code, not
        configuration: editing a prompt does not change the fingerprint (change the config's
        ``name`` to count a prompt edit as a new trial)."""
        dump = self.model_dump(exclude={"runs_dir", "evaluation"})
        if not dump["universe"]["ciks"]:      # added later: an unused pin must not change existing fingerprints
            del dump["universe"]["ciks"]
        if self.evaluation.cutoff_buffer_days != DEFAULT_CUTOFF_BUFFER_DAYS:
            dump["cutoff_buffer_days"] = self.evaluation.cutoff_buffer_days
        scr = dump["universe"]["screen"]
        if scr["enabled"]:
            try:
                resolved = self.investable_tickers()
            except (OSError, KeyError, ValueError):
                resolved = None
            if resolved is not None:
                for k in ("preset", "file", "tickers"):
                    scr.pop(k, None)
                scr["resolved_universe"] = resolved
        blob = json.dumps(dump, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def model_cutoff(model: str) -> str | None:
    return MODEL_TRAINING_CUTOFFS.get(model)
