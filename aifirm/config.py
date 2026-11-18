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


class UniverseConfig(_Section):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "PG"])
    benchmark: str = "SPY"
    sectors: dict[str, str] = Field(default_factory=dict, description="Optional manual sector labels (ticker -> sector) used for sector limits. Defaults to SEC SIC descriptions.")
    ciks: dict[str, str | int] = Field(default_factory=dict, description="Optional SEC filer pins (ticker -> CIK) used for that ticker's fundamentals and filing events on every date. Use when SEC's current ticker table maps a ticker to the wrong filer for your dates (e.g. after a holding-company reorganization).")

    def _cross_checks(self) -> list[str]:
        bench = self.benchmark.strip().upper()
        if not bench:
            return ["benchmark must not be empty"]
        if not any(t.strip() and t.strip().upper() != bench for t in self.tickers):
            return ["tickers must list at least one stock other than the benchmark"]
        bad = [f"{t} = {c!r}" for t, c in self.ciks.items()
               if isinstance(c, bool) or not str(c).strip().isdigit() or len(str(c).strip()) > 10]
        if bad:
            return [f"ciks must map tickers to SEC CIK numbers (up to 10 digits): {', '.join(bad)}"]
        return []

    def cik_pins(self) -> dict[str, str]:
        """``ciks`` normalized: upper-case tickers -> 10-digit CIK strings."""
        return {t.strip().upper(): str(c).strip().zfill(10) for t, c in self.ciks.items()}


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

    _RANGES: ClassVar[dict[str, _Range]] = {
        "horizon_days": _Range(1, 252),                 # trading days: one day to one year
        "rebalance_every_days": _Range(1, 252),
        "challenge_dispersion_threshold": _fraction(0.0, 0.5),   # std-dev of probabilities is at most 0.5
        "trim_fraction": _fraction(0.0, 0.5, hi_open=True),
        "memo_min_edge": _fraction(0.0, 0.5),
        "max_llm_cost_usd_per_run": _Range(0.0, lo_open=True),
    }


class PortfolioConfig(_Section):
    core_weight: float = Field(0.0, description="Fraction of capital held passively in the benchmark (core-satellite). The agents manage the rest.")
    long_only: bool = True
    max_position: float = 0.10
    max_sector: float = 0.35
    min_cash: float = 0.02
    max_gross: float = 1.0
    min_prob_to_hold: float = Field(0.52, description="Minimum pooled P(outperform) to hold a long.")
    kelly_fraction: float = Field(0.25, description="Fraction of the Kelly-optimal weight to use (full Kelly is far too aggressive with estimated edges).")
    target_vol: float = Field(0.18, description="Annualized volatility target for the active sleeve.")
    no_trade_band: float = Field(0.01, description="Skip trades that change a weight by less than this.")
    max_turnover: float = Field(0.5, description="Max one-way turnover per rebalance (fraction of NAV).")
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
        "dispersion_shrink": _Range(0.0),
    }


class RiskConfig(_Section):
    drawdown_derisk: float = Field(0.10, description="At this drawdown from peak NAV, gross exposure of the active sleeve is halved.")
    drawdown_freeze: float = Field(0.15, description="At this drawdown, the active sleeve goes to cash/core until a human resets it.")
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
    runs_dir: str = Field("runs", description="Book and backtest directory. Relative paths are relative to the "
                                              "config file's directory (see FirmConfig.runs_path).")

    # Directory of the TOML file this config was loaded from (None when built in code).
    # A private attribute: not part of model_dump(), so it never changes the fingerprint.
    _source_dir: Path | None = PrivateAttr(default=None)

    def _cross_checks(self) -> list[str]:
        return [] if self.runs_dir.strip() else ["runs_dir must not be empty"]

    @classmethod
    def load(cls, path: str | Path) -> "FirmConfig":
        """Load and validate a TOML file. Raises ``FileNotFoundError``, ``tomllib.TOMLDecodeError``
        (a ``ValueError``) or ``pydantic.ValidationError`` (unknown keys, values out of range)."""
        path = Path(path).expanduser()
        with open(path, "rb") as fh:
            cfg = cls.model_validate(tomllib.load(fh))
        cfg._source_dir = path.resolve().parent
        return cfg

    def runs_path(self) -> Path:
        """Absolute ``runs_dir``. A relative value is anchored at the directory of the config
        file it was loaded from, so the book, the trial ledger and the kill switch do not
        depend on the directory a command runs from; without a file (a config built in
        code), at the current working directory."""
        p = Path(self.runs_dir).expanduser()
        if not p.is_absolute():
            p = (self._source_dir if self._source_dir is not None else Path.cwd()) / p
        return p.resolve()

    def fingerprint(self) -> str:
        """Stable hash of the configuration (used to count distinct trials for the deflated Sharpe).

        ``runs_dir`` is left out: where results are written is not a strategy choice, and
        the same config must not count as a new trial because its path was resolved or moved."""
        dump = self.model_dump(exclude={"runs_dir"})
        if not dump["universe"]["ciks"]:      # added later: an unused pin must not change existing fingerprints
            del dump["universe"]["ciks"]
        blob = json.dumps(dump, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def model_cutoff(model: str) -> str | None:
    return MODEL_TRAINING_CUTOFFS.get(model)
