"""Firm configuration. Load from a TOML file (see ``examples/firm.toml``) or build in code.

Every default below is a deliberate, research-backed choice; see docs/ARCHITECTURE.md
for the reasoning and docs/research/RESEARCH_REPORT.md for the evidence.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

# Training-data cutoffs per model. Forecasts made on dates BEFORE a model's cutoff may be
# contaminated by memorized outcomes, so backtests report those periods separately and
# never count them as evidence of skill. If a model is missing here, the cutoff is
# assumed to be "today" (i.e. every historical date is treated as contaminated).
# Verify these against Anthropic's models overview page when you change models.
MODEL_TRAINING_CUTOFFS: dict[str, str] = {
    "claude-opus-5-5": "2026-06-30",
}


class ModelConfig(BaseModel):
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


class UniverseConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "PG"])
    benchmark: str = "SPY"
    sectors: dict[str, str] = Field(default_factory=dict, description="Optional manual sector labels (ticker -> sector) used for sector limits. Defaults to SEC SIC descriptions.")


class AnalystsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["fundamental", "technical", "macro", "events", "quant"],
                               description="Which analysts sit on the committee. 'quant' is deterministic (no LLM).")
    max_lessons_in_prompt: int = 5


class ProcessConfig(BaseModel):
    horizon_days: int = Field(21, description="Forecast horizon in trading days (21 ~ 1 month).")
    rebalance_every_days: int = Field(21, description="Trading days between decision cycles in backtests.")
    challenge_enabled: bool = True
    challenge_dispersion_threshold: float = Field(0.12, description="Std-dev of analyst probabilities above which a red-team challenge round runs.")
    trim_fraction: float = Field(0.2, description="Fraction trimmed from each tail when pooling probabilities in log-odds space.")
    memo_enabled: bool = True
    memo_min_edge: float = Field(0.03, description="Write an IC memo only when |pooled prob - 0.5| >= this, or the name is currently held.")
    risk_review_enabled: bool = True
    blind_mode: bool = Field(False, description="Hide ticker, company name and absolute dates from LLM analysts (for historical research: reduces memorization leakage).")
    news_enabled: bool = True
    max_llm_cost_usd_per_run: float = 10.0


class PortfolioConfig(BaseModel):
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


class RiskConfig(BaseModel):
    drawdown_derisk: float = Field(0.10, description="At this drawdown from peak NAV, gross exposure of the active sleeve is halved.")
    drawdown_freeze: float = Field(0.15, description="At this drawdown, the active sleeve goes to cash/core until a human resets it.")
    daily_loss_halt: float = Field(0.05, description="A one-day NAV loss this large halts new buys for the next cycle.")
    max_adv_participation: float = Field(0.02, description="Max order size as a fraction of 20-day average dollar volume.")
    restricted: list[str] = Field(default_factory=list)
    kill_switch_file: str = Field("KILL", description="If this file exists in the runs directory, no orders are generated.")
    min_price: float = 2.0


class CostConfig(BaseModel):
    commission_bps: float = 0.5
    slippage_bps: float = 5.0
    impact_coefficient: float = Field(0.1, description="Square-root market impact: cost_bps += coef * 1e4 * sqrt(order / ADV) * daily_vol.")


class ExecutionConfig(BaseModel):
    broker: str = Field("paper", description="'paper' (simulated) or 'alpaca' (Alpaca paper/live API).")
    require_approval: bool = Field(True, description="Ask a human to approve orders before sending them to a real broker.")
    initial_cash: float = 100_000.0


class FirmConfig(BaseModel):
    name: str = "AI Investment Firm"
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    analysts: AnalystsConfig = Field(default_factory=AnalystsConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    runs_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path) -> "FirmConfig":
        with open(path, "rb") as fh:
            return cls.model_validate(tomllib.load(fh))

    def fingerprint(self) -> str:
        """Stable hash of the configuration (used to count distinct trials for the deflated Sharpe)."""
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def model_cutoff(model: str) -> str | None:
    return MODEL_TRAINING_CUTOFFS.get(model)
