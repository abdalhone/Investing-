"""Memory and learning layer: the audit ledger, the forecast track record and lessons.

* :mod:`.ledger`: append-only, hash-chained JSONL record of every decision (P10).
* :mod:`.forecasts`: every view becomes a resolvable forecast; Brier-scored track
  records set believability weights, strictly point-in-time (P7, P3).
* :mod:`.lessons`: process lessons retrievable only once their forecast has resolved (P3).
* :mod:`.calibration`: per-agent Platt calibration and rank IC from resolved forecasts,
  strictly point-in-time.
"""

from .calibration import (Calibrator, PairedIC, RankIC, calibrate_view, fit_calibrators, paired_rank_ic,
                          resolved_rank_ic)
from .forecasts import (COMMITTEE_AGENT, LLM_POOL_AGENT, POOLED_AGENTS, SHADOW_AGENT, ForecastStore,
                        believability_weights, is_pre_cutoff, make_forecast_records, question_key, resolve_date,
                        scorecard)
from .ledger import GENESIS_HASH, Ledger, LedgerAudit, LedgerError, canonical_json, to_jsonable
from .lessons import LessonStore

__all__ = [
    "COMMITTEE_AGENT",
    "LLM_POOL_AGENT",
    "POOLED_AGENTS",
    "SHADOW_AGENT",
    "question_key",
    "Calibrator",
    "PairedIC",
    "RankIC",
    "paired_rank_ic",
    "calibrate_view",
    "fit_calibrators",
    "resolved_rank_ic",
    "GENESIS_HASH",
    "ForecastStore",
    "Ledger",
    "LedgerAudit",
    "LedgerError",
    "LessonStore",
    "believability_weights",
    "canonical_json",
    "is_pre_cutoff",
    "make_forecast_records",
    "resolve_date",
    "scorecard",
    "to_jsonable",
]
