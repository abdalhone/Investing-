"""The deterministic quant analyst: a cross-sectional factor model that votes in the
committee like any analyst but cannot hallucinate and has no training cutoff (P5, P10).
See docs/ARCHITECTURE.md section 3.7."""

from .screen import ScreenEntry, ScreenResult, screen_universe
from .signals import (COMPOSITE, FACTORS, QUANT_AGENT, QUANT_MODEL, composite_probability, factor_scores,
                      horizon_sigma_pct, quant_views)

__all__ = [
    "COMPOSITE",
    "FACTORS",
    "QUANT_AGENT",
    "QUANT_MODEL",
    "ScreenEntry",
    "ScreenResult",
    "composite_probability",
    "factor_scores",
    "horizon_sigma_pct",
    "quant_views",
    "screen_universe",
]
