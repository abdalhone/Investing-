"""Deterministic money core: pooling (aggregate), sizing (construct) and the invariant risk core (risk_engine).

LLMs propose; this package disposes (P1). Nothing here calls a model.
"""

from .aggregate import aggregate_views, pool_probabilities
from .construct import construct_portfolio
from .overlay import OverlayBound, adaptive_bound, combine, overlay_aggregate
from .risk_engine import RiskState, apply_risk_limits, orders_from_target, update_risk_state

__all__ = [
    "OverlayBound",
    "RiskState",
    "adaptive_bound",
    "combine",
    "overlay_aggregate",
    "aggregate_views",
    "apply_risk_limits",
    "construct_portfolio",
    "orders_from_target",
    "pool_probabilities",
    "update_risk_state",
]
