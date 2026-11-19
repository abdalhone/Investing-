"""The LLM agent layer: analysts, grounding, committee, risk review, reflection, offline responders.

Agents propose; code disposes (P1). Every agent here outputs probabilities, evidence,
multipliers that can only shrink, or lessons. None outputs a weight, a share count or an
order. See docs/ARCHITECTURE.md sections 3.2-3.6.
"""

from .analysts import (analyst_system_prompt, analyst_user_prompt, blind_factsheet, role_slice,
                       run_analyst)
from .committee import CommitteeResult, llm_analysts, run_committee
from .grounding import GroundingReport, check_view
from .news_reader import NewsReadResult, read_headlines
from .offline import make_offline_llm
from .reflection import write_lesson
from .reid import ReidReport, run_reid_probe
from .risk_review import apply_risk_review, review_portfolio
from .roles import ROLES, AnalystRole
from .tier_gate import apply_tier_gate, tier_gate

__all__ = [
    "ROLES",
    "AnalystRole",
    "CommitteeResult",
    "GroundingReport",
    "NewsReadResult",
    "ReidReport",
    "analyst_system_prompt",
    "analyst_user_prompt",
    "apply_risk_review",
    "apply_tier_gate",
    "blind_factsheet",
    "check_view",
    "llm_analysts",
    "make_offline_llm",
    "read_headlines",
    "review_portfolio",
    "role_slice",
    "run_analyst",
    "run_committee",
    "run_reid_probe",
    "tier_gate",
    "write_lesson",
]
