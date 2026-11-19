"""Deterministic cross-sectional factor model: the committee's "quant" analyst.

Why a code analyst sits on an LLM committee
-------------------------------------------
* **A different slice of information (P5).** Every LLM analyst sees one stock at a time.
  The quant sees the whole universe on the same date and reports where this stock
  *ranks* on four well-documented equity factors. That cross-sectional view is
  information the others structurally lack, which is what makes pooling useful.
* **An honest baseline (P10).** It is cheap, reproducible, cannot hallucinate and has no
  training cutoff, so it is the "quant-only" baseline in backtests and a sanity anchor
  for the pool. If the LLM committee cannot beat it, the LLMs are not adding value.
* **Deliberately modest (P8).** Factor premia are real but small at a one-month horizon,
  so the probability mapping ``p = 0.5 + 0.06 * tanh(composite)`` never leaves
  [0.44, 0.56]. The quant tilts the pool; it cannot dominate it.

Method (ARCHITECTURE.md 3.7)
----------------------------
* Factors: momentum (``mom_12_1``), low volatility (``-vol_60d``), quality (``roe``,
  ``operating_income_margin``) and value (``earnings_yield``, ``fcf_yield``).
* Each input is turned into a cross-sectional z-score (population std-dev) and winsorized
  at +/-3, so one extreme name cannot dominate. A two-input factor averages the z-scores
  of the inputs a name has, then is z-scored and winsorized again, so every factor has
  the same scale and therefore the same say in the composite.
* Missing data scores 0 (neutral) and is *excluded* from the composite: composite = mean
  of the factor z-scores the name actually has. A z-score needs at least three names with
  data and some dispersion; otherwise the factor is unavailable for everyone.
* Expected excess return = ``sigma_h * Phi^-1(p)`` with ``sigma_h`` from ``vol_60d``
  scaled to the horizon. Scenarios are the 3-point Gauss-Hermite discretisation of a
  normal with that mean and ``sigma_h`` (nodes ``mu`` and ``mu +/- sqrt(3) sigma_h`` with
  weights 2/3 and 1/6 each), so their probability-weighted mean equals the expected excess
  exactly and the view is internally consistent.
* A name with no scorable factor abstains (fail closed: no data, no opinion). With fewer
  than ``MIN_NAMES`` names in the universe nothing can be ranked, so the quant abstains on
  every name and says so in ``abstain_reason`` (a quant-only firm then never trades; the
  pipeline surfaces that reason as a warning rather than reporting an all-cash book
  silently).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..data.factsheet import FactSheet
from ..schemas import AnalystView, EvidenceItem, Scenario, ViewRecord, sanitize_view

QUANT_AGENT = "quant"
QUANT_MODEL = "deterministic"
COMPOSITE = "composite"

# factor -> (fact key, sign) inputs; the sign makes "higher is better" for every input.
FACTORS: dict[str, tuple[tuple[str, float], ...]] = {
    "momentum": (("mom_12_1", 1.0),),
    "low_vol": (("vol_60d", -1.0),),
    "quality": (("roe", 1.0), ("operating_income_margin", 1.0)),
    "value": (("earnings_yield", 1.0), ("fcf_yield", 1.0)),
}
FACTOR_LABELS = {
    "momentum": "Momentum (12-1 month return)",
    "low_vol": "Low volatility (60-day realized volatility, lower is better)",
    "quality": "Quality (return on equity, operating margin)",
    "value": "Value (earnings yield, free-cash-flow yield)",
}
WINSOR_Z = 3.0
MIN_NAMES = 3              # fewer names and a "z-score" describes a pair, not a cross-section
PROB_SPREAD = 0.06         # max deviation of p from 0.5 (contract)
NEUTRAL_Z = 0.25           # within a quarter std-dev of the universe mean: not worth arguing about
FALLBACK_VOL_PCT = 30.0    # same assumption as the portfolio engine when vol_60d is missing
TRADING_DAYS = 252
_GH_SPREAD = math.sqrt(3.0)
_GH_WEIGHTS = {"bull": 1 / 6, "base": 2 / 3, "bear": 1 / 6}


@dataclass(frozen=True)
class _Scores:
    z: dict[str, float]                       # factor -> winsorized z (0.0 when unavailable)
    available: tuple[str, ...]                # factors this name has, in FACTORS order
    fact_ids: dict[str, tuple[str, ...]]      # factor -> fact IDs that fed it
    names: dict[str, int]                     # factor -> names in the cross-section with data
    composite: float | None                   # None when no factor is available


def composite_probability(composite: float) -> float:
    """``0.5 + 0.06 * tanh(composite)``: strictly increasing, always inside (0.44, 0.56)."""
    return 0.5 + PROB_SPREAD * math.tanh(composite)


def _metric(fs: FactSheet, key: str) -> float | None:
    """A finite numeric fact for this ticker, else ``None`` (missing)."""
    fact = fs.get(f"{fs.ticker}.{key}")
    if fact is None or isinstance(fact.value, bool) or not isinstance(fact.value, numbers.Real):
        return None
    v = float(fact.value)
    if not math.isfinite(v) or (key == "vol_60d" and v <= 0):
        return None
    return v


def _zscores(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-scores winsorized at +/-``WINSOR_Z``. Empty when fewer than
    ``MIN_NAMES`` names or no dispersion. Computed in ticker order, so the result does not
    depend on the order the fact sheets were passed in."""
    if len(values) < MIN_NAMES:
        return {}
    names = sorted(values)
    x = np.array([values[t] for t in names])
    mu, sd = float(x.mean()), float(x.std())
    if not sd > 1e-12 * max(1.0, abs(mu)):
        return {}
    return {t: float(np.clip((values[t] - mu) / sd, -WINSOR_Z, WINSOR_Z)) for t in names}


def _score_universe(factsheets: Sequence[FactSheet]) -> dict[str, _Scores]:
    tickers = [fs.ticker for fs in factsheets]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"duplicate tickers in the cross-section: {tickers}")
    by_ticker = {fs.ticker: fs for fs in factsheets}
    factor_z: dict[str, dict[str, float]] = {}
    factor_ids: dict[str, dict[str, tuple[str, ...]]] = {}
    for factor, inputs in FACTORS.items():
        input_z = {}
        for key, sign in inputs:
            raw = {t: sign * v for t in tickers if (v := _metric(by_ticker[t], key)) is not None}
            # a single-input factor is z-scored once below; z-scoring it here too would be redundant
            input_z[key] = raw if len(inputs) == 1 else _zscores(raw)
        combined, ids = {}, {}
        for t in tickers:
            have = [key for key, _ in inputs if t in input_z[key]]
            if have:
                combined[t] = sum(input_z[key][t] for key in have) / len(have)
                ids[t] = tuple(f"{t}.{key}" for key in have)
        factor_z[factor] = _zscores(combined)
        factor_ids[factor] = ids
    out = {}
    for t in tickers:
        available = tuple(f for f in FACTORS if t in factor_z[f])
        out[t] = _Scores(
            z={f: factor_z[f].get(t, 0.0) for f in FACTORS},
            available=available,
            fact_ids={f: factor_ids[f][t] for f in available},
            names={f: len(factor_z[f]) for f in FACTORS},
            composite=sum(factor_z[f][t] for f in available) / len(available) if available else None,
        )
    return out


def factor_scores(factsheets: list[FactSheet]) -> dict[str, dict[str, float]]:
    """Per ticker: winsorized cross-sectional z-score per factor (0.0 when missing) and the
    ``composite`` (mean of the available factor z-scores; 0.0 when none is available)."""
    return {t: {**s.z, COMPOSITE: s.composite if s.composite is not None else 0.0}
            for t, s in _score_universe(factsheets).items()}


def _annual_vol_pct(fs: FactSheet) -> tuple[float, bool]:
    """(annualized vol in %, True if it is the fallback)."""
    v = _metric(fs, "vol_60d")
    return (v, False) if v is not None else (FALLBACK_VOL_PCT, True)


def horizon_sigma_pct(fs: FactSheet, horizon_days: int) -> float:
    """The stock's volatility over ``horizon_days`` in percent, as the quant uses it:
    ``vol_60d`` (or the 30% fallback) times ``sqrt(horizon_days / 252)``. The overlay uses
    the same scale to turn a final probability into an expected excess return."""
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    return _annual_vol_pct(fs)[0] * math.sqrt(horizon_days / TRADING_DAYS)


def _direction(z: float) -> str:
    if z > NEUTRAL_Z:
        return "positive"
    if z < -NEUTRAL_Z:
        return "negative"
    return "neutral"


def _abstain(ticker: str, n_universe: int) -> AnalystView:
    if n_universe < MIN_NAMES:
        reason = (f"The universe has only {n_universe} name(s) with a fact sheet; ranking a cross-section needs "
                  f"at least {MIN_NAMES}, so the quant abstains on every name (add tickers to use it).")
    else:
        reason = (f"No factor could be scored: each factor needs data for at least {MIN_NAMES} names "
                  "with some dispersion, and this name had none.")
    return AnalystView(
        ticker=ticker, abstain=True,
        abstain_reason=reason,
        prob_outperform=0.5, expected_excess_return_pct=0.0, scenarios=[], evidence=[], key_risks=[],
        what_would_change_my_mind="", private_insight="")


def _view(fs: FactSheet, s: _Scores, horizon_days: int, n_universe: int) -> AnalystView:
    if s.composite is None:
        return _abstain(fs.ticker, n_universe)
    p = composite_probability(s.composite)
    vol_fallback = _annual_vol_pct(fs)[1]
    sigma_h_pct = horizon_sigma_pct(fs, horizon_days)
    excess = sigma_h_pct * float(norm.ppf(p))
    spread = _GH_SPREAD * sigma_h_pct
    scenarios = [
        Scenario(name="bull", probability=_GH_WEIGHTS["bull"], excess_return_pct=excess + spread,
                 description="Stock-specific noise breaks favourably: about 1.7 horizon standard deviations above the factor expectation."),
        Scenario(name="base", probability=_GH_WEIGHTS["base"], excess_return_pct=excess,
                 description="The factor tilt delivers its average effect relative to the benchmark."),
        Scenario(name="bear", probability=_GH_WEIGHTS["bear"], excess_return_pct=excess - spread,
                 description="Stock-specific noise breaks unfavourably by the same amount."),
    ]
    evidence = [
        EvidenceItem(point=(f"{FACTOR_LABELS[f]}: {s.z[f]:+.2f} standard deviations versus "
                            f"the {s.names[f]} names in the universe with data."),
                     fact_ids=list(s.fact_ids[f]), direction=_direction(s.z[f]))
        for f in s.available
    ]
    risks = [
        "Factor premia are small and regime-dependent; over one horizon most of the return is stock-specific noise.",
        "Ranks are relative to this universe: if the whole universe lags the benchmark, a top-ranked name can still lag.",
        "Factor crowding and sharp momentum reversals can turn the tilt negative quickly.",
    ]
    missing = [f for f in FACTORS if f not in s.available]
    if missing:
        risks.append(f"Not scored for lack of data (counted as neutral): {', '.join(missing)}.")
    if vol_fallback:
        risks.append(f"60-day volatility unavailable: the expected excess return assumes {FALLBACK_VOL_PCT:.0f}% annual volatility.")
    return AnalystView(
        ticker=fs.ticker, abstain=False, abstain_reason="", prob_outperform=p, expected_excess_return_pct=excess,
        scenarios=scenarios, evidence=evidence, key_risks=risks,
        what_would_change_my_mind=(f"Inputs that move the composite factor score (now {s.composite:+.2f}) across zero, "
                                   "such as a momentum reversal, a volatility spike or a new filing that changes "
                                   "the quality or value ranks."),
        private_insight=("Where this stock ranks against the rest of the universe on the same date on momentum, "
                         "low volatility, quality and value; analysts who read one stock's facts in isolation "
                         "cannot see relative standing."))


def quant_views(factsheets: list[FactSheet], horizon_days: int) -> dict[str, ViewRecord]:
    """The quant analyst's view for every ticker (``agent="quant"``, ``model="deterministic"``).

    Pass the whole investable universe for one date: the scores are relative to it.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    scores = _score_universe(factsheets)
    return {fs.ticker: ViewRecord(agent=QUANT_AGENT, model=QUANT_MODEL, sample=0,
                                  view=sanitize_view(_view(fs, scores[fs.ticker], horizon_days, len(factsheets))))
            for fs in factsheets}
