"""Deterministic portfolio construction: pooled forecasts in, target weights out (P1, P8).

LLMs never size positions (P1). This module turns each ticker's pooled probability and
expected excess return into a weight with a fixed, auditable recipe:

1. **Edge.** ``mu_prob = sigma_h * Phi^-1(p)`` is the excess return a normal distribution
   with the stock's horizon volatility implies for ``P(beat benchmark) = p``.
   ``mu_exp = expected_excess / 100`` is the analysts' own number. The two are averaged
   because each guards against a different failure: probabilities are scored and
   calibrated (P7), while point forecasts carry magnitude. The edge is then shrunk by the
   chair's ``edge_multiplier`` (which can only shrink it) and by analyst disagreement,
   ``max(0, 1 - dispersion_shrink * dispersion)``. Disagreement is information about
   uncertainty, so it costs size.
2. **Fractional Kelly.** ``w = kelly_fraction * mu / sigma_h^2``. Full Kelly maximizes
   growth only when the edge is known exactly; with estimated edges it is ruinous, hence
   the default quarter Kelly (P8).
3. **Caps and targets.** Per-name cap, sector caps, then sleeve volatility targeting
   (sample covariance of the last 126 days when returns are given, else a
   constant-correlation model with rho = 0.3).
4. **Core-satellite.** ``core_weight`` of NAV sits passively in the benchmark. The agents
   manage the rest, so a bad committee can only lose relative to part of the book.
5. **Book limits** (gross and cash floor), stated as fractions of total NAV.
6. **Trade path.** A no-trade band and a turnover cap avoid paying costs for noise.

Cost controls yield to risk limits: if the band or the turnover cap would leave the book
breaching a hard limit (position cap, sector cap, gross, cash floor, volatility target,
long-only), the exposure-adding part of each trade is trimmed first; if that is not
enough (the current book itself is out of bounds, e.g. after price drift), the book moves
toward the ideal target just far enough to comply and the turnover cap is overridden with
a note. The final book therefore always satisfies every hard limit. In a long-only book
whose current weights are within limits, the turnover cap is also respected (strictly
guaranteed when no pairwise correlation is negative, as in the fallback model; selling a
negatively correlated hedge can raise volatility). In a long/short book, covering a short
raises net exposure, so in rare cases a limit can still override the turnover cap.

Every binding constraint is recorded per position (``constraints_hit``) or in the
portfolio ``notes``, so a report can explain each weight.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import FirmConfig, PortfolioConfig
from ..data.factsheet import FactSheet
from ..schemas import AggregatedView, TargetPortfolio, TargetPosition, clamp

TRADING_DAYS = 252
FALLBACK_VOL = 0.30     # annualized vol assumed when vol_60d is unavailable (contract)
MIN_VOL = 0.05          # a single stock below 5% realized vol almost always means stale prices
FALLBACK_CORR = 0.30    # average pairwise correlation of US large caps, used without returns
COV_WINDOW = 126        # ~6 months of daily returns for the sample covariance
MIN_COV_OBS = 60        # fewer complete rows than this and the sample covariance is too noisy
PROB_CLIP = 0.01        # keeps Phi^-1(p) finite
TOL = 1e-12             # slack for limit checks: above float error, far below any meaningful weight
DUST = 1e-12            # weights smaller than this are floating-point noise


# --------------------------------------------------------------------------- helpers


def _finite_or(x: object, default: float) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _validate(pc: PortfolioConfig) -> None:
    """Reject nonsensical limits up front: a misconfigured book must not trade (fail closed)."""
    problems = []
    if not 0.0 <= pc.min_cash < 1.0:
        problems.append(f"min_cash={pc.min_cash} must be in [0, 1)")
    if not 0.0 <= pc.core_weight <= 1.0:
        problems.append(f"core_weight={pc.core_weight} must be in [0, 1]")
    for name in ("max_position", "max_sector", "kelly_fraction", "no_trade_band", "max_turnover", "dispersion_shrink"):
        if not getattr(pc, name) >= 0:
            problems.append(f"{name}={getattr(pc, name)} must be >= 0")
    for name in ("max_gross", "target_vol"):
        if not getattr(pc, name) > 0:
            problems.append(f"{name}={getattr(pc, name)} must be > 0")
    if problems:
        raise ValueError("invalid portfolio config: " + "; ".join(problems))


def _clean_current(current_weights: Mapping[str, float]) -> dict[str, float]:
    """Current weights must be real numbers. A NaN here means the account state is broken,
    and guessing 0 could make us buy more of something we already hold, so we raise."""
    out = {}
    for t, w in current_weights.items():
        v = _finite_or(w, math.nan)
        if math.isnan(v):
            raise ValueError(f"current weight for {t} is not a finite number: {w!r}")
        if v != 0.0:
            out[t] = v
    return out


def _annual_vol(fs: FactSheet | None, ticker: str) -> tuple[float, str]:
    fact = fs.get(f"{ticker}.vol_60d") if fs is not None else None
    v = _finite_or(fact.value, math.nan) / 100.0 if fact is not None else math.nan
    if not math.isfinite(v) or v <= 0:
        return FALLBACK_VOL, "vol_60d unavailable, fallback 30%"
    if v < MIN_VOL:
        return MIN_VOL, f"vol_60d {v:.1%} floored at {MIN_VOL:.0%}"
    return v, "vol_60d"


@dataclass(frozen=True)
class _Cov:
    """Annualized covariance over a fixed ticker list; weights outside the list count as 0."""

    tickers: tuple[str, ...]
    matrix: np.ndarray
    source: str

    def vol(self, weights: Mapping[str, float]) -> float:
        x = np.array([weights.get(t, 0.0) for t in self.tickers], dtype=float)
        return math.sqrt(max(0.0, float(x @ self.matrix @ x))) if x.size else 0.0


def _covariance(names: list[str], sigmas: Mapping[str, float], returns: pd.DataFrame | None,
                as_of: str) -> _Cov:
    """Sample covariance of the last ``COV_WINDOW`` rows of ``returns`` (sliced to ``as_of``,
    P3), or the constant-correlation model when returns are absent or too sparse."""
    if returns is not None and names:
        r = returns.loc[: pd.Timestamp(as_of)] if isinstance(returns.index, pd.DatetimeIndex) else returns
        window = r.reindex(columns=names).tail(COV_WINDOW).dropna()
        if len(window) >= MIN_COV_OBS:
            m = window.cov().to_numpy() * TRADING_DAYS
            if np.isfinite(m).all():
                return _Cov(tuple(names), m, f"sample covariance of {len(window)} daily returns")
        fallback_reason = f"only {len(window)} complete return rows for {len(names)} names; "
    else:
        fallback_reason = ""
    s = np.array([sigmas[t] for t in names], dtype=float)
    m = FALLBACK_CORR * np.outer(s, s)
    np.fill_diagonal(m, s ** 2)
    return _Cov(tuple(names), m, f"{fallback_reason}constant-correlation model (rho={FALLBACK_CORR}, vol_60d)")


@dataclass(frozen=True)
class _Limits:
    """Hard limits on the final book, as fractions of total NAV (the core leg is ``bench``)."""

    bench: str
    long_only: bool
    max_position: float
    max_sector: float
    max_gross: float
    min_cash: float
    vol_cap: float                  # target_vol * (1 - core): the sleeve target in NAV terms
    sector_of: Mapping[str, str]    # classified tickers only
    cov: _Cov

    def violations(self, book: Mapping[str, float]) -> list[str]:
        out = []
        active = {t: w for t, w in book.items() if t != self.bench}
        if self.long_only and any(w < -TOL for w in book.values()):
            out.append("short position in a long-only book")
        over = sorted(t for t, w in active.items() if abs(w) > self.max_position + TOL)
        if over:
            out.append(f"position cap ({', '.join(over)})")
        sector_gross: dict[str, float] = defaultdict(float)
        for t, w in active.items():
            if self.sector_of.get(t):
                sector_gross[self.sector_of[t]] += abs(w)
        over_s = sorted(s for s, g in sector_gross.items() if g > self.max_sector + TOL)
        if over_s:
            out.append(f"sector cap ({', '.join(over_s)})")
        if sum(abs(w) for w in book.values()) > self.max_gross + TOL:
            out.append("max gross")
        if 1.0 - sum(book.values()) < self.min_cash - TOL:
            out.append("cash floor")
        if self.cov.vol(active) > self.vol_cap + TOL:
            out.append("volatility target")
        return out


def _toward_zero_part(current: float, delta: float) -> float:
    """The part of trade ``delta`` that shrinks ``current`` toward zero (never past it)."""
    if current * delta < 0:
        return math.copysign(min(abs(delta), abs(current)), delta)
    return 0.0


def _largest_feasible(ok_at: Callable[[float], bool]) -> float:
    """Largest u in [0, 1] with ``ok_at(u)``, given ``ok_at(0)`` and a convex feasible set
    (every limit here is convex in the weights, so feasibility along a segment is an interval)."""
    if ok_at(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if ok_at(mid):
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- steps


@dataclass
class _Sized:
    weight: float
    sigma_ann: float
    rationale: str


def _size_name(t: str, agg: AggregatedView, multiplier: object, fs: FactSheet | None,
               cfg: FirmConfig, hits: list[str]) -> _Sized:
    """Steps 1-2 for one ticker: edge, shrinkage, gating and capped fractional Kelly."""
    pc = cfg.portfolio
    sigma_ann, vol_src = _annual_vol(fs, t)
    sigma_h = sigma_ann * math.sqrt(cfg.process.horizon_days / TRADING_DAYS)
    p = clamp(agg.prob_outperform, 0.0, 1.0)
    mu_prob = sigma_h * float(norm.ppf(min(max(p, PROB_CLIP), 1.0 - PROB_CLIP)))
    mu_exp = _finite_or(agg.expected_excess_return_pct, 0.0) / 100.0
    # The chair can only shrink the edge; a missing or broken multiplier is treated as 0 (no trade).
    m = clamp(_finite_or(multiplier, 0.0), 0.0, 1.0)
    disp = _finite_or(agg.dispersion, math.nan)
    disp_factor = max(0.0, 1.0 - pc.dispersion_shrink * max(0.0, disp)) if math.isfinite(disp) else 0.0
    mu = 0.5 * (mu_prob + mu_exp) * m * disp_factor

    long_ok = p >= pc.min_prob_to_hold
    short_ok = (not pc.long_only) and p <= 1.0 - pc.min_prob_to_hold
    weight = 0.0
    if (long_ok and mu > 0) or (short_ok and mu < 0):
        raw = pc.kelly_fraction * mu / sigma_h ** 2
        weight = max(-pc.max_position, min(pc.max_position, raw))
        if weight != raw:
            hits.append("max_position")
        sizing = f"Kelly {pc.kelly_fraction:g} x mu / sigma_h^2 = {raw:+.4f}"
    elif long_ok or short_ok:
        hits.append("no_edge")
        sizing = "probability passes the hold threshold but the shrunk edge has the wrong sign: no position"
    else:
        hits.append("min_prob_to_hold")
        sizing = f"p below the hold threshold {pc.min_prob_to_hold:g}: no position"
    rationale = (
        f"p={p:.3f} (equal-weight {agg.prob_equal_weight:.3f}, {agg.n_views} views, dispersion {agg.dispersion:.3f}); "
        f"E[excess]={agg.expected_excess_return_pct:+.2f}%; sigma_ann={sigma_ann:.1%} ({vol_src}); "
        f"mu_prob={mu_prob:+.4f}, mu_exp={mu_exp:+.4f}, edge x{m:.2f}, dispersion x{disp_factor:.2f} "
        f"-> mu={mu:+.4f}; {sizing}"
    )
    return _Sized(weight=weight, sigma_ann=sigma_ann, rationale=rationale)


def _apply_sector_caps(sleeve: dict[str, float], sector_of: Mapping[str, str], cap: float,
                       hits: dict[str, list[str]], notes: list[str]) -> None:
    members: dict[str, list[str]] = defaultdict(list)
    for t, w in sleeve.items():
        if w != 0.0 and sector_of.get(t):
            members[sector_of[t]].append(t)
    for sector in sorted(members):
        gross = sum(abs(sleeve[t]) for t in members[sector])
        if gross > cap:
            factor = cap / gross
            for t in members[sector]:
                sleeve[t] *= factor
                hits[t].append("sector_cap")
            names = ", ".join(sorted(members[sector]))
            notes.append(f"sector cap: {sector} {gross:.1%} -> {cap:.1%} ({names})")


def _apply_book_limits(active: dict[str, float], core: float, pc: PortfolioConfig,
                       hits: dict[str, list[str]], notes: list[str]) -> None:
    """Scale the active legs so that core + gross <= max_gross and cash >= min_cash.

    The limits are measured on the whole book (core included), so ``min_cash`` means real
    cash in the account whatever the core weight is."""
    gross = sum(abs(w) for w in active.values())
    net = sum(active.values())
    factors = {}
    if gross > pc.max_gross - core:
        factors["max_gross"] = (pc.max_gross - core) / gross
    if net > 1.0 - pc.min_cash - core:
        factors["min_cash"] = (1.0 - pc.min_cash - core) / net
    if not factors:
        return
    f = min(factors.values())
    binding = sorted(k for k, v in factors.items() if v == f)
    for t in active:
        if active[t] != 0.0:
            active[t] *= f
            hits[t].extend(binding)
    notes.append(f"book limits ({', '.join(binding)}): active sleeve scaled by {f:.3f}")


def _trade_path(ideal: dict[str, float], current: dict[str, float], pc: PortfolioConfig, limits: _Limits,
                hits: dict[str, list[str]], notes: list[str]) -> dict[str, float]:
    """No-trade band, turnover cap, then the compliance repair described in the module docstring."""
    legs = sorted(set(ideal) | set(current))
    delta = {t: ideal.get(t, 0.0) - current.get(t, 0.0) for t in legs}
    for t in legs:
        if delta[t] != 0.0 and abs(delta[t]) < pc.no_trade_band:
            delta[t] = 0.0
            hits[t].append("no_trade_band")
    turnover = sum(abs(d) for d in delta.values())
    if turnover > pc.max_turnover:
        s = pc.max_turnover / turnover
        for t in legs:
            if delta[t] != 0.0:
                delta[t] *= s
                hits[t].append("turnover_cap")
        notes.append(f"turnover cap: sum|dw| {turnover:.3f} -> {pc.max_turnover:.3f} (all trades scaled by {s:.3f})")
    book = {t: current.get(t, 0.0) + delta[t] for t in legs}
    problems = limits.violations(book)
    if not problems:
        return book

    # Split each trade into the part that shrinks the existing position toward zero and the part
    # that adds exposure. In a long-only book these are exactly the sells and the buys.
    reduce = {t: _toward_zero_part(current.get(t, 0.0), delta[t]) for t in legs}
    reduced = {t: current.get(t, 0.0) + reduce[t] for t in legs}
    if not limits.violations(reduced):
        def trimmed(b: float) -> dict[str, float]:
            return {t: reduced[t] + b * (delta[t] - reduce[t]) for t in legs}

        b = _largest_feasible(lambda u: not limits.violations(trimmed(u)))
        for t in legs:
            if delta[t] != reduce[t]:
                hits[t].append("limit_trim")
        notes.append(f"band/turnover path breached {', '.join(problems)}: exposure-adding trades trimmed to {b:.1%}")
        return trimmed(b)

    def mixed(u: float) -> dict[str, float]:   # u=0 -> ideal target (compliant), u=1 -> banded path
        return {t: (1.0 - u) * ideal.get(t, 0.0) + u * book[t] for t in legs}

    u = _largest_feasible(lambda x: not limits.violations(mixed(x)))
    repaired = mixed(u)
    for t in legs:
        if repaired[t] != book[t]:
            hits[t].append("limit_repair")
    notes.append(f"current book breaches {', '.join(problems)}: moved {1.0 - u:.1%} of the remaining way to the "
                 f"ideal target to comply (risk limits override the turnover cap)")
    return repaired


# --------------------------------------------------------------------------- public API


def construct_portfolio(aggs: dict[str, AggregatedView], edge_multipliers: dict[str, float],
                        factsheets: dict[str, FactSheet], current_weights: dict[str, float],
                        cfg: FirmConfig, as_of: str, sectors: dict[str, str],
                        returns: pd.DataFrame | None = None) -> TargetPortfolio:
    """Convert pooled views into a :class:`TargetPortfolio` (see the module docstring).

    * ``edge_multipliers``: the chair's multiplier per ticker. A ticker missing from the
      dict had no memo, which per the committee contract means 1.0; a non-finite value
      means 0 (no trade).
    * ``current_weights``: current fraction of NAV per ticker, including the benchmark
      (the core leg). Held tickers absent from ``aggs`` get target 0 (they are sold,
      subject to the band and turnover cap).
    * ``sectors``: ticker -> sector, falling back to ``FactSheet.sector``. Unclassified
      names are not grouped under sector caps (only the position cap applies) and are
      listed in the notes.
    * ``returns``: daily returns (e.g. ``PriceData.returns()``); rows after ``as_of`` are
      ignored (P3).

    The benchmark is the passive core, never an active name: if it appears in ``aggs`` it
    is skipped with a note. ``core_weight`` in the result is the benchmark weight after
    the band and turnover cap; ``cash_weight = 1 - core - sum(active weights)``.
    """
    pc = cfg.portfolio
    _validate(pc)
    bench = cfg.universe.benchmark
    current = _clean_current(current_weights)
    notes: list[str] = []
    hits: dict[str, list[str]] = defaultdict(list)
    rationale: dict[str, str] = {}

    # 1-2. per-name edge and capped fractional Kelly (sleeve weights, fractions of sleeve capital)
    sleeve: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for t in sorted(aggs):
        if t == bench:
            notes.append(f"{t} is the benchmark (passive core); its committee view is not sized")
            continue
        sized = _size_name(t, aggs[t], edge_multipliers.get(t, 1.0), factsheets.get(t), cfg, hits[t])
        sleeve[t], sigma[t], rationale[t] = sized.weight, sized.sigma_ann, sized.rationale
    for t in sorted(current):
        if t != bench and t not in sleeve:
            sleeve[t] = 0.0
            sigma[t] = _annual_vol(factsheets.get(t), t)[0]
            rationale[t] = "held, but no committee view this cycle: target 0 (exit)"

    # 3. sector caps
    sector_of = {}
    for t in sleeve:
        fs = factsheets.get(t)
        label = (sectors.get(t) or (fs.sector if fs is not None else "") or "").strip()
        if label:
            sector_of[t] = label
    unclassified = sorted(t for t, w in sleeve.items() if w != 0.0 and t not in sector_of)
    if unclassified:
        notes.append(f"no sector label (sector cap not applied): {', '.join(unclassified)}")
    _apply_sector_caps(sleeve, sector_of, pc.max_sector, hits, notes)

    # 4. sleeve volatility target
    names = sorted(t for t in sleeve if sleeve[t] != 0.0 or t in current)
    cov = _covariance(names, sigma, returns, as_of)
    sleeve_vol = cov.vol(sleeve)
    if sleeve_vol > pc.target_vol:
        f = pc.target_vol / sleeve_vol
        for t in sleeve:
            if sleeve[t] != 0.0:
                sleeve[t] *= f
                hits[t].append("vol_target")
        notes.append(f"vol target: sleeve vol {sleeve_vol:.1%} -> {pc.target_vol:.1%} ({cov.source})")

    # 5. core-satellite and book limits (fractions of total NAV from here on)
    core = min(pc.core_weight, 1.0 - pc.min_cash, pc.max_gross)
    if core < pc.core_weight:
        notes.append(f"core weight {pc.core_weight:.1%} reduced to {core:.1%} to respect the cash floor / max gross")
    ideal = {t: w * (1.0 - core) for t, w in sleeve.items()}
    _apply_book_limits(ideal, core, pc, hits, notes)
    ideal[bench] = core

    # 6. trade path: band, turnover, compliance
    limits = _Limits(bench=bench, long_only=pc.long_only, max_position=pc.max_position, max_sector=pc.max_sector,
                     max_gross=pc.max_gross, min_cash=pc.min_cash, vol_cap=pc.target_vol * (1.0 - core),
                     sector_of=sector_of, cov=cov)
    final = _trade_path(ideal, current, pc, limits, hits, notes)
    final = {t: (0.0 if abs(w) < DUST else w) for t, w in final.items()}
    if hits.get(bench):
        notes.append(f"core leg ({bench}): {', '.join(dict.fromkeys(hits[bench]))}")

    positions = []
    for t in sorted(set(final) - {bench}):
        agg = aggs.get(t)
        positions.append(TargetPosition(
            ticker=t,
            weight=final[t],
            prob_outperform=agg.prob_outperform if agg is not None else 0.5,
            expected_excess_return_pct=agg.expected_excess_return_pct if agg is not None else 0.0,
            rationale=rationale.get(t, ""),
            constraints_hit=list(dict.fromkeys(hits.get(t, []))),
        ))
    core_final = final.get(bench, 0.0)
    return TargetPortfolio(as_of=as_of, positions=positions, core_weight=core_final,
                           cash_weight=1.0 - sum(final.values()), notes=notes)
