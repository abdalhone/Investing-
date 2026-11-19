"""The invariant risk core (P9): deterministic rules nothing upstream can override.

Everything before this module (analysts, chair, LLM risk reviewer, even the portfolio
constructor) proposes. This module disposes, with rules modelled on multi-manager "pod"
risk practice and broker-dealer pre-trade controls (SEC 15c3-5, after Knight Capital):

* **Kill switch**: if ``<runs_dir>/<kill_switch_file>`` exists, the target becomes the
  current book, so no orders are generated. A human creates or removes the file.
* **Drawdown freeze**: at ``drawdown_freeze`` below peak NAV the active sleeve goes to 0
  and ``state.frozen`` is set. It stays set until a human resets it, even if NAV
  recovers (section 4 of ARCHITECTURE.md: resetting a frozen book is a human decision).
  The passive core is kept, because the core is not the thing that failed. The freeze
  also records ``state.frozen_at_nav``. A human unfreezes the book with ``aifirm unfreeze``
  (the supported path: it records who reset the book and why in the ledger); setting
  ``frozen`` to false by hand in ``risk_state.json`` still works. Either way the next
  cycle sees ``frozen_at_nav`` still set, treats that as the human reset, restarts the
  high-water mark at the current NAV and clears the marker. Without that, a book frozen to cash could never recover the
  old peak and would re-freeze on the very next run.
* **Drawdown de-risk and re-risk** (RESEARCH 6.4): at ``drawdown_derisk`` active weights
  are halved and ``state.derisked`` is set. The book stays halved until it has recovered
  half of the drawdown, measured from the peak to the lowest NAV since that peak
  (``state.trough_nav``): full gross returns once ``nav >= trough + 0.5 * (peak - trough)``
  (or at a new peak), while unfrozen. Without this hysteresis the book would re-risk the
  moment the drawdown crept back under the threshold and could flip between half and full
  gross on every small move around it.

  "Halved" means half of the *un-derisked plan*, and it is idempotent: applying it every
  cycle of an episode does not compound. A leg built by construction carries its
  ``ideal_weight`` (the target before the trade path), and while the de-risk is in force
  its weight is capped at ``0.5 * |ideal_weight|``. Halving the incoming target instead
  would compound with partial rebalancing (the target is a step from a book that was
  already halved last cycle), selling into the drawdown every cycle toward about a third of
  the ideal. The plan includes the LLM risk reviewer's cut: a reviewer multiplier ``m``
  scales ``ideal_weight`` too (:func:`aifirm.agents.risk_review.apply_risk_review`), so a
  reviewed leg is capped at ``0.5 * m * |ideal|``, half of the reviewed plan. A *deliberate
  hold* (:data:`HOLD_LABELS`: an LLM-outage hold, a held name whose memo failed, a held name
  with no view, a risk-review halt) is halved on the cycle the de-risk triggers, when the
  held book is still full size. The weight the de-risk sets for every leg is recorded in
  ``state.derisk_caps``; on every later cycle of the episode a hold keeps its current weight
  up to that recorded level (an LLM outage must never force a sale, and a hold is never sold
  below it), and is cut to it only when the earlier sale did not execute (an order blocked
  for a missing price or ADV, or rejected). The record is cleared on re-risk, on a freeze
  and on a human reset. A leg without an ``ideal_weight`` (a hand-built target) is halved.
* **Restricted list / minimum price**: weight 0. A name whose price cannot be verified
  may not grow (we will not add risk we cannot check, nor force an exit on a data glitch).
* **Daily-loss halt**: after a one-day NAV loss of ``daily_loss_halt`` or more on *any*
  day since the previous cycle (``worst_day_return``, or ``nav / prev_nav``), no weight
  may increase this cycle (sells and reductions still go through). The high-water mark
  must likewise see every daily NAV (:func:`update_risk_state` per mark), not only the
  decision-day NAVs, or drawdowns inside a cycle are measured from a too-low peak.
* **Backstops**: non-finite weights -> 0, shorts in a long-only book -> 0, the hard
  per-name cap ``risk.max_position_hard`` (construction sizes to the lower target
  ``portfolio.max_position``; a drifted holding between the two is not cut here), max gross
  and cash floor. Construction already enforces these; repeating them here
  means a bug upstream cannot reach the broker.

Invariant: apart from the kill switch (which returns the current book unchanged), every
rule only moves a weight toward zero, so ``|out_w| <= |target_w|`` for every name and the
engine can never raise gross exposure above the proposed target.

``orders_from_target`` turns weights into share orders with the same fail-closed stance:
no price, no ADV or no valid NAV means no order for that name, with a message. A full
exit (target exactly 0 on a held name: a freeze, the restricted list, the minimum price,
a name no longer covered) is never skipped by the no-trade band; otherwise a position
below the band could never be exited by any rule.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import FirmConfig
from ..data.factsheet import FactSheet
from ..schemas import Order, TargetPortfolio, TargetPosition

DUST = 1e-12
CAP_NOISE = 1e-9   # a weight this close above the position cap is float rounding, not a breach
DERISK_FACTOR = 0.5
# Constraint labels of legs the proposer deliberately kept at the current weight (no view, an LLM
# failure or outage, a risk-review halt): the de-risk halves them once, when it triggers.
HOLD_LABELS = frozenset({"pipeline:hold", "unreviewed_hold", "no_view_hold", "risk_review_halt"})


@dataclass
class RiskState:
    """Book-level risk memory carried between cycles. ``frozen`` is only ever cleared by a human.

    ``last_nav`` / ``last_as_of``: the NAV and date of the last decision (set by the
    pipeline), so the next cycle can see the path since then. ``frozen_at_nav``: set when
    the book freezes and kept after a human clears ``frozen``, which is how the next cycle
    recognises the human reset (see the module docstring). ``trough_nav``: the lowest NAV
    mark since the last peak (:func:`update_risk_state`); ``derisked``: the drawdown
    de-risk is in force until half of the drawdown is recovered; ``derisk_caps``: during a
    de-risk episode, the absolute weight the de-risk last set for each leg (a deliberate
    hold is never kept above it). They default so that a state saved before they existed
    loads unchanged."""

    peak_nav: float
    frozen: bool = False
    last_nav: float | None = None
    last_as_of: str | None = None
    frozen_at_nav: float | None = None
    trough_nav: float | None = None
    derisked: bool = False
    derisk_caps: dict[str, float] = field(default_factory=dict)


def _valid_positive(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def update_risk_state(state: RiskState, nav: float) -> RiskState:
    """Record a new NAV mark: raise the peak if exceeded, track the lowest NAV since the
    peak (``trough_nav``, reset to the NAV at each new peak) and remember ``last_nav``.

    Returns a new state (the input is not modified). A non-finite or non-positive NAV means
    the accounting is broken, so the book is frozen (fail closed). A corrupt peak is reset
    to ``nav`` and the book frozen, so drawdown math works again once a human unfreezes it.
    """
    if not _valid_positive(nav):
        marker = state.frozen_at_nav if state.frozen_at_nav is not None else (state.last_nav or 0.0)
        return replace(state, frozen=True, frozen_at_nav=marker)
    nav = float(nav)
    if not _valid_positive(state.peak_nav):
        return replace(state, peak_nav=nav, last_nav=nav, frozen=True, trough_nav=nav,
                       frozen_at_nav=state.frozen_at_nav if state.frozen_at_nav is not None else nav)
    if nav >= float(state.peak_nav):
        return replace(state, peak_nav=nav, last_nav=nav, trough_nav=nav)
    trough = min(float(state.trough_nav), nav) if _valid_positive(state.trough_nav) else nav
    return replace(state, last_nav=nav, trough_nav=trough)


def _no_increase(target: float, current: float) -> float:
    """The largest-magnitude weight that does not add exposure relative to ``current``:
    same side as ``current`` and no larger; opening or flipping a position gives 0."""
    if target * current > 0:
        return math.copysign(min(abs(target), abs(current)), current)
    return 0.0


def _hold_current(target: TargetPortfolio, current_weights: dict[str, float], bench: str,
                  message: str) -> TargetPortfolio:
    """A target equal to the current book, so that no orders follow."""
    by_ticker = {p.ticker: p for p in target.positions}
    positions = []
    for t in sorted(t for t in current_weights if t != bench):
        prior = by_ticker.get(t)
        positions.append(TargetPosition(
            ticker=t, weight=float(current_weights[t]),
            prob_outperform=prior.prob_outperform if prior else 0.5,
            expected_excess_return_pct=prior.expected_excess_return_pct if prior else 0.0,
            rationale=message, constraints_hit=["risk:hold_current"],
        ))
    core = float(current_weights.get(bench, 0.0))
    return TargetPortfolio(as_of=target.as_of, positions=positions, core_weight=core,
                           cash_weight=1.0 - core - sum(p.weight for p in positions),
                           notes=[*target.notes, message])


def apply_risk_limits(target: TargetPortfolio, state: RiskState, nav: float, prev_nav: float | None,
                      current_weights: dict[str, float], cfg: FirmConfig, factsheets: dict[str, FactSheet],
                      runs_dir: str, *, worst_day_return: float | None = None) -> tuple[TargetPortfolio, list[str]]:
    """Apply the invariant rules (module docstring) and return ``(target, messages)``.

    ``state.frozen`` / ``state.frozen_at_nav`` are set in place when the freeze triggers,
    and ``state.peak_nav`` is reset in place when a human has unfrozen the book. Drawdown
    is measured from ``max(state.peak_nav, nav)`` so a caller that has not yet called
    :func:`update_risk_state` cannot under-state it; callers should feed every daily NAV
    to :func:`update_risk_state` so the peak is the true high-water mark.
    ``worst_day_return`` is the worst one-day NAV return since the previous cycle (a
    fraction; None if unknown); the daily-loss halt triggers if it, or ``nav / prev_nav - 1``,
    is a loss of ``daily_loss_halt`` or more. ``current_weights`` includes the benchmark
    (core) leg. Messages are also appended to the returned target's notes.
    """
    rc, pc = cfg.risk, cfg.portfolio
    bench = cfg.universe.benchmark
    for t, w in current_weights.items():
        if not math.isfinite(float(w)):
            raise ValueError(f"current weight for {t} is not finite: {w!r}")

    kill = Path(runs_dir) / rc.kill_switch_file
    if kill.exists():
        msg = f"KILL SWITCH: {kill} exists, no trades this cycle (target = current book)"
        return _hold_current(target, current_weights, bench, msg), [msg]
    if not _valid_positive(nav) or not _valid_positive(state.peak_nav):
        msg = f"invalid NAV ({nav!r}) or peak NAV ({state.peak_nav!r}): no trades this cycle"
        return _hold_current(target, current_weights, bench, msg), [msg]

    msgs: list[str] = []
    hits: dict[str, list[str]] = defaultdict(list)
    weights: dict[str, float] = {}
    for p in target.positions:
        w = float(p.weight)
        if not math.isfinite(w):
            msgs.append(f"{p.ticker}: non-finite target weight -> 0")
            hits[p.ticker].append("risk:invalid_weight")
            w = 0.0
        if pc.long_only and w < 0:
            msgs.append(f"{p.ticker}: short target {w:.2%} in a long-only book -> 0")
            hits[p.ticker].append("risk:long_only")
            w = 0.0
        if abs(w) > rc.max_position_hard:
            # Construction meets the cap up to float rounding (e.g. 0.10000000000100001); clip
            # that silently and report only a real breach.
            if abs(w) > rc.max_position_hard + CAP_NOISE:
                msgs.append(f"{p.ticker}: {w:.2%} exceeds the position cap (risk.max_position_hard) -> "
                                f"{rc.max_position_hard:.2%}")
                hits[p.ticker].append("risk:max_position")
            w = math.copysign(rc.max_position_hard, w)
        weights[p.ticker] = w
    core = float(target.core_weight) if math.isfinite(float(target.core_weight)) else 0.0
    core_limit = min(pc.max_gross, 1.0 - pc.min_cash)
    if not 0.0 <= core <= core_limit:
        msgs.append(f"core weight {core:.2%} outside [0, {core_limit:.2%}] -> clipped")
        core = min(max(core, 0.0), core_limit)

    # A human cleared ``frozen`` after a freeze: the reset restarts the high-water mark here,
    # otherwise the book (in cash, so unable to recover the old peak) would re-freeze at once.
    if not state.frozen and state.frozen_at_nav is not None:
        msgs.append(f"book unfrozen by a human since the freeze (NAV then {state.frozen_at_nav:,.2f}): "
                    f"high-water mark reset from {float(state.peak_nav):,.2f} to the current NAV {float(nav):,.2f}")
        state.peak_nav = float(nav)
        state.frozen_at_nav = None
        state.trough_nav = float(nav)
        state.derisked = False
        state.derisk_caps = {}

    # Drawdown: freeze (sticky) or de-risk.
    peak = max(float(state.peak_nav), float(nav))
    drawdown = 1.0 - float(nav) / peak
    if state.frozen or drawdown >= rc.drawdown_freeze:
        if state.frozen_at_nav is None:
            state.frozen_at_nav = float(nav)       # marks the freeze, so a human reset can be recognised
        if not state.frozen:
            state.frozen = True
            msgs.append(f"DRAWDOWN FREEZE: {drawdown:.1%} from peak >= {rc.drawdown_freeze:.0%}; "
                        f"active sleeve to 0 until a human resets the book")
        else:
            msgs.append(f"book is frozen (drawdown {drawdown:.1%}); active sleeve stays at 0 until a human resets it")
        for t in weights:
            if weights[t] != 0.0:
                hits[t].append("risk:drawdown_freeze")
            weights[t] = 0.0
        state.derisk_caps = {}
    else:
        trough = min(float(state.trough_nav), float(nav)) if _valid_positive(state.trough_nav) else float(nav)
        recovery = trough + 0.5 * (peak - trough)        # half of the peak-to-trough drawdown recovered
        halve = False
        newly = False
        if drawdown >= rc.drawdown_derisk:
            halve = True
            newly = not state.derisked
            state.derisked = True
            why = (f"drawdown de-risk: {drawdown:.1%} from peak >= {rc.drawdown_derisk:.0%}; active weights capped at "
                   f"half of the un-derisked plan until half of the drawdown is recovered")
        elif state.derisked and float(nav) < recovery and float(nav) < peak:
            halve = True
            why = (f"drawdown de-risk still in force: {drawdown:.1%} from peak, but less than half of the drawdown "
                   f"from {peak:,.2f} to the trough {trough:,.2f} is recovered (full gross at NAV >= {recovery:,.2f}); "
                   f"active weights capped at half of the un-derisked plan")
        elif state.derisked:
            state.derisked = False
            state.derisk_caps = {}
            msgs.append(f"re-risk: half of the drawdown from {peak:,.2f} to the trough {trough:,.2f} recovered (NAV "
                        f"{float(nav):,.2f}); full gross restored")
        if halve:
            by_ticker = {p.ticker: p for p in target.positions}
            caps = {} if newly else dict(state.derisk_caps)   # rebound, never mutated: copies of the state share it
            cut, kept, finished = [], [], []
            for t in weights:
                if weights[t] == 0.0:
                    continue
                pos = by_ticker[t]
                ideal = pos.ideal_weight
                if HOLD_LABELS.intersection(pos.constraints_hit):
                    if newly:          # the held book is still full size: halve it once
                        new_w = weights[t] * DERISK_FACTOR
                    elif t in caps and abs(weights[t]) - caps[t] >= max(pc.no_trade_band, CAP_NOISE):
                        # the level the de-risk set earlier in the episode was not reached (the sale did
                        # not execute): finish it, but never sell a hold below that level. A gap under
                        # the no-trade band (share rounding, a small drift) would not trade anyway.
                        new_w = math.copysign(caps[t], weights[t])
                        finished.append(t)
                    else:
                        new_w = weights[t]
                        kept.append(t)
                elif ideal is not None and math.isfinite(float(ideal)):
                    cap = DERISK_FACTOR * abs(float(ideal))
                    new_w = math.copysign(min(abs(weights[t]), cap), weights[t])
                else:
                    new_w = weights[t] * DERISK_FACTOR
                if t in kept:
                    caps.setdefault(t, abs(new_w))     # a hold's drift never moves the recorded level
                else:
                    caps[t] = abs(new_w)
                if new_w != weights[t]:
                    weights[t] = new_w
                    hits[t].append("risk:drawdown_derisk")
                    cut.append(t)
            state.derisk_caps = caps
            msgs.append(why + (f"; cut: {', '.join(sorted(cut))}" if cut else
                               "; no leg is above its de-risk level, nothing cut"))
            if kept:
                msgs.append(f"drawdown de-risk: deliberate hold(s) {', '.join(sorted(kept))} kept at the current "
                            "weight (not above the level the de-risk set, give or take the no-trade band; a hold is never "
                            "sold below that level)")
            if finished:
                msgs.append(f"drawdown de-risk: deliberate hold(s) {', '.join(sorted(finished))} cut to the level the "
                            "de-risk set earlier in this episode (that sale did not execute)")

    # Restricted list and minimum price.
    restricted = {x.upper() for x in rc.restricted}
    for t in sorted(weights):
        if weights[t] == 0.0:
            continue
        if t.upper() in restricted:
            msgs.append(f"{t}: restricted -> 0")
            hits[t].append("risk:restricted")
            weights[t] = 0.0
            continue
        fs = factsheets.get(t)
        fact = fs.get(f"{t}.price") if fs is not None else None
        price = float(fact.value) if fact is not None and isinstance(fact.value, (int, float)) else math.nan
        if not math.isfinite(price):
            capped = _no_increase(weights[t], float(current_weights.get(t, 0.0)))
            if capped != weights[t]:
                msgs.append(f"{t}: price unavailable, position may not grow ({weights[t]:.2%} -> {capped:.2%})")
                hits[t].append("risk:no_price")
                weights[t] = capped
        elif price < rc.min_price:
            msgs.append(f"{t}: price {price:.2f} below minimum {rc.min_price:.2f} -> 0")
            hits[t].append("risk:min_price")
            weights[t] = 0.0
    if core > 0 and bench.upper() in restricted:
        msgs.append(f"{bench} (core) is restricted -> core 0")
        core = 0.0

    # Daily-loss halt: nothing may grow this cycle. Any one-day loss since the previous
    # cycle counts, not only the decision day's own return. A non-finite return (a broken
    # NAV) counts as a loss (fail closed).
    losses: list[tuple[float, str]] = []
    if prev_nav is not None:
        day = float(nav) / float(prev_nav) - 1.0 if _valid_positive(prev_nav) else -math.inf
        losses.append((day, f"one-day NAV change {day:.1%}" if math.isfinite(day)
                       else f"invalid previous NAV {prev_nav!r}"))
    if worst_day_return is not None:
        w = float(worst_day_return)
        losses.append((w if math.isfinite(w) else -math.inf,
                       f"worst one-day NAV change since the last cycle {w:.1%}" if math.isfinite(w)
                       else "invalid NAV in the path since the last cycle"))
    halt = min(losses, key=lambda x: x[0]) if losses else None
    if halt is not None and halt[0] <= -rc.daily_loss_halt:
        msgs.append(f"DAILY-LOSS HALT: {halt[1]}; no weight may increase this cycle")
        for t in weights:
            capped = _no_increase(weights[t], float(current_weights.get(t, 0.0)))
            if capped != weights[t]:
                hits[t].append("risk:daily_loss_halt")
                weights[t] = capped
        core = max(0.0, _no_increase(core, float(current_weights.get(bench, 0.0))))

    # Backstop: gross and cash floor on the whole book (only scales active weights down).
    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    factor = 1.0
    if gross > pc.max_gross - core + 1e-12:
        factor = min(factor, (pc.max_gross - core) / gross)
    if net > 1.0 - pc.min_cash - core + 1e-12:
        factor = min(factor, (1.0 - pc.min_cash - core) / net)
    if factor < 1.0:
        msgs.append(f"backstop: active weights scaled by {factor:.3f} to respect max gross / cash floor")
        for t in weights:
            if weights[t] != 0.0:
                weights[t] *= factor
                hits[t].append("risk:book_limits")

    positions = [p.model_copy(update={
        "weight": 0.0 if abs(weights[p.ticker]) < DUST else weights[p.ticker],
        "constraints_hit": list(dict.fromkeys([*p.constraints_hit, *hits.get(p.ticker, [])])),
    }) for p in target.positions]
    out = TargetPortfolio(as_of=target.as_of, positions=positions, core_weight=core,
                          cash_weight=1.0 - core - sum(p.weight for p in positions),
                          notes=[*target.notes, *msgs])
    return out, msgs


def orders_from_target(target: TargetPortfolio, positions: dict[str, float], prices: dict[str, float], nav: float,
                       adv: dict[str, float], cfg: FirmConfig, benchmark: str) -> tuple[list[Order], list[str]]:
    """Share orders that move the book from ``positions`` (shares) to ``target`` (weights).

    * The benchmark's target is ``core_weight`` (plus any active weight on it). Held
      tickers absent from the target are sold: a target is the complete desired book.
    * Trades with ``|delta notional| < no_trade_band * nav`` are skipped (not worth the cost),
      except a full exit: a held name whose target is exactly 0 (frozen book, restricted,
      below the minimum price, no longer covered, or simply absent from the target) is
      always sold. Upstream construction already applies the band to discretionary
      trades (it keeps the current weight, not 0), so a 0 here is a decision to exit, and
      skipping it would leave a sub-band position that no rule could ever remove.
    * Each order's notional is capped at ``max_adv_participation * ADV`` (``adv`` is the
      average daily *dollar* volume) so we never are the market; a message says so.
    * A missing or invalid price or ADV blocks that ticker's order with a message (fail
      closed); this includes the benchmark, so callers must pass its ADV too.
    * In a long-only book sells never exceed the shares held, so rounding cannot create a short.
    * Sells come before buys (they fund them), each group sorted by ticker for determinism.
    """
    if not _valid_positive(nav):
        return [], [f"invalid NAV {nav!r}: no orders"]
    rc, pc = cfg.risk, cfg.portfolio
    msgs: list[str] = []
    weights = dict(target.weights())
    weights[benchmark] = weights.get(benchmark, 0.0) + target.core_weight
    tickers = sorted(set(weights) | {t for t, q in positions.items() if q != 0})
    band = pc.no_trade_band * nav
    sells: list[Order] = []
    buys: list[Order] = []
    for t in tickers:
        w = float(weights.get(t, 0.0))
        if not math.isfinite(w):
            msgs.append(f"{t}: non-finite target weight, order blocked")
            continue
        if pc.long_only and w < 0:
            msgs.append(f"{t}: short target {w:.2%} in a long-only book treated as 0")
            w = 0.0
        held = float(positions.get(t, 0.0))
        if not math.isfinite(held):
            msgs.append(f"{t}: non-finite share count {held!r}, order blocked")
            continue
        price = prices.get(t)
        if not _valid_positive(price):
            msgs.append(f"{t}: no valid price ({price!r}), order blocked")
            continue
        price = float(price)
        delta = w * nav - held * price
        full_exit = w == 0.0 and held != 0.0
        if delta == 0.0 or (abs(delta) < band and not full_exit):
            continue
        a = adv.get(t)
        if not _valid_positive(a):
            msgs.append(f"{t}: no valid ADV ({a!r}), order of {delta:+,.0f} blocked (liquidity unknown)")
            continue
        cap = rc.max_adv_participation * float(a)
        reason = f"target {w:.2%} of NAV (now {held * price / nav:.2%})"
        if abs(delta) > cap:
            msgs.append(f"{t}: order {delta:+,.0f} capped at {rc.max_adv_participation:.1%} of ADV ({cap:,.0f})")
            reason += f"; capped at {rc.max_adv_participation:.1%} of ADV"
            delta = math.copysign(cap, delta)
        qty = abs(delta) / price
        if delta < 0 and pc.long_only:
            qty = min(qty, max(held, 0.0))
        if qty <= 0:
            continue
        order = Order(ticker=t, side="buy" if delta > 0 else "sell", quantity=qty, reference_price=price,
                      notional=qty * price, reason=reason)
        (buys if delta > 0 else sells).append(order)
    return sells + buys, msgs
