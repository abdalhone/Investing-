"""Command-line interface: ``aifirm factsheet | run | compare | screen | backtest | score | reid-probe |
verify-ledger | unfreeze``.

Safety defaults (P9, ARCHITECTURE.md section 4):

* ``run`` trades the **paper broker** unless the config says ``broker = "alpaca"``, and even
  then it asks a human to approve the orders (``--yes`` skips the question; use it only
  in a supervised schedule). ``--offline`` always uses the paper broker and keeps its
  state apart (``<runs_dir>/offline``), so heuristic trades never mix with a real book.
* Paper orders decided at a close are filled at the **next trading day's open** on the
  following run (P3), never at the price that produced the decision. A second run before
  that open refuses to decide again, so a plan cannot be doubled.
* **Quant-only shadow book** (RESEARCH 6.6, 7.4 S6). After the live decision, ``run`` also
  decides a paper-only book with the language models removed (:mod:`aifirm.shadow`) in
  ``<book>/shadow``, so ``compare`` can report the AI's value added net of its LLM cost and
  the paired per-date IC. The shadow runs after the live decision is persisted, never
  sends orders to a real broker, and a shadow failure is a printed and ledgered warning,
  never a failed run (``--no-shadow`` or ``evaluation.shadow_book = false`` turn it off).
* Runs that call Claude check for credentials first and stop with a clear message
  instead of failing halfway (fail closed). ``ANTHROPIC_API_KEY`` is enough; an
  ``ant auth login`` profile, ``ANTHROPIC_AUTH_TOKEN`` or workload identity also work.
  ``reid-probe`` refuses to run without them unless ``--offline``.
* Decisions use **completed sessions only**. ``--as-of`` (and ``backtest --end``,
  ``score --as-of``) may not be after :func:`last_completed_session`: today before the
  close settles, or a future date, would treat a partial intraday bar as a close (P3).
* A book **never decides twice on the same session or on an earlier one** (no duplicate
  forecasts, no backdated orders filled at historical prices behind later fills).
  ``run --force`` overrides that, and the override is written to the ledger.
* ``unfreeze`` is the human reset of a drawdown freeze: it asks for a reason and a
  confirmation, restarts the high-water mark at the NAV of the book's last decision and
  records the change in the ledger (ARCHITECTURE.md section 4).
* **Strategy flags** on ``run`` and ``backtest`` (``--preset``, ``--combination``,
  ``--no-calibration`` ...) change the configuration, so they change its fingerprint (a new
  trial for the deflated Sharpe ratio). The backtest's **evaluation flags**
  (``--random-books``, ``--cost-multipliers`` ...) only change what the report measures and
  never the fingerprint. Every flag is validated like the config file (ranges and
  cross-checks), never assigned unchecked.

Book layout under ``runs_dir`` (``runs_dir/offline`` with ``--offline``): see
:mod:`aifirm.books`; the shadow book is ``<book>/shadow``. The kill switch is always
``<runs_dir>/<risk.kill_switch_file>`` and applies to the shadow too. A relative
``runs_dir`` is relative to the config file's directory (:meth:`FirmConfig.runs_path`),
never to the shell's working directory, so a scheduled run from elsewhere finds the same
book and the same kill switch; every run prints the resolved path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .agents import ROLES, make_offline_llm, run_reid_probe
from .backtest.engine import run_backtest
from .backtest.ic import overlap_lags
from .books import (BROKER_FILE, FORECASTS_FILE, LEDGER_FILE, BookError, append_anchor,
                    last_decision, load_risk_state, marks, pending_as_of, pending_tickers, queue_paper_orders,
                    settle_paper_book, waiting_message, write_json)
from .config import FirmConfig, describe_config_error
# The session clock lives with the price loader (every caller needs it); MARKET_TZ and
# last_completed_session stay importable from here.
from .data.prices import MARKET_TZ, SESSION_CLOSE_SETTLED, PriceData, last_completed_session, load_prices  # noqa: F401
from .execution.broker import AlpacaBroker, PaperBroker
from .journal import (COMMITTEE_AGENT, LLM_POOL_AGENT, SHADOW_AGENT, ForecastStore, Ledger, LedgerError,
                      LessonStore, paired_rank_ic, scorecard)
from .llm.client import LLM, ClaudeLLM
from .pipeline import PRICE_LOOKBACK_DAYS, Decision, decide, gather_data, needs_llm, universe
from .quant import QUANT_AGENT, quant_views, screen_universe
from .report import render_backtest_report, render_decision_report, six_lines
from .shadow import (BookComparison, BookSnapshot, ShadowOutcome, compare_books, run_shadow, shadow_applies,
                     shadow_book_dir)

REID_DIR = "reid"


class CLIError(BookError):
    """A problem the user can fix; printed without a traceback."""


# --------------------------------------------------------------------------- helpers


def credentials_problem(alternative: str = "add --offline to use the deterministic offline rules (not AI)") -> str | None:
    """None if the Anthropic SDK can find credentials, else a message saying how to fix it.
    ``alternative`` is the command's own way to run without a model (commands differ: ``run``
    takes ``--offline``, ``backtest`` takes ``--mode offline`` or ``--mode quant``).

    ``ANTHROPIC_API_KEY`` is accepted without further checks. Otherwise the SDK's own
    resolution chain decides (auth token, ``ANTHROPIC_PROFILE``, workload identity, or the
    active ``ant auth login`` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic.lib.credentials import default_credentials
        if default_credentials() is not None:
            return None
    except Exception as exc:  # a broken profile must surface, not silently fall through
        return f"Anthropic credentials could not be loaded: {str(exc).rstrip('.')}. Fix that, or {alternative}."
    return ("No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login` to create a profile, "
            f"or {alternative}.")


def _load_config(path: str | None) -> FirmConfig:
    """The validated config, with ``runs_dir`` made absolute (relative to the config file's
    directory; the working directory only for the built-in defaults). Everything derived
    from it (book, kill switch, backtests, trial ledger) then ignores the shell's cwd."""
    if path is None:
        cfg = FirmConfig()
    else:
        try:
            cfg = FirmConfig.load(path)
        except FileNotFoundError as exc:
            raise CLIError(f"config file not found: {path}" if not Path(path).expanduser().exists() else str(exc)) \
                from exc
        except ValidationError as exc:                   # unknown keys, values out of range
            raise CLIError(f"invalid config {path}:\n  {describe_config_error(exc)}") from exc
        except ValueError as exc:                        # TOMLDecodeError is a ValueError
            raise CLIError(f"invalid config {path}: {exc}") from exc
    return cfg.model_copy(update={"runs_dir": str(cfg.runs_path())})


def _parse_multipliers(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise CLIError(f"--cost-multipliers {text!r}: use comma-separated numbers, e.g. 0,1,2") from exc


def _apply_flags(cfg: FirmConfig, args: argparse.Namespace) -> FirmConfig:
    """``cfg`` with the command-line overrides in ``args`` applied and the result validated
    like a config file (ranges, cross-checks, the screen file's existence). Flags absent
    from ``args`` (another command's) or left at their defaults change nothing; with no
    override the same object is returned.

    Strategy flags (screen, combination, tilt, calibration, news reader, tier gate,
    rebalance fraction) change the fingerprint; evaluation flags and ``--no-shadow`` live in
    ``evaluation`` and do not."""
    data = cfg.model_dump()
    scr, proc, ev = data["universe"]["screen"], data["process"], data["evaluation"]
    changed = False

    def flag(name: str) -> object:
        return getattr(args, name, None)

    def put(section: dict, key: str, value: object) -> None:
        nonlocal changed
        section[key] = value
        changed = True

    if flag("preset"):
        put(scr, "preset", str(flag("preset")))
    if flag("universe_file"):
        put(scr, "file", str(Path(str(flag("universe_file"))).expanduser().resolve()))
    if flag("deep_dive") is not None:
        put(scr, "deep_dive_top_n", flag("deep_dive"))
    if flag("min_dollar_volume") is not None:
        put(scr, "min_dollar_volume", flag("min_dollar_volume"))
    if changed:
        scr["enabled"] = True                   # any screen flag turns the screen on
    if flag("combination"):
        put(proc, "combination", flag("combination"))
    if flag("max_tilt") is not None:
        put(proc, "overlay_max_tilt", flag("max_tilt"))
    if flag("no_calibration"):
        put(data["calibration"], "enabled", False)
    if flag("no_news_reader"):
        put(proc, "news_reader_enabled", False)
    if flag("no_tier_gate"):
        put(proc, "tier_gate_enabled", False)
    if flag("rebalance_fraction") is not None:
        put(data["portfolio"], "rebalance_fraction", flag("rebalance_fraction"))
    if flag("random_books") is not None:
        put(ev, "n_random_books", flag("random_books"))
    if flag("seed") is not None:
        put(ev, "random_seed", flag("seed"))
    if flag("sector_matched_books"):
        put(ev, "random_books_sector_matched", True)
    if flag("cost_multipliers"):
        put(ev, "cost_multipliers", _parse_multipliers(str(flag("cost_multipliers"))))
    if flag("breakeven_cap") is not None:
        put(ev, "breakeven_cap", flag("breakeven_cap"))
    if flag("cutoff_buffer_days") is not None:
        put(ev, "cutoff_buffer_days", flag("cutoff_buffer_days"))
    if flag("ic_min_names") is not None:
        put(ev, "ic_min_names", flag("ic_min_names"))
    if flag("no_factor_proxies"):
        put(ev, "factor_proxies", False)
    if flag("no_shadow"):
        put(ev, "shadow_book", False)
    if not changed:
        return cfg
    try:
        return cfg.revalidated(data)
    except ValidationError as exc:
        raise CLIError(f"invalid command-line option(s):\n  {describe_config_error(exc)}") from exc
    except FileNotFoundError as exc:
        raise CLIError(str(exc)) from exc


def _book_dir(cfg: FirmConfig, offline: bool, shadow: bool = False) -> Path:
    root = cfg.runs_path()
    book = root / "offline" if offline else root
    return shadow_book_dir(book) if shadow else book


def _completed_session(value: str | None, flag: str = "--as-of") -> str:
    """``value`` (YYYY-MM-DD), or the last completed session when None.

    A date after :func:`last_completed_session` is refused: its bar is partial (today
    before the close settles) or does not exist yet (a future date, which the price
    loader would silently map to today's partial bar). Deciding, resolving forecasts or
    backtesting on it would treat an intraday print as a close (P3)."""
    latest = last_completed_session()
    if value is None:
        return latest
    try:
        day = str(date.fromisoformat(value.strip()))
    except ValueError as exc:
        raise CLIError(f"{flag} {value!r} is not a date; use YYYY-MM-DD") from exc
    if day > latest:
        raise CLIError(f"{flag} {day} is after the last completed session ({latest}). Only completed sessions "
                       f"are used (a session's close is final after {SESSION_CLOSE_SETTLED[0]}:"
                       f"{SESSION_CLOSE_SETTLED[1]:02d} New York time); a later date would use a partial "
                       "intraday bar as if it were the close")
    return day


def _refuse_redecision(book: Path, as_of: str, last: str | None, force: bool) -> bool:
    """Refuse a decision at the close of ``as_of`` when the book already decided on that
    session or a later one. Re-deciding a session stores its forecasts twice (they then
    count twice in the scorecard and the believability weights); an earlier session would
    queue orders that fill at a historical open, with hindsight, behind fills the book
    already holds. Returns True when ``force`` overrides the refusal (the caller records it).
    Queued orders that are still waiting for their open are never overridden: that would
    double the plan."""
    if last is None or as_of > last:
        return False
    pending = pending_as_of(book)
    if pending is not None and as_of <= pending:
        raise CLIError(waiting_message(pending))
    if force:
        return True
    what = ("would repeat that decision (its forecasts would be stored and scored twice)" if as_of == last else
            "would be backdated: its orders would fill at a historical open, with hindsight, behind later fills")
    raise CLIError(f"this book ({book}) already decided at the close of {last}; a decision at the close of {as_of} "
                   f"{what}. Decide on a later completed session, or pass --force to override "
                   "(the override is recorded in the ledger).")


def _claude(cfg: FirmConfig, budget_usd: float) -> LLM:
    problem = credentials_problem()
    if problem:
        raise CLIError(problem)
    return ClaudeLLM(budget_usd=budget_usd, use_fallbacks=cfg.models.use_server_fallbacks)


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _print_scorecard(card: dict[str, dict]) -> None:
    if not card:
        print("No resolved forecasts yet (a forecast resolves after its horizon).")
        return
    print(f"{'agent':<12} {'n':>5} {'pre-cut':>7} {'brier':>7} {'skill':>7} {'logloss':>8} {'hit':>6} "
          f"{'meanP':>6} {'base':>6}")
    for agent, c in card.items():
        hit = f"{c['hit_rate']:.0%}" if c["hit_rate"] is not None else "-"
        print(f"{agent:<12} {c['n']:>5} {c['n_pre_cutoff']:>7} {c['brier']:>7.4f} {c['brier_skill']:>+7.3f} "
              f"{c['log_loss']:>8.4f} {hit:>6} {c['mean_prob']:>6.3f} {c['base_rate']:>6.0%}")
    print("brier: 0 is perfect, 0.25 = always saying 0.5; skill > 0 beats a coin flip. "
          "Pre-cutoff forecasts may be contaminated by training data. 'shadow' is the quant-only shadow book's "
          "final probability; 'llm_pool' the pooled LLM analysts before the overlay.")


def _print_paired_ic(records: list, as_of: str, cfg: FirmConfig, include_pre_cutoff: bool) -> None:
    """Paired per-date IC lines for the comparisons the firm acts on: the live final
    probability vs the shadow book, and the LLM pool vs the quant (the adaptive overlay
    bound's evidence). Printed only when both agents have records."""
    agents = {r.agent for r in records}
    lags = overlap_lags(cfg.process.horizon_days, cfg.process.rebalance_every_days)
    for a, b, label in ((COMMITTEE_AGENT, SHADOW_AGENT, "live final P vs quant-only shadow"),
                        (LLM_POOL_AGENT, QUANT_AGENT, "LLM pool vs quant (overlay bound evidence)")):
        if a not in agents or b not in agents:
            continue
        res = paired_rank_ic(records, as_of, a, b, include_pre_cutoff=include_pre_cutoff,
                             min_names=cfg.evaluation.ic_min_names, nw_lags=lags)
        if res.n_dates == 0:
            print(f"paired IC, {label}: no resolved date with both yet")
            continue
        t = f"{res.t_stat:.2f}" if math.isfinite(res.t_stat) else "n/a"
        print(f"paired IC, {label}: mean difference {res.mean_diff:+.3f} (t = {t}) over {res.n_dates} date(s); "
              f"mean IC {res.mean_a:+.3f} vs {res.mean_b:+.3f}")


def _resolve_due(store: ForecastStore, as_of: str) -> None:
    """Resolve every due forecast in ``store`` with prices through ``as_of`` (downloaded)."""
    pending = [r for r in store.all() if not r.resolved and r.resolve_on <= as_of]
    if not pending:
        return
    tickers = sorted({r.ticker for r in pending} | {r.benchmark for r in pending})
    start = pd.Timestamp(min(r.as_of for r in pending)) - pd.Timedelta(days=10)
    newly = store.resolve(load_prices(tickers, start, as_of), as_of)
    print(f"resolved {len(newly)} of {len(pending)} due forecast(s) with prices through {as_of}")


# --------------------------------------------------------------------------- commands


def cmd_factsheet(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config).model_copy(deep=True)
    ticker = args.ticker.upper()
    if ticker == cfg.universe.benchmark.upper():
        raise CLIError(f"{ticker} is the benchmark; fact sheets are built for stocks measured against it")
    cfg.universe.tickers = [ticker]
    cfg.universe.screen.enabled = False          # one sheet: do not build the whole screened universe
    as_of = _completed_session(args.as_of)
    _, sheets = gather_data(cfg, as_of, live=args.as_of is None)   # headlines only for "now"
    if ticker not in sheets:
        raise CLIError(f"no price data for {ticker} as of {as_of}")
    print(sheets[ticker].render())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _apply_flags(_load_config(args.config), args)
    offline = bool(args.offline)
    book = _book_dir(cfg, offline)
    runs_root = cfg.runs_path()
    # Dates first: a session that is not complete, or one this book has already decided,
    # is refused before anything is loaded, called or written.
    requested = _completed_session(args.as_of)
    last_decided = last_decision(book)
    forced = _refuse_redecision(book, requested, last_decided, args.force)
    print(f"book: {book} (kill switch: {runs_root / cfg.risk.kill_switch_file})")
    if offline:
        llm: LLM | None = make_offline_llm()
        print("OFFLINE MODE: deterministic rules stand in for every LLM. Not AI; do not act on the result.")
    else:
        llm = _claude(cfg, cfg.process.max_llm_cost_usd_per_run) if needs_llm(cfg) else None
    if cfg.execution.broker not in ("paper", "alpaca"):
        raise CLIError(f"unknown broker {cfg.execution.broker!r}; use 'paper' or 'alpaca'")
    use_alpaca = cfg.execution.broker == "alpaca" and not offline
    if offline and cfg.execution.broker == "alpaca":
        print("note: offline mode always uses the paper broker")

    book.mkdir(parents=True, exist_ok=True)
    as_of = requested
    ledger = Ledger(book / LEDGER_FILE)
    if use_alpaca:
        broker: PaperBroker | AlpacaBroker = AlpacaBroker(require_approval=cfg.execution.require_approval)
    else:
        broker = PaperBroker(cfg.execution.initial_cash, cfg.costs, state_path=str(book / BROKER_FILE))
        broker.load()
    # Prices for everything the book touches: a name dropped from the universe but still held
    # must be priced, or it could never be sold.
    queued = pending_tickers(book)
    tickers = {*universe(cfg), cfg.universe.benchmark.upper(), *broker.positions(), *queued}
    start = pd.Timestamp(as_of) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
    prices = load_prices(sorted(tickers), start, as_of).as_of(as_of)
    # A holiday or weekend maps to the previous trading day, which may already be decided:
    # check the session the prices actually end on, before any queued order is filled.
    if len(prices.close.index):
        forced = _refuse_redecision(book, str(prices.close.index[-1].date()), last_decided, args.force) or forced
    prev_nav = None
    if isinstance(broker, PaperBroker):         # dividends and queued fills since the last run
        prev_nav = settle_paper_book(broker, book, prices, ledger, last_decided)
    # The session whose close we decide on (a holiday or weekend -> the previous session),
    # resolved before any fact sheet is built: filings, fundamentals and macro are loaded as
    # of that close, never as of the requested calendar day (P3).
    if len(prices.close.index):
        as_of = str(prices.close.index[-1].date())
    prices, sheets = gather_data(cfg, as_of, live=args.as_of is None, prices=prices)
    as_of = str(prices.close.index[-1].date())
    forced = _refuse_redecision(book, as_of, last_decided, args.force) or forced   # backstop
    run_id = f"run-{as_of}-{secrets.token_hex(3)}"
    if forced:
        ledger.append("decision_date_forced", {"run_id": run_id, "as_of": as_of, "requested_as_of": requested,
                                               "last_decision": last_decided, "approved_by": "--force"})
        print(f"warning: --force: deciding at the close of {as_of} although this book already decided at the "
              f"close of {last_decided} (recorded in the ledger)")
    mk = marks(prices)
    positions = broker.positions()
    if not isinstance(broker, PaperBroker):
        prev_marks = marks(prices, -2)
        prev_nav = broker.nav(prev_marks) if prev_marks else None
    nav = broker.nav(mk)
    current = {t: q * mk[t] / nav for t, q in positions.items() if t in mk}
    state_path = book / "risk_state.json"
    state = load_risk_state(state_path, nav)
    # The live book as it stands before its decision: a new shadow book is forked from it.
    snapshot = BookSnapshot(as_of=as_of, cash=float(broker.cash()), positions=dict(positions), nav=nav,
                            risk_state=replace(state), marks=dict(mk), prev_nav=prev_nav)

    with ForecastStore(book / FORECASTS_FILE) as store:
        decision = decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=nav, prev_nav=prev_nav,
                          risk_state=state, forecast_store=store, lesson_store=LessonStore(book / "lessons.jsonl"),
                          ledger=ledger, run_id=run_id, positions=positions, runs_dir=str(runs_root))
    write_json(state_path, asdict(decision.risk_state))
    _execute(decision, broker, book, ledger, args.yes)

    # The quant-only shadow book, strictly after the live decision is persisted and queued.
    outcome, comparison = _run_shadow_book(cfg, book, runs_root, as_of, prices, sheets, run_id, snapshot, ledger,
                                           offline or llm is None)

    report = book / "reports" / f"decision_{as_of}_{run_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_decision_report(decision, shadow=outcome, comparison=comparison))
    if outcome is not None and outcome.decision is not None:
        sreport = Path(outcome.book) / "reports" / f"decision_{as_of}_{outcome.run_id}.md"
        sreport.parent.mkdir(parents=True, exist_ok=True)
        sreport.write_text(render_decision_report(outcome.decision))
    append_anchor(book, ledger, run_id, as_of)
    print(f"decision {run_id} at the close of {as_of}: {len(decision.orders)} order(s), "
          f"LLM cost ${decision.usage.get('cost_usd', 0.0):.2f}")
    _print_decision_messages(decision)
    _print_shadow(outcome, comparison)
    print(f"report: {report}")
    return 0


def _run_shadow_book(cfg: FirmConfig, book: Path, runs_root: Path, as_of: str, prices: PriceData, sheets: dict,
                     run_id: str, snapshot: BookSnapshot, ledger: Ledger,
                     include_pre_cutoff: bool) -> tuple[ShadowOutcome | None, BookComparison | None]:
    """Decide the shadow book and compare it with the live book. Never raises: every
    problem is a warning (the live decision is already made and must stand)."""
    wanted, why = shadow_applies(cfg)
    if not wanted:
        print(f"shadow book: off ({why})")
        return None, None
    outcome = run_shadow(cfg, book=book, runs_root=runs_root, as_of=as_of, prices=prices, sheets=sheets,
                         live_run_id=run_id, live_start=snapshot, live_forecasts=book / FORECASTS_FILE,
                         price_loader=lambda tickers, start, end: load_prices(tickers, start, end))
    try:
        ledger.append("shadow", {"run_id": run_id, "as_of": as_of, **outcome.to_dict()})
    except Exception as exc:  # noqa: BLE001 - recording the shadow must not undo the live run
        print(f"warning: shadow book result could not be written to the live ledger ({type(exc).__name__}: {exc})")
    comparison = None
    try:
        with ForecastStore(book / FORECASTS_FILE) as store:
            records = store.all()
        comparison = compare_books(cfg, book, records, as_of=as_of, include_pre_cutoff=include_pre_cutoff)
    except Exception as exc:  # noqa: BLE001 - the comparison is a report, never a reason to fail the run
        print(f"warning: live-vs-shadow comparison failed ({type(exc).__name__}: {exc})")
    return outcome, comparison


def _print_shadow(outcome: ShadowOutcome | None, comparison: BookComparison | None) -> None:
    if outcome is None:
        return
    if outcome.status == "failed":
        print(f"warning: shadow book FAILED (the live decision above is unaffected): {outcome.reason}")
    elif outcome.status == "skipped":
        print(f"shadow book: skipped ({outcome.reason})")
    else:
        born = " (started today from the live book's state)" if outcome.inception else ""
        print(f"shadow book{born}: quant-only decision {outcome.run_id}, {outcome.n_orders} paper order(s); "
              f"{outcome.book}")
        for w in outcome.warnings:
            print(f"shadow warning: {w}")
    if comparison is not None:
        for line in comparison.lines():
            print(f"compare: {line}")


def _print_decision_messages(decision: Decision) -> None:
    """Everything an operator must see without opening the report: pipeline warnings,
    every rule the deterministic risk engine applied (kill switch, drawdown freeze or
    de-risk, daily-loss halt, a human unfreeze, caps) and orders it blocked or capped."""
    for w in decision.warnings:
        print(f"warning: {w}")
    for m in decision.risk_messages:
        print(f"risk: {m}")
    for m in decision.order_messages:
        print(f"orders: {m}")
    if decision.screen is not None:
        sc = decision.screen
        deep = ", ".join(sc.deep_dive) or "none"
        print(f"screen: {sc.n_universe} names, {sc.n_eligible} eligible; deep dive: {deep}")
    ex = decision.exposure()
    print(f"exposure: {int(ex['n_names'])} name(s), active gross {ex['active_gross']:.1%}, core {ex['core']:.1%}, "
          f"cash {ex['cash']:.1%}")
    b = decision.overlay_bound
    if b is not None and b.max_tilt < b.configured:
        print(f"overlay: bound {b.max_tilt:.3f} (configured {b.configured:.3f}): {b.reason}")


def _execute(decision: Decision, broker: PaperBroker | AlpacaBroker, book: Path, ledger: Ledger, yes: bool) -> None:
    """Paper: queue the orders for the next open. Alpaca: ask a human, then send."""
    if not decision.orders:
        return
    if isinstance(broker, PaperBroker):
        n = queue_paper_orders(decision, book)
        print(f"{n} paper order(s) queued; they fill at the next trading day's open on the next run")
        return
    for o in decision.orders:
        print(f"  {o.side:<4} {o.quantity:>12.4f} {o.ticker:<6} ~${o.notional:>12,.2f}  {o.reason}")
    approved = yes or not broker.require_approval or _confirm(f"Send {len(decision.orders)} order(s) to Alpaca?")
    if not approved:
        ledger.append("orders_not_approved", {"run_id": decision.run_id, "as_of": decision.as_of,
                                              "n_orders": len(decision.orders)})
        print("orders NOT sent (not approved)")
        return
    fills = broker.submit(decision.orders, decision.market.prices, decision.as_of, approve=True)
    ledger.append("broker_submit", {"run_id": decision.run_id, "as_of": decision.as_of, "broker": "alpaca",
                                    "approved_by": "--yes" if yes else "interactive", "fills": fills,
                                    "responses": broker.last_orders})
    print(f"sent {len(decision.orders)} order(s) to Alpaca; {len(fills)} reported filled so far")


def cmd_compare(args: argparse.Namespace) -> int:
    """Live book vs its quant-only shadow (see :mod:`aifirm.shadow`)."""
    cfg = _load_config(args.config)
    as_of = _completed_session(args.as_of)
    book = _book_dir(cfg, args.offline)
    if not (book / LEDGER_FILE).exists():
        print(f"no book at {book} yet (run `aifirm run` first)")
        return 0
    path = book / FORECASTS_FILE
    records = []
    if path.exists():
        with ForecastStore(path) as store:
            if not args.no_resolve:
                _resolve_due(store, as_of)
            records = store.all()
    include_pre = bool(args.offline) or not needs_llm(cfg)
    comp = compare_books(cfg, book, records, as_of=as_of, include_pre_cutoff=include_pre)
    print(f"book: {book}\nshadow: {shadow_book_dir(book)}")
    if args.offline:
        print("OFFLINE BOOK: its 'AI' is deterministic rules, so this measures the plumbing, not a language model.")
    for line in comp.lines():
        print(line)
    if args.json:
        print(json.dumps(comp.to_dict(), indent=2, default=str))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    """Rank the universe with the deterministic quant and show the deep-dive set (no LLM, no orders)."""
    cfg = _load_config(args.config)
    as_of = _completed_session(args.as_of)
    book = _book_dir(cfg, args.offline)
    sc = cfg.universe.screen
    if not sc.enabled:
        print("note: universe.screen is off in this config, so every universe name would go to the committee; "
              "ranking universe.tickers anyway")
    top_n = args.top if args.top is not None else sc.deep_dive_top_n
    if top_n < 1:
        raise CLIError("--top must be at least 1")
    bench = cfg.universe.benchmark.upper()
    held: set[str] = set()
    if (book / BROKER_FILE).exists():
        paper = PaperBroker(cfg.execution.initial_cash, cfg.costs, state_path=str(book / BROKER_FILE))
        paper.load()
        held = {t for t in paper.positions() if t != bench}
    names = universe(cfg)
    start = pd.Timestamp(as_of) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
    prices = load_prices(sorted({*names, bench, *held}), start, as_of).as_of(as_of)
    prices, sheets = gather_data(cfg, as_of, live=False, prices=prices)
    session = str(prices.close.index[-1].date())
    in_universe = {t: sheets[t] for t in names if t in sheets}
    quant = quant_views(list(in_universe.values()), cfg.process.horizon_days)
    result = screen_universe(in_universe, quant, prices, session, top_n=top_n,
                             min_dollar_volume=sc.min_dollar_volume, min_price=cfg.risk.min_price, held=held)
    print(f"screen at the close of {session}: {result.n_universe} names with prices, {result.n_eligible} eligible "
          f"(price >= {cfg.risk.min_price:g}, 20-day dollar volume >= ${sc.min_dollar_volume:,.0f}); deep dive = top "
          f"{top_n} with quant P > 0.5 plus held names ({len(held)} held in {book})")
    print(f"{'rank':>4} {'ticker':<7} {'quant P':>8} {'$ volume 20d':>15} {'price':>9}  deep dive / reason")
    shown = result.entries if args.rows is None else result.entries[:args.rows]
    for e in shown:
        rank = str(e.rank) if e.rank is not None else "-"
        prob = f"{e.quant_prob:.3f}" if e.quant_prob is not None else "-"
        vol = f"${e.adv_usd:,.0f}" if e.adv_usd is not None else "-"
        px = f"{e.last_price:,.2f}" if e.last_price is not None else "-"
        print(f"{rank:>4} {e.ticker:<7} {prob:>8} {vol:>15} {px:>9}  {e.deep_dive_reason or e.reason or '-'}")
    if len(shown) < len(result.entries):
        print(f"... {len(result.entries) - len(shown)} more (use --rows to show them)")
    missing = [t for t in names if t not in sheets]
    if missing:
        print(f"no prices (no fact sheet): {', '.join(missing)}")
    print(f"deep-dive set ({len(result.deep_dive)}): {', '.join(result.deep_dive) or 'none'}")
    if args.record:
        ledger = Ledger(book / LEDGER_FILE)
        ledger.append("screen_preview", {"as_of": session, "top_n": top_n, "held": sorted(held),
                                         "screen": result.to_dict()})
        append_anchor(book, ledger, None, session)
        print(f"recorded in {book / LEDGER_FILE} (event screen_preview; not a decision)")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _apply_flags(_load_config(args.config), args)
    end = _completed_session(args.end, "--end")
    llm = None
    if args.mode == "llm":
        problem = credentials_problem(            # before the budget prompt: answering it must not be wasted
            "use --mode quant (no model) or --mode offline (the plumbing with deterministic rules, not AI)")
        if problem:
            raise CLIError(problem)
        cycles = len(pd.bdate_range(args.start, end)) // max(1, cfg.process.rebalance_every_days) + 1
        budget = cfg.process.max_llm_cost_usd_per_run * cycles
        print(f"LLM backtest: about {cycles} decision cycles, LLM budget up to ${budget:,.2f} "
              f"(max_llm_cost_usd_per_run x cycles). Decisions on or before the model's training cutoff plus "
              f"{cfg.evaluation.cutoff_buffer_days} days are contaminated and are reported separately.")
        if not args.yes and not _confirm("Continue?"):
            print("cancelled")
            return 1
        llm = _claude(cfg, budget)
    print(f"backtest {args.mode} {args.start} to {end}: progress lines go to stderr (a 100-name screened run takes "
          "tens of minutes)", file=sys.stderr, flush=True)
    result = run_backtest(cfg, args.start, end, args.mode, llm=llm,
                          progress=lambda line: print(f"  {line}", file=sys.stderr, flush=True))
    report = Path(result.run_dir) / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_backtest_report(result))
    if result.universe_note:
        print(f"SURVIVORSHIP: {result.universe_note}")
    for i, line in enumerate(six_lines(result), start=1):
        print(f"{i}. {line.replace('**', '')}")
    print(f"{'':<14} {'total':>8} {'sharpe':>7} {'max dd':>8} {'turnover':>9}")
    for name, rep in result.reports.items():
        turn = f"{rep.avg_turnover_pct:.1f}%" if rep.avg_turnover_pct is not None else "-"
        print(f"{name:<14} {rep.total_return_pct:>7.1f}% {rep.sharpe:>7.2f} {rep.max_drawdown_pct:>7.1f}% {turn:>9}")
    s = result.reports["strategy"]
    dsr = f"{s.deflated_sharpe:.2f}" if s.deflated_sharpe is not None else "n/a (1 trial)"
    print(f"deflated Sharpe: {dsr} over {result.n_trials} recorded configuration(s); PSR {s.psr:.2f}")
    ex = result.exposure_summary()
    if math.isfinite(ex["invested"]):
        print(f"exposure: {ex['invested']:.1%} invested, {ex['cash']:.1%} cash on average")
    ov = result.overlay_value()
    if ov is not None:
        def signed(x: float) -> str:
            return f"{'-' if x < 0 else '+'}${abs(x):,.2f}"
        print(f"LLM value added vs the quant-only baseline, net of LLM cost: {signed(ov.value_added_usd)} "
              f"(NAV difference {signed(ov.difference_usd)}, LLM cost ${ov.llm_cost_usd:,.2f})")
    if result.model_cutoff is not None:
        print(f"model training cutoff {result.model_cutoff} (+{result.cutoff_buffer_days} days: contaminated through "
              f"{result.contamination_end}): pre-cutoff results are not evidence of skill (see report)")
    print(f"report: {report}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    as_of = _completed_session(args.as_of)
    path = Path(args.db) if args.db else _book_dir(cfg, args.offline, args.shadow) / FORECASTS_FILE
    if not path.exists():
        print(f"no forecasts yet ({path} does not exist)")
        return 0
    with ForecastStore(path) as store:
        _resolve_due(store, as_of)
        records = store.all()
    print(f"forecasts: {len(records)} total, {sum(r.resolved for r in records)} resolved")
    _print_scorecard(scorecard(records, as_of=as_of))
    _print_paired_ic(records, as_of, cfg, include_pre_cutoff=bool(args.offline) or not needs_llm(cfg))
    return 0


def _reid_sample(tickers: list[str], prices: PriceData, start: str, end: str, step: int, n: int,
                 seed: int) -> list[tuple[str, str]]:
    """``n`` (decision date, ticker) probes drawn without replacement, reproducibly from
    ``seed``, from every rebalance date (every ``step`` trading days from ``start``) and
    every ticker with a close on that date; sorted by date then ticker."""
    idx = prices.close.index
    days = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))][::max(1, step)]
    pool = [(str(d.date()), t) for d in days for t in tickers
            if t in prices.close.columns and pd.notna(prices.close.at[d, t])]
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return sorted(pool[i] for i in pick)


def cmd_reid_probe(args: argparse.Namespace) -> int:
    """Re-identification gate on the exact blind payload (RESEARCH 7.4 S2, 9.3 item 2)."""
    cfg = _load_config(args.config)
    end = _completed_session(args.end, "--end")
    try:
        start = str(date.fromisoformat(args.start.strip()))
    except ValueError as exc:
        raise CLIError(f"--start {args.start!r} is not a date; use YYYY-MM-DD") from exc
    if start >= end:
        raise CLIError(f"--start {start} must be before --end {end}")
    if args.n < 1:
        raise CLIError("--n must be at least 1")
    model = args.model or cfg.models.committee_model
    if args.offline:
        llm: LLM = make_offline_llm()
        print("OFFLINE: a seeded random guesser stands in for the model. This tests the harness and shows chance "
              "levels; it says nothing about what a real model can re-identify.")
    else:
        problem = credentials_problem()
        if problem:
            raise CLIError(f"{problem} (reid-probe sends {args.n} calls to the model; --offline runs the harness "
                           "with a random guesser)")
        budget = args.budget if args.budget is not None else cfg.process.max_llm_cost_usd_per_run
        print(f"re-identification probe: up to {args.n} calls to {model}, LLM budget ${budget:,.2f}")
        if not args.yes and not _confirm("Continue?"):
            print("cancelled")
            return 1
        llm = ClaudeLLM(budget_usd=budget, use_fallbacks=cfg.models.use_server_fallbacks)
    names = universe(cfg)
    bench = cfg.universe.benchmark.upper()
    prices = load_prices([*names, bench], pd.Timestamp(start) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS), end)
    probes = _reid_sample(names, prices, start, end, cfg.process.rebalance_every_days, args.n, args.seed)
    if not probes:
        raise CLIError(f"no (date, ticker) with prices between {start} and {end}")
    sheets = []
    by_date: dict[str, list[str]] = {}
    for d, t in probes:
        by_date.setdefault(d, []).append(t)
    for d, ts in sorted(by_date.items()):
        one = cfg.model_copy(deep=True)
        one.universe.tickers = list(ts)
        one.universe.screen.enabled = False
        _, built = gather_data(one, d, live=False, prices=prices)    # historical: no headlines
        sheets += [built[t] for t in ts if t in built]
    print(f"{len(sheets)} probes over {len(by_date)} dates, {len(names)} candidate tickers, months "
          f"{start[:7]} to {end[:7]}" + (f", {args.role} slice" if args.role else ", full blind sheet"))
    rep = run_reid_probe(llm, sheets, model=model, role=args.role, candidate_tickers=names,
                         month_window=(start[:7], end[:7]), max_parallel=cfg.models.max_parallel_calls)
    print(rep.summary())
    out_dir = cfg.runs_path() / REID_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"reid_{stamp}_{'offline' if args.offline else model}.json"
    write_json(path, {"offline": bool(args.offline), "start": start, "end": end, "seed": args.seed,
                      "report": rep.to_dict()})
    summary = {k: v for k, v in rep.to_dict().items() if k != "probes"}
    ledger = Ledger(out_dir / LEDGER_FILE)
    ledger.append("reid_probe", {"offline": bool(args.offline), "start": start, "end": end, "seed": args.seed,
                                 "report_path": str(path), **summary})
    print(f"report: {path}")
    return 0 if rep.gate_passed else 1


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    path = Path(args.path) if args.path else _book_dir(cfg, args.offline, args.shadow) / LEDGER_FILE
    if not path.exists():
        raise CLIError(f"no ledger at {path}")
    anchor = args.anchor
    anchors = path.parent / "ledger_anchors.jsonl"
    if anchor is None and anchors.exists():
        lines = [x for x in anchors.read_text().splitlines() if x.strip()]
        if lines:
            anchor = json.loads(lines[-1])["head"]
    try:
        audit = Ledger(path).audit(anchor)
    except LedgerError as exc:
        print(f"LEDGER DAMAGED: {exc}")
        return 1
    if audit.ok:
        print(f"OK: {audit.checked} entries verified" + (f"; anchor {anchor[:16]}... found" if anchor else ""))
        return 0
    print(f"LEDGER FAILED VERIFICATION after {audit.checked} good entries: {audit.problem}")
    return 1


def cmd_unfreeze(args: argparse.Namespace) -> int:
    """The human reset of a drawdown freeze (ARCHITECTURE.md section 4).

    A frozen book holds only cash and the passive core, so it can never regain its old
    peak: clearing ``frozen`` alone would re-freeze it on the next run. The reset therefore
    also restarts the high-water mark at the NAV recorded at the book's last decision (the
    latest NAV the book has measured; the next run rebuilds the NAV path from that decision
    and measures any later loss from there, so a drop since then is not forgiven). The
    before/after state, the reason and who approved go to the ledger before the state file
    changes; if writing the ledger fails, the book stays frozen. ``--shadow`` resets the
    shadow book (its own drawdown rules; the same procedure)."""
    cfg = _load_config(args.config)
    book = _book_dir(cfg, args.offline, args.shadow)
    state_path = book / "risk_state.json"
    if not state_path.exists():
        raise CLIError(f"no risk state at {state_path}: this book has not run yet, so it is not frozen")
    state = load_risk_state(state_path, math.nan)
    if not state.frozen:
        print(f"the book at {book} is not frozen; nothing changed")
        return 0
    reason = (args.reason or "").strip()
    if not reason:
        raise CLIError("--reason is required: say what you reviewed and why the book may trade again "
                       "(it is written to the ledger)")
    nav = state.last_nav
    if not (isinstance(nav, (int, float)) and math.isfinite(nav) and nav > 0):
        raise CLIError(f"{state_path} records no valid NAV (last_nav={nav!r}), so the high-water mark cannot be "
                       "restarted. Run one cycle first (a frozen book still runs and records its NAV), then unfreeze.")
    nav = float(nav)
    peak = state.peak_nav
    below = (f" ({1.0 - nav / peak:.1%} below the peak of {peak:,.2f})"
             if isinstance(peak, (int, float)) and math.isfinite(peak) and peak > 0 else f" (peak {peak!r})")
    print(f"book: {book}")
    print(f"FROZEN. NAV at the last decision ({state.last_as_of or 'date unknown'}): {nav:,.2f}{below}.")
    print(f"Unfreezing lets the active sleeve trade again and restarts the high-water mark at {nav:,.2f}.")
    if not args.yes and not _confirm("Unfreeze this book?"):
        print("cancelled: the book stays frozen")
        return 1
    new = replace(state, frozen=False, peak_nav=nav, frozen_at_nav=None, trough_nav=nav, derisked=False)
    ledger = Ledger(book / LEDGER_FILE)
    ledger.append("risk_unfrozen", {"as_of": state.last_as_of, "book": str(book), "reason": reason,
                                    "approved_by": "--yes" if args.yes else "interactive",
                                    "before": asdict(state), "after": asdict(new)})
    write_json(state_path, asdict(new))
    append_anchor(book, ledger, None, state.last_as_of)
    print(f"unfrozen: high-water mark {nav:,.2f}; recorded in {book / LEDGER_FILE}")
    return 0


# --------------------------------------------------------------------------- parser


def _strategy_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("strategy overrides (each changes the config fingerprint: a new trial)")
    g.add_argument("--preset", metavar="NAME", help="screen the bundled universe list NAME (e.g. sp100; current members, so "
                                    "survivorship-biased) and turn the screen on")
    g.add_argument("--universe-file", metavar="PATH", help="screen the tickers in PATH (one per line, '#' comments; "
                                                           "relative to the shell's directory) and turn the screen on")
    g.add_argument("--deep-dive", type=int, metavar="N",
                   help="names sent to the LLM committee per cycle (top N by quant P above 0.5, plus held names); "
                        "turns the screen on")
    g.add_argument("--min-dollar-volume", type=float, metavar="USD", help="screen liquidity floor (20-day average "
                                                                          "dollar volume); turns the screen on")
    g.add_argument("--combination", choices=["overlay", "pool"], help="overlay: LLM pool tilts the quant core within "
                                                                      "a bound; pool: every analyst is one pool member")
    g.add_argument("--max-tilt", type=float, metavar="X", help="largest move in probability the LLM pool may apply to "
                                                               "the quant core (overlay mode; e.g. 0.05)")
    g.add_argument("--no-calibration", action="store_true", help="pool raw LLM probabilities (no per-agent Platt map)")
    g.add_argument("--no-news-reader", action="store_true", help="withhold headlines instead of reading them into "
                                                                 "tier-labelled events")
    g.add_argument("--no-tier-gate", action="store_true", help="let bullish views rest on unverified (tier C) news")
    g.add_argument("--rebalance-fraction", type=float, metavar="F", help="move this fraction of the way to target "
                                                                         "per rebalance, in (0, 1]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aifirm", description="A research-grade multi-agent AI investment firm "
                                     "(paper trading by default; not investment advice).")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks for errors")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("factsheet", help="print the point-in-time fact sheet for one ticker")
    p.add_argument("ticker")
    p.add_argument("--as-of", help="decision date YYYY-MM-DD (default: the last completed session)")
    p.add_argument("--config", help="firm config TOML (for the benchmark); default: built-in defaults")
    p.set_defaults(func=cmd_factsheet)

    p = sub.add_parser("run", help="one decision cycle with the paper broker (or Alpaca, with approval), plus the "
                                   "quant-only shadow book")
    p.add_argument("--config", required=True, help="firm config TOML")
    p.add_argument("--as-of", help="decision date YYYY-MM-DD (default: the last completed session)")
    p.add_argument("--offline", action="store_true", help="deterministic offline rules instead of Claude (not AI)")
    p.add_argument("--yes", action="store_true", help="approve orders for a non-paper broker without asking")
    p.add_argument("--force", action="store_true",
                   help="decide even if this book already decided on this session or a later one "
                        "(duplicate forecasts or backdated orders; recorded in the ledger)")
    p.add_argument("--no-shadow", action="store_true",
                   help="do not decide the quant-only shadow book this run (evaluation.shadow_book = false does it "
                        "for every run; the comparison then has a gap)")
    _strategy_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="live book vs its quant-only shadow: NAV, return, LLM spend, AI value added "
                                       "net of LLM cost, paired per-date IC with a verdict")
    p.add_argument("--config", help="firm config TOML (default: built-in defaults)")
    p.add_argument("--offline", action="store_true", help="compare the offline book with its shadow")
    p.add_argument("--as-of", help="count decisions and resolved forecasts up to this date (default: the last "
                                   "completed session)")
    p.add_argument("--no-resolve", action="store_true", help="do not download prices to resolve due forecasts first")
    p.add_argument("--json", action="store_true", help="also print the comparison as JSON")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("screen", help="rank the universe with the deterministic quant and print the deep-dive set "
                                      "(no LLM, nothing traded)")
    p.add_argument("--config", help="firm config TOML (default: built-in defaults)")
    p.add_argument("--as-of", help="screen at this close YYYY-MM-DD (default: the last completed session)")
    p.add_argument("--top", type=int, help="deep-dive size (default: universe.screen.deep_dive_top_n)")
    p.add_argument("--offline", action="store_true", help="take held names from the offline book")
    p.add_argument("--rows", type=int, help="print only the first N ranking rows (default: all)")
    p.add_argument("--record", action="store_true", help="append the ranking to the book's ledger as a "
                                                         "screen_preview entry (not a decision)")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("backtest", help="walk-forward backtest with baselines, a model-cutoff split and the "
                                        "evaluation (IC, power, random books, costs, factors)")
    p.add_argument("--config", required=True, help="firm config TOML")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="last date YYYY-MM-DD (at most the last completed session)")
    p.add_argument("--mode", choices=["quant", "offline", "llm"], default="quant")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation in llm mode")
    g = p.add_argument_group("evaluation overrides (change what the report measures, never the fingerprint)")
    g.add_argument("--random-books", type=int, metavar="N", help="composition-matched random books (0 = off)")
    g.add_argument("--seed", type=int, help="seed of the random books")
    g.add_argument("--sector-matched-books", action="store_true", help="draw random replacements from the same sector")
    g.add_argument("--cost-multipliers", metavar="K,K,...", help="re-execute the same decisions with every cost "
                                                                "component times each k, e.g. 0,1,2")
    g.add_argument("--breakeven-cap", type=float, metavar="X",
                   help="largest cost multiplier searched for the breakeven")
    g.add_argument("--cutoff-buffer-days", type=int, metavar="N", help="days after the model's training cutoff still "
                                                                       "counted as contaminated (also which LLM forecasts "
                                                                       "count as clean track record, so a non-default "
                                                                       "value changes the fingerprint)")
    g.add_argument("--ic-min-names", type=int, metavar="N", help="fewest names for one date's rank IC to count")
    g.add_argument("--no-factor-proxies", action="store_true", help="skip the ETF factor regression (no download)")
    _strategy_flags(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("score", help="resolve due forecasts and print the per-agent scorecard and paired ICs")
    p.add_argument("--config", help="firm config TOML (default: built-in defaults)")
    p.add_argument("--offline", action="store_true", help="score the offline book")
    p.add_argument("--shadow", action="store_true", help="score the shadow book's own forecast store")
    p.add_argument("--db", help="path to a forecasts.sqlite (for example inside a backtest run directory)")
    p.add_argument("--as-of", help="score as of this date (default: the last completed session)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("reid-probe", help="re-identification gate: can the model name the stock and month of a blind "
                                          "fact sheet? (needs credentials unless --offline)")
    p.add_argument("--config", required=True, help="firm config TOML")
    p.add_argument("--start", required=True, help="first decision date YYYY-MM-DD of the probe window")
    p.add_argument("--end", required=True, help="last decision date (at most the last completed session)")
    p.add_argument("--n", type=int, default=400, help="probes to draw (the gate needs 400 answered; default 400)")
    p.add_argument("--role", choices=sorted(ROLES), help="probe one analyst's blind slice (default: the full blind "
                                                         "sheet, the conservative choice)")
    p.add_argument("--model", help="model to probe (default: models.committee_model)")
    p.add_argument("--seed", type=int, default=0, help="seed of the (date, ticker) sample")
    p.add_argument("--budget", type=float, help="LLM budget in USD (default: process.max_llm_cost_usd_per_run)")
    p.add_argument("--offline", action="store_true", help="a seeded random guesser instead of the model (tests the "
                                                          "harness; no credentials needed)")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    p.set_defaults(func=cmd_reid_probe)

    p = sub.add_parser("verify-ledger", help="verify the hash chain of a decision ledger")
    p.add_argument("--config", help="firm config TOML (default: built-in defaults)")
    p.add_argument("--offline", action="store_true", help="verify the offline book's ledger")
    p.add_argument("--shadow", action="store_true", help="verify the shadow book's ledger")
    p.add_argument("--path", help="ledger file (default: <runs_dir>/ledger.jsonl)")
    p.add_argument("--anchor", help="a head hash recorded earlier (default: the last one in ledger_anchors.jsonl)")
    p.set_defaults(func=cmd_verify_ledger)

    p = sub.add_parser("unfreeze", help="human reset of a drawdown freeze (restarts the high-water mark; ledgered)")
    p.add_argument("--config", required=True, help="firm config TOML")
    p.add_argument("--offline", action="store_true", help="unfreeze the offline book")
    p.add_argument("--shadow", action="store_true", help="unfreeze the shadow book")
    p.add_argument("--reason", required=True, help="what you reviewed and why the book may trade again (ledgered)")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_unfreeze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BookError as exc:          # CLIError included: a problem the user can fix
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (LedgerError, RuntimeError, PermissionError, ValueError, OSError) as exc:
        # Operational failures (network, data, a damaged ledger, a refused broker endpoint):
        # stop without trading and say why. --debug shows the traceback.
        if args.debug:
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
