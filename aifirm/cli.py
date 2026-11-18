"""Command-line interface: ``aifirm factsheet | run | backtest | score | verify-ledger | unfreeze``.

Safety defaults (P9, ARCHITECTURE.md section 4):

* ``run`` trades the **paper broker** unless the config says ``broker = "alpaca"``, and even
  then it asks a human to approve the orders (``--yes`` skips the question; use it only
  in a supervised schedule). ``--offline`` always uses the paper broker and keeps its
  state apart (``<runs_dir>/offline``), so heuristic trades never mix with a real book.
* Paper orders decided at a close are filled at the **next trading day's open** on the
  following run (P3), never at the price that produced the decision. A second run before
  that open refuses to decide again, so a plan cannot be doubled.
* Runs that call Claude check for credentials first and stop with a clear message
  instead of failing halfway (fail closed). ``ANTHROPIC_API_KEY`` is enough; an
  ``ant auth login`` profile, ``ANTHROPIC_AUTH_TOKEN`` or workload identity also work.
* Decisions use **completed sessions only**. ``--as-of`` (and ``backtest --end``,
  ``score --as-of``) may not be after :func:`last_completed_session`: today before the
  close settles, or a future date, would treat a partial intraday bar as a close (P3).
* A book **never decides twice on the same session or on an earlier one** (no duplicate
  forecasts, no backdated orders filled at historical prices behind later fills).
  ``run --force`` overrides that, and the override is written to the ledger.
* ``unfreeze`` is the human reset of a drawdown freeze: it asks for a reason and a
  confirmation, restarts the high-water mark at the NAV of the book's last decision and
  records the change in the ledger (ARCHITECTURE.md section 4).

Live-book layout under ``runs_dir`` (``runs_dir/offline`` with ``--offline``):
``ledger.jsonl``, ``ledger_anchors.jsonl``, ``forecasts.sqlite``, ``lessons.jsonl``,
``paper_broker.json``, ``risk_state.json``, ``pending_orders.json`` and ``reports/``.
The kill switch is always ``<runs_dir>/<risk.kill_switch_file>``. A relative ``runs_dir``
is relative to the config file's directory (:meth:`FirmConfig.runs_path`), never to the
shell's working directory, so a scheduled run from elsewhere finds the same book and the
same kill switch; every run prints the resolved path.
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

import pandas as pd
from pydantic import ValidationError

from .agents import make_offline_llm
from .backtest.engine import run_backtest
from .config import FirmConfig, describe_config_error
# The session clock lives with the price loader (every caller needs it); MARKET_TZ and
# last_completed_session stay importable from here.
from .data.prices import MARKET_TZ, SESSION_CLOSE_SETTLED, PriceData, last_completed_session, load_prices  # noqa: F401
from .execution.broker import AlpacaBroker, PaperBroker
from .journal import ForecastStore, Ledger, LedgerError, LessonStore, scorecard
from .llm.client import LLM, ClaudeLLM
from .pipeline import PRICE_LOOKBACK_DAYS, Decision, decide, gather_data, implied_dividends, needs_llm, universe
from .portfolio import RiskState
from .report import render_backtest_report, render_decision_report
from .schemas import Fill, Order

STALE_FILL_DAYS = 5


class CLIError(RuntimeError):
    """A problem the user can fix; printed without a traceback."""


# --------------------------------------------------------------------------- helpers


def credentials_problem() -> str | None:
    """None if the Anthropic SDK can find credentials, else a message saying how to fix it.

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
        return f"Anthropic credentials could not be loaded: {exc}"
    return ("No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login` to create a profile, "
            "or add --offline to use the deterministic offline rules (not AI).")


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
            raise CLIError(f"config file not found: {path}") from exc
        except ValidationError as exc:                   # unknown keys, values out of range
            raise CLIError(f"invalid config {path}:\n  {describe_config_error(exc)}") from exc
        except ValueError as exc:                        # TOMLDecodeError is a ValueError
            raise CLIError(f"invalid config {path}: {exc}") from exc
    return cfg.model_copy(update={"runs_dir": str(cfg.runs_path())})


def _book_dir(cfg: FirmConfig, offline: bool) -> Path:
    root = cfg.runs_path()
    return root / "offline" if offline else root


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


def _waiting_message(decided: str) -> str:
    return (f"orders decided at the close of {decided} are waiting for the next trading day's open; "
            "run again after that session (deciding again now would double the plan)")


def _pending_as_of(book: Path) -> str | None:
    path = book / "pending_orders.json"
    if not path.exists():
        return None
    return str(json.loads(path.read_text())["as_of"])[:10]


def _last_decision(book: Path) -> str | None:
    """The latest session this book has decided on, from every record that carries one:
    the ledger's completed cycles (``cycle_end``), ``risk_state.json`` (``last_as_of``)
    and queued orders. None for a new book."""
    dates: list[str] = []
    pending = _pending_as_of(book)
    if pending:
        dates.append(pending)
    state = book / "risk_state.json"
    if state.exists():
        try:
            last = json.loads(state.read_text()).get("last_as_of")
        except (json.JSONDecodeError, AttributeError):
            last = None                     # a corrupt file is reported by _load_risk_state
        if last:
            dates.append(str(last)[:10])
    ledger = book / "ledger.jsonl"
    if ledger.exists():
        with open(ledger, "rb") as fh:      # canonical JSON: the event key is a literal substring
            for raw in fh:
                if b'"event":"cycle_end"' not in raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue                # a damaged line is the ledger audit's business
                if entry.get("event") == "cycle_end" and entry.get("payload", {}).get("as_of"):
                    dates.append(str(entry["payload"]["as_of"])[:10])
    return max(dates) if dates else None


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
    pending = _pending_as_of(book)
    if pending is not None and as_of <= pending:
        raise CLIError(_waiting_message(pending))
    if force:
        return True
    what = ("would repeat that decision (its forecasts would be stored and scored twice)" if as_of == last else
            "would be backdated: its orders would fill at a historical open, with hindsight, behind later fills")
    raise CLIError(f"this book ({book}) already decided at the close of {last}; a decision at the close of {as_of} "
                   f"{what}. Decide on a later completed session, or pass --force to override "
                   "(the override is recorded in the ledger).")


def _append_anchor(book: Path, ledger: Ledger, run_id: str | None, as_of: str | None) -> None:
    with open(book / "ledger_anchors.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "as_of": as_of,
                             "head": ledger.head, "entries": len(ledger)}) + "\n")


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


def _marks(prices: PriceData, row: int = -1) -> dict[str, float]:
    """Last known close per ticker at ``row`` (forward-filled, so a missing print uses the prior close)."""
    closes = prices.close.ffill()
    if closes.empty or len(closes) < abs(row):
        return {}
    return {t: float(v) for t, v in closes.iloc[row].items() if math.isfinite(v) and v > 0}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _load_risk_state(path: Path, nav: float) -> RiskState:
    if not path.exists():
        return RiskState(peak_nav=nav)
    try:
        return RiskState(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as exc:
        raise CLIError(f"corrupt risk state {path}: {exc}. Fix or remove it by hand (a human decision).") from exc


def _last_fill_date(book: Path) -> str | None:
    """The latest paper fill date in the ledger (None if the book never filled)."""
    path = book / "ledger.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, "rb") as fh:
        for raw in fh:
            if b'"event":"fills"' not in raw:
                continue
            try:
                day = json.loads(raw).get("payload", {}).get("fill_date")
            except json.JSONDecodeError:
                continue
            if day and (last is None or str(day) > last):
                last = str(day)[:10]
    return last


def _settle_paper_book(broker: PaperBroker, book: Path, prices: PriceData, ledger: Ledger,
                       last_decided: str | None) -> float | None:
    """Bring the paper book from where the last run left it to the close of the decision
    session, day by day, as the backtest does (P3, P10). Returns the book's NAV at the
    previous session's close (None if unknown).

    Between runs a paper book changes only through (1) cash dividends and (2) the fill of
    the orders queued at the last decision. On each session after the last settled one
    (``PaperBroker.dividends_through``): first the dividends with that ex-date are credited
    to the shares held at the previous close (from ``implied_dividends``, as in backtests;
    otherwise the paper book would earn the price return only, while forecasts and the
    benchmark are measured on total return), then queued orders fill at that day's open if
    it is the first session after their decision. A book saved before dividends were
    tracked starts from its last fill in the ledger (its shares are unchanged since then).

    The previous close's NAV is taken during that replay, with the shares and cash the book
    really had then; computed after the fill it would mark today's shares at yesterday's
    prices (the daily-loss halt uses it when no NAV path is on record)."""
    idx = prices.close.index
    if len(idx) == 0:
        return None
    path = book / "pending_orders.json"
    pending = json.loads(path.read_text()) if path.exists() else None
    fill_day = None
    if pending is not None:
        decided = pending["as_of"]
        after = idx[idx > pd.Timestamp(decided)]
        if len(after) == 0:
            raise CLIError(_waiting_message(decided))
        fill_day = after[0]
    since = broker.dividends_through or _last_fill_date(book) or (pending or {}).get("as_of") or last_decided
    start = pd.Timestamp(since) if since else idx[-1]      # no history at all: nothing is owed
    if broker.positions() and since and start < idx[0]:
        print(f"note: dividends with ex-dates from {since} to {idx[0].date()} are outside the loaded prices "
              "and were not credited")
    dividends = implied_dividends(prices)
    closes = prices.close.ffill()
    prev_day = idx[-2] if len(idx) >= 2 else None
    prev_nav = None
    if prev_day is not None and prev_day <= start:          # nothing happened since that close
        prev_nav = broker.nav(_marks(prices, -2))
    credited: list[dict] = []
    for d in idx[idx > start]:
        row = dividends.loc[d]
        per_share = {t: float(row[t]) for t, q in broker.positions().items() if t in row.index and row[t] > 0}
        held = broker.positions()
        if per_share and broker.credit_dividends(per_share, str(d.date())):
            credited += [{"ex_date": str(d.date()), "ticker": t, "per_share": v, "shares": held[t],
                          "amount": held[t] * v} for t, v in sorted(per_share.items())]
        if fill_day is not None and d == fill_day:
            _fill_pending(broker, book, pending, prices, d, ledger)
        if d == prev_day:
            prev_nav = broker.nav({t: float(v) for t, v in closes.loc[d].items() if math.isfinite(v) and v > 0})
    through = str(idx[-1].date())
    broker.dividends_through = max(through, broker.dividends_through or through)
    broker.save()
    if credited:
        ledger.append("dividends", {"as_of": through, "credited": credited,
                                    "total": sum(c["amount"] for c in credited)})
    return prev_nav


def _fill_pending(broker: PaperBroker, book: Path, pending: dict, prices: PriceData, day: pd.Timestamp,
                  ledger: Ledger) -> list[Fill]:
    """Fill orders decided at an earlier close at the open of ``day``, the next trading day (P3)."""
    decided = pending["as_of"]
    orders = [Order.model_validate(o) for o in pending["orders"]]
    opens = (prices.open if prices.open is not None else prices.close).loc[day]
    fill_px = {o.ticker: float(opens[o.ticker]) for o in orders
               if o.ticker in opens.index and math.isfinite(opens[o.ticker]) and opens[o.ticker] > 0}
    fills = broker.submit(orders, fill_px, str(day.date()), pending.get("adv"), pending.get("daily_vol"))
    ledger.append("fills", {"run_id": pending.get("run_id"), "as_of": decided, "fill_date": str(day.date()),
                            "fills": fills, "rejections": broker.last_rejections})
    (book / "pending_orders.json").unlink()
    age = (prices.close.index[-1] - day).days
    if age > STALE_FILL_DAYS:
        print(f"note: pending orders from {decided} were filled at the open of {day.date()}, {age} days ago")
    return fills


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
          "Pre-cutoff forecasts may be contaminated by training data.")


# --------------------------------------------------------------------------- commands


def cmd_factsheet(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config).model_copy(deep=True)
    ticker = args.ticker.upper()
    if ticker == cfg.universe.benchmark.upper():
        raise CLIError(f"{ticker} is the benchmark; fact sheets are built for stocks measured against it")
    cfg.universe.tickers = [ticker]
    as_of = _completed_session(args.as_of)
    _, sheets = gather_data(cfg, as_of, live=args.as_of is None)   # headlines only for "now"
    if ticker not in sheets:
        raise CLIError(f"no price data for {ticker} as of {as_of}")
    print(sheets[ticker].render())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    offline = bool(args.offline)
    book = _book_dir(cfg, offline)
    runs_root = cfg.runs_path()
    # Dates first: a session that is not complete, or one this book has already decided,
    # is refused before anything is loaded, called or written.
    requested = _completed_session(args.as_of)
    last_decided = _last_decision(book)
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
    ledger = Ledger(book / "ledger.jsonl")
    if use_alpaca:
        broker: PaperBroker | AlpacaBroker = AlpacaBroker(require_approval=cfg.execution.require_approval)
    else:
        broker = PaperBroker(cfg.execution.initial_cash, cfg.costs, state_path=str(book / "paper_broker.json"))
        broker.load()
    # Prices for everything the book touches: a name dropped from the universe but still held
    # must be priced, or it could never be sold.
    pending = book / "pending_orders.json"
    queued = [o["ticker"] for o in json.loads(pending.read_text())["orders"]] if pending.exists() else []
    tickers = {*universe(cfg), cfg.universe.benchmark.upper(), *broker.positions(), *queued}
    start = pd.Timestamp(as_of) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
    prices = load_prices(sorted(tickers), start, as_of).as_of(as_of)
    # A holiday or weekend maps to the previous trading day, which may already be decided:
    # check the session the prices actually end on, before any queued order is filled.
    if len(prices.close.index):
        forced = _refuse_redecision(book, str(prices.close.index[-1].date()), last_decided, args.force) or forced
    prev_nav = None
    if isinstance(broker, PaperBroker):         # dividends and queued fills since the last run
        prev_nav = _settle_paper_book(broker, book, prices, ledger, last_decided)
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
    marks = _marks(prices)
    positions = broker.positions()
    if not isinstance(broker, PaperBroker):
        prev_marks = _marks(prices, -2)
        prev_nav = broker.nav(prev_marks) if prev_marks else None
    nav = broker.nav(marks)
    current = {t: q * marks[t] / nav for t, q in positions.items() if t in marks}
    state_path = book / "risk_state.json"
    state = _load_risk_state(state_path, nav)

    with ForecastStore(book / "forecasts.sqlite") as store:
        decision = decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=nav, prev_nav=prev_nav,
                          risk_state=state, forecast_store=store, lesson_store=LessonStore(book / "lessons.jsonl"),
                          ledger=ledger, run_id=run_id, positions=positions, runs_dir=str(runs_root))
    _write_json(state_path, asdict(decision.risk_state))
    _execute(decision, broker, book, ledger, args.yes)

    report = book / "reports" / f"decision_{as_of}_{run_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_decision_report(decision))
    _append_anchor(book, ledger, run_id, as_of)
    print(f"decision {run_id} at the close of {as_of}: {len(decision.orders)} order(s), "
          f"LLM cost ${decision.usage.get('cost_usd', 0.0):.2f}")
    _print_decision_messages(decision)
    print(f"report: {report}")
    return 0


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


def _execute(decision: Decision, broker: PaperBroker | AlpacaBroker, book: Path, ledger: Ledger, yes: bool) -> None:
    """Paper: queue the orders for the next open. Alpaca: ask a human, then send."""
    if not decision.orders:
        return
    if isinstance(broker, PaperBroker):
        _write_json(book / "pending_orders.json", {
            "run_id": decision.run_id, "as_of": decision.as_of, "orders": [o.model_dump() for o in decision.orders],
            "adv": decision.market.adv, "daily_vol": decision.market.daily_vol})
        print(f"{len(decision.orders)} paper order(s) queued; they fill at the next trading day's open on the next run")
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


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    end = _completed_session(args.end, "--end")
    llm = None
    if args.mode == "llm":
        cycles = len(pd.bdate_range(args.start, end)) // max(1, cfg.process.rebalance_every_days) + 1
        budget = cfg.process.max_llm_cost_usd_per_run * cycles
        print(f"LLM backtest: about {cycles} decision cycles, LLM budget up to ${budget:,.2f} "
              f"(max_llm_cost_usd_per_run x cycles). Decisions on or before the model's training cutoff are "
              "contaminated and are reported separately.")
        if not args.yes and not _confirm("Continue?"):
            print("cancelled")
            return 1
        llm = _claude(cfg, budget)
    result = run_backtest(cfg, args.start, end, args.mode, llm=llm)
    report = Path(result.run_dir) / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_backtest_report(result))
    print(f"{'':<14} {'total':>8} {'sharpe':>7} {'max dd':>8} {'turnover':>9}")
    for name, rep in result.reports.items():
        turn = f"{rep.avg_turnover_pct:.1f}%" if rep.avg_turnover_pct is not None else "-"
        print(f"{name:<14} {rep.total_return_pct:>7.1f}% {rep.sharpe:>7.2f} {rep.max_drawdown_pct:>7.1f}% {turn:>9}")
    s = result.reports["strategy"]
    dsr = f"{s.deflated_sharpe:.2f}" if s.deflated_sharpe is not None else "n/a (1 trial)"
    print(f"deflated Sharpe: {dsr} over {result.n_trials} recorded configuration(s); PSR {s.psr:.2f}")
    if result.model_cutoff is not None:
        print(f"model training cutoff {result.model_cutoff}: pre-cutoff results are contaminated (see report)")
    print(f"report: {report}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    as_of = _completed_session(args.as_of)
    path = Path(args.db) if args.db else _book_dir(cfg, args.offline) / "forecasts.sqlite"
    if not path.exists():
        print(f"no forecasts yet ({path} does not exist)")
        return 0
    with ForecastStore(path) as store:
        pending = [r for r in store.all() if not r.resolved and r.resolve_on <= as_of]
        if pending:
            tickers = sorted({r.ticker for r in pending} | {r.benchmark for r in pending})
            start = pd.Timestamp(min(r.as_of for r in pending)) - pd.Timedelta(days=10)
            newly = store.resolve(load_prices(tickers, start, as_of), as_of)
            print(f"resolved {len(newly)} of {len(pending)} due forecast(s) with prices through {as_of}")
        records = store.all()
    print(f"forecasts: {len(records)} total, {sum(r.resolved for r in records)} resolved")
    _print_scorecard(scorecard(records, as_of=as_of))
    return 0


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    path = Path(args.path) if args.path else _book_dir(cfg, args.offline) / "ledger.jsonl"
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
    changes; if writing the ledger fails, the book stays frozen."""
    cfg = _load_config(args.config)
    book = _book_dir(cfg, args.offline)
    state_path = book / "risk_state.json"
    if not state_path.exists():
        raise CLIError(f"no risk state at {state_path}: this book has not run yet, so it is not frozen")
    state = _load_risk_state(state_path, math.nan)
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
    new = replace(state, frozen=False, peak_nav=nav, frozen_at_nav=None)
    ledger = Ledger(book / "ledger.jsonl")
    ledger.append("risk_unfrozen", {"as_of": state.last_as_of, "book": str(book), "reason": reason,
                                    "approved_by": "--yes" if args.yes else "interactive",
                                    "before": asdict(state), "after": asdict(new)})
    _write_json(state_path, asdict(new))
    _append_anchor(book, ledger, None, state.last_as_of)
    print(f"unfrozen: high-water mark {nav:,.2f}; recorded in {book / 'ledger.jsonl'}")
    return 0


# --------------------------------------------------------------------------- parser


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

    p = sub.add_parser("run", help="one decision cycle with the paper broker (or Alpaca, with approval)")
    p.add_argument("--config", required=True)
    p.add_argument("--as-of", help="decision date YYYY-MM-DD (default: the last completed session)")
    p.add_argument("--offline", action="store_true", help="deterministic offline rules instead of Claude (not AI)")
    p.add_argument("--yes", action="store_true", help="approve orders for a non-paper broker without asking")
    p.add_argument("--force", action="store_true",
                   help="decide even if this book already decided on this session or a later one "
                        "(duplicate forecasts or backdated orders; recorded in the ledger)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("backtest", help="walk-forward backtest with baselines and a model-cutoff split")
    p.add_argument("--config", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="last date YYYY-MM-DD (at most the last completed session)")
    p.add_argument("--mode", choices=["quant", "offline", "llm"], default="quant")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation in llm mode")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("score", help="resolve due forecasts and print the per-agent scorecard")
    p.add_argument("--config")
    p.add_argument("--offline", action="store_true", help="score the offline book")
    p.add_argument("--db", help="path to a forecasts.sqlite (for example inside a backtest run directory)")
    p.add_argument("--as-of", help="score as of this date (default: the last completed session)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("verify-ledger", help="verify the hash chain of the decision ledger")
    p.add_argument("--config")
    p.add_argument("--offline", action="store_true", help="verify the offline book's ledger")
    p.add_argument("--path", help="ledger file (default: <runs_dir>/ledger.jsonl)")
    p.add_argument("--anchor", help="a head hash recorded earlier (default: the last one in ledger_anchors.jsonl)")
    p.set_defaults(func=cmd_verify_ledger)

    p = sub.add_parser("unfreeze", help="human reset of a drawdown freeze (restarts the high-water mark; ledgered)")
    p.add_argument("--config", required=True)
    p.add_argument("--offline", action="store_true", help="unfreeze the offline book")
    p.add_argument("--reason", required=True, help="what you reviewed and why the book may trade again (ledgered)")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_unfreeze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CLIError as exc:
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
