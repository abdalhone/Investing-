"""Command-line interface: ``aifirm factsheet | run | backtest | score | verify-ledger``.

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

Live-book layout under ``runs_dir`` (``runs_dir/offline`` with ``--offline``):
``ledger.jsonl``, ``ledger_anchors.jsonl``, ``forecasts.sqlite``, ``lessons.jsonl``,
``paper_broker.json``, ``risk_state.json``, ``pending_orders.json`` and ``reports/``.
The kill switch is always ``<runs_dir>/<risk.kill_switch_file>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from pydantic import ValidationError

from .agents import make_offline_llm
from .backtest.engine import run_backtest
from .config import FirmConfig
from .data.prices import PriceData, load_prices
from .execution.broker import AlpacaBroker, PaperBroker
from .journal import ForecastStore, Ledger, LedgerError, LessonStore, scorecard
from .llm.client import LLM, ClaudeLLM
from .pipeline import PRICE_LOOKBACK_DAYS, Decision, decide, gather_data, needs_llm, universe
from .portfolio import RiskState
from .report import render_backtest_report, render_decision_report
from .schemas import Fill, Order


def _market_tz() -> tzinfo:
    """New York time; without a time-zone database (e.g. Windows without tzdata) fixed EST,
    which reads the clock an hour early in summer and so errs toward the previous session."""
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=-5), "EST")


MARKET_TZ = _market_tz()
SESSION_CLOSE_SETTLED = (16, 30)    # after 16:30 New York time the day's close is final
STALE_FILL_DAYS = 5


class CLIError(RuntimeError):
    """A problem the user can fix; printed without a traceback."""


# --------------------------------------------------------------------------- helpers


def last_completed_session(now: datetime | None = None) -> str:
    """The latest weekday whose close is final in New York (a holiday resolves to the last
    trading day through the price index). Deciding on a partial intraday bar would be
    deciding on a price that is not a close (P3)."""
    t = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    day = t.date()
    if t.weekday() < 5 and (t.hour, t.minute) < SESSION_CLOSE_SETTLED:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return str(day)


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
    if path is None:
        return FirmConfig()
    try:
        return FirmConfig.load(path)
    except FileNotFoundError as exc:
        raise CLIError(f"config file not found: {path}") from exc
    except (ValueError, ValidationError) as exc:      # TOMLDecodeError is a ValueError
        raise CLIError(f"invalid config {path}: {exc}") from exc


def _book_dir(cfg: FirmConfig, offline: bool) -> Path:
    return Path(cfg.runs_dir) / "offline" if offline else Path(cfg.runs_dir)


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


def _fill_pending(broker: PaperBroker, book: Path, prices: PriceData, ledger: Ledger) -> list[Fill]:
    """Fill orders decided at an earlier close at the next trading day's open (P3)."""
    path = book / "pending_orders.json"
    if not path.exists():
        return []
    pending = json.loads(path.read_text())
    decided = pending["as_of"]
    idx = prices.close.index
    after = idx[idx > pd.Timestamp(decided)]
    if len(after) == 0:
        raise CLIError(f"orders decided at the close of {decided} are waiting for the next trading day's open; "
                       "run again after that session (deciding again now would double the plan)")
    day = after[0]
    orders = [Order.model_validate(o) for o in pending["orders"]]
    opens = (prices.open if prices.open is not None else prices.close).loc[day]
    fill_px = {o.ticker: float(opens[o.ticker]) for o in orders
               if o.ticker in opens.index and math.isfinite(opens[o.ticker]) and opens[o.ticker] > 0}
    fills = broker.submit(orders, fill_px, str(day.date()), pending.get("adv"), pending.get("daily_vol"))
    ledger.append("fills", {"run_id": pending.get("run_id"), "as_of": decided, "fill_date": str(day.date()),
                            "fills": fills, "rejections": broker.last_rejections})
    path.unlink()
    age = (idx[-1] - day).days
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
    as_of = args.as_of or last_completed_session()
    _, sheets = gather_data(cfg, as_of, live=args.as_of is None)   # headlines only for "now"
    if ticker not in sheets:
        raise CLIError(f"no price data for {ticker} as of {as_of}")
    print(sheets[ticker].render())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    offline = bool(args.offline)
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

    book = _book_dir(cfg, offline)
    book.mkdir(parents=True, exist_ok=True)
    as_of = args.as_of or last_completed_session()
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
    if isinstance(broker, PaperBroker):
        _fill_pending(broker, book, prices, ledger)
    prices, sheets = gather_data(cfg, as_of, live=args.as_of is None, prices=prices)
    as_of = str(prices.close.index[-1].date())              # the session whose close we decide on
    marks, prev_marks = _marks(prices), _marks(prices, -2)
    positions = broker.positions()
    prev_nav = broker.nav(prev_marks) if prev_marks else None
    nav = broker.nav(marks)
    current = {t: q * marks[t] / nav for t, q in positions.items() if t in marks}
    state_path = book / "risk_state.json"
    state = _load_risk_state(state_path, nav)

    run_id = f"run-{as_of}-{secrets.token_hex(3)}"
    with ForecastStore(book / "forecasts.sqlite") as store:
        decision = decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=nav, prev_nav=prev_nav,
                          risk_state=state, forecast_store=store, lesson_store=LessonStore(book / "lessons.jsonl"),
                          ledger=ledger, run_id=run_id, positions=positions)
    _write_json(state_path, asdict(decision.risk_state))
    _execute(decision, broker, book, ledger, args.yes)

    report = book / "reports" / f"decision_{as_of}_{run_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_decision_report(decision))
    with open(book / "ledger_anchors.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "as_of": as_of,
                             "head": ledger.head, "entries": len(ledger)}) + "\n")
    print(f"decision {run_id} at the close of {as_of}: {len(decision.orders)} order(s), "
          f"LLM cost ${decision.usage.get('cost_usd', 0.0):.2f}")
    for w in decision.warnings:
        print(f"warning: {w}")
    print(f"report: {report}")
    return 0


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
    llm = None
    if args.mode == "llm":
        cycles = len(pd.bdate_range(args.start, args.end)) // max(1, cfg.process.rebalance_every_days) + 1
        budget = cfg.process.max_llm_cost_usd_per_run * cycles
        print(f"LLM backtest: about {cycles} decision cycles, LLM budget up to ${budget:,.2f} "
              f"(max_llm_cost_usd_per_run x cycles). Decisions on or before the model's training cutoff are "
              "contaminated and are reported separately.")
        if not args.yes and not _confirm("Continue?"):
            print("cancelled")
            return 1
        llm = _claude(cfg, budget)
    result = run_backtest(cfg, args.start, args.end, args.mode, llm=llm)
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
    path = Path(args.db) if args.db else _book_dir(cfg, args.offline) / "forecasts.sqlite"
    if not path.exists():
        print(f"no forecasts yet ({path} does not exist)")
        return 0
    as_of = args.as_of or last_completed_session()
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
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("backtest", help="walk-forward backtest with baselines and a model-cutoff split")
    p.add_argument("--config", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
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
