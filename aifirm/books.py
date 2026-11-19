"""Persistent paper books between runs: state files, settlement of dividends and queued fills.

A *book* is a directory holding everything one paper portfolio needs to continue from one
run to the next: ``ledger.jsonl`` (hash-chained journal), ``ledger_anchors.jsonl``,
``forecasts.sqlite``, ``lessons.jsonl``, ``paper_broker.json`` (cash and shares),
``risk_state.json`` (high-water mark, freeze, de-risk episode), ``pending_orders.json``
(orders decided at a close, waiting for the next open) and ``reports/``.

The live book (``aifirm run``) and its quant-only shadow book (:mod:`aifirm.shadow`) are
both books and go through exactly the same code here, so the two are settled the same way:

* **Fills at the next open (P3).** Orders decided at the close of a session fill at the
  open of the first later session in the price data, never at the price that produced
  the decision (:func:`settle_paper_book`, :func:`fill_pending`).
* **Total-return accounting.** Cash dividends are credited on each ex-date to the shares
  held at the previous close, before that day's fills, from
  :func:`aifirm.pipeline.implied_dividends` (the same amounts the backtest credits), and
  ``PaperBroker.dividends_through`` records the last settled session, so no ex-date is
  credited twice or skipped.
* **One decision per session.** :func:`last_decision` reads every record that carries a
  decision date (ledger ``cycle_end`` entries, ``risk_state.json``, queued orders), so a
  caller can refuse to decide the same session twice or an earlier one.

Fail closed: a corrupt state file raises :class:`BookError` (a human must look), and
settling a book whose queued orders have no later session to fill in raises instead of
deciding again (that would double the plan).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .data.prices import PriceData
from .execution.broker import PaperBroker
from .journal import Ledger
from .pipeline import implied_dividends
from .portfolio import RiskState
from .schemas import Fill, Order

if TYPE_CHECKING:        # import cycle otherwise (pipeline is imported above only for implied_dividends)
    from .pipeline import Decision

STALE_FILL_DAYS = 5      # a queued order filled more than this many calendar days ago gets a note
LEDGER_FILE = "ledger.jsonl"
ANCHORS_FILE = "ledger_anchors.jsonl"
FORECASTS_FILE = "forecasts.sqlite"
LESSONS_FILE = "lessons.jsonl"
BROKER_FILE = "paper_broker.json"
RISK_STATE_FILE = "risk_state.json"
PENDING_FILE = "pending_orders.json"
REPORTS_DIR = "reports"

Notify = Callable[[str], None]


class BookError(RuntimeError):
    """A problem with a book's saved state or schedule that a person must resolve (a corrupt
    file, orders still waiting for their open). The CLI prints it without a traceback."""


def waiting_message(decided: str) -> str:
    """The message for queued orders that have not reached their fill session yet."""
    return (f"orders decided at the close of {decided} are waiting for the next trading day's open; "
            "run again after that session (deciding again now would double the plan)")


def write_json(path: Path, data: object) -> None:
    """Write ``data`` as indented JSON atomically (temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def pending_as_of(book: Path) -> str | None:
    """The decision date of the orders queued in ``book`` (None when nothing is queued)."""
    path = book / PENDING_FILE
    if not path.exists():
        return None
    return str(json.loads(path.read_text())["as_of"])[:10]


def pending_tickers(book: Path) -> list[str]:
    """Tickers of the orders queued in ``book`` ([] when nothing is queued)."""
    path = book / PENDING_FILE
    if not path.exists():
        return []
    return [str(o["ticker"]) for o in json.loads(path.read_text())["orders"]]


def last_decision(book: Path) -> str | None:
    """The latest session ``book`` has decided on, from every record that carries one: the
    ledger's completed cycles (``cycle_end``), ``risk_state.json`` (``last_as_of``) and
    queued orders. None for a new book."""
    dates: list[str] = []
    pending = pending_as_of(book)
    if pending:
        dates.append(pending)
    state = book / RISK_STATE_FILE
    if state.exists():
        try:
            last = json.loads(state.read_text()).get("last_as_of")
        except (json.JSONDecodeError, AttributeError):
            last = None                     # a corrupt file is reported by load_risk_state
        if last:
            dates.append(str(last)[:10])
    ledger = book / LEDGER_FILE
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


def append_anchor(book: Path, ledger: Ledger, run_id: str | None, as_of: str | None) -> None:
    """Record the ledger head after a run in ``ledger_anchors.jsonl`` (``verify-ledger``
    checks the chain still reaches it)."""
    with open(book / ANCHORS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "as_of": as_of,
                             "head": ledger.head, "entries": len(ledger)}) + "\n")


def marks(prices: PriceData, row: int = -1) -> dict[str, float]:
    """Last known close per ticker at ``row`` (forward-filled, so a missing print uses the prior close)."""
    closes = prices.close.ffill()
    if closes.empty or len(closes) < abs(row):
        return {}
    return {t: float(v) for t, v in closes.iloc[row].items() if math.isfinite(v) and v > 0}


def load_risk_state(path: Path, nav: float) -> RiskState:
    """The saved risk state, or a fresh one with the high-water mark at ``nav``. A corrupt
    file raises :class:`BookError`: resetting a drawdown state is a human decision."""
    if not path.exists():
        return RiskState(peak_nav=nav)
    try:
        return RiskState(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as exc:
        raise BookError(f"corrupt risk state {path}: {exc}. Fix or remove it by hand (a human decision).") from exc


def last_fill_date(book: Path) -> str | None:
    """The latest paper fill date in the book's ledger (None if the book never filled)."""
    path = book / LEDGER_FILE
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


def settle_paper_book(broker: PaperBroker, book: Path, prices: PriceData, ledger: Ledger,
                      last_decided: str | None, notify: Notify = print) -> float | None:
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
    prices (the daily-loss halt uses it when no NAV path is on record).

    Raises :class:`BookError` when queued orders have no later session in ``prices``.
    ``notify`` receives operator notes (dividends outside the loaded prices, stale fills)."""
    idx = prices.close.index
    if len(idx) == 0:
        return None
    path = book / PENDING_FILE
    pending = json.loads(path.read_text()) if path.exists() else None
    fill_day = None
    if pending is not None:
        decided = pending["as_of"]
        after = idx[idx > pd.Timestamp(decided)]
        if len(after) == 0:
            raise BookError(waiting_message(decided))
        fill_day = after[0]
    since = broker.dividends_through or last_fill_date(book) or (pending or {}).get("as_of") or last_decided
    start = pd.Timestamp(since) if since else idx[-1]      # no history at all: nothing is owed
    if broker.positions() and since and start < idx[0]:
        notify(f"note: dividends with ex-dates from {since} to {idx[0].date()} are outside the loaded prices "
               "and were not credited")
    dividends = implied_dividends(prices)
    closes = prices.close.ffill()
    prev_day = idx[-2] if len(idx) >= 2 else None
    prev_nav = None
    if prev_day is not None and prev_day <= start:          # nothing happened since that close
        prev_nav = broker.nav(marks(prices, -2))
    credited: list[dict] = []
    for d in idx[idx > start]:
        row = dividends.loc[d]
        per_share = {t: float(row[t]) for t, q in broker.positions().items() if t in row.index and row[t] > 0}
        held = broker.positions()
        if per_share and broker.credit_dividends(per_share, str(d.date())):
            credited += [{"ex_date": str(d.date()), "ticker": t, "per_share": v, "shares": held[t],
                          "amount": held[t] * v} for t, v in sorted(per_share.items())]
        if fill_day is not None and d == fill_day:
            fill_pending(broker, book, pending, prices, d, ledger, notify)   # type: ignore[arg-type]
        if d == prev_day:
            prev_nav = broker.nav({t: float(v) for t, v in closes.loc[d].items() if math.isfinite(v) and v > 0})
    through = str(idx[-1].date())
    broker.dividends_through = max(through, broker.dividends_through or through)
    broker.save()
    if credited:
        ledger.append("dividends", {"as_of": through, "credited": credited,
                                    "total": sum(c["amount"] for c in credited)})
    return prev_nav


def fill_pending(broker: PaperBroker, book: Path, pending: dict, prices: PriceData, day: pd.Timestamp,
                 ledger: Ledger, notify: Notify = print) -> list[Fill]:
    """Fill orders decided at an earlier close at the open of ``day``, the next trading day
    (P3), record the fills in the ledger and remove the queue file."""
    decided = pending["as_of"]
    orders = [Order.model_validate(o) for o in pending["orders"]]
    opens = (prices.open if prices.open is not None else prices.close).loc[day]
    fill_px = {o.ticker: float(opens[o.ticker]) for o in orders
               if o.ticker in opens.index and math.isfinite(opens[o.ticker]) and opens[o.ticker] > 0}
    fills = broker.submit(orders, fill_px, str(day.date()), pending.get("adv"), pending.get("daily_vol"))
    ledger.append("fills", {"run_id": pending.get("run_id"), "as_of": decided, "fill_date": str(day.date()),
                            "fills": fills, "rejections": broker.last_rejections})
    (book / PENDING_FILE).unlink()
    age = (prices.close.index[-1] - day).days
    if age > STALE_FILL_DAYS:
        notify(f"note: pending orders from {decided} were filled at the open of {day.date()}, {age} days ago")
    return fills


def queue_paper_orders(decision: Decision, book: Path) -> int:
    """Queue ``decision.orders`` in ``book`` for the next trading day's open (the next
    :func:`settle_paper_book` fills them). Returns the number queued (nothing is written
    when there are none)."""
    if not decision.orders:
        return 0
    write_json(book / PENDING_FILE, {
        "run_id": decision.run_id, "as_of": decision.as_of, "orders": [o.model_dump() for o in decision.orders],
        "adv": decision.market.adv, "daily_vol": decision.market.daily_vol})
    return len(decision.orders)


def ledger_events(path: Path, *events: str) -> list[dict]:
    """Entries of the ledger at ``path`` whose event is one of ``events`` (all entries when
    none is given), in file order, without verification ([] when there is no ledger)."""
    if not path.exists():
        return []
    wanted = set(events)
    out = []
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue                    # a damaged line is the ledger audit's business
            if isinstance(entry, dict) and (not wanted or entry.get("event") in wanted):
                out.append(entry)
    return out
