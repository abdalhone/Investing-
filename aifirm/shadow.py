"""The quant-only shadow book: is the language model worth what it costs? (RESEARCH 6.6, 7.4 S6, 9.3 item 6)

Research 6.6: "keep a quant-only shadow book on the same constraints, and let measured
net-of-cost IC improvement set the overlay's weight". Every ``aifirm run`` therefore also
decides a second paper book, at the same close, with the same configuration minus every
language-model stage (:func:`aifirm.pipeline.quant_only_config`: the deterministic quant
alone, no challenge, memo or risk review). Everything else is the same: the universe and
its screen, the point-in-time fact sheets of that session, the sizing rules, the risk
engine (kill switch included), the costs, next-open fills and dividend crediting. The only
difference between the two books is what the language models did, so

    AI value added net of LLM cost = (live NAV - shadow NAV) - LLM spend since inception

and the paired per-date rank IC of the live book's final probability against the shadow's
says whether the LLMs improved the *ranking*, the quantity that survives costs and luck
better than a NAV gap over a few weeks.

Layout and lifecycle
--------------------
* The shadow of a book lives in ``<book>/shadow`` (``<runs_dir>/shadow`` for the live book,
  ``<runs_dir>/offline/shadow`` for the offline book), a complete book of its own
  (:mod:`aifirm.books`): its own ledger, forecasts, risk state, paper broker and queue.
* **Inception (fork).** The first time a book runs with the shadow enabled, the shadow is
  forked from the live book's state at the start of that cycle (after settlement, before
  the live decision): the same cash, shares and risk state, recorded in a
  ``shadow_inception`` ledger entry. Both books then decide from an identical position, so
  later differences come from the decisions alone. A shadow directory with some state
  files but not all is refused (fail closed): remove it to start a new shadow.
* **Each later cycle**, the shadow's queued orders fill at the same next-session open as
  the live book's and its dividends are credited the same way (the same
  :func:`aifirm.books.settle_paper_book`), then it decides at the same close.

Isolation guarantees (tested)
-----------------------------
* **The shadow never trades for real.** It only ever uses a
  :class:`~aifirm.execution.broker.PaperBroker`, whatever ``execution.broker`` says, and its
  orders are only queued in its own directory.
* **The shadow cannot block or alter the live decision.** :func:`run_shadow` runs after the
  live decision has been made, persisted and queued; it never raises (any error becomes a
  ``failed`` :class:`ShadowOutcome`, which the CLI prints and ledgers as a warning); it gets
  deep copies of the fact sheets; and it never writes to the live book's ledger, broker,
  risk state or queue.
* **Its only write into the live book** is one forecast record per name it scored, tagged
  ``agent = "shadow"`` (:data:`~aifirm.journal.forecasts.SHADOW_AGENT`), in the live
  forecast store, so both books' forecasts resolve and are scored together. Those records
  are excluded from everything a live decision reads: believability weights and
  calibration take explicit agent lists, the adaptive overlay bound compares
  ``llm_pool`` with ``quant``, and the pooled analyst IC excludes
  :data:`~aifirm.journal.forecasts.POOLED_AGENTS`. The shadow's own per-analyst
  ``quant`` records are not mirrored: the live store already holds the quant's forecasts,
  so mirroring them would count the quant twice.

Point-in-time: the shadow decides on exactly the data the live decision used (prices cut
at the same close, the same sheets), and the comparison counts only forecasts resolved on
or before its ``as_of``.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TypeGuard

import numpy as np
import pandas as pd

from .backtest.ic import overlap_lags
from .books import (BROKER_FILE, FORECASTS_FILE, LEDGER_FILE, LESSONS_FILE, RISK_STATE_FILE, BookError,
                    append_anchor, last_decision, ledger_events, load_risk_state, marks, pending_tickers,
                    queue_paper_orders, settle_paper_book, write_json)
from .config import FirmConfig
from .data.factsheet import FactSheet
from .data.prices import PriceData
from .execution.broker import PaperBroker
from .journal import (COMMITTEE_AGENT, SHADOW_AGENT, ForecastStore, Ledger, LessonStore, paired_rank_ic)
from .pipeline import Decision, decide, needs_llm, quant_only_config
from .portfolio import RiskState
from .portfolio.overlay import thin_dates_note
from .quant import QUANT_AGENT
from .schemas import ForecastRecord

SHADOW_DIR = "shadow"
INCEPTION_EVENT = "shadow_inception"
FORWARD_MIN_YEARS = 2.0     # research 7.4 S6: >= 2 years of forward decisions for monthly horizons / <= 100 names
VERDICT_T = 2.0             # research 7.4 S4: prefer the more complex design only at t >= 2 with a consistent sign
TRADING_DAYS = 252

PriceLoader = Callable[[list[str], pd.Timestamp, str], PriceData]   # (tickers, start, end) -> prices

log = logging.getLogger(__name__)


def _day(d: object) -> str:
    return str(pd.Timestamp(d).date())  # type: ignore[arg-type]


def shadow_book_dir(book: Path) -> Path:
    """Directory of the shadow of ``book`` (``<book>/shadow``)."""
    return Path(book) / SHADOW_DIR


def shadow_applies(cfg: FirmConfig) -> tuple[bool, str]:
    """Whether a run with ``cfg`` keeps a shadow book, and why not when it does not: the
    setting is off (``evaluation.shadow_book``), or the live configuration has no language
    model stage, so the shadow would be the live book itself."""
    if not cfg.evaluation.shadow_book:
        return False, "evaluation.shadow_book is off"
    if not needs_llm(cfg):
        return False, "the live configuration is already quant-only (no language model stage), so it is its own shadow"
    return True, ""


# --------------------------------------------------------------------------- running the shadow


@dataclass(frozen=True)
class BookSnapshot:
    """The live book at the start of a cycle: after its dividends and queued fills were
    settled, before its decision. A new shadow book is forked from it.

    ``positions`` are shares, ``marks`` the closes it was valued at, ``nav`` its value at
    the close of ``as_of``, ``prev_nav`` its value at the previous close (None if unknown)
    and ``risk_state`` the state the live decision started from."""

    as_of: str
    cash: float
    positions: Mapping[str, float]
    nav: float
    risk_state: RiskState
    marks: Mapping[str, float] = field(default_factory=dict)
    prev_nav: float | None = None


@dataclass
class ShadowOutcome:
    """What one shadow cycle did. ``status``: ``"ok"`` (decided), ``"skipped"`` (not run, with
    ``reason``) or ``"failed"`` (an error; ``reason`` says which; the live run is unaffected).
    ``mirrored`` counts the ``"shadow"`` forecasts written to the live store."""

    status: str
    reason: str
    as_of: str
    book: str
    run_id: str = ""
    decision: Decision | None = None
    nav: float | None = None
    n_orders: int = 0
    mirrored: int = 0
    inception: bool = False
    ledger_head: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Ledger-safe summary (the decision object itself is in the shadow's own ledger)."""
        out = {k: getattr(self, k) for k in ("status", "reason", "as_of", "book", "run_id", "nav", "n_orders",
                                             "mirrored", "inception", "ledger_head", "warnings")}
        if self.decision is not None:
            out["exposure"] = self.decision.exposure()
            out["held_by_outage"] = self.decision.held_by_outage
        return out


def run_shadow(cfg: FirmConfig, *, book: Path, runs_root: Path, as_of: str, prices: PriceData,
               sheets: Mapping[str, FactSheet], live_run_id: str, live_start: BookSnapshot,
               live_forecasts: Path | None = None, price_loader: PriceLoader | None = None,
               notify: Callable[[str], None] = print) -> ShadowOutcome:
    """Settle and decide the quant-only shadow of ``book`` at the close of ``as_of``.

    Call it after the live decision for ``as_of`` has been made and persisted, with the
    prices (cut at ``as_of``) and fact sheets that decision used. ``live_start`` is the
    live book before its decision (used only on the first run, to fork the shadow).
    ``live_forecasts`` is the live forecast store, which receives the shadow's final
    probabilities tagged ``"shadow"`` (None: not mirrored). ``price_loader(tickers, start,
    end)`` fetches prices for names the shadow holds that ``prices`` lacks (None: such a
    shadow fails). ``runs_root`` is where the kill switch lives (the same as the live
    book's). ``notify`` receives operator notes.

    Never raises (except ``KeyboardInterrupt``-like exits): a skipped or failed shadow is
    reported in the returned :class:`ShadowOutcome`. A shadow that already decided on
    ``as_of`` or later (a forced live re-decision) is skipped, never re-decided."""
    as_of_s = _day(as_of)
    sdir = shadow_book_dir(book)
    try:
        return _run_shadow(cfg, sdir=sdir, runs_root=Path(runs_root), as_of=as_of_s, prices=prices, sheets=sheets,
                           live_run_id=live_run_id, live_start=live_start, live_forecasts=live_forecasts,
                           price_loader=price_loader, notify=notify)
    except Exception as exc:  # noqa: BLE001 - the shadow must never take the live run down with it
        log.warning("shadow book failed", exc_info=True)
        text = f"{type(exc).__name__}: {exc}"
        return ShadowOutcome(status="failed", reason=text if len(text) <= 300 else text[:297] + "...",
                             as_of=as_of_s, book=str(sdir))


def _run_shadow(cfg: FirmConfig, *, sdir: Path, runs_root: Path, as_of: str, prices: PriceData,
                sheets: Mapping[str, FactSheet], live_run_id: str, live_start: BookSnapshot,
                live_forecasts: Path | None, price_loader: PriceLoader | None,
                notify: Callable[[str], None]) -> ShadowOutcome:
    def skipped(reason: str) -> ShadowOutcome:
        return ShadowOutcome(status="skipped", reason=reason, as_of=as_of, book=str(sdir))

    wanted, why = shadow_applies(cfg)
    if not wanted:
        return skipped(why)
    last = last_decision(sdir)
    if last is not None and as_of <= last:
        return skipped(f"the shadow book already decided at the close of {last}; it never decides a session twice")
    files = [sdir / LEDGER_FILE, sdir / BROKER_FILE, sdir / RISK_STATE_FILE]
    present = [f.name for f in files if f.exists()]
    if present and len(present) < len(files):
        raise BookError(f"the shadow book in {sdir} is incomplete (has {', '.join(present)}, missing "
                        f"{', '.join(f.name for f in files if not f.exists())}); remove that directory to start a "
                        "new shadow from the live book (a human decision)")
    inception = not present
    qcfg = quant_only_config(cfg)

    def note(text: str) -> None:
        notify(f"shadow: {text}")

    sdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(sdir / LEDGER_FILE)
    if inception:
        if live_start.as_of != as_of:
            raise ValueError(f"live snapshot is from {live_start.as_of}, not {as_of}")
        broker = PaperBroker.from_snapshot(live_start.cash, live_start.positions, cfg.costs,
                                           state_path=str(sdir / BROKER_FILE), marks=live_start.marks,
                                           dividends_through=as_of)
        state = replace(live_start.risk_state)
        write_json(sdir / RISK_STATE_FILE, asdict(state))
        ledger.append(INCEPTION_EVENT, {
            "as_of": as_of, "live_book": str(sdir.parent), "live_run_id": live_run_id, "nav": live_start.nav,
            "cash": live_start.cash, "positions": dict(live_start.positions), "risk_state": state,
            "config_fingerprint": qcfg.fingerprint(), "live_config_fingerprint": cfg.fingerprint()})
        sprices = prices
        prev_nav = live_start.prev_nav
    else:
        broker = PaperBroker(cfg.execution.initial_cash, cfg.costs, state_path=str(sdir / BROKER_FILE))
        if not broker.load():
            raise BookError(f"shadow broker state {sdir / BROKER_FILE} could not be loaded")
        sprices = _covering_prices(prices, [*broker.positions(), *pending_tickers(sdir)], as_of, price_loader)
        prev_nav = settle_paper_book(broker, sdir, sprices, ledger, last, notify=note)
        state = None

    positions = broker.positions()
    mk = marks(sprices)
    nav = broker.nav(mk)
    if not (math.isfinite(nav) and nav > 0):
        raise BookError(f"shadow NAV is not positive ({nav!r})")
    current = {t: q * mk[t] / nav for t, q in positions.items() if t in mk}
    if state is None:
        state = load_risk_state(sdir / RISK_STATE_FILE, nav)
    run_id = f"{live_run_id}-shadow"
    with ForecastStore(sdir / FORECASTS_FILE) as store:
        dec = decide(qcfg, None, as_of, sprices, copy.deepcopy(dict(sheets)), current_weights=current, nav=nav,
                     prev_nav=prev_nav, risk_state=state, forecast_store=store,
                     lesson_store=LessonStore(sdir / LESSONS_FILE), ledger=ledger, run_id=run_id,
                     positions=positions, runs_dir=str(runs_root))
    write_json(sdir / RISK_STATE_FILE, asdict(dec.risk_state))
    n_orders = queue_paper_orders(dec, sdir)
    append_anchor(sdir, ledger, run_id, as_of)
    warnings = list(dec.warnings)
    mirrored = 0
    if live_forecasts is not None:
        try:
            with ForecastStore(live_forecasts) as live_store:
                mirrored = len(live_store.add(shadow_forecast_records(dec)))
        except Exception as exc:  # noqa: BLE001 - the shadow decided; only this session's comparison is lost
            warnings.append(f"shadow forecasts were not recorded in the live store ({type(exc).__name__}: {exc}); "
                            "this session is missing from the IC comparison")
    return ShadowOutcome(status="ok", reason="", as_of=as_of, book=str(sdir), run_id=run_id, decision=dec, nav=nav,
                         n_orders=n_orders, mirrored=mirrored, inception=inception, ledger_head=ledger.head,
                         warnings=warnings)


def _covering_prices(prices: PriceData, needed: Iterable[str], as_of: str,
                     price_loader: PriceLoader | None) -> PriceData:
    """``prices`` plus columns for every ``needed`` ticker it lacks (fetched with
    ``price_loader`` and aligned to ``prices``' dates, so the names both books price are
    priced identically). Raises when a needed ticker cannot be priced."""
    have = {t for t in prices.close.columns if prices.close[t].notna().any()}
    missing = sorted(set(needed) - have)
    if not missing:
        return prices
    if price_loader is None:
        raise BookError(f"no prices for shadow holding(s) {', '.join(missing)} and no price loader")
    extra = price_loader(missing, prices.close.index[0], as_of).as_of(as_of)
    idx = prices.close.index
    absent = [t for t in missing if t not in extra.close.columns or extra.close[t].reindex(idx).dropna().empty]
    if absent:
        raise BookError(f"no prices for shadow holding(s) {', '.join(absent)}")

    def join(base: pd.DataFrame | None, add: pd.DataFrame | None, fill: float | None = None) -> pd.DataFrame | None:
        if base is None:
            return None
        cols = [t for t in missing if add is not None and t in add.columns]
        right = add[cols].reindex(base.index) if add is not None else pd.DataFrame(index=base.index)
        if fill is not None:
            right = right.fillna(fill)
        return pd.concat([base.drop(columns=[c for c in cols if c in base.columns]), right], axis=1)

    splits = prices.splits
    if extra.splits is not None and splits is not None:
        added = extra.splits[[t for t in missing if t in extra.splits.columns]]
        splits = pd.concat([splits, added], axis=1).fillna(1.0)
    return PriceData(close=join(prices.close, extra.close), adj_close=join(prices.adj_close, extra.adj_close),
                     volume=join(prices.volume, extra.volume), open=join(prices.open, extra.open),
                     raw_close=join(prices.raw_close, extra.raw_close if extra.raw_close is not None else extra.close),
                     splits=splits)


def shadow_forecast_records(dec: Decision) -> list[ForecastRecord]:
    """The shadow decision's final probability for every name it scored, tagged
    ``agent = "shadow"`` for the live forecast store.

    Per ticker: the shadow's ``committee`` record (its final probability) when the name
    went through its committee, else its ``quant`` record (screened names outside its
    deep-dive set). Abstentions have no record, so they are not forecasts here either. The
    shadow's per-analyst ``quant`` records are never mirrored as such (the live store
    already has the quant's forecasts). IDs are ``<shadow run_id>|<as_of>|<ticker>|shadow``.
    """
    best: dict[str, ForecastRecord] = {}
    for r in dec.forecasts:
        if r.agent == COMMITTEE_AGENT or (r.agent == QUANT_AGENT and r.ticker not in best):
            best[r.ticker] = r
    return [r.model_copy(update={"id": f"{dec.run_id}|{_day(r.as_of)}|{t}|{SHADOW_AGENT}", "agent": SHADOW_AGENT,
                                 "run_id": dec.run_id})
            for t, r in sorted(best.items())]


# --------------------------------------------------------------------------- comparing the books


@dataclass
class BookComparison:
    """Live book vs its quant-only shadow (module docstring), as of ``as_of``.

    NAVs are the books' values at the close of the latest session both decided
    (``nav_date``); returns run from the shadow's inception NAV (both books started from
    it). A session counts as *decided* only when its cycle completed (a ``cycle_end`` entry);
    a cycle that started and then failed (``cycle_error``) is a missed session, for the live
    book and the shadow alike. The NAV at ``nav_date`` is the NAV *before* that session's
    decision (its orders fill at the next open), so ``llm_spend_usd`` sums the live book's
    LLM cost over the cycles from inception up to but excluding ``nav_date``, failed
    attempts included (their spend is logged on ``cycle_error``): exactly the decisions
    whose effect the compared NAVs contain. The spend on ``nav_date`` and on every later
    cycle up to ``as_of`` (a failed latest cycle included) is ``llm_spend_pending_usd``,
    charged once its fills reach a compared NAV.
    ``value_added_usd = difference_usd - llm_spend_usd``. ``live_failed`` lists sessions up
    to ``as_of`` whose live cycle failed and was not decided. The IC fields
    come from :func:`aifirm.journal.paired_rank_ic` of the live final probability
    (``committee``) against the shadow's, on the names both forecast, with forecasts
    resolved on or before ``as_of``; ``min_dates`` is the forward record needed before a
    verdict (research 7.4 S6). ``status``: ``no_shadow``, ``no_common_decision``,
    ``not_measurable``, ``better``, ``worse`` or ``no_difference``."""

    status: str
    as_of: str | None
    verdict: str
    inception_as_of: str | None = None
    inception_nav: float | None = None
    nav_date: str | None = None
    live_nav: float | None = None
    shadow_nav: float | None = None
    live_return_pct: float | None = None
    shadow_return_pct: float | None = None
    difference_usd: float | None = None
    difference_pp: float | None = None
    llm_spend_usd: float = 0.0
    llm_spend_pending_usd: float = 0.0
    value_added_usd: float | None = None
    value_added_pct: float | None = None
    live_cycles: int = 0
    shadow_missed: list[str] = field(default_factory=list)
    live_failed: list[str] = field(default_factory=list)
    ic_dates: int = 0
    ic_mean_diff: float | None = None
    ic_t_stat: float | None = None
    ic_live: float | None = None
    ic_shadow: float | None = None
    ic_lags: int = 0
    step_days: int = 0
    min_dates: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def value_sentence(self) -> str:
        """The value-added line in plain words."""
        if self.value_added_usd is None:
            return "AI value added net of LLM cost: not available yet (no session decided by both books)."
        pct = f" ({self.value_added_pct:+.2f}% of the starting NAV)" if _finite(self.value_added_pct) else ""
        return (f"AI value added net of LLM cost: {_usd_signed(self.value_added_usd)}{pct} = live minus shadow "
                f"{_usd_signed(self.difference_usd)} minus LLM spend {_usd(self.llm_spend_usd)}, over "
                f"{self.live_cycles} live decision(s) since {self.inception_as_of}.")

    def lines(self) -> list[str]:
        """Plain-text lines for the console and the run report."""
        if self.status in ("no_shadow", "no_common_decision"):
            return [self.verdict, *self.notes]
        out = [f"Shadow inception {self.inception_as_of} at NAV {_usd(self.inception_nav)}; compared at the close of "
               f"{self.nav_date}.",
               f"NAV: live {_usd(self.live_nav)} ({_pp(self.live_return_pct)}) vs quant-only shadow "
               f"{_usd(self.shadow_nav)} ({_pp(self.shadow_return_pct)}); difference "
               f"{_usd_signed(self.difference_usd)} "
               f"({_pp(self.difference_pp)} points).",
               f"Cumulative LLM spend since inception: {_usd(self.llm_spend_usd)} on the decisions reflected in these "
               f"NAVs" + (f" (plus {_usd(self.llm_spend_pending_usd)} spent on the {self.nav_date} decision and any "
                          "later attempt, failed ones included, charged once their fills reach a compared NAV)."
                          if self.llm_spend_pending_usd else "."),
               self.value_sentence(),
               f"Ranking (paired per-date rank IC, live final P minus shadow P on common names): {self.verdict}"]
        return [*out, *self.notes]


def _finite(x: object) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _usd(x: object) -> str:
    return f"${float(x):,.2f}" if _finite(x) else "n/a"


def _usd_signed(x: object) -> str:
    if not _finite(x):
        return "n/a"
    v = float(x)  # type: ignore[arg-type]
    return f"{'-' if v < 0 else '+'}${abs(v):,.2f}"


def _pp(x: object) -> str:
    return f"{float(x):+.2f}%" if _finite(x) else "n/a"


def _nan_none(x: float) -> float | None:
    return float(x) if _finite(x) else None


def decision_step_days(dates: Iterable[str], default: int) -> int:
    """Median number of business days between consecutive distinct decision dates (the
    book's actual cadence), or ``default`` with fewer than two dates."""
    ds = sorted({_day(d) for d in dates})
    if len(ds) < 2:
        return int(default)
    gaps = [int(np.busday_count(a, b)) for a, b in zip(ds[:-1], ds[1:], strict=True)]
    return max(1, int(round(float(np.median(gaps)))))


def forward_min_dates(cfg: FirmConfig, step_days: int) -> int:
    """Resolved decision dates needed before the forward IC comparison is read: about
    :data:`FORWARD_MIN_YEARS` years of decisions at ``step_days`` (research 7.4 S6: at
    least two years for monthly horizons or books of 100 names or fewer), and never fewer
    than ``process.overlay_min_dates``."""
    return max(int(cfg.process.overlay_min_dates), math.ceil(FORWARD_MIN_YEARS * TRADING_DAYS / max(1, step_days)))


def ic_verdict(n_dates: int, min_dates: int, mean_diff: float | None, t_stat: float | None,
               horizon_days: int) -> tuple[str, str]:
    """``(status, sentence)`` for a paired IC comparison of the live book against its shadow."""
    so_far = ""
    if n_dates >= 2 and mean_diff is not None:
        so_far = (f"; so far the mean IC difference is {mean_diff:+.3f} (t = "
                  f"{f'{t_stat:.2f}' if t_stat is not None else 'n/a'})")
    if n_dates < min_dates:
        wait = (f" (a forecast resolves {horizon_days} trading days after its decision)" if n_dates == 0 else "")
        return "not_measurable", f"not yet measurable: {n_dates} of {min_dates} resolved decision dates{wait}{so_far}."
    if t_stat is not None and t_stat >= VERDICT_T and (mean_diff or 0) > 0:
        return "better", (f"the live book ranks stocks better than its quant-only shadow: mean IC difference "
                          f"{mean_diff:+.3f}, t = {t_stat:.2f} over {n_dates} dates (bar t >= {VERDICT_T:g}).")
    if t_stat is not None and t_stat <= -VERDICT_T:
        return "worse", (f"the live book ranks stocks WORSE than its quant-only shadow: mean IC difference "
                         f"{mean_diff:+.3f}, t = {t_stat:.2f} over {n_dates} dates; the LLM overlay is subtracting "
                         "value and should be shrunk or switched off (research 6.6).")
    t_text = f"{t_stat:.2f}" if t_stat is not None else "n/a"
    return "no_difference", (f"no measurable difference over {n_dates} dates (mean IC difference "
                             f"{mean_diff if mean_diff is not None else float('nan'):+.3f}, t = {t_text}, bar "
                             f"|t| >= {VERDICT_T:g}): by research 7.4, default to the simpler quant-only design.")


def _live_failed_note(failed: list[str], last: str | None = None) -> str:
    shown = ", ".join(failed[:5]) + (", ..." if len(failed) > 5 else "")
    after = [d for d in failed if last is None or d >= last]
    tail = (" Their LLM spend is in the spend shown above, not yet in any compared NAV." if last is None else
            " The spend of failed attempts is counted: in the LLM spend when it lies before the compared session, "
            "else as spend not yet in the compared NAVs." if after else
            " The LLM spend of the failed attempts is counted.")
    return f"The live cycle failed and was not decided on {len(failed)} session(s) ({shown}).{tail}"


def compare_books(cfg: FirmConfig, book: Path, records: Iterable[ForecastRecord], *, as_of: str | None = None,
                  include_pre_cutoff: bool = False) -> BookComparison:
    """Compare ``book`` (the live book's directory) with its shadow (module docstring).

    Reads both ledgers (``shadow_inception``, ``cycle_start`` NAVs, the live ``cycle_end``
    LLM costs) and the live store's ``records`` (pass ``ForecastStore.all()``, resolved as
    far as the caller could). Only decisions on or before ``as_of`` (default: all) and
    forecasts resolved on or before ``as_of`` (default: the latest common decision) count.
    ``include_pre_cutoff`` follows the believability rule (True for offline or quant books,
    whose rules cannot have memorised history). Read-only; never raises for missing data
    (the status says what is missing)."""
    as_of_s = _day(as_of) if as_of is not None else None
    sdir = shadow_book_dir(book)
    inc = ledger_events(sdir / LEDGER_FILE, INCEPTION_EVENT)
    h = cfg.process.horizon_days
    if not inc:
        return BookComparison(status="no_shadow", as_of=as_of_s,
                              verdict=f"No shadow book yet in {sdir}: the next `aifirm run` starts one.")
    inc_p = inc[0].get("payload") or {}
    inc_day = _day(inc_p["as_of"])
    inc_nav = float(inc_p["nav"])

    def navs(path: Path) -> tuple[dict[str, float], list[str]]:
        """(decided session -> NAV before its decision, sessions that started and only failed)."""
        starts: dict[str, float] = {}
        for e in ledger_events(path, "cycle_start"):
            p = e.get("payload") or {}
            if p.get("as_of") and _finite(p.get("nav")):
                starts[_day(p["as_of"])] = float(p["nav"])  # a forced re-decision repeats the date: same NAV
        ended = {_day(p["as_of"]) for e in ledger_events(path, "cycle_end") if (p := e.get("payload") or {}).get("as_of")}
        errored = {_day(p["as_of"]) for e in ledger_events(path, "cycle_error")
                   if (p := e.get("payload") or {}).get("as_of")}
        return {d: v for d, v in starts.items() if d in ended}, sorted(d for d in errored if d not in ended)

    (live_nav, live_failed), (shadow_nav, shadow_failed) = navs(Path(book) / LEDGER_FILE), navs(sdir / LEDGER_FILE)
    in_window = [d for d in sorted(live_nav) if d >= inc_day and (as_of_s is None or d <= as_of_s)]
    common = [d for d in in_window if d in shadow_nav]
    step = decision_step_days(live_nav, cfg.process.rebalance_every_days)
    lags = overlap_lags(h, step)
    min_dates = forward_min_dates(cfg, step)
    out = BookComparison(status="no_common_decision", as_of=as_of_s, verdict="", inception_as_of=inc_day,
                         inception_nav=inc_nav, step_days=step, ic_lags=lags, min_dates=min_dates)
    out.live_failed = [d for d in live_failed if d >= inc_day and (as_of_s is None or d <= as_of_s)]
    def spend(before: str | None) -> tuple[float, float]:
        """(spend on sessions before ``before``, spend on ``before`` and later), failed attempts included."""
        done = pending = 0.0
        for event in ("cycle_end", "cycle_error"):          # a failed attempt spent money too
            for e in ledger_events(Path(book) / LEDGER_FILE, event):
                p = e.get("payload") or {}
                d = _day(p["as_of"]) if p.get("as_of") else None
                cost = (p.get("usage") or {}).get("cost_usd", 0.0)
                if d is None or not _finite(cost) or d < inc_day or (as_of_s is not None and d > as_of_s):
                    continue
                if before is not None and d < before:
                    done += float(cost)                    # every cycle's spend, a forced re-decision's too
                else:   # `before` and any later attempt (a failed latest cycle too): not in the compared NAVs yet
                    pending += float(cost)
        return done, pending

    if not common:
        out.verdict = f"The live book and its shadow (inception {inc_day}) have no decided session in common yet."
        _, out.llm_spend_pending_usd = spend(None)
        if out.llm_spend_pending_usd:
            out.notes.append(f"LLM spend since the shadow's inception: {_usd(out.llm_spend_pending_usd)} (failed "
                             "attempts included), not yet in any compared NAV.")
        if out.live_failed:
            out.notes.append(_live_failed_note(out.live_failed))
        return out
    last = common[-1]
    out.nav_date, out.live_nav, out.shadow_nav = last, live_nav[last], shadow_nav[last]
    out.live_return_pct = (out.live_nav / inc_nav - 1.0) * 100.0
    out.shadow_return_pct = (out.shadow_nav / inc_nav - 1.0) * 100.0
    out.difference_usd = out.live_nav - out.shadow_nav
    out.difference_pp = out.live_return_pct - out.shadow_return_pct
    out.llm_spend_usd, out.llm_spend_pending_usd = spend(last)
    out.value_added_usd = out.difference_usd - out.llm_spend_usd
    out.value_added_pct = out.value_added_usd / inc_nav * 100.0 if inc_nav > 0 else None
    upto = [d for d in in_window if d <= last]
    out.live_cycles = len(upto)
    out.shadow_missed = [d for d in upto if d not in shadow_nav]
    if out.shadow_missed:
        shown = ", ".join(out.shadow_missed[:5]) + (", ..." if len(out.shadow_missed) > 5 else "")
        failed_inside = [d for d in out.shadow_missed if d in shadow_failed]
        inside = (f"; {len(failed_inside)} of them failed inside the decision (cycle_error in the shadow ledger)"
                  if failed_inside else "")
        out.notes.append(f"The shadow did not decide on {len(out.shadow_missed)} of the {len(upto)} live sessions "
                         f"since inception ({shown}){inside}: see the 'shadow' entries in the live ledger; the NAV "
                         "comparison includes those gaps.")
    if out.live_failed:
        out.notes.append(_live_failed_note(out.live_failed, last))

    res = paired_rank_ic(list(records), as_of_s or last, COMMITTEE_AGENT, SHADOW_AGENT,
                         include_pre_cutoff=include_pre_cutoff, min_names=cfg.evaluation.ic_min_names, nw_lags=lags)
    out.ic_dates = res.n_dates
    out.ic_mean_diff, out.ic_t_stat = _nan_none(res.mean_diff), _nan_none(res.t_stat)
    out.ic_live, out.ic_shadow = _nan_none(res.mean_a), _nan_none(res.mean_b)
    out.status, out.verdict = ic_verdict(res.n_dates, min_dates, out.ic_mean_diff, out.ic_t_stat, h)
    if res.n_thin:
        out.notes.append(thin_dates_note(cfg, res.n_thin, cfg.evaluation.ic_min_names)
                         .replace("had both forecasts", "had both a live and a shadow forecast") + ".")
    if out.status == "not_measurable":
        out.notes.append("A NAV gap over a short window is mostly noise; the ranking comparison is the test, and it "
                         f"needs about {FORWARD_MIN_YEARS:g} years of decisions at this cadence ({min_dates} dates "
                         f"{step} trading day{'s' if step != 1 else ''} apart).")
    return out
