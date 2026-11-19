"""Broad-universe quant screen: which names get an LLM deep dive this cycle (RESEARCH 7.3, 9.2, 9.3 item 1).

Why
---
Statistical power comes from breadth: with about ten names a month, skill cannot be
confirmed for decades, while a few hundred names give a usable rank IC within a few
years (RESEARCH 7.3). LLM analysis is expensive, so the research recommends a cheap
deterministic screen over a broad liquid universe and LLM deep dives only for the
candidates and the holdings (RESEARCH 9.2 "cadence and universe").

Contract of :func:`screen_universe`
-----------------------------------
* **Input.** The fact sheets of every universe name priced at the decision close (built
  without news or LLMs), the deterministic quant views scored over that whole
  cross-section (z-scores need the broad universe), the prices and the held tickers.
* **Eligibility (point-in-time).** A name is eligible when (1) the quant has a view on it
  (it did not abstain), (2) its last close on or before ``as_of`` is at least
  ``min_price`` and no more than :data:`STALE_DAYS` calendar days old, and (3) its average
  daily dollar volume (close x volume) over the last :data:`LIQUIDITY_WINDOW` rows up to
  ``as_of`` is at least ``min_dollar_volume`` (with at least :data:`MIN_LIQUIDITY_OBS`
  valid days; unknown liquidity is not eligible). Prices are cut at ``as_of`` here, so a
  caller cannot leak later prints into the filters.
* **Ranking.** Eligible names are ranked by quant probability, highest first, ties broken
  by ticker (deterministic, independent of input order).
* **Deep-dive set.** The top ``top_n`` eligible names whose quant probability is above 0.5
  (a name the quant expects to lag is never a candidate), then every held name with a fact
  sheet that is not already in that band, whatever its rank or eligibility: a held position
  must be reviewed by the committee, and leaving it out would force a sale. Names outside
  the set get no new position this cycle.

Known caveat: the universe *list* itself is not point-in-time. A list of current index
members (the bundled presets) omits names dropped or delisted during a backtest window,
so backtest results on it are biased upward (survivorship); reports say so.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..data.factsheet import FactSheet
from ..data.prices import PriceData
from ..schemas import ViewRecord

LIQUIDITY_WINDOW = 20          # trading days for the average dollar volume (same window as the pipeline's ADV)
MIN_LIQUIDITY_OBS = 5          # fewer valid days and liquidity is unknown (not eligible)
STALE_DAYS = 5                 # a last close older than this (calendar days) is stale (halted or delisted)

TOP_BAND = "top band"
HELD = "held"


@dataclass(frozen=True)
class ScreenEntry:
    """One universe name in a screen. ``rank`` is 1 for the highest quant probability among
    eligible names (None when not eligible); ``reason`` says why a name is not eligible
    ("" when it is); ``deep_dive_reason`` is :data:`TOP_BAND`, :data:`HELD` or ""."""

    ticker: str
    quant_prob: float | None
    eligible: bool
    reason: str
    rank: int | None
    adv_usd: float | None
    last_price: float | None
    held: bool
    deep_dive: bool
    deep_dive_reason: str


@dataclass
class ScreenResult:
    """The outcome of one screen (module docstring). ``entries``: every name with a fact
    sheet, eligible names first in rank order, then the rest by ticker. ``deep_dive``: the
    top band in rank order, then held names outside it in ticker order."""

    as_of: str
    top_n: int
    min_dollar_volume: float
    min_price: float
    entries: list[ScreenEntry] = field(default_factory=list)
    deep_dive: list[str] = field(default_factory=list)

    @property
    def n_universe(self) -> int:
        return len(self.entries)

    @property
    def n_eligible(self) -> int:
        return sum(1 for e in self.entries if e.eligible)

    @property
    def top_band(self) -> list[str]:
        return [e.ticker for e in self.entries if e.deep_dive_reason == TOP_BAND]

    def quant_probs(self) -> dict[str, float]:
        """Ticker -> quant probability for every scored name (abstentions omitted)."""
        return {e.ticker: e.quant_prob for e in self.entries if e.quant_prob is not None}

    def to_dict(self) -> dict:
        return {"as_of": self.as_of, "top_n": self.top_n, "min_dollar_volume": self.min_dollar_volume,
                "min_price": self.min_price, "n_universe": self.n_universe, "n_eligible": self.n_eligible,
                "deep_dive": list(self.deep_dive), "top_band": self.top_band,
                "ranking": [asdict(e) for e in self.entries]}


def _liquidity(p: PriceData, ticker: str) -> tuple[float | None, float | None, pd.Timestamp | None]:
    """(average dollar volume, last as-traded close, date of the last close) from prices
    already cut at the decision close; None where unknown.

    The last close is the **as-traded** level (:meth:`PriceData.price_level`), the price that
    printed that day, as the minimum-price rule and the risk engine's price fact use. The
    ``close`` series is back-adjusted for every split up to the download date, so in a
    backtest it would divide a historical level by splits that had not happened yet (NVDA
    at 2016-06-30 shows 1.18 instead of the 47.01 that traded) and screen names out on
    future corporate actions. Dollar volume uses ``close x volume``, which is split-invariant."""
    if ticker not in p.close.columns:
        return None, None, None
    closes = p.close[ticker].dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None, None, None
    last_day = closes.index[-1]
    level = p.price_level(ticker)
    raw = level.get(last_day) if last_day in level.index else None
    last = float(raw) if raw is not None and math.isfinite(float(raw)) and float(raw) > 0 else None
    if last is None:
        return None, None, None
    adv = None
    if ticker in p.volume.columns:
        dollar = (p.close[ticker] * p.volume[ticker]).iloc[-LIQUIDITY_WINDOW:].dropna()
        if len(dollar) >= MIN_LIQUIDITY_OBS and math.isfinite(float(dollar.mean())):
            adv = float(dollar.mean())
    return adv, last, last_day


def screen_universe(factsheets: Mapping[str, FactSheet], quant: Mapping[str, ViewRecord], prices: PriceData,
                    as_of: str, *, top_n: int, min_dollar_volume: float, min_price: float,
                    held: Iterable[str] = ()) -> ScreenResult:
    """Rank the universe by quant probability and choose the deep-dive set (module docstring).

    ``factsheets``: every universe name priced at ``as_of``; ``quant``: the quant view per
    ticker, scored over that whole cross-section (a missing or abstaining view makes a
    name ineligible); ``held``: tickers currently held (non-zero weight). Deterministic:
    the same inputs give the same result whatever their order."""
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    as_of_ts = pd.Timestamp(as_of)
    p = prices.as_of(str(as_of_ts.date()))
    held_set = {str(t) for t in held}
    rows = []
    for t in sorted(factsheets):
        q = quant.get(t)
        prob = float(q.view.prob_outperform) if q is not None and not q.view.abstain else None
        adv, last, last_day = _liquidity(p, t)
        if prob is None or not math.isfinite(prob):
            prob, reason = None, "the quant has no view (no scorable factor)"
        elif last is None or last_day is None:
            reason = "no price on or before the decision close"
        elif (as_of_ts - last_day).days > STALE_DAYS:
            reason = f"stale price: last close {last_day.date()} is more than {STALE_DAYS} days old"
        elif last < min_price:
            reason = f"price {last:.2f} below the minimum {min_price:.2f}"
        elif adv is None:
            reason = f"liquidity unknown (fewer than {MIN_LIQUIDITY_OBS} days of dollar volume)"
        elif adv < min_dollar_volume:
            reason = f"20-day average dollar volume {adv:,.0f} below the minimum {min_dollar_volume:,.0f}"
        else:
            reason = ""
        rows.append((t, prob, reason, adv, last))
    eligible = sorted((r for r in rows if not r[2]), key=lambda r: (-r[1], r[0]))
    rank = {r[0]: i for i, r in enumerate(eligible, start=1)}
    band = [r[0] for r in eligible if r[1] > 0.5][:top_n]
    extra = sorted(t for t in held_set if t in factsheets and t not in band)
    deep = [*band, *extra]
    order = [r[0] for r in eligible] + sorted(r[0] for r in rows if r[2])
    by_ticker = {r[0]: r for r in rows}
    entries = []
    for t in order:
        _, prob, reason, adv, last = by_ticker[t]
        why = TOP_BAND if t in band else (HELD if t in extra else "")
        entries.append(ScreenEntry(ticker=t, quant_prob=prob, eligible=not reason, reason=reason, rank=rank.get(t),
                                   adv_usd=adv, last_price=last, held=t in held_set, deep_dive=bool(why),
                                   deep_dive_reason=why))
    return ScreenResult(as_of=str(as_of_ts.date()), top_n=top_n, min_dollar_volume=float(min_dollar_volume),
                        min_price=float(min_price), entries=entries, deep_dive=deep)
