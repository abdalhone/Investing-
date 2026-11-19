"""Composition-matched random books: the null distribution for "is the stock selection
better than chance?" (research report 7.4 S4 and 9.3 item 6).

A benchmark comparison mixes selection with everything else the strategy does (how much
cash it holds, how many names, how concentrated, when it trades, what it pays). A random
book that copies all of that and differs *only* in which names it holds isolates the
selection. If the strategy does not beat about 95% of such books, its stock picks are not
distinguishable from random picks, whatever its return.

Contract
--------
* **Schedule.** The strategy's recorded decisions become a list of rebalances: one per
  decision that produced orders, executed at the open of the next trading day (the fill
  day). A decision without orders leaves every book as it is (drifting), exactly as the
  strategy's book was left. Each rebalance has a target weight vector: active names, the
  core weight on the benchmark, and cash for the rest.
* **Random books.** At each rebalance every random book gets the strategy's weights
  (same count, same sizes, same core, same cash), each attached to a name drawn at random
  from that decision's eligible universe: the universe names with a close on or within
  :data:`ELIGIBLE_LAG_ROWS` rows before the decision date (point-in-time: a name not yet
  listed is never drawn), and always including the strategy's own names. Holding periods
  are matched too: while the strategy keeps a name from one rebalance to the next, the
  random book keeps the name standing in for it (re-sized to the new weight), and when
  the strategy buys a new name the random book draws a new one it does not already hold.
  So the random books trade about as much as the strategy; redrawing every name at every
  date would charge them turnover the strategy never paid. With ``sector_matched`` each
  replacement is drawn from the same sector as the name it replaces (unlabelled names
  form one group); if a sector has too few free names the draw falls back to any free
  eligible name.
* **Accounting** (the same for the strategy's own book, so the comparison is like for
  like): buy at the fill-day open, hold (weights drift with prices) to the next fill-day
  open, mark at every close. Prices are adjusted (total return): closes are adjusted
  closes, opens are put on the adjusted basis with the same day's factor
  (``open * adj_close / close``); a missing open is replaced by the previous adjusted
  close. At each rebalance every book pays ``cost_rate * traded fraction of its value``,
  where the traded fraction is ``sum |new weight - drifted weight|`` over securities and
  ``cost_rate`` is the strategy's realised cost per dollar traded in the engine (the
  same turnover-proportional cost rate). Cash earns nothing.
* **Outputs.** Total return and annualised daily Sharpe for each book; the strategy's
  percentile (share of random books strictly below it, ties counted half) and the
  empirical one-sided p-value ``(1 + #books >= strategy) / (n_books + 1)``; the same
  percentile gross of costs. The strategy's own figure here comes from this accounting,
  not from the engine (whose fills, ADV caps and no-trade band differ slightly); the
  engine's total return is reported beside it.
* Reproducible: the same seed, decisions and prices give the same books. Vectorised with
  numpy: per rebalance one ``(books x names)`` draw and one matrix product.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.prices import PriceData
from .metrics import TRADING_DAYS

ELIGIBLE_LAG_ROWS = 5          # a name counts as priced if it has a close this many rows before the decision
VERDICT_PERCENTILE = 95.0       # below this the selection is "not distinguishable from random books"
MAX_DISTINCT = 10 ** 15         # distinct-book counts are capped here (only "fewer than n_books" matters)


@dataclass
class BookSchedule:
    """The strategy's book as rebalances in matrix form (the input of the null).

    ``assets`` are the universe names followed by the benchmark (last column). Row ``i`` of
    ``weights`` is the target at rebalance ``i`` (active weights, then the core on the
    benchmark column; cash is ``1 - sum``), executed at the open of ``days[fill_pos[i]]``.
    ``eligible[i]`` marks the names the null may draw at that decision."""

    days: pd.DatetimeIndex
    assets: list[str]
    fill_pos: np.ndarray            # (m,) strictly increasing positions in ``days``
    weights: np.ndarray             # (m, n_assets)
    eligible: np.ndarray            # (m, n_names) bool
    sectors: list[str]              # per name; "" = unlabelled

    @property
    def n_names(self) -> int:
        return len(self.assets) - 1


def build_schedule(decisions: Sequence[Mapping], prices: PriceData, days: pd.DatetimeIndex,
                   names: Sequence[str], benchmark: str,
                   sectors: Mapping[str, str] | None = None) -> BookSchedule:
    """Turn the backtest's decision records (``as_of``, ``orders``, ``target``,
    ``core_weight``) into a :class:`BookSchedule`. Decisions without orders, or on the last
    day (no next open), are not rebalances. A target name outside ``names`` (cannot happen
    for a valid run) raises ``ValueError`` rather than being dropped silently."""
    names = list(names)
    assets = [*names, benchmark]
    col = {t: i for i, t in enumerate(assets)}
    close = prices.close.reindex(columns=names).ffill(limit=ELIGIBLE_LAG_ROWS)
    pos_of = {d: i for i, d in enumerate(days)}
    fill_pos, rows, elig = [], [], []
    for rec in decisions:
        d = pd.Timestamp(rec["as_of"])
        if not rec.get("orders") or d not in pos_of or pos_of[d] + 1 >= len(days):
            continue
        w = np.zeros(len(assets))
        for t, x in (rec.get("target") or {}).items():
            if t not in col:
                raise ValueError(f"target name {t} is not in the universe {names}")
            w[col[t]] += float(x)
        w[-1] += float(rec.get("core_weight") or 0.0)
        priced = close.loc[d].to_numpy(dtype=float) if d in close.index else np.full(len(names), np.nan)
        fill_pos.append(pos_of[d] + 1)
        rows.append(w)
        elig.append(np.isfinite(priced) & (priced > 0) | (w[:-1] != 0))
    labels = sectors or {}
    return BookSchedule(days=days, assets=assets, fill_pos=np.asarray(fill_pos, dtype=int),
                        weights=np.asarray(rows, dtype=float).reshape(len(rows), len(assets)),
                        eligible=np.asarray(elig, dtype=bool).reshape(len(elig), len(names)),
                        sectors=[str(labels.get(t, "") or "") for t in names])


def _price_arrays(prices: PriceData, days: pd.DatetimeIndex, assets: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted closes (forward-filled) and adjusted opens on ``days``, shape (T, assets)."""
    close = prices.close.reindex(columns=list(assets))
    adj = prices.adj_close.reindex(index=close.index, columns=list(assets))
    ac = adj.ffill()
    if prices.open is not None:
        ao = prices.open.reindex(index=close.index, columns=list(assets)) * (adj / close)
        ao = ao.where(ao > 0).fillna(ac.shift(1))           # a missing open: the previous adjusted close
    else:
        ao = ac.shift(1)
    return ac.reindex(days).to_numpy(dtype=float), ao.reindex(days).to_numpy(dtype=float)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Growth ratios with unknown values (no price yet) as 1.0: a name without prices has
    no move, and it is never held (only priced names are drawn), so this only keeps
    ``0 * NaN`` out of the matrix products."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / den
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def value_paths(schedule: BookSchedule, ac: np.ndarray, ao: np.ndarray, weights: Iterator[np.ndarray],
                n_books: int, cost_rate: float) -> np.ndarray:
    """Book values (``n_books`` x days, starting at 1.0 in cash) under the accounting of the
    module docstring. ``weights`` yields one ``(n_books, n_assets)`` matrix per rebalance."""
    t_len, m = len(schedule.days), len(schedule.fill_pos)
    values = np.ones((n_books, t_len))
    prev_w: np.ndarray | None = None
    entry = np.ones(ac.shape[1])
    v_post = np.ones(n_books)
    for i, f in enumerate(schedule.fill_pos):
        w = next(weights)
        if prev_w is None:
            v_pre, drift = np.ones(n_books), np.zeros_like(w)
        else:
            held = prev_w * _ratio(ao[f], entry)
            rel = held.sum(axis=1) + (1.0 - prev_w.sum(axis=1))
            v_pre, drift = v_post * rel, held / rel[:, None]
        traded = np.abs(w - drift).sum(axis=1)
        v_post = v_pre * (1.0 - cost_rate * traded)
        nxt = schedule.fill_pos[i + 1] if i + 1 < m else t_len
        growth = _ratio(ac[f:nxt], ao[f])                       # (L, assets)
        values[:, f:nxt] = v_post[:, None] * (w @ growth.T + (1.0 - w.sum(axis=1))[:, None])
        prev_w, entry = w, ao[f]
    return values


def _strategy_weights(schedule: BookSchedule) -> Iterator[np.ndarray]:
    for w in schedule.weights:
        yield w[None, :]


def _draw(rng: np.random.Generator, used: np.ndarray, allowed: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k`` distinct random columns per row among ``allowed`` columns not ``used``;
    returns ``(indices (rows, k), ok (rows, k))`` where ``ok`` is False if a row ran out."""
    keys = rng.random(used.shape)
    keys[:, ~allowed] = np.inf
    keys[used] = np.inf
    idx = np.argsort(keys, axis=1)[:, :k]
    return idx, np.isfinite(np.take_along_axis(keys, idx, axis=1))


def _random_weights(schedule: BookSchedule, n_books: int, rng: np.random.Generator,
                    sector_matched: bool) -> Iterator[np.ndarray]:
    """Weight matrices of the random books, rebalance by rebalance (module docstring)."""
    n = schedule.n_names
    rows = np.arange(n_books)[:, None]
    shadow = np.full((n_books, n), -1, dtype=int)     # strategy name j -> the name standing in for it
    prev_active = np.zeros(n, dtype=bool)
    sector_arr = np.asarray(schedule.sectors, dtype=object)
    for i, w in enumerate(schedule.weights):
        active = w[:n] != 0
        elig = schedule.eligible[i]
        kept = np.flatnonzero(active & prev_active)
        new = np.flatnonzero(active & ~prev_active)
        shadow[:, ~active] = -1
        used = np.zeros((n_books, n), dtype=bool)
        redraw: list[tuple[int, np.ndarray]] = []          # kept names whose stand-in lost its price
        for j in kept:
            ok = elig[shadow[:, j]]
            used[rows[ok, 0], shadow[ok, j]] = True
            if not ok.all():
                redraw.append((j, ~ok))
        groups = [new] if not sector_matched else [new[sector_arr[new] == s] for s in dict.fromkeys(sector_arr[new])]
        for group in groups:
            if not len(group):
                continue
            allowed = elig & (sector_arr == sector_arr[group[0]]) if sector_matched else elig
            idx, ok = _draw(rng, used, allowed, len(group))
            if not ok.all():                                 # too few free names in the sector: any eligible
                short = ~ok.all(axis=1)
                idx_any, _ = _draw(rng, used[short], elig, len(group))
                idx[short] = idx_any
            shadow[:, group] = idx
            used[rows, idx] = True
        for j, need in redraw:
            allowed = elig & (sector_arr == sector_arr[j]) if sector_matched else elig
            idx, ok = _draw(rng, used[need], allowed, 1)
            if not ok.all():
                idx[~ok[:, 0]], _ = _draw(rng, used[need][~ok[:, 0]], elig, 1)
            shadow[need, j] = idx[:, 0]
            used[np.flatnonzero(need), idx[:, 0]] = True
        mat = np.zeros((n_books, len(schedule.assets)))
        act = np.flatnonzero(active)
        mat[rows, shadow[:, act]] = w[act][None, :]
        mat[:, n] = w[n]                                     # the core stays on the benchmark
        prev_active = active
        yield mat


def _sharpe_rows(values: np.ndarray) -> np.ndarray:
    r = values[:, 1:] / values[:, :-1] - 1.0
    sd = r.std(axis=1, ddof=1) if r.shape[1] > 1 else np.zeros(len(r))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = r.mean(axis=1) / sd * math.sqrt(TRADING_DAYS)
    return np.where(sd > 0, s, 0.0)


def ordinal(x: float) -> str:
    """``62.4 -> "62nd"``, ``1 -> "1st"``, ``13 -> "13th"`` (rounded to a whole percentile)."""
    n = round(x)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _percentile(sample: np.ndarray, x: float) -> float:
    return float(100.0 * ((sample < x).mean() + 0.5 * (sample == x).mean()))


def _p_value(sample: np.ndarray, x: float) -> float:
    return float((1 + int((sample >= x).sum())) / (len(sample) + 1))


@dataclass
class RandomBooksResult:
    """Where the strategy sits in the composition-matched null (see module docstring)."""

    n_books: int
    seed: int
    sector_matched: bool
    cost_rate_bps: float                    # cost per dollar traded charged to every book
    n_rebalances: int
    strategy_total_return_pct: float        # the strategy's own book in the null's accounting
    strategy_sharpe: float
    engine_total_return_pct: float          # the engine's figure over the same period, for reference
    percentile_return: float                # 0-100
    p_value_return: float
    percentile_sharpe: float
    p_value_sharpe: float
    gross_percentile_return: float          # nobody pays costs
    return_quantiles_pct: dict[str, float]  # p5 / p50 / p95 of the random books' total return
    sharpe_quantiles: dict[str, float]
    avg_names_held: float
    avg_eligible: float
    min_distinct_books: float               # fewest distinct name sets at a rebalance with an active name
    notes: list[str] = field(default_factory=list)
    # None: the engine figure is the whole run. A date: the schedule holds only the decisions
    # from that date on (the clean ones), and the engine figure is its NAV return from that
    # date's close to the end, the same period (the engine enters it with the book it held).
    engine_period_start: str | None = None

    @property
    def distinguishable(self) -> bool:
        return self.percentile_return >= VERDICT_PERCENTILE

    def verdict(self) -> str:
        where = (f"{ordinal(self.percentile_return)} percentile of {self.n_books:,} composition-matched random books "
                 f"(one-sided p = {self.p_value_return:.3f})")
        if self.distinguishable:
            return f"the strategy's total return is at the {where}: its stock selection beat random picks"
        return (f"the strategy's total return is at the {where}: not distinguishable from random books "
                f"(below the {VERDICT_PERCENTILE:.0f}th percentile)")


def random_books(schedule: BookSchedule, prices: PriceData, n_books: int, seed: int, cost_rate: float,
                 engine_total_return_pct: float, sector_matched: bool = False,
                 engine_period_start: str | None = None) -> RandomBooksResult:
    """Simulate ``n_books`` composition-matched random books and the strategy's own book
    through the same vectorised accounting and place the strategy in the distribution.
    ``engine_total_return_pct`` is the engine's NAV return over the period the schedule
    covers, from ``engine_period_start`` (None: the whole run), shown for reference.
    Raises ``ValueError`` when there is nothing to compare (no rebalance, ``n_books < 1``)."""
    if n_books < 1:
        raise ValueError("n_books must be at least 1")
    if not len(schedule.fill_pos):
        raise ValueError("the strategy never traded, so there is no composition to match")
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError(f"cost_rate must be finite and >= 0, got {cost_rate!r}")
    ac, ao = _price_arrays(prices, schedule.days, schedule.assets)

    def run(rate: float) -> tuple[np.ndarray, np.ndarray]:
        strat = value_paths(schedule, ac, ao, _strategy_weights(schedule), 1, rate)
        books = value_paths(schedule, ac, ao, _random_weights(schedule, n_books, np.random.default_rng(seed),
                                                              sector_matched), n_books, rate)
        return strat, books

    strat, books = run(cost_rate)
    gross_strat, gross_books = run(0.0) if cost_rate > 0 else (strat, books)
    s_ret, b_ret = float(strat[0, -1] - 1.0), books[:, -1] - 1.0
    s_sh, b_sh = float(_sharpe_rows(strat)[0]), _sharpe_rows(books)
    held = (schedule.weights[:, :-1] != 0).sum(axis=1)
    n_elig = schedule.eligible.sum(axis=1)
    # the narrowest rebalance with at least one active name (C(eligible, held) possible name sets)
    combos = [(min(math.comb(int(e), int(k)), MAX_DISTINCT), int(k), int(e))
              for e, k in zip(n_elig, held, strict=True) if k > 0]
    distinct, k_min, e_min = min(combos) if combos else (1, 0, int(n_elig.min()) if len(n_elig) else 0)
    notes = []
    if not combos:
        notes.append("The strategy never held an active name, so every random book equals it: the null says nothing.")
    elif distinct < n_books:
        books = "book" if distinct == 1 else "books"
        notes.append(f"At one rebalance the strategy held {k_min} of the {e_min} eligible names, which leaves only "
                     f"{distinct:,} distinct {books}: many of the {n_books:,} random books repeat, and a narrow "
                     "universe makes a weak null.")
    q = (5, 50, 95)
    return RandomBooksResult(
        n_books=n_books, seed=seed, sector_matched=sector_matched, cost_rate_bps=cost_rate * 1e4,
        n_rebalances=len(schedule.fill_pos), strategy_total_return_pct=s_ret * 100, strategy_sharpe=s_sh,
        engine_total_return_pct=engine_total_return_pct,
        percentile_return=_percentile(b_ret, s_ret), p_value_return=_p_value(b_ret, s_ret),
        percentile_sharpe=_percentile(b_sh, s_sh), p_value_sharpe=_p_value(b_sh, s_sh),
        gross_percentile_return=_percentile(gross_books[:, -1], float(gross_strat[0, -1])),
        return_quantiles_pct={f"p{x}": float(np.percentile(b_ret, x) * 100) for x in q},
        sharpe_quantiles={f"p{x}": float(np.percentile(b_sh, x)) for x in q},
        avg_names_held=float(held.mean()), avg_eligible=float(n_elig.mean()),
        min_distinct_books=float(distinct), notes=notes, engine_period_start=engine_period_start)


def realised_cost_rate(decisions: Sequence[Mapping]) -> float:
    """The strategy's cost per dollar traded in the engine: (commission + slippage and
    impact) / traded notional over every fill. 0.0 when nothing was filled."""
    traded = cost = 0.0
    for rec in decisions:
        for f in rec.get("fills") or ():
            traded += abs(float(f["quantity"]) * float(f["price"]))
            cost += float(f.get("commission", 0.0)) + float(f.get("slippage_cost", 0.0))
    return cost / traded if traded > 0 else 0.0
