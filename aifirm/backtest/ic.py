"""Signal-level evaluation: per-date rank information coefficient (IC), its significance,
and how long it would take to know (research report 7.3, 7.4 and 9.3 item 6).

A portfolio's return mixes stock selection with sizing, cash, costs and luck. The IC
isolates the forecasts: on each decision date, did the names the committee rated more
likely to beat the benchmark actually do better? Because the date is the unit of
replication (seeds add no market information), significance is judged across dates.

Contract
--------
* **Forward excess return** of a ticker for a decision at the close of ``d``: its total
  return minus the benchmark's over the same window, from the *open of the execution day*
  (the first trading day after ``d``, when the backtest fills) to the *close of the
  resolution date* (``horizon_days`` trading days after ``d``: the same ``resolve_on``
  date the forecast is scored at). Total return uses adjusted prices: the exit is the
  adjusted close and the entry is that day's open put on the adjusted basis with the same
  day's adjustment factor (``open * adj_close / close``), so dividends count and the
  overnight move from ``d``'s close to the execution open, which no trade could capture,
  is excluded. Without open prices the entry is the adjusted close of ``d`` (the
  forecasts' own window). Closes are forward-filled at most :data:`MAX_PRINT_LAG_ROWS`
  rows; anything still missing makes that ticker-date unknown and it is left out, never
  guessed.
* **Ex post only.** Forward returns use prices after ``d`` by construction. They are
  computed after the backtest has finished, never reach a decision, and use no price
  after the backtest's ``end``: a date whose window ends after ``end`` is not evaluated.
* **Per-date IC**: Spearman rank correlation (average ranks for ties) between the
  probabilities and the forward excess returns of the tickers that have both. A date with
  fewer than ``min_names`` such tickers, or no variation in either, has no IC (it is not
  counted as zero).
* **t-stat of the mean IC**: ``mean / std * sqrt(n_dates)`` when decision dates are at
  least one horizon apart. When ``rebalance_every_days < horizon_days`` the windows of
  consecutive dates overlap, the ICs are autocorrelated and the plain t-stat overstates
  significance, so Newey-West (Bartlett kernel) with ``ceil(h / r) - 1`` lags is used
  instead (:func:`overlap_lags`). The method is recorded with every statistic.
* **Power**: the number of decision dates needed for the observed mean IC to reach
  ``t = 2``, ``n = (2 * s / IC)^2``, and for 80% power at the 5% level (two-sided),
  ``n = ((z_0.975 + z_0.80) * s / IC)^2 = (2.80 * s / IC)^2``, where ``s`` is the per-date
  IC standard deviation (the Newey-West long-run one when dates overlap). Years =
  dates x ``rebalance_every_days`` / 252.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from ..data.prices import PriceData
from .metrics import (
    TRADING_DAYS,
    long_run_variance,
    mean_t_stat,
    min_track_record_length,
    sharpe,
)

MAX_PRINT_LAG_ROWS = 5          # a missing close is bridged by the previous one at most this many rows back
Z_T2 = 2.0
Z_POWER_80 = float(norm.ppf(0.975) + norm.ppf(0.80))     # 2.80: 80% power, 5% two-sided

POOL = "pool"                          # the firm's final probability (the strategy's signal; see probability_tables)
POOL_EQUAL = "pool_equal_weight"       # the same with the LLM pool's weights equal (P7 shadow)
QUANT_ONLY = "quant_only"              # the quant-only baseline's pool (its own run)
AGENT_PREFIX = "agent:"

FORWARD_RETURN_DEFINITION = (
    "forward excess return = ticker total return minus benchmark total return from the open of the execution "
    "day (the trading day after the decision) to the close of the forecast's resolution date (horizon_days "
    "trading days after the decision), on adjusted prices")


def label_name(label: str) -> str:
    """Human name of an IC series label."""
    if label == POOL:
        return "Firm's final probability (traded)"
    if label == POOL_EQUAL:
        return "Final probability, equal-weight pool"
    if label == QUANT_ONLY:
        return "Quant-only baseline"
    if label.startswith(AGENT_PREFIX):
        return f"{label[len(AGENT_PREFIX):]} analyst"
    return label


def overlap_lags(horizon_days: int, step_days: int) -> int:
    """Newey-West lags for per-date statistics: ``ceil(h / r) - 1`` (0 when dates are at
    least one horizon apart, so the forward windows do not overlap)."""
    if horizon_days <= 0 or step_days <= 0:
        raise ValueError("horizon_days and step_days must be positive")
    return max(0, math.ceil(horizon_days / step_days) - 1)


def t_method(lags: int) -> str:
    return "iid" if lags == 0 else f"Newey-West ({lags} lag{'s' if lags != 1 else ''})"


# --------------------------------------------------------------------------- inputs


def forward_excess_returns(prices: PriceData, decision_dates: Iterable[str | pd.Timestamp],
                           tickers: Sequence[str], benchmark: str, horizon_days: int,
                           end: str | pd.Timestamp) -> pd.DataFrame:
    """Realised forward excess returns (fractions) per decision date (rows) and ticker
    (columns), as defined in the module docstring. Rows whose window ends after ``end`` (or
    after the last price) are absent; unknown ticker-dates are NaN."""
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    p = prices.as_of(end)                                        # nothing after the backtest's end
    cols = [*dict.fromkeys([*tickers, benchmark])]
    close = p.close.reindex(columns=cols)
    adj = p.adj_close.reindex(index=close.index, columns=cols)
    adj_ff = adj.ffill(limit=MAX_PRINT_LAG_ROWS)
    if p.open is not None:
        factor = adj / close                                     # same-day adjustment factor
        entry_px = p.open.reindex(index=close.index, columns=cols) * factor
    else:
        entry_px = None
    cal = close.index
    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in decision_dates:
        ts = pd.Timestamp(d)
        pos = int(cal.searchsorted(ts, side="right")) - 1
        if pos < 0 or pos + horizon_days >= len(cal):
            continue                                             # before the data, or not resolved by end
        exit_px = adj_ff.iloc[pos + horizon_days]
        start_px = entry_px.iloc[pos + 1] if entry_px is not None else adj_ff.iloc[pos]
        ret = exit_px / start_px - 1.0
        ret = ret.where(np.isfinite(ret) & (start_px > 0) & (exit_px > 0))
        rows[cal[pos]] = ret[list(tickers)] - ret[benchmark]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=list(tickers))
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def probability_tables(decisions: Sequence[Mapping], quant_decisions: Sequence[Mapping] | None = None
                       ) -> dict[str, pd.DataFrame]:
    """Probability of beating the benchmark per decision date (rows) and ticker (columns),
    one table per signal, from the backtest's decision records:

    * :data:`POOL`: the firm's final probability, the one the strategy traded on (after any
      challenge round). For a deep-dive name it is the committee's aggregate: in overlay
      mode (the default) the quant core plus the bounded LLM tilt, in pool mode the
      believability-weighted pool of every member. For a screened name outside the deep
      dive it is the quant's probability;
    * :data:`POOL_EQUAL`: the same with the LLM views pooled at equal weights;
    * ``agent:<name>``: each analyst's sealed probability (the mean over its samples);
    * :data:`QUANT_ONLY`: the quant-only baseline's pool, when ``quant_decisions`` is given.

    A ticker whose views all abstained has no probability on that date (NaN), not 0.5:
    an abstention is not a forecast."""
    tables: dict[str, dict[pd.Timestamp, dict[str, float]]] = defaultdict(dict)

    def add(label: str, d: pd.Timestamp, t: str, p: object) -> None:
        if isinstance(p, (int, float)) and math.isfinite(float(p)):
            tables[label].setdefault(d, {})[t] = float(p)

    for rec in decisions:
        d = pd.Timestamp(rec["as_of"])
        counted = rec.get("n_views") or {}
        for t, p in (rec.get("pooled_prob") or {}).items():
            if counted.get(t, 1) > 0:
                add(POOL, d, t, p)
                add(POOL_EQUAL, d, t, (rec.get("prob_equal_weight") or {}).get(t))
        for t, by_agent in (rec.get("agent_probs") or {}).items():
            for agent, p in by_agent.items():
                add(AGENT_PREFIX + agent, d, t, p)
    for rec in quant_decisions or ():
        d = pd.Timestamp(rec["as_of"])
        counted = rec.get("n_views") or {}
        for t, p in (rec.get("pooled_prob") or {}).items():
            if counted.get(t, 1) > 0:
                add(QUANT_ONLY, d, t, p)
    out = {}
    for label, by_date in tables.items():
        df = pd.DataFrame.from_dict(by_date, orient="index")
        df.index = pd.DatetimeIndex(df.index)
        out[label] = df.sort_index()
    return out


# --------------------------------------------------------------------------- statistics


def rank_ic(probs: pd.Series, forward: pd.Series, min_names: int) -> tuple[float, int]:
    """``(Spearman IC, n)`` over the tickers present in both series; NaN IC when fewer
    than ``min_names`` tickers or no variation in either ranking."""
    both = pd.concat([probs.rename("p"), forward.rename("r")], axis=1, join="inner").dropna()
    n = len(both)
    if n < max(2, min_names):
        return float("nan"), n
    a, b = rankdata(both["p"].to_numpy()), rankdata(both["r"].to_numpy())
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), n
    return float(np.corrcoef(a, b)[0, 1]), n


@dataclass
class ICStats:
    """Per-date rank IC of one signal and its summary statistics (see module docstring)."""

    label: str
    n_dates: int                 # dates with an IC
    mean: float
    std: float                   # sample std of the per-date ICs (ddof=1)
    long_run_std: float          # Newey-West long-run std (equals ``std`` when lags == 0)
    t_stat: float
    lags: int
    hit_rate: float              # share of dates with IC > 0
    avg_names: float             # names per date behind each IC
    per_date: pd.Series = field(repr=False)

    @property
    def method(self) -> str:
        return t_method(self.lags)

    @property
    def name(self) -> str:
        return label_name(self.label)


def _summary(label: str, ics: pd.Series, names: pd.Series, lags: int) -> ICStats:
    ics = ics.dropna()
    n = len(ics)
    std = float(ics.std(ddof=1)) if n >= 2 else float("nan")
    lrv = long_run_variance(ics, lags)
    return ICStats(label=label, n_dates=n, mean=float(ics.mean()) if n else float("nan"), std=std,
                   long_run_std=math.sqrt(lrv) if math.isfinite(lrv) else float("nan"),
                   t_stat=mean_t_stat(ics, lags), lags=lags,
                   hit_rate=float((ics > 0).mean()) if n else float("nan"),
                   avg_names=float(names.reindex(ics.index).mean()) if n else float("nan"), per_date=ics)


def ic_stats(label: str, probs: pd.DataFrame, forward: pd.DataFrame, min_names: int, lags: int) -> ICStats:
    """Per-date rank IC of ``probs`` against ``forward`` (dates as rows, tickers as columns)
    and its mean, std, t-stat (iid or Newey-West with ``lags``) and hit rate."""
    ics, names = {}, {}
    for d in probs.index.intersection(forward.index):
        ic, n = rank_ic(probs.loc[d], forward.loc[d], min_names)
        ics[d], names[d] = ic, n
    return _summary(label, pd.Series(ics, dtype=float), pd.Series(names, dtype=float), lags)


@dataclass
class ICDifference:
    """Paired per-date IC difference ``a - b`` over the dates where both have an IC: the
    right test for "does design a rank stocks better than design b" (research 7.4, S4)."""

    a: str
    b: str
    n_dates: int
    mean_diff: float
    std: float
    t_stat: float
    lags: int
    per_date: pd.Series = field(repr=False)

    @property
    def method(self) -> str:
        return t_method(self.lags)


def paired_ic_difference(a: ICStats, b: ICStats) -> ICDifference:
    diff = (a.per_date - b.per_date).dropna()
    lags = max(a.lags, b.lags)
    n = len(diff)
    return ICDifference(a=a.label, b=b.label, n_dates=n, mean_diff=float(diff.mean()) if n else float("nan"),
                        std=float(diff.std(ddof=1)) if n >= 2 else float("nan"), t_stat=mean_t_stat(diff, lags),
                        lags=lags, per_date=diff)


def _signal_order(label: str) -> tuple[int, str]:
    rank = {POOL: 0, POOL_EQUAL: 1, QUANT_ONLY: 2}
    return rank.get(label, 3), label


def information_coefficients(decisions: Sequence[Mapping], prices: PriceData, tickers: Sequence[str],
                             benchmark: str, horizon_days: int, step_days: int, end: str,
                             min_names: int, quant_decisions: Sequence[Mapping] | None = None
                             ) -> tuple[dict[str, ICStats], list[ICDifference]]:
    """IC statistics for every signal in the decision records (see
    :func:`probability_tables`) and the paired tests: pool vs quant-only (when the
    quant-only baseline ran) and believability-weighted vs equal-weight pool."""
    tables = probability_tables(decisions, quant_decisions)
    dates = sorted({d for df in tables.values() for d in df.index})
    forward = forward_excess_returns(prices, dates, tickers, benchmark, horizon_days, end)
    lags = overlap_lags(horizon_days, step_days)
    stats = {label: ic_stats(label, tables[label], forward, min_names, lags)
             for label in sorted(tables, key=_signal_order)}
    diffs = []
    for a, b in ((POOL, QUANT_ONLY), (POOL, POOL_EQUAL)):
        if a in stats and b in stats:
            diffs.append(paired_ic_difference(stats[a], stats[b]))
    return stats, diffs


# --------------------------------------------------------------------------- power


@dataclass
class PowerAnalysis:
    """How many decision dates (and years) it would take to know, at the observed IC.

    ``dates_for_t2`` / ``dates_for_80pct`` are infinite when the mean IC is not positive
    (no amount of data at this IC would show skill) and NaN when there is no IC at all."""

    label: str
    ic_mean: float
    ic_std: float                # the per-date std used (long-run std when dates overlap)
    n_dates: int
    step_days: int
    dates_for_t2: float
    dates_for_80pct: float
    strategy_sharpe: float
    min_trl_years: float         # MinTRL (95% that the true Sharpe > 0) at the strategy's Sharpe
    track_years: float           # length of the strategy's daily return series in years
    avg_names: float = float("nan")     # breadth: names behind each date's IC

    @property
    def have_years(self) -> float:
        return self.n_dates * self.step_days / TRADING_DAYS

    @property
    def years_for_t2(self) -> float:
        return self.dates_for_t2 * self.step_days / TRADING_DAYS

    @property
    def years_for_80pct(self) -> float:
        return self.dates_for_80pct * self.step_days / TRADING_DAYS

    @property
    def anecdotal(self) -> bool:
        """True unless the sample already has the dates for 80% power at the observed IC."""
        return not (math.isfinite(self.dates_for_80pct) and self.n_dates >= self.dates_for_80pct)

    def ic_sentence(self) -> str:
        if not math.isfinite(self.ic_mean):
            return "No date had enough names with a forecast and a realised return to compute an IC."
        have = f"the backtest has {self.n_dates} ({self.have_years:.1f} years)"
        if not self.ic_mean > 0:
            return (f"The mean IC ({self.ic_mean:+.3f}) is not positive, so no amount of data at this IC would "
                    f"show skill; {have}.")
        breadth = f" and breadth (~{self.avg_names:.0f} names per date)" if math.isfinite(self.avg_names) else ""
        return (f"At this IC ({self.ic_mean:+.3f} per date, std {self.ic_std:.3f}){breadth} you would need ~"
                f"{self.dates_for_80pct:,.0f} decision dates (~{self.years_for_80pct:,.1f} years at one decision "
                f"every {self.step_days} trading days) for 80% power at the 5% level, and ~"
                f"{self.dates_for_t2:,.0f} (~{self.years_for_t2:,.1f} years) to reach t = 2; {have}.")

    def sharpe_sentence(self) -> str:
        if not math.isfinite(self.min_trl_years) and self.min_trl_years != math.inf:
            return "The strategy's minimum track record could not be computed (too few returns or no variation)."
        if self.min_trl_years == math.inf:
            return (f"At the strategy's Sharpe ({self.strategy_sharpe:.2f}) no track record is long enough for 95% "
                    "confidence that the true Sharpe is above zero.")
        return (f"At the strategy's Sharpe ({self.strategy_sharpe:.2f}) you would need ~{self.min_trl_years:,.1f} "
                f"years of daily returns for 95% confidence that the true Sharpe is above zero (MinTRL); the "
                f"backtest has {self.track_years:.1f}.")


def _dates_needed(z: float, mean: float, std: float) -> float:
    if not (math.isfinite(mean) and math.isfinite(std)):
        return float("nan")
    if mean <= 0:
        return math.inf
    return (z * std / mean) ** 2


def power_analysis(stats: ICStats, step_days: int, returns: pd.Series) -> PowerAnalysis:
    """Dates needed for t = 2 and for 80% power at the observed mean IC of ``stats`` (with
    its long-run std, so overlapping dates are not counted as independent), plus MinTRL in
    years from the strategy's daily ``returns``."""
    s = stats.long_run_std
    r = returns.dropna()
    mtrl = min_track_record_length(r)
    return PowerAnalysis(label=stats.label, ic_mean=stats.mean, ic_std=s, n_dates=stats.n_dates, step_days=step_days,
                         dates_for_t2=_dates_needed(Z_T2, stats.mean, s),
                         dates_for_80pct=_dates_needed(Z_POWER_80, stats.mean, s),
                         strategy_sharpe=sharpe(r), min_trl_years=mtrl / TRADING_DAYS if mtrl != math.inf else mtrl,
                         track_years=len(r) / TRADING_DAYS, avg_names=stats.avg_names)
