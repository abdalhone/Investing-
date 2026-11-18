"""Tests for the walk-forward backtest engine and its report, fully offline (synthetic data)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, make_config, make_prices, perturb_after

from aifirm.agents import make_offline_llm
from aifirm.backtest.engine import contamination_cutoff, run_backtest
from aifirm.data.prices import from_frames
from aifirm.report import render_backtest_report

START, END = 300, 559          # indices into the synthetic calendar (300 days of history first)


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


def _cfg(runs_dir, **process):
    cfg = make_config(str(runs_dir), **process)
    cfg.portfolio.min_prob_to_hold = 0.5
    return cfg


def _bt(cfg, prices, mode="quant", start=START, end=END, **kw):
    return run_backtest(cfg, _day(prices, start), _day(prices, end), mode, prices=prices, **LOADERS, **kw)


@pytest.fixture(scope="module")
def quant_result(prices, tmp_path_factory):
    return _bt(_cfg(tmp_path_factory.mktemp("quant")), prices)


@pytest.fixture(scope="module")
def offline_result(prices, tmp_path_factory):
    return _bt(_cfg(tmp_path_factory.mktemp("offline")), prices, mode="offline", start=380, end=500)


# ----------------------------------------------------------------------------- results and baselines


def test_quant_backtest_reports_strategy_and_baselines(quant_result, prices):
    r = quant_result
    days = prices.close.index[START:END + 1]
    assert list(r.nav.index) == list(days)
    assert r.nav.iloc[0] == pytest.approx(100_000.0)               # nothing filled before the first close
    assert len(r.returns) == len(days) - 1 and r.returns.notna().all()
    assert set(r.reports) == {"strategy", "benchmark", "equal_weight"}   # no quant-only baseline of itself
    assert set(r.baseline_returns) == {"benchmark", "equal_weight"}
    assert r.model_cutoff is None and r.pre_cutoff_report is None and r.post_cutoff_report is None
    n_cycles = math.ceil((len(days) - 1) / 21)
    assert len(r.decisions) == n_cycles
    assert r.turnover.sum() > 0 and r.costs.sum() > 0
    assert r.costs_usd["commission"] > 0 and r.costs_usd["slippage"] > 0
    assert {"quant", "committee"} <= set(r.scorecard)
    assert r.usage == {}
    run_dir = Path(r.run_dir)
    assert (run_dir / "ledger.jsonl").exists() and (run_dir / "forecasts.sqlite").exists()
    # benchmark buy-and-hold is the benchmark's own total return
    bench = prices.adj_close["SPY"].iloc[START:END + 1]
    assert r.reports["benchmark"].total_return_pct == pytest.approx((bench.iloc[-1] / bench.iloc[0] - 1) * 100)


def test_offline_backtest_adds_quant_only_baseline_through_the_same_engine(offline_result, prices, tmp_path):
    r = offline_result
    assert set(r.reports) == {"strategy", "benchmark", "equal_weight", "quant_only"}
    assert r.usage["calls"] > 0 and r.usage["cost_usd"] == 0
    # the quant-only baseline is exactly a quant-mode run of the same engine
    q = _bt(_cfg(tmp_path), prices, start=380, end=500)
    pd.testing.assert_series_equal(r.baseline_returns["quant_only"], q.returns, check_names=False)
    # the LLM committee's decisions differ from the quant's (it is a different strategy)
    assert not r.returns.equals(q.returns)


def test_offline_backtest_splits_contaminated_returns_at_the_model_cutoff(offline_result):
    r = offline_result
    assert r.model_cutoff == "2026-06-30"                          # claude-opus-5-5 in MODEL_TRAINING_CUTOFFS
    assert r.pre_cutoff_report is not None and r.pre_cutoff_report.days == len(r.returns)
    assert r.post_cutoff_report is None                            # every synthetic date precedes the cutoff
    md = render_backtest_report(r)
    assert "CONTAMINATION WARNING" in md and "NOT evidence of skill" in md and "OFFLINE MODE" in md


def test_unknown_model_cutoff_counts_everything_as_contaminated(tmp_path):
    cfg = _cfg(tmp_path)
    assert contamination_cutoff(cfg, "quant") is None
    cfg.models.analyst_models = ["some-unlisted-model"]
    assert contamination_cutoff(cfg, "llm") >= str(pd.Timestamp.today().date())


# ----------------------------------------------------------------------------- execution realism


def test_orders_fill_at_the_next_trading_days_open_with_costs(quant_result, prices):
    idx = prices.close.index
    filled = [d for d in quant_result.decisions if d.get("fills")]
    assert filled
    for d in filled:
        decided = pd.Timestamp(d["as_of"])
        next_day = idx[idx.searchsorted(decided, side="right")]
        assert d["fill_date"] == str(next_day.date())
        for f in d["fills"]:
            open_px = prices.open.loc[next_day, f["ticker"]]
            if f["side"] == "buy":
                assert open_px <= f["price"] <= open_px * 1.01       # the open plus slippage
            else:
                assert open_px * 0.99 <= f["price"] <= open_px
            assert f["price"] != pytest.approx(prices.close.loc[decided, f["ticker"]], rel=1e-9)


def test_costs_reduce_returns(prices, tmp_path):
    costly = _bt(_cfg(tmp_path / "a"), prices)
    free_cfg = _cfg(tmp_path / "b")
    free_cfg.costs.commission_bps = free_cfg.costs.slippage_bps = free_cfg.costs.impact_coefficient = 0.0
    free = _bt(free_cfg, prices)
    assert free.costs.sum() == 0 and costly.costs.sum() > 0
    assert costly.nav.iloc[-1] < free.nav.iloc[-1]
    assert costly.reports["strategy"].total_costs_pct > 0


def test_backtest_has_no_look_ahead(prices, tmp_path):
    """Scrambling every price after a date leaves the NAV path and every decision up to that
    date unchanged (P3): decisions only see data up to their close."""
    cut = 450
    base = _bt(_cfg(tmp_path / "a"), prices)
    changed = _bt(_cfg(tmp_path / "b"), perturb_after(prices, _day(prices, cut)))
    pd.testing.assert_series_equal(base.nav.iloc[:cut - START + 1], changed.nav.iloc[:cut - START + 1])
    assert not base.nav.equals(changed.nav)                       # the future really did change

    def upto(r):
        return [{k: d[k] for k in ("as_of", "orders", "target", "pooled_prob")} for d in r.decisions
                if d["as_of"] <= _day(prices, cut)]

    assert upto(base) == upto(changed) and upto(base)


def test_drawdown_freeze_moves_the_active_book_to_cash(tmp_path):
    """A market crash trips the drawdown freeze; from then on the active sleeve stays at 0."""
    p = make_prices()
    crash = np.ones(len(p.close))
    crash[420:440] = 0.97                                          # 20 days of -3% for every stock
    factor = pd.Series(np.cumprod(crash), index=p.close.index)
    close = p.close.copy()
    for t in NAMES:
        close[t] = close[t] * factor
    crashed = from_frames(close, volume=p.volume, open_=close.shift(1).bfill())
    cfg = _cfg(tmp_path)
    cfg.risk.drawdown_derisk, cfg.risk.drawdown_freeze = 0.02, 0.04
    r = _bt(cfg, crashed, start=380, end=520)
    frozen = [d for d in r.decisions if d["frozen"]]
    assert frozen, "the crash should have frozen the book"
    assert all(d["target"] == {} for d in frozen)
    assert any("DRAWDOWN FREEZE" in m for d in frozen for m in d["risk_messages"])


# ----------------------------------------------------------------------------- trials and validation


def test_trial_ledger_counts_distinct_configurations(prices, tmp_path):
    cfg = _cfg(tmp_path)
    first = _bt(cfg, prices, start=480, end=540)
    again = _bt(cfg, prices, start=480, end=540)
    assert first.n_trials == again.n_trials == 1                  # re-running the same config is not a new trial
    cfg.portfolio.kelly_fraction = 0.5
    assert _bt(cfg, prices, start=480, end=540).n_trials == 2
    lines = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and {x["mode"] for x in lines} == {"quant"}
    assert all({"fingerprint", "run_id", "sharpe", "start", "end"} <= set(x) for x in lines)
    report = _bt(cfg, prices, start=480, end=540).reports["strategy"]
    assert report.deflated_sharpe is not None                      # DSR is computed once trials > 1


def test_invalid_arguments_fail_loudly(prices, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        _bt(cfg, prices, mode="yolo")
    with pytest.raises(ValueError, match="needs an LLM"):
        _bt(cfg, prices, mode="llm")
    with pytest.raises(ValueError, match="does not take an llm"):
        _bt(cfg, prices, mode="offline", llm=make_offline_llm())
    with pytest.raises(ValueError, match="before end"):
        _bt(cfg, prices, start=500, end=400)


def test_llm_mode_runs_with_any_llm_client(prices, tmp_path):
    r = _bt(_cfg(tmp_path), prices, mode="llm", start=480, end=530, llm=make_offline_llm())
    assert r.mode == "llm" and "quant_only" in r.reports and r.usage["calls"] > 0


def test_backtest_report_renders_every_section(quant_result):
    md = render_backtest_report(quant_result)
    for heading in ("# Backtest report", "## Performance vs baselines", "## Statistical significance",
                    "## Pre / post model-cutoff split", "## Trading activity and costs", "## Forecast scorecard",
                    "## Honest interpretation"):
        assert heading in md
    assert "Equal weight" in md and "Benchmark buy-and-hold" in md and "Deflated Sharpe" in md
    assert "Not applicable: no language model was used" in md
    assert "survivors" in md


def test_recurring_data_gaps_are_reported_once_with_a_count(prices, tmp_path):
    def fundamentals(t, d):
        if t == "BBB":
            raise RuntimeError(f"no filing for {t} as of {d}")
        return LOADERS["fundamentals_loader"](t, d)

    loaders = {**LOADERS, "fundamentals_loader": fundamentals}
    r = run_backtest(_cfg(tmp_path), _day(prices, 480), _day(prices, 540), "quant", prices=prices, **loaders)
    gaps = [w for w in r.warnings if w.startswith("BBB: SEC fundamentals could not be loaded")]
    n = len(r.decisions)
    assert len(gaps) == 1 and gaps[0].endswith(f"({n} of {n} decisions)") and "<date>" in gaps[0]


# ----------------------------------------------------------------------------- regressions (money accounting)


def _with_dividends(p, quarterly_yield=0.01, every=63):
    """A copy of ``p`` whose adj_close embeds a cash dividend of ``quarterly_yield`` of the
    previous close every ``every`` days for every ticker (Yahoo-style: adj == close at the end)."""
    close = p.close
    div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    div.iloc[every::every] = (close.shift(1) * quarterly_yield).iloc[every::every]
    growth = ((close + div) / close.shift(1)).fillna(1.0)
    adj = growth.cumprod()
    adj = adj / adj.iloc[-1] * close.iloc[-1]
    return from_frames(close, adj_close=adj, volume=p.volume, open_=p.open), div


def _core_only_cfg(runs_dir):
    cfg = _cfg(runs_dir)
    cfg.portfolio.core_weight, cfg.portfolio.min_cash, cfg.portfolio.max_turnover = 1.0, 0.0, 5.0
    cfg.costs.commission_bps = cfg.costs.slippage_bps = cfg.costs.impact_coefficient = 0.0
    return cfg


def test_books_earn_dividends_like_the_benchmark(tmp_path):
    """Regression: books are marked at raw closes, so without dividend cash a book holding
    only the benchmark lagged the (total-return) benchmark by the dividend yield. Holding
    100% SPY at zero cost it must now track the benchmark's total return."""
    p, div = _with_dividends(make_prices())
    from aifirm.pipeline import implied_dividends
    assert np.allclose(implied_dividends(p).to_numpy(), div.to_numpy(), atol=1e-9)
    r = _bt(_core_only_cfg(tmp_path), p)
    assert r.dividends_usd > 0
    from aifirm.report import render_backtest_report
    assert f"Dividends credited to the strategy book: ${r.dividends_usd:,.2f}" in render_backtest_report(r)
    days = p.close.index[START:END + 1]
    n_ex = int((div["SPY"].loc[days[2]:days[-1]] > 0).sum())
    assert n_ex >= 3                                               # several ex-dates while fully invested
    strat = (1 + r.returns.iloc[1:]).prod() - 1                    # from the first fill onwards
    bench = (1 + r.benchmark_returns.iloc[1:]).prod() - 1
    assert strat == pytest.approx(bench, abs=0.003)                # was ~n_ex percentage points short
    adj = p.adj_close["SPY"]
    assert r.reports["benchmark"].total_return_pct == pytest.approx((adj[days[-1]] / adj[days[0]] - 1) * 100)


def test_a_missing_benchmark_print_does_not_drop_the_move_across_the_gap(tmp_path):
    """Regression: a NaN benchmark close zeroed two days of benchmark return."""
    p = make_prices()
    close, adj = p.close.copy(), p.adj_close.copy()
    gap = p.close.index[490]
    close.loc[gap, "SPY"] = np.nan
    adj.loc[gap, "SPY"] = np.nan
    holed = from_frames(close, adj_close=adj, volume=p.volume, open_=p.open)
    r = _bt(_cfg(tmp_path), holed, start=480, end=500)
    a, b = p.close.index[480], p.close.index[500]
    expected = (p.adj_close.loc[b, "SPY"] / p.adj_close.loc[a, "SPY"] - 1) * 100
    assert r.reports["benchmark"].total_return_pct == pytest.approx(expected)


def _spy_path(p, multiplier):
    """``p`` with SPY's close, adj close and open scaled by ``multiplier`` (a Series)."""
    close, adj, open_ = p.close.copy(), p.adj_close.copy(), p.open.copy()
    for frame in (close, adj, open_):
        frame["SPY"] = frame["SPY"] * multiplier
    open_["SPY"] = close["SPY"].shift(1).bfill()
    return from_frames(close, adj_close=adj, volume=p.volume, open_=open_)


def test_drawdown_is_measured_from_the_daily_high_water_mark(tmp_path):
    """Regression: the risk state saw only decision-day NAVs, so a rally inside a cycle that
    was given back never counted as a drawdown. Every daily NAV now raises the peak."""
    p = make_prices()
    idx = p.close.index
    m = pd.Series(1.0, index=idx)
    up, top, down = START + 22, START + 31, START + 42
    m.iloc[up:top + 1] = np.linspace(1.0, 1.3, top + 1 - up)
    m.iloc[top:down + 1] = np.linspace(1.3, 1.0, down + 1 - top)
    r = _bt(_core_only_cfg(tmp_path), _spy_path(p, m), start=START, end=START + 60)
    dec = {d["as_of"]: d for d in r.decisions}
    at = dec[_day(p, down)]
    daily_peak = r.nav.loc[:idx[down]].max()
    decision_peak = max(d["nav"] for d in r.decisions if d["as_of"] <= _day(p, down))
    assert 1 - at["nav"] / decision_peak < 0.15 <= 1 - at["nav"] / daily_peak   # only the daily peak sees it
    assert at["frozen"] and any("DRAWDOWN FREEZE" in m for m in at["risk_messages"])


def test_a_loss_day_between_rebalances_halts_buying_at_the_next_decision(tmp_path):
    """Regression: the daily-loss halt compared only the decision day with the day before."""
    p = make_prices()
    m = pd.Series(1.0, index=p.close.index)
    m.iloc[START + 3:] = 0.90                                       # a -10% day three days into the cycle
    r = _bt(_core_only_cfg(tmp_path), _spy_path(p, m), start=START, end=START + 30)
    nxt = next(d for d in r.decisions if d["as_of"] == _day(p, START + 21))
    assert any("DAILY-LOSS HALT" in msg and "since the last cycle" in msg for msg in nxt["risk_messages"])


def test_trial_count_includes_the_backtest_window(prices, tmp_path):
    """Regression: one configuration run over many windows is many trials for the DSR."""
    cfg = _cfg(tmp_path)
    assert _bt(cfg, prices, start=480, end=540).n_trials == 1
    assert _bt(cfg, prices, start=470, end=540).n_trials == 2
    assert _bt(cfg, prices, start=470, end=540).n_trials == 2        # the identical trial again is not new
    third = _bt(cfg, prices, start=460, end=540)
    assert third.n_trials == 3 and third.reports["strategy"].deflated_sharpe is not None


def test_turnover_is_reported_one_way(quant_result):
    """Turnover = half the traded notional / NAV, the same definition max_turnover caps."""
    r = quant_result
    for d in r.decisions:
        if d.get("fills"):
            traded = sum(f["quantity"] * f["price"] for f in d["fills"])
            assert r.turnover[pd.Timestamp(d["fill_date"])] == pytest.approx(0.5 * traded / d["nav"])


def test_quant_only_firm_with_too_few_names_says_why_it_does_not_trade(prices, tmp_path):
    """Regression: with fewer than three names the quant abstains on everything; the result
    must say so instead of reporting an unexplained all-cash book."""
    cfg = _cfg(tmp_path)
    cfg.universe.tickers = ["EEE", "FFF"]
    r = _bt(cfg, prices, start=480, end=540)
    assert all(not d["orders"] for d in r.decisions)
    assert any(w.startswith("quant abstained on all 2 names") and "at least 3" in w for w in r.warnings)


def test_library_backtest_and_kill_switch_follow_the_config_file_not_the_cwd(tmp_path, prices, monkeypatch):
    """Open request (cli -> pipeline/engine): run_backtest and decide used ``cfg.runs_dir`` as a
    path, so for a library caller a relative runs_dir (relative to the config file, per the
    config contract) resolved against the shell's working directory instead."""
    from aifirm.config import FirmConfig
    from aifirm.journal import ForecastStore, Ledger, LessonStore
    from aifirm.pipeline import decide, gather_data, quant_only_config
    from aifirm.portfolio import RiskState

    proj, elsewhere = tmp_path / "proj", tmp_path / "elsewhere"
    proj.mkdir()
    elsewhere.mkdir()
    (proj / "firm.toml").write_text(f'runs_dir = "runs"\n[universe]\ntickers = {json.dumps(NAMES)}\n')
    monkeypatch.chdir(elsewhere)
    cfg = FirmConfig.load(proj / "firm.toml")
    r = _bt(cfg, prices, start=480, end=520)
    assert Path(r.run_dir).resolve().is_relative_to((proj / "runs").resolve())
    assert (proj / "runs" / "trials.jsonl").exists() and not (elsewhere / "runs").exists()

    (proj / "runs" / "KILL").touch()
    q = quant_only_config(cfg)
    as_of = _day(prices, 500)
    _, sheets = gather_data(q, as_of, live=False, prices=prices, **LOADERS)
    with ForecastStore(tmp_path / "f.sqlite") as store:
        dec = decide(q, None, as_of, prices, sheets, current_weights={}, nav=1e5, prev_nav=None,
                     risk_state=RiskState(peak_nav=1e5), forecast_store=store,
                     lesson_store=LessonStore(tmp_path / "l.jsonl"), ledger=Ledger(tmp_path / "ledger.jsonl"),
                     run_id="r", trading_calendar=prices.close.index)
    assert dec.orders == [] and any("KILL SWITCH" in m for m in dec.risk_messages)
