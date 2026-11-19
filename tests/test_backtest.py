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
    cfg.portfolio.rebalance_fraction = 1.0       # buy the whole core at the first open (no partial steps)
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


# ----------------------------------------------------------------------------- evaluation: is there skill, and would we know?


def test_decision_records_carry_every_signal_for_the_evaluation(offline_result):
    r = offline_result
    rec = r.decisions[0]
    assert {"pooled_prob", "prob_equal_weight", "n_views", "agent_probs", "sectors"} <= set(rec)
    agents = {a for by_agent in rec["agent_probs"].values() for a in by_agent}
    assert {"fundamental", "technical", "macro", "events", "quant"} & agents
    assert all(0.0 < p < 1.0 for by_agent in rec["agent_probs"].values() for p in by_agent.values())
    assert rec["sectors"]["AAA"] == "Tech"


def test_every_backtest_is_evaluated(quant_result, offline_result):
    from aifirm.backtest.ic import POOL, QUANT_ONLY
    ev = quant_result.evaluation                       # no model, no contamination: every decision counts
    assert ev is not None and ev.horizon_days == 21 and ev.step_days == 21 and ev.clean_after is None
    assert ev.ic[POOL].n_dates > 0 and math.isfinite(ev.ic[POOL].mean) and ev.ic[POOL].method == "iid"
    assert ev.power is not None and ev.power.n_dates == ev.ic[POOL].n_dates
    assert ev.random_books is not None and ev.random_books.n_books == 1000
    assert 0.0 <= ev.random_books.percentile_return <= 100.0 and 0.0 < ev.random_books.p_value_return <= 1.0
    assert ev.costs is not None and ev.costs.replay_matches_engine
    assert ev.factors is not None and ev.factors.factors == ["market"]     # synthetic data holds SPY only
    assert any("size factor skipped" in n for n in ev.notes) and not ev.contaminated_ic
    assert QUANT_ONLY not in ev.ic and ev.costs.quant_total_return_pct is None
    # offline on synthetic 2024 data: every decision precedes the configured models' cutoff + buffer, so
    # no skill evidence is clean; the whole-window IC is kept apart, labelled contaminated
    off = offline_result.evaluation
    assert off.clean_after == offline_result.contamination_end and off.n_clean_decisions == 0
    assert not off.ic and not off.ic_differences and off.power is None and off.random_books is None
    assert off.factors is None and any("no decision was made after" in n for n in off.notes)
    assert QUANT_ONLY in off.contaminated_ic and any((d.a, d.b) == (POOL, QUANT_ONLY)
                                                     for d in off.contaminated_ic_differences)
    assert off.costs.quant_total_return_pct is not None and off.costs.vs_quant is not None
    assert {"agent:technical", "agent:quant"} <= set(off.contaminated_ic)
    # the quant analyst inside the committee and the quant-only baseline forecast identically
    assert off.contaminated_ic["agent:quant"].mean == pytest.approx(off.contaminated_ic[QUANT_ONLY].mean)


def test_skill_evidence_uses_only_decisions_after_the_contaminated_period(prices, tmp_path, monkeypatch):
    """Regression (P10): IC, paired IC, power, random books and factor alpha were computed over
    every decision, so a memorising model's pre-cutoff 'skill' reached the verdicts. With the
    cutoff mid-window they use only the clean decisions; the whole window is kept apart."""
    from aifirm import config as config_mod
    from aifirm.backtest.engine import run_models
    from aifirm.backtest.ic import POOL
    cfg = _cfg(tmp_path)
    mid = _day(prices, 440)
    for m in run_models(cfg, "offline"):
        monkeypatch.setitem(config_mod.MODEL_TRAINING_CUTOFFS, m, mid)
    cfg.evaluation.cutoff_buffer_days = 0
    r = _bt(cfg, prices, mode="offline", start=380, end=540)
    ev = r.evaluation
    clean = [d for d in r.decisions if d["as_of"] > mid]
    assert r.contamination_end == mid and ev.clean_after == mid
    assert 0 < ev.n_clean_decisions == len(clean) < ev.n_decisions == len(r.decisions)
    assert ev.ic[POOL].n_dates <= len(clean) < ev.contaminated_ic[POOL].n_dates
    assert ev.random_books is not None and ev.random_books.n_rebalances <= len(clean)
    assert ev.factors is not None and ev.factors.n_obs < len(r.returns)
    md = render_backtest_report(r)
    assert "**Clean decisions only.**" in md and "contaminated decisions included (for reference only" in md


def test_replaying_the_recorded_decisions_reproduces_the_run_and_doubling_costs_lowers_returns(quant_result,
                                                                                                prices):
    from aifirm.backtest.engine import replay_total_return
    from aifirm.pipeline import quant_only_config
    r = quant_result
    c = r.evaluation.costs
    assert c.multipliers == [0.0, 1.0, 2.0]
    assert c.strategy_total_return_pct[1.0] == pytest.approx(r.reports["strategy"].total_return_pct, abs=1e-9)
    assert c.strategy_total_return_pct[0.0] > c.strategy_total_return_pct[1.0] > c.strategy_total_return_pct[2.0]
    assert c.excess_vs_benchmark(2.0) < c.excess_vs_benchmark(1.0)
    assert c.realised_cost_bps > 5.0                                  # at least the 5 bps spread plus commission
    cfg = quant_only_config(_cfg(Path(r.run_dir).parents[1]))
    days = prices.close.index[START:END + 1]
    rebal = set(days[::21])
    assert replay_total_return(prices, cfg, days, rebal, r.decisions, 1.0) == pytest.approx(
        r.reports["strategy"].total_return_pct, abs=1e-9)


def test_engine_breakeven_is_where_the_replayed_excess_crosses_zero(quant_result, prices):
    """Consistency of the bisection on the real re-simulation: against a benchmark whose
    return equals the strategy's at 3x costs, the breakeven must come out at ~3x."""
    from aifirm.backtest.engine import replay_total_return
    from aifirm.backtest.evaluation import cost_sensitivity
    from aifirm.pipeline import quant_only_config
    r = quant_result
    cfg = quant_only_config(_cfg(Path(r.run_dir).parents[1]))
    days = prices.close.index[START:END + 1]
    rebal = set(days[::21])
    markets: dict = {}

    def replay(k):
        return replay_total_return(prices, cfg, days, rebal, r.decisions, k, markets)

    target = replay(3.0)
    cs = cost_sensitivity(replay, target, [0.0, 1.0, 2.0], 50.0, r.evaluation.costs.realised_cost_bps,
                          r.reports["strategy"].total_return_pct)
    assert cs.vs_benchmark.status == "found" and cs.vs_benchmark.multiplier == pytest.approx(3.0, abs=0.03)
    assert cs.vs_benchmark.bps_one_way == pytest.approx(cs.vs_benchmark.multiplier * cs.realised_cost_bps)


def test_contamination_buffer_extends_the_contaminated_period(prices, tmp_path, monkeypatch):
    """Regression guard: vendor cutoffs are unreliable (research 7.2), so decisions within
    cutoff_buffer_days after a model's training cutoff still count as contaminated."""
    from aifirm import config
    cutoff_i = 430
    monkeypatch.setitem(config.MODEL_TRAINING_CUTOFFS, "claude-opus-5-5", _day(prices, cutoff_i))
    cfg = _cfg(tmp_path / "a")
    assert contamination_cutoff(cfg, "llm") == str((prices.close.index[cutoff_i] + pd.Timedelta(days=31)).date())
    assert contamination_cutoff(cfg, "llm", buffer_days=0) == _day(prices, cutoff_i)
    assert contamination_cutoff(cfg, "quant") is None
    with pytest.raises(ValueError):
        contamination_cutoff(cfg, "llm", buffer_days=-1)
    cfg.evaluation.cutoff_buffer_days = 0
    unbuffered = _bt(cfg, prices, mode="offline", start=380, end=500)
    buffered_cfg = _cfg(tmp_path / "b")
    buffered_cfg.evaluation.cutoff_buffer_days = 45
    buffered = _bt(buffered_cfg, prices, mode="offline", start=380, end=500)
    assert unbuffered.model_cutoff == buffered.model_cutoff == _day(prices, cutoff_i)
    assert buffered.contamination_end > unbuffered.contamination_end == _day(prices, cutoff_i)
    in_buffer = [d["as_of"] for d in buffered.decisions
                 if unbuffered.contamination_end < d["as_of"] <= buffered.contamination_end]
    assert in_buffer, "a decision must fall inside the buffer for this test to mean anything"
    assert buffered.pre_cutoff_report.days > unbuffered.pre_cutoff_report.days
    md = render_backtest_report(buffered)
    assert "plus a 45-day buffer" in md and buffered.contamination_end in md
    assert "the test straddles it" in md


def test_a_failing_evaluation_part_is_reported_missing_not_fatal(prices, tmp_path, monkeypatch):
    """Fail closed: a part that cannot be computed is None with a note, never a number, and
    never sinks the backtest (whose decisions may have cost real LLM money)."""
    from aifirm.backtest import engine

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "random_books", broken)
    r = _bt(_cfg(tmp_path), prices, start=480, end=540)
    assert r.evaluation.random_books is None
    assert any(n.startswith("random books could not be computed (RuntimeError: boom)") for n in r.evaluation.notes)
    md = render_backtest_report(r)
    assert "composition-matched random books not computed" in md and "### Composition-matched random books" in md


def test_factor_proxies_are_never_downloaded_when_prices_are_passed_in(prices, tmp_path, monkeypatch):
    from aifirm.backtest import engine

    def no_network(*a, **k):
        raise AssertionError("load_prices must not be called when prices are passed in")

    monkeypatch.setattr(engine, "load_prices", no_network)
    p = make_prices()
    r = _bt(_cfg(tmp_path / "a"), p, start=480, end=540)
    assert r.evaluation.factors is not None                          # SPY (the benchmark) is a proxy itself
    assert any("size factor skipped" in n for n in r.evaluation.notes)
    # injected proxies are used as given
    idx = p.close.index
    rng = np.random.default_rng(0)
    proxies = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
                            for t in ("SPY", "IWM", "IWD", "IWF", "MTUM", "QUAL", "USMV", "BIL")}, index=idx)
    r2 = _bt(_cfg(tmp_path / "b"), p, start=480, end=540, factor_prices=from_frames(proxies))
    assert r2.evaluation.factors.factors == ["market", "size", "value", "momentum", "quality", "low_volatility"]
    assert r2.evaluation.factors.cash_proxy == "BIL"


def test_report_states_the_six_lines_and_answers_whether_there_is_skill(offline_result):
    md = render_backtest_report(offline_result)
    six = md.split("## The six lines", 1)[1].split("## Performance vs baselines", 1)[0]
    for i, needle in enumerate(("**Window and assets:**", "**Model and cutoff:**", "**Cost model and breakeven:**",
                                "**Configurations tried:**", "**Deflated Sharpe at", "**Benchmark and null:**"),
                               start=1):
        assert f"{i}. {needle}" in six
    assert "6 stocks plus the benchmark SPY" in six and "claude-opus-5-5" in six and "31-day buffer" in six
    assert "the whole test lies before it" in six and "composition-matched random books" in six
    for heading in ("## Is there skill, and would we know?", "### Signal: rank information coefficient (IC)",
                    "### Power: how long until we would know", "### Composition-matched random books",
                    "### Cost sensitivity and breakeven", "### Factor attribution (ETF proxies)"):
        assert heading in md
    assert "Quant-only baseline" in md and "minus Quant-only baseline" in md and "Excess vs quant-only" in md


def test_report_says_plainly_when_skill_is_not_shown(quant_result):
    """Plain-language verdicts: below the 95th percentile of random books the selection is
    'not distinguishable from random books', and too few dates for 80% power is 'anecdotal'."""
    import copy
    r = copy.deepcopy(quant_result)
    ev = r.evaluation
    ev.random_books.percentile_return, ev.random_books.p_value_return = 62.0, 0.38
    ev.power.dates_for_80pct, ev.power.dates_for_t2 = 4000.0, 2000.0
    md = render_backtest_report(r)
    assert "not distinguishable from random books" in md
    assert "Anecdotal: fewer than 4,000 dates" in md
    assert "at the 62nd percentile of 1,000 composition-matched random books" in md
    ev.random_books.percentile_return, ev.random_books.p_value_return = 99.0, 0.004
    assert "its stock selection beat random picks" in render_backtest_report(r)


def test_a_downloading_backtest_fetches_the_factor_proxies_even_though_spy_is_the_benchmark(prices, tmp_path,
                                                                                            monkeypatch):
    """Regression: the benchmark SPY is itself a factor proxy, so "prices hold a proxy" must not
    stop a run that downloads its own prices from fetching IWM, IWD, MTUM and the rest."""
    from aifirm.backtest import engine
    from aifirm.backtest.factors import PROXY_TICKERS
    idx = prices.close.index
    rng = np.random.default_rng(0)
    proxies = from_frames(pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
                                        for t in PROXY_TICKERS}, index=idx))
    requested: list[list[str]] = []

    def fake_load(tickers, start, end, cache=None):
        requested.append(sorted(tickers))
        return proxies if "IWM" in tickers else prices

    monkeypatch.setattr(engine, "load_prices", fake_load)
    r = run_backtest(_cfg(tmp_path), _day(prices, 480), _day(prices, 540), "quant", **LOADERS)
    assert requested[0] == sorted([*NAMES, "SPY"]) and requested[1] == sorted(PROXY_TICKERS)
    assert len(r.evaluation.factors.factors) == 6 and r.evaluation.factors.cash_proxy == "BIL"


def test_interpretation_reads_the_evaluation_without_overclaiming(quant_result):
    """Regressions from the real-data smoke: a significantly negative factor alpha was described
    as "explained by exposure", "does not survive 2x costs" was said of a strategy with no edge
    even at zero cost, and a non-positive IC produced a garbled anecdotal sentence."""
    import copy
    from aifirm.backtest.evaluation import Breakeven
    r = copy.deepcopy(quant_result)
    ev = r.evaluation
    ev.factors.alpha_annual_pct, ev.factors.alpha_t = -4.0, -2.5
    ev.costs.vs_benchmark = Breakeven("no_edge", 0.0, 50.0, 1)
    ev.power.ic_mean, ev.power.dates_for_80pct, ev.power.dates_for_t2 = -0.05, math.inf, math.inf
    md = render_backtest_report(r)
    assert "Factor-adjusted alpha is negative (-4.0% a year, t = -2.50)" in md
    assert "Even with zero trading costs the strategy did not beat the benchmark" in md
    assert "does not survive" not in md
    assert "is not positive, so no number of dates at this IC would show skill" in md
    ev.costs.vs_benchmark = Breakeven("found", 1.5, 50.0, 8, 8.0)
    ev.costs.strategy_total_return_pct[2.0] = ev.costs.benchmark_total_return_pct - 1.0
    assert "The result does not survive 2x costs" in render_backtest_report(r)


def test_a_prompt_edit_is_a_new_trial_for_llm_stages_only(tmp_path):
    """Regression (doc mismatch): prompt edits are code, so they never counted as trials; the
    trial key of an offline or llm run now carries a digest of the prompt modules."""
    import json as _json
    from aifirm.backtest.engine import _count_trials, prompt_digest
    path = tmp_path / "trials.jsonl"
    rows = [dict(fingerprint="f", mode="llm", start="2024-01-01", end="2024-06-30", prompts="p1"),
            dict(fingerprint="f", mode="quant", start="2024-01-01", end="2024-06-30")]
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    assert _count_trials(path, "f", "llm", "2024-01-01", "2024-06-30", "p1") == 2       # same prompts: no new trial
    assert _count_trials(path, "f", "llm", "2024-01-01", "2024-06-30", "p2") == 3       # edited prompts: a new one
    assert _count_trials(path, "f", "quant", "2024-01-01", "2024-06-30", "p2") == 2     # quant sends no prompt
    d = prompt_digest()
    assert len(d) == 16 and d == prompt_digest()
