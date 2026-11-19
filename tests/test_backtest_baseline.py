"""The backtest's quant-only baseline is the strategy minus its language models: the same
screen, limits and costs (RESEARCH 6.6, 9.3 item 6), and the report states what the language
models added over it net of their cost. Fully offline (synthetic market, offline rules)."""

from dataclasses import replace

import pytest
from synthetic_market import LOADERS, NAMES, make_config, make_prices

from aifirm.backtest.engine import _screen_mismatch, run_backtest
from aifirm.config import ScreenConfig
from aifirm.journal import Ledger
from aifirm.report import render_backtest_report


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


@pytest.fixture(scope="module")
def screened(prices, tmp_path_factory):
    cfg = make_config(str(tmp_path_factory.mktemp("screened")), memo_min_edge=0.005)
    cfg.portfolio.min_prob_to_hold = 0.5
    cfg.universe.tickers = NAMES[:1]
    cfg.universe.screen = ScreenConfig(enabled=True, tickers=list(NAMES[1:]), deep_dive_top_n=2,
                                       min_dollar_volume=1e6)
    cfg.evaluation.n_random_books = 20
    return run_backtest(cfg, _day(prices, 380), _day(prices, 470), "offline", prices=prices, **LOADERS)


def _screens(path):
    return [(e["payload"]["as_of"], e["payload"]["n_universe"], e["payload"]["n_eligible"], e["payload"]["top_band"])
            for e in Ledger(path).entries() if e["event"] == "screen"]


def test_the_quant_only_baseline_screens_the_same_universe(screened):
    run_dir = screened.run_dir
    strategy = _screens(f"{run_dir}/ledger.jsonl")
    baseline = _screens(f"{run_dir}/quant_only/ledger.jsonl")
    assert strategy and strategy == baseline           # same dates, names, eligibility and top band
    assert all(n == len(NAMES) for _, n, _, _ in strategy)
    assert not any("did not screen the same universe" in w for w in screened.warnings)


def test_the_screen_parity_guard_names_a_mismatch():
    a = [{"as_of": "2024-01-02", "screen": {"n_universe": 6, "n_eligible": 6}}]
    assert _screen_mismatch(a, [dict(a[0])]) == ""
    assert "did not screen the same universe" in _screen_mismatch(a, [{"as_of": "2024-01-02",
                                                                       "screen": {"n_universe": 5, "n_eligible": 5}}])
    assert "did not screen the same universe" in _screen_mismatch(a, [])


def test_value_added_over_the_baseline_is_net_of_llm_cost(screened):
    ov = screened.overlay_value()
    assert ov is not None and ov.offline
    assert ov.initial_nav == pytest.approx(100_000.0)
    assert ov.strategy_nav == pytest.approx(float(screened.nav.iloc[-1]))
    assert ov.quant_nav == pytest.approx(float(screened.baseline_nav["quant_only"].iloc[-1]))
    q_ret = screened.baseline_returns["quant_only"]
    assert ov.quant_nav == pytest.approx(100_000.0 * float((1 + q_ret).prod()))   # the NAV behind the returns
    assert ov.llm_cost_usd == 0.0 and ov.value_added_usd == pytest.approx(ov.difference_usd)
    paid = replace(screened, usage={**screened.usage, "cost_usd": 12.5}).overlay_value()
    assert paid.value_added_usd == pytest.approx(ov.difference_usd - 12.5)
    assert paid.value_added_pct == pytest.approx((ov.difference_usd - 12.5) / 1_000.0)
    text = render_backtest_report(screened)
    assert "## Value added by the language models (vs the quant-only baseline, net of LLM cost)" in text
    assert "offline rules standing in for the language models" in text


def test_quant_mode_has_no_value_added_section(prices, tmp_path):
    cfg = make_config(str(tmp_path))
    cfg.evaluation.n_random_books = 0
    r = run_backtest(cfg, _day(prices, 480), _day(prices, 520), "quant", prices=prices, **LOADERS)
    assert r.overlay_value() is None and "quant_only" not in r.baseline_nav
    assert "Value added by the language models" not in render_backtest_report(r)
