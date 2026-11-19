"""Integration tests for the wiring added on top of the decision cycle: the broad screen and
its deep-dive set, the quarantined news reader, the tier gate, point-in-time calibration and
the bounded overlay, in ``decide`` and in the backtest. Fully offline (synthetic data,
offline LLM responders; no network, no API key)."""

import re

import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, make_config, make_prices, perturb_after
from test_news_quarantine import RAW_MARKERS, RecordingLLM, fragments, hostile_headlines

from aifirm.agents import make_offline_llm
from aifirm.backtest.engine import run_backtest
from aifirm.config import ScreenConfig
from aifirm.journal import ForecastStore, Ledger, LessonStore
from aifirm.llm.client import LLMCallFailed
from aifirm.pipeline import decide, gather_data, market_snapshot, quant_only_config
from aifirm.portfolio import RiskState
from aifirm.report import render_backtest_report, render_decision_report
from aifirm.schemas import ForecastRecord

DAY = 400
NAV = 100_000.0
MODEL = "claude-opus-5-5"


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


def _cfg(tmp_path, **process):
    cfg = make_config(str(tmp_path / "runs"), memo_min_edge=0.005, **process)
    cfg.portfolio.min_prob_to_hold = 0.5
    return cfg


class Book:
    def __init__(self, d):
        d.mkdir(parents=True, exist_ok=True)
        self.store = ForecastStore(d / "forecasts.sqlite")
        self.lessons = LessonStore(d / "lessons.jsonl")
        self.ledger = Ledger(d / "ledger.jsonl")

    def events(self, name):
        return [e["payload"] for e in self.ledger.entries() if e["event"] == name]


@pytest.fixture
def books(tmp_path):
    made = []

    def make(name):
        made.append(Book(tmp_path / name))
        return made[-1]

    yield make
    for b in made:
        b.store.close()


def _decide(cfg, llm, prices, book, *, as_of=None, current=None, live=False, news=None, sheets=None):
    as_of = as_of or _day(prices, DAY)
    if sheets is None:
        _, sheets = gather_data(cfg, as_of, live=live, prices=prices, headlines_loader=news, **LOADERS)
    current = current or {}
    px = market_snapshot(prices, as_of, current).prices
    return decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=NAV, prev_nav=None,
                  risk_state=RiskState(peak_nav=NAV), forecast_store=book.store, lesson_store=book.lessons,
                  ledger=book.ledger, run_id="r1", positions={t: w * NAV / px[t] for t, w in current.items()},
                  trading_calendar=prices.close.index)


def _recording():
    return RecordingLLM(dict(make_offline_llm().responders))


# ----------------------------------------------------------------------------- the screen in decide


def _screen_cfg(tmp_path, top_n=2, **process):
    cfg = _cfg(tmp_path, **process)
    cfg.universe.tickers = []
    cfg.universe.screen = ScreenConfig(enabled=True, tickers=list(NAMES), deep_dive_top_n=top_n,
                                       min_dollar_volume=1e6)
    return cfg


def test_only_the_top_band_and_held_names_reach_the_llm_committee(tmp_path, prices, books):
    cfg = _screen_cfg(tmp_path)
    llm = _recording()
    book = books("screen")
    dec = _decide(cfg, llm, prices, book, current={"AAA": 0.05})             # AAA ranks last but is held
    sc = dec.screen
    assert sc is not None and sc.n_universe == len(NAMES) and len(sc.top_band) == 2
    assert dec.screen.deep_dive == [*sc.top_band, "AAA"] and list(dec.committee) == sc.deep_dive
    analysed = {re.search(r'Use ticker "([^"]+)"', u).group(1) for r, _, u in llm.calls if r.startswith("analyst.")}
    assert analysed == set(sc.deep_dive)                                    # no LLM call outside the deep dive
    outside = set(NAMES) - set(sc.deep_dive)
    assert not outside & set(dec.target_final.weights())                    # no position outside the deep dive
    quant_only = {f.ticker for f in dec.forecasts if f.agent == "quant"}
    assert quant_only == set(NAMES)                                         # breadth: the quant forecasts every name
    assert not {f.ticker for f in dec.forecasts if f.agent == "committee"} & outside
    logged = book.events("screen")
    assert len(logged) == 1 and len(logged[0]["ranking"]) == len(NAMES) and logged[0]["deep_dive"] == sc.deep_dive
    assert dec.final_probabilities().keys() == set(NAMES)
    md = render_decision_report(dec)
    assert "## Screen" in md and "The deterministic quant ranked 6 names" in md


def test_the_screen_and_the_quant_run_without_any_language_model(tmp_path, prices, books):
    cfg = quant_only_config(_screen_cfg(tmp_path, top_n=3))
    dec = _decide(cfg, None, prices, books("q"))
    assert dec.mode == "quant" and len(dec.committee) == 3 and dec.usage["calls"] == 0
    assert all(a.combination == "overlay" and a.prob_outperform == a.prob_quant for a in dec.aggregates.values())


# ----------------------------------------------------------------------------- the news reader in decide


def test_live_headlines_are_read_into_events_and_never_reach_privileged_prompts(tmp_path, prices, books):
    cfg = _cfg(tmp_path, news_enabled=True)
    cfg.universe.tickers = ["AAA", "BBB", "CCC"]
    llm = _recording()
    book = books("news")
    as_of = _day(prices, DAY)
    _, sheets = gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=lambda t, d: hostile_headlines(d),
                            **LOADERS)
    dec = _decide(cfg, llm, prices, book, sheets=sheets)
    readings = book.events("news_reading")
    assert {r["ticker"] for r in readings} == {"AAA", "BBB", "CCC"} and all(not r["failed"] for r in readings)
    assert all(dec.news[t]["read"] and dec.news[t]["n_events"] > 0 for t in ("AAA", "BBB", "CCC"))
    privileged = [u for role, _, u in llm.calls if role != "news_reader"]
    assert privileged
    for raw in RAW_MARKERS:
        for frag in fragments(raw):
            assert all(frag not in u for u in privileged), frag
    logged = book.events("factsheet")
    assert all(p["headlines"] == [] and p["quarantined_headlines"] and p["news_quarantined"] for p in logged)
    assert all(sheets[t].headlines for t in sheets)                          # the caller's sheets are untouched


def test_a_failing_news_reader_is_a_data_gap_not_an_outage(tmp_path, prices, books):
    cfg = _cfg(tmp_path, news_enabled=True)
    cfg.universe.tickers = ["AAA", "BBB", "CCC"]
    llm = make_offline_llm()

    def broken(role, system, user, schema):
        raise LLMCallFailed("news_reader: overloaded")

    llm.register("news_reader", broken)
    as_of = _day(prices, DAY)
    _, sheets = gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=lambda t, d: hostile_headlines(d),
                            **LOADERS)
    dec = _decide(cfg, llm, prices, books("fail"), sheets=sheets, current={"AAA": 0.05})
    assert not dec.held_by_outage and not any("OUTAGE" in w for w in dec.warnings)
    assert all(any(g.startswith("News could not be read") for g in dec.data_gaps[t]) for t in ("AAA", "BBB", "CCC"))
    assert all(dec.news[t]["failed"] for t in ("AAA", "BBB", "CCC"))


def test_without_a_model_headlines_are_withheld_not_shown(tmp_path, prices, books):
    cfg = quant_only_config(_cfg(tmp_path, news_enabled=True))
    as_of = _day(prices, DAY)
    _, sheets = gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=lambda t, d: hostile_headlines(d),
                            **LOADERS)
    book = books("quant-news")
    dec = _decide(cfg, None, prices, book, sheets=sheets)
    assert all(not s["read"] and "no language model" in s["reason"] for s in dec.news.values())
    assert book.events("news_reading") == []
    assert all(p["headlines"] == [] for p in book.events("factsheet"))


def test_headlines_outside_the_deep_dive_are_withheld_and_not_read(tmp_path, prices, books):
    cfg = _screen_cfg(tmp_path, top_n=2, news_enabled=True)
    llm = _recording()
    as_of = _day(prices, DAY)
    _, sheets = gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=lambda t, d: hostile_headlines(d),
                            **LOADERS)
    dec = _decide(cfg, llm, prices, books("dd-news"), sheets=sheets)
    read = {t for t, s in dec.news.items() if s["read"]}
    assert read == set(dec.screen.deep_dive)
    assert all("outside this cycle's deep-dive set" in s["reason"] for t, s in dec.news.items() if t not in read)
    reader_tickers = {re.search(r"COMPANY TICKER: (\S+)", u).group(1) for r, _, u in llm.calls if r == "news_reader"}
    assert reader_tickers == read


# ----------------------------------------------------------------------------- calibration and the overlay in decide


def _overconfident_track(agent, as_of, n=40, resolved_before=True):
    """n resolved questions on distinct dates: the agent said 0.8 and was right half the time."""
    days = pd.bdate_range(end=pd.Timestamp(as_of) - pd.Timedelta(days=60), periods=n)
    out = []
    for i, d in enumerate(days):
        resolve = d + pd.Timedelta(days=30) if resolved_before else pd.Timestamp(as_of) + pd.Timedelta(days=5)
        out.append(ForecastRecord(
            id=f"old|{d.date()}|T{i}|{agent}|s0", run_id="old", agent=agent, model=MODEL, ticker=f"T{i}",
            benchmark="SPY", as_of=str(d.date()), horizon_days=21, resolve_on=str(resolve.date()),
            prob_outperform=0.8, expected_excess_return_pct=2.0, pre_cutoff=True, resolved=True, outcome=i % 2,
            realized_excess_return_pct=1.0 if i % 2 else -1.0, brier=(0.8 - i % 2) ** 2))
    return out


def test_calibration_is_fitted_point_in_time_and_forecasts_keep_the_raw_probability(tmp_path, prices, books):
    cfg = _cfg(tmp_path, combination="pool")
    as_of = _day(prices, DAY)
    book = books("cal")
    book.store.add(_overconfident_track("technical", as_of))
    dec = _decide(cfg, make_offline_llm(), prices, book)
    cal = dec.calibrators["technical"]
    assert cal.fitted and cal.b < 1.0 and cal.n == 40 and dec.calibrators["fundamental"].is_identity
    logged = book.events("calibration")[0]["calibrators"]["technical"]
    assert logged["fitted"] and logged["b"] == pytest.approx(cal.b)
    for t, r in dec.committee.items():
        raw = next(v for v in r.views if v.agent == "technical")
        pooled = next(v for v in r.pool_views if v.agent == "technical")
        if raw.view.abstain:
            continue
        assert pooled.view.prob_outperform == pytest.approx(cal.apply(raw.view.prob_outperform))
        stored = next(f for f in dec.forecasts if f.ticker == t and f.agent == "technical")
        assert stored.prob_outperform == raw.view.prob_outperform                   # RAW, not calibrated
    inputs = [p for p in book.events("pool_inputs") if p["stage"] == "sealed"]
    assert inputs and all("raw_prob" in row and "pool_prob" in row for p in inputs for row in p["views"])


def test_calibration_never_uses_outcomes_resolved_after_the_decision(tmp_path, prices, books):
    """Regression guard (P3): the same overconfident track, resolving after the decision
    close, must leave every map at the identity."""
    cfg = _cfg(tmp_path, combination="pool")
    as_of = _day(prices, DAY)
    book = books("cal-future")
    book.store.add(_overconfident_track("technical", as_of, resolved_before=False))
    dec = _decide(cfg, make_offline_llm(), prices, book)
    assert all(c.is_identity for c in dec.calibrators.values())


def test_overlay_keeps_every_final_probability_within_the_bound_and_scores_both(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    book = books("overlay")
    dec = _decide(cfg, make_offline_llm(), prices, book)
    bound = book.events("overlay_bound")[0]
    assert bound["max_tilt"] == cfg.process.overlay_max_tilt and bound["n_dates"] == 0
    for a in dec.aggregates.values():
        assert a.combination == "overlay" and a.prob_quant is not None
        assert abs(a.prob_outperform - a.prob_quant) <= cfg.process.overlay_max_tilt + 1e-12
    agents = {(f.ticker, f.agent) for f in dec.forecasts}
    for t, a in dec.aggregates.items():
        if a.prob_llm is not None:
            assert (t, "llm_pool") in agents and (t, "committee") in agents
            llm = next(f for f in dec.forecasts if f.ticker == t and f.agent == "llm_pool")
            fin = next(f for f in dec.forecasts if f.ticker == t and f.agent == "committee")
            assert llm.prob_outperform == pytest.approx(a.prob_llm) and fin.prob_outperform == a.prob_outperform
    md = render_decision_report(dec)
    assert "Overlay mode" in md and "| Final P | Quant P | LLM P | Tilt |" in md and "exposure:" in md


def test_overlay_without_the_quant_warns_and_bounds_around_one_half(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    cfg.analysts.enabled = ["fundamental", "technical", "macro", "events"]
    dec = _decide(cfg, make_offline_llm(), prices, books("noquant"))
    assert any("quant analyst is not enabled" in w for w in dec.warnings)
    assert all(abs(a.prob_outperform - 0.5) <= cfg.process.overlay_max_tilt + 1e-12 for a in dec.aggregates.values())


# ----------------------------------------------------------------------------- the backtest


START, END = 300, 559


def test_backtest_with_a_screen_is_point_in_time_and_records_the_deep_dive(tmp_path, prices):
    cut = 450

    def run(p, d):
        cfg = quant_only_config(_screen_cfg(tmp_path / d, top_n=2))
        cfg.evaluation.n_random_books = 50
        return run_backtest(cfg, _day(p, START), _day(p, END), "quant", prices=p, **LOADERS)

    base = run(prices, "a")
    changed = run(perturb_after(prices, _day(prices, cut)), "b")

    def upto(r):
        return [{k: d[k] for k in ("as_of", "deep_dive", "target", "pooled_prob", "orders")} for d in r.decisions
                if d["as_of"] <= _day(prices, cut)]

    assert upto(base) == upto(changed) and upto(base)
    for d in base.decisions:
        held = set(d["target"])
        assert len(set(d["deep_dive"]) - held) <= 2 and set(d["pooled_prob"]) == set(NAMES)
        assert d["screen"] == {"n_universe": len(NAMES), "n_eligible": d["screen"]["n_eligible"]}
    ex = base.exposure_summary()
    assert len(base.exposure) == len(base.nav) and ((base.exposure >= -1e-9) & (base.exposure <= 1 + 1e-9)).all()
    assert 0 < ex["invested"] < 1 and ex["cash"] == pytest.approx(1 - ex["invested"])
    assert base.universe_note.startswith("screened universe from a file or inline list")
    md = render_backtest_report(base)
    assert "Exposure: on average" in md and "SURVIVORSHIP BIAS" in md


def test_a_preset_universe_is_flagged_as_survivorship_biased(tmp_path, prices):
    cfg = quant_only_config(_cfg(tmp_path))
    cfg.universe.screen = ScreenConfig(enabled=True, preset="sp100", deep_dive_top_n=3, min_dollar_volume=1e6)
    cfg.evaluation.n_random_books = 0
    r = run_backtest(cfg, _day(prices, 500), _day(prices, END), "quant", prices=prices, **LOADERS)
    assert r.universe_note.startswith("preset 'sp100'") and "survivorship" in r.universe_note
    assert r.warnings[0].startswith("SURVIVORSHIP")
    md = render_backtest_report(r)
    assert "SURVIVORSHIP BIAS" in md and "SURVIVORSHIP-BIASED universe list" in md


def test_the_factsheet_command_builds_one_sheet_even_with_a_screen(tmp_path, prices, monkeypatch, capsys):
    """Regression: with the screen enabled, `aifirm factsheet T` built fact sheets for the whole
    screened universe (a hundred SEC downloads) to print one."""
    from aifirm import cli
    from aifirm.pipeline import universe

    (tmp_path / "firm.toml").write_text('[universe.screen]\nenabled = true\npreset = "sp100"\n')
    seen = []

    def fake_gather(cfg, as_of, live, prices=None):
        seen.append(universe(cfg))
        return gather_data(cfg, as_of, False, prices=prices_all.as_of(as_of), **LOADERS)

    prices_all = prices
    monkeypatch.setattr(cli, "gather_data", fake_gather)
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: _day(prices, DAY))
    assert cli.main(["factsheet", "AAA", "--config", str(tmp_path / "firm.toml"), "--as-of", _day(prices, DAY)]) == 0
    assert seen == [["AAA"]] and "FACT SHEET: AAA" in capsys.readouterr().out


def test_the_live_headline_feed_honours_news_max_headlines(tmp_path, prices, monkeypatch):
    """Regression: gather_data called recent_headlines with its own default of 12 items, so
    process.news_max_headlines (20) never bound with the Yahoo feed."""
    from aifirm import pipeline
    seen = []

    def feed(t, d, max_items=12, max_age_days=14, close_time=None):
        seen.append(max_items)
        return []

    monkeypatch.setattr(pipeline, "recent_headlines", feed)
    cfg = _cfg(tmp_path)
    cfg.process.news_max_headlines = 17
    gather_data(cfg, _day(prices, DAY), live=True, prices=prices, **LOADERS)
    assert seen and set(seen) == {17}


def test_a_failed_cycle_logs_the_llm_spend_it_already_made(tmp_path, prices, books, monkeypatch):
    """Regression: cycle_error carried no usage, so the cost of a live attempt that called the
    committee and then failed was dropped from the value-added arithmetic."""
    from aifirm import pipeline

    def broken(*a, **k):
        raise RuntimeError("construct failed")

    monkeypatch.setattr(pipeline, "construct_portfolio", broken)
    book = books("failed")
    with pytest.raises(RuntimeError, match="construct failed"):
        _decide(_cfg(tmp_path), make_offline_llm(), prices, book)
    err = book.events("cycle_error")
    assert len(err) == 1 and err[0]["error"].startswith("RuntimeError: construct failed")
    assert err[0]["usage"]["calls"] > 0 and "cost_usd" in err[0]["usage"]
