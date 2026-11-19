"""Tests of the quant-only shadow book (aifirm/shadow.py, aifirm/books.py and its CLI wiring),
fully offline: synthetic market, offline LLM responders, no network, no API key.

What is pinned here (RESEARCH 6.6, 7.4 S6):

* every ``aifirm run`` also decides a quant-only paper book in ``<book>/shadow``, forked from
  the live book on its first run, with its own ledger, forecasts, risk state and queue;
* the shadow's orders fill at the same next open as the live book's, and it is credited the
  same dividends, through the same settlement code;
* the shadow can never change the live book: identical live orders, state, forecasts and
  ledger decisions with and without it, over cycles in which its forecasts resolve; a
  failing shadow (or comparison) is a warning, never a failed or altered live run; it only
  ever uses the paper broker and obeys the kill switch;
* its forecasts are mirrored once into the live store as agent ``shadow`` (the quant's own
  forecasts are not duplicated) and excluded from everything a live decision reads;
* ``compare_books``: NAV, returns, LLM spend and value added arithmetic, paired IC verdicts.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, make_config, make_prices

from aifirm import cli, shadow
from aifirm.books import BookError
from aifirm.data.prices import PriceData
from aifirm.execution.broker import PaperBroker
from aifirm.journal import (SHADOW_AGENT, ForecastStore, Ledger, believability_weights, fit_calibrators,
                            resolved_rank_ic)
from aifirm.pipeline import gather_data, quant_only_config
from aifirm.portfolio import RiskState
from aifirm.schemas import ForecastRecord

DAY = 400
TODAY = 540
ANTHROPIC_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                  "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE")


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


def _window(p, tickers, start, end):
    lo, hi, cols = pd.Timestamp(start), pd.Timestamp(end), sorted(tickers)
    cut = lambda df: None if df is None else df.loc[lo:hi, cols]   # noqa: E731
    return PriceData(close=cut(p.close), adj_close=cut(p.adj_close), volume=cut(p.volume), open=cut(p.open),
                     raw_close=cut(p.raw_close), splits=None)


def _with_dividend(prices, ex_day, y=0.01):
    adj = prices.adj_close.copy()
    adj.loc[adj.index >= pd.Timestamp(ex_day)] *= 1.0 + y
    return PriceData(close=prices.close, adj_close=adj, volume=prices.volume, open=prices.open,
                     raw_close=prices.raw_close)


CONFIG = '[universe]\ntickers = {names}\nbenchmark = "SPY"\n[process]\nmemo_min_edge = 0.005\n' \
         '[portfolio]\nmin_prob_to_hold = 0.5\n'


def _make_env(tmp_path, prices, monkeypatch, name="proj", extra=""):
    data = {"prices": prices}

    def fake_gather(cfg, as_of, live, prices=None):
        return gather_data(cfg, as_of, False, prices=prices, **LOADERS)

    def fake_load_prices(tickers, start, end):
        return _window(data["prices"], tickers, start, end)

    monkeypatch.setattr(cli, "gather_data", fake_gather)
    monkeypatch.setattr(cli, "load_prices", fake_load_prices)
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: _day(prices, TODAY))
    for var in ANTHROPIC_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    proj = tmp_path / name
    proj.mkdir()
    config = proj / "firm.toml"
    config.write_text('runs_dir = "runs"\n' + CONFIG.format(names=json.dumps(NAMES)) + extra)
    runs = (proj / "runs").resolve()
    book = runs / "offline"
    return SimpleNamespace(config=str(config), runs=runs, book=book, shadow=book / "shadow", data=data)


@pytest.fixture
def env(tmp_path, prices, monkeypatch):
    return _make_env(tmp_path, prices, monkeypatch)


def _run(env, i, *extra, prices=None):
    p = prices if prices is not None else env.data["prices"]
    return cli.main(["run", "--config", env.config, "--offline", "--as-of", _day(p, i), *extra])


def _events(path, event=None):
    if not path.exists():
        return []
    return [e for e in Ledger(path).entries() if event is None or e["event"] == event]


def _records(book):
    with ForecastStore(book / "forecasts.sqlite") as store:
        return store.all()


# ----------------------------------------------------------------------------- the shadow decides and trades


def test_first_run_forks_the_shadow_from_the_live_book_and_both_decide(env, prices, capsys):
    assert _run(env, DAY) == 0
    out = capsys.readouterr().out
    assert "shadow book (started today from the live book's state)" in out and "compare: " in out
    inception = _events(env.shadow / "ledger.jsonl", "shadow_inception")
    assert len(inception) == 1
    p = inception[0]["payload"]
    assert p["as_of"] == _day(prices, DAY) and p["nav"] == pytest.approx(100_000.0) and p["positions"] == {}
    cfg = cli._load_config(env.config)
    starts = _events(env.shadow / "ledger.jsonl", "cycle_start")
    assert len(starts) == 1 and starts[0]["payload"]["mode"] == "quant"
    assert starts[0]["payload"]["config_fingerprint"] == quant_only_config(cfg).fingerprint()
    assert json.loads((env.shadow / "pending_orders.json").read_text())["orders"]    # the shadow trades
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"]      # and so does the live book
    live_event = _events(env.book / "ledger.jsonl", "shadow")
    assert len(live_event) == 1 and live_event[0]["payload"]["status"] == "ok"
    assert live_event[0]["payload"]["inception"] is True
    report = next((env.book / "reports").glob("decision_*.md")).read_text()
    assert "## Quant-only shadow book" in report and "AI value added net of LLM cost" in report
    assert list((env.shadow / "reports").glob("decision_*.md"))
    assert cli.main(["verify-ledger", "--config", env.config, "--offline"]) == 0
    assert cli.main(["verify-ledger", "--config", env.config, "--offline", "--shadow"]) == 0


def test_shadow_orders_fill_at_the_same_next_open_as_the_live_book(env, prices):
    assert _run(env, DAY) == 0 and _run(env, DAY + 1) == 0
    d1 = _day(prices, DAY + 1)
    live = _events(env.book / "ledger.jsonl", "fills")
    shadow_fills = _events(env.shadow / "ledger.jsonl", "fills")
    assert len(live) == 1 and len(shadow_fills) == 1
    assert live[0]["payload"]["fill_date"] == shadow_fills[0]["payload"]["fill_date"] == d1
    for f in shadow_fills[0]["payload"]["fills"]:
        assert f["side"] == "buy" and f["date"] == d1
        assert prices.open.loc[d1, f["ticker"]] <= f["price"] <= prices.open.loc[d1, f["ticker"]] * 1.01
    state = json.loads((env.shadow / "paper_broker.json").read_text())
    assert set(state["positions"]) == {f["ticker"] for f in shadow_fills[0]["payload"]["fills"]}
    assert state["dividends_through"] == d1


def test_shadow_book_is_credited_dividends_like_the_live_book_once(env, prices):
    y = 0.01
    env.data["prices"] = _with_dividend(_with_dividend(prices, _day(prices, DAY + 1), y), _day(prices, DAY + 3), y)
    assert _run(env, DAY, prices=prices) == 0
    assert _run(env, DAY + 2, prices=prices) == 0
    shares = json.loads((env.shadow / "paper_broker.json").read_text())["positions"]
    assert shares and json.loads((env.shadow / "paper_broker.json").read_text())["total_dividends"] == 0.0
    assert _run(env, DAY + 4, prices=prices) == 0
    ex = _day(prices, DAY + 3)
    expected = sum(q * y * prices.close.loc[ex, t] for t, q in shares.items())
    state = json.loads((env.shadow / "paper_broker.json").read_text())
    assert state["total_dividends"] == pytest.approx(expected, rel=1e-9) and expected > 0
    credited = _events(env.shadow / "ledger.jsonl", "dividends")
    assert len(credited) == 1 and {c["ex_date"] for c in credited[0]["payload"]["credited"]} == {ex}
    assert _run(env, DAY + 5, prices=prices) == 0                   # never credited twice
    assert json.loads((env.shadow / "paper_broker.json").read_text())["total_dividends"] == pytest.approx(expected)
    assert json.loads((env.book / "paper_broker.json").read_text())["total_dividends"] > 0   # the live book too


def test_the_shadow_obeys_the_kill_switch(env, prices):
    env.runs.mkdir(parents=True)
    (env.runs / "KILL").touch()
    assert _run(env, DAY) == 0
    assert not (env.book / "pending_orders.json").exists() and not (env.shadow / "pending_orders.json").exists()
    assert any("KILL SWITCH" in m for e in _events(env.shadow / "ledger.jsonl", "risk_engine")
               for m in e["payload"]["messages"])


def test_a_forced_live_redecision_never_makes_the_shadow_decide_twice(env, prices, capsys):
    env.runs.mkdir(parents=True)
    (env.runs / "KILL").touch()                                     # nothing queued, so --force is allowed
    assert _run(env, DAY) == 0
    capsys.readouterr()
    assert _run(env, DAY, "--force") == 0
    assert "shadow book: skipped (the shadow book already decided" in capsys.readouterr().out
    assert len(_events(env.shadow / "ledger.jsonl", "cycle_end")) == 1
    assert [e["payload"]["status"] for e in _events(env.book / "ledger.jsonl", "shadow")] == ["ok", "skipped"]


def test_a_quant_only_live_book_keeps_no_shadow(tmp_path, prices, monkeypatch, capsys):
    env = _make_env(tmp_path, prices, monkeypatch, extra='[analysts]\nenabled = ["quant"]\n')
    path = Path(env.config)
    path.write_text(path.read_text().replace("memo_min_edge = 0.005\n", "memo_min_edge = 0.005\nmemo_enabled = false\n"
                                             "risk_review_enabled = false\nchallenge_enabled = false\n"))
    assert _run(env, DAY) == 0
    assert "shadow book: off (the live configuration is already quant-only" in capsys.readouterr().out
    assert not env.shadow.exists()


def test_no_shadow_flag_and_setting_turn_it_off(env, prices, capsys):
    assert _run(env, DAY, "--no-shadow") == 0
    assert "shadow book: off (evaluation.shadow_book is off)" in capsys.readouterr().out
    assert not env.shadow.exists() and not _events(env.book / "ledger.jsonl", "shadow")


# ----------------------------------------------------------------------------- isolation from the live book


_DECISION_EVENTS = ("believability", "calibration", "overlay_bound", "aggregate", "target_proposed", "risk_engine",
                    "orders", "view")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in ("run_id", "ts")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_the_shadow_never_changes_the_live_book(tmp_path, prices, monkeypatch):
    """Two identical books over four monthly cycles (forecasts resolve in between, so the
    shadow's mirrored records sit in the live store when later live decisions read it): one
    with the shadow, one without. Every live decision, order, state file and forecast is the same."""
    with_shadow = _make_env(tmp_path, prices, monkeypatch, name="a")
    without = _make_env(tmp_path, prices, monkeypatch, name="b")
    for i in (DAY, DAY + 21, DAY + 42, DAY + 63):
        assert _run(with_shadow, i) == 0
        assert _run(without, i, "--no-shadow") == 0
        a = json.loads((with_shadow.book / "pending_orders.json").read_text()) \
            if (with_shadow.book / "pending_orders.json").exists() else {"orders": []}
        b = json.loads((without.book / "pending_orders.json").read_text()) \
            if (without.book / "pending_orders.json").exists() else {"orders": []}
        assert a["orders"] == b["orders"]
    for f in ("paper_broker.json", "risk_state.json"):
        assert json.loads((with_shadow.book / f).read_text()) == json.loads((without.book / f).read_text())
    for event in _DECISION_EVENTS:
        ea = [_strip(e["payload"]) for e in _events(with_shadow.book / "ledger.jsonl", event)]
        eb = [_strip(e["payload"]) for e in _events(without.book / "ledger.jsonl", event)]
        assert ea == eb and ea, event
    ra = [r for r in _records(with_shadow.book) if r.agent != SHADOW_AGENT]
    rb = _records(without.book)
    key = lambda r: (r.as_of, r.ticker, r.agent, r.prob_outperform, r.resolved, r.outcome)   # noqa: E731
    assert sorted(map(key, ra)) == sorted(map(key, rb))
    shadow_recs = [r for r in _records(with_shadow.book) if r.agent == SHADOW_AGENT]
    assert shadow_recs and any(r.resolved for r in shadow_recs)    # they were in the store while it decided


def test_a_failing_shadow_never_fails_or_alters_the_live_run(env, prices, monkeypatch, capsys, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("shadow exploded")

    monkeypatch.setattr(shadow, "decide", boom)
    assert _run(env, DAY) == 0
    out = capsys.readouterr().out
    assert "warning: shadow book FAILED (the live decision above is unaffected): RuntimeError: shadow exploded" in out
    event = _events(env.book / "ledger.jsonl", "shadow")
    assert event[0]["payload"]["status"] == "failed" and "shadow exploded" in event[0]["payload"]["reason"]
    report = next((env.book / "reports").glob("decision_*.md")).read_text()
    assert "The shadow book failed this cycle" in report
    assert cli.main(["verify-ledger", "--config", env.config, "--offline"]) == 0
    other = _make_env(tmp_path, prices, monkeypatch, name="control")
    monkeypatch.setattr(shadow, "decide", boom)
    assert _run(other, DAY, "--no-shadow") == 0
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"] == \
        json.loads((other.book / "pending_orders.json").read_text())["orders"]

    monkeypatch.setattr(shadow, "decide", cli.decide)    # the next cycle: the shadow carries on from its fork
    assert _run(env, DAY + 1) == 0
    assert _events(env.book / "ledger.jsonl", "shadow")[-1]["payload"]["status"] == "ok"
    comp = shadow.compare_books(cli._load_config(env.config), env.book, _records(env.book),
                                as_of=_day(prices, DAY + 1), include_pre_cutoff=True)
    assert comp.shadow_missed == [_day(prices, DAY)] and "did not decide on 1 of the 2" in " ".join(comp.notes)


def test_a_failed_mirror_keeps_the_shadow_decision_and_says_what_is_lost(env, prices, monkeypatch, capsys):
    def boom(dec):
        raise OSError("store locked")

    monkeypatch.setattr(shadow, "shadow_forecast_records", boom)
    assert _run(env, DAY) == 0
    out = capsys.readouterr().out
    assert "shadow warning: shadow forecasts were not recorded in the live store (OSError: store locked)" in out
    assert _events(env.book / "ledger.jsonl", "shadow")[0]["payload"]["status"] == "ok"
    assert json.loads((env.shadow / "pending_orders.json").read_text())["orders"]
    assert not [r for r in _records(env.book) if r.agent == SHADOW_AGENT]


def test_a_failing_comparison_is_only_a_warning(env, prices, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("comparison exploded")

    monkeypatch.setattr(cli, "compare_books", boom)
    assert _run(env, DAY) == 0
    assert "warning: live-vs-shadow comparison failed (ValueError: comparison exploded)" in capsys.readouterr().out
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"]


def test_an_incomplete_shadow_is_refused_not_silently_reforked(env, prices, capsys):
    env.shadow.mkdir(parents=True)
    (env.shadow / "paper_broker.json").write_text(json.dumps({"cash": 5.0, "positions": {}}))
    assert _run(env, DAY) == 0
    out = capsys.readouterr().out
    assert "shadow book FAILED" in out and "incomplete" in out
    assert json.loads((env.shadow / "paper_broker.json").read_text()) == {"cash": 5.0, "positions": {}}
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"]


def test_the_shadow_only_ever_uses_the_paper_broker(tmp_path, prices, monkeypatch):
    """Even when the live book trades through Alpaca, the shadow is a paper book."""
    import aifirm.execution.broker as broker_mod

    def refuse(*a, **k):
        raise AssertionError("the shadow must never construct a real broker")

    monkeypatch.setattr(broker_mod.AlpacaBroker, "__init__", refuse)
    assert not hasattr(shadow, "AlpacaBroker")
    cfg = make_config(str(tmp_path / "runs"), memo_min_edge=0.005)
    cfg.portfolio.min_prob_to_hold = 0.5
    cfg.execution.broker = "alpaca"
    as_of = _day(prices, DAY)
    p, sheets = gather_data(cfg, as_of, False, prices=prices, **LOADERS)
    book = tmp_path / "runs"
    snap = shadow.BookSnapshot(as_of=as_of, cash=100_000.0, positions={}, nav=100_000.0,
                               risk_state=RiskState(peak_nav=100_000.0))
    out = shadow.run_shadow(cfg, book=book, runs_root=book, as_of=as_of, prices=p, sheets=sheets, live_run_id="r1",
                            live_start=snap)
    assert out.status == "ok", out.reason
    assert (book / "shadow" / "paper_broker.json").exists()
    assert json.loads((book / "shadow" / "pending_orders.json").read_text())["orders"]
    assert out.decision is not None and out.decision.mode == "quant" and out.decision.usage["calls"] == 0


def test_run_shadow_never_raises(tmp_path, prices, monkeypatch):
    cfg = make_config(str(tmp_path / "runs"))
    snap = shadow.BookSnapshot(as_of="2020-01-01", cash=1.0, positions={}, nav=1.0, risk_state=RiskState(peak_nav=1.0))
    out = shadow.run_shadow(cfg, book=tmp_path, runs_root=tmp_path, as_of=_day(prices, DAY), prices=prices, sheets={},
                            live_run_id="r", live_start=snap)
    assert out.status == "failed" and "live snapshot is from 2020-01-01" in out.reason


# ----------------------------------------------------------------------------- scoring


def test_shadow_forecasts_are_mirrored_once_and_scored(env, prices, capsys):
    assert _run(env, DAY) == 0
    recs = _records(env.book)
    shadow_recs = [r for r in recs if r.agent == SHADOW_AGENT]
    live_run = next(e["payload"]["run_id"] for e in _events(env.book / "ledger.jsonl", "cycle_start"))
    assert sorted(r.ticker for r in shadow_recs) == sorted(NAMES)          # one per name the shadow scored
    assert all(r.model == "deterministic" and not r.pre_cutoff and r.run_id == f"{live_run}-shadow"
               for r in shadow_recs)
    # the shadow's own quant analyst records stay in its own store: the live quant is not counted twice
    assert all(r.run_id == live_run for r in recs if r.agent != SHADOW_AGENT)
    with ForecastStore(env.shadow / "forecasts.sqlite") as store:
        own = {(r.ticker, r.prob_outperform) for r in store.all() if r.agent == "committee"}
    assert own == {(r.ticker, r.prob_outperform) for r in shadow_recs}
    committee = {r.ticker: r.resolve_on for r in recs if r.agent == "committee"}
    assert all(r.resolve_on == committee[r.ticker] for r in shadow_recs if r.ticker in committee)

    for i in (DAY + 21, DAY + 42):
        assert _run(env, i) == 0
    capsys.readouterr()
    assert cli.main(["score", "--config", env.config, "--offline", "--as-of", _day(prices, DAY + 70)]) == 0
    out = capsys.readouterr().out
    assert "\nshadow " in out and "paired IC, live final P vs quant-only shadow:" in out
    assert cli.main(["compare", "--config", env.config, "--offline", "--as-of", _day(prices, DAY + 70)]) == 0
    out = capsys.readouterr().out
    assert "AI value added net of LLM cost" in out and "not yet measurable: 3 of 24 resolved decision dates" in out


def test_the_shadow_agent_never_reaches_a_live_decision_input():
    """Regression guard: mirrored shadow records must not change the pooled analyst IC, the
    believability weights or the calibration maps that live decisions read."""
    def rec(agent, t, d, p, x):
        return ForecastRecord(id=f"r|{d}|{t}|{agent}", run_id="r", agent=agent, model="deterministic", ticker=t,
                              benchmark="SPY", as_of=d, horizon_days=21, resolve_on="2024-03-01",
                              prob_outperform=p, expected_excess_return_pct=0.0, resolved=True, outcome=int(x > 0),
                              realized_excess_return_pct=x, brier=(p - int(x > 0)) ** 2)
    base = [rec("quant", t, "2024-01-02", 0.5 + 0.01 * k, 1.0 * k - 2) for k, t in enumerate("ABCDE")]
    base += [rec("quant", t, "2024-01-03", 0.5 + 0.01 * k, 1.0 * k - 2) for k, t in enumerate("ABCDE")]
    poison = [rec(SHADOW_AGENT, t, d, 0.9 - 0.1 * k, 1.0 * k - 2)
              for d in ("2024-01-02", "2024-01-03") for k, t in enumerate("ABCDE")]
    for f in (lambda rs: resolved_rank_ic(rs, "2024-12-31", agent=None, include_pre_cutoff=True, min_names=3),
              lambda rs: believability_weights(rs, "2024-12-31", ["quant"], min_n=1, min_dates=1),
              lambda rs: {a: c.to_dict() for a, c in fit_calibrators(rs, "2024-12-31", ["quant"], min_n=1).items()}):
        assert f(base) == f(base + poison)


# ----------------------------------------------------------------------------- comparing the books


def _ledger(path, entries):
    led = Ledger(path)
    for event, payload in entries:
        led.append(event, payload)


def test_compare_books_arithmetic_and_point_in_time_cut(tmp_path):
    cfg = make_config(str(tmp_path))
    book = tmp_path / "book"
    _ledger(book / "ledger.jsonl", [
        ("cycle_start", {"as_of": "2024-01-02", "nav": 100_000.0}),          # before the shadow existed
        ("cycle_end", {"as_of": "2024-01-02", "usage": {"cost_usd": 50.0}}),
        ("cycle_start", {"as_of": "2024-02-01", "nav": 101_000.0}),
        ("cycle_end", {"as_of": "2024-02-01", "usage": {"cost_usd": 3.0}}),
        ("cycle_start", {"as_of": "2024-03-01", "nav": 103_000.0}),
        ("cycle_end", {"as_of": "2024-03-01", "usage": {"cost_usd": 4.0}}),
        ("cycle_start", {"as_of": "2024-04-01", "nav": 104_000.0}),
        ("cycle_end", {"as_of": "2024-04-01", "usage": {"cost_usd": 5.0}}),
        ("cycle_start", {"as_of": "2024-05-01", "nav": 90_000.0}),           # after the as_of cut
        ("cycle_end", {"as_of": "2024-05-01", "usage": {"cost_usd": 6.0}})])
    _ledger(book / "shadow" / "ledger.jsonl", [
        ("shadow_inception", {"as_of": "2024-02-01", "nav": 101_000.0}),
        ("cycle_start", {"as_of": "2024-02-01", "nav": 101_000.0}),
        ("cycle_start", {"as_of": "2024-04-01", "nav": 102_000.0}),           # missed 2024-03-01
        ("cycle_start", {"as_of": "2024-05-01", "nav": 95_000.0})])
    c = shadow.compare_books(cfg, book, [], as_of="2024-04-15")
    assert c.nav_date == "2024-04-01" and c.live_nav == 104_000.0 and c.shadow_nav == 102_000.0
    assert c.difference_usd == pytest.approx(2_000.0)
    assert c.llm_spend_usd == pytest.approx(3.0 + 4.0 + 5.0)               # inception through the NAV date only
    assert c.value_added_usd == pytest.approx(2_000.0 - 12.0)
    assert c.value_added_pct == pytest.approx((2_000.0 - 12.0) / 101_000.0 * 100)
    assert c.live_return_pct == pytest.approx((104 / 101 - 1) * 100)
    assert c.shadow_return_pct == pytest.approx((102 / 101 - 1) * 100)
    assert c.live_cycles == 3 and c.shadow_missed == ["2024-03-01"]
    assert c.min_dates == shadow.forward_min_dates(cfg, c.step_days) and c.step_days in (21, 22)
    assert c.status == "not_measurable" and c.verdict.startswith(f"not yet measurable: 0 of {c.min_dates} ")
    assert any("AI value added net of LLM cost: +$1,988.00" in line for line in c.lines())
    later = shadow.compare_books(cfg, book, [])
    assert later.nav_date == "2024-05-01" and later.value_added_usd == pytest.approx(-5_000.0 - 18.0)


def test_compare_without_a_shadow_or_a_common_session_says_so(tmp_path, capsys):
    cfg = make_config(str(tmp_path))
    c = shadow.compare_books(cfg, tmp_path / "nobook", [])
    assert c.status == "no_shadow" and "No shadow book yet" in c.verdict and c.value_added_usd is None
    book = tmp_path / "book"
    _ledger(book / "shadow" / "ledger.jsonl", [("shadow_inception", {"as_of": "2024-02-01", "nav": 1.0})])
    c = shadow.compare_books(cfg, book, [])
    assert c.status == "no_common_decision" and "no decided session in common" in c.verdict


def test_ic_verdicts_are_plain_and_need_a_forward_record():
    status, text = shadow.ic_verdict(0, 24, None, None, 21)
    assert status == "not_measurable" and text.startswith("not yet measurable: 0 of 24 resolved decision dates")
    status, text = shadow.ic_verdict(10, 24, 0.05, 3.1, 21)
    assert status == "not_measurable" and "10 of 24" in text and "+0.050 (t = 3.10)" in text   # promising is not proof
    assert shadow.ic_verdict(30, 24, 0.04, 2.5, 21)[0] == "better"
    assert shadow.ic_verdict(30, 24, -0.04, -2.5, 21)[0] == "worse"
    status, text = shadow.ic_verdict(30, 24, 0.01, 1.2, 21)
    assert status == "no_difference" and "default to the simpler quant-only design" in text
    assert shadow.ic_verdict(30, 24, 0.01, None, 21)[0] == "no_difference"


def test_the_forward_record_needed_follows_the_cadence():
    cfg = make_config("runs")
    assert shadow.forward_min_dates(cfg, 21) == 24                  # monthly decisions: two years
    assert shadow.forward_min_dates(cfg, 5) == math.ceil(504 / 5)
    assert shadow.forward_min_dates(cfg, 1) == 504
    cfg.process.overlay_min_dates = 40
    assert shadow.forward_min_dates(cfg, 21) == 40
    assert shadow.decision_step_days(["2024-01-02"], 21) == 21
    assert shadow.decision_step_days(["2024-01-02", "2024-01-09", "2024-01-16", "2024-02-13"], 21) == 5


# ----------------------------------------------------------------------------- book plumbing


def test_paper_broker_fork_copies_state_and_refuses_nonsense(tmp_path):
    b = PaperBroker.from_snapshot(500.0, {"AAA": 3.0, "BBB": 0.0}, make_config("x").costs,
                                  state_path=str(tmp_path / "b.json"), marks={"AAA": 10.0},
                                  dividends_through="2024-01-05")
    assert b.positions() == {"AAA": 3.0} and b.cash() == 500.0 and b.nav({}) == pytest.approx(530.0)
    again = PaperBroker(1.0, make_config("x").costs, state_path=str(tmp_path / "b.json"))
    assert again.load() and again.positions() == {"AAA": 3.0} and again.dividends_through == "2024-01-05"
    assert again.total_commission == 0.0 and again.total_dividends == 0.0
    for bad in ({"AAA": -1.0}, {"AAA": float("nan")}):
        with pytest.raises(ValueError):
            PaperBroker.from_snapshot(1.0, bad, make_config("x").costs)
    with pytest.raises(ValueError):
        PaperBroker.from_snapshot(-1.0, {}, make_config("x").costs)


def test_shadow_holdings_outside_the_live_prices_are_fetched_on_the_same_dates(prices):
    base = _window(prices, ["AAA", "SPY"], prices.close.index[0], prices.close.index[DAY])
    asked = []

    def loader(tickers, start, end):
        asked.append((list(tickers), str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))
        return _window(prices, tickers, start, end)

    as_of = _day(prices, DAY)
    assert shadow._covering_prices(base, ["AAA"], as_of, None) is base           # nothing missing: unchanged
    got = shadow._covering_prices(base, ["AAA", "FFF"], as_of, loader)
    assert asked == [(["FFF"], _day(prices, 0), as_of)]
    assert list(got.close.index) == list(base.close.index) and set(got.close.columns) == {"AAA", "SPY", "FFF"}
    assert got.close["FFF"].equals(prices.close["FFF"].loc[:as_of])
    assert got.close["AAA"].equals(base.close["AAA"])
    with pytest.raises(BookError, match="no price loader"):
        shadow._covering_prices(base, ["FFF"], as_of, None)
    with pytest.raises(BookError, match="no prices for shadow holding"):
        shadow._covering_prices(base, ["ZZZ"], as_of, lambda t, s, e: _window(prices, ["FFF"], s, e))
