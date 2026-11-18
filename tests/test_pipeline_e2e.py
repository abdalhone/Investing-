"""End-to-end tests of one decision cycle and of the CLI, fully offline (synthetic data,
offline LLM responders; no network, no API key)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, fake_filings, fake_fundamentals, make_config, make_prices, perturb_after

from aifirm import cli
from aifirm.agents import make_offline_llm
from aifirm.data.news import Headline
from aifirm.data.prices import PriceData
from aifirm.journal import ForecastStore, Ledger, LessonStore
from aifirm.llm.client import OfflineLLM
from aifirm.pipeline import MAX_LESSONS_PER_CYCLE, decide, gather_data, market_snapshot, quant_only_config
from aifirm.portfolio import RiskState
from aifirm.report import render_decision_report

DAY = 400
NAV = 100_000.0


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


def _cfg(tmp_path, **process):
    cfg = make_config(str(tmp_path / "runs"), memo_min_edge=0.005, **process)
    cfg.portfolio.min_prob_to_hold = 0.5       # the offline rules are modest; let small edges trade
    return cfg


class Book:
    """The persistent stores of one book (forecasts, lessons, ledger) in a directory."""

    def __init__(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        self.store = ForecastStore(d / "forecasts.sqlite")
        self.lessons = LessonStore(d / "lessons.jsonl")
        self.ledger = Ledger(d / "ledger.jsonl")


@pytest.fixture
def books(tmp_path):
    """Book factory; closes every forecast store at teardown (no leaked connections)."""
    made = []

    def make(name: str) -> Book:
        made.append(Book(tmp_path / name))
        return made[-1]

    yield make
    for b in made:
        b.store.close()


def _holdings(prices, as_of, weights):
    """Shares for target weights at the as-of close (so current weights and positions agree)."""
    px = market_snapshot(prices, as_of, weights).prices
    return {t: w * NAV / px[t] for t, w in weights.items()}


def _decide(cfg, llm, prices, as_of, book, current=None, run_id="r1", prices_for_data=None, **kw):
    _, sheets = gather_data(cfg, as_of, live=False, prices=prices_for_data or prices, **LOADERS)
    current = current or {}
    return decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=kw.pop("nav", NAV),
                  prev_nav=kw.pop("prev_nav", None), risk_state=kw.pop("state", RiskState(peak_nav=NAV)),
                  forecast_store=book.store,
                  lesson_store=book.lessons, ledger=book.ledger, run_id=run_id,
                  positions=_holdings(prices, as_of, current), trading_calendar=prices.close.index, **kw)


# ----------------------------------------------------------------------------- one cycle


def test_offline_cycle_runs_end_to_end_and_journals_everything(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    book = books("book")
    as_of = _day(prices, DAY)
    dec = _decide(cfg, make_offline_llm(), prices, as_of, book)

    assert dec.mode == "offline" and dec.as_of == as_of and dec.benchmark == "SPY"
    assert list(dec.committee) == NAMES                           # config order
    for r in dec.committee.values():
        assert sorted(v.agent for v in r.views) == ["events", "fundamental", "macro", "quant", "technical"]
    assert any(r.memo is not None for r in dec.committee.values())  # memo threshold lowered for the test

    # code sized the book within the hard limits, from cash: buys only, capped by ADV
    w = dec.target_final.weights()
    assert all(0 < x <= cfg.portfolio.max_position + 1e-9 for x in w.values())
    assert dec.target_final.cash_weight >= cfg.portfolio.min_cash - 1e-9
    assert dec.orders and all(o.side == "buy" for o in dec.orders)
    for o in dec.orders:
        assert o.notional <= cfg.risk.max_adv_participation * dec.market.adv[o.ticker] + 1e-6
        assert o.reference_price == pytest.approx(prices.close.loc[as_of, o.ticker])

    # every non-abstaining view plus each committee pool became a stored forecast
    n_expected = sum(sum(not v.view.abstain for v in r.views) + 1 for r in dec.committee.values())
    assert len(dec.forecasts) == n_expected == len(book.store.all())
    assert all(f.resolve_on > as_of for f in dec.forecasts)

    # the ledger holds every step and verifies
    events = {e["event"] for e in book.ledger.entries()}
    assert {"cycle_start", "factsheet", "believability", "view", "aggregate", "memo", "target_proposed",
            "risk_review", "risk_engine", "orders", "forecasts_written", "cycle_end"} <= events
    assert book.ledger.verify() == (True, len(book.ledger))
    assert dec.ledger_head == book.ledger.head

    md = render_decision_report(dec)
    assert "# Decision report" in md and "OFFLINE MODE" in md and "## Orders" in md and "## IC memos" in md
    for t in NAMES:
        assert f"| {t} |" in md


def test_gather_data_turns_source_failures_into_data_gaps(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    cfg.universe.tickers = [*NAMES, "ZZZ"]                        # ZZZ has no prices at all
    as_of = _day(prices, DAY)

    def fundamentals(t, d):
        if t == "BBB":
            raise RuntimeError("SEC is down")
        return fake_fundamentals(t, d)

    def filings(t, d):
        if t == "CCC":
            raise TimeoutError("timeout")
        return fake_filings(t, d)

    def macro(d):
        raise ConnectionError("FRED unreachable")

    p, sheets = gather_data(cfg, as_of, live=False, prices=prices, fundamentals_loader=fundamentals,
                            macro_loader=macro, filings_loader=filings)
    assert p.close.index[-1] <= pd.Timestamp(as_of)               # returned prices are cut at as_of
    assert sorted(sheets) == NAMES                                 # ZZZ skipped, nothing crashed
    assert any("SEC fundamentals could not be loaded" in g for g in sheets["BBB"].data_gaps)
    assert any("filing events could not be loaded" in g for g in sheets["CCC"].data_gaps)
    assert all(any("Macro data could not be loaded" in g for g in fs.data_gaps) for fs in sheets.values())
    assert sheets["AAA"].get("AAA.pe_ttm") is not None            # healthy sources still used
    assert sheets["AAA"].sector == "Tech"                          # manual sector label

    book = books("g")
    dec = decide(cfg, make_offline_llm(), as_of, prices, sheets, current_weights={}, nav=NAV, prev_nav=None,
                 risk_state=RiskState(peak_nav=NAV), forecast_store=book.store, lesson_store=book.lessons,
                 ledger=book.ledger, run_id="g")
    assert any("ZZZ" in w for w in dec.warnings)
    assert "BBB" in dec.data_gaps


def test_headlines_only_live_and_only_when_news_is_enabled(tmp_path, prices):
    as_of = _day(prices, DAY)
    calls = []

    def news(t, d):
        calls.append(t)
        return [Headline(title=f"{t} news", publisher="Wire", published=f"{d}T12:00:00Z")]

    cfg = _cfg(tmp_path)
    _, sheets = gather_data(cfg, as_of, live=False, prices=prices, headlines_loader=news, **LOADERS)
    assert calls == [] and all(fs.headlines == [] for fs in sheets.values())
    _, sheets = gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=news, **LOADERS)
    assert calls == NAMES and sheets["AAA"].headlines[0].title == "AAA news"
    calls.clear()
    cfg.process.news_enabled = False
    gather_data(cfg, as_of, live=True, prices=prices, headlines_loader=news, **LOADERS)
    assert calls == []


def test_missing_benchmark_stops_the_run(tmp_path, prices):
    cfg = _cfg(tmp_path)
    cfg.universe.benchmark = "QQQ"
    with pytest.raises(RuntimeError, match="benchmark"):
        gather_data(cfg, _day(prices, DAY), live=False, prices=prices, **LOADERS)


def test_decision_is_unchanged_when_prices_after_the_decision_date_change(tmp_path, prices, books):
    """No look-ahead (P3): scrambling every price and volume after the decision date
    leaves the decision on that date bit-for-bit identical."""
    as_of = _day(prices, DAY)
    current = {"AAA": 0.05, "CCC": 0.04, "SPY": 0.0}
    future_changed = perturb_after(prices, as_of)
    assert not future_changed.close.equals(prices.close)
    a = _decide(_cfg(tmp_path / "a"), make_offline_llm(), prices, as_of, books("a"), current)
    b = _decide(_cfg(tmp_path / "b"), make_offline_llm(), future_changed, as_of, books("b"), current)

    assert {t: x.model_dump() for t, x in a.aggregates.items()} == {t: x.model_dump() for t, x in b.aggregates.items()}
    assert {t: r.memo.model_dump() if r.memo else None for t, r in a.committee.items()} == \
           {t: r.memo.model_dump() if r.memo else None for t, r in b.committee.items()}
    assert a.target_final.model_dump() == b.target_final.model_dump()
    assert [o.model_dump() for o in a.orders] == [o.model_dump() for o in b.orders]
    assert [f.model_dump() for f in a.forecasts] == [f.model_dump() for f in b.forecasts]


def test_kill_switch_blocks_every_order(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    as_of = _day(prices, DAY)
    current = {"AAA": 0.05, "SPY": 0.0}
    free = _decide(cfg, make_offline_llm(), prices, as_of, books("free"), current)
    assert free.orders                                             # without the switch there is something to do

    kill = Path(cfg.runs_dir) / cfg.risk.kill_switch_file
    kill.parent.mkdir(parents=True, exist_ok=True)
    kill.touch()
    dec = _decide(cfg, make_offline_llm(), prices, as_of, books("killed"), current)
    assert dec.orders == []
    assert any("KILL SWITCH" in m for m in dec.risk_messages)
    assert dec.target_final.weights() == {"AAA": pytest.approx(0.05)}
    assert "KILL SWITCH" in render_decision_report(dec)


def test_quant_only_cycle_needs_no_llm_and_llm_configs_refuse_none(tmp_path, prices, books):
    as_of = _day(prices, DAY)
    cfg = _cfg(tmp_path)
    dec = _decide(quant_only_config(cfg), None, prices, as_of, books("q"))
    assert dec.mode == "quant" and dec.usage["calls"] == 0
    assert all([v.agent for v in r.views] == ["quant"] for r in dec.committee.values())
    with pytest.raises(ValueError, match="no LLM"):
        _decide(cfg, None, prices, as_of, books("x"))


def test_llm_outage_holds_the_book_instead_of_trading(tmp_path, prices, books):
    """Every LLM call fails: no orders (fail closed), the held book is kept, and it says why."""
    as_of = _day(prices, DAY)
    current = {"AAA": 0.05, "CCC": 0.04, "SPY": 0.0}
    dec = _decide(_cfg(tmp_path), OfflineLLM({}), prices, as_of, books("o"), current)
    assert dec.held_by_outage and dec.orders == []
    assert any("LLM OUTAGE" in w for w in dec.warnings)
    assert dec.target_final.weights() == {"AAA": pytest.approx(0.05), "CCC": pytest.approx(0.04)}
    assert dec.risk_review is None


def test_lessons_are_learned_after_resolution_and_only_used_from_then_on(tmp_path, prices, books):
    cfg = _cfg(tmp_path)
    book = books("book")
    d1, d2 = _day(prices, DAY), _day(prices, DAY + cfg.process.horizon_days + 2)
    first = _decide(cfg, make_offline_llm(), prices, d1, book, run_id="run1")
    assert first.resolved == [] and first.lessons_written == []
    second = _decide(cfg, make_offline_llm(), prices, d2, book, run_id="run2")

    assert second.resolved and all(r.resolve_on <= d2 for r in second.resolved)
    assert 0 < len(second.lessons_written) <= MAX_LESSONS_PER_CYCLE
    assert {x.agent for x in second.lessons_written} <= {"fundamental", "technical", "macro", "events", "committee"}
    agent = second.lessons_written[0].agent
    resolved_on = min(r.resolve_on for r in second.resolved)
    assert book.lessons.retrieve(agent, d1) == []                 # unknown at the first decision
    assert book.lessons.retrieve(agent, resolved_on)              # known from the resolution date on
    assert sum(1 for e in book.ledger.entries() if e["event"] == "lesson") == len(second.lessons_written)
    assert all(w == 1.0 for w in second.believability.values())   # far below min_n resolved questions


def test_market_snapshot_leaves_unknown_values_out(prices):
    as_of = _day(prices, DAY)
    snap = market_snapshot(prices, as_of, ["AAA", "SPY", "NOPE"])
    assert set(snap.prices) == set(snap.adv) == set(snap.daily_vol) == {"AAA", "SPY"}
    assert snap.adv["AAA"] > 0 and 0 < snap.daily_vol["AAA"] < 0.2


# ----------------------------------------------------------------------------- CLI


def _window(p, tickers, start, end):
    lo, hi, cols = pd.Timestamp(start), pd.Timestamp(end), sorted(tickers)
    cut = lambda df: None if df is None else df.loc[lo:hi, cols]   # noqa: E731
    return PriceData(close=cut(p.close), adj_close=cut(p.adj_close), volume=cut(p.volume), open=cut(p.open),
                     raw_close=cut(p.raw_close), splits=None)


@pytest.fixture
def cli_env(tmp_path, prices, monkeypatch):
    """A config file, with prices and fact-sheet sources served from the synthetic market."""
    real_gather = gather_data

    def fake_gather(cfg, as_of, live, prices=None):
        return real_gather(cfg, as_of, False, prices=prices, **LOADERS)

    def fake_load_prices(tickers, start, end):
        # like the real loader: only the requested tickers, rows from start to end inclusive
        assert set(tickers) <= {*synthetic.close.columns}
        return _window(synthetic, tickers, start, end)

    synthetic = prices
    monkeypatch.setattr(cli, "gather_data", fake_gather)
    monkeypatch.setattr(cli, "load_prices", fake_load_prices)
    runs = tmp_path / "runs"
    config = tmp_path / "firm.toml"
    config.write_text(f'runs_dir = "{runs}"\n[universe]\ntickers = {json.dumps(NAMES)}\nbenchmark = "SPY"\n'
                      '[process]\nmemo_min_edge = 0.005\n[portfolio]\nmin_prob_to_hold = 0.5\n')
    return config, runs


def test_cli_offline_run_queues_orders_then_fills_them_at_the_next_open(cli_env, prices, capsys):
    config, runs = cli_env
    book = runs / "offline"
    d1, d2 = _day(prices, DAY), _day(prices, DAY + 1)
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", d1]) == 0
    pending = json.loads((book / "pending_orders.json").read_text())
    assert pending["as_of"] == d1 and pending["orders"]
    assert list((book / "reports").glob(f"decision_{d1}_*.md"))
    assert json.loads((book / "risk_state.json").read_text())["peak_nav"] == pytest.approx(100_000.0)

    # deciding again before the next open would double the plan: refused
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", d1]) == 1
    assert "waiting for the next trading day's open" in capsys.readouterr().err

    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", d2]) == 0
    fills = [e for e in Ledger(book / "ledger.jsonl").entries() if e["event"] == "fills"]
    assert len(fills) == 1 and fills[0]["payload"]["fill_date"] == d2
    for f in fills[0]["payload"]["fills"]:
        assert f["side"] == "buy" and f["date"] == d2
        assert f["price"] >= prices.open.loc[d2, f["ticker"]]          # the next open plus slippage,
        assert f["price"] <= prices.open.loc[d2, f["ticker"]] * 1.01   # not the decision-day close
    state = json.loads((book / "paper_broker.json").read_text())
    assert set(state["positions"]) == {f["ticker"] for f in fills[0]["payload"]["fills"]}

    assert cli.main(["verify-ledger", "--config", str(config), "--offline"]) == 0
    assert cli.main(["score", "--config", str(config), "--offline", "--as-of", d2]) == 0
    assert "No resolved forecasts yet" in capsys.readouterr().out

    # tampering with any entry is detected
    path = book / "ledger.jsonl"
    lines = path.read_text().splitlines()
    lines[3] = lines[3].replace('"run_id"', '"run_iD"', 1)
    path.write_text("\n".join(lines) + "\n")
    assert cli.main(["verify-ledger", "--config", str(config), "--offline"]) == 1


def test_cli_sells_a_held_name_that_left_the_universe(cli_env, prices):
    config, runs = cli_env
    book = runs / "offline"
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 1)]) == 0
    held = set(json.loads((book / "paper_broker.json").read_text())["positions"])
    dropped = sorted(held)[0]
    config.write_text(config.read_text().replace(f'"{dropped}", ', "").replace(f', "{dropped}"', ""))
    assert dropped not in cli._load_config(str(config)).universe.tickers
    (book / "pending_orders.json").unlink(missing_ok=True)
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 2)]) == 0
    orders = json.loads((book / "pending_orders.json").read_text())["orders"]
    assert any(o["ticker"] == dropped and o["side"] == "sell" for o in orders)


def test_cli_documented_unfreeze_procedure_works(cli_env, prices):
    """Regression: after a freeze, setting "frozen": false in risk_state.json (the README's
    procedure) must unfreeze the book. Before, the unchanged peak re-froze it on the next run
    because a frozen book sits in cash and can never regain the old peak."""
    config, runs = cli_env
    book = runs / "offline"
    book.mkdir(parents=True, exist_ok=True)
    state_path = book / "risk_state.json"
    state_path.write_text(json.dumps({"peak_nav": 125_000.0, "frozen": True, "last_nav": 100_000.0}))
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    frozen = json.loads(state_path.read_text())
    assert frozen["frozen"] and frozen["frozen_at_nav"] == pytest.approx(100_000.0)
    assert not (book / "pending_orders.json").exists()             # frozen: the active sleeve stays in cash
    frozen["frozen"] = False                                       # the human decision, as documented
    state_path.write_text(json.dumps(frozen))
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 1)]) == 0
    after = json.loads(state_path.read_text())
    assert not after["frozen"] and after["peak_nav"] == pytest.approx(100_000.0) and after["frozen_at_nav"] is None
    assert json.loads((book / "pending_orders.json").read_text())["orders"]   # trading again


def test_cli_kill_switch_file_stops_live_orders(cli_env, prices):
    config, runs = cli_env
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "KILL").touch()
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert not (runs / "offline" / "pending_orders.json").exists()


def test_cli_refuses_llm_runs_without_credentials(cli_env, tmp_path, monkeypatch, capsys):
    config, _ = cli_env
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    assert cli.credentials_problem() is not None
    assert cli.main(["run", "--config", str(config)]) == 1
    assert "credentials" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")            # a key is enough; never blocks
    assert cli.credentials_problem() is None


def test_last_completed_session_never_uses_an_unfinished_day():
    ny = cli.MARKET_TZ
    assert cli.last_completed_session(pd.Timestamp("2026-11-17 10:00", tz=ny).to_pydatetime()) == "2026-11-16"
    assert cli.last_completed_session(pd.Timestamp("2026-11-17 17:00", tz=ny).to_pydatetime()) == "2026-11-17"
    assert cli.last_completed_session(pd.Timestamp("2026-11-16 09:00", tz=ny).to_pydatetime()) == "2026-11-13"
    assert cli.last_completed_session(pd.Timestamp("2026-11-15 12:00", tz=ny).to_pydatetime()) == "2026-11-13"



def test_example_config_documents_exactly_the_defaults():
    from aifirm.config import FirmConfig

    example = Path(__file__).resolve().parents[1] / "examples" / "firm.toml"
    assert FirmConfig.load(example).model_dump() == FirmConfig().model_dump()


# ----------------------------------------------------------------------------- regressions (risk path, abstention)


def _abstaining_llm():
    from aifirm.schemas import AnalystView

    llm = make_offline_llm()
    llm.register("analyst.", lambda role, system, user, schema: AnalystView(
        ticker="STOCK", abstain=True, abstain_reason="Key data for this slice are missing.", prob_outperform=0.5,
        expected_excess_return_pct=0.0, scenarios=[], evidence=[], key_risks=[], what_would_change_my_mind="",
        private_insight=""))
    return llm


def test_every_analyst_abstaining_holds_the_book_instead_of_liquidating(tmp_path, prices, books):
    """Regression: valid abstentions from every analyst pooled to 0.5 and sold the held names,
    although no information was produced. They are now held (like an outage), with a warning."""
    cfg = _cfg(tmp_path)
    cfg.analysts.enabled = ["fundamental", "technical", "macro", "events"]
    cfg.process.risk_review_enabled = False
    as_of = _day(prices, DAY)
    current = {"AAA": 0.08, "BBB": 0.05, "SPY": 0.0}
    dec = _decide(cfg, _abstaining_llm(), prices, as_of, books("abstain"), current)
    assert not dec.held_by_outage
    assert all(a.n_views == 0 for a in dec.aggregates.values())
    assert dec.orders == []
    assert dec.target_final.weights() == {"AAA": pytest.approx(0.08), "BBB": pytest.approx(0.05)}
    assert any("every analyst abstained on held name(s) AAA, BBB" in w for w in dec.warnings)
    assert any(w.startswith("technical abstained on all 6 names") for w in dec.warnings)


def test_quant_with_fewer_than_three_names_explains_its_abstention(tmp_path, prices, books):
    cfg = quant_only_config(_cfg(tmp_path))
    cfg.universe.tickers = ["EEE", "FFF"]
    dec = _decide(cfg, None, prices, _day(prices, DAY), books("q2"))
    assert dec.orders == [] and all(r.views[0].view.abstain for r in dec.committee.values())
    assert any(w.startswith("quant abstained on all 2 names") for w in dec.warnings)


def test_nav_marks_since_the_last_decision_reach_the_risk_state(tmp_path, prices, books):
    """Regression: only decision-day NAVs reached the risk state, so an intra-cycle peak and a
    loss day between rebalances were invisible to the drawdown freeze and the daily-loss halt."""
    cfg = quant_only_config(_cfg(tmp_path))
    as_of = _day(prices, DAY)
    # a peak of 130k inside the cycle: today's 100k is a 23% drawdown -> freeze
    dec = _decide(cfg, None, prices, as_of, books("peak"), prev_nav=NAV, nav_marks=[NAV, 130_000.0, 101_000.0])
    assert dec.risk_state.peak_nav == pytest.approx(130_000.0) and dec.risk_state.frozen
    assert dec.risk_state.last_as_of == as_of
    # a -8% day in the middle of the cycle -> no buys this cycle, although today is flat
    dec2 = _decide(cfg, None, prices, as_of, books("loss"), prev_nav=NAV, nav_marks=[NAV, 92_000.0, NAV])
    assert any("DAILY-LOSS HALT" in m for m in dec2.risk_messages) and dec2.orders == []
    free = _decide(cfg, None, prices, as_of, books("calm"), prev_nav=NAV, nav_marks=[NAV, 99_000.0, NAV])
    assert free.orders and not any("HALT" in m for m in free.risk_messages)


def test_live_book_rebuilds_the_nav_path_since_the_previous_run(tmp_path, prices, books):
    """Without nav_marks (the CLI), the path since the last decision is rebuilt from the held
    shares, starting at the NAV recorded at that decision; a stale caller prev_nav is not used."""
    cfg = quant_only_config(_cfg(tmp_path))
    last, today = DAY - 5, DAY
    shares = {"SPY": NAV / float(prices.close["SPY"].iloc[last])}       # 100% SPY since the last decision
    closes = prices.close["SPY"].iloc[last:today + 1]
    path = list(shares["SPY"] * closes)
    nav = path[-1]
    state = RiskState(peak_nav=NAV, last_nav=NAV, last_as_of=_day(prices, last))
    _, sheets = gather_data(cfg, _day(prices, today), live=False, prices=prices, **LOADERS)
    book = books("live")
    dec = decide(cfg, None, _day(prices, today), prices, sheets, current_weights={"SPY": 1.0}, nav=nav,
                 prev_nav=nav * 2, risk_state=state, forecast_store=book.store, lesson_store=book.lessons,
                 ledger=book.ledger, run_id="live", positions=shares)
    entry = next(e["payload"] for e in book.ledger.entries() if e["event"] == "risk_engine")
    assert entry["nav_path_days"] == today - last
    worst = min(b / a - 1 for a, b in zip(path, path[1:]))
    assert entry["worst_day_return"] == pytest.approx(worst)
    assert dec.risk_state.peak_nav == pytest.approx(max(path))
    assert not any("HALT" in m for m in dec.risk_messages)              # the bogus prev_nav (-50%) is ignored


# ----------------------------------------------------------------------------- regressions (committee/risk-review wiring)


def test_failed_memo_on_a_held_name_keeps_its_weight_through_the_pipeline(tmp_path, prices, books):
    """Regression (#18): the committee flags a held name whose required memo failed with
    hold_current, but the pipeline ignored the flag and re-sized the name from the unreviewed
    pooled view (FFF 3% -> 9.55%, a $6,552 buy caused by an LLM failure, with no warning)."""
    from aifirm.agents.offline import memo_responder
    from aifirm.llm.client import LLMCallFailed

    def failing_memo(role, system, user, schema):
        if 'Use ticker "FFF"' in user:
            raise LLMCallFailed("memo: model refused")
        return memo_responder(role, system, user, schema)

    llm = make_offline_llm()
    llm.register("memo", failing_memo)
    current = {"FFF": 0.03, "EEE": 0.08}
    dec = _decide(_cfg(tmp_path), llm, prices, _day(prices, DAY), books("memo_fail"), current)
    assert dec.committee["FFF"].hold_current and not dec.held_by_outage
    assert dec.target_final.weights()["FFF"] == pytest.approx(0.03)
    assert not [o for o in dec.orders if o.ticker == "FFF"]
    assert any("required IC memo failed on held name(s) FFF" in w for w in dec.warnings)


def test_risk_review_halt_keeps_the_book_instead_of_liquidating(tmp_path, prices, books):
    """Regression (#19): the reviewer is told that halt_trading keeps every position at its
    current weight, but the pipeline did not pass the current book to apply_risk_review, so a
    halt zeroed the active sleeve (sell AAA $8,000 and BBB $6,000)."""
    from aifirm.schemas import RiskReview

    llm = make_offline_llm()
    llm.register("risk_review", lambda role, system, user, schema: RiskReview(
        summary="one value looks broken; pause", concerns=[], hidden_concentrations=[], halt_trading=True))
    current = {"AAA": 0.08, "BBB": 0.06, "SPY": 0.0}
    dec = _decide(_cfg(tmp_path), llm, prices, _day(prices, DAY), books("halt"), current)
    assert dec.risk_review is not None and dec.risk_review.halt_trading
    assert dec.target_reviewed.weights() == {"AAA": pytest.approx(0.08), "BBB": pytest.approx(0.06)}
    assert dec.orders == []
    assert any("RISK REVIEW HALT" in w for w in dec.warnings)


def test_shipped_defaults_run_end_to_end(tmp_path, prices, books):
    """The other e2e tests lower memo_min_edge and min_prob_to_hold so that the modest offline
    rules trade; this one runs the shipped defaults (FirmConfig()) unchanged: every name that
    could be sized is reviewed by the chair, and the book stays within its hard limits."""
    from aifirm.agents.committee import memo_required
    from aifirm.config import FirmConfig

    cfg = make_config(str(tmp_path / "runs"))
    defaults = FirmConfig()
    assert cfg.process == defaults.process and cfg.portfolio == defaults.portfolio and cfg.risk == defaults.risk
    traded = 0
    for i in (DAY - 100, DAY, DAY + 100):
        dec = _decide(cfg, make_offline_llm(), prices, _day(prices, i), books(f"defaults{i}"))
        traded += len(dec.orders)
        assert not dec.held_by_outage
        for t, r in dec.committee.items():
            if memo_required(r.aggregate, cfg, held=False):
                assert r.memo is not None, t
            else:
                assert r.memo is None and r.edge_multiplier == 0.0, t
        w = dec.target_final.weights()
        assert all(0 < x <= cfg.portfolio.max_position + 1e-9 for x in w.values())
        assert all(dec.committee[t].memo is not None for t in w)    # nothing is held unreviewed
        assert dec.target_final.cash_weight >= cfg.portfolio.min_cash - 1e-9
        assert all(o.side == "buy" for o in dec.orders)
    assert traded > 0                                                # the defaults do trade on this market


# ----------------------------------------------------------------------------- regressions (decision session)


def _without_day(prices, day):
    keep = prices.close.index != pd.Timestamp(day)
    return PriceData(close=prices.close[keep], adj_close=prices.adj_close[keep], volume=prices.volume[keep],
                     open=prices.open[keep], raw_close=prices.raw_close[keep] if prices.raw_close is not None else None)


def test_gather_data_on_a_holiday_loads_everything_as_of_the_previous_session(tmp_path, prices):
    """Regression (#1): on a market holiday (Good Friday: NYSE closed, EDGAR open) the fact sheets
    were built for the calendar day, so filings accepted on the holiday reached a decision made
    at the previous session's close. Every loader now receives the decision session."""
    holiday, session = _day(prices, DAY), _day(prices, DAY - 1)
    seen: list[tuple[str, str]] = []

    def rec(name, fn):
        def wrapped(*args):
            seen.append((name, args[-1]))
            return fn(*args)
        return wrapped

    loaders = {k: rec(k, v) for k, v in LOADERS.items()}
    p, sheets = gather_data(_cfg(tmp_path), holiday, live=True, prices=_without_day(prices, holiday),
                            headlines_loader=rec("headlines", lambda t, d: []), **loaders)
    assert str(p.close.index[-1].date()) == session
    assert seen and {d for _, d in seen} == {session}
    assert {name for name, _ in seen} == {*LOADERS, "headlines"}
    assert all(fs.as_of == session for fs in sheets.values())


def test_cli_run_on_a_holiday_decides_the_previous_session(cli_env, prices, monkeypatch):
    config, runs = cli_env
    holiday, session = _day(prices, DAY), _day(prices, DAY - 1)
    gapped = _without_day(prices, holiday)
    asked: list[str] = []
    real_gather = gather_data

    def filings(t, d):
        asked.append(d)
        return fake_filings(t, d)

    def fake_gather(cfg, as_of, live, prices=None):
        return real_gather(cfg, as_of, False, prices=prices, **{**LOADERS, "filings_loader": filings})

    monkeypatch.setattr(cli, "gather_data", fake_gather)
    monkeypatch.setattr(cli, "load_prices", lambda tickers, start, end: _window(gapped, tickers, start, end))
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", holiday]) == 0
    assert asked and set(asked) == {session}
    sheets = [e["payload"] for e in Ledger(runs / "offline" / "ledger.jsonl").entries() if e["event"] == "factsheet"]
    assert sheets and all(e["as_of"] == session for e in sheets)


def test_cli_run_prints_kill_switch_and_freeze_messages(cli_env, prices, capsys):
    """Regression (#30): `aifirm run` printed only the pipeline warnings, so a kill switch or a
    drawdown freeze showed as "0 order(s)" on the console and was explained only in the report."""
    config, runs = cli_env
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "KILL").touch()
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert "risk: KILL SWITCH" in capsys.readouterr().out
    (runs / "KILL").unlink()
    book = runs / "offline"
    (book / "risk_state.json").write_text(json.dumps({"peak_nav": 125_000.0, "frozen": False, "last_nav": 100_000.0,
                                                      "last_as_of": _day(prices, DAY)}))
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 1)]) == 0
    out = capsys.readouterr().out
    assert "risk: DRAWDOWN FREEZE" in out and "0 order(s)" in out


# ----------------------------------------------------------------------------- regressions (live paper book between runs)


def _with_dividend(prices, ex_day, y=0.01):
    """A copy in which every name pays a cash dividend of ``y`` x its close on ``ex_day``
    (the adjusted close gains ``y`` on the raw close from that day on)."""
    adj = prices.adj_close.copy()
    adj.loc[adj.index >= pd.Timestamp(ex_day)] *= 1.0 + y
    return PriceData(close=prices.close, adj_close=adj, volume=prices.volume, open=prices.open,
                     raw_close=prices.raw_close)


def test_cli_credits_dividends_to_the_live_paper_book_once(cli_env, prices, monkeypatch):
    """Open request (money -> cli): the backtest credited dividends but `aifirm run` never did, so
    the live paper book earned the price return only. A dividend on the fill day belongs to the
    holder at the previous close (here: nobody), one on a later ex-date to the shares then held,
    and a later run never credits the same ex-date again."""
    config, runs = cli_env
    book = runs / "offline"
    y = 0.01
    paid = _with_dividend(_with_dividend(prices, _day(prices, DAY + 1), y), _day(prices, DAY + 3), y)
    monkeypatch.setattr(cli, "load_prices", lambda tickers, start, end: _window(paid, tickers, start, end))
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 2)]) == 0
    state = json.loads((book / "paper_broker.json").read_text())
    assert state["total_dividends"] == 0.0                     # ex-date DAY+1 is the fill day: not ours yet
    assert state["dividends_through"] == _day(prices, DAY + 2)
    shares = state["positions"]                                   # held at the close of DAY+2
    assert shares
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 4)]) == 0
    # the DAY+3 dividend goes to the shares held at the DAY+2 close (before any fill at the DAY+3 open)
    ex = _day(prices, DAY + 3)
    expected = sum(q * y * prices.close.loc[ex, t] for t, q in shares.items())
    state = json.loads((book / "paper_broker.json").read_text())
    assert state["total_dividends"] == pytest.approx(expected, rel=1e-9) and expected > 0
    events = [e["payload"] for e in Ledger(book / "ledger.jsonl").entries() if e["event"] == "dividends"]
    assert len(events) == 1 and {c["ex_date"] for c in events[0]["credited"]} == {ex}
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 5)]) == 0
    assert json.loads((book / "paper_broker.json").read_text())["total_dividends"] == pytest.approx(expected)


def test_cli_prev_nav_uses_the_shares_held_at_the_previous_close(cli_env, prices):
    """Open request (money -> cli): prev_nav was computed after the queued orders filled, i.e. today's
    shares marked at yesterday's close. With an old-format risk_state.json (no last_as_of, so no NAV
    path) that number fed the daily-loss halt. It is now the book's real NAV at the previous close."""
    config, runs = cli_env
    book = runs / "offline"
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert json.loads((book / "pending_orders.json").read_text())["orders"]
    state = json.loads((book / "risk_state.json").read_text())
    (book / "risk_state.json").write_text(json.dumps({"peak_nav": state["peak_nav"], "frozen": False,
                                                      "last_nav": state["last_nav"]}))   # old format
    assert cli.main(["run", "--config", str(config), "--offline", "--as-of", _day(prices, DAY + 1)]) == 0
    start = [e["payload"] for e in Ledger(book / "ledger.jsonl").entries() if e["event"] == "cycle_start"][-1]
    assert start["prev_nav"] == pytest.approx(100_000.0)       # all cash at the previous close
