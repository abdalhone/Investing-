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
    return decide(cfg, llm, as_of, prices, sheets, current_weights=current, nav=NAV, prev_nav=kw.pop("prev_nav", None),
                  risk_state=kw.pop("state", RiskState(peak_nav=NAV)), forecast_store=book.store,
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


@pytest.fixture
def cli_env(tmp_path, prices, monkeypatch):
    """A config file, with prices and fact-sheet sources served from the synthetic market."""
    real_gather = gather_data

    def fake_gather(cfg, as_of, live, prices=None):
        return real_gather(cfg, as_of, False, prices=prices, **LOADERS)

    def fake_load_prices(tickers, start, end):
        assert set(tickers) <= {*synthetic.close.columns}
        return synthetic

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
