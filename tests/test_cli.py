"""Tests of the CLI's environment handling (aifirm/cli.py), fully offline.

Unlike the CLI tests in test_pipeline_e2e.py, the config here uses a *relative*
``runs_dir`` and every command runs from a different working directory; the fake price
loader honours ``end`` like Yahoo does, and the session clock is pinned to a synthetic
date. Regressions covered:

* the book, the kill switch and the trial ledger followed the shell's cwd, so a run from
  another directory ignored ``runs/KILL`` and started a fresh $100k book;
* ``--as-of`` today (before the close) or in the future decided on a partial intraday bar;
* the live book accepted decisions dated on or before its last decision (backdated fills,
  duplicate forecasts);
* there was no tested unfreeze procedure;
* config typos and percent-style limits were accepted silently.
"""

import builtins
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, make_prices

from aifirm import cli
from aifirm.backtest.engine import run_backtest
from aifirm.journal import ForecastStore, Ledger
from aifirm.pipeline import gather_data

DAY = 400          # 2024-07-15, a Monday
FRIDAY = DAY + 4   # 2024-07-19
TODAY = DAY + 40   # the pinned "last completed session"
ANTHROPIC_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                  "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE")


@pytest.fixture(scope="module")
def prices():
    return make_prices()


def _day(prices, i):
    return str(prices.close.index[i].date())


@pytest.fixture
def env(tmp_path, prices, monkeypatch):
    """proj/firm.toml with ``runs_dir = "runs"``; the shell sits in a sibling directory."""

    def fake_gather(cfg, as_of, live, prices=None):
        return gather_data(cfg, as_of, False, prices=prices, **LOADERS)

    def fake_load_prices(tickers, start, end):
        assert set(tickers) <= set(prices.close.columns)
        return prices.as_of(end)                 # like Yahoo: nothing after `end`

    clock = {"today": _day(prices, TODAY)}
    monkeypatch.setattr(cli, "gather_data", fake_gather)
    monkeypatch.setattr(cli, "load_prices", fake_load_prices)
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: clock["today"])
    for var in ANTHROPIC_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))

    proj = tmp_path / "proj"
    proj.mkdir()
    config = proj / "firm.toml"
    config.write_text(f'runs_dir = "runs"\n[universe]\ntickers = {json.dumps(NAMES)}\nbenchmark = "SPY"\n'
                      '[process]\nmemo_min_edge = 0.005\n[portfolio]\nmin_prob_to_hold = 0.5\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runs = (proj / "runs").resolve()
    return SimpleNamespace(config=str(config), proj=proj, runs=runs, book=runs / "offline",
                           elsewhere=elsewhere, clock=clock)


def _run(env, *extra):
    return cli.main(["run", "--config", env.config, "--offline", *extra])


def _events(book, event=None):
    path = book / "ledger.jsonl"
    if not path.exists():
        return []
    return [e for e in Ledger(path).entries() if event is None or e["event"] == event]


def _n_forecasts(book):
    with ForecastStore(book / "forecasts.sqlite") as store:
        return len(store.all())


def _snapshot(book):
    files = ("ledger.jsonl", "paper_broker.json", "risk_state.json", "pending_orders.json")
    return {f: (book / f).read_text() if (book / f).exists() else None for f in files}


# ----------------------------------------------------------------------------- runs_dir and the kill switch


def test_book_and_kill_switch_follow_the_config_file_not_the_cwd(env, prices, capsys):
    env.runs.mkdir(parents=True)
    (env.runs / "KILL").touch()
    assert _run(env, "--as-of", _day(prices, DAY)) == 0
    out = capsys.readouterr().out
    assert f"book: {env.book}" in out and f"kill switch: {env.runs / 'KILL'}" in out
    assert not (env.elsewhere / "runs").exists()              # before: a fresh book here, KILL unseen
    assert not (env.book / "pending_orders.json").exists()    # the kill switch held
    assert any("KILL SWITCH" in m for e in _events(env.book, "risk_engine") for m in e["payload"]["messages"])

    (env.runs / "KILL").unlink()
    assert _run(env, "--as-of", _day(prices, DAY + 1)) == 0
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"]
    assert not (env.elsewhere / "runs").exists()


def test_relative_config_path_from_another_directory_finds_the_same_book(env, prices, monkeypatch):
    monkeypatch.chdir(env.proj.parent)
    assert cli.main(["run", "--config", "proj/firm.toml", "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert (env.book / "ledger.jsonl").exists()
    assert cli.main(["verify-ledger", "--config", "proj/firm.toml", "--offline"]) == 0
    monkeypatch.chdir(env.elsewhere)
    assert cli.main(["verify-ledger", "--config", env.config, "--offline"]) == 0
    assert cli.main(["score", "--config", env.config, "--offline"]) == 0


def test_backtest_trial_ledger_follows_the_config_file(env, prices, monkeypatch):
    def offline_backtest(cfg, start, end, mode, llm=None):
        return run_backtest(cfg, start, end, mode, llm=llm, prices=prices, **LOADERS)

    monkeypatch.setattr(cli, "run_backtest", offline_backtest)
    args = ["backtest", "--config", env.config, "--start", _day(prices, DAY), "--end", _day(prices, DAY + 30),
            "--mode", "quant"]
    assert cli.main(args) == 0
    assert (env.runs / "trials.jsonl").exists() and list((env.runs / "backtests").glob("*/report.md"))
    assert not (env.elsewhere / "runs").exists()


# ----------------------------------------------------------------------------- completed sessions only


def test_run_refuses_a_session_that_is_not_complete(env, prices, capsys):
    today = env.clock["today"]
    tomorrow = _day(prices, TODAY + 1)
    for bad in (tomorrow, "2099-01-01"):
        assert _run(env, "--as-of", bad) == 1
        assert "after the last completed session" in capsys.readouterr().err
    assert _run(env, "--as-of", "2024/07/15") == 1
    assert "not a date" in capsys.readouterr().err
    assert not env.runs.exists()                               # refused before anything is written

    # the same rule for every command that reads a close
    assert cli.main(["score", "--config", env.config, "--offline", "--as-of", tomorrow]) == 1
    assert cli.main(["factsheet", "AAA", "--as-of", tomorrow]) == 1
    assert cli.main(["backtest", "--config", env.config, "--start", _day(prices, DAY), "--end", tomorrow]) == 1
    assert capsys.readouterr().err.count("after the last completed session") == 3

    # without --as-of the run decides on the last completed session
    assert _run(env) == 0
    assert list((env.book / "reports").glob(f"decision_{today}_*.md"))


def test_completed_session_with_the_real_clock(monkeypatch):
    """At 10:01 New York time on 2026-11-17 the 17th is still trading: refused (the review's case)."""
    real = cli.last_completed_session
    morning = pd.Timestamp("2026-11-17 10:01", tz=cli.MARKET_TZ).to_pydatetime()
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: real(morning))
    assert cli._completed_session(None) == "2026-11-16"
    assert cli._completed_session("2026-11-16") == "2026-11-16"
    assert cli._completed_session("2026-01-02") == "2026-01-02"
    for bad in ("2026-11-17", "2026-11-18"):
        with pytest.raises(cli.CLIError, match="after the last completed session"):
            cli._completed_session(bad)
    evening = pd.Timestamp("2026-11-17 17:00", tz=cli.MARKET_TZ).to_pydatetime()
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: real(evening))
    assert cli._completed_session("2026-11-17") == "2026-11-17"


# ----------------------------------------------------------------------------- no backdating, no re-deciding


def test_live_book_refuses_backdated_decisions(env, prices, capsys):
    d0, d1 = _day(prices, DAY), _day(prices, DAY + 1)
    assert _run(env, "--as-of", d0) == 0
    assert _run(env, "--as-of", d1) == 0                     # fills d0's orders at the d1 open
    assert _events(env.book, "fills")
    (env.book / "pending_orders.json").unlink(missing_ok=True)   # as after a cycle that queued nothing
    before, n_forecasts = _snapshot(env.book), _n_forecasts(env.book)
    capsys.readouterr()

    assert _run(env, "--as-of", _day(prices, DAY - 50)) == 1
    err = capsys.readouterr().err
    assert f"already decided at the close of {d1}" in err and "backdated" in err and "--force" in err
    assert _snapshot(env.book) == before and _n_forecasts(env.book) == n_forecasts


def test_same_session_cannot_be_decided_twice_unless_forced(env, prices, capsys):
    """The review's /robust/twice: with KILL present nothing is queued, and each re-run of the
    same date used to store another full set of forecasts for the same questions."""
    env.runs.mkdir(parents=True)
    (env.runs / "KILL").touch()
    fri, sat = _day(prices, FRIDAY), "2024-07-20"
    assert pd.Timestamp(sat) not in prices.close.index
    assert _run(env, "--as-of", fri) == 0
    n, before = _n_forecasts(env.book), _snapshot(env.book)
    assert n > 0
    capsys.readouterr()

    assert _run(env, "--as-of", fri) == 1
    assert "would repeat that decision" in capsys.readouterr().err
    assert _run(env, "--as-of", sat) == 1                    # the weekend maps back to the decided Friday
    assert f"already decided at the close of {fri}" in capsys.readouterr().err
    assert _snapshot(env.book) == before and _n_forecasts(env.book) == n

    assert _run(env, "--as-of", fri, "--force") == 0
    forced = _events(env.book, "decision_date_forced")
    assert len(forced) == 1 and forced[0]["payload"]["last_decision"] == fri
    out = capsys.readouterr().out
    assert "--force" in out
    # Open request (cli -> journal): the forced re-decision's forecasts used to be stored as well,
    # so every question of that session counted twice in the scorecard and the weights.
    assert _n_forecasts(env.book) == n
    assert "forecast(s) were not stored" in out
    assert _run(env, "--as-of", _day(prices, FRIDAY + 1)) == 0   # a later session is fine


def test_orders_waiting_for_their_open_cannot_be_forced_through(env, prices, capsys):
    d0 = _day(prices, DAY)
    assert _run(env, "--as-of", d0) == 0
    assert (env.book / "pending_orders.json").exists()
    capsys.readouterr()
    assert _run(env, "--as-of", d0, "--force") == 1
    assert "waiting for the next trading day's open" in capsys.readouterr().err


# ----------------------------------------------------------------------------- unfreeze


def _freeze(env, prices):
    """A book frozen by the risk engine: a 20% drawdown from a 125k peak at the first run."""
    env.book.mkdir(parents=True)
    (env.book / "risk_state.json").write_text(json.dumps({"peak_nav": 125_000.0, "frozen": True,
                                                          "last_nav": 100_000.0}))
    assert _run(env, "--as-of", _day(prices, DAY)) == 0
    state = json.loads((env.book / "risk_state.json").read_text())
    assert state["frozen"] and state["last_as_of"] == _day(prices, DAY)
    assert not (env.book / "pending_orders.json").exists()     # frozen: the active sleeve stays in cash
    return state


def _unfreeze(env, *extra):
    return cli.main(["unfreeze", "--config", env.config, "--offline", *extra])


def test_unfreeze_resets_the_freeze_and_the_high_water_mark_and_is_ledgered(env, prices):
    frozen = _freeze(env, prices)
    assert _unfreeze(env, "--reason", "reviewed the drawdown; data were fine", "--yes") == 0
    state = json.loads((env.book / "risk_state.json").read_text())
    assert state["frozen"] is False and state["frozen_at_nav"] is None
    assert state["peak_nav"] == pytest.approx(frozen["last_nav"])
    assert state["last_as_of"] == frozen["last_as_of"]
    entry = _events(env.book)[-1]
    assert entry["event"] == "risk_unfrozen"
    assert entry["payload"]["reason"] == "reviewed the drawdown; data were fine"
    assert entry["payload"]["before"]["frozen"] is True and entry["payload"]["after"]["frozen"] is False
    assert entry["payload"]["approved_by"] == "--yes"
    assert cli.main(["verify-ledger", "--config", env.config, "--offline"]) == 0

    assert _run(env, "--as-of", _day(prices, DAY + 1)) == 0     # trading again, not re-frozen
    assert json.loads((env.book / "pending_orders.json").read_text())["orders"]
    assert json.loads((env.book / "risk_state.json").read_text())["frozen"] is False


def test_unfreeze_is_a_deliberate_human_step(env, prices, monkeypatch, capsys):
    assert _unfreeze(env, "--reason", "x", "--yes") == 1
    assert "has not run yet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _unfreeze(env, "--yes")                                    # --reason is required

    _freeze(env, prices)
    before = _snapshot(env.book)
    assert _unfreeze(env, "--reason", "   ", "--yes") == 1
    assert "--reason is required" in capsys.readouterr().err
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    assert _unfreeze(env, "--reason", "looked at it") == 1          # not confirmed
    assert "stays frozen" in capsys.readouterr().out
    assert _snapshot(env.book) == before

    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    assert _unfreeze(env, "--reason", "looked at it") == 0
    assert _events(env.book)[-1]["payload"]["approved_by"] == "interactive"
    n = len(_events(env.book))
    assert _unfreeze(env, "--reason", "again", "--yes") == 0         # not frozen: nothing changes
    assert "not frozen" in capsys.readouterr().out and len(_events(env.book)) == n


def test_unfreeze_needs_a_recorded_nav(env):
    env.book.mkdir(parents=True)
    (env.book / "risk_state.json").write_text(json.dumps({"peak_nav": 125_000.0, "frozen": True}))
    assert _unfreeze(env, "--reason", "x", "--yes") == 1
    assert json.loads((env.book / "risk_state.json").read_text())["frozen"] is True


# ----------------------------------------------------------------------------- config and credentials


def test_invalid_config_stops_the_cli_before_anything_is_written(env, capsys):
    bad = env.proj / "bad.toml"
    bad.write_text('runs_dir = "runs"\n[portfolio]\nmax_positon = 0.02\n[risk]\ndrawdown_freeze = 15\n'
                   'restricted_list = ["NVDA"]\n')
    assert cli.main(["run", "--config", str(bad), "--offline"]) == 1
    err = capsys.readouterr().err
    assert "portfolio.max_positon: unknown setting" in err and "risk.restricted_list: unknown setting" in err
    assert "use 0.15 for 15%" in err
    assert not env.runs.exists()


def test_offline_runs_need_no_credentials_and_llm_runs_refuse_before_writing(env, prices, capsys):
    assert cli.credentials_problem() is not None
    assert cli.main(["run", "--config", env.config, "--as-of", _day(prices, DAY)]) == 1
    assert "credentials" in capsys.readouterr().err
    assert not env.runs.exists()
    assert _run(env, "--as-of", _day(prices, DAY)) == 0
