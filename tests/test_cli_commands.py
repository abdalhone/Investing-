"""Tests of the operator commands and flags added to the CLI, fully offline (synthetic market,
offline LLM rules, no network, no API key): ``screen``, ``compare``, ``reid-probe``, the
backtest's evaluation flags, the strategy override flags, and ``--shadow`` on ``score``,
``verify-ledger`` and ``unfreeze``."""

import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from synthetic_market import LOADERS, NAMES, make_prices

from aifirm import cli
from aifirm.backtest.engine import run_backtest
from aifirm.data.prices import PriceData
from aifirm.journal import Ledger
from aifirm.pipeline import gather_data

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


@pytest.fixture
def env(tmp_path, prices, monkeypatch):
    """A config with a screen over the synthetic names (two in universe.tickers, the rest inline)."""
    monkeypatch.setattr(cli, "gather_data", lambda cfg, as_of, live, prices=None:
                        gather_data(cfg, as_of, False, prices=prices, **LOADERS))
    monkeypatch.setattr(cli, "load_prices", lambda tickers, start, end: _window(prices, tickers, start, end))
    monkeypatch.setattr(cli, "last_completed_session", lambda now=None: _day(prices, TODAY))
    monkeypatch.setattr(cli, "run_backtest", lambda cfg, start, end, mode, llm=None, **kw:
                        run_backtest(cfg, start, end, mode, llm=llm, prices=prices, **LOADERS, **kw))
    for var in ANTHROPIC_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    proj = tmp_path / "proj"
    proj.mkdir()
    config = proj / "firm.toml"
    config.write_text(f'runs_dir = "runs"\n[universe]\ntickers = {json.dumps(NAMES[:2])}\nbenchmark = "SPY"\n'
                      f'[universe.screen]\nenabled = true\ntickers = {json.dumps(NAMES[2:])}\ndeep_dive_top_n = 2\n'
                      'min_dollar_volume = 1000000.0\n[process]\nmemo_min_edge = 0.005\n'
                      '[portfolio]\nmin_prob_to_hold = 0.5\n')
    runs = (proj / "runs").resolve()
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(config=str(config), proj=proj, runs=runs)


# ----------------------------------------------------------------------------- screen


def test_screen_prints_the_ranking_and_the_deep_dive_set_with_held_names(env, prices, capsys):
    env.runs.mkdir(parents=True)
    (env.runs / "paper_broker.json").write_text(json.dumps({"cash": 1000.0, "positions": {"AAA": 5.0}}))
    assert cli.main(["screen", "--config", env.config, "--as-of", _day(prices, DAY)]) == 0
    out = capsys.readouterr().out
    assert f"screen at the close of {_day(prices, DAY)}: 6 names with prices, 6 eligible" in out
    lines = [x for x in out.splitlines() if x.strip().split(" ")[0].isdigit()]
    assert len(lines) == 6 and [x.split()[0] for x in lines] == [str(i) for i in range(1, 7)]
    deep = out.split("deep-dive set (")[1].splitlines()[0]
    assert "AAA" in deep and "held" in out                   # a held name is always deep-dived
    assert not (env.runs / "ledger.jsonl").exists()          # a preview writes nothing unless asked


def test_screen_top_override_and_record(env, prices, capsys):
    assert cli.main(["screen", "--config", env.config, "--as-of", _day(prices, DAY), "--top", "1", "--rows", "2",
                     "--record"]) == 0
    out = capsys.readouterr().out
    assert "deep dive = top 1" in out and "... 4 more" in out
    entries = [e for e in Ledger(env.runs / "ledger.jsonl").entries()]
    assert [e["event"] for e in entries] == ["screen_preview"]
    assert entries[0]["payload"]["screen"]["top_n"] == 1 and entries[0]["payload"]["as_of"] == _day(prices, DAY)
    assert cli.main(["verify-ledger", "--config", env.config]) == 0
    assert cli.main(["screen", "--config", env.config, "--top", "0"]) == 1
    assert "--top must be at least 1" in capsys.readouterr().err


# ----------------------------------------------------------------------------- reid-probe


def test_reid_probe_refuses_without_credentials_unless_offline(env, prices, capsys):
    args = ["reid-probe", "--config", env.config, "--start", _day(prices, 300), "--end", _day(prices, 500), "--n", "12"]
    assert cli.main(args) == 1
    err = capsys.readouterr().err
    assert "credentials" in err and "add --offline to run the harness with a seeded random guesser" in err and "deterministic" not in err
    assert not env.runs.exists()

    assert cli.main([*args, "--offline"]) == 1                # 12 probes cannot pass a 400-probe gate
    out = capsys.readouterr().out
    assert "OFFLINE: a seeded random guesser" in out and "12 answered" in out and "gate: FAILED" in out
    reports = list((env.runs / "reid").glob("reid_*_offline.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text())
    assert saved["offline"] is True and saved["report"]["n_probes"] == 12
    assert len(saved["report"]["candidates"]) == len(NAMES)  # closed world: the whole universe
    ledger = list(Ledger(env.runs / "reid" / "ledger.jsonl").entries())
    assert ledger[-1]["event"] == "reid_probe" and ledger[-1]["payload"]["gate_passed"] is False
    assert "probes" not in ledger[-1]["payload"]
    assert cli.main([*args, "--offline", "--role", "technical"]) == 1
    assert "technical slice" in capsys.readouterr().out


def test_reid_sample_is_reproducible_and_only_uses_priced_names(prices):
    a = cli._reid_sample(NAMES, prices, _day(prices, 300), _day(prices, 500), 21, 15, seed=3)
    assert a == cli._reid_sample(NAMES, prices, _day(prices, 300), _day(prices, 500), 21, 15, seed=3)
    assert a != cli._reid_sample(NAMES, prices, _day(prices, 300), _day(prices, 500), 21, 15, seed=4)
    assert len(a) == 15 == len(set(a)) and a == sorted(a)
    dates = {str(d.date()) for d in prices.close.index[300:501:21]}
    assert {d for d, _ in a} <= dates
    everything = cli._reid_sample(NAMES, prices, _day(prices, 300), _day(prices, 500), 21, 10_000, seed=0)
    assert len(everything) == len(dates) * len(NAMES)


# ----------------------------------------------------------------------------- flags


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_evaluation_flags_never_change_the_fingerprint_and_strategy_flags_do(env):
    cfg = cli._load_config(env.config)
    assert cli._apply_flags(cfg, _ns()) is cfg
    ev = cli._apply_flags(cfg, _ns(random_books=50, seed=7, sector_matched_books=True, cost_multipliers="0,1,3",
                                   breakeven_cap=20.0, cutoff_buffer_days=31, ic_min_names=4, no_factor_proxies=True,
                                   no_shadow=True))
    assert ev.fingerprint() == cfg.fingerprint()
    e = ev.evaluation
    assert (e.n_random_books, e.random_seed, e.random_books_sector_matched, e.cost_multipliers, e.breakeven_cap,
            e.cutoff_buffer_days, e.ic_min_names, e.factor_proxies, e.shadow_book) == \
        (50, 7, True, [0.0, 1.0, 3.0], 20.0, 31, 4, False, False)
    # a non-default buffer also moves the pre_cutoff flags (which track record counts): a new trial
    buf = cli._apply_flags(cfg, _ns(cutoff_buffer_days=10))
    assert buf.evaluation.cutoff_buffer_days == 10 and buf.fingerprint() != cfg.fingerprint()
    for flags, check in ((dict(combination="pool"), lambda c: c.process.combination == "pool"),
                         (dict(max_tilt=0.02), lambda c: c.process.overlay_max_tilt == 0.02),
                         (dict(no_calibration=True), lambda c: not c.calibration.enabled),
                         (dict(no_news_reader=True), lambda c: not c.process.news_reader_enabled),
                         (dict(no_tier_gate=True), lambda c: not c.process.tier_gate_enabled),
                         (dict(rebalance_fraction=1.0), lambda c: c.portfolio.rebalance_fraction == 1.0),
                         (dict(deep_dive=5), lambda c: c.universe.screen.deep_dive_top_n == 5),
                         (dict(min_dollar_volume=5e6), lambda c: c.universe.screen.min_dollar_volume == 5e6)):
        new = cli._apply_flags(cfg, _ns(**flags))
        assert check(new) and new.fingerprint() != cfg.fingerprint(), flags
        assert new.runs_path() == cfg.runs_path()


def test_screen_flags_turn_the_screen_on_and_resolve_paths_from_the_shell(env, tmp_path):
    base = env.proj / "plain.toml"
    base.write_text('runs_dir = "runs"\n[universe]\ntickers = ["AAA"]\nbenchmark = "SPY"\n')
    cfg = cli._load_config(str(base))
    assert not cfg.universe.screen.enabled
    preset = cli._apply_flags(cfg, _ns(preset="sp100"))
    assert preset.universe.screen.enabled and preset.universe.screen.preset == "sp100"
    (tmp_path / "list.txt").write_text("BBB\nCCC\n")                      # relative to the shell (cwd = tmp_path)
    listed = cli._apply_flags(cfg, _ns(universe_file="list.txt"))
    assert listed.universe.screen.enabled and listed.screen_file_path() == (tmp_path / "list.txt").resolve()
    with pytest.raises(cli.CLIError, match="not found"):
        cli._apply_flags(cfg, _ns(universe_file="missing.txt"))
    with pytest.raises(cli.CLIError, match="needs a universe source"):
        cli._apply_flags(cfg, _ns(deep_dive=3))                           # a screen with nothing to screen
    with pytest.raises(cli.CLIError, match="unknown"):
        cli._apply_flags(cfg, _ns(preset="nosuch"))


def test_an_llm_backtest_without_credentials_fails_before_asking_to_spend(env, prices, monkeypatch, capsys):
    """Regression: '--mode llm' printed the budget and asked 'Continue?' before checking for an
    Anthropic key; answering yes then failed with 'No Anthropic credentials found'."""
    def no_prompt(*a, **k):
        raise AssertionError("asked to confirm a budget although no credentials exist")

    monkeypatch.setattr("builtins.input", no_prompt)
    assert cli.main(["backtest", "--config", env.config, "--start", _day(prices, 380), "--end", _day(prices, 460),
                     "--mode", "llm"]) == 1
    err = capsys.readouterr()
    assert "Anthropic credentials" in err.err and "--mode offline" in err.err and "LLM budget" not in err.out
    assert "--offline" not in err.err.replace("--mode offline", "")     # backtest has no --offline flag
    assert not (env.runs / "backtests").exists()


def test_invalid_flags_stop_before_anything_runs(env, prices, capsys):
    base = ["backtest", "--config", env.config, "--start", _day(prices, 300), "--end", _day(prices, 400)]
    for bad, message in ((["--max-tilt", "7"], "use 0.07 for 7%"), (["--cost-multipliers", "0,x"], "comma-separated"),
                         (["--cost-multipliers", "0,100"], "breakeven_cap"),
                         (["--random-books", "-1"], "n_random_books"),
                         (["--rebalance-fraction", "0"], "rebalance_fraction")):
        assert cli.main([*base, *bad]) == 1 and message in capsys.readouterr().err, bad
    assert not env.runs.exists()                                          # refused before anything ran


def test_backtest_flags_reach_the_run_and_the_console_states_the_six_lines(env, prices, monkeypatch, capsys):
    seen = {}
    real = cli.run_backtest

    def spy(cfg, start, end, mode, llm=None, **kw):
        seen["cfg"] = cfg
        return real(cfg, start, end, mode, llm=llm, **kw)

    monkeypatch.setattr(cli, "run_backtest", spy)
    assert cli.main(["backtest", "--config", env.config, "--start", _day(prices, 380), "--end", _day(prices, 460),
                     "--mode", "offline", "--random-books", "30", "--cost-multipliers", "0,1,4", "--cutoff-buffer-days",
                     "5", "--no-factor-proxies", "--seed", "11"]) == 0
    ev = seen["cfg"].evaluation
    assert (ev.n_random_books, ev.cost_multipliers, ev.cutoff_buffer_days, ev.factor_proxies, ev.random_seed) == \
        (30, [0.0, 1.0, 4.0], 5, False, 11)
    captured = capsys.readouterr()
    out = captured.out
    assert "1. Window and assets:" in out and "6. Benchmark and null:" in out and "**" not in out
    # a long run must not look like a hang: one progress line per decision (strategy and baseline) on stderr
    assert "offline strategy: decision 1 of " in captured.err and "quant-only baseline: decision 1 of " in captured.err
    assert "evaluating: signal IC" in captured.err and "decision 1 of" not in out
    assert "LLM value added vs the quant-only baseline, net of LLM cost:" in out
    assert "+5 days: contaminated through" in out and "SURVIVORSHIP" in out
    report = next((env.runs / "backtests").glob("*offline*/report.md")).read_text()
    assert "## Value added by the language models (vs the quant-only baseline, net of LLM cost)" in report
    # every synthetic decision precedes the cutoff: the random books (skill evidence) are not computed on
    # contaminated decisions, and the report says why
    assert "Each of 30 random books" not in report and "**Clean decisions only.**" in report


# ----------------------------------------------------------------------------- compare and --shadow


def test_compare_with_no_book_and_shadow_flags(env, prices, capsys):
    assert cli.main(["compare", "--config", env.config, "--offline"]) == 0
    assert "no book at" in capsys.readouterr().out
    assert cli.main(["run", "--config", env.config, "--offline", "--as-of", _day(prices, DAY)]) == 0
    assert cli.main(["run", "--config", env.config, "--offline", "--as-of", _day(prices, DAY + 1)]) == 0
    capsys.readouterr()
    assert cli.main(["compare", "--config", env.config, "--offline", "--as-of", _day(prices, DAY + 1), "--json"]) == 0
    out = capsys.readouterr().out
    assert "OFFLINE BOOK" in out and "Cumulative LLM spend since inception: $0.00 on the decisions reflected" in out
    assert "not yet measurable: 0 of " in out
    payload = json.loads(out[out.index("{"):])
    assert payload["nav_date"] == _day(prices, DAY + 1) and payload["live_cycles"] == 2
    assert cli.main(["score", "--config", env.config, "--offline", "--shadow", "--as-of", _day(prices, DAY + 1)]) == 0
    assert "forecasts:" in capsys.readouterr().out
    assert cli.main(["verify-ledger", "--config", env.config, "--offline", "--shadow"]) == 0

    shadow_state = env.runs / "offline" / "shadow" / "risk_state.json"
    state = json.loads(shadow_state.read_text())
    shadow_state.write_text(json.dumps({**state, "frozen": True, "frozen_at_nav": state["last_nav"]}))
    assert cli.main(["unfreeze", "--config", env.config, "--offline", "--shadow", "--reason", "reviewed", "--yes"]) == 0
    assert json.loads(shadow_state.read_text())["frozen"] is False
    assert not json.loads((env.runs / "offline" / "risk_state.json").read_text())["frozen"]
    last = list(Ledger(env.runs / "offline" / "shadow" / "ledger.jsonl").entries())[-1]
    assert last["event"] == "risk_unfrozen"
