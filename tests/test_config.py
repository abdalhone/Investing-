"""Tests for configuration loading and validation (aifirm/config.py). No network.

Regressions covered: unknown keys and sections were silently ignored (a typo left the
default in force, e.g. an empty restricted list), percent-style risk limits (15 for 15%)
were accepted and switched the drawdown freeze and the daily-loss halt off, and a relative
``runs_dir`` depended on the shell's working directory instead of the config file.
"""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from aifirm.config import (
    CostConfig,
    EvaluationConfig,
    ExecutionConfig,
    FirmConfig,
    PortfolioConfig,
    ProcessConfig,
    RiskConfig,
    describe_config_error,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "firm.toml"
# FirmConfig().fingerprint(): a change of any default is a new strategy. (Changed deliberately when
# calibration.min_dates was added: the default calibration now waits for 20 resolved dates.)
DEFAULT_FINGERPRINT = "2e5f0d6108e79f5f"


def _write(tmp_path: Path, text: str, name: str = "firm.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _errors(text: str, tmp_path: Path) -> str:
    with pytest.raises(ValidationError) as info:
        FirmConfig.load(_write(tmp_path, text))
    return describe_config_error(info.value)


# ----------------------------------------------------------------------------- the example file


def test_example_config_loads_and_equals_the_defaults():
    cfg = FirmConfig.load(EXAMPLE)
    assert cfg.model_dump() == FirmConfig().model_dump()
    assert cfg.fingerprint() == FirmConfig().fingerprint()
    assert cfg.runs_path() == (EXAMPLE.parent / "runs").resolve()


# ----------------------------------------------------------------------------- unknown keys


def test_typos_in_keys_and_sections_are_errors_not_silent_defaults(tmp_path):
    """The reviewer's typo.toml: before, it loaded with max_position 0.1, drawdown_freeze 0.15,
    restricted [] and core_weight 0.0, so `aifirm run` would have traded the restricted name."""
    text = ('[portfolio]\nmax_positon = 0.02\n'
            '[risk]\ndrawdown_freez = 0.05\nrestricted_list = ["NVDA"]\n'
            '[portfollio]\ncore_weight = 0.5\n')
    msg = _errors(text, tmp_path)
    for loc in ("portfolio.max_positon", "risk.drawdown_freez", "risk.restricted_list", "portfollio"):
        assert f"{loc}: unknown setting" in msg
    with pytest.raises(ValidationError):
        FirmConfig.model_validate({"runs_dirr": "x"})        # top level too
    with pytest.raises(ValidationError):
        RiskConfig(restricted_list=["NVDA"])                  # and in code


# ----------------------------------------------------------------------------- ranges


@pytest.mark.parametrize("section,key,value,hint", [
    ("portfolio", "max_position", 10, "use 0.10 for 10%"),
    ("portfolio", "min_cash", 2, "use 0.02 for 2%"),
    ("portfolio", "core_weight", 50, "use 0.50 for 50%"),
    ("portfolio", "min_prob_to_hold", 52, "use 0.52 for 52%"),
    ("portfolio", "target_vol", 18, "use 0.18 for 18%"),
    ("portfolio", "max_turnover", 50, "use 0.50 for 50%"),
    ("risk", "drawdown_freeze", 15, "use 0.15 for 15%"),
    ("risk", "daily_loss_halt", 5, "use 0.05 for 5%"),
    ("risk", "max_adv_participation", 2.5, "use 0.025 for 2.5%"),
])
def test_percent_style_values_are_rejected_with_the_fraction_they_meant(tmp_path, section, key, value, hint):
    msg = _errors(f"[{section}]\n{key} = {value}\n", tmp_path)
    assert f"{section}.{key}: " in msg and "is out of range" in msg and hint in msg


def test_percent_style_risk_limits_can_no_longer_switch_the_risk_core_off():
    """Before: drawdown_freeze=15, drawdown_derisk=10, daily_loss_halt=5 were accepted, and
    a 50% drawdown with a 29% one-day loss fired neither the freeze nor the halt."""
    with pytest.raises(ValidationError) as info:
        FirmConfig(risk=RiskConfig(drawdown_freeze=15, drawdown_derisk=10, daily_loss_halt=5))
    msg = describe_config_error(info.value)
    assert msg.count("It looks like a percentage") == 3


@pytest.mark.parametrize("model,kwargs,needle", [
    (ProcessConfig, {"horizon_days": 0}, "horizon_days: 0 is out of range"),
    (ProcessConfig, {"horizon_days": 1000}, "horizon_days: 1000 is out of range"),
    (ProcessConfig, {"rebalance_every_days": 0}, "rebalance_every_days: 0 is out of range"),
    (ProcessConfig, {"trim_fraction": 0.5}, "trim_fraction: 0.5 is out of range"),
    (ProcessConfig, {"max_llm_cost_usd_per_run": 0.0}, "max_llm_cost_usd_per_run: 0.0 is out of range"),
    (PortfolioConfig, {"min_cash": 1.0}, "min_cash: 1.0 is out of range"),
    (PortfolioConfig, {"max_position": -0.1}, "max_position: -0.1 is out of range"),
    (PortfolioConfig, {"max_gross": 0.0}, "max_gross: 0.0 is out of range"),
    (PortfolioConfig, {"min_prob_to_hold": 0.4}, "min_prob_to_hold: 0.4 is out of range"),
    (PortfolioConfig, {"kelly_fraction": 1.5}, "kelly_fraction: 1.5 is out of range"),
    (RiskConfig, {"drawdown_derisk": 0.0}, "drawdown_derisk: 0.0 is out of range"),
    (RiskConfig, {"drawdown_freeze": 1.0}, "drawdown_freeze: 1.0 is out of range"),
    (RiskConfig, {"drawdown_derisk": 0.2, "drawdown_freeze": 0.1}, "must not exceed drawdown_freeze"),
    (RiskConfig, {"max_adv_participation": 0.0}, "max_adv_participation: 0.0 is out of range"),
    (RiskConfig, {"min_price": -1.0}, "min_price: -1.0 is out of range"),
    (RiskConfig, {"kill_switch_file": ""}, "kill_switch_file"),
    (RiskConfig, {"kill_switch_file": "../KILL"}, "kill_switch_file"),
    (CostConfig, {"commission_bps": -0.5}, "commission_bps: -0.5 is out of range"),
    (CostConfig, {"impact_coefficient": -1.0}, "impact_coefficient: -1.0 is out of range"),
    (ExecutionConfig, {"initial_cash": 0.0}, "initial_cash: 0.0 is out of range"),
    (ExecutionConfig, {"broker": "ib"}, "broker = 'ib' is unknown"),
])
def test_out_of_range_values_are_rejected(model, kwargs, needle):
    with pytest.raises(ValidationError) as info:
        model(**kwargs)
    assert needle in describe_config_error(info.value)


@pytest.mark.parametrize("payload,needle", [
    ({"models": {"samples_per_analyst": 0}}, "models.samples_per_analyst: 0 is out of range"),
    ({"models": {"analyst_models": []}}, "models: analyst_models must name at least one model"),
    ({"analysts": {"enabled": []}}, "analysts: enabled must list at least one analyst"),
    ({"universe": {"tickers": ["SPY"], "benchmark": "SPY"}}, "universe: tickers must list at least one stock"),
    ({"runs_dir": "  "}, "runs_dir must not be empty"),
])
def test_structural_problems_are_rejected(payload, needle):
    with pytest.raises(ValidationError) as info:
        FirmConfig.model_validate(payload)
    assert needle in describe_config_error(info.value)


def test_non_finite_numbers_are_rejected(tmp_path):
    msg = _errors("[risk]\ndrawdown_freeze = nan\n[portfolio]\nmax_position = inf\n", tmp_path)
    assert "risk.drawdown_freeze" in msg and "portfolio.max_position" in msg


def test_every_problem_is_reported_on_its_own_line(tmp_path):
    msg = _errors("[risk]\ndrawdown_freeze = 15\ndaily_loss_halt = 5\n[process]\nhorizon_days = 0\n", tmp_path)
    lines = msg.split("\n  ")
    assert [line.split(":")[0] for line in lines] == ["process.horizon_days", "risk.drawdown_freeze",
                                                       "risk.daily_loss_halt"]


def test_unknown_keys_and_bad_values_in_one_section_are_reported_together(tmp_path):
    msg = _errors('[risk]\ndrawdown_freeze = 15\nrestricted_list = ["NVDA"]\n', tmp_path)
    assert "risk.restricted_list: unknown setting" in msg and "risk.drawdown_freeze: 15.0 is out of range" in msg


def test_boundary_values_that_switch_a_rule_off_are_still_allowed():
    """Values tests and users rely on to isolate rules (no turnover cap, no vol target,
    full Kelly, a fully passive book) stay valid."""
    pc = PortfolioConfig(max_turnover=10.0, target_vol=5.0, kelly_fraction=1.0, core_weight=1.0,
                         min_cash=0.0, no_trade_band=0.0, max_sector=1.0, max_gross=1.6, long_only=False)
    assert pc.max_turnover == 10.0
    RiskConfig(drawdown_derisk=0.15, drawdown_freeze=0.15, max_adv_participation=1.0, min_price=0.0)
    ProcessConfig(horizon_days=1, rebalance_every_days=252, trim_fraction=0.0, memo_min_edge=0.0)


# ----------------------------------------------------------------------------- runs_dir


def test_relative_runs_dir_is_relative_to_the_config_file_not_the_cwd(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg = FirmConfig.load(_write(proj, 'runs_dir = "runs"\n'))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cfg.runs_dir == "runs"                       # stored as written (model_dump is unchanged)
    assert cfg.runs_path() == (proj / "runs").resolve()
    assert cfg.model_copy(deep=True).runs_path() == (proj / "runs").resolve()   # copies keep the anchor
    nested = FirmConfig.load(_write(proj, 'runs_dir = "../data/book"\n', "nested.toml"))
    assert nested.runs_path() == (tmp_path / "data" / "book").resolve()


def test_absolute_and_built_in_runs_dirs(tmp_path, monkeypatch):
    absolute = tmp_path / "abs-runs"
    cfg = FirmConfig.load(_write(tmp_path, f"runs_dir = {json.dumps(str(absolute))}\n"))
    assert cfg.runs_path() == absolute.resolve()
    monkeypatch.chdir(tmp_path)
    assert FirmConfig().runs_path() == (tmp_path / "runs").resolve()   # no file: the cwd, as documented


def test_fingerprint_ignores_where_results_are_written(tmp_path):
    cfg = FirmConfig.load(_write(tmp_path, 'runs_dir = "runs"\n'))
    resolved = cfg.model_copy(update={"runs_dir": str(cfg.runs_path())})
    assert resolved.fingerprint() == cfg.fingerprint() == FirmConfig(runs_dir="/elsewhere").fingerprint()
    assert FirmConfig(portfolio=PortfolioConfig(max_position=0.05)).fingerprint() != cfg.fingerprint()


def test_toml_syntax_errors_still_surface_as_value_errors(tmp_path):
    with pytest.raises(ValueError):
        FirmConfig.load(_write(tmp_path, "[risk\n"))
    assert math.isfinite(FirmConfig().risk.drawdown_freeze)


# ----------------------------------------------------------------------------- SEC filer pins


def test_universe_ciks_pin_sec_filers_and_reach_the_default_loaders(tmp_path, monkeypatch):
    """Open request (data -> cli/config): with extra="forbid", a CIK pin could only be set through
    the AIFIRM_CIK_OVERRIDES environment variable. ``universe.ciks`` now pins a filer in the
    config, and gather_data's default SEC loaders use it."""
    cfg = FirmConfig.load(_write(tmp_path, '[universe]\ntickers = ["xom", "AAA"]\n[universe.ciks]\nxom = 34088\n'
                                           'AAA = "0000000042"\n'))
    assert cfg.universe.cik_pins() == {"XOM": "0000034088", "AAA": "0000000042"}
    assert cfg.fingerprint() != FirmConfig().fingerprint()
    assert FirmConfig().model_copy(update={}).fingerprint() == FirmConfig().fingerprint()
    msg = _errors('[universe.ciks]\nXOM = "abc"\n', tmp_path)
    assert "universe" in msg and "ciks must map tickers to SEC CIK numbers" in msg

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from synthetic_market import make_prices

    from aifirm import pipeline
    seen: dict[str, list] = {"fund": [], "filings": []}
    monkeypatch.setattr(pipeline, "load_fundamentals",
                        lambda t, d, cik=None: seen["fund"].append((t, cik)))
    monkeypatch.setattr(pipeline, "recent_filings",
                        lambda t, d, days, cik=None: seen["filings"].append((t, cik)) or [])
    cfg.universe.tickers = ["AAA", "BBB"]
    pipeline.gather_data(cfg, "2024-06-03", live=False, prices=make_prices(), macro_loader=lambda d: [])
    assert sorted(seen["fund"]) == sorted(seen["filings"]) == [("AAA", "0000000042"), ("BBB", None)]


# ----------------------------------------------------------------------------- evaluation section


@pytest.mark.parametrize("kwargs,needle", [
    ({"n_random_books": -1}, "n_random_books: -1 is out of range"),
    ({"ic_min_names": 2}, "ic_min_names: 2 is out of range"),
    ({"cutoff_buffer_days": -5}, "cutoff_buffer_days: -5 is out of range"),
    ({"breakeven_cap": 1.0}, "breakeven_cap: 1.0 is out of range"),
    ({"random_seed": -1}, "random_seed: -1 is out of range"),
    ({"cost_multipliers": []}, "cost_multipliers must list at least one multiplier"),
    ({"cost_multipliers": [0.0, -1.0]}, "cost_multipliers [-1.0] must be finite, >= 0"),
    ({"cost_multipliers": [1.0, 80.0]}, "<= breakeven_cap (50)"),
])
def test_evaluation_settings_are_validated(kwargs, needle):
    with pytest.raises(ValidationError) as info:
        EvaluationConfig(**kwargs)
    assert needle in describe_config_error(info.value)


def test_evaluation_section_loads_from_toml_and_rejects_typos(tmp_path):
    cfg = FirmConfig.load(_write(tmp_path, "[evaluation]\nn_random_books = 200\ncost_multipliers = [0.0, 1.0, 3.0]\n"
                                           "cutoff_buffer_days = 60\nfactor_proxies = false\n"))
    assert cfg.evaluation.n_random_books == 200 and cfg.evaluation.cost_multipliers == [0.0, 1.0, 3.0]
    assert cfg.evaluation.cutoff_buffer_days == 60 and cfg.evaluation.factor_proxies is False
    assert "evaluation.n_random_bookz: unknown setting" in _errors("[evaluation]\nn_random_bookz = 5\n", tmp_path)
    defaults = EvaluationConfig()
    assert (defaults.n_random_books, defaults.cost_multipliers, defaults.cutoff_buffer_days) == (1000, [0.0, 1.0, 2.0],
                                                                                                31)


def test_evaluation_settings_are_not_part_of_the_fingerprint():
    """Regression guard: re-scoring a strategy with another null, cost grid or cutoff buffer
    is not a new trial, and adding the [evaluation] section must not change the fingerprint
    of any existing configuration (that would silently reset every trial ledger count)."""
    changed = FirmConfig(evaluation=EvaluationConfig(n_random_books=7, random_seed=1, cost_multipliers=[3.0],
                                                     factor_proxies=False))
    assert changed.fingerprint() == FirmConfig().fingerprint()
    # ... except a non-default cutoff buffer: it also decides which LLM forecasts are flagged
    # pre_cutoff, i.e. which track record believability, calibration and the overlay bound use
    buffered = FirmConfig(evaluation=EvaluationConfig(cutoff_buffer_days=0))
    assert buffered.fingerprint() != FirmConfig().fingerprint()
    assert FirmConfig(evaluation=EvaluationConfig(cutoff_buffer_days=31)).fingerprint() == FirmConfig().fingerprint()
    # Pinned: evaluation settings never move it. (The value changed once, deliberately, when the
    # research defaults, the screen, the overlay and calibration settings were added: a new strategy.)
    assert FirmConfig().fingerprint() == DEFAULT_FINGERPRINT


def _screen_config(directory, tickers, file_value="universe.txt"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "universe.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")
    (directory / "firm.toml").write_text(
        f'[universe]\ntickers = []\n[universe.screen]\nenabled = true\nfile = "{file_value}"\n', encoding="utf-8")
    return FirmConfig.load(directory / "firm.toml")


def test_fingerprint_covers_the_screened_universe_contents(tmp_path):
    """Regression: the fingerprint hashed only the path ``universe.screen.file``, so editing the
    file and re-running counted as the same trial and the deflated Sharpe never saw the
    universe selection; the same file given by an absolute path counted as a new trial."""
    a = _screen_config(tmp_path / "a", ["AAPL", "MSFT"])
    b = _screen_config(tmp_path / "b", ["NVDA", "XOM", "KO", "JPM"])
    assert a.fingerprint() != b.fingerprint()
    assert a.universe_digest() != b.universe_digest()
    same = _screen_config(tmp_path / "c", ["AAPL", "MSFT"])                     # another directory, same names
    absolute = _screen_config(tmp_path / "d", ["AAPL", "MSFT"], file_value=str(tmp_path / "a" / "universe.txt"))
    assert same.fingerprint() == a.fingerprint() == absolute.fingerprint()
    (tmp_path / "a" / "universe.txt").write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")   # edited in place
    assert a.fingerprint() != same.fingerprint()
    preset = FirmConfig()
    preset.universe.tickers = []
    preset.universe.screen.enabled, preset.universe.screen.preset = True, "sp100"
    inline = FirmConfig()
    inline.universe.tickers = []
    inline.universe.screen.enabled, inline.universe.screen.tickers = True, preset.investable_tickers()
    assert preset.fingerprint() == inline.fingerprint()                         # same names, same trial


def test_trial_ledger_rows_record_the_universe_digest(tmp_path):
    from aifirm.backtest.engine import _count_trials
    a = _screen_config(tmp_path / "a", ["AAPL", "MSFT"])
    assert a.universe_digest() == FirmConfig.load(tmp_path / "a" / "firm.toml").universe_digest()
    (tmp_path / "a" / "universe.txt").unlink()
    assert a.universe_digest() == "unreadable" and a.fingerprint()              # never raises
    assert _count_trials(tmp_path / "none.jsonl", "f", "quant", "2024-01-01", "2024-06-30") == 1
