"""Tests for the broad-universe screen: bundled universe lists, the screen config, universe
resolution and the deterministic deep-dive selection (aifirm/universes, aifirm/quant/screen.py).
No network."""

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tomllib
from pydantic import ValidationError
from synthetic_market import make_prices, perturb_after

from aifirm.config import FirmConfig, ScreenConfig, UniverseConfig
from aifirm.data.factsheet import FactSheet
from aifirm.data.prices import from_frames
from aifirm.pipeline import universe
from aifirm.quant.screen import HELD, TOP_BAND, screen_universe
from aifirm.schemas import AnalystView, ViewRecord
from aifirm.universes import PRESETS, SURVIVORSHIP_NOTE, file_tickers, parse_tickers, preset_tickers

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- universe lists


def test_parse_tickers_handles_comments_blanks_case_and_duplicates():
    text = "# header\naapl\n\n  MSFT  # trailing comment\nAAPL\nbrk-b\n"
    assert parse_tickers(text) == ["AAPL", "MSFT", "BRK-B"]


@pytest.mark.parametrize("bad", ["AAPL MSFT", "AAPL,MSFT"])
def test_parse_tickers_refuses_more_than_one_token_per_line(bad):
    with pytest.raises(ValueError, match="one ticker per line"):
        parse_tickers(f"AAPL\n{bad}\n")


def test_bundled_sp100_preset_is_about_a_hundred_unique_names_and_warns_of_survivorship():
    names = preset_tickers("sp100")
    assert 90 <= len(names) <= 105 and len(set(names)) == len(names)
    assert {"AAPL", "MSFT", "JPM", "XOM", "BRK-B"} <= set(names)
    assert "GOOG" not in names and "SPY" not in names            # one share class per company; no benchmark
    header = resources.files("aifirm.universes").joinpath("sp100.txt").read_text()
    assert "SURVIVORSHIP" in header and "biased UPWARD" in header
    assert preset_tickers("SP100") == names                       # case-insensitive
    with pytest.raises(KeyError, match="unknown universe preset"):
        preset_tickers("ftse")
    assert "survivorship" in SURVIVORSHIP_NOTE and "biased upward" in SURVIVORSHIP_NOTE


def test_presets_ship_as_package_data():
    """Regression guard: the .txt lists must be installed with the package, not only present
    in a source checkout."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "*.txt" in data["tool"]["setuptools"]["package-data"]["aifirm.universes"]
    for name in PRESETS:
        assert (ROOT / "aifirm" / "universes" / f"{name}.txt").is_file()


# ----------------------------------------------------------------------------- config


def test_screen_config_needs_a_source_and_a_known_preset():
    with pytest.raises(ValidationError, match="needs a universe source"):
        ScreenConfig(enabled=True)
    with pytest.raises(ValidationError, match="unknown"):
        ScreenConfig(enabled=True, preset="nasdaq9000")
    with pytest.raises(ValidationError):
        ScreenConfig(deep_dive_top_n=0)
    assert ScreenConfig(enabled=True, preset="sp100").deep_dive_top_n == 12
    assert ScreenConfig().enabled is False                                       # off by default


def test_a_screened_universe_may_leave_the_ticker_list_empty():
    UniverseConfig(tickers=[], screen=ScreenConfig(enabled=True, tickers=["AAA", "BBB", "CCC"]))
    with pytest.raises(ValidationError, match="at least one stock"):
        UniverseConfig(tickers=[])


def test_screen_file_is_relative_to_the_config_file_and_missing_files_fail_at_load(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "names.txt").write_text("# my list\nzzz\nAAPL\nSPY\n")
    (d / "firm.toml").write_text('[universe]\ntickers = ["MSFT"]\n[universe.screen]\nenabled = true\n'
                                 'file = "names.txt"\ntickers = ["yyy"]\n')
    monkeypatch.chdir(tmp_path)                                   # not the config's directory
    cfg = FirmConfig.load(d / "firm.toml")
    assert cfg.screen_file_path() == (d / "names.txt").resolve()
    assert file_tickers(cfg.screen_file_path()) == ["ZZZ", "AAPL", "SPY"]
    # universe.tickers first, then the file, then inline; benchmark excluded; de-duplicated
    assert universe(cfg) == ["MSFT", "ZZZ", "AAPL", "YYY"]
    (d / "bad.toml").write_text('[universe.screen]\nenabled = true\nfile = "missing.txt"\n')
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        FirmConfig.load(d / "bad.toml")


def test_universe_with_a_preset_and_without_the_screen():
    cfg = FirmConfig()
    assert universe(cfg) == [t.upper() for t in cfg.universe.tickers]            # screen off: unchanged
    cfg.universe.screen = ScreenConfig(enabled=True, preset="sp100")
    names = universe(cfg)
    assert names[:len(cfg.universe.tickers)] == cfg.universe.tickers and len(names) == len(set(names))
    assert set(names) == set(cfg.universe.tickers) | set(preset_tickers("sp100"))


# ----------------------------------------------------------------------------- screen_universe


def _view(t, p, abstain=False):
    return ViewRecord(agent="quant", model="deterministic", view=AnalystView(
        ticker=t, abstain=abstain, abstain_reason="no data" if abstain else "", prob_outperform=0.5 if abstain else p,
        expected_excess_return_pct=0.0, scenarios=[], evidence=[], key_risks=[], what_would_change_my_mind="",
        private_insight=""))


def _market(names, n=60, price=50.0, volume=1e6, end="2024-03-29"):
    idx = pd.bdate_range(end=end, periods=n)
    close = pd.DataFrame({t: np.full(n, price) for t in names}, index=idx)
    vol = pd.DataFrame({t: np.full(n, volume) for t in names}, index=idx)
    return from_frames(close, volume=vol)


AS_OF = "2024-03-29"


def _sheets(names):
    return {t: FactSheet(ticker=t, as_of=AS_OF) for t in names}


def _screen(probs, prices=None, held=(), top_n=3, min_dv=1e6, min_price=2.0, quant=None):
    names = sorted(probs)
    quant = quant if quant is not None else {t: _view(t, p) for t, p in probs.items()}
    return screen_universe(_sheets(names), quant, prices or _market(names), AS_OF, top_n=top_n,
                           min_dollar_volume=min_dv, min_price=min_price, held=held)


def test_ranks_by_quant_probability_and_deep_dives_only_the_top_band_above_one_half():
    probs = {"A": 0.53, "B": 0.55, "C": 0.51, "D": 0.49, "E": 0.55, "F": 0.50}
    res = _screen(probs, top_n=4)
    assert [e.ticker for e in res.entries] == ["B", "E", "A", "C", "F", "D"]      # ties broken by ticker
    assert [e.rank for e in res.entries] == [1, 2, 3, 4, 5, 6]
    assert res.deep_dive == ["B", "E", "A", "C"] and res.top_band == res.deep_dive
    # a name the quant expects to lag (or at a coin flip) is never a candidate, however small the universe
    small = _screen({"A": 0.53, "D": 0.49, "F": 0.50}, top_n=5)
    assert small.deep_dive == ["A"]
    assert res.n_universe == 6 and res.n_eligible == 6


def test_held_names_are_always_deep_dived_even_when_ranked_low_or_ineligible():
    probs = {"A": 0.56, "B": 0.55, "C": 0.54, "LOW": 0.45, "THIN": 0.56}
    prices = _market(["A", "B", "C", "LOW"]).close.join(_market(["THIN"], volume=10.0).close)
    vols = pd.concat([_market(["A", "B", "C", "LOW"]).volume, _market(["THIN"], volume=10.0).volume], axis=1)
    res = _screen(probs, prices=from_frames(prices, volume=vols), held=["LOW", "THIN"], top_n=2)
    assert res.deep_dive == ["A", "B", "LOW", "THIN"]
    by = {e.ticker: e for e in res.entries}
    assert by["LOW"].deep_dive_reason == HELD and by["LOW"].held and by["LOW"].eligible
    assert by["THIN"].deep_dive_reason == HELD and not by["THIN"].eligible and "dollar volume" in by["THIN"].reason
    assert by["A"].deep_dive_reason == TOP_BAND and not by["C"].deep_dive


def test_point_in_time_filters_price_liquidity_staleness_and_quant_abstention():
    names = ["OK", "PENNY", "ILLIQ", "STALE", "NOVIEW", "SHORT"]
    idx = pd.bdate_range(end=AS_OF, periods=40)
    close = pd.DataFrame({t: np.full(40, 50.0) for t in names}, index=idx)
    close["PENNY"] = 1.5
    close.loc[idx[-10]:, "STALE"] = np.nan                        # last print two weeks before the decision
    close.loc[:idx[-4], "SHORT"] = np.nan                         # only 3 days of history: liquidity unknown
    volume = pd.DataFrame(1e6, index=idx, columns=names)
    volume["ILLIQ"] = 100.0
    prices = from_frames(close, volume=volume)
    quant = {t: _view(t, 0.55, abstain=(t == "NOVIEW")) for t in names}
    res = screen_universe(_sheets(names), quant, prices, AS_OF, top_n=10, min_dollar_volume=1e6, min_price=2.0)
    why = {e.ticker: e.reason for e in res.entries}
    assert res.deep_dive == ["OK"] and why["OK"] == ""
    assert "below the minimum 2.00" in why["PENNY"]
    assert "dollar volume" in why["ILLIQ"]
    assert "stale price" in why["STALE"]
    assert "no view" in why["NOVIEW"]
    assert "liquidity unknown" in why["SHORT"]
    assert all(e.rank is None for e in res.entries if e.ticker != "OK")


def test_the_screen_never_sees_prices_after_the_decision_close():
    """Regression guard (P3): the liquidity and price filters cut the prices at as_of
    themselves, so passing the full history (as a backtest does) changes nothing."""
    full = make_prices()
    as_of = str(full.close.index[300].date())
    names = [c for c in full.close.columns if c != "SPY"]
    quant = {t: _view(t, 0.51 + 0.01 * i) for i, t in enumerate(names)}
    sheets = {t: FactSheet(ticker=t, as_of=as_of) for t in names}
    kw = dict(top_n=3, min_dollar_volume=5e7, min_price=2.0)
    a = screen_universe(sheets, quant, full, as_of, **kw)
    b = screen_universe(sheets, quant, perturb_after(full, as_of), as_of, **kw)
    c = screen_universe(sheets, quant, full.as_of(as_of), as_of, **kw)
    assert a.to_dict() == b.to_dict() == c.to_dict()


def test_screen_is_deterministic_whatever_the_input_order():
    probs = {f"T{i:02d}": 0.5 + ((i * 7) % 11) / 200 for i in range(15)}
    a = _screen(probs, top_n=5)
    shuffled = dict(sorted(probs.items(), reverse=True))
    b = _screen(shuffled, top_n=5)
    assert a.to_dict() == b.to_dict()
    assert set(a.to_dict()) >= {"ranking", "deep_dive", "n_universe", "n_eligible", "top_band"}
    with pytest.raises(ValueError):
        _screen(probs, top_n=0)


def test_min_price_uses_the_as_traded_level_not_a_later_split_adjusted_close():
    """Regression (look-ahead through splits): ``close`` is back-adjusted for every split up to
    the download date. A stock that later split 40:1 shows 1.18 on the decision date although
    it traded at 47.01, and the screen excluded it on a corporate action that had not happened
    yet. The rule now reads the as-traded level, like the risk engine's price fact."""
    idx = pd.bdate_range(end=AS_OF, periods=60)
    adjusted = pd.DataFrame({"NV": np.full(60, 47.01 / 40), "OK": np.full(60, 50.0)}, index=idx)
    raw = pd.DataFrame({"NV": np.full(60, 47.01), "OK": np.full(60, 50.0)}, index=idx)
    vol = pd.DataFrame({"NV": np.full(60, 4e7), "OK": np.full(60, 1e6)}, index=idx)
    splits = pd.DataFrame({"NV": [4.0, 10.0], "OK": [1.0, 1.0]},
                          index=pd.to_datetime(["2025-07-20", "2028-06-10"]))     # both after the decision
    prices = from_frames(adjusted, volume=vol, raw_close=raw, splits=splits)
    res = _screen({"NV": 0.6, "OK": 0.55}, prices=prices)
    nv = next(e for e in res.entries if e.ticker == "NV")
    assert nv.eligible and nv.last_price == pytest.approx(47.01) and "NV" in res.deep_dive
    assert nv.adv_usd == pytest.approx(47.01 / 40 * 4e7)              # dollar volume: close x volume, unchanged
    low = from_frames(adjusted, volume=vol)                            # no separate as-traded series: close is it
    res_low = _screen({"NV": 0.6, "OK": 0.55}, prices=low)
    assert not next(e for e in res_low.entries if e.ticker == "NV").eligible
