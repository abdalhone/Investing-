"""Tests for the re-identification probe (RESEARCH 7.2, 9.3 item 2). Offline; no API key."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic_market import make_prices

from aifirm.agents import blind_factsheet, make_offline_llm, run_reid_probe
from aifirm.agents import reid as reid_mod
from aifirm.agents.reid import (REID_ROLE, months_between, normalize_month, normalize_ticker, probe_payload,
                                wilson_interval)
from aifirm.data.factsheet import Fact, FactSheet, build_factsheet
from aifirm.journal.ledger import canonical_json, to_jsonable
from aifirm.llm.client import CallRecord, LLM, LLMCallFailed, OfflineLLM, Usage
from aifirm.schemas import ReidGuess

TICKERS = [f"T{k:02d}" for k in range(10)]


def toy_sheets(n: int, seed: int = 0) -> list[FactSheet]:
    """``n`` small fact sheets over 10 tickers and the 12 months of 2024 (cheap, deterministic)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    out = []
    for i in range(n):
        t = TICKERS[i % len(TICKERS)]
        d = str(days[int(rng.integers(len(days)))].date())
        fs = FactSheet(ticker=t, as_of=d, name=f"{t} Holdings Inc", sector="Widgets")
        for key, v in (("ret_3m", rng.normal(0, 10)), ("vol_60d", abs(rng.normal(30, 5))), ("rsi_14", rng.uniform(20, 80))):
            fs.add(Fact(f"{t}.{key}", key, float(v), "%", d, "prices"))
        fs.add(Fact("MACRO.VIXCLS", "VIX", float(rng.uniform(12, 30)), "index", d, "FRED VIXCLS"))
        out.append(fs)
    return out


class Memoriser(LLM):
    """A model that recognises every payload it has 'seen in training'."""

    def __init__(self, sheets: list[FactSheet]):
        self.usage = Usage(budget_usd=float("inf"))
        self.truth = {probe_payload(fs): (fs.ticker, fs.as_of[:7]) for fs in sheets}

    def structured(self, *, role, model, system, user, schema, effort="medium", max_tokens=16000):
        payload = user.split("<fact_sheet>\n", 1)[1].split("\n</fact_sheet>", 1)[0]
        t, m = self.truth[payload]
        return ReidGuess(ticker_guess=t.lower(), month_guess=f"{m}-15", confidence=0.9, clues="memory"), \
            CallRecord(role=role, model=model, prompt_hash="x")


# --------------------------------------------------------------------------- statistics helpers

def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_interval(0, 400)
    assert lo == 0.0 and hi == pytest.approx(0.00952, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_months_and_normalisation():
    assert months_between("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]
    assert months_between("2024-02-10", "2023-11-01")[0] == "2023-11"
    assert normalize_ticker(" $brk.b ") == "BRK-B" and normalize_ticker("BRK/B") == "BRK-B"
    assert normalize_month("2024-03") == normalize_month("2024-03-15") == normalize_month("March 2024") == "2024-03"
    assert normalize_month("") == normalize_month("no idea") == ""


# --------------------------------------------------------------------------- the probe

def test_random_guesser_matches_chance_and_passes_the_gate():
    sheets = toy_sheets(400)
    rep = run_reid_probe(make_offline_llm(), sheets)
    assert rep.n_probes == rep.n_answered == 400 and rep.n_failed == 0
    assert rep.candidates == TICKERS and len(rep.month_window) == 12
    assert rep.ticker.chance == pytest.approx(0.1) and rep.month.chance == pytest.approx(1 / 12)
    assert rep.joint.chance == pytest.approx(1 / 120)
    for h in (rep.ticker, rep.month, rep.joint):
        assert h.lo <= h.chance <= h.hi                      # random guessing is consistent with chance
    assert rep.gate_passed, rep.gate_notes
    assert "PASSED" in rep.summary()


def test_probe_is_deterministic():
    sheets = toy_sheets(60)
    a, b = (run_reid_probe(make_offline_llm(), sheets, min_probes=1) for _ in range(2))
    assert a.to_dict() == b.to_dict()
    c = run_reid_probe(make_offline_llm(), sheets, min_probes=1, max_parallel=4)
    assert [p.ticker_guess for p in c.probes] == [p.ticker_guess for p in a.probes]


def test_a_memorising_model_fails_the_gate():
    sheets = toy_sheets(400)
    rep = run_reid_probe(Memoriser(sheets), sheets)
    assert rep.ticker.rate == rep.month.rate == rep.joint.rate == 1.0
    assert not rep.gate_passed and any("above chance" in n for n in rep.gate_notes)


def test_too_few_probes_fail_the_gate():
    rep = run_reid_probe(make_offline_llm(), toy_sheets(50))
    assert not rep.gate_passed and any("only 50 answered probes" in n for n in rep.gate_notes)


def test_failed_calls_are_excluded_not_counted_as_misses():
    calls = {"n": 0}
    base = make_offline_llm().responders["reid_probe"]

    def flaky(role, system, user, schema):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise LLMCallFailed("reid_probe: model refused")
        return base(role, system, user, schema)

    rep = run_reid_probe(OfflineLLM({REID_ROLE: flaky}), toy_sheets(400))
    assert rep.n_failed == 100 and rep.n_answered == 300 and rep.joint.n == 300
    assert not rep.gate_passed and any("only 300 answered" in n for n in rep.gate_notes)
    assert all(p.error for p in rep.probes if not (p.ticker_guess or p.month_guess))


def test_payload_is_the_blind_sheet_and_never_names_the_company():
    prices = make_prices()
    days = [str(d.date()) for d in prices.close.index[300:420:20]]
    sheets = []
    for d in days:
        fs = build_factsheet("AAA", d, prices.as_of(d))
        fs.name, fs.sector = "Alpha Widgets Inc", "Alpha Widgets sector"
        fs.events = [f"{d}: 8-K - results of operations (earnings release)"]
        sheets.append(fs)
    llm = OfflineLLM({REID_ROLE: make_offline_llm().responders["reid_probe"]})
    seen: list[str] = []
    orig = llm.structured

    def spy(**kw):
        seen.append(kw["user"])
        return orig(**kw)

    llm.structured = spy                                       # type: ignore[method-assign]
    rep = run_reid_probe(llm, sheets, candidate_tickers=["BBB", "CCC"], min_probes=1)
    assert rep.candidates == ["AAA", "BBB", "CCC"] and not any(p.payload_leak for p in rep.probes)
    for d, user in zip(days, seen):
        assert "AAA" in user.split("\n", 1)[0]                 # only in the candidate list ...
        body = user.split("<fact_sheet>", 1)[1]
        assert "AAA" not in body and d not in body and "Alpha Widgets" not in body   # ... never in the payload
        assert "STOCK.ret_3m" in body and blind_factsheet(sheets[0]).render().splitlines()[0] in body


def test_identity_leak_in_the_payload_fails_the_gate(monkeypatch):
    monkeypatch.setattr(reid_mod, "blind_factsheet", lambda fs: fs)        # simulate a blinding bug
    rep = run_reid_probe(make_offline_llm(), toy_sheets(400))
    assert not rep.gate_passed and any("blinding bug" in n for n in rep.gate_notes)
    assert all("ticker" in p.payload_leak for p in rep.probes)


def test_role_slice_payload_and_validation():
    sheets = toy_sheets(20)
    rep = run_reid_probe(make_offline_llm(), sheets, role="technical", min_probes=1)
    assert rep.payload == "technical slice" and "MACRO.VIXCLS" not in probe_payload(sheets[0], "technical")
    with pytest.raises(ValueError):
        probe_payload(sheets[0], "astrologer")
    with pytest.raises(ValueError):
        run_reid_probe(make_offline_llm(), sheets, month_window=("2025-01", "2025-06"))
    with pytest.raises(ValueError):
        run_reid_probe(make_offline_llm(), [])


def test_report_is_ledger_safe():
    rep = run_reid_probe(make_offline_llm(), toy_sheets(30), min_probes=1)
    d = rep.to_dict()
    json.loads(canonical_json(to_jsonable(d)))
    assert d["month_window"] == ["2024-01", "2024-12"] and len(d["probes"]) == 30


def test_blind_factsheet_scrubs_the_sector_label():
    """Regression: the sector label was passed through unscrubbed in blind mode."""
    fs = FactSheet(ticker="AAA", as_of="2024-05-01", name="Alpha Widgets Inc", sector="AAA / Alpha Widgets division")
    assert "AAA" not in blind_factsheet(fs).sector and "Alpha Widgets" not in blind_factsheet(fs).sector


class Dater(LLM):
    """Dates every payload exactly (as frontier models date macro snapshots) but guesses the ticker."""

    def __init__(self, sheets: list[FactSheet], ticker_too: bool = False):
        self.usage = Usage(budget_usd=float("inf"))
        self.truth = {probe_payload(fs): (fs.ticker, fs.as_of[:7]) for fs in sheets}
        self.ticker_too = ticker_too
        self.k = 0

    def structured(self, *, role, model, system, user, schema, effort="medium", max_tokens=16000):
        payload = user.split("<fact_sheet>\n", 1)[1].split("\n</fact_sheet>", 1)[0]
        t, m = self.truth[payload]
        cands = user.split("CANDIDATE TICKERS: ", 1)[1].split("\n", 1)[0].split(", ")
        self.k += 1
        guess = t if self.ticker_too else cands[(self.k * 7) % len(cands)]
        return ReidGuess(ticker_guess=guess, month_guess=m, confidence=0.9, clues="macro level"), \
            CallRecord(role=role, model=model, prompt_hash="x")


def test_a_payload_that_reveals_the_month_fails_even_when_the_joint_rate_passes():
    """Regression: the gate read only the joint hit rate. With 100 candidate tickers, a model
    that dates every sheet but never names the ticker sits near joint chance and passed."""
    sheets = toy_sheets(1200)
    many = [f"C{k:03d}" for k in range(90)] + TICKERS                  # a closed-world list like sp100
    rep = run_reid_probe(Dater(sheets), sheets, candidate_tickers=many)
    assert rep.month.rate == 1.0 and rep.joint.excess <= 0.01 and rep.joint.hi - rep.joint.chance <= 0.02
    assert not rep.gate_passed
    assert any(n.startswith("month hit rate") and "reveals the month" in n for n in rep.gate_notes)
    assert not any(n.startswith("ticker hit rate") for n in rep.gate_notes)


def test_a_small_but_significant_marginal_leak_fails():
    """A month leak under the 5-point excess cap still fails when it is significantly (95%
    lower bound) more than 2 points above chance."""
    hr = reid_mod.HitRate.of(hits=int(0.125 * 4000), n=4000, chance=1 / 12)       # +4.2 points, very tight CI
    assert hr.excess < reid_mod.GATE_MAX_MARGINAL_EXCESS and hr.lo - hr.chance > reid_mod.GATE_MAX_MARGINAL_LOWER
    sheets = toy_sheets(400)
    rep = run_reid_probe(make_offline_llm(), sheets, max_marginal_lower=-1.0)      # any significance fails now
    assert not rep.gate_passed and any("significantly above chance" in n for n in rep.gate_notes)
