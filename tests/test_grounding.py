"""Tests for the numeric grounding checker (P2). No network, hand-built fact sheets."""

import pytest

from aifirm.agents.grounding import check_view, extract_numbers
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.schemas import AnalystView, EvidenceItem, Scenario


def _fs() -> FactSheet:
    fs = FactSheet(ticker="NVDA", as_of="2026-06-30", name="N Corp", sector="Semiconductors")
    for f in [
        Fact("NVDA.ret_3m", "Return, 3 months (63 trading days)", 12.34, "%", "2026-06-30", "prices"),
        Fact("NVDA.price", "Last close", 123.45, "USD", "2026-06-30", "prices"),
        Fact("NVDA.revenue", "Revenue (TTM)", 130.5e9, "USD", "2026-04-30", "SEC"),
        Fact("NVDA.pe_ttm", "Price / earnings (TTM)", 45.67, "x", "2026-06-30", "computed"),
        Fact("NVDA.debt_to_equity", "Long-term debt / equity", 0.42, "ratio", "2026-04-30", "computed"),
        Fact("NVDA.vol_60d", "Realized volatility, 60 days (annualized)", 48.2, "%", "2026-06-30", "prices"),
        Fact("NVDA.shares_out", "Shares outstanding", 2.45e10, "count", "2026-04-30", "SEC"),
        Fact("MACRO.DGS10_chg_3m", "10-year Treasury yield: change over ~3 months", 0.45, "%pts", "2026-06-29", "FRED"),
    ]:
        fs.add(f)
    fs.events = ["2026-05-28: 8-K - results of operations (earnings release)",
                 "4 insider transaction reports (Form 4) filed in the window (direction not parsed)"]
    return fs


def _view(points: list[str], fact_ids: list[str] | None = None, prob: float = 0.56, excess: float = 1.2,
          risks: list[str] | None = None, insight: str = "") -> AnalystView:
    return AnalystView(
        ticker="NVDA", abstain=False, abstain_reason="", prob_outperform=prob, expected_excess_return_pct=excess,
        scenarios=[Scenario(name="bull", probability=0.3, excess_return_pct=6.0, description="Momentum continues."),
                   Scenario(name="base", probability=0.5, excess_return_pct=1.0, description="Drift."),
                   Scenario(name="bear", probability=0.2, excess_return_pct=-5.5, description="Reversal.")],
        evidence=[EvidenceItem(point=p, fact_ids=fact_ids or ["NVDA.ret_3m"], direction="positive") for p in points],
        key_risks=risks or [], what_would_change_my_mind="A price below 100 would change my mind.",
        private_insight=insight,
    )


def test_supported_numbers_in_every_display_format_pass():
    view = _view([
        "The 3-month return was 12.34%, rounded 12.3%.",          # raw and rounded
        "Revenue is $130.5B, i.e. $130.50 billion.",               # $B and spelled-out scale
        "P/E of 45.7x and debt/equity of 42%.",                   # multiple; ratio shown as a percent
        "Last close $123.45; 24.5 billion shares outstanding.",   # raw USD; scaled count
        "The 10-year yield rose 45 bps; 60-day volatility is 48.2%.",  # bps; window number from a label
    ])
    report = check_view(view, _fs())
    assert report.ok
    assert report.unsupported_numbers == []
    assert report.unknown_fact_ids == []
    assert report.checked_numbers == 11


def test_wrong_scale_is_unsupported():
    report = check_view(_view(["Revenue is $130.5M."]), _fs())
    assert report.unsupported_numbers == ["$130.5M"]


@pytest.mark.parametrize("n_bad, ok", [(2, True), (3, False)])
def test_more_than_two_unsupported_numbers_fail(n_bad, ok):
    bad = ["Margins are 77.7%.", "Backlog is $999B.", "Market share is 88.8%."][:n_bad]
    report = check_view(_view(["Return of 12.34%."] + bad), _fs())
    assert len(report.unsupported_numbers) == n_bad
    assert report.ok is ok


def test_repeated_unsupported_number_counts_once():
    report = check_view(_view(["Margins are 77.7%.", "Again, 77.7%.", "Still 77.7%."]), _fs())
    assert report.unsupported_numbers == ["77.7%"]
    assert report.ok


def test_unknown_fact_id_fails_even_without_numbers():
    report = check_view(_view(["Momentum is strong."], fact_ids=["NVDA.made_up", "[NVDA.ret_3m]"]), _fs())
    assert not report.ok
    assert report.unknown_fact_ids == ["NVDA.made_up"]   # the bracketed ID is normalized and known


def test_non_claims_are_ignored():
    view = _view(
        ["In 2025 the stock rose over 3 months and a 12-month window (see NVDA.ret_3m).",
         "The 8-K filed 2026-05-28 under Item 2.02 matters; the S&P 500 is the reference.",
         "My probability is 56% (0.56) versus the 0.5 base rate (50%); expected excess 1.2%."],
        risks=["The bull case of +6.0% may not happen; the bear case is -5.5%."],
    )
    report = check_view(view, _fs())
    assert report.checked_numbers == 0
    assert report.ok


def test_numbers_from_fact_sheet_text_are_supported():
    report = check_view(_view(["There were 4 insider transaction reports."]), _fs())
    assert report.checked_numbers == 1
    assert report.unsupported_numbers == []


def test_free_text_fields_beyond_evidence_are_checked():
    report = check_view(_view(["Return of 12.34%."], risks=["Debt of $77B."], insight="Gross margin 66.6%."), _fs())
    assert sorted(report.unsupported_numbers) == ["$77B", "66.6%"]


def test_check_against_a_slice_flags_ids_the_analyst_was_not_shown():
    fs = _fs()
    sliced = FactSheet(ticker="NVDA", as_of=fs.as_of, facts=[f for f in fs.facts if f.id == "NVDA.ret_3m"])
    report = check_view(_view(["Return of 12.34%."], fact_ids=["NVDA.pe_ttm"]), sliced)
    assert report.unknown_fact_ids == ["NVDA.pe_ttm"]
    assert not report.ok


def test_extract_numbers_reads_presentation():
    toks = {t.text: t for t in extract_numbers("Sales $1,234.5M, growth 12% and 45 bps; 3m return; 2s10s; 52-week high")}
    assert toks["$1,234.5M"].value == pytest.approx(1234.5) and toks["$1,234.5M"].scale == 1e6
    assert toks["12%"].percent
    assert toks["45 bps"].bps
    assert toks["3m"].time_unit                      # a bare lowercase "m" after a number is a month
    assert toks["52-week"].time_unit and toks["52-week"].value == 52
    assert not any(t.value in (2, 10) for t in toks.values())   # "2s10s" is a curve name, not numbers


def test_numbers_stated_in_the_prompt_are_supported():
    view = _view(["Over the next 21 trading days the 12.34% return may fade."])
    assert check_view(view, _fs()).unsupported_numbers == ["21 trading days"]
    assert check_view(view, _fs(), prompt_numbers=(21,)).unsupported_numbers == []
