"""Scheduled NYSE half days and the point-in-time cutoff they move (P3). No network."""

import pandas as pd
import pytest

from aifirm.agents.news_reader import reader_user_prompt
from aifirm.data.calendar import is_early_close, session_close_time, thanksgiving
from aifirm.data.fundamentals import decision_cutoff
from aifirm.data.news import Headline, decision_close_utc, headlines_as_of


@pytest.mark.parametrize("day,early", [
    ("2026-11-27", True), ("2025-11-28", True), ("2024-11-29", True),      # day after Thanksgiving
    ("2024-12-24", True), ("2025-12-24", True), ("2026-12-24", True),      # Christmas Eve, Mon-Thu
    ("2021-12-24", False), ("2022-12-23", False),                          # Friday 24th is the holiday
    ("2025-07-03", True), ("2024-07-03", True), ("2023-07-03", True),
    ("2026-07-03", False), ("2020-07-03", False),                          # Friday 3rd is the holiday
    ("2026-11-26", False), ("2026-11-30", False), ("2026-06-15", False),
])
def test_scheduled_half_days(day, early):
    assert is_early_close(day) is early
    assert session_close_time(day) == ("13:00" if early else "16:00")


def test_thanksgiving_is_the_fourth_thursday():
    assert [str(thanksgiving(y)) for y in (2022, 2023, 2024, 2025, 2026)] == [
        "2022-11-24", "2023-11-23", "2024-11-28", "2025-11-27", "2026-11-26"]


def test_a_headline_after_a_13_00_close_is_not_read():
    """Regression: the news cutoff was always 16:00, so on the day after Thanksgiving a 14:30
    New York headline reached a decision (and the forecast was credited with the repricing)."""
    late = Headline("Alpha raises guidance", "Reuters", "2026-11-27T14:30:00-05:00")
    early = Headline("Alpha files 8-K", "Reuters", "2026-11-27T12:59:00-05:00")
    kept, dropped = headlines_as_of([late, early], "2026-11-27")
    assert kept == [early] and dropped == 1
    assert decision_close_utc("2026-11-27") == pd.Timestamp("2026-11-27 18:00", tz="UTC")
    assert decision_close_utc("2026-11-25") == pd.Timestamp("2026-11-25 21:00", tz="UTC")
    assert len(headlines_as_of([late], "2026-11-27", close_time="16:00")[0]) == 1      # explicit override
    assert "13:00 New York close" in reader_user_prompt("AAA", "2026-11-27", [(1, early)])
    assert "16:00 New York close" in reader_user_prompt("AAA", "2026-11-25", [(1, early)])


def test_filings_accepted_after_a_half_day_close_are_not_known_at_it():
    assert decision_cutoff("2026-11-27") == pd.Timestamp("2026-11-27 13:00")
    assert decision_cutoff("2026-11-25") == pd.Timestamp("2026-11-25 16:00")
