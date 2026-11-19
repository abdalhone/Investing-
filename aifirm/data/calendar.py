"""New York Stock Exchange session close times, for the point-in-time cutoff (P3).

A decision is made at the close of the as-of session. On most sessions that is 16:00
America/New_York; on the exchange's scheduled half days it is 13:00. Anything published or
accepted after the real close (a headline, an SEC filing) cannot have been acted on at that
close, and the forecast is scored from that close's price, so the cutoff must move with it.

Scheduled early closes (NYSE rules in force since 2010):

* the day after Thanksgiving (the Friday after the fourth Thursday of November);
* Christmas Eve, 24 December, when it falls on Monday to Thursday (on a Friday it is the
  observed Christmas holiday; at weekends there is no session);
* 3 July, when it falls on Monday to Thursday (on a Friday it is the observed Independence
  Day holiday when 4 July is a Saturday; at weekends there is no session).

This module has no list of full holidays (a decision is only made on a session that has a
price), and ad-hoc closures are not scheduled half days. Pure date arithmetic: no network,
no state.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

REGULAR_CLOSE = "16:00"
EARLY_CLOSE = "13:00"


def _day(d: date | str | pd.Timestamp) -> date:
    return pd.Timestamp(d).date()


def thanksgiving(year: int) -> date:
    """The fourth Thursday of November."""
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


def is_early_close(day: date | str | pd.Timestamp) -> bool:
    """True if ``day`` is a scheduled NYSE half day (module docstring)."""
    d = _day(day)
    if d == thanksgiving(d.year) + timedelta(days=1):
        return True
    if (d.month, d.day) in ((12, 24), (7, 3)):
        return d.weekday() <= 3          # Monday..Thursday
    return False


def session_close_time(day: date | str | pd.Timestamp) -> str:
    """The New York close of the session on ``day``: "13:00" on a scheduled half day, else "16:00"."""
    return EARLY_CLOSE if is_early_close(day) else REGULAR_CLOSE


def session_close_hours(day: date | str | pd.Timestamp) -> int:
    """The close as whole hours after midnight New York time (13 or 16)."""
    return int(session_close_time(day).split(":", 1)[0])
