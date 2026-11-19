"""Bundled universe lists and ticker-file parsing for the broad-universe screen.

A *universe list* is a plain text file with one ticker per line; ``#`` starts a comment
(whole-line or trailing) and blank lines are ignored. Tickers are upper-cased and
de-duplicated in file order.

Bundled presets live next to this module as package data (``<name>.txt``, see
``pyproject.toml``). They are lists of CURRENT index members, so they carry
survivorship bias in backtests: names that were dropped, acquired or delisted during a
backtest window are missing and later joiners (often recent winners) are present, which
biases backtest returns upward. :data:`SURVIVORSHIP_NOTE` is the warning reports print.

This module does not import :mod:`aifirm.config` (the config validates preset names
against :data:`PRESETS`), so it has no dependency on the rest of the firm.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

PRESETS: tuple[str, ...] = ("sp100",)

SURVIVORSHIP_NOTE = ("the screened universe comes from a list of CURRENT index members, so names that were dropped, "
                     "acquired or delisted during the window are missing (survivorship bias): backtest returns on it "
                     "are biased upward")


def parse_tickers(text: str) -> list[str]:
    """Tickers from universe-list text: one per line, ``#`` comments, blank lines ignored,
    upper-cased, whitespace-stripped, de-duplicated in order. A line with more than one
    token is an error (a pasted CSV row must not silently become a strange ticker)."""
    out: list[str] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if len(line.split()) != 1 or "," in line:
            raise ValueError(f"line {n}: expected one ticker per line, got {raw.strip()!r}")
        t = line.upper()
        if t not in out:
            out.append(t)
    return out


def preset_tickers(name: str) -> list[str]:
    """The tickers of bundled preset ``name`` (case-insensitive). Raises ``KeyError`` for
    an unknown preset."""
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown universe preset {name!r}; bundled presets: {', '.join(PRESETS)}")
    return parse_tickers(resources.files(__name__).joinpath(f"{key}.txt").read_text(encoding="utf-8"))


def file_tickers(path: str | Path) -> list[str]:
    """The tickers listed in a universe file (``FileNotFoundError`` if it does not exist)."""
    return parse_tickers(Path(path).read_text(encoding="utf-8"))
