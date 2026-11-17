"""Point-in-time store of process lessons (P3, P7).

A lesson (written by :func:`aifirm.agents.reflection.write_lesson` after a forecast
resolves) can only be known once its forecast has resolved. Retrieval therefore filters
on the forecast's ``resolved_on`` date: a decision on date ``D`` sees only lessons with
``resolved_on <= D``. Without this filter a backtest would hand analysts hindsight from
the future, which is exactly the look-ahead bug that inflates published LLM-trading
results (see the "Profit Mirage" and Look-Ahead-Bench studies cited in ARCHITECTURE.md).

Other guards:

* **Most recent first, at most ``k``.** Prompts stay short and recent experience wins.
* **Short, de-duplicated strings.** Each lesson is rendered as one line (the rule plus
  when it applies) and capped in length; identical lines are shown once.
* **No stock names.** Lessons are retrieved for unrelated tickers; the ticker the lesson
  came from is masked as "STOCK" (the blind-mode alias) in case the writer named it anyway.
* **A damaged line is skipped, never repaired.** A missing lesson is harmless; a garbled
  one is not.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from ..schemas import Lesson
from .ledger import canonical_json

MAX_LESSON_CHARS = 300
MASK = "STOCK"                 # the same alias blind mode uses for the ticker
MIN_MASKED_TICKER_LEN = 2      # masking one-letter tickers ("A", "T") would mangle ordinary words

log = logging.getLogger(__name__)


def _day(d: str) -> str:
    return str(pd.Timestamp(d).date())


class LessonStore:
    """Append-only JSONL file of ``{"resolved_on": "YYYY-MM-DD", "lesson": {...}}`` lines."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: list[tuple[str, Lesson]] = []
        self._lines: set[str] = set()
        if self.path.exists():
            for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.strip():
                    self._load_line(line, lineno)

    def _load_line(self, line: str, lineno: int) -> None:
        try:
            raw = json.loads(line)
            entry = (_day(raw["resolved_on"]), Lesson.model_validate(raw["lesson"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            log.warning("%s line %d skipped: %s", self.path, lineno, exc)
            return
        self._entries.append(entry)
        self._lines.add(self._line(*entry))

    @staticmethod
    def _line(resolved_on: str, lesson: Lesson) -> str:
        return canonical_json({"resolved_on": resolved_on, "lesson": lesson.model_dump(mode="json")})

    def add(self, lesson: Lesson, resolved_on: str) -> None:
        """Store ``lesson``, usable from ``resolved_on`` (its forecast's resolution date)
        onward. Adding an identical lesson for the same date again is a no-op."""
        entry = (_day(resolved_on), lesson.model_copy(deep=True))
        line = self._line(*entry)
        with self._lock:
            if line in self._lines:
                return
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._entries.append(entry)
            self._lines.add(line)

    def retrieve(self, agent: str, as_of: str, k: int = 5) -> list[str]:
        """Up to ``k`` lessons for ``agent`` that were known on ``as_of``, most recent first."""
        if k <= 0:
            return []
        cutoff = _day(as_of)
        with self._lock:
            known = [(when, i, lesson) for i, (when, lesson) in enumerate(self._entries)
                     if lesson.agent == agent and when <= cutoff]
        out: list[str] = []
        for _, _, lesson in sorted(known, key=lambda x: (x[0], x[1]), reverse=True):
            text = _render(lesson)
            if text and text not in out:
                out.append(text)
                if len(out) == k:
                    break
        return out


def _render(lesson: Lesson) -> str:
    """One short line: the process rule, then when it applies, with the source ticker masked."""
    rule = " ".join(lesson.process_lesson.split())
    if not rule:
        return ""
    when = " ".join(lesson.applies_when.split())
    text = f"{rule} (applies when: {when})" if when else rule
    if len(lesson.ticker) >= MIN_MASKED_TICKER_LEN:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(lesson.ticker)}(?![A-Za-z0-9])", MASK, text)
    return text if len(text) <= MAX_LESSON_CHARS else text[:MAX_LESSON_CHARS - 3].rstrip() + "..."
