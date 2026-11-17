"""Append-only, hash-chained decision ledger (JSONL).

Why a hash chain (P10, P9): an honest track record needs an audit trail that cannot be
quietly rewritten after the fact. Every decision-cycle event (fact sheets, sealed views,
pooled views, memos, target portfolios, risk-engine overrides, orders, fills) is written
as one JSON line, and each line commits to the line before it:

    hash = SHA-256( prev_hash + canonical_json({seq, ts, event, payload, prev_hash}) )

so editing, deleting, inserting or reordering any line breaks the chain at that point and
:meth:`Ledger.verify` reports it. This is *tamper evidence*, not tamper proofing: someone
who rewrites the whole file from the edit onward can forge a valid chain, and a pure
suffix deletion leaves a valid shorter chain. Both are caught by comparing against a head
hash recorded elsewhere (``verify(anchor=...)``; the pipeline can put ``ledger.head`` in
each run report), and this process's own last-known head is always checked.

Design choices
--------------
* **Canonical JSON** (sorted keys, no whitespace, UTF-8, no NaN) makes the hash a pure
  function of the content, and every stored line must *be* its canonical form, so even a
  whitespace-only edit is detected.
* **Restart-safe**: a new ``Ledger`` on an existing file resumes the chain from the last
  line (read from the end of the file, so start-up cost does not grow with the ledger).
* **Fail closed**: an incomplete last line (crash mid-write), a last line whose own hash
  does not verify, or a file that changed underneath the writer raises
  :class:`LedgerError` instead of silently chaining onto damaged history. A pipeline that
  cannot write its audit trail must not trade.
* **Single writer per file.** A thread lock serialises appends within a process; before
  each append the on-disk tail is compared with the in-memory head, so a second writer
  or an external edit is detected rather than forked over.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

GENESIS_HASH = "0" * 64
ENTRY_KEYS = frozenset({"seq", "ts", "event", "payload", "prev_hash", "hash"})
_TAIL_CHUNK = 8192


class LedgerError(RuntimeError):
    """The ledger file is damaged or changed underneath this writer; a human must look."""


# --------------------------------------------------------------------------- JSON helpers


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON data (dict / list / str / int / float / bool / None).

    * pydantic models via ``model_dump(mode="json")``; dataclasses field by field (the
      result equals ``dataclasses.asdict`` without its deep copy, which fails on fields
      such as locks);
    * dates and timestamps as ISO strings; numpy scalars and arrays as Python values;
      enums by value; paths as strings; sets sorted by their canonical JSON so the
      output (and therefore the hash) is deterministic;
    * non-finite floats (NaN, +/-inf) become ``None``: strict JSON has no NaN, and a
      ledger other tools cannot parse is not an audit trail.

    Anything else raises ``TypeError``: writing an opaque ``repr`` would make the entry
    unverifiable and unreadable, so a caller bug should surface immediately.
    """
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(to_jsonable(k))
            if key in out:
                raise ValueError(f"two keys map to the same JSON key {key!r}")
            out[key] = to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v) for v in obj), key=canonical_json)
    if obj is pd.NaT:
        return None
    if isinstance(obj, (datetime, date)):          # includes pandas.Timestamp
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, Decimal):
        return to_jsonable(float(obj))
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot write {type(obj).__name__} to the ledger; convert it to plain data first")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, UTF-8 characters kept, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def entry_hash(body: Mapping[str, Any]) -> str:
    """SHA-256 of ``prev_hash`` followed by the canonical JSON of the entry body (all
    fields except ``hash``)."""
    return hashlib.sha256((body["prev_hash"] + canonical_json(dict(body))).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- entry checks


def _parse_line(raw: bytes) -> tuple[dict | None, str | None]:
    """Parse one stored line (without its newline). Returns (entry, problem)."""
    try:
        text = raw.decode("utf-8")
        entry = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"not valid JSON ({exc.__class__.__name__})"
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        return None, "not a ledger entry (wrong fields)"
    seq = entry["seq"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return None, "invalid seq"
    if not all(isinstance(entry[k], str) for k in ("ts", "event", "prev_hash", "hash")):
        return None, "invalid field types"
    if not isinstance(entry["payload"], dict):
        return None, "payload is not an object"
    try:
        canonical = canonical_json(entry)
    except ValueError:
        return None, "contains non-finite numbers"
    if canonical != text:
        return None, "not in canonical form (the line was edited)"
    return entry, None


def _self_hash_ok(entry: dict) -> bool:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return entry_hash(body) == entry["hash"]


def _read_last_line(path: Path) -> bytes | None:
    """The last line of ``path`` without its newline, reading backwards from the end.
    ``None`` for a missing or empty file. Raises if the file does not end in a newline
    (an interrupted write)."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as fh:
        end = fh.seek(0, os.SEEK_END)
        fh.seek(end - 1)
        if fh.read(1) != b"\n":
            raise LedgerError(f"{path}: the last line is incomplete (interrupted write?); run verify and repair by hand")
        data, pos = b"", end - 1
        while pos > 0:
            step = min(_TAIL_CHUNK, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
            nl = data.rfind(b"\n")
            if nl >= 0:
                return data[nl + 1:]
        return data


@dataclass(frozen=True)
class LedgerAudit:
    ok: bool
    checked: int                 # entries that verified before the first problem (all of them if ok)
    problem: str | None = None   # human-readable description of the first problem


# --------------------------------------------------------------------------- the ledger


class Ledger:
    """Append-only JSONL ledger with a SHA-256 hash chain.

    Each line is ``{seq, ts, event, payload, prev_hash, hash}``; ``seq`` counts from 0 and
    the first ``prev_hash`` is :data:`GENESIS_HASH`. ``clock`` returns the ``ts`` string
    (UTC ISO time by default); inject a fixed clock for reproducible tests.
    """

    def __init__(self, path: str | os.PathLike[str], clock: Callable[[], str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or _utc_now
        self._lock = threading.Lock()
        self._seq, self._head = self._tail_state()

    @property
    def head(self) -> str:
        """Hash of the last entry (``GENESIS_HASH`` when empty). Record it elsewhere and
        pass it to ``verify(anchor=...)`` later to detect truncation across restarts."""
        return self._head

    def __len__(self) -> int:
        return self._seq

    def _tail_state(self) -> tuple[int, str]:
        """(next seq, head hash) from the last line on disk, which must verify on its own."""
        raw = _read_last_line(self.path)
        if raw is None:
            return 0, GENESIS_HASH
        entry, problem = _parse_line(raw)
        if entry is None:
            raise LedgerError(f"{self.path}: last line is damaged ({problem}); run verify")
        if not _self_hash_ok(entry):
            raise LedgerError(f"{self.path}: last line fails its own hash check; run verify")
        return entry["seq"] + 1, entry["hash"]

    def append(self, event: str, payload: Mapping[str, Any]) -> str:
        """Append one event and return its hash. ``payload`` may contain pydantic models,
        dataclasses, numpy values and dates; see :func:`to_jsonable`."""
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")
        data = to_jsonable(payload)
        if not isinstance(data, dict):
            raise TypeError("payload must convert to a JSON object")
        with self._lock:
            if self._tail_state() != (self._seq, self._head):
                raise LedgerError(f"{self.path} changed on disk since this writer last saw it "
                                  "(another writer, truncation or an edit); run verify")
            body = {"seq": self._seq, "ts": self._clock(), "event": event, "payload": data,
                    "prev_hash": self._head}
            h = entry_hash(body)
            line = canonical_json({**body, "hash": h}) + "\n"
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._seq, self._head = self._seq + 1, h
        return h

    def entries(self) -> Iterator[dict]:
        """Parsed entries in file order, without verification (use :meth:`verify` for that)."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as fh:
            for raw in fh:
                yield json.loads(raw.decode("utf-8"))

    def audit(self, anchor: str | None = None) -> LedgerAudit:
        """Walk the whole chain and describe the first problem, if any.

        Checks every line is a canonical entry with the expected ``seq``, links to the
        previous hash and matches its own hash. Also requires that the head this object
        last saw and, if given, ``anchor`` (a head hash recorded earlier) are still in the
        chain, which catches deletion of the most recent entries.
        """
        required = {h for h in (self._head, anchor) if h and h != GENESIS_HASH}
        prev, n = GENESIS_HASH, 0
        if self.path.exists():
            with open(self.path, "rb") as fh:
                for raw in fh:
                    where = f"line {n + 1}"
                    if not raw.endswith(b"\n"):
                        return LedgerAudit(False, n, f"{where}: incomplete line (interrupted write?)")
                    entry, problem = _parse_line(raw[:-1])
                    if entry is None:
                        return LedgerAudit(False, n, f"{where}: {problem}")
                    if not _self_hash_ok(entry):
                        return LedgerAudit(False, n, f"{where}: hash mismatch (the entry was edited)")
                    if entry["prev_hash"] != prev or entry["seq"] != n:
                        return LedgerAudit(False, n, f"{where}: chain broken (an entry was deleted, inserted or reordered)")
                    prev = entry["hash"]
                    required.discard(prev)
                    n += 1
        if required:
            return LedgerAudit(False, n, "a known head hash is missing: the most recent entries were removed or rewritten")
        return LedgerAudit(True, n)

    def verify(self, anchor: str | None = None) -> tuple[bool, int]:
        """(ok, entries checked). See :meth:`audit` for what is checked and why."""
        result = self.audit(anchor)
        return result.ok, result.checked
