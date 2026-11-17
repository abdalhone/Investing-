"""Tests for the hash-chained ledger (journal/ledger.py). No network."""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aifirm.data.factsheet import Fact
from aifirm.journal.ledger import (GENESIS_HASH, Ledger, LedgerError, canonical_json, entry_hash,
                                   to_jsonable)
from aifirm.schemas import AnalystView, ViewRecord


def _clock():
    return "2026-01-30T21:00:00+00:00"


def _ledger(path, n=5):
    led = Ledger(path, clock=_clock)
    for i in range(n):
        led.append("event", {"i": i, "note": f"entry {i}"})
    return led


def _lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines(keepends=True)


def _write(path, lines):
    Path(path).write_text("".join(lines), encoding="utf-8")


def _view():
    v = AnalystView(ticker="AAA", abstain=False, abstain_reason="", prob_outperform=0.6,
                    expected_excess_return_pct=1.5, scenarios=[], evidence=[], key_risks=[],
                    what_would_change_my_mind="", private_insight="")
    return ViewRecord(agent="technical", model="claude-opus-5-5", view=v)


# ----------------------------------------------------------------------------- chain structure


def test_entries_have_contract_fields_and_link_to_previous_hash(tmp_path):
    led = _ledger(tmp_path / "ledger.jsonl", n=3)
    entries = list(led.entries())
    assert [e["seq"] for e in entries] == [0, 1, 2]
    assert all(set(e) == {"seq", "ts", "event", "payload", "prev_hash", "hash"} for e in entries)
    assert entries[0]["prev_hash"] == GENESIS_HASH
    assert entries[1]["prev_hash"] == entries[0]["hash"]
    assert entries[2]["prev_hash"] == entries[1]["hash"]
    assert led.head == entries[-1]["hash"] and len(led) == 3


def test_append_returns_hash_of_prev_hash_plus_canonical_body(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", clock=_clock)
    h = led.append("view", {"b": 2, "a": 1})
    body = {"seq": 0, "ts": _clock(), "event": "view", "payload": {"a": 1, "b": 2}, "prev_hash": GENESIS_HASH}
    expected = hashlib.sha256((GENESIS_HASH + canonical_json(body)).encode()).hexdigest()
    assert h == expected == entry_hash(body)


def test_same_events_and_clock_give_identical_hashes(tmp_path):
    a, b = _ledger(tmp_path / "a.jsonl"), _ledger(tmp_path / "b.jsonl")
    assert a.head == b.head
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_clean_ledger_verifies_and_counts_entries(tmp_path):
    led = _ledger(tmp_path / "l.jsonl", n=7)
    assert led.verify() == (True, 7)


def test_empty_or_missing_ledger_verifies(tmp_path):
    led = Ledger(tmp_path / "sub" / "new.jsonl")
    assert led.verify() == (True, 0)
    assert led.head == GENESIS_HASH


def test_rejects_bad_event_and_non_object_payload(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(ValueError):
        led.append("", {})
    with pytest.raises(TypeError):
        led.append("x", [1, 2])  # type: ignore[arg-type]
    assert led.verify() == (True, 0)


# ----------------------------------------------------------------------------- restarts


def test_restart_resumes_the_chain_from_the_last_line(tmp_path):
    path = tmp_path / "l.jsonl"
    first = _ledger(path, n=3)
    second = Ledger(path, clock=_clock)
    assert len(second) == 3 and second.head == first.head
    second.append("after_restart", {"x": 1})
    entries = list(second.entries())
    assert entries[3]["seq"] == 3 and entries[3]["prev_hash"] == entries[2]["hash"]
    assert Ledger(path).verify() == (True, 4)


def test_resume_reads_long_last_line(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path, clock=_clock)
    led.append("small", {})
    led.append("big", {"blob": "x" * 50_000})          # longer than one tail-read chunk
    assert Ledger(path).head == led.head


def test_interrupted_write_fails_closed_on_restart(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=2)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"seq":2,"ts":"x","ev')                # crash mid-write: no newline
    with pytest.raises(LedgerError, match="incomplete"):
        Ledger(path)


def test_damaged_last_line_fails_closed_on_restart(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=2)
    lines = _lines(path)
    lines[-1] = lines[-1].replace('"i":1', '"i":9')
    _write(path, lines)
    with pytest.raises(LedgerError):
        Ledger(path)


def test_append_refuses_when_file_changed_underneath(tmp_path):
    path = tmp_path / "l.jsonl"
    led = _ledger(path, n=3)
    _write(path, _lines(path)[:-1])                     # someone truncates the tail
    with pytest.raises(LedgerError, match="changed on disk"):
        led.append("next", {})


# ----------------------------------------------------------------------------- tamper detection


def test_edited_payload_is_detected_at_that_line(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=5)
    lines = _lines(path)
    lines[2] = lines[2].replace('"i":2', '"i":3')
    _write(path, lines)
    led = Ledger(path)                                   # the intact last line still resumes
    assert led.verify() == (False, 2)
    assert "line 3" in led.audit().problem


def test_edited_and_rehashed_entry_breaks_the_next_link(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=5)
    lines = _lines(path)
    e = json.loads(lines[1])
    e["payload"]["note"] = "rewritten history"
    body = {k: v for k, v in e.items() if k != "hash"}
    e["hash"] = entry_hash(body)                         # forger fixes the entry's own hash
    lines[1] = canonical_json(e) + "\n"
    _write(path, lines)
    ok, checked = Ledger(path).verify()
    assert not ok and checked == 2                       # entry 2 no longer links to entry 1


def test_deleted_middle_line_is_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=5)
    lines = _lines(path)
    del lines[2]
    _write(path, lines)
    led = Ledger(path)
    assert led.verify() == (False, 2)
    assert "chain broken" in led.audit().problem


def test_reordered_lines_are_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=5)
    lines = _lines(path)
    lines[1], lines[2] = lines[2], lines[1]
    _write(path, lines)
    assert Ledger(path).verify() == (False, 1)


def test_whitespace_only_edit_is_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=3)
    lines = _lines(path)
    lines[0] = lines[0].replace('"event":', '"event": ')
    _write(path, lines)
    led = Ledger(path)
    assert led.verify() == (False, 0)
    assert "canonical" in led.audit().problem


def test_inserted_blank_or_garbage_line_is_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    _ledger(path, n=3)
    lines = _lines(path)
    lines.insert(1, "\n")
    _write(path, lines)
    assert Ledger(path).verify() == (False, 1)


def test_deleting_the_tail_is_caught_by_the_known_head(tmp_path):
    path = tmp_path / "l.jsonl"
    led = _ledger(path, n=5)
    anchor = led.head
    _write(path, _lines(path)[:-2])
    assert led.verify() == (False, 3)                    # this process remembers its head
    fresh = Ledger(path)
    assert fresh.verify() == (True, 3)                   # a bare suffix cut is a valid shorter chain...
    assert fresh.verify(anchor=anchor) == (False, 3)     # ...which an externally recorded head exposes
    assert "head" in fresh.audit(anchor=anchor).problem


def test_anchor_inside_the_chain_verifies(tmp_path):
    path = tmp_path / "l.jsonl"
    led = _ledger(path, n=2)
    anchor = led.head
    led.append("later", {"x": 1})
    assert Ledger(path).verify(anchor=anchor) == (True, 3)


# ----------------------------------------------------------------------------- payload conversion


class Color(Enum):
    RED = "red"


@dataclass
class Point:
    x: float
    tags: tuple[str, ...]


def test_to_jsonable_handles_models_dataclasses_numpy_dates_and_nan():
    out = to_jsonable({
        "view": _view(),
        "fact": Fact("AAA.price", "Last close", 101.5, "USD", "2026-01-30", "test"),
        "point": Point(x=np.float64(1.25), tags=("a", "b")),
        "np_int": np.int64(7), "np_bool": np.bool_(True), "arr": np.array([1.0, 2.0]),
        "day": date(2026, 1, 30), "ts": pd.Timestamp("2026-01-30"), "nat": pd.NaT,
        "nan": float("nan"), "inf": math.inf, "enum": Color.RED, "set": {3, 1, 2},
        "path": Path("/tmp/x"), 5: "int key",
    })
    assert out["view"]["view"]["prob_outperform"] == 0.6 and out["view"]["agent"] == "technical"
    assert out["fact"] == {"id": "AAA.price", "label": "Last close", "value": 101.5, "unit": "USD",
                           "as_of": "2026-01-30", "source": "test"}
    assert out["point"] == {"x": 1.25, "tags": ["a", "b"]}
    assert out["np_int"] == 7 and type(out["np_int"]) is int and out["np_bool"] is True
    assert out["arr"] == [1.0, 2.0]
    assert out["day"] == "2026-01-30" and out["ts"].startswith("2026-01-30") and out["nat"] is None
    assert out["nan"] is None and out["inf"] is None
    assert out["enum"] == "red" and out["set"] == [1, 2, 3] and out["path"] == "/tmp/x"
    assert out["5"] == "int key"
    canonical_json(out)                                  # strict JSON: no NaN, all keys strings


def test_to_jsonable_rejects_unknown_types_and_colliding_keys():
    with pytest.raises(TypeError):
        to_jsonable({"x": object()})
    with pytest.raises(ValueError):
        to_jsonable({1: "a", "1": "b"})


def test_pydantic_payloads_round_trip_through_the_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path)
    led.append("view", {"ticker": "AAA", "record": _view(), "score": np.float32(0.5)})
    entry = next(Ledger(path).entries())
    assert ViewRecord.model_validate(entry["payload"]["record"]) == _view()
    assert entry["payload"]["score"] == 0.5
    assert Ledger(path).verify() == (True, 1)
