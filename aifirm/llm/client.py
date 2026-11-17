"""Claude client used by every LLM agent in the firm.

Features
--------
* **Structured outputs**: every call returns a validated Pydantic object
  (``client.beta.messages.parse(output_format=...)``), so no regex parsing of prose.
* **Content-hash cache**: identical (model, system, prompt, schema) calls are served from
  disk. Agents only re-reason when their inputs change, and backtests are reproducible.
* **Budget**: a per-run dollar ceiling; exceeding it raises ``BudgetExceeded`` (fail closed).
* **Audit trail**: every call is recorded (prompt hash, model, tokens, cost, latency).
* **Refusal handling**: server-side fallbacks are on by default; a final refusal or a
  truncated answer raises ``LLMCallFailed`` and the caller abstains (never a silent "hold").
* **Offline mode**: ``OfflineLLM`` answers with deterministic heuristics so the whole
  pipeline runs and tests without an API key. It is NOT an AI and must never be used for
  real decisions.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, TypeVar

from pydantic import BaseModel

from ..data.cache import DiskCache, default_cache

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Cache reads are billed at ~0.1x input and cache
# writes at ~1.25x input. Update when prices change.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class BudgetExceeded(RuntimeError):
    pass


class LLMCallFailed(RuntimeError):
    pass


@dataclass
class CallRecord:
    role: str
    model: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    served_by: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Usage:
    budget_usd: float
    spent_usd: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, rec: CallRecord) -> None:
        with self._lock:
            self.spent_usd += rec.cost_usd
            self.calls.append(rec)

    def check(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"LLM budget of ${self.budget_usd:.2f} exhausted (spent ${self.spent_usd:.2f})")

    def summary(self) -> dict:
        live = [c for c in self.calls if not c.cached]
        return {
            "calls": len(self.calls),
            "live_calls": len(live),
            "cached_calls": len(self.calls) - len(live),
            "input_tokens": sum(c.input_tokens for c in live),
            "output_tokens": sum(c.output_tokens for c in live),
            "cost_usd": round(self.spent_usd, 4),
            "failures": sum(1 for c in self.calls if c.error),
        }


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = PRICES.get(model, (5.0, 25.0))
    return (input_tokens * pin + output_tokens * pout + cache_read * pin * 0.1 + cache_write * pin * 1.25) / 1e6


def prompt_hash(model: str, system: str, user: str, schema: type[BaseModel], effort: str) -> str:
    blob = json.dumps({"m": model, "s": system, "u": user, "f": schema.__name__,
                       "schema": schema.model_json_schema(), "e": effort}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLM:
    """Interface: ``structured(role, model, system, user, schema, effort) -> (obj, CallRecord)``."""

    usage: Usage

    def structured(self, *, role: str, model: str, system: str, user: str, schema: type[T],
                   effort: str = "medium", max_tokens: int = 16000) -> tuple[T, CallRecord]:
        raise NotImplementedError

    @property
    def is_offline(self) -> bool:
        return False


class ClaudeLLM(LLM):
    def __init__(self, budget_usd: float = 10.0, cache: DiskCache | None = None,
                 use_fallbacks: bool = True, client=None):
        import anthropic

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic(max_retries=4)
        self.cache = cache or default_cache()
        self.use_fallbacks = use_fallbacks
        self.usage = Usage(budget_usd=budget_usd)

    def structured(self, *, role: str, model: str, system: str, user: str, schema: type[T],
                   effort: str = "medium", max_tokens: int = 16000) -> tuple[T, CallRecord]:
        h = prompt_hash(model, system, user, schema, effort)
        rec = CallRecord(role=role, model=model, prompt_hash=h)
        cached = self._cache_get(h)
        if cached is not None:
            rec.cached = True
            rec.served_by = cached.get("served_by", model)
            self.usage.charge(rec)
            return schema.model_validate(cached["output"]), rec

        self.usage.check()
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
        )
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        t0 = time.monotonic()
        try:
            resp = self.client.beta.messages.parse(**kwargs)
        except self._anthropic.BadRequestError as exc:
            rec.error = f"bad request: {exc.message}"
            self.usage.charge(rec)
            raise LLMCallFailed(rec.error) from exc
        except self._anthropic.APIStatusError as exc:
            rec.error = f"API error {exc.status_code}: {exc.message}"
            self.usage.charge(rec)
            raise LLMCallFailed(rec.error) from exc
        except self._anthropic.APIConnectionError as exc:
            rec.error = f"connection error: {exc}"
            self.usage.charge(rec)
            raise LLMCallFailed(rec.error) from exc
        rec.latency_s = round(time.monotonic() - t0, 2)
        u = resp.usage
        rec.input_tokens = u.input_tokens or 0
        rec.output_tokens = u.output_tokens or 0
        rec.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
        rec.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
        rec.served_by = resp.model
        rec.cost_usd = estimate_cost(resp.model, rec.input_tokens, rec.output_tokens,
                                     rec.cache_read_tokens, rec.cache_write_tokens)
        if resp.stop_reason == "refusal":
            rec.error = "refusal"
            self.usage.charge(rec)
            raise LLMCallFailed(f"{role}: model refused")
        if resp.stop_reason == "max_tokens":
            rec.error = "max_tokens"
            self.usage.charge(rec)
            raise LLMCallFailed(f"{role}: output truncated at max_tokens={max_tokens}")
        parsed = resp.parsed_output
        if parsed is None:
            rec.error = "no parsed output"
            self.usage.charge(rec)
            raise LLMCallFailed(f"{role}: response had no structured output")
        self.usage.charge(rec)
        self._cache_put(h, {"output": parsed.model_dump(), "served_by": rec.served_by})
        return parsed, rec

    # cache helpers ---------------------------------------------------------
    def _cache_get(self, h: str) -> dict | None:
        if not self.cache.enabled:
            return None
        path = self.cache._path("llm", h, "json")
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    def _cache_put(self, h: str, payload: dict) -> None:
        if self.cache.enabled:
            self.cache.get_or_set("llm", h, lambda: payload, kind="json")


Responder = Callable[[str, str, str, type[BaseModel]], BaseModel]


class OfflineLLM(LLM):
    """Deterministic stand-in for Claude. Each role registers a responder that builds a
    schema-valid object from the prompt text (usually by reading fact values). Used for
    tests, CI and dry runs. Its 'opinions' are simple rules, not analysis."""

    def __init__(self, responders: dict[str, Responder] | None = None):
        self.responders: dict[str, Responder] = responders or {}
        self.usage = Usage(budget_usd=float("inf"))

    @property
    def is_offline(self) -> bool:
        return True

    def register(self, role_prefix: str, fn: Responder) -> None:
        self.responders[role_prefix] = fn

    def structured(self, *, role: str, model: str, system: str, user: str, schema: type[T],
                   effort: str = "medium", max_tokens: int = 16000) -> tuple[T, CallRecord]:
        fn = None
        for prefix in sorted(self.responders, key=len, reverse=True):
            if role.startswith(prefix):
                fn = self.responders[prefix]
                break
        if fn is None:
            raise LLMCallFailed(f"OfflineLLM has no responder for role '{role}'")
        obj = fn(role, system, user, schema)
        rec = CallRecord(role=role, model="offline", prompt_hash=prompt_hash("offline", system, user, schema, effort),
                         served_by="offline")
        self.usage.charge(rec)
        return schema.model_validate(obj.model_dump()), rec
