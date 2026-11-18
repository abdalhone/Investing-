"""Claude client used by every LLM agent in the firm.

Features
--------
* **Structured outputs**: every call asks for the schema's JSON (``output_config.format``)
  and returns a validated Pydantic object, so no regex parsing of prose. Validation runs
  in this module, *after* the stop reason is checked, so a refused, truncated or otherwise
  malformed answer becomes ``LLMCallFailed`` (never a raw ``ValidationError`` that would
  abort the whole decision cycle).
* **Prompt caching**: the system prompt (stable per role, never per ticker) carries an
  explicit ``cache_control`` breakpoint, so every request that shares it reads it from the
  cache. The per-ticker user message gets no breakpoint: it is unique, and paying the
  cache-write premium on bytes nobody reads back would be a pure surcharge. (Identical
  repeats are served by the disk cache below before any API call.) The minimum cacheable
  prefix is model-dependent; a shorter system prompt is simply not cached.
* **Content-hash cache**: identical (model, system, prompt, schema) calls are served from
  disk. Agents only re-reason when their inputs change, and backtests are reproducible.
* **Budget**: a per-run dollar ceiling; exceeding it raises ``BudgetExceeded`` (fail closed).
  Each live call first *reserves* its worst-case cost under a lock and settles to the
  actual cost afterwards, so concurrent calls cannot all pass the check before any of
  them is charged: spend can exceed the ceiling by at most one call.
* **Audit trail**: every call is recorded (prompt hash, requested model, the model that
  actually served it, tokens, cost, latency).
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

from pydantic import BaseModel, ValidationError

from ..data.cache import DiskCache, default_cache

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Cache writes (5-minute TTL) are billed at 1.25x
# input; cache reads at a model-dependent fraction of input (CACHE_READ_MULTIPLIER, 0.1x
# when not listed). Update when prices change.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_PRICE = (5.0, 25.0)              # unknown model: assume Opus-class prices
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER: dict[str, float] = {
    "claude-fable-5-2": 0.025,           # $0.20 / MTok
    "claude-opus-5-5": 0.05,             # $0.20 / MTok
    "claude-sonnet-5-5": 0.05,           # $0.10 / MTok
}
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
_CHARS_PER_TOKEN_LOW = 3.0               # conservative (low) chars-per-token for the worst-case reservation

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
    served_by: str = ""        # the model that actually answered (differs from ``model`` after a fallback)
    fallback_used: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Usage:
    """Spend tracking and the per-run budget.

    Live calls go through :meth:`reserve` / :meth:`settle`: a call reserves its worst-case
    cost under the lock *before* it is sent and is charged its actual cost afterwards. A
    call is admitted only if it fits in the budget next to the calls already in flight, or
    if nothing is in flight and the budget is not yet spent; otherwise it waits for the
    in-flight calls to settle and checks again. So the ceiling can be overshot by at most
    one call, whatever the thread-pool size, and ``BudgetExceeded`` is raised only once
    ``spent_usd >= budget_usd`` (which is what callers test to detect an exhausted budget)."""

    budget_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _in_flight: int = field(default=0, repr=False)

    def charge(self, rec: CallRecord) -> None:
        """Record a call that needed no reservation (cache hits, offline calls)."""
        with self._lock:
            self.spent_usd += rec.cost_usd
            self.calls.append(rec)
            self._lock.notify_all()

    def check(self) -> None:
        with self._lock:
            self._raise_if_spent()

    def _raise_if_spent(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"LLM budget of ${self.budget_usd:.2f} exhausted (spent ${self.spent_usd:.2f})")

    def reserve(self, estimate_usd: float) -> float:
        """Reserve ``estimate_usd`` for one live call (see the class docstring). Returns the
        amount reserved, to be passed to :meth:`settle`. Raises ``BudgetExceeded``."""
        estimate_usd = max(0.0, float(estimate_usd))
        with self._lock:
            while True:
                self._raise_if_spent()
                if self._in_flight == 0 or self.spent_usd + self.reserved_usd + estimate_usd <= self.budget_usd:
                    self._in_flight += 1
                    self.reserved_usd += estimate_usd
                    return estimate_usd
                self._lock.wait()

    def settle(self, rec: CallRecord, reserved_usd: float) -> None:
        """Release a reservation and charge the call's actual cost (successful or not)."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd) if self._in_flight else 0.0
            self.spent_usd += rec.cost_usd
            self.calls.append(rec)
            self._lock.notify_all()

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
            "fallbacks": sum(1 for c in self.calls if c.fallback_used),
        }


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = PRICES.get(model, DEFAULT_PRICE)
    read = CACHE_READ_MULTIPLIER.get(model, DEFAULT_CACHE_READ_MULTIPLIER)
    return (input_tokens * pin + output_tokens * pout + cache_read * pin * read
            + cache_write * pin * CACHE_WRITE_MULTIPLIER) / 1e6


def json_output_format(schema: type[BaseModel]) -> dict:
    """The ``output_config.format`` that asks for JSON matching ``schema`` (the same
    transformation the SDK's ``parse`` helper applies)."""
    import anthropic

    return {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}


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
        """One structured call. Every failure (API error, refusal, truncation, an answer
        that does not validate against ``schema``) raises ``LLMCallFailed``; the call is
        charged to ``usage`` either way."""
        h = prompt_hash(model, system, user, schema, effort)
        rec = CallRecord(role=role, model=model, prompt_hash=h)
        cached = self._cache_get(h)
        if cached is not None:
            try:
                obj = schema.model_validate(cached["output"])
            except (ValidationError, KeyError, TypeError):
                obj = None                 # unreadable entry: ask the model again
            if obj is not None:
                rec.cached = True
                rec.served_by = cached.get("served_by") or model
                rec.fallback_used = rec.served_by != model
                self.usage.charge(rec)
                return obj, rec

        reserved = self.usage.reserve(self._worst_case_cost(model, system, user, schema, max_tokens))
        try:
            parsed = self._live_call(rec, role=role, model=model, system=system, user=user, schema=schema,
                                     effort=effort, max_tokens=max_tokens)
        finally:
            self.usage.settle(rec, reserved)
        self._cache_put(h, {"output": parsed.model_dump(), "served_by": rec.served_by})
        return parsed, rec

    def _worst_case_cost(self, model: str, system: str, user: str, schema: type[BaseModel], max_tokens: int) -> float:
        """Upper estimate of one call's cost, reserved against the budget before sending: a
        low chars-per-token ratio, every input token at the cache-write price and the full
        ``max_tokens`` of output, at the dearest model the call may be served by."""
        chars = len(system) + len(user) + len(json.dumps(schema.model_json_schema()))
        tokens_in = int(chars / _CHARS_PER_TOKEN_LOW) + 1
        candidates = [model, *PRICES] if self.use_fallbacks else [model]
        return max(estimate_cost(m, 0, max_tokens, cache_write=tokens_in) for m in candidates)

    def _live_call(self, rec: CallRecord, *, role: str, model: str, system: str, user: str, schema: type[T],
                   effort: str, max_tokens: int) -> T:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            # Explicit breakpoint at the end of the stable system prompt (shared by every
            # ticker for this role); no top-level automatic caching, which would put the
            # breakpoint after the unique user message and only add write charges.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": json_output_format(schema)},
        )
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        a = self._anthropic
        t0 = time.monotonic()
        try:
            resp = self.client.beta.messages.create(**kwargs)
        except a.BadRequestError as exc:
            rec.error = f"bad request: {exc.message}"
            raise LLMCallFailed(rec.error) from exc
        except a.APIStatusError as exc:
            rec.error = f"API error {exc.status_code}: {exc.message}"
            raise LLMCallFailed(rec.error) from exc
        except a.APIConnectionError as exc:
            rec.error = f"connection error: {exc}"
            raise LLMCallFailed(rec.error) from exc
        except a.AnthropicError as exc:           # e.g. a response the SDK could not read
            rec.error = f"SDK error: {type(exc).__name__}: {exc}"
            raise LLMCallFailed(rec.error) from exc
        rec.latency_s = round(time.monotonic() - t0, 2)
        u = resp.usage
        rec.input_tokens = getattr(u, "input_tokens", 0) or 0
        rec.output_tokens = getattr(u, "output_tokens", 0) or 0
        rec.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
        rec.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
        rec.served_by = getattr(resp, "model", "") or model
        rec.fallback_used = (rec.served_by != model
                             or any(getattr(b, "type", "") == "fallback" for b in (resp.content or []))
                             or any(getattr(i, "type", "") == "fallback_message"
                                    for i in (getattr(u, "iterations", None) or [])))
        rec.cost_usd = estimate_cost(rec.served_by, rec.input_tokens, rec.output_tokens,
                                     rec.cache_read_tokens, rec.cache_write_tokens)
        if resp.stop_reason == "refusal":
            rec.error = "refusal"
            raise LLMCallFailed(f"{role}: model refused")
        if resp.stop_reason == "max_tokens":
            rec.error = "max_tokens"
            raise LLMCallFailed(f"{role}: output truncated at max_tokens={max_tokens}")
        text = _final_text(resp.content or [])
        if not text.strip():
            rec.error = "no structured output"
            raise LLMCallFailed(f"{role}: response had no structured output")
        try:
            return schema.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            rec.error = f"invalid structured output: {str(exc)[:200]}"
            raise LLMCallFailed(f"{role}: the answer did not match the {schema.__name__} schema") from exc

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


def _final_text(content: list) -> str:
    """The text of the answer. With server-side fallbacks, a model that declined may leave
    partial text before a ``fallback`` block; only the text after the last switch counts."""
    parts: list[str] = []
    for block in content:
        kind = getattr(block, "type", "")
        if kind == "fallback":
            parts = []
        elif kind == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


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
