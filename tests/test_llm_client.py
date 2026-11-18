"""Tests for the Claude client (aifirm.llm.client) against a mocked HTTP transport.

No API key and no network: every request goes to ``httpx2.MockTransport``. The tests check
what is sent (prompt-caching breakpoints, structured-output format) and how every kind of
answer is turned into either a validated object or ``LLMCallFailed`` (never a raw
``ValidationError`` that would abort the decision cycle), plus the budget reservation and
the served-model audit trail.
"""

import json
import threading
import time

import anthropic
import httpx2
import pytest

from aifirm.agents.analysts import run_analyst
from aifirm.config import FirmConfig
from aifirm.data.cache import DiskCache
from aifirm.data.factsheet import Fact, FactSheet
from aifirm.llm.client import (FALLBACK_BETA, BudgetExceeded, ClaudeLLM, LLMCallFailed, estimate_cost)
from aifirm.schemas import AnalystView, Revision

MODEL = "claude-opus-5-5"
REVISION = {"ticker": "AAA", "prob_outperform": 0.55, "expected_excess_return_pct": 1.0, "changed": False,
            "reason": "", "new_evidence_fact_ids": []}


def message(text: str | None = None, *, stop_reason: str = "end_turn", model: str = MODEL, content=None,
            input_tokens: int = 1000, output_tokens: int = 200, cache_read: int = 0, cache_write: int = 0) -> dict:
    if content is None:
        content = [] if text is None else [{"type": "text", "text": text}]
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write}}


class Transport:
    """Records every request; answers with ``reply(request_body)`` after an optional delay."""

    def __init__(self, reply, delay: float = 0.0):
        self.reply, self.delay = reply, delay
        self.requests: list[tuple[dict, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append((dict(request.headers), body))
        if self.delay:
            time.sleep(self.delay)
        out = self.reply(body)
        return out if isinstance(out, httpx2.Response) else httpx2.Response(200, json=out)


def make_llm(transport: Transport, budget: float = 10.0, cache: DiskCache | None = None, fallbacks: bool = True):
    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(transport)))
    return ClaudeLLM(budget_usd=budget, cache=cache or DiskCache(enabled=False), use_fallbacks=fallbacks,
                     client=client)


def call(llm: ClaudeLLM, schema=Revision, user: str = "USER", system: str = "SYSTEM PROMPT", max_tokens: int = 16000):
    return llm.structured(role="revision.technical", model=MODEL, system=system, user=user, schema=schema,
                          effort="high", max_tokens=max_tokens)


# --------------------------------------------------------------------------- request shape

def test_request_caches_the_system_prompt_not_the_unique_tail():
    t = Transport(lambda body: message(json.dumps(REVISION)))
    obj, rec = call(make_llm(t), system="STABLE ROLE PROMPT", user="per-ticker prompt")
    assert obj.prob_outperform == pytest.approx(0.55) and rec.error == ""
    headers, body = t.requests[0]
    # explicit breakpoint at the end of the shared system prompt ...
    assert body["system"] == [{"type": "text", "text": "STABLE ROLE PROMPT", "cache_control": {"type": "ephemeral"}}]
    # ... and no top-level automatic caching, which would write the unique user message at 1.25x every call
    assert "cache_control" not in body
    assert body["messages"] == [{"role": "user", "content": "per-ticker prompt"}]
    assert "cache_control" not in json.dumps(body["messages"])
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert set(body["output_config"]["format"]["schema"]["properties"]) == set(Revision.model_fields)
    assert body["fallbacks"] == "default" and FALLBACK_BETA in headers.get("anthropic-beta", "")


def test_no_fallback_parameters_when_disabled():
    t = Transport(lambda body: message(json.dumps(REVISION)))
    call(make_llm(t, fallbacks=False))
    headers, body = t.requests[0]
    assert "fallbacks" not in body and FALLBACK_BETA not in headers.get("anthropic-beta", "")


# --------------------------------------------------------------------------- failures become LLMCallFailed

@pytest.mark.parametrize("stop_reason, text", [
    ("max_tokens", json.dumps(REVISION)[:40]),             # truncated JSON
    ("refusal", "I can't help with that."),                # refusal with prose
    ("refusal", None),                                     # refusal with no content
    ("end_turn", '{"ticker": "AAA", "changed": "maybe"}'),  # answer that does not match the schema
    ("end_turn", "not json at all"),
    ("end_turn", None),                                    # no text block
])
def test_bad_answers_raise_llm_call_failed_and_are_charged(stop_reason, text):
    t = Transport(lambda body: message(text, stop_reason=stop_reason))
    llm = make_llm(t)
    with pytest.raises(LLMCallFailed):
        call(llm)
    assert len(llm.usage.calls) == 1 and llm.usage.calls[0].error
    assert llm.usage.spent_usd == pytest.approx(estimate_cost(MODEL, 1000, 200))   # the tokens were billed


def test_api_error_raises_llm_call_failed_and_is_recorded():
    t = Transport(lambda body: httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error",
                                                                                     "message": "Overloaded"}}))
    llm = make_llm(t)
    with pytest.raises(LLMCallFailed, match="529"):
        call(llm)
    assert llm.usage.calls[0].error.startswith("API error 529")


def test_truncated_analyst_answer_becomes_an_abstention_not_a_crash():
    fs = FactSheet(ticker="AAA", as_of="2025-03-21", name="Alpha Corp", sector="Tech",
                   facts=[Fact("AAA.mom_12_1", "Momentum 12-1", 23.4, "%", "2025-03-21", "prices")])
    t = Transport(lambda body: message('{"ticker": "AAA", "abstain": false, "prob_outp', stop_reason="max_tokens"))
    rec = run_analyst(make_llm(t), "technical", fs, FirmConfig(), MODEL)
    assert rec.view.abstain and rec.view.prob_outperform == 0.5
    assert rec.grounding_issues[0].startswith("llm failure:") and "truncated" in rec.grounding_issues[0]
    assert "analyst.technical" not in rec.view.abstain_reason       # the chair sees reasons under anonymous letters


# --------------------------------------------------------------------------- served model

def _analyst_json() -> str:
    view = AnalystView(ticker="AAA", abstain=False, abstain_reason="", prob_outperform=0.56,
                       expected_excess_return_pct=1.0, scenarios=[], key_risks=[], what_would_change_my_mind="",
                       private_insight="", evidence=[{"point": "Momentum is 23.4%.", "fact_ids": ["AAA.mom_12_1"],
                                                      "direction": "positive"}])
    return view.model_dump_json()


def test_served_model_is_recorded_and_credited_to_the_view():
    fs = FactSheet(ticker="AAA", as_of="2025-03-21", name="Alpha Corp", sector="Tech",
                   facts=[Fact("AAA.mom_12_1", "Momentum 12-1", 23.4, "%", "2025-03-21", "prices")])
    fallback_block = {"type": "fallback", "from": {"model": MODEL}, "to": {"model": "claude-fable-5-2"},
                      "trigger": {"category": None}}
    t = Transport(lambda body: message(model="claude-fable-5-2", content=[
        {"type": "text", "text": '{"ticker": "AAA", "abst'},          # partial output of the model that declined
        fallback_block,
        {"type": "text", "text": _analyst_json()}]))
    llm = make_llm(t)
    rec = run_analyst(llm, "technical", fs, FirmConfig(), MODEL)
    call_rec = llm.usage.calls[0]
    assert call_rec.model == MODEL and call_rec.served_by == "claude-fable-5-2" and call_rec.fallback_used
    assert call_rec.cost_usd == pytest.approx(estimate_cost("claude-fable-5-2", 1000, 200))
    assert not rec.view.abstain and rec.view.prob_outperform == pytest.approx(0.56)
    assert rec.model == "claude-fable-5-2"           # the forecast is credited to the model that made it


def test_cached_answer_keeps_the_served_model(tmp_path):
    t = Transport(lambda body: message(json.dumps(REVISION), model="claude-sonnet-5-5"))
    cache = DiskCache(tmp_path)
    first = call(make_llm(t, cache=cache))[1]
    second = call(make_llm(t, cache=cache))[1]
    assert len(t.requests) == 1 and second.cached
    assert first.served_by == second.served_by == "claude-sonnet-5-5"


# --------------------------------------------------------------------------- budget

def test_concurrent_calls_cannot_overshoot_the_budget_by_more_than_one_call():
    per_call = estimate_cost(MODEL, 2000, 11600)          # about $0.24 per answer
    t = Transport(lambda body: message(json.dumps(REVISION), input_tokens=2000, output_tokens=11600), delay=0.05)
    llm = make_llm(t, budget=1.0)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            call(llm, user=f"prompt {i}")
            result = "ok"
        except BudgetExceeded:
            result = "budget"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # Without the reservation all 8 calls pass the check before any is charged ($1.92 spent).
    assert llm.usage.spent_usd <= llm.usage.budget_usd + per_call + 1e-9
    assert outcomes.count("ok") == len(t.requests) == 5 and outcomes.count("budget") == 3
    assert llm.usage.spent_usd >= llm.usage.budget_usd       # exhaustion is visible to the pipeline's check
    assert llm.usage.reserved_usd == 0.0


def test_a_failed_call_releases_its_reservation():
    t = Transport(lambda body: message("garbage"))
    llm = make_llm(t, budget=5.0)
    for _ in range(3):
        with pytest.raises(LLMCallFailed):
            call(llm)
    assert llm.usage.reserved_usd == 0.0 and len(llm.usage.calls) == 3


def test_exhausted_budget_raises_before_any_request():
    t = Transport(lambda body: message(json.dumps(REVISION)))
    llm = make_llm(t, budget=0.0)
    with pytest.raises(BudgetExceeded):
        call(llm)
    assert t.requests == []


# --------------------------------------------------------------------------- pricing

def test_cache_reads_are_priced_per_model():
    assert estimate_cost("claude-opus-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)
    assert estimate_cost("claude-fable-5-2", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)
    assert estimate_cost("claude-sonnet-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.10)
    assert estimate_cost("claude-opus-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.50)
    assert estimate_cost("claude-opus-5-5", 0, 0, cache_write=1_000_000) == pytest.approx(5.0)
