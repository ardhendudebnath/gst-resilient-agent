"""Recovery policy: what the loop does about a failure, per failure class.

Two things are being defended here. That a transient failure is re-issued
*without* spending a model turn — which is the whole saving — and that a
recovery policy cannot become the retry storm it exists to prevent.
"""

from __future__ import annotations

import json

import pytest

from agent.contract import ERROR_CODES, ToolCall, ToolResult
from agent.llm import ScriptedModel
from agent.loop import run_task
from agent.policy import Policy
from agent.recovery import ABORT, ADVISE, RETRY, RecoveryPolicy, covered_codes
from agent.registry import IdempotencyCache, Registry
from agent.tools import build_registry
from agent.trace import Tracer

pytest.importorskip("pypdf", reason="the tools read the archived notifications")

LINE = {
    "line_id": "inv-test",
    "description": "Copper handi, 2 litre, for kitchen use",
    "declared_hsn": "7418",
    "declared_rate": "12",
    "taxable_value_inr": 100000.00,
    "invoice_date": "2026-03-14",
}


def act(tool: str, **arguments) -> str:
    return json.dumps({"thought": "t", "tool": tool, "arguments": arguments})


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def test_every_error_code_the_contract_allows_has_a_rule_or_a_documented_default():
    """A code with no rule falls back to advising, which is the baseline path.
    This test exists so that adding a code to ERROR_CODES surfaces the gap."""
    uncovered = set(ERROR_CODES) - covered_codes()
    # These are the ones deliberately left to the fallback.
    assert uncovered == {"not_found"} or not uncovered - {"not_found"}, uncovered


def test_transient_codes_retry_and_permanent_ones_do_not():
    policy = RecoveryPolicy()
    for code in ("timeout", "rate_limited", "unavailable"):
        assert policy.decide(f"k-{code}", ToolResult.err(code)).action == RETRY
    for code in ("bad_argument", "malformed", "internal"):
        assert policy.decide(f"k-{code}", ToolResult.err(code)).action == ADVISE


def test_a_corpus_mismatch_aborts_rather_than_retrying():
    """No number of retries conjures the right Gazette, and a rate read from
    the wrong document would be traceable to nothing."""
    policy = RecoveryPolicy()
    for code in ("source_missing", "source_mismatch"):
        assert policy.decide("k", ToolResult.err(code)).action == ABORT


def test_a_successful_result_needs_no_recovery():
    assert RecoveryPolicy().decide("k", ToolResult.ok_({"a": 1})).action == ""


def test_an_unknown_code_falls_back_to_advising_not_to_inventing_behaviour():
    policy = RecoveryPolicy()
    decision = policy.decide("k", ToolResult(ok=False, error="not_found"))
    assert decision.action == ADVISE
    assert decision.guidance


# --------------------------------------------------------------------------
# Retries are bounded
# --------------------------------------------------------------------------


def test_retries_are_bounded_per_call_and_then_hand_over_to_the_model():
    """A policy that can retry without limit is a worse failure than the one
    it handles."""
    policy = RecoveryPolicy()
    result = ToolResult.err("timeout")
    actions = [policy.decide("same-key", result).action for _ in range(5)]
    assert actions[0] == RETRY and actions[1] == RETRY
    assert actions[2:] == [ADVISE, ADVISE, ADVISE]
    assert policy.retries == 2


def test_two_different_calls_each_get_their_own_retry_budget():
    """Keyed on the call, not the tool: two lookups that both time out are two
    problems, and one that keeps timing out is still bounded."""
    policy = RecoveryPolicy()
    assert policy.decide("key-a", ToolResult.err("timeout")).action == RETRY
    assert policy.decide("key-b", ToolResult.err("timeout")).action == RETRY
    assert policy.retries == 2


def test_the_exhausted_message_tells_the_model_to_stop_calling_it():
    policy = RecoveryPolicy()
    for _ in range(3):
        decision = policy.decide("k", ToolResult.err("timeout"))
    assert decision.action == ADVISE
    assert "Do not keep calling it" in decision.guidance


# --------------------------------------------------------------------------
# In the loop
# --------------------------------------------------------------------------


class FlakyRegistry:
    """Fails a named tool `n` times, then delegates. Same signature as Registry."""

    def __init__(self, inner: Registry, tool: str, failures: int, code: str = "timeout"):
        self.inner = inner
        self.tool = tool
        self.left = failures
        self.code = code
        self.attempts = 0

    def specs(self):
        return self.inner.specs()

    def names(self):
        return self.inner.names()

    def get(self, name):
        return self.inner.get(name)

    def __contains__(self, name):
        return name in self.inner

    def invoke(self, call: ToolCall, **kw):
        if call.name == self.tool:
            self.attempts += 1
            if self.left > 0:
                self.left -= 1
                return ToolResult.err(self.code, "injected").stamped(
                    tool_call_id=call.call_id, latency_ms=0.1
                )
        return self.inner.invoke(call, **kw)


def _run(responses, *, policy, dispatcher, monkeypatch=None):
    tracer = Tracer(memory_only=True)
    model = ScriptedModel(responses)
    result = run_task(
        LINE,
        model=model,
        dispatcher=dispatcher,
        policy=policy,
        tracer=tracer,
        system_prompt="SYSTEM",
    )
    return result, model, tracer


def test_a_transient_failure_is_retried_without_spending_a_model_turn(monkeypatch):
    """The whole saving: routing a timeout through the model costs an
    iteration and a few thousand tokens of resent history to reach the
    decision the error code already implied."""
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    flaky = FlakyRegistry(build_registry(), "lookup_schedule", failures=1)
    responses = [
        act("screen_scope", description=LINE["description"]),
        act("lookup_schedule", heading="7418", on_date="2026-03-14"),
        act("draft_opinion", terminal="unanswerable", invoice_date="2026-03-14",
            reason="rate-fact-absent"),
    ]
    result, model, tracer = _run(
        responses, policy=Policy(recovery_policies=True), dispatcher=flaky
    )

    assert flaky.attempts == 2, "the call should have been re-issued"
    assert len(model.calls) == 3, "the retry must not have cost a model turn"
    assert result.recovery["retries"] == 1
    assert any(e["event"] == "policy" for e in tracer.events)


def test_the_baseline_hands_the_same_failure_to_the_model_instead(monkeypatch):
    """The before side. If this ever starts retrying, the baseline has grown a
    defence and the before/after gap is fiction."""
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    flaky = FlakyRegistry(build_registry(), "lookup_schedule", failures=1)
    responses = [
        act("screen_scope", description=LINE["description"]),
        act("lookup_schedule", heading="7418", on_date="2026-03-14"),
        act("lookup_schedule", heading="7418", on_date="2026-03-14"),
        act("draft_opinion", terminal="unanswerable", invoice_date="2026-03-14",
            reason="rate-fact-absent"),
    ]
    result, model, _ = _run(
        responses, policy=Policy.baseline(), dispatcher=flaky
    )
    assert result.recovery is None
    assert len(model.calls) == 4, "the baseline spends a model turn on the retry"


def test_an_abort_ends_the_run_naming_the_tool_error(monkeypatch):
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    flaky = FlakyRegistry(
        build_registry(), "lookup_schedule", failures=99, code="source_mismatch"
    )
    responses = [
        act("screen_scope", description=LINE["description"]),
        act("lookup_schedule", heading="7418", on_date="2026-03-14"),
    ] + [act("screen_scope", description="x")] * 6
    result, _, tracer = _run(
        responses, policy=Policy(recovery_policies=True), dispatcher=flaky
    )
    assert result.terminal == "budget_exhausted"
    assert "source_mismatch" in result.reason
    assert "aborted" in result.reason


def test_advice_reaches_the_model_alongside_the_error(monkeypatch):
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    flaky = FlakyRegistry(
        build_registry(), "lookup_schedule", failures=99, code="bad_argument"
    )
    responses = [
        act("screen_scope", description=LINE["description"]),
        act("lookup_schedule", heading="7418", on_date="2026-03-14"),
        act("draft_opinion", terminal="unanswerable", invoice_date="2026-03-14",
            reason="rate-fact-absent"),
    ]
    _, model, _ = _run(
        responses, policy=Policy(recovery_policies=True), dispatcher=flaky
    )
    history = "\n".join(m.content for _s, msgs in model.calls for m in msgs)
    assert "RECOVERY:" in history
    assert "Fix the arguments" in history


def test_a_retry_storm_is_still_bounded_by_the_per_tool_cap(monkeypatch):
    """Recovery re-issues through the ledger, so the registry's existing cap
    still applies. A policy that could bypass it would be the storm."""
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    from agent.budget import Budget

    flaky = FlakyRegistry(build_registry(), "screen_scope", failures=99)
    responses = [act("screen_scope", description=f"x{i}") for i in range(10)]
    tracer = Tracer(memory_only=True)
    result = run_task(
        LINE,
        model=ScriptedModel(responses),
        dispatcher=flaky,
        policy=Policy(recovery_policies=True),
        budget=Budget(max_iterations=6, max_calls_per_tool=3),
        tracer=tracer,
        system_prompt="SYSTEM",
    )
    assert result.terminal == "budget_exhausted"
    assert result.ledger["calls_by_tool"].get("screen_scope", 0) <= 3


def test_hardened_policy_turns_recovery_on():
    assert Policy.hardened().recovery_policies is True
    assert Policy.baseline().recovery_policies is False
