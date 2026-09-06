"""The registry is where the contract stops being a convention.

Each test below corresponds to a guarantee the agent loop relies on. If one of
these breaks, a failure class the taxonomy claims is fixed quietly returns.
"""

from __future__ import annotations

import pytest

from agent.budget import Budget, Ledger
from agent.contract import ContractError, ToolResult
from agent.jsonschema_lite import UnsupportedSchema
from agent.registry import IdempotencyCache, Registry, ToolSpec, build_call
from agent.trace import Tracer

SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 0}},
    "required": ["n"],
    "additionalProperties": False,
}


def _spec(name: str, handler, **kw) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=kw.pop("description", "test tool"),
        parameters=kw.pop("parameters", SCHEMA),
        handler=handler,
        **kw,
    )


def double(n: int) -> ToolResult:
    return ToolResult.ok_({"n": n * 2})


def explode(n: int) -> ToolResult:
    raise ZeroDivisionError("boom")


def wrong_type(n: int):
    return {"n": n}  # not a ToolResult


# --------------------------------------------------------------------------


def test_registration_rejects_a_schema_it_cannot_enforce():
    reg = Registry()
    with pytest.raises(UnsupportedSchema):
        reg.register(_spec("t", double, parameters={"type": "object", "$ref": "#/x"}))


def test_duplicate_registration_is_refused():
    reg = Registry()
    reg.register(_spec("t", double))
    with pytest.raises(ContractError, match="already registered"):
        reg.register(_spec("t", double))


def test_a_handler_that_raises_becomes_a_structured_error():
    """The guarantee: nothing a tool does can unwind the loop."""
    reg = Registry()
    reg.register(_spec("explode", explode))
    r = reg.invoke(build_call("explode", {"n": 1}))
    assert r.ok is False
    assert r.error == "internal"
    assert "ZeroDivisionError" in r.message
    assert r.retryable is False


def test_a_handler_returning_the_wrong_type_is_caught():
    reg = Registry()
    reg.register(_spec("wrong", wrong_type))
    r = reg.invoke(build_call("wrong", {"n": 1}))
    assert r.error == "internal" and "not a ToolResult" in r.message


def test_bad_arguments_come_back_with_the_specific_problems():
    """The agent has to be able to correct its call, so it is told what was wrong."""
    reg = Registry()
    reg.register(_spec("double", double))
    r = reg.invoke(build_call("double", {"n": -1}))
    assert r.error == "bad_argument"
    assert "below minimum" in r.message
    assert r.data["problems"]


def test_unknown_tool_is_not_found_not_internal():
    """A tool that does not exist is a fact about the request, not a crash."""
    reg = Registry()
    reg.register(_spec("double", double))
    r = reg.invoke(build_call("nope", {}))
    assert r.error == "not_found" and "double" in r.message


def test_the_handler_never_sees_invalid_arguments():
    seen: list[int] = []

    def record(n: int) -> ToolResult:
        seen.append(n)
        return ToolResult.ok_({})

    reg = Registry()
    reg.register(_spec("record", record))
    reg.invoke(build_call("record", {"n": "not an int"}))
    assert seen == []


# -- idempotency -----------------------------------------------------------


def test_repeat_calls_are_served_from_cache():
    """The decision that pre-empts the double-counting failure class."""
    calls: list[int] = []

    def counted(n: int) -> ToolResult:
        calls.append(n)
        return ToolResult.ok_({"n": n})

    reg = Registry()
    reg.register(_spec("counted", counted))
    cache = IdempotencyCache()
    ledger = Ledger()

    for _ in range(3):
        r = reg.invoke(build_call("counted", {"n": 7}), cache=cache, ledger=ledger)
        assert r.data["n"] == 7

    assert calls == [7], "the handler ran more than once for identical arguments"
    assert ledger.tool_calls == 1
    assert ledger.cache_hits == 2


def test_cache_records_which_keys_repeated():
    """Raw material for the duplicate-call failure class, without reading a trace."""
    reg = Registry()
    reg.register(_spec("double", double))
    cache = IdempotencyCache()
    for _ in range(3):
        reg.invoke(build_call("double", {"n": 1}), cache=cache)
    assert list(cache.to_json()["repeated_keys"].values()) == [3]


def test_different_arguments_are_different_calls():
    reg = Registry()
    reg.register(_spec("double", double))
    cache = IdempotencyCache()
    assert reg.invoke(build_call("double", {"n": 1}), cache=cache).data["n"] == 2
    assert reg.invoke(build_call("double", {"n": 2}), cache=cache).data["n"] == 4


def test_failures_are_not_cached():
    """Memoising a timeout turns one bad attempt into a permanently dead tool."""
    attempts: list[int] = []

    def flaky(n: int) -> ToolResult:
        attempts.append(n)
        if len(attempts) == 1:
            return ToolResult.err("timeout", "hung")
        return ToolResult.ok_({"n": n})

    reg = Registry()
    reg.register(_spec("flaky", flaky))
    cache = IdempotencyCache()
    assert reg.invoke(build_call("flaky", {"n": 1}), cache=cache).error == "timeout"
    assert reg.invoke(build_call("flaky", {"n": 1}), cache=cache).ok is True


# -- budget ----------------------------------------------------------------


def test_per_tool_cap_stops_a_retry_storm():
    reg = Registry()
    reg.register(_spec("double", double))
    ledger = Ledger(budget=Budget(max_calls_per_tool=2))
    # No cache, so each identical call is a real invocation.
    assert reg.invoke(build_call("double", {"n": 1}), ledger=ledger).ok
    assert reg.invoke(build_call("double", {"n": 1}), ledger=ledger).ok
    blocked = reg.invoke(build_call("double", {"n": 1}), ledger=ledger)
    assert blocked.error == "rate_limited"
    assert blocked.retryable is False
    assert "per-run limit" in blocked.message


# -- tracing ---------------------------------------------------------------


def test_every_invocation_traces_a_call_and_a_result():
    reg = Registry()
    reg.register(_spec("double", double))
    tracer = Tracer(memory_only=True)
    reg.invoke(build_call("double", {"n": 1}), tracer=tracer)
    events = [e["event"] for e in tracer.events]
    assert events == ["tool_call", "tool_result"]
    assert tracer.events[1]["result"]["data"]["n"] == 2
    assert "duration_ms" in tracer.events[1]


def test_a_cache_hit_is_visible_in_the_trace():
    reg = Registry()
    reg.register(_spec("double", double))
    tracer = Tracer(memory_only=True)
    cache = IdempotencyCache()
    reg.invoke(build_call("double", {"n": 1}), tracer=tracer, cache=cache)
    reg.invoke(build_call("double", {"n": 1}), tracer=tracer, cache=cache)
    assert [e["event"] for e in tracer.events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "cache_hit",
    ]


def test_a_failing_tool_still_traces_its_result():
    reg = Registry()
    reg.register(_spec("explode", explode))
    tracer = Tracer(memory_only=True)
    reg.invoke(build_call("explode", {"n": 1}), tracer=tracer)
    assert tracer.events[-1]["result"]["error"] == "internal"
