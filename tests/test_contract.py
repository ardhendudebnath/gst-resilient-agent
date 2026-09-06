"""The tool contract is a guarantee, so it is tested as one."""

from __future__ import annotations

import pytest

from agent.contract import (
    ERROR_CODES,
    ContractError,
    Evidence,
    ToolCall,
    ToolResult,
    call_key,
)


def test_error_code_set_is_closed():
    """An invented code would silently miss the week-6 recovery policy table."""
    with pytest.raises(ContractError, match="unknown error code"):
        ToolResult.err("flaky")


def test_error_codes_carry_a_default_retryability():
    assert ERROR_CODES["timeout"] is True
    assert ERROR_CODES["bad_argument"] is False
    # A hash mismatch does not resolve itself.
    assert ERROR_CODES["source_mismatch"] is False


def test_retryable_can_be_overridden_but_the_code_cannot():
    r = ToolResult.err("rate_limited", "hard quota", retryable=False)
    assert r.error == "rate_limited" and r.retryable is False


def test_evidence_must_be_evidence():
    """Verbatim document text may not arrive as a bare string.

    The trusted/untrusted split is the whole basis of the week-5 defences; a
    tool that smuggles document text in as a plain value defeats them silently.
    """
    with pytest.raises(ContractError, match="must be Evidence"):
        ToolResult.ok_({"slab": "18"}, evidence=["some raw text"])  # type: ignore[list-item]


def test_ok_result_shape():
    r = ToolResult.ok_({"slab": "18"}, [Evidence("09-2025-CTR.pdf", "Sch II", "6810 ...")])
    assert r.ok and r.error is None
    j = r.to_json()
    assert j["data"]["slab"] == "18"
    assert j["evidence"][0]["source"] == "09-2025-CTR.pdf"


def test_the_envelope_has_the_same_shape_on_success_and_failure():
    """Every field present either way, so a consumer never branches on key
    existence. An envelope whose shape depends on the outcome is one that gets
    parsed two different ways, and the second way is always the buggy one."""
    required = {"ok", "data", "error", "retryable", "tool_call_id", "latency_ms"}
    good = ToolResult.ok_({"slab": "18"}).to_json()
    bad = ToolResult.err("timeout", "took too long").to_json()
    assert required <= set(good)
    assert required <= set(bad)
    assert good["error"] is None and good["retryable"] is False
    assert bad["error"] == "timeout" and bad["retryable"] is True


def test_an_unstamped_result_reports_no_id_or_latency():
    """Only `registry.invoke` stamps them, so a hand-built result says so
    rather than carrying a plausible-looking zero."""
    r = ToolResult.ok_({"a": 1})
    assert r.tool_call_id is None
    assert r.latency_ms is None


def test_stamping_preserves_everything_else():
    r = ToolResult.err("timeout", "slow", data={"x": 1}).stamped(
        tool_call_id="abc123", latency_ms=12.5
    )
    assert r.tool_call_id == "abc123"
    assert r.latency_ms == 12.5
    assert r.error == "timeout" and r.retryable is True
    assert r.data == {"x": 1}


def test_registry_stamps_id_and_latency_onto_every_result():
    """Stamped on the one path every call goes through, so a tool cannot forget
    and cannot lie about its own latency."""
    from agent.registry import build_call
    from agent.tools import build_registry

    registry = build_registry()
    call = build_call("screen_scope", {"description": "copper pipe fittings"})
    result = registry.invoke(call)
    assert result.tool_call_id == call.call_id
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0

    missing = registry.invoke(build_call("no_such_tool", {}))
    assert missing.tool_call_id is not None, "failures are stamped too"
    assert missing.latency_ms is not None


def test_call_key_is_canonical_over_argument_order():
    """The model orders keys arbitrarily; the cache must not care."""
    assert call_key("t", {"a": 1, "b": 2}) == call_key("t", {"b": 2, "a": 1})


def test_call_key_separates_tools_and_arguments():
    assert call_key("t", {"a": 1}) != call_key("u", {"a": 1})
    assert call_key("t", {"a": 1}) != call_key("t", {"a": 2})


def test_call_id_is_per_invocation_and_key_is_per_call():
    """Two invocations of the same call share a key and not an id."""
    a = ToolCall("lookup_schedule", {"heading": "6810"})
    b = ToolCall("lookup_schedule", {"heading": "6810"})
    assert a.key == b.key
    assert a.call_id != b.call_id


def test_chaos_tagging_survives_serialisation():
    """An injected failure must never look organic in a trace."""
    r = ToolResult.err("timeout", "tool hung").with_chaos("timeout@0.25")
    assert r.to_json()["chaos"] == "timeout@0.25"


def test_chaos_tagging_preserves_the_result():
    r = ToolResult.ok_({"slab": "18"}).with_chaos("duplicate")
    assert r.ok and r.data["slab"] == "18" and r.chaos == "duplicate"
