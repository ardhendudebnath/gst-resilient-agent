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
    assert "retryable" not in j  # only meaningful on failures


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
