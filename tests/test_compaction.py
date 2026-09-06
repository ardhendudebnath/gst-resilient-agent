"""History compaction: what re-enters the conversation, and what does not.

The measured problem: the model spent 1,000-1,900 tokens on replies whose
payload is a JSON object needing about 150, and every one of those was resent
on every later turn. A 1,894-token reply was followed by an input that grew by
2,073.

The trace keeps the raw reply in full — diagnosis needs it. Only the *history*
is compacted.
"""

from __future__ import annotations

import json

import pytest

from agent.llm import OpenAICompatModel, ScriptedModel
from agent.loop import MAX_THOUGHT_CHARS, Action, compact_action, run_task
from agent.policy import Policy
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


def test_compaction_keeps_the_action_and_drops_the_preamble():
    action = Action(tool="lookup_schedule", arguments={"heading": "7418"}, thought="x" * 4000)
    out = compact_action(action)
    parsed = json.loads(out)
    assert parsed["tool"] == "lookup_schedule"
    assert parsed["arguments"] == {"heading": "7418"}
    assert len(out) < 600, "a 4,000-character thought must not survive intact"
    assert len(parsed["thought"]) <= MAX_THOUGHT_CHARS


def test_compaction_is_lossless_for_short_replies():
    action = Action(tool="screen_scope", arguments={"description": "pens"}, thought="short")
    assert json.loads(compact_action(action))["thought"] == "short"


def test_a_verbose_reply_does_not_reach_the_next_turn(monkeypatch):
    """The whole point: a 4,000-character preamble is paid for once, not on
    every subsequent turn."""
    rambling = (
        "Let me think about this at considerable length. " * 90
        + json.dumps({
            "thought": "screen first",
            "tool": "screen_scope",
            "arguments": {"description": LINE["description"]},
        })
    )
    finish = json.dumps({
        "thought": "done",
        "tool": "draft_opinion",
        "arguments": {
            "terminal": "unanswerable",
            "invoice_date": "2026-03-14",
            "reason": "rate-fact-absent",
        },
    })
    model = ScriptedModel([rambling, finish])
    run_task(
        LINE,
        model=model,
        dispatcher=build_registry(),
        policy=Policy.baseline(),
        tracer=Tracer(memory_only=True),
        system_prompt="SYSTEM",
    )

    assert len(model.calls) == 2
    _system, second_turn = model.calls[1]
    history = "\n".join(m.content for m in second_turn)
    assert "considerable length" not in history, "the preamble was resent"
    assert "screen_scope" in history, "the action itself must survive"


def test_the_raw_reply_still_reaches_the_trace(monkeypatch):
    """Compaction is about what is *resent*, never about what is recorded.
    A trace that lost the model's actual words could not diagnose anything."""
    rambling = "Thinking out loud. " * 60 + json.dumps(
        {"thought": "t", "tool": "screen_scope", "arguments": {"description": "pens"}}
    )
    tracer = Tracer(memory_only=True)
    run_task(
        LINE,
        model=ScriptedModel([rambling]),
        dispatcher=build_registry(),
        policy=Policy.baseline(),
        tracer=tracer,
        system_prompt="SYSTEM",
    )
    texts = [e.get("text", "") for e in tracer.events if e["event"] == "llm_result"]
    assert any("Thinking out loud" in t for t in texts)


def test_an_unparseable_reply_is_resent_verbatim():
    """The correction that follows refers to it, so the model has to see what
    it actually sent."""
    good = json.dumps({
        "thought": "screen",
        "tool": "screen_scope",
        "arguments": {"description": LINE["description"]},
    })
    model = ScriptedModel(["I am not JSON, sorry", good])
    run_task(
        LINE,
        model=model,
        dispatcher=build_registry(),
        policy=Policy.baseline(),
        tracer=Tracer(memory_only=True),
        system_prompt="SYSTEM",
    )
    _system, second_turn = model.calls[1]
    history = "\n".join(m.content for m in second_turn)
    assert "I am not JSON, sorry" in history
    assert "PROTOCOL ERROR" in history


def test_the_client_reply_cap_is_sized_for_the_protocol():
    """One JSON action is about 150 tokens; 2048 invited 1,900-token replies
    that were then resent on every later turn."""
    model = OpenAICompatModel.__new__(OpenAICompatModel)
    import inspect

    default = inspect.signature(OpenAICompatModel.__init__).parameters["max_tokens"].default
    assert default == 768, "reply cap moved without the design note moving"
