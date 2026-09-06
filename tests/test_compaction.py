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


def test_the_reply_cap_is_high_enough_to_never_sever_an_object():
    """768 was tried and was a regression: the model writes long `thought`
    fields, the cap truncated replies at exactly 768 tokens, and truncated JSON
    does not parse — six unparseable replies in a four-run smoke check.

    Compaction, not the cap, is what fixes the compounding cost: a verbose
    reply is now paid for once instead of on every later turn. So the ceiling
    only has to be high enough never to cut an object in half. Observed replies
    reached 1,894 tokens.
    """
    import inspect

    default = inspect.signature(OpenAICompatModel.__init__).parameters["max_tokens"].default
    assert default >= 2048, "a cap this low severs the payload; see the docstring"


def test_a_truncated_reply_is_reported_as_truncation_not_as_bad_json():
    """One is fixed by raising a limit, the other by changing a prompt.
    Counting them together is how a config bug gets blamed on the model."""
    from agent.llm import Completion

    class Truncating:
        provider = "t"
        model = "t"

        def __init__(self):
            self.calls = []

        def complete(self, system, messages):
            self.calls.append((system, list(messages)))
            return Completion(
                text='{"thought": "I will begin by considering',  # cut mid-object
                model="t",
                provider="t",
                tokens_out=2048,
                stop_reason="length",
            )

    tracer = Tracer(memory_only=True)
    result = run_task(
        LINE,
        model=Truncating(),
        dispatcher=build_registry(),
        policy=Policy.baseline(),
        tracer=tracer,
        system_prompt="SYSTEM",
    )
    notes = [e for e in tracer.events if e["event"] == "note"]
    assert any(e.get("truncated") for e in notes), "truncation was not identified"
    assert any("token limit" in e.get("note", "") for e in notes)
    assert result.terminal == "budget_exhausted"


def test_the_prompt_asks_for_a_short_thought():
    """The cap cannot be the only thing keeping replies short, because a cap
    that bites truncates rather than shortens."""
    from agent.loop import build_system_prompt

    prompt = build_system_prompt(build_registry())
    assert "single short sentence" in prompt
