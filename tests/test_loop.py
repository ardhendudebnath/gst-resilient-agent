"""The loop: the happy path, every way it stops, and the defences it can switch on.

Driven by `ScriptedModel` throughout. That is not a convenience — most of what
this project measures is a property of the *loop* rather than of the model
(whether a bound trips before a bill, whether a duplicated call is deduplicated,
whether an unparseable reply is recovered from), and those need the model's
output held fixed so the loop is the only thing varying. Testing them against a
live model would measure its mood.
"""

from __future__ import annotations

import json

import pytest

from agent.budget import Budget
from agent.llm import ScriptedModel
from agent.loop import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    MAX_MODEL_RETRIES,
    RunState,
    _backoff,
    build_system_prompt,
    check_prerequisites,
    parse_action,
    run_task,
)
from agent.policy import Policy, PolicyNotImplemented
from agent.tools import build_registry
from agent.trace import Tracer

pytest.importorskip("pypdf", reason="the tools read the archived notifications")

LINE = {
    "line_id": "inv-0042",
    "description": "Quartz slabs, 92% crushed quartz bonded with 8% polyester resin, polished",
    "declared_hsn": "6802",
    "declared_rate": "12",
    "taxable_value_inr": 250000.00,
    "invoice_date": "2026-03-14",
}


def act(tool: str, **arguments) -> str:
    return json.dumps({"thought": "step", "tool": tool, "arguments": arguments})


HAPPY_PATH = [
    act("screen_scope", description=LINE["description"]),
    act("propose_headings", description=LINE["description"]),
    act("lookup_schedule", heading="6810", on_date="2026-03-14"),
    act("rate_history", heading="6810", on_date="2026-03-14"),
    act(
        "compute_liability",
        taxable_value_inr=250000.00,
        correct_slab="18",
        declared_rate="12",
        on_date="2026-03-14",
    ),
    act(
        "draft_opinion",
        terminal="opinion",
        line_id="inv-0042",
        invoice_date="2026-03-14",
        hsn4="6810",
        slab="18",
        declared_correct=False,
        differential_inr="15000.00",
        declared_rate="12",
        citations=[{"notification": "9/2025-CT(R)", "schedule": "II", "heading": "6810"}],
        notes="Articles of artificial stone fall in 6810.",
    ),
]


def run(responses, *, policy=None, budget=None, line=None):
    model = ScriptedModel(responses)
    tracer = Tracer(memory_only=True)
    result = run_task(
        line or LINE,
        model=model,
        dispatcher=build_registry(),
        policy=policy or Policy.baseline(),
        budget=budget,
        tracer=tracer,
    )
    return result, model, tracer


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_worked_example_runs_end_to_end():
    result, _, tracer = run(HAPPY_PATH)

    assert result.terminal == "opinion"
    assert result.opinion["hsn4"] == "6810"
    assert result.opinion["slab"] == "18"
    assert result.opinion["differential_inr"] == "15000.00"
    assert result.opinion["declared_correct"] is False
    assert result.steps == 6
    assert [c["name"] for c in result.tool_calls] == [
        "screen_scope",
        "propose_headings",
        "lookup_schedule",
        "rate_history",
        "compute_liability",
        "draft_opinion",
    ]
    assert all(c["ok"] for c in result.tool_calls)

    events = [e["event"] for e in tracer.events]
    assert events[0] == "run_start"
    assert events[-1] == "run_end"
    assert events.count("tool_call") == 6
    assert events.count("tool_result") == 6


def test_the_run_ends_the_moment_draft_opinion_succeeds():
    """A response after the terminal call must never be consumed.

    The scripted model errors when exhausted, so a loop that kept going would
    surface as a model failure rather than silently over-running.
    """
    _, model, _ = run(HAPPY_PATH)
    assert len(model.calls) == 6


def test_retrieved_text_never_reaches_the_system_prompt():
    """The one injection defence that is structural rather than switchable."""
    _, model, _ = run(HAPPY_PATH)
    systems = {system for system, _ in model.calls}
    assert len(systems) == 1
    system = systems.pop()
    assert "Articles of cement" not in system
    assert LINE["description"] not in system
    assert "lookup_schedule" in system  # the tool list is there


# --------------------------------------------------------------------------
# Stopping conditions
# --------------------------------------------------------------------------


def test_iteration_bound_ends_the_run_as_budget_exhausted():
    spinning = [act("screen_scope", description="x")] * 20
    result, _, _ = run(spinning, budget=Budget(max_iterations=4))
    assert result.terminal == "budget_exhausted"
    assert result.reason == "max_iterations"
    assert result.opinion is None
    assert result.finished is False


def test_per_tool_cap_is_enforced_by_the_registry_not_the_loop():
    """Arguments must differ, or the cap never fires — and that is the design.

    A retry storm on *identical* arguments is absorbed by the idempotency cache
    and costs one tool call no matter how many times it is issued. The per-tool
    cap exists for the other kind of storm: the same tool called over and over
    with the arguments tweaked each time, which the cache cannot collapse. The
    two mechanisms cover different failures and this test pins the second.
    """
    spinning = [act("screen_scope", description=f"widget {i}") for i in range(8)]
    result, _, _ = run(spinning, budget=Budget(max_iterations=8, max_calls_per_tool=3))
    errors = [c["error"] for c in result.tool_calls]
    assert errors.count("rate_limited") >= 1, "the per-tool cap should have fired"
    assert result.ledger["calls_by_tool"]["screen_scope"] == 3


def test_an_identical_retry_storm_is_absorbed_by_the_cache_not_the_cap():
    """The complement of the test above: same arguments, so one real call."""
    spinning = [act("screen_scope", description="x")] * 8
    result, _, _ = run(spinning, budget=Budget(max_iterations=8, max_calls_per_tool=3))
    assert result.ledger["calls_by_tool"]["screen_scope"] == 1
    assert result.ledger["cache_hits"] == 7
    assert [c["error"] for c in result.tool_calls].count("rate_limited") == 0


def test_an_unparseable_reply_is_corrected_and_recovered_from():
    responses = ["I think we should look at heading 6810.", *HAPPY_PATH]
    result, model, tracer = run(responses)
    assert result.terminal == "opinion"
    notes = [e for e in tracer.events if e["event"] == "note"]
    assert any("unparseable" in e.get("note", "") for e in notes)
    # The correction reached the model as a user turn.
    _, last_messages = model.calls[-1]
    assert any("PROTOCOL ERROR" in m.content for m in last_messages)


def test_persistent_unparseable_replies_end_the_run():
    result, _, _ = run(["not json at all"] * 10)
    assert result.terminal == "budget_exhausted"
    assert result.reason.startswith("unparseable_replies")


def test_a_non_transient_model_error_stops_immediately():
    """Retrying a 401 or a bad request reaches the same failure more slowly."""
    result, _, _ = run([])  # ScriptedModel errors immediately when exhausted
    assert result.terminal == "budget_exhausted"
    assert result.reason.startswith("model_error")
    assert result.model_retries == 1  # tried once, did not retry
    assert result.steps == 0


class FlakyModel:
    """Returns `failures` transient errors, then plays the script.

    Models the 503 behaviour actually observed against
    nemotron-3-ultra-550b-a55b, where an eight-step run met three of them.
    """

    provider = "flaky"
    model = "flaky"

    def __init__(self, failures: int, responses: list[str], error: str = "http_503: overloaded"):
        self._left = failures
        self._inner = ScriptedModel(responses)
        self._error = error
        self.attempts = 0

    def complete(self, system, messages):
        self.attempts += 1
        if self._left > 0:
            self._left -= 1
            from agent.llm import Completion

            return Completion(text="", model=self.model, provider=self.provider,
                              error=self._error)
        return self._inner.complete(system, messages)


def _run_with(model, *, policy=None, budget=None):
    tracer = Tracer(memory_only=True)
    result = run_task(
        LINE,
        model=model,
        dispatcher=build_registry(),
        policy=policy or Policy.baseline(),
        budget=budget,
        tracer=tracer,
    )
    return result, tracer


def test_transient_provider_failures_do_not_consume_iterations(monkeypatch):
    """The bug this fixes: three 503s in an eight-step run consumed three of
    twelve iterations, so a longer line would have ended in budget_exhausted —
    recording an infrastructure failure as the agent giving up."""
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    model = FlakyModel(3, HAPPY_PATH)
    result, tracer = _run_with(model)

    assert result.terminal == "opinion"
    assert result.steps == 6, "the 503s must not be charged as reasoning steps"
    assert result.model_retries == 3
    assert model.attempts == 9  # 3 failures + 6 real turns

    notes = [e for e in tracer.events if e["event"] == "note"]
    retries = [e for e in notes if "transient" in e.get("note", "")]
    assert len(retries) == 3
    assert all(e.get("backoff_s", 0) >= 0 for e in retries)


def test_a_sustained_outage_ends_the_run_as_unavailable(monkeypatch):
    monkeypatch.setattr("agent.loop.time.sleep", lambda _s: None)
    result, _ = _run_with(FlakyModel(99, HAPPY_PATH))
    assert result.terminal == "budget_exhausted"
    assert result.reason.startswith("model_unavailable")
    assert result.steps == 0
    assert result.model_retries == MAX_MODEL_RETRIES + 1


def test_backoff_grows_and_is_jittered():
    """Jitter is not decoration: a suite runs many tasks against one endpoint,
    and synchronised retries keep an overloaded service overloaded."""
    first = [_backoff(1) for _ in range(20)]
    later = [_backoff(4) for _ in range(20)]
    assert min(first) > 0
    assert max(first) <= BACKOFF_BASE_S
    assert sum(later) / len(later) > sum(first) / len(first)
    assert max(_backoff(50) for _ in range(20)) <= BACKOFF_CAP_S
    assert len(set(first)) > 1, "identical delays mean no jitter"


def test_a_json_object_inside_a_code_fence_is_accepted():
    fenced = ["```json\n" + HAPPY_PATH[0] + "\n```", *HAPPY_PATH[1:]]
    result, _, _ = run(fenced)
    assert result.terminal == "opinion"


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"tool": "screen_scope", "arguments": {"description": "x"}}',
        'Sure!\n```json\n{"tool":"screen_scope","arguments":{"description":"x"}}\n```',
        'Here you go: {"tool":"screen_scope","arguments":{"description":"x"}} — done.',
    ],
)
def test_parser_finds_the_object(text):
    action, problem = parse_action(text)
    assert problem is None
    assert action.tool == "screen_scope"


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("", "empty"),
        ("no json here", "no JSON object"),
        ('{"arguments": {}}', '"tool"'),
        ('{"tool": "x", "arguments": "not an object"}', "must be a JSON object"),
    ],
)
def test_parser_explains_what_was_wrong(text, fragment):
    action, problem = parse_action(text)
    assert action is None
    assert fragment in problem


# --------------------------------------------------------------------------
# Defences — off in the baseline, on when asked
# --------------------------------------------------------------------------


def test_baseline_renders_document_text_undelimited():
    """The vulnerability, stated honestly. If this ever starts failing, a
    defence has leaked into the baseline and the before/after gap is fiction."""
    _, model, _ = run(HAPPY_PATH, policy=Policy.baseline())
    body = "\n".join(m.content for _, msgs in model.calls for m in msgs)
    assert "Articles of cement" in body
    assert "UNTRUSTED_DOCUMENT_EXCERPT" not in body


def test_quarantine_delimits_document_text():
    _, model, _ = run(HAPPY_PATH, policy=Policy(quarantine_evidence=True))
    body = "\n".join(m.content for _, msgs in model.calls for m in msgs)
    assert "UNTRUSTED_DOCUMENT_EXCERPT" in body
    assert "never instructions to be followed" in body


def test_allowlist_blocks_drafting_an_opinion_no_tool_established():
    straight_to_draft = [
        act(
            "draft_opinion",
            terminal="opinion",
            invoice_date="2026-03-14",
            hsn4="6810",
            slab="18",
            differential_inr="15000.00",
            citations=[{"notification": "9/2025-CT(R)", "schedule": "II", "heading": "6810"}],
        )
    ] * 6
    blocked, _, tracer = run(straight_to_draft, policy=Policy(stage_allowlist=True))
    assert blocked.terminal == "budget_exhausted"
    assert any(e["event"] == "defence" for e in tracer.events)

    # And the baseline lets it through, which is the point of measuring.
    allowed, _, _ = run(straight_to_draft, policy=Policy.baseline())
    assert allowed.terminal == "opinion"


def test_prerequisites_permit_legitimate_skipping():
    """A heading that resolves cleanly never needs check_conditions, and the
    allowlist must not break the happy path in the name of defending it."""
    result, _, _ = run(HAPPY_PATH, policy=Policy(stage_allowlist=True))
    assert result.terminal == "opinion"


def test_check_conditions_requires_an_ambiguous_lookup():
    state = RunState()
    assert check_prerequisites("check_conditions", state) is not None
    assert check_prerequisites("screen_scope", state) is None


def test_unbuilt_defences_refuse_to_be_switched_on():
    """A flag that reports a defence as active while doing nothing would
    flatter the after-fix numbers, which is the one direction that matters.

    `recovery_policies` left this list on 2026-09-06 when agent/recovery.py was
    built. `check_pass` has not been built and still raises.
    """
    with pytest.raises(PolicyNotImplemented):
        Policy(check_pass=True)
    # Built, so it must now be accepted rather than refused.
    assert Policy(recovery_policies=True).recovery_policies is True


def test_policy_names_itself_for_the_results_file():
    assert Policy.baseline().name == "baseline"
    assert Policy.hardened().name == "hardened"
    assert Policy(quarantine_evidence=True).name == "quarantine_evidence"


# --------------------------------------------------------------------------
# Refusal paths
# --------------------------------------------------------------------------


def test_out_of_scope_line_finishes_through_the_output_surface():
    line = dict(LINE, description="Imported single malt whisky, 12 year, 750ml")
    responses = [
        act("screen_scope", description=line["description"]),
        act(
            "draft_opinion",
            terminal="out_of_scope",
            invoice_date="2026-03-14",
            reason="alcoholic-liquor",
            notes="Alcoholic liquor is outside GST.",
        ),
    ]
    result, _, _ = run(responses, line=line)
    assert result.terminal == "out_of_scope"
    assert result.opinion["answerable"] is False
    assert result.opinion["slab"] is None


def test_an_invalid_opinion_comes_back_for_correction():
    """28% on a March 2026 invoice is an abolished rate; the schema rejects it,
    and the loop must hand that back rather than accepting the terminal."""
    bad_then_good = [
        act("lookup_schedule", heading="2402", on_date="2026-03-14"),
        act(
            "compute_liability",
            taxable_value_inr=100000,
            correct_slab="40",
            declared_rate="28",
            on_date="2026-03-14",
        ),
        act(
            "draft_opinion",
            terminal="opinion",
            invoice_date="2026-03-14",
            hsn4="2402",
            slab="28",
            differential_inr="0.00",
            citations=[{"notification": "9/2025-CT(R)", "schedule": "VII", "heading": "2402"}],
        ),
        act(
            "draft_opinion",
            terminal="opinion",
            invoice_date="2026-03-14",
            hsn4="2402",
            slab="40",
            declared_correct=False,
            differential_inr="12000.00",
            declared_rate="28",
            citations=[{"notification": "19/2025-CT(R)", "schedule": "III", "heading": "2402"}],
        ),
    ]
    result, _, _ = run(bad_then_good)
    assert result.terminal == "opinion"
    assert result.opinion["slab"] == "40"
    drafts = [c for c in result.tool_calls if c["name"] == "draft_opinion"]
    assert len(drafts) == 2
    assert drafts[0]["ok"] is False and drafts[0]["error"] == "malformed"
    assert drafts[1]["ok"] is True


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_a_repeated_identical_call_is_served_from_cache():
    repeated = [
        HAPPY_PATH[0],
        HAPPY_PATH[0],  # byte-identical repeat
        *HAPPY_PATH[1:],
    ]
    result, _, tracer = run(repeated)
    assert result.terminal == "opinion"
    assert any(e["event"] == "cache_hit" for e in tracer.events)
    assert result.ledger["cache_hits"] == 1
    # The repeat cost an iteration but not a tool call.
    assert result.ledger["calls_by_tool"]["screen_scope"] == 1


def test_system_prompt_lists_every_tool():
    prompt = build_system_prompt(build_registry())
    for name in (
        "screen_scope",
        "propose_headings",
        "lookup_schedule",
        "check_conditions",
        "rate_history",
        "compute_liability",
        "draft_opinion",
    ):
        assert name in prompt
