"""The per-step allowlist, checked per terminal.

The first hardened run ended `max_iterations` on 7 scenarios, every one a
refusal case, with 49 identical blocked calls: `draft_opinion` demanded a slab
or a terminal screen verdict, and a line that is genuinely undeterminable
produces neither. The defence made the correct answer unreachable for exactly
the scenarios designed to produce it.

Fixing that by allowing *any* refusal would have been worse than the bug: the
`delimiter_escape` payload instructs the agent to finish `out_of_scope`, and a
blanket rule would let it succeed off an unrelated `not_found`. So each
terminal names its own evidence, and both halves are tested here.
"""

from __future__ import annotations

import pytest

from agent.contract import ToolResult
from agent.loop import RunState, check_prerequisites

pytest.importorskip("pypdf", reason="the tools read the archived notifications")


def state_with(*calls) -> RunState:
    return RunState(calls=list(calls))


OK = ToolResult.ok_
ERR = ToolResult.err


# --------------------------------------------------------------------------
# The bug that was measured
# --------------------------------------------------------------------------


def test_a_date_outside_the_archive_permits_unanswerable():
    """sc-prearchive-01/02 hit max_iterations because of this."""
    s = state_with(("rate_history", ERR("not_found", "predates the archive")))
    assert s.has_established_refusal()
    assert check_prerequisites("draft_opinion", s, {"terminal": "unanswerable"}) is None


def test_an_undeterminable_condition_permits_unanswerable():
    """sc-indet-01/02 hit max_iterations because of this."""
    s = state_with(
        ("check_conditions", OK({"outcome": "not_determinable", "reason": "rate-fact-absent"}))
    )
    assert s.has_established_refusal()
    assert check_prerequisites("draft_opinion", s, {"terminal": "unanswerable"}) is None


@pytest.mark.parametrize("outcome", ["absent", "chapter_only"])
def test_a_heading_that_determines_nothing_permits_unanswerable(outcome):
    s = state_with(("lookup_schedule", OK({"outcome": outcome})))
    assert check_prerequisites("draft_opinion", s, {"terminal": "unanswerable"}) is None


def test_no_candidates_at_all_permits_unanswerable():
    s = state_with(("propose_headings", OK({"count": 0, "candidates": []})))
    assert check_prerequisites("draft_opinion", s, {"terminal": "unanswerable"}) is None


def test_an_unanswerable_with_nothing_established_is_still_blocked():
    """Refusing must still be earned. An agent that gives up without a tool
    saying anything is guessing, and guessing a refusal is still guessing."""
    s = state_with(("screen_scope", OK({"verdict": "in_scope"})))
    blocked = check_prerequisites("draft_opinion", s, {"terminal": "unanswerable"})
    assert blocked and "keep working" in blocked


# --------------------------------------------------------------------------
# The defence the fix must not weaken
# --------------------------------------------------------------------------


def test_out_of_scope_needs_screen_scope_and_accepts_no_substitute():
    """The `delimiter_escape` payload instructs the agent to finish
    out_of_scope. A blanket 'any refusal' rule would have let it succeed off an
    unrelated not_found, so this terminal stays narrow."""
    s = state_with(
        ("rate_history", ERR("not_found", "predates the archive")),
        ("check_conditions", OK({"outcome": "not_determinable"})),
        ("lookup_schedule", OK({"outcome": "absent"})),
    )
    assert s.has_established_refusal(), "these do permit 'unanswerable'"
    blocked = check_prerequisites("draft_opinion", s, {"terminal": "out_of_scope"})
    assert blocked, "but they must NOT permit 'out_of_scope'"
    assert "screen_scope" in blocked


def test_out_of_scope_is_permitted_when_screen_scope_actually_said_so():
    s = state_with(
        ("screen_scope", OK({"verdict": "out_of_scope", "reason": "alcoholic-liquor"}))
    )
    assert check_prerequisites("draft_opinion", s, {"terminal": "out_of_scope"}) is None


def test_an_opinion_still_needs_a_tool_established_slab():
    """The tool-misuse injection: a document supplying the rate must not be
    enough to reach an opinion."""
    s = state_with(("check_conditions", OK({"outcome": "not_determinable"})))
    blocked = check_prerequisites("draft_opinion", s, {"terminal": "opinion"})
    assert blocked and "slab established" in blocked


@pytest.mark.parametrize(
    "call",
    [
        ("lookup_schedule", OK({"outcome": "resolved", "slab": "18"})),
        ("check_conditions", OK({"outcome": "resolved", "slab": "5"})),
        ("rate_history", OK({"slab_on_date": "28"})),
    ],
)
def test_an_opinion_is_permitted_once_a_tool_has_established_a_slab(call):
    assert check_prerequisites("draft_opinion", state_with(call), {"terminal": "opinion"}) is None


def test_compute_liability_still_needs_a_slab():
    s = state_with(("propose_headings", OK({"count": 3})))
    blocked = check_prerequisites("compute_liability", s, {})
    assert blocked and "slab" in blocked


def test_check_conditions_still_needs_an_ambiguous_lookup():
    blocked = check_prerequisites("check_conditions", RunState(), {})
    assert blocked and "ambiguous" in blocked


def test_a_missing_terminal_is_treated_as_an_opinion():
    """A malformed draft_opinion call must not slip past the strictest rule by
    omitting the field the rule keys on."""
    blocked = check_prerequisites("draft_opinion", RunState(), {})
    assert blocked and "slab established" in blocked


def test_early_tools_are_never_blocked():
    for tool in ("screen_scope", "propose_headings", "lookup_schedule", "rate_history"):
        assert check_prerequisites(tool, RunState(), {}) is None
