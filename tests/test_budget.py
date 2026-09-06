"""Bounds, set before the first chaos run rather than after the bill."""

from __future__ import annotations

from agent.budget import SLOW_MODE_WALL_CLOCK_S, Budget, Ledger, SuiteBudget


def test_defaults_match_the_pinned_design():
    """docs/DESIGN.md §10, as amended by §11.

    If these change, the design note changes with them — which is exactly what
    happened on 2026-09-06: this test caught the wall-clock reversal and the
    reversal was written up before the number moved. That is the whole reason
    the assertion is here rather than in a comment.
    """
    b = Budget()
    assert (b.max_iterations, b.max_tool_calls, b.max_calls_per_tool) == (12, 20, 4)
    assert b.max_tokens == 60_000
    # Was 120 s. Raised because it bounded the provider rather than the agent:
    # the first full baseline ended budget_exhausted on 15 of 15 derived
    # scenarios, all on this bound, while iterations and tokens never tripped.
    assert b.max_wall_clock_s == 600.0


def test_a_fresh_ledger_is_within_budget():
    assert Ledger().exceeded() is None


def test_iterations_are_bounded():
    led = Ledger(budget=Budget(max_iterations=2))
    led.note_iteration()
    assert led.exceeded() is None
    led.note_iteration()
    assert led.exceeded() == "max_iterations"


def test_tool_calls_are_bounded():
    led = Ledger(budget=Budget(max_tool_calls=1))
    led.note_tool_call("lookup_schedule")
    assert led.exceeded() == "max_tool_calls"


def test_tokens_are_bounded():
    led = Ledger(budget=Budget(max_tokens=100))
    led.note_tokens(60, 41)
    assert led.exceeded() == "max_tokens"


def test_per_tool_exhaustion_names_the_tool():
    """A retry storm on one tool and a long path are different faults."""
    led = Ledger(budget=Budget(max_calls_per_tool=2))
    led.note_tool_call("lookup_schedule")
    led.note_tool_call("lookup_schedule")
    led.note_tool_call("rate_history")
    assert led.tool_exhausted("lookup_schedule")
    assert not led.tool_exhausted("rate_history")
    # And the global budget is untouched, so the two are distinguishable.
    assert led.exceeded() is None


def test_cache_hits_are_counted_but_not_charged():
    """A deduplicated repeat did not cost a call; billing it would make the
    cache look like it consumes the budget it saves."""
    led = Ledger(budget=Budget(max_tool_calls=2))
    led.note_tool_call("double")
    for _ in range(10):
        led.note_cache_hit()
    assert led.cache_hits == 10
    assert led.tool_calls == 1
    assert led.exceeded() is None


def test_slow_mode_raises_only_the_wall_clock():
    """Slow-but-correct tools break agents differently from failing ones, so
    the mode is kept and the clock is raised — not the other way round."""
    b = Budget().for_slow_mode()
    assert b.max_wall_clock_s == SLOW_MODE_WALL_CLOCK_S
    assert b.max_iterations == Budget().max_iterations
    assert b.max_tokens == Budget().max_tokens


def test_suite_budget_stops_before_the_call_that_would_cross_it():
    suite = SuiteBudget(max_usd=1.00)
    suite.note_spend(0.90)
    assert suite.exceeded() is None
    assert suite.exceeded(next_call_usd=0.20) == "suite_max_usd"


def test_suite_cost_check_can_be_disabled_for_model_free_suites():
    suite = SuiteBudget(max_usd=0.0)
    suite.note_spend(1000.0)
    assert suite.exceeded() is None


def test_ledger_serialises_its_budget_with_it():
    """A results file whose caps are not recorded cannot be compared to another."""
    j = Ledger().to_json()
    assert j["budget"]["max_iterations"] == 12
