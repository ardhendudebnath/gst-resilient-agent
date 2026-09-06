"""Attribution: can a results file say what caused what?

A full chaos ladder was run before these fields existed, and the +10.8 point
gain from hardening could not be attributed to a mechanism — `recovery` never
reached the Score, and the ChaosReport was reduced to a single count on the way
into the results file. A number you cannot attribute is not evidence.
"""

from __future__ import annotations

import json

import pytest

from agent.llm import ScriptedModel
from agent.policy import Policy
from suite import scenarios as S
from suite.run import run_suite
from suite.score import score_run, summarise

pytest.importorskip("pypdf", reason="scenario construction reads the archived corpus")


def scenario(**kw) -> S.Scenario:
    base = dict(
        id="sc-test",
        line={"line_id": "inv-test", "invoice_date": "2026-03-14"},
        expect_terminal="unanswerable",
        expect_reason="rate-fact-absent",
    )
    base.update(kw)
    return S.Scenario(**base)


def result(**kw) -> dict:
    base = {
        "terminal": "unanswerable",
        "opinion": {
            "terminal": "unanswerable", "invoice_date": "2026-03-14",
            "answerable": False, "hsn4": None, "slab": None,
            "differential_inr": None, "reason": "rate-fact-absent",
            "citations": [], "justification": "Cannot be determined.",
        },
        "steps": 4,
        "ledger": {"tool_calls": 4},
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


def test_recovery_reaches_the_score():
    """It never did. `score_run` read model_retries and parse_retries and
    dropped `recovery` on the floor."""
    s = score_run(
        scenario(),
        result(recovery={"retries": 3, "decisions": 5,
                         "by_action": {"retry": 3, "advise": 2, "abort": 0}}),
    )
    assert s.recovery_retries == 3
    assert s.recovery_actions["retry"] == 3
    assert s.recovery_actions["advise"] == 2
    assert s.to_json()["recovery_retries"] == 3


def test_a_baseline_run_reports_no_recovery_rather_than_zero_silently():
    s = score_run(scenario(), result())
    assert s.recovery_retries == 0
    assert s.recovery_actions == {}


def test_recovery_aggregates_across_a_suite():
    scores = [
        score_run(scenario(id="a"), result(recovery={"retries": 2, "by_action": {"retry": 2}})),
        score_run(scenario(id="b"), result(recovery={"retries": 1, "by_action": {"retry": 1, "advise": 4}})),
    ]
    b = summarise(scores)["overall"]
    assert b["recovery_retries"] == 3
    assert b["recovery_by_action"] == {"advise": 4, "retry": 3}


# --------------------------------------------------------------------------
# Chaos attribution
# --------------------------------------------------------------------------


def test_the_modes_that_fired_are_recorded_next_to_the_scenario():
    """A failure is only attributable to an injected mode if the mode is
    recorded beside it. A suite-level aggregate cannot say which run met what."""
    s = score_run(
        scenario(),
        result(),
        chaos_report={"perturbed_calls": 3, "by_mode": {"stale": 2, "timeout": 1}},
    )
    assert s.chaos_perturbations == 3
    assert s.chaos_modes == {"stale": 2, "timeout": 1}


def test_chaos_modes_aggregate_across_a_suite():
    scores = [
        score_run(scenario(id="a"), result(),
                  chaos_report={"perturbed_calls": 2, "by_mode": {"stale": 2}}),
        score_run(scenario(id="b"), result(),
                  chaos_report={"perturbed_calls": 3, "by_mode": {"stale": 1, "empty": 2}}),
    ]
    b = summarise(scores)["overall"]
    assert b["chaos_perturbations"] == 5
    assert b["chaos_by_mode"] == {"stale": 3, "empty": 2}


# --------------------------------------------------------------------------
# Idempotency — three-valued on purpose
# --------------------------------------------------------------------------


def test_no_duplicate_check_reports_none_not_true():
    """"No disagreement was observed" and "no check was made" are different
    statements, and only one of them supports a claim."""
    s = score_run(scenario(), result(), chaos_report={"duplicate_checks": 0})
    assert s.idempotency_held is None
    assert summarise([s])["overall"]["idempotency_held"] is None


def test_a_passing_duplicate_check_is_recorded_as_held():
    s = score_run(
        scenario(), result(),
        chaos_report={"duplicate_checks": 2, "idempotency_held": True},
    )
    assert s.idempotency_held is True
    b = summarise([s])["overall"]
    assert b["idempotency_held"] is True
    assert b["idempotency_checks"] == 1


def test_one_broken_run_makes_the_whole_suite_report_broken():
    """The brief predicts double-counting as the week-5 failure. If a single
    tool ever returns different data for identical arguments, the suite must
    say so rather than averaging it away."""
    good = score_run(scenario(id="a"), result(),
                     chaos_report={"duplicate_checks": 3, "idempotency_held": True})
    bad = score_run(scenario(id="b"), result(),
                    chaos_report={"duplicate_checks": 1, "idempotency_held": False})
    b = summarise([good, bad])["overall"]
    assert b["idempotency_held"] is False
    assert b["idempotency_checks"] == 2


# --------------------------------------------------------------------------
# End to end through the runner
# --------------------------------------------------------------------------


def _act(tool: str, **arguments) -> str:
    return json.dumps({"thought": "t", "tool": tool, "arguments": arguments})


def test_the_runner_puts_all_of_it_in_the_results_file():
    """The gap that mattered: run_suite kept only `perturbed_calls` from the
    ChaosReport, so a whole chaos ladder was unreadable.

    Driven by a script that makes REAL tool calls. An unparseable model never
    reaches `dispatcher.invoke`, so chaos would have nothing to perturb and the
    test would pass against a still-broken pipeline.
    """
    script = [
        _act("screen_scope", description="Copper handi, kitchen use"),
        _act("lookup_schedule", heading="7418", on_date="2026-03-14"),
        _act("rate_history", heading="7418", on_date="2026-03-14"),
        _act(
            "draft_opinion",
            terminal="unanswerable",
            invoice_date="2026-03-14",
            reason="rate-fact-absent",
        ),
    ] * 12

    run = run_suite(
        model=ScriptedModel(script),
        scenario_list=S.all_scenarios()[:3],
        chaos_rate=1.0,
        chaos_modes=("duplicate",),
        seed=5,
        policy=Policy.baseline(),
        name="attribution-test",
        trace=False,
    )
    body = run.to_json()
    json.dumps(body)  # must serialise

    b = body["summary"]["overall"]
    for key in ("chaos_perturbations", "chaos_by_mode", "recovery_retries",
                "recovery_by_action", "idempotency_held", "idempotency_checks"):
        assert key in b, f"{key} missing from the summary"

    assert b["chaos_perturbations"] > 0, "chaos ran but was not recorded"
    assert "duplicate" in b["chaos_by_mode"]
    # Every tool is a pure function of its arguments, so duplicate delivery
    # must be harmless. If this is ever False it is a finding, not a flake.
    assert b["idempotency_held"] is True
    assert all("chaos_modes" in s for s in body["scores"])
