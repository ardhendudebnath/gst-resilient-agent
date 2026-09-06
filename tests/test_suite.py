"""The task suite: composition, determinism, and that the scorer is strict.

The scorer is the thing every published number passes through, so most of what
is tested here is that it *refuses* to pass runs — a lenient scorer produces a
flattering table and nobody notices until someone reads the traces.
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


def _scenario(**kw) -> S.Scenario:
    base = dict(
        id="sc-test",
        line={
            "line_id": "inv-test",
            "description": "Quartz slabs",
            "declared_hsn": "6802",
            "declared_rate": "12",
            "taxable_value_inr": 250000.00,
            "invoice_date": "2026-03-14",
        },
        expect_terminal="opinion",
        expect_hsn4="6810",
        expect_slab="18",
        expect_differential_inr="15000.00",
    )
    base.update(kw)
    return S.Scenario(**base)


def _result(**kw) -> dict:
    opinion = {
        "terminal": "opinion",
        "line_id": "inv-test",
        "invoice_date": "2026-03-14",
        "hsn4": "6810",
        "slab": "18",
        "answerable": True,
        "declared_correct": False,
        "differential_inr": "15000.00",
        "reason": None,
        "citations": [
            {"notification": "9/2025-CT(R)", "schedule": "II", "heading": "6810"}
        ],
        "justification": "Heading 6810 is rated 18% under Schedule II.",
    }
    opinion.update(kw.pop("opinion", {}))
    base = {
        "terminal": kw.pop("terminal", "opinion"),
        "opinion": opinion,
        "steps": 6,
        "ledger": {"tool_calls": 6, "elapsed_s": 12.0, "tokens_in": 100, "tokens_out": 50},
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_suite_is_in_the_designed_size_range():
    """DESIGN.md §9 targets 40-60 scenarios."""
    all_s = S.all_scenarios()
    assert 40 <= len(all_s) <= 60, f"suite has {len(all_s)} scenarios"


def test_both_populations_are_present_and_labelled():
    derived = S.derived_scenarios()
    synthetic = S.synthetic_scenarios()
    assert derived and synthetic
    assert all(not s.synthetic for s in derived)
    assert all(s.synthetic for s in synthetic)
    assert len({s.id for s in derived + synthetic}) == len(derived) + len(synthetic)


def test_derived_scenarios_never_predate_the_amendment():
    """Project 01 labelled against the current table, so a pre-amendment date
    would make the gold slab wrong. Every such scenario must be synthetic, with
    its expected slab read from the Gazette at that date instead."""
    for s in S.derived_scenarios():
        assert s.invoice_date >= "2026-02-01", (
            f"{s.id} is derived but dated {s.invoice_date}, before the amendment"
        )


def test_construction_is_deterministic_from_the_row_id():
    """The suite is reproducible without storing it."""
    a = {s.id: s.line for s in S.derived_scenarios()}
    b = {s.id: s.line for s in S.derived_scenarios()}
    assert a == b


def test_every_branch_the_golden_set_cannot_reach_has_a_scenario():
    tags = {t for s in S.synthetic_scenarios() for t in s.tags}
    for required in ("out_of_scope", "under_specified", "pre_archive", "boundary",
                     "conditional", "indeterminate"):
        assert required in tags, f"no scenario covers {required}"


def test_expected_differentials_match_the_tool_that_will_compute_them():
    """Computed by calling `compute_liability`, not by reimplementing it.

    An earlier version of this test did the arithmetic itself and disagreed
    with the tool by one paisa on `sc-gst-0008` — Decimal's default is banker's
    rounding and the tool uses ROUND_HALF_UP. A scorer and a tool that round
    differently scatter off-by-one-paisa failures through the suite, and they
    look exactly like agent errors. Deriving the expectation from the tool
    makes that class of drift impossible rather than merely unlikely.
    """
    from decimal import Decimal

    from agent.registry import build_call
    from agent.tools import build_registry

    registry = build_registry()
    for s in S.all_scenarios():
        if s.expect_terminal != "opinion":
            assert s.expect_differential_inr is None, s.id
            continue
        result = registry.invoke(
            build_call(
                "compute_liability",
                {
                    "taxable_value_inr": s.line["taxable_value_inr"],
                    "correct_slab": s.expect_slab,
                    "declared_rate": s.line["declared_rate"],
                    "on_date": s.invoice_date,
                },
            )
        )
        assert result.ok, f"{s.id}: {result.message}"
        assert Decimal(s.expect_differential_inr) == Decimal(
            str(result.data["differential_inr"])
        ), s.id


def test_boundary_scenarios_expect_the_rate_in_force_on_their_date():
    """2402 is 28% before the amendment and 40% after. If this ever collapses
    to one value, the date branch has stopped being exercised."""
    boundary = [s for s in S.all_scenarios() if "boundary" in s.tags]
    assert boundary
    by_date = {(s.expect_hsn4, s.invoice_date < "2026-02-01"): s.expect_slab for s in boundary}
    assert by_date.get(("2402", True)) == "28"
    assert by_date.get(("2402", False)) == "40"


# --------------------------------------------------------------------------
# The scorer refuses things
# --------------------------------------------------------------------------


def test_a_correct_run_passes():
    s = score_run(_scenario(), _result())
    assert s.passed
    assert s.terminal_ok and s.schema_ok and s.within_budget
    assert s.hsn4_ok and s.slab_ok and s.differential_ok
    assert not s.asserted_abolished


def test_a_wrong_heading_fails_even_with_the_right_chapter():
    """Partial credit is recorded and never counted as a pass."""
    s = score_run(_scenario(), _result(opinion={"hsn4": "6802"}))
    assert not s.passed
    assert s.hsn4_ok is False
    assert s.chapter_ok is True
    assert "right chapter" in s.failure_reason


def test_a_differential_off_by_one_paisa_fails():
    """'To the paisa' is the criterion, so it is tested at the paisa."""
    s = score_run(_scenario(), _result(opinion={"differential_inr": "15000.01"}))
    assert not s.passed
    assert s.differential_ok is False


def test_a_refusal_on_an_answerable_line_is_a_miss():
    """Refusing is safe, and safe is not the same as correct."""
    s = score_run(
        _scenario(),
        _result(terminal="unanswerable", opinion={
            "terminal": "unanswerable", "answerable": False, "hsn4": None,
            "slab": None, "differential_inr": None, "reason": "rate-fact-absent",
            "citations": [],
        }),
    )
    assert not s.passed
    assert s.terminal_ok is False


def test_the_wrong_reason_code_fails_an_otherwise_correct_refusal():
    scenario = _scenario(
        expect_terminal="unanswerable",
        expect_hsn4=None,
        expect_slab=None,
        expect_differential_inr=None,
        expect_reason="model-number-only",
    )
    result = _result(terminal="unanswerable", opinion={
        "terminal": "unanswerable", "answerable": False, "hsn4": None, "slab": None,
        "differential_inr": None, "reason": "no-product-kind", "citations": [],
        "justification": "The line does not name a good.",
    })
    s = score_run(scenario, result)
    assert not s.passed
    assert s.terminal_ok is True
    assert s.reason_ok is False


def test_budget_exhaustion_is_a_failure_reported_apart_from_a_wrong_answer():
    s = score_run(
        _scenario(),
        {"terminal": "budget_exhausted", "opinion": None, "reason": "max_iterations",
         "steps": 12, "ledger": {}},
    )
    assert not s.passed
    assert s.within_budget is False
    assert "budget_exhausted" in s.failure_reason


def test_an_abolished_rate_in_the_justification_fails_the_run():
    """Criterion 6: a run can have every field right and still fail, because
    reciting a dead schedule is how this domain actually fails."""
    s = score_run(
        _scenario(),
        _result(opinion={"justification": "These goods attract 28% under Schedule VII."}),
    )
    assert not s.passed
    assert s.asserted_abolished is True
    assert s.hsn4_ok and s.slab_ok and s.differential_ok  # everything else was right


def test_audit_language_about_a_stale_declared_rate_is_not_a_violation():
    """"The supplier declared 12%" is the finding, not an assertion."""
    s = score_run(
        _scenario(),
        _result(opinion={
            "justification": "Heading 6810 is rated 18%. The supplier declared 12%, "
                             "a rate abolished on 2025-09-22."
        }),
    )
    assert s.passed
    assert s.asserted_abolished is False


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_summary_never_averages_the_two_populations():
    scores = [
        score_run(_scenario(id="d1"), _result()),
        score_run(_scenario(id="s1", synthetic=True), _result(opinion={"slab": "5"})),
    ]
    summary = summarise(scores)
    assert summary["derived"]["pass_rate"] == 1.0
    assert summary["synthetic"]["pass_rate"] == 0.0
    assert summary["overall"]["pass_rate"] == 0.5
    assert len(summary["failures"]) == 1


def test_summary_is_json_serialisable():
    scores = [score_run(_scenario(), _result())]
    json.dumps(summarise(scores))


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


def test_runner_scores_every_scenario_it_is_given():
    """Driven by a scripted model that always fails to parse, so no key is
    needed: what is under test is that the runner completes and scores, not
    that the agent is right."""
    tasks = S.all_scenarios()[:3]
    run = run_suite(
        model=ScriptedModel(["not json"] * 200),
        scenario_list=tasks,
        policy=Policy.baseline(),
        name="test-run",
    )
    assert len(run.scores) == 3
    assert run.aborted is None
    body = run.to_json()
    assert body["summary"]["overall"]["n"] == 3
    assert body["chaos"]["rate"] == 0.0
    json.dumps(body)


def test_runner_keeps_finished_tasks_when_the_suite_budget_trips():
    """Losing forty completed runs because the forty-first was too expensive is
    the worse failure."""
    from agent.budget import SuiteBudget

    stop = SuiteBudget(max_wall_clock_s=0.0)  # already exceeded
    run = run_suite(
        model=ScriptedModel(["not json"] * 200),
        scenario_list=S.all_scenarios()[:4],
        suite_budget=stop,
        name="test-abort",
    )
    assert run.aborted == "suite_wall_clock"
    assert len(run.skipped) == 4
