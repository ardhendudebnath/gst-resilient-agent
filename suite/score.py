"""Scoring. The criteria are pinned in docs/DESIGN.md §5 and not negotiable here.

A task **passes** if and only if every one of these holds:

1. the terminal state equals the expected terminal state;
2. if `opinion`: `hsn4` exact-matches gold, `slab` exact-matches gold, and
   `differential_inr` matches the independently computed value to the paisa;
3. if `unanswerable`: the reason code matches gold's;
4. the final object validates against the output schema;
5. the run finished inside every budget;
6. no abolished slab is asserted as current anywhere in the output.

Criterion 6 is a **domain safety property, not an accuracy metric**, and it gets
its own column. An agent that gets the slab right while reciting a dead schedule
in its justification has failed in the way this domain actually fails, and
averaging that into an accuracy number would hide it.

**Partial credit is recorded and never counted as a pass.** Chapter-level
agreement — the right chapter, the wrong heading — goes in the results file for
diagnosis, because "wrong by a chapter" and "wrong by a mile" are different
failures with different fixes, and an accuracy column cannot tell them apart.

### Two things this scorer will not do

It will not average derived and synthetic scenarios into one rate. Twenty-eight
real classification problems and sixteen scenarios written to exercise the
author's own branches are not the same population.

It will not treat a refusal as a pass because refusing is safe. An
`unanswerable` on a line that has a determinable rate is a miss, and it is
counted as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from agent.opinion import Opinion, stale_rate_mentions, validate
from suite.scenarios import Scenario


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


@dataclass(slots=True)
class Score:
    """One scenario's outcome, with every criterion recorded separately.

    Separately, because a single boolean cannot tell you whether the agent got
    the heading wrong, the arithmetic wrong, or simply ran out of steps — and
    those need different fixes.
    """

    scenario_id: str
    synthetic: bool
    tags: tuple[str, ...] = ()

    passed: bool = False
    terminal_ok: bool = False
    schema_ok: bool = False
    within_budget: bool = False

    hsn4_ok: bool | None = None
    slab_ok: bool | None = None
    differential_ok: bool | None = None
    reason_ok: bool | None = None

    #: Criterion 6, reported on its own. True means the output asserted a rate
    #: that did not exist on the invoice date.
    asserted_abolished: bool = False
    abolished_detail: list[dict[str, Any]] = field(default_factory=list)

    #: Partial credit. Recorded, never counted.
    chapter_ok: bool | None = None

    #: True when the run died because the provider did, not because the agent
    #: did — `model_unavailable` after exhausting retries, or a hard
    #: `model_error`. Still a failure, and still counted in `pass_rate`, but
    #: reported separately and excluded from `pass_rate_excl_infra`.
    #:
    #: Added after two runs of the same suite were compared across a 2.5x
    #: difference in provider failures (10 against 25) and one 429 killed a
    #: scenario outright. Attributing that to a policy would be the same error
    #: as charging a 503 to the agent's iteration budget, which this project
    #: has now made twice.
    infrastructure_failure: bool = False

    expected_terminal: str = ""
    actual_terminal: str = ""
    expected: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""

    #: Secondary metrics, gated on nothing.
    steps: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model_retries: int = 0
    parse_retries: int = 0
    justification_source: str | None = None
    chaos_perturbations: int = 0

    #: What the recovery policy did on this run, or None when it was off.
    #: Without this the results file could not say whether a hardened suite
    #: improved because it reasoned better or because it silently retried, and
    #: those are different claims. The chaos ladder was run once without it and
    #: the +10.8 points could not be attributed to a mechanism.
    recovery_retries: int = 0
    recovery_actions: dict[str, int] = field(default_factory=dict)

    #: Which chaos modes actually fired on THIS scenario. A failure is only
    #: attributable to an injected mode if the mode is recorded next to it —
    #: an aggregate over the suite cannot say which run met what.
    chaos_modes: dict[str, int] = field(default_factory=dict)

    #: False if a tool returned different data for identical arguments under
    #: the `duplicate` mode. None when the mode never fired. The brief predicts
    #: double-counting as the week-5 failure; this is the field that would show
    #: it, and a suite that never records it cannot claim the failure is absent.
    idempotency_held: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "synthetic": self.synthetic,
            "tags": list(self.tags),
            "passed": self.passed,
            "terminal_ok": self.terminal_ok,
            "schema_ok": self.schema_ok,
            "within_budget": self.within_budget,
            "hsn4_ok": self.hsn4_ok,
            "slab_ok": self.slab_ok,
            "differential_ok": self.differential_ok,
            "reason_ok": self.reason_ok,
            "asserted_abolished": self.asserted_abolished,
            "abolished_detail": self.abolished_detail,
            "chapter_ok": self.chapter_ok,
            "infrastructure_failure": self.infrastructure_failure,
            "expected_terminal": self.expected_terminal,
            "actual_terminal": self.actual_terminal,
            "expected": self.expected,
            "actual": self.actual,
            "failure_reason": self.failure_reason,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "elapsed_s": self.elapsed_s,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model_retries": self.model_retries,
            "parse_retries": self.parse_retries,
            "justification_source": self.justification_source,
            "chaos_perturbations": self.chaos_perturbations,
            "recovery_retries": self.recovery_retries,
            "recovery_actions": dict(self.recovery_actions),
            "chaos_modes": dict(self.chaos_modes),
            "idempotency_held": self.idempotency_held,
        }


def score_run(
    scenario: Scenario,
    result: dict[str, Any],
    *,
    chaos_perturbations: int = 0,
    chaos_report: dict[str, Any] | None = None,
) -> Score:
    """Grade one finished run against its scenario.

    `chaos_report` is the per-scenario `ChaosReport.to_json()`. Taken whole
    rather than as a single count, because a failure is only attributable to an
    injected mode if the mode is recorded beside it.
    """
    opinion = result.get("opinion") or {}
    terminal = str(result.get("terminal") or "")
    ledger = result.get("ledger") or {}
    recovery = result.get("recovery") or {}
    chaos_report = chaos_report or {}

    s = Score(
        scenario_id=scenario.id,
        synthetic=scenario.synthetic,
        tags=scenario.tags,
        expected_terminal=scenario.expect_terminal,
        actual_terminal=terminal,
        steps=int(result.get("steps") or 0),
        tool_calls=int(ledger.get("tool_calls") or 0),
        elapsed_s=float(ledger.get("elapsed_s") or 0.0),
        tokens_in=int(ledger.get("tokens_in") or 0),
        tokens_out=int(ledger.get("tokens_out") or 0),
        model_retries=int(result.get("model_retries") or 0),
        parse_retries=int(result.get("parse_retries") or 0),
        justification_source=result.get("justification_source"),
        chaos_perturbations=chaos_perturbations or int(
            chaos_report.get("perturbed_calls") or 0
        ),
        recovery_retries=int(recovery.get("retries") or 0),
        recovery_actions=dict(recovery.get("by_action") or {}),
        chaos_modes=dict(chaos_report.get("by_mode") or {}),
        # None, not False, when `duplicate` never fired: "no disagreement was
        # observed" and "no check was made" are different statements and only
        # one of them supports a claim.
        idempotency_held=(
            bool(chaos_report.get("idempotency_held"))
            if chaos_report.get("duplicate_checks")
            else None
        ),
        expected={
            "terminal": scenario.expect_terminal,
            "hsn4": scenario.expect_hsn4,
            "slab": scenario.expect_slab,
            "reason": scenario.expect_reason,
            "differential_inr": scenario.expect_differential_inr,
        },
        actual={
            "terminal": terminal,
            "hsn4": opinion.get("hsn4"),
            "slab": opinion.get("slab"),
            "reason": opinion.get("reason"),
            "differential_inr": opinion.get("differential_inr"),
        },
    )

    # -- 5. bounds -------------------------------------------------------
    s.within_budget = terminal != "budget_exhausted"
    if not s.within_budget:
        why = str(result.get("reason") or "")
        s.failure_reason = f"budget_exhausted: {why}"
        # The provider died, not the agent. Marked so a policy comparison run
        # under a degraded endpoint can be read honestly rather than being
        # attributed to the policy.
        s.infrastructure_failure = why.startswith(
            ("model_unavailable", "model_error")
        )

    # -- 1. terminal -----------------------------------------------------
    s.terminal_ok = terminal == scenario.expect_terminal
    if not s.terminal_ok and s.within_budget and not s.failure_reason:
        s.failure_reason = (
            f"terminal {terminal!r}, expected {scenario.expect_terminal!r}"
        )

    # -- 4. schema -------------------------------------------------------
    if not opinion:
        # A run with no opinion cannot have produced a valid one. Only a
        # budget-exhausted run legitimately has none.
        s.schema_ok = False
        if s.within_budget and not s.failure_reason:
            s.failure_reason = "no opinion object was produced"
    else:
        try:
            when = date.fromisoformat(str(opinion.get("invoice_date") or "")[:10])
        except ValueError:
            when = date.fromisoformat(scenario.invoice_date)
        problems = validate(Opinion.from_json(opinion), invoice_date=when)
        s.schema_ok = not problems
        if problems and not s.failure_reason:
            s.failure_reason = "schema: " + "; ".join(problems[:2])

        # -- 6. domain safety, on its own column -------------------------
        findings = stale_rate_mentions(str(opinion.get("justification") or ""), when)
        slab = opinion.get("slab")
        if slab is not None:
            from agent.opinion import slab_valid_on

            if not slab_valid_on(str(slab), when):
                findings.append(
                    {
                        "rate": str(slab),
                        "abolished_on": "see agent.gazette.SLAB_ABOLISHED_ON",
                        "excerpt": "asserted as the applicable slab",
                    }
                )
        s.asserted_abolished = bool(findings)
        s.abolished_detail = findings

    # -- 2 / 3. the answer itself ----------------------------------------
    if scenario.expect_terminal == "opinion":
        s.hsn4_ok = str(opinion.get("hsn4") or "") == str(scenario.expect_hsn4 or "")
        s.slab_ok = str(opinion.get("slab") or "") == str(scenario.expect_slab or "")

        got = _decimal(opinion.get("differential_inr"))
        want = _decimal(scenario.expect_differential_inr)
        s.differential_ok = got is not None and want is not None and got == want

        # Partial credit: right chapter, wrong heading.
        actual_h = str(opinion.get("hsn4") or "")
        want_h = str(scenario.expect_hsn4 or "")
        s.chapter_ok = (
            len(actual_h) == 4 and len(want_h) == 4 and actual_h[:2] == want_h[:2]
        )

        if s.terminal_ok and not s.failure_reason:
            if not s.hsn4_ok:
                s.failure_reason = (
                    f"heading {opinion.get('hsn4')!r}, expected "
                    f"{scenario.expect_hsn4!r}"
                    + (" (right chapter)" if s.chapter_ok else "")
                )
            elif not s.slab_ok:
                s.failure_reason = (
                    f"slab {opinion.get('slab')!r}, expected {scenario.expect_slab!r}"
                )
            elif not s.differential_ok:
                s.failure_reason = (
                    f"differential {opinion.get('differential_inr')!r}, expected "
                    f"{scenario.expect_differential_inr!r}"
                )
    else:
        s.reason_ok = str(opinion.get("reason") or "") == str(
            scenario.expect_reason or ""
        )
        if s.terminal_ok and not s.reason_ok and not s.failure_reason:
            s.failure_reason = (
                f"reason {opinion.get('reason')!r}, expected {scenario.expect_reason!r}"
            )

    # -- the conjunction -------------------------------------------------
    checks = [s.terminal_ok, s.schema_ok, s.within_budget, not s.asserted_abolished]
    if scenario.expect_terminal == "opinion":
        checks += [bool(s.hsn4_ok), bool(s.slab_ok), bool(s.differential_ok)]
    else:
        checks.append(bool(s.reason_ok))
    s.passed = all(checks)

    if s.passed:
        s.failure_reason = ""
    elif not s.failure_reason and s.asserted_abolished:
        s.failure_reason = "asserted an abolished rate as current"
    return s


# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _merge_counts(dicts: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for d in dicts:
        for key, value in (d or {}).items():
            merged[key] = merged.get(key, 0) + int(value)
    return dict(sorted(merged.items(), key=lambda kv: -kv[1]))


def _bucket(scores: list[Score]) -> dict[str, Any]:
    n = len(scores)
    if not n:
        return {"n": 0}
    infra = sum(s.infrastructure_failure for s in scores)
    # Two rates, both reported, neither hidden. `pass_rate` is what the agent
    # achieved on the day; `pass_rate_excl_infra` is what it achieved on the
    # scenarios the provider let it attempt. Comparing two policies across
    # different endpoint conditions needs the second, and quoting only the
    # second would be flattering — so both are always present.
    return {
        "n": n,
        "passed": sum(s.passed for s in scores),
        "pass_rate": _rate(sum(s.passed for s in scores), n),
        "infrastructure_failures": infra,
        "pass_rate_excl_infra": _rate(sum(s.passed for s in scores), n - infra),
        "terminal_ok": _rate(sum(s.terminal_ok for s in scores), n),
        "schema_ok": _rate(sum(s.schema_ok for s in scores), n),
        "within_budget": _rate(sum(s.within_budget for s in scores), n),
        "budget_exhausted": sum(not s.within_budget for s in scores),
        "asserted_abolished": sum(s.asserted_abolished for s in scores),
        "chapter_only_credit": sum(
            1 for s in scores if s.hsn4_ok is False and s.chapter_ok
        ),
        "mean_steps": round(sum(s.steps for s in scores) / n, 2),
        "mean_tool_calls": round(sum(s.tool_calls for s in scores) / n, 2),
        "mean_elapsed_s": round(sum(s.elapsed_s for s in scores) / n, 2),
        "tokens_in": sum(s.tokens_in for s in scores),
        "tokens_out": sum(s.tokens_out for s in scores),
        "model_retries": sum(s.model_retries for s in scores),
        "parse_retries": sum(s.parse_retries for s in scores),
        "templated_justifications": sum(
            1 for s in scores if s.justification_source == "template"
        ),
        # -- what chaos and recovery actually did -------------------------
        "chaos_perturbations": sum(s.chaos_perturbations for s in scores),
        "chaos_by_mode": _merge_counts(s.chaos_modes for s in scores),
        "recovery_retries": sum(s.recovery_retries for s in scores),
        "recovery_by_action": _merge_counts(s.recovery_actions for s in scores),
        # Three-valued on purpose. True means the duplicate mode fired and
        # every repeated call agreed; False means a tool returned different
        # data for identical arguments, which is the failure the brief predicts
        # and would be a finding; None means the check never ran, which is not
        # evidence of anything.
        "idempotency_held": (
            None
            if all(s.idempotency_held is None for s in scores)
            else all(s.idempotency_held is not False for s in scores)
        ),
        "idempotency_checks": sum(
            1 for s in scores if s.idempotency_held is not None
        ),
    }


def summarise(scores: Iterable[Score]) -> dict[str, Any]:
    """Overall, and split by population.

    The split is not optional. Derived and synthetic scenarios measure
    different things, and a headline that averaged them would be a number about
    nothing in particular.
    """
    scores = list(scores)
    derived = [s for s in scores if not s.synthetic]
    synthetic = [s for s in scores if s.synthetic]

    by_terminal: dict[str, dict[str, int]] = {}
    for s in scores:
        row = by_terminal.setdefault(
            s.expected_terminal, {"n": 0, "passed": 0}
        )
        row["n"] += 1
        row["passed"] += int(s.passed)

    return {
        "overall": _bucket(scores),
        "derived": _bucket(derived),
        "synthetic": _bucket(synthetic),
        "by_expected_terminal": by_terminal,
        "failures": [
            {
                "scenario_id": s.scenario_id,
                "synthetic": s.synthetic,
                "expected": s.expected_terminal,
                "actual": s.actual_terminal,
                "why": s.failure_reason,
            }
            for s in scores
            if not s.passed
        ],
    }
