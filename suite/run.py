"""Run the task suite at a chaos level, and write a results file.

    python -m suite.run --chaos 0.0    # the baseline everything else is measured against
    python -m suite.run --chaos 0.25 --seed 7
    python -m suite.run --chaos 0.25 --policy hardened
    python -m suite.run --dry-run      # what would run, calling nothing

One command per row of the before/after table. Each writes
`results/<name>.json` holding every scenario's score, the configuration that
produced it, and the secondary metrics — so a published number can be traced to
the run that produced it rather than to a screenshot.

**The suite budget is checked between tasks, not inside them.** A suite that ran
out of money halfway keeps the tasks it finished, and says how many it skipped.
Losing forty completed runs because the forty-first was too expensive is a worse
failure than stopping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.budget import Budget, SuiteBudget
from agent.config import REPO_ROOT, agent_model, load_env, max_run_usd
from agent.llm import Model, ModelError, build
from agent.loop import build_system_prompt, run_task
from agent.policy import Policy
from agent.tools import build_registry
from agent.trace import Tracer
from chaos import MODE_NAMES, wrap
from suite import scenarios as S
from suite.score import Score, score_run, summarise

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "") or REPO_ROOT / "results")


@dataclass(slots=True)
class SuiteRun:
    """Everything needed to reproduce a row of the results table."""

    name: str
    model: str
    policy: dict[str, Any]
    chaos: dict[str, Any]
    retrieval_mode: str
    budget: dict[str, Any]
    scores: list[Score] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    aborted: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "policy": self.policy,
            "chaos": self.chaos,
            "retrieval_mode": self.retrieval_mode,
            "budget": self.budget,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "aborted": self.aborted,
            "skipped": self.skipped,
            "summary": summarise(self.scores),
            "scores": [s.to_json() for s in self.scores],
        }


def run_suite(
    *,
    model: Model,
    chaos_rate: float = 0.0,
    chaos_modes: tuple[str, ...] = MODE_NAMES,
    payload: str | None = None,
    seed: int = 1729,
    policy: Policy | None = None,
    budget: Budget | None = None,
    scenario_list: list[S.Scenario] | None = None,
    suite_budget: SuiteBudget | None = None,
    name: str = "",
    trace_dir: Path | None = None,
    on_progress: Any = None,
    concurrency: int = 1,
    trace: bool = True,
) -> SuiteRun:
    """Run every scenario once and score it. Never raises for a task failure."""
    from agent import retrieval

    policy = policy or Policy.baseline()
    budget = budget or Budget()
    tasks = scenario_list if scenario_list is not None else S.all_scenarios()
    stop = suite_budget or SuiteBudget(max_usd=max_run_usd())

    registry = build_registry()
    system_prompt = build_system_prompt(registry)

    run = SuiteRun(
        name=name or f"chaos{int(chaos_rate * 100):02d}-{policy.name}",
        model=getattr(model, "model", "?"),
        policy=policy.to_json(),
        chaos={
            "rate": chaos_rate,
            "modes": list(chaos_modes),
            "payload": payload,
            "seed": seed,
        },
        retrieval_mode=retrieval.retrieval_mode(),
        budget=budget.to_json(),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    def run_one(i: int, scenario: S.Scenario) -> Score:
        dispatcher: Any = registry
        chaos_dispatcher = None
        if chaos_rate > 0:
            # A fresh dispatcher per task: the chaos state (`contradictory`
            # needs a fact to contradict) is per-run by design, and one carried
            # between tasks would contradict a fact from a different invoice.
            chaos_dispatcher = wrap(
                registry,
                rate=chaos_rate,
                modes=chaos_modes,
                payload=payload,
                # Seeded per scenario, so a task's injections do not depend on
                # how many calls the previous task happened to make — which is
                # also what makes concurrency safe to turn on without changing
                # what gets injected.
                seed=seed + i,
            )
            dispatcher = chaos_dispatcher

        path = (trace_dir / f"{run.name}-{scenario.id}.jsonl") if trace_dir else None
        # `trace=False` keeps events in memory. Tests use it: a default Tracer
        # writes into `traces/`, so the test suite was quietly depositing
        # scripted-model traces in the repository alongside real ones — which
        # then turned up in a diagnostic of a live run and read as a genuine
        # failure mode. Test artefacts must not be indistinguishable from
        # evidence.
        with Tracer(path=path, memory_only=not trace) as tracer:
            result = run_task(
                scenario.line,
                model=model,
                dispatcher=dispatcher,
                policy=policy,
                budget=budget,
                tracer=tracer,
                system_prompt=system_prompt,
            )
        perturbed = chaos_dispatcher.report.perturbed_calls if chaos_dispatcher else 0
        return score_run(scenario, result.to_json(), chaos_perturbations=perturbed)

    if concurrency <= 1:
        for i, scenario in enumerate(tasks, 1):
            if broke := stop.exceeded():
                # Keep what finished. Losing forty completed runs because the
                # forty-first was too expensive is the worse failure.
                run.aborted = broke
                run.skipped = [s.id for s in tasks[i - 1 :]]
                break
            score = run_one(i, scenario)
            run.scores.append(score)
            if on_progress:
                on_progress(i, len(tasks), scenario, score)
    else:
        # Every task already owns its tracer, its idempotency cache and its
        # chaos dispatcher, so tasks share nothing mutable and the only
        # contention is the provider. `Tracer` is documented as not
        # thread-safe; that holds, and is why there is one per task rather than
        # one per suite.
        #
        # The suite budget is checked before *submitting* rather than before
        # each result, so an abort stops new work without discarding work in
        # flight.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for i, scenario in enumerate(tasks, 1):
                if broke := stop.exceeded():
                    run.aborted = broke
                    run.skipped = [s.id for s in tasks[i - 1 :]]
                    break
                futures[pool.submit(run_one, i, scenario)] = scenario
            for future in as_completed(futures):
                scenario = futures[future]
                score = future.result()
                run.scores.append(score)
                done += 1
                if on_progress:
                    on_progress(done, len(futures), scenario, score)

        # Completion order is arbitrary under concurrency; the results file is
        # sorted so two runs of the same suite diff cleanly.
        order = {s.id: n for n, s in enumerate(tasks)}
        run.scores.sort(key=lambda s: order.get(s.scenario_id, 0))

    run.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return run


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _progress(i: int, total: int, scenario: S.Scenario, score: Score) -> None:
    mark = "pass" if score.passed else "FAIL"
    why = "" if score.passed else f"  {score.failure_reason[:70]}"
    print(
        f"  [{i:>3}/{total}] {mark}  {scenario.id:<22} "
        f"{score.actual_terminal:<16} {score.elapsed_s:>6.1f}s{why}",
        flush=True,
    )


def _print_summary(run: SuiteRun) -> None:
    summary = summarise(run.scores)
    print(f"\n=== {run.name} — {run.model} ===")
    print(
        f"policy {run.policy['name']}   retrieval {run.retrieval_mode}   "
        f"chaos {run.chaos['rate']:.0%} (seed {run.chaos['seed']})"
    )
    if run.aborted:
        print(f"ABORTED: {run.aborted} — {len(run.skipped)} scenario(s) not run")

    header = (
        f"{'population':<12}{'n':>4}{'pass':>7}{'rate':>8}{'ex-infra':>10}"
        f"{'sched':>8}{'budget':>8}{'infra':>7}{'stale':>7}"
    )
    print("\n" + header)
    for label in ("overall", "derived", "synthetic"):
        b = summary[label]
        if not b.get("n"):
            continue
        print(
            f"{label:<12}{b['n']:>4}{b['passed']:>7}{b['pass_rate']:>8.1%}"
            f"{b['pass_rate_excl_infra']:>10.1%}{b['schema_ok']:>8.1%}"
            f"{b['budget_exhausted']:>8}{b['infrastructure_failures']:>7}"
            f"{b['asserted_abolished']:>7}"
        )

    b = summary["overall"]
    print(
        f"\nsteps {b['mean_steps']}  tool calls {b['mean_tool_calls']}  "
        f"{b['mean_elapsed_s']}s/run  tokens {b['tokens_in']}+{b['tokens_out']}"
    )
    print(
        f"recovered: {b['model_retries']} provider failure(s), "
        f"{b['parse_retries']} unparseable reply(s), "
        f"{b['templated_justifications']} templated justification(s)"
    )
    if b["chapter_only_credit"]:
        print(
            f"partial credit (right chapter, wrong heading): "
            f"{b['chapter_only_credit']} — recorded, not counted as a pass"
        )

    if summary["failures"]:
        print(f"\nfailures ({len(summary['failures'])}):")
        for f in summary["failures"][:15]:
            tag = "syn" if f["synthetic"] else "der"
            print(f"  [{tag}] {f['scenario_id']:<22} {f['why'][:88]}")
        if len(summary["failures"]) > 15:
            print(f"  ... and {len(summary['failures']) - 15} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="suite.run", description=__doc__)
    parser.add_argument("--chaos", type=float, default=0.0, metavar="RATE")
    parser.add_argument("--chaos-modes", default="")
    parser.add_argument("--payload", default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--policy", choices=["baseline", "hardened"], default="baseline")
    parser.add_argument("--retrieval", choices=["keyword", "semantic", "hybrid"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument("--tag", default="", help="only scenarios carrying this tag")
    parser.add_argument("--derived-only", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--name", default="", help="results file name (default: derived)")
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "scenarios to run at once. The suite is I/O-bound on the provider "
            "and every task owns its tracer, cache and chaos dispatcher, so "
            "this is safe. Default 1; raising it also raises the 503 rate."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would run, call nothing")
    args = parser.parse_args(argv)

    load_env()
    if args.retrieval:
        os.environ["RETRIEVAL_MODE"] = args.retrieval

    tasks = S.load(
        include_derived=not args.synthetic_only,
        include_synthetic=not args.derived_only,
        tag=args.tag,
    )
    if args.limit:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print(json.dumps(S.summary(), indent=2))
        print(f"\nwould run {len(tasks)} scenario(s):")
        for s in tasks:
            print(
                f"  {s.id:<22} expect {s.expect_terminal:<14} "
                f"{'syn' if s.synthetic else 'der'}  {','.join(s.tags)}"
            )
        return 0

    try:
        model = build(args.model)
        model._ensure_client()
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    modes = tuple(m.strip() for m in args.chaos_modes.split(",") if m.strip()) or MODE_NAMES
    print(
        f"running {len(tasks)} scenario(s) on {getattr(model, 'model', '?')} "
        f"at {args.chaos:.0%} chaos, policy {args.policy}"
    )

    run = run_suite(
        model=model,
        chaos_rate=args.chaos,
        chaos_modes=modes,
        payload=args.payload,
        seed=args.seed,
        policy=Policy.hardened() if args.policy == "hardened" else Policy.baseline(),
        scenario_list=tasks,
        suite_budget=SuiteBudget(
            max_usd=args.max_usd if args.max_usd is not None else max_run_usd()
        ),
        name=args.name,
        on_progress=_progress,
        concurrency=args.concurrency,
    )

    _print_summary(run)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run.name}.json"
    out.write_text(
        json.dumps(run.to_json(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nwritten to {out}")
    return 0 if not run.aborted else 1


if __name__ == "__main__":
    raise SystemExit(main())
