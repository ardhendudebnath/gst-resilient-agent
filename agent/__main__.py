"""Audit one invoice line from the command line.

    python -m agent --line examples/inv-0042.json
    python -m agent --line - < line.json
    python -m agent --demo

Prints the opinion and the ledger, and reports where the run's event stream was
written. Read a trace back with `agent.trace.read_trace`, or set TRACE_DIR to
put it somewhere other than `traces/`.

This is the single-line entry point. Running the whole suite at a chaos level
is `python -m suite.run`, which does not exist yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent.budget import Budget
from agent.config import agent_model, load_env
from agent.llm import ModelError, build
from agent.loop import build_system_prompt, run_task
from agent.policy import Policy
from agent.tools import build_registry
from agent.trace import Tracer
from chaos import MODE_NAMES, PAYLOAD_NAMES, wrap

DEMO_LINE = {
    "line_id": "inv-0042",
    "description": (
        "Quartz slabs, 92% crushed quartz bonded with 8% polyester resin, polished"
    ),
    "declared_hsn": "6802",
    "declared_rate": "12",
    "taxable_value_inr": 250000.00,
    "invoice_date": "2026-03-14",
}


def _load_line(spec: str) -> dict:
    if spec == "-":
        return json.loads(sys.stdin.read())
    path = Path(spec)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    # A bare JSON object is convenient enough to be worth supporting.
    return json.loads(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--line", help="path to a JSON invoice line, '-' for stdin, or JSON")
    source.add_argument(
        "--demo", action="store_true", help="run the worked example from docs/DESIGN.md §1"
    )
    parser.add_argument(
        "--policy",
        choices=["baseline", "hardened"],
        default="baseline",
        help="which defences are in force (default: baseline, i.e. none)",
    )
    parser.add_argument("--model", default=None, help=f"model id (default: {agent_model()})")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "enable the model's reasoning mode. Off by default: the chain bills "
            "as output tokens and a 12-turn loop would exhaust the 60k run "
            "budget. Raise --max-tokens-budget if you turn this on."
        ),
    )
    parser.add_argument(
        "--retrieval",
        choices=["keyword", "semantic", "hybrid"],
        default=None,
        help=(
            "candidate retrieval backend (default: keyword). 'semantic' uses "
            "NVIDIA embeddings and needs NVIDIA_API_KEY; it degrades to keyword "
            "and says so if unavailable."
        ),
    )
    parser.add_argument(
        "--chaos",
        type=float,
        default=0.0,
        metavar="RATE",
        help="failure-injection rate, 0..1. The suite runs 0, 0.10, 0.25, 0.50.",
    )
    parser.add_argument(
        "--chaos-modes",
        default="",
        help=(
            "comma-separated modes to inject (default: all eleven). "
            "One of: " + ",".join(MODE_NAMES)
        ),
    )
    parser.add_argument(
        "--payload",
        default=None,
        help=(
            "pin one injection payload instead of choosing at random: "
            + ",".join(PAYLOAD_NAMES)
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=1729, help="chaos seed; a seed reproduces a run"
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--max-tokens-budget",
        type=int,
        default=None,
        help="override the per-run token budget (default 60000)",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    load_env()
    if args.retrieval:
        # Set before the registry is built, because the retrieval mode is read
        # per call and a run that changed backend halfway would be two runs.
        os.environ["RETRIEVAL_MODE"] = args.retrieval
    line = DEMO_LINE if args.demo else _load_line(args.line)

    budget = Budget()
    if args.max_iterations is not None or args.max_tokens_budget is not None:
        budget = Budget(
            max_iterations=args.max_iterations or budget.max_iterations,
            max_tool_calls=budget.max_tool_calls,
            max_calls_per_tool=budget.max_calls_per_tool,
            max_tokens=args.max_tokens_budget or budget.max_tokens,
            max_wall_clock_s=budget.max_wall_clock_s,
        )

    try:
        model = build(args.model, thinking=args.thinking)
        model._ensure_client()
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nThe tools, the chaos harness and the tests all run without a key "
            "(`make test`). Only the loop and draft_opinion need one.",
            file=sys.stderr,
        )
        return 2

    registry = build_registry()
    dispatcher = registry
    if args.chaos > 0:
        dispatcher = wrap(
            registry,
            rate=args.chaos,
            modes=(
                tuple(m.strip() for m in args.chaos_modes.split(",") if m.strip())
                or MODE_NAMES
            ),
            payload=args.payload,
            seed=args.seed,
        )

    with Tracer() as tracer:
        result = run_task(
            line,
            model=model,
            dispatcher=dispatcher,
            policy=Policy.hardened() if args.policy == "hardened" else Policy.baseline(),
            budget=budget,
            tracer=tracer,
            # The chaos wrapper cannot render a tool list, so the prompt is
            # built from the registry it wraps. Identical either way, which is
            # what keeps a chaos run comparable to a clean one.
            system_prompt=build_system_prompt(registry),
        )

    if args.json:
        print(json.dumps(result.to_json(), indent=2, ensure_ascii=False, default=str))
        return 0 if result.finished else 1

    print(f"line     {result.line_id or '(unnamed)'}")
    print(f"terminal {result.terminal}" + (f"  ({result.reason})" if result.reason else ""))
    if result.opinion:
        op = result.opinion
        print(f"heading  {op['hsn4']}")
        print(f"slab     {op['slab']}%")
        print(f"delta    Rs {op['differential_inr']}")
        print(f"\n{op['justification']}\n")
        for c in op["citations"]:
            print(f"  cited: {c['notification']}, Schedule {c['schedule']}, heading {c['heading']}")
    print(
        f"\nsteps {result.steps}  tool calls {result.ledger['tool_calls']}  "
        f"cache hits {result.ledger['cache_hits']}  "
        f"tokens {result.ledger['tokens_in']}+{result.ledger['tokens_out']}  "
        f"{result.ledger['elapsed_s']}s"
    )
    if result.justification_source == "template":
        print(
            "note: the justification is the deterministic template — tool 7's "
            "model call did not succeed. The determination above is unaffected."
        )
    if result.model_retries or result.parse_retries:
        # Surfaced rather than buried: a run that quietly absorbed six provider
        # failures is not the same run as one that absorbed none, and a suite
        # whose numbers move with the endpoint's mood is measuring the endpoint.
        print(
            f"recovered from {result.model_retries} provider failure(s), "
            f"{result.parse_retries} unparseable reply(s)"
        )
    print(f"policy {result.policy['name']}   trace {result.trace_path}")

    if args.chaos > 0:
        rep = dispatcher.report.to_json()
        fired = ", ".join(f"{k}×{v}" for k, v in sorted(rep["by_mode"].items())) or "none"
        print(
            f"chaos  configured {rep['configured_rate']:.0%}  "
            f"effective {rep['effective_rate']:.0%}  "
            f"({rep['perturbed_calls']}/{rep['eligible_calls']} calls, "
            f"{rep['skipped_cached']} cached)  seed {args.seed}"
        )
        print(f"       fired: {fired}")
        if rep["duplicate_checks"]:
            held = "held" if rep["idempotency_held"] else "BROKEN"
            print(f"       idempotency {held} over {rep['duplicate_checks']} duplicate call(s)")
    return 0 if result.finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
