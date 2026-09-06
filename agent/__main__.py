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
import sys
from pathlib import Path

from agent.budget import Budget
from agent.config import agent_model, load_env
from agent.llm import AnthropicModel, ModelError
from agent.loop import run_task
from agent.policy import Policy
from agent.tools import build_registry
from agent.trace import Tracer

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
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    load_env()
    line = DEMO_LINE if args.demo else _load_line(args.line)

    budget = Budget()
    if args.max_iterations is not None:
        budget = Budget(
            max_iterations=args.max_iterations,
            max_tool_calls=budget.max_tool_calls,
            max_calls_per_tool=budget.max_calls_per_tool,
            max_tokens=budget.max_tokens,
            max_wall_clock_s=budget.max_wall_clock_s,
        )

    try:
        model = AnthropicModel(args.model)
        model._ensure_client()
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nThe tools, the chaos harness and the tests all run without a key "
            "(`make test`). Only the loop and draft_opinion need one.",
            file=sys.stderr,
        )
        return 2

    with Tracer() as tracer:
        result = run_task(
            line,
            model=model,
            dispatcher=build_registry(),
            policy=Policy.hardened() if args.policy == "hardened" else Policy.baseline(),
            budget=budget,
            tracer=tracer,
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
    print(f"policy {result.policy['name']}   trace {result.trace_path}")
    return 0 if result.finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
