"""The seven tools, and the registry that owns them.

One module per tool, each exporting a `SPEC`. `build_registry()` assembles them
in workflow order, and that order is asserted rather than assumed: a tool that
fails to register is a silent hole in the agent's capability — the loop would
simply never call it, and the run would look like a reasoning failure rather
than a missing tool.

Ordering is not cosmetic either. `STAGES` is the sequence the workflow actually
runs in, and it is what the per-step allowlist defence keys on in week 5: a call
that jumps ahead is refusable because the stages are ordered data from day one,
which makes that defence a policy change rather than a refactor.

Nothing here calls a tool. `Registry.invoke` is the one path, and going through
it is what makes the contract in `agent/contract.py` a guarantee.
"""

from __future__ import annotations

from agent.registry import Registry, ToolSpec
from agent.tools import (
    check_conditions,
    compute_liability,
    draft_opinion,
    lookup_schedule,
    propose_headings,
    rate_history,
    screen_scope,
)

#: The seven, in workflow order. The tuple is the source of truth for both the
#: registration order and the stage sequence.
SPECS: tuple[ToolSpec, ...] = (
    screen_scope.SPEC,       # 1. is there anything to classify
    propose_headings.SPEC,   # 2. which heading could it be
    lookup_schedule.SPEC,    # 3. what does the Gazette say, on the invoice date
    check_conditions.SPEC,   # 4. settle an ambiguous heading
    rate_history.SPEC,       # 5. which notification governed, and did it move
    compute_liability.SPEC,  # 6. the arithmetic
    draft_opinion.SPEC,      # 7. the output surface
)

TOOL_NAMES: tuple[str, ...] = tuple(s.name for s in SPECS)

#: Workflow stages in order. Week 5's allowlist refuses a call whose stage is
#: further along than the run has reached.
STAGES: tuple[str, ...] = tuple(s.stage for s in SPECS)

EXPECTED_NAMES: tuple[str, ...] = (
    "screen_scope",
    "propose_headings",
    "lookup_schedule",
    "check_conditions",
    "rate_history",
    "compute_liability",
    "draft_opinion",
)

if TOOL_NAMES != EXPECTED_NAMES:
    raise ImportError(
        f"tool set is {TOOL_NAMES}, expected {EXPECTED_NAMES}. A tool module "
        "changed its SPEC name or the workflow order moved."
    )

if len(set(STAGES)) != len(STAGES):
    raise ImportError(f"two tools share a stage: {STAGES}")


def build_registry() -> Registry:
    """A fresh registry holding all seven tools.

    Fresh rather than shared: `Registry.register` refuses a duplicate name, and
    a module-level singleton would make two suite runs in one process fail on
    the second. The registry is cheap; the ledger, cache and tracer that go with
    it are per-run by design.
    """
    registry = Registry()
    for spec in SPECS:
        registry.register(spec)
    return registry


def stage_index(stage: str) -> int:
    """Where `stage` sits in the workflow. -1 for an unknown stage."""
    return STAGES.index(stage) if stage in STAGES else -1


__all__ = [
    "SPECS",
    "STAGES",
    "TOOL_NAMES",
    "build_registry",
    "stage_index",
    "check_conditions",
    "compute_liability",
    "draft_opinion",
    "lookup_schedule",
    "propose_headings",
    "rate_history",
    "screen_scope",
]
