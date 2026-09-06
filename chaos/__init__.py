"""Chaos testing: the half of this project that is the point.

The agent is the setup. This package sits between the loop and its tools and
breaks them on purpose, eleven ways, at a configured rate, reproducibly — so
that "the agent is reliable" becomes a number with a failure rate attached
rather than an impression from a demo.

    from agent.tools import build_registry
    from chaos import wrap

    dispatcher = wrap(build_registry(), rate=0.25, seed=7)
    result = run_task(line, model=model, dispatcher=dispatcher)
    print(dispatcher.report.to_json())

Three properties everything here is built around:

**The agent cannot tell.** `ChaosDispatcher` has the same `invoke` signature as
`Registry`, and the loop strips the chaos label before a result enters the
message history. An agent that could detect the test would not be under test.

**Every fabricated result is labelled twice** — on the result and as its own
trace event — so no injected failure can ever be mistaken for an organic one
when the taxonomy is written.

**A seed reproduces a run.** A failure you cannot reproduce is an anecdote.
"""

from chaos.middleware import ChaosConfig, ChaosDispatcher, ChaosReport, wrap
from chaos.modes import MODE_NAMES, MODES, ChaosState, Mode
from chaos.payloads import EXFIL_MARKER, PAYLOAD_NAMES, PAYLOADS, compliance

__all__ = [
    "ChaosConfig",
    "ChaosDispatcher",
    "ChaosReport",
    "ChaosState",
    "EXFIL_MARKER",
    "MODES",
    "MODE_NAMES",
    "Mode",
    "PAYLOADS",
    "PAYLOAD_NAMES",
    "compliance",
    "wrap",
]
