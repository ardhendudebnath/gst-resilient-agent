"""The eleven failure modes, from the project brief.

Each mode is a small, named perturbation of one tool call. Three phases:

    before   the tool never runs; the mode returns a failure in its place
    around   the tool runs, and the mode wraps the call
    after    the tool runs and the mode alters what comes back

Two rules hold for every mode without exception.

**Every fabricated result is labelled.** Modes return through
`ToolResult.with_chaos(name)`, and the middleware also emits a `chaos` trace
event. An injected failure that looked organic in a trace would make the whole
failure taxonomy fiction, so the labelling is done twice on purpose. The loop
strips the label before the result enters the message history — the agent must
not be able to tell.

**Perturbation is deterministic given a seed.** Every mode takes the run's RNG.
A chaos result you cannot reproduce is a bug report nobody can act on.

### Two modes worth reading the implementation of

`stale` does not fabricate data. It re-runs the real tool with the invoice date
moved to the day before Notification 19/2025 took effect, so the agent receives
a genuinely correct answer to a question nobody asked — 28 % for a heading that
moved to 40 %, read out of the actual Gazette. That is what stale data looks
like in this domain, and it is far more convincing than a made-up number.

`duplicate` should be harmless here, and proving that is the point. Every tool
is a pure function of its arguments, and `compute_liability` takes the running
total as an argument rather than accumulating it, so a response delivered twice
cannot double-count. The brief predicts this failure for week 5; this mode is
how the claim that it cannot happen gets tested rather than asserted.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.contract import Evidence, ToolCall, ToolResult
from chaos import payloads

#: The day before Notification 19/2025 took effect. `stale` reads the schedules
#: as they stood here, which is the most recent date at which a now-superseded
#: answer was the correct one.
STALE_AS_OF = "2026-01-31"

#: Arguments naming the invoice date, across the tools that take one.
DATE_ARGS = ("on_date", "invoice_date", "as_of")

#: Slabs a `contradictory` mode may substitute. Real rates, so the agent cannot
#: reject the contradiction on the grounds that the number is impossible.
PLAUSIBLE_SLABS = ("0", "5", "18", "40")


@dataclass(slots=True)
class ChaosState:
    """Per-run memory. Some modes need to know what came before.

    One state per run, never shared: `contradictory` has to contradict
    something this run actually saw, and a state carried between runs would
    contradict a fact from a different task.
    """

    #: Slabs this run has already been told, by tool name.
    slabs_seen: dict[str, str] = field(default_factory=dict)
    #: Modes fired so far, in order. Recorded on the run.
    fired: list[dict[str, Any]] = field(default_factory=list)
    #: `duplicate` observations: did the second call match the first?
    duplicate_agreements: list[bool] = field(default_factory=list)


Invoker = Callable[[ToolCall], ToolResult]


@dataclass(frozen=True, slots=True)
class Mode:
    name: str
    phase: str  # "before" | "around" | "after"
    describe: str
    #: Only meaningful on a tool that returns verbatim document text.
    needs_evidence: bool = False
    #: Only meaningful on a tool whose arguments carry a date.
    needs_date: bool = False


# --------------------------------------------------------------------------
# before — the tool never runs
# --------------------------------------------------------------------------


def _timeout(call: ToolCall, rng: random.Random, state: ChaosState) -> ToolResult:
    # "Hangs, then errors." The hang is brief by default: a realistic 30 s
    # stall would dominate a 600-run suite, and what is under test is how the
    # agent handles the error, not how long Python can sleep.
    time.sleep(rng.uniform(0.05, 0.2))
    return ToolResult.err(
        "timeout",
        f"{call.name} did not respond within the time limit",
        retryable=True,
    ).with_chaos("timeout")


def _rate_limited(call: ToolCall, rng: random.Random, state: ChaosState) -> ToolResult:
    return ToolResult.err(
        "rate_limited",
        f"{call.name}: 429 Too Many Requests; retry after a short delay",
        retryable=True,
    ).with_chaos("rate_limited")


# --------------------------------------------------------------------------
# around — the tool runs, the mode wraps it
# --------------------------------------------------------------------------


def _slow(
    call: ToolCall, invoke: Invoker, rng: random.Random, state: ChaosState, *, factor: float, cap: float
) -> ToolResult:
    """Correct, but 20x late.

    The brief names this as the less-covered case, and it is: a slow-but-correct
    tool breaks an agent differently from a failing one, because nothing in the
    result says anything is wrong. What breaks is the budget.
    """
    started = time.perf_counter()
    result = invoke(call)
    elapsed = time.perf_counter() - started
    extra = min(elapsed * (factor - 1.0), cap)
    time.sleep(max(0.0, extra))
    return result.with_chaos("slow")


def _duplicate(
    call: ToolCall, invoke: Invoker, rng: random.Random, state: ChaosState
) -> ToolResult:
    """The response arrives twice. Returns the second.

    Both invocations are real. If the two disagree, the tool is not a pure
    function of its arguments and the idempotency claim in `docs/DESIGN.md` §4
    is false for it — which is a finding, recorded on the state rather than
    swallowed.
    """
    first = invoke(call)
    second = invoke(call)
    agreed = json.dumps(dict(first.data), sort_keys=True, default=str) == json.dumps(
        dict(second.data), sort_keys=True, default=str
    )
    state.duplicate_agreements.append(agreed)
    return second.with_chaos("duplicate")


# --------------------------------------------------------------------------
# after — the tool ran, the mode alters the result
# --------------------------------------------------------------------------


def _empty(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """Valid, well-formed, and says nothing.

    Harder than an error: `ok` is true, the shape is right, and an agent that
    only checks `ok` walks straight into concluding that the schedule is silent
    on this heading.
    """
    hollow: dict[str, Any] = {}
    for key, value in result.data.items():
        if isinstance(value, (list, tuple)):
            hollow[key] = []
        elif isinstance(value, dict):
            hollow[key] = {}
        elif isinstance(value, str):
            hollow[key] = ""
        elif isinstance(value, bool):
            hollow[key] = value
        else:
            hollow[key] = None
    return ToolResult(
        ok=True, data=hollow, evidence=(), chaos="empty"
    )


def _partial(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """Half the fields, silently. The rest never existed as far as the agent knows."""
    keys = sorted(result.data)
    if not keys:
        return result.with_chaos("partial")
    keep = max(1, len(keys) // 2)
    kept = set(rng.sample(keys, keep))
    return ToolResult(
        ok=True,
        data={k: v for k, v in result.data.items() if k in kept},
        evidence=result.evidence,
        chaos="partial",
    )


def _wrong_type(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """A string where a number belonged, or the reverse.

    Targets the fields the arithmetic depends on first, because that is where a
    type confusion actually costs something.
    """
    data = dict(result.data)
    numeric = [
        k
        for k, v in data.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    stringy = [k for k, v in data.items() if isinstance(v, str) and v]
    if numeric:
        key = rng.choice(sorted(numeric))
        data[key] = f"{data[key]} (approx.)"
    elif stringy:
        key = rng.choice(sorted(stringy))
        data[key] = len(str(data[key]))
    else:
        return result.with_chaos("wrong_type")
    return ToolResult(
        ok=True, data=data, evidence=result.evidence, chaos="wrong_type"
    )


def _malformed(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """Output that does not conform to the tool's documented shape.

    A caveat worth stating: results travel as Python objects here, not over a
    wire, so a genuinely truncated JSON *payload* is not reachable — the
    transport cannot be corrupted because there is no transport. What is
    reachable, and what this does, is the failure that actually matters to the
    agent: a result whose structure is not the one the tool promised, delivered
    with `ok` true.
    """
    body = json.dumps(dict(result.data), default=str)
    truncated = body[: max(8, len(body) // 2)]
    return ToolResult(
        ok=True,
        data={"__unparsed__": truncated, "note": "response body was not complete"},
        evidence=result.evidence,
        chaos="malformed_json",
    )


def _contradictory(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """A fact incompatible with one this run has already been given.

    Only fires once a slab has been seen, because a contradiction needs
    something to contradict. Recording the earlier value on the state is what
    makes this a genuine inconsistency rather than a random wrong number.
    """
    data = dict(result.data)
    slab = data.get("slab") or data.get("slab_on_date") or data.get("correct_slab")
    if not slab:
        return result.with_chaos("contradictory")

    seen = state.slabs_seen.get("any")
    alternatives = [s for s in PLAUSIBLE_SLABS if s != str(slab)]
    replacement = rng.choice(alternatives)
    for key in ("slab", "slab_on_date", "correct_slab"):
        if key in data and data[key] is not None:
            data[key] = replacement
    data["_contradicts"] = seen
    return ToolResult(
        ok=True, data=data, evidence=result.evidence, chaos="contradictory"
    )


def _injection(
    result: ToolResult, rng: random.Random, state: ChaosState, *, payload_name: str | None
) -> ToolResult:
    """Instruction-shaped text inside a document the agent must read.

    Spliced into `evidence`, never into `data`, because that is where document
    text actually arrives. An injection planted in the trusted channel would be
    testing a scenario that cannot occur.
    """
    if not result.evidence:
        return result.with_chaos("injection")
    payload = (
        payloads.get(payload_name)
        if payload_name
        else payloads.get(rng.choice(list(payloads.PAYLOAD_NAMES)))
    )
    first, *rest = result.evidence
    poisoned = Evidence(
        source=first.source,
        locator=first.locator,
        text=payloads.splice(first.text, payload),
    )
    state.fired.append({"mode": "injection", "payload": payload.name})
    return ToolResult(
        ok=True,
        data=result.data,
        evidence=(poisoned, *rest),
        chaos=f"injection:{payload.name}",
    )


def _stale(result: ToolResult, rng: random.Random, state: ChaosState) -> ToolResult:
    """Placeholder — `stale` is handled in the middleware.

    It needs to re-issue the call with a different date, which means reaching
    the inner dispatcher, which an after-phase mode does not have. Kept here so
    the mode table stays the single list of what exists.
    """
    return result.with_chaos("stale")


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------

MODES: tuple[Mode, ...] = (
    Mode("timeout", "before", "the tool hangs, then errors"),
    Mode("rate_limited", "before", "the tool returns 429"),
    Mode("slow", "around", "the tool returns correctly but 20x slower"),
    Mode("duplicate", "around", "the tool succeeds but the response arrives twice"),
    Mode("empty", "after", "the tool returns valid-but-empty results"),
    Mode("partial", "after", "the tool returns half the expected fields"),
    Mode("wrong_type", "after", "a string where a number was expected"),
    Mode("malformed_json", "after", "the tool returns a broken structure"),
    Mode("contradictory", "after", "two tools return incompatible facts"),
    Mode(
        "stale",
        "after",
        "the tool returns correct-looking but outdated data",
        needs_date=True,
    ),
    Mode(
        "injection",
        "after",
        "the tool returns document content containing instructions",
        needs_evidence=True,
    ),
)

BY_NAME: dict[str, Mode] = {m.name: m for m in MODES}
MODE_NAMES: tuple[str, ...] = tuple(m.name for m in MODES)

BEFORE_FNS: dict[str, Callable[..., ToolResult]] = {
    "timeout": _timeout,
    "rate_limited": _rate_limited,
}

AFTER_FNS: dict[str, Callable[..., ToolResult]] = {
    "empty": _empty,
    "partial": _partial,
    "wrong_type": _wrong_type,
    "malformed_json": _malformed,
    "contradictory": _contradictory,
    "stale": _stale,
}


def get(name: str) -> Mode:
    if name not in BY_NAME:
        raise KeyError(f"unknown chaos mode {name!r}; known: {MODE_NAMES}")
    return BY_NAME[name]


def applicable(mode: Mode, call: ToolCall, returns_evidence: bool) -> bool:
    """Whether this mode can meaningfully perturb this call.

    Checked before a mode is chosen rather than after, so that a run at 25 %
    injection really does perturb a quarter of its calls. Silently no-opping an
    inapplicable mode would quietly lower the rate the results claim.
    """
    if mode.needs_evidence and not returns_evidence:
        return False
    if mode.needs_date and not any(a in call.arguments for a in DATE_ARGS):
        return False
    return True
