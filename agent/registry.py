"""The tool registry, and the one path through which a tool is ever called.

Every invocation goes through `Registry.invoke`, and that is what makes the
contract in `agent/contract.py` a guarantee rather than a convention:

- **arguments are validated** before the handler sees them, and a bad call
  comes back as `error="bad_argument"` carrying the specific problems, so the
  agent can correct it rather than guess;
- **exceptions are converted**, so a handler that raises produces
  `error="internal"` instead of unwinding the loop. A tool that breaks its own
  contract still cannot break the agent;
- **repeats are deduplicated** on the idempotency key, which is the single
  decision that pre-empts the double-counting failure class;
- **the call is traced**, both the call and its result, with timing.

The chaos middleware wraps this object rather than replacing it — same
signature, so `chaos.middleware.ChaosDispatcher(registry).invoke` is a drop-in
and the agent loop cannot tell which one it holds. That is deliberate: an agent
that could detect it was being tested would not be being tested.

**Division of budget responsibility.** The registry enforces the per-tool cap,
because that is a fact about a tool. The loop enforces iteration, token and
wall-clock caps, because those are facts about a run. Splitting it this way
keeps `invoke` callable from a test with no ledger at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from agent.budget import Ledger
from agent.contract import ContractError, ToolCall, ToolResult
from agent.jsonschema_lite import check_schema, validate
from agent.trace import Tracer

Handler = Callable[..., ToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool, as the agent sees it and as the registry enforces it."""

    name: str
    #: Shown to the model. Says what the tool does and, where it matters, what
    #: it deliberately will not do — `lookup_schedule` refusing to resolve an
    #: ambiguous heading is a feature the agent has to know about to plan
    #: around it.
    description: str
    #: JSON Schema for the arguments, in the subset `jsonschema_lite` enforces.
    parameters: dict[str, Any]
    handler: Handler
    #: True when this tool can return verbatim document text. Used by the
    #: chaos harness to pick injection targets, and by the week-5 defences to
    #: decide what needs delimiting. Declared rather than inferred: a tool that
    #: starts returning evidence must say so.
    returns_evidence: bool = False
    #: Which stage of the workflow this tool belongs to. Recorded now,
    #: *enforced* in week 5 as the per-step allowlist defence — one of the
    #: named OWASP LLM01 mitigations. Kept as data from day one so the
    #: defence is a policy change rather than a refactor.
    stage: str = ""
    #: False when the handler is not a pure function of its arguments and is
    #: only idempotent because the cache makes it so. Recorded because the
    #: residual risk differs: for these, a genuinely changed value within one
    #: run is masked by the cache.
    pure: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "stage": self.stage,
            "returns_evidence": self.returns_evidence,
        }


@dataclass(slots=True)
class IdempotencyCache:
    """Per-run memo of `(tool, arguments) -> result`.

    One run, one cache. Deliberately not shared across runs: the Gazette does
    not change mid-suite, but a cache that outlived a run would also outlive
    the chaos configuration that produced its entries, and a result injected
    with `malformed_json` under 50 % injection would leak into a clean run.
    That would corrupt the baseline, which is the number everything else is
    measured against.

    **Residual risk, stated rather than discovered later:** within one run, a
    tool whose answer legitimately changes between two identical calls is
    masked. None of this agent's seven tools does that — six are pure and the
    seventh reads a hash-pinned document — but a tool that later does would
    need an explicit opt-out, and there is none yet.
    """

    entries: dict[str, ToolResult] = field(default_factory=dict)
    #: Keys seen more than once, with how many times. This is the raw material
    #: for the duplicate-call failure class: a run whose agent re-issues the
    #: same call five times is visible here without reading the trace.
    repeats: dict[str, int] = field(default_factory=dict)

    def get(self, key: str) -> ToolResult | None:
        hit = self.entries.get(key)
        if hit is not None:
            self.repeats[key] = self.repeats.get(key, 1) + 1
        return hit

    def put(self, key: str, result: ToolResult) -> None:
        # Failures are not cached. A timeout is a fact about one attempt, not
        # about the call, and memoising it would turn a transient failure into
        # a permanent one for the rest of the run — which is itself a failure
        # class, and one worth not building in on purpose.
        if result.ok:
            self.entries[key] = result

    def to_json(self) -> dict[str, Any]:
        return {
            "size": len(self.entries),
            "repeated_keys": {k: v for k, v in self.repeats.items() if v > 1},
        }


class Registry:
    """The tools available to one agent."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # -- registration ----------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ContractError(f"tool {spec.name!r} is already registered")
        # Fail here, with a human watching, rather than at call time.
        check_schema(spec.parameters, where=f"{spec.name}.parameters")
        self._specs[spec.name] = spec
        return spec

    def tool(self, **kwargs: Any) -> Callable[[Handler], Handler]:
        """Decorator form. The handler's name is the tool's name by default."""

        def decorate(fn: Handler) -> Handler:
            kwargs.setdefault("name", fn.__name__)
            kwargs.setdefault("description", (fn.__doc__ or "").strip())
            self.register(ToolSpec(handler=fn, **kwargs))
            return fn

        return decorate

    # -- lookup ----------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return sorted(self._specs)

    # -- invocation ------------------------------------------------------

    def invoke(
        self,
        call: ToolCall,
        *,
        ledger: Ledger | None = None,
        tracer: Tracer | None = None,
        cache: IdempotencyCache | None = None,
    ) -> ToolResult:
        """Run one tool call. Never raises for anything a tool did."""
        if tracer is not None:
            tracer.tool_call(call)

        spec = self._specs.get(call.name)
        if spec is None:
            # Not `internal`: the agent asked for a tool that does not exist,
            # which is a fact about its request and something it can correct.
            return self._finish(
                call,
                ToolResult.err(
                    "not_found",
                    f"no tool named {call.name!r}; available: {', '.join(self.names())}",
                ),
                tracer=tracer,
                started=time.perf_counter(),
            )

        started = time.perf_counter()

        if ledger is not None and ledger.tool_exhausted(call.name):
            return self._finish(
                call,
                ToolResult.err(
                    "rate_limited",
                    f"{call.name} has been called "
                    f"{ledger.budget.max_calls_per_tool} times in this run, "
                    "which is its per-run limit",
                    retryable=False,
                ),
                tracer=tracer,
                started=started,
            )

        if problems := validate(dict(call.arguments), spec.parameters):
            return self._finish(
                call,
                ToolResult.err(
                    "bad_argument",
                    "; ".join(problems),
                    data={"problems": problems},
                ),
                tracer=tracer,
                started=started,
            )

        if cache is not None and (hit := cache.get(call.key)) is not None:
            if tracer is not None:
                tracer.cache_hit(call)
            if ledger is not None:
                ledger.note_cache_hit()
            return hit

        if ledger is not None:
            ledger.note_tool_call(call.name)

        try:
            result = spec.handler(**call.arguments)
        except Exception as exc:  # noqa: BLE001 — the whole point is to catch everything
            result = ToolResult.err(
                "internal",
                f"{type(exc).__name__}: {exc}",
                data={"tool": call.name},
            )
        else:
            if not isinstance(result, ToolResult):
                result = ToolResult.err(
                    "internal",
                    f"{call.name} returned {type(result).__name__}, not a ToolResult",
                    data={"tool": call.name},
                )

        if cache is not None:
            cache.put(call.key, result)
        return self._finish(call, result, tracer=tracer, started=started)

    @staticmethod
    def _finish(
        call: ToolCall,
        result: ToolResult,
        *,
        tracer: Tracer | None,
        started: float,
    ) -> ToolResult:
        if tracer is not None:
            tracer.tool_result(
                call, result, duration_ms=int((time.perf_counter() - started) * 1000)
            )
        return result


def build_call(name: str, arguments: Mapping[str, Any] | None = None, *, step: int = 0) -> ToolCall:
    """Small helper so callers do not construct ToolCall with positional args."""
    return ToolCall(name=name, arguments=dict(arguments or {}), step=step)
