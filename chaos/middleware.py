"""The chaos middleware: a dispatcher that breaks tools on purpose.

Wraps a `Registry` rather than replacing it, with the same `invoke` signature,
so `ChaosDispatcher(registry)` is a drop-in and **the agent loop cannot tell
which one it is holding.** That is not tidiness — an agent that could detect it
was being tested would not be being tested.

### Where chaos sits, and why it matters

    loop ──▶ ChaosDispatcher ──▶ Registry.invoke ──▶ tool
                    │                   │
                    │                   └── validates args, dedupes, caps, traces
                    └── decides whether to perturb this call

**A call the cache will serve is never perturbed.** A repeat call is answered
from memory and never reaches the thing that would have failed, so injecting a
timeout into it would be simulating a failure that cannot physically occur.
This has a consequence worth stating in advance rather than discovering in the
results: deduplication reduces an agent's exposure to chaos, so an agent that
repeats calls is *less* affected by a given injection rate than one that does
not. Effective rate is reported alongside the configured rate for that reason.

### Reproducibility, and its limit

Every decision comes from a seeded RNG keyed on `(seed, tool name, how many
times that tool has been called)`. So a given seed reproduces a given pattern
of failures against a given sequence of calls.

It does **not** guarantee that a baseline run and a hardened run of the same
task see identical injections, because the two agents make different calls in
different orders. That is a real limitation of before/after comparison under
chaos and it is why the tables report rates over a suite rather than diffing
individual runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agent.budget import Ledger
from agent.contract import ToolCall, ToolResult
from agent.registry import IdempotencyCache, Registry
from agent.trace import Tracer
from chaos import modes as M


@dataclass(frozen=True, slots=True)
class ChaosConfig:
    """How much chaos, of what kind, reproducibly."""

    #: Probability that any one eligible call is perturbed. The suite runs
    #: 0.0 / 0.10 / 0.25 / 0.50.
    rate: float = 0.0
    #: Which modes are in play. Defaults to all eleven.
    modes: tuple[str, ...] = M.MODE_NAMES
    seed: int = 1729
    #: `slow` multiplies the tool's real latency by this, capped.
    slow_factor: float = 20.0
    slow_cap_s: float = 5.0
    #: Pin one injection payload instead of choosing at random. The injection
    #: suite runs each payload as its own condition so compliance is countable
    #: per attack shape rather than averaged into one number.
    payload: str | None = None
    #: Restrict perturbation to these tools. Empty means all of them.
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in 0..1, got {self.rate}")
        for name in self.modes:
            M.get(name)  # raises on an unknown mode
        if self.payload is not None:
            from chaos import payloads

            payloads.get(self.payload)

    @property
    def enabled(self) -> bool:
        return self.rate > 0 and bool(self.modes)

    def to_json(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "modes": list(self.modes),
            "seed": self.seed,
            "payload": self.payload,
            "targets": list(self.targets),
            "slow_factor": self.slow_factor,
        }


@dataclass(slots=True)
class ChaosReport:
    """What actually happened, for the results file."""

    configured_rate: float = 0.0
    eligible_calls: int = 0
    perturbed_calls: int = 0
    skipped_cached: int = 0
    by_mode: dict[str, int] = field(default_factory=dict)
    payloads_fired: dict[str, int] = field(default_factory=dict)
    #: False in any entry means a tool returned different data for identical
    #: arguments — the idempotency claim broken, and a finding on its own.
    duplicate_agreements: list[bool] = field(default_factory=list)

    @property
    def effective_rate(self) -> float:
        """Share of eligible calls actually perturbed.

        Reported next to the configured rate because they diverge: cached calls
        are skipped, and some modes do not apply to some tools.
        """
        return self.perturbed_calls / self.eligible_calls if self.eligible_calls else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "configured_rate": self.configured_rate,
            "effective_rate": round(self.effective_rate, 4),
            "eligible_calls": self.eligible_calls,
            "perturbed_calls": self.perturbed_calls,
            "skipped_cached": self.skipped_cached,
            "by_mode": dict(self.by_mode),
            "payloads_fired": dict(self.payloads_fired),
            "idempotency_held": all(self.duplicate_agreements),
            "duplicate_checks": len(self.duplicate_agreements),
        }


class ChaosDispatcher:
    """A `Registry`-shaped object that breaks things on the way through."""

    def __init__(self, inner: Registry, config: ChaosConfig | None = None) -> None:
        self.inner = inner
        self.config = config or ChaosConfig()
        self.state = M.ChaosState()
        self.report = ChaosReport(configured_rate=self.config.rate)
        self._counts: dict[str, int] = {}

    # -- pass-throughs, so the loop can render a tool list from us -------
    def specs(self):  # pragma: no cover - trivial delegation
        return self.inner.specs()

    def names(self):  # pragma: no cover
        return self.inner.names()

    def get(self, name: str):  # pragma: no cover
        return self.inner.get(name)

    def __contains__(self, name: object) -> bool:  # pragma: no cover
        return name in self.inner

    # -- the decision ----------------------------------------------------

    def _rng_for(self, call: ToolCall) -> random.Random:
        ordinal = self._counts.get(call.name, 0)
        return random.Random(f"{self.config.seed}|{call.name}|{ordinal}")

    def _choose_mode(self, call: ToolCall, rng: random.Random) -> M.Mode | None:
        if not self.config.enabled:
            return None
        if self.config.targets and call.name not in self.config.targets:
            return None

        spec = self.inner.get(call.name)
        returns_evidence = bool(spec and spec.returns_evidence)
        eligible = [
            M.get(name)
            for name in self.config.modes
            if M.applicable(M.get(name), call, returns_evidence)
        ]
        if not eligible:
            return None

        self.report.eligible_calls += 1
        if rng.random() >= self.config.rate:
            return None
        return rng.choice(eligible)

    # -- invocation ------------------------------------------------------

    def invoke(
        self,
        call: ToolCall,
        *,
        ledger: Ledger | None = None,
        tracer: Tracer | None = None,
        cache: IdempotencyCache | None = None,
    ) -> ToolResult:
        def passthrough(c: ToolCall) -> ToolResult:
            return self.inner.invoke(c, ledger=ledger, tracer=tracer, cache=cache)

        # A call the cache will answer never reaches the failing thing.
        if cache is not None and call.key in cache.entries:
            self.report.skipped_cached += 1
            return passthrough(call)

        rng = self._rng_for(call)
        mode = self._choose_mode(call, rng)
        self._counts[call.name] = self._counts.get(call.name, 0) + 1

        if mode is None:
            return passthrough(call)

        result = self._apply(mode, call, passthrough, rng)

        self.report.perturbed_calls += 1
        self.report.by_mode[mode.name] = self.report.by_mode.get(mode.name, 0) + 1
        if result.chaos and result.chaos.startswith("injection:"):
            name = result.chaos.split(":", 1)[1]
            self.report.payloads_fired[name] = self.report.payloads_fired.get(name, 0) + 1

        if tracer is not None:
            # Emitted in addition to the label on the result itself. Two places
            # on purpose: a trace that could not distinguish an injected failure
            # from an organic one would make the taxonomy fiction.
            tracer.chaos(
                mode=mode.name,
                target=call.name,
                call_id=call.call_id,
                label=result.chaos,
                seed=self.config.seed,
            )
        return result

    def _apply(
        self,
        mode: M.Mode,
        call: ToolCall,
        passthrough: Any,
        rng: random.Random,
    ) -> ToolResult:
        if mode.phase == "before":
            return M.BEFORE_FNS[mode.name](call, rng, self.state)

        if mode.name == "slow":
            return M._slow(
                call,
                passthrough,
                rng,
                self.state,
                factor=self.config.slow_factor,
                cap=self.config.slow_cap_s,
            )
        if mode.name == "duplicate":
            result = M._duplicate(call, passthrough, rng, self.state)
            self.report.duplicate_agreements = list(self.state.duplicate_agreements)
            return result

        if mode.name == "stale":
            return self._stale(call, passthrough, rng)

        result = passthrough(call)
        if not result.ok:
            # Perturbing a failure is not interesting and would double-count:
            # the run already has a failure to handle.
            return result

        if mode.name == "injection":
            return M._injection(result, rng, self.state, payload_name=self.config.payload)

        self._remember(result)
        return M.AFTER_FNS[mode.name](result, rng, self.state)

    def _stale(self, call: ToolCall, passthrough: Any, rng: random.Random) -> ToolResult:
        """Re-issue the call as if the invoice were dated before the amendment.

        The answer that comes back is *real*: it is what the archived Gazette
        said on 2026-01-31, read by the same tool from the same pinned PDF. No
        number is fabricated. That is what makes it a convincing staleness —
        28 % for a heading that moved to 40 % is not a wrong answer, it is
        last month's right answer.
        """
        shifted = dict(call.arguments)
        for arg in M.DATE_ARGS:
            if arg in shifted:
                shifted[arg] = M.STALE_AS_OF
        stale_call = ToolCall(
            name=call.name, arguments=shifted, call_id=call.call_id, step=call.step
        )
        result = passthrough(stale_call)
        self._remember(result)
        # The date the agent asked about is put back, so the result *claims* to
        # answer the question that was asked. Leaving the shifted date visible
        # would announce the injection.
        data = dict(result.data)
        for arg in M.DATE_ARGS:
            if arg in data and arg in call.arguments:
                data[arg] = call.arguments[arg]
        return ToolResult(
            ok=result.ok,
            data=data,
            evidence=result.evidence,
            error=result.error,
            message=result.message,
            retryable=result.retryable,
            chaos="stale",
        )

    def _remember(self, result: ToolResult) -> None:
        slab = result.data.get("slab") or result.data.get("slab_on_date")
        if slab:
            self.state.slabs_seen["any"] = str(slab)


def wrap(registry: Registry, **kwargs: Any) -> ChaosDispatcher:
    """`wrap(registry, rate=0.25, seed=7)` — the shorthand the suite uses."""
    return ChaosDispatcher(registry, ChaosConfig(**kwargs))
