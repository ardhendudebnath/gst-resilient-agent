"""Hard limits on a run, set before the first chaos run rather than after the bill.

An unbounded agent loop is two things at once: a bug, because a loop with no
stopping condition is not an algorithm, and an invoice, because chaos testing
deliberately induces the retry behaviour that runs it up. The brief is explicit
that the surprise API bill is a named pitfall, so the caps in `Budget` are the
values pinned in `docs/DESIGN.md` §10 and the defaults here are those values.

Two design choices worth stating:

**Exhaustion is a return value, not an exception.** `Ledger.exceeded()` returns
the name of the bound that broke, or None. The loop turns that into the
terminal state `budget_exhausted`, which the scorer counts as a failure but
reports in its own column — running out of steps and answering incorrectly are
different faults with different fixes, and a table that merges them hides which
one the fix addressed.

**`max_calls_per_tool` exists separately from `max_tool_calls`.** A retry storm
against one flaky tool and a genuinely long path both consume the global
budget, but only the first is a failure of the agent's judgement. Capping per
tool makes the two distinguishable in the trace without any further analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

#: Wall-clock ceiling under the `slow` chaos mode. That mode returns correct
#: results at 20x latency, so the ordinary 120 s cap would turn every slow run
#: into a timeout and measure nothing except the cap itself. Slow-but-correct
#: tools break agents differently from failing ones, and the brief names not
#: testing that as a pitfall — so the budget is raised rather than the mode
#: dropped, and the raised value is recorded on the run.
SLOW_MODE_WALL_CLOCK_S = 600.0


@dataclass(frozen=True, slots=True)
class Budget:
    """The limits for one task. Pinned in docs/DESIGN.md §10."""

    #: Longest legitimate path is 7 tools + 2 recoveries + the draft.
    max_iterations: int = 12
    max_tool_calls: int = 20
    max_calls_per_tool: int = 4
    max_tokens: int = 60_000
    max_wall_clock_s: float = 120.0

    def for_slow_mode(self) -> "Budget":
        """The same budget with the wall clock raised. See SLOW_MODE_WALL_CLOCK_S."""
        return Budget(
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            max_calls_per_tool=self.max_calls_per_tool,
            max_tokens=self.max_tokens,
            max_wall_clock_s=SLOW_MODE_WALL_CLOCK_S,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "max_tokens": self.max_tokens,
            "max_wall_clock_s": self.max_wall_clock_s,
        }


@dataclass(slots=True)
class Ledger:
    """The running tally for one task.

    Every counter is incremented by the loop, never by a tool. A tool that
    could charge its own budget could also decline to.
    """

    budget: Budget = field(default_factory=Budget)
    iterations: int = 0
    tool_calls: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    #: Calls served from the idempotency cache. Counted, but charged against
    #: nothing: a deduplicated repeat did not cost a call, and billing it would
    #: make the cache look like it was consuming the budget it saves.
    cache_hits: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    # -- accounting ------------------------------------------------------

    def note_iteration(self) -> None:
        self.iterations += 1

    def note_tool_call(self, name: str) -> None:
        self.tool_calls += 1
        self.calls_by_tool[name] = self.calls_by_tool.get(name, 0) + 1

    def note_cache_hit(self) -> None:
        self.cache_hits += 1

    def note_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    # -- limits ----------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def exceeded(self) -> str | None:
        """The name of the first bound that broke, or None.

        Checked before issuing work, not after, so the cap is a limit on what
        the run does rather than a description of what it already did.
        """
        b = self.budget
        if self.iterations >= b.max_iterations:
            return "max_iterations"
        if self.tool_calls >= b.max_tool_calls:
            return "max_tool_calls"
        if self.tokens >= b.max_tokens:
            return "max_tokens"
        if self.elapsed_s >= b.max_wall_clock_s:
            return "max_wall_clock_s"
        return None

    def tool_exhausted(self, name: str) -> bool:
        """True when this specific tool has been called its maximum times.

        Distinct from the global cap: this one names the tool the agent is
        stuck on, which is the thing worth knowing.
        """
        return self.calls_by_tool.get(name, 0) >= self.budget.max_calls_per_tool

    def to_json(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "calls_by_tool": dict(self.calls_by_tool),
            "cache_hits": self.cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "elapsed_s": round(self.elapsed_s, 3),
            "budget": self.budget.to_json(),
        }


@dataclass(slots=True)
class SuiteBudget:
    """The stop that prevents an overnight bill across a whole suite run.

    Separate from `Budget` because the failure it guards against is different:
    one task cannot run up a bill, and forty tasks retrying under 50 % failure
    injection very much can. Checked between tasks, so a suite aborts with the
    tasks it finished intact rather than losing the run.
    """

    max_wall_clock_s: float = 45 * 60
    #: Estimated spend ceiling in USD. Abort before the call that would cross
    #: it, not after. Zero disables the check — for suites that call no model.
    max_usd: float = 5.00
    spent_usd: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    def note_spend(self, usd: float) -> None:
        self.spent_usd += usd

    def exceeded(self, *, next_call_usd: float = 0.0) -> str | None:
        if self.elapsed_s >= self.max_wall_clock_s:
            return "suite_wall_clock"
        if self.max_usd > 0 and self.spent_usd + next_call_usd > self.max_usd:
            return "suite_max_usd"
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "max_wall_clock_s": self.max_wall_clock_s,
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 4),
            "elapsed_s": round(self.elapsed_s, 3),
        }
