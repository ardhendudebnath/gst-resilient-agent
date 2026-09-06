"""Which defences are switched on. See docs/DESIGN.md §6.

The before/after table only means anything if the "before" is a fair
implementation, so the baseline is **the agent an ordinary careful engineer
writes on the first pass**: bounded loops, structured tool errors, idempotent
tools, tracing. Those are day-one decisions and pretending otherwise would be
building a strawman to knock down.

What the baseline does *not* have is listed in DESIGN.md §6, written before any
measurement existed, so the eventual improvement cannot be read as having been
engineered into the "before". Each flag here is one of those, off by default:

    quarantine_evidence   delimit untrusted document text before it enters the
                          message history (agent/render.py)
    stage_allowlist       refuse a tool call that skips ahead in the workflow
    check_pass            a second look at tool results before they are used
    recovery_policies     failure-class-specific recovery rather than "tell the
                          model and let it decide"

`check_pass` and `recovery_policies` are declared but not yet implemented; the
loop raises if you switch one on, rather than accepting the flag and silently
doing nothing. A defence that reports itself as enabled while doing nothing
would corrupt the measurement in the direction that flatters the result, which
is the one direction that matters.

Turning a flag on is a policy change, not a refactor. That is the whole reason
`stage` has been data on every ToolSpec since the first commit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


class PolicyNotImplemented(NotImplementedError):
    """A defence was switched on that does not exist yet."""


@dataclass(frozen=True, slots=True)
class Policy:
    """The defences in force for one run."""

    quarantine_evidence: bool = False
    stage_allowlist: bool = False
    check_pass: bool = False
    recovery_policies: bool = False

    #: How many malformed model replies to correct before giving up. Not a
    #: defence — a loop with no parse-retry budget would score a formatting
    #: slip as a task failure, which measures the parser rather than the agent.
    #: Counted against iterations either way, so it cannot run away.
    max_parse_retries: int = 2

    def __post_init__(self) -> None:
        unbuilt = [
            name
            for name in ("check_pass", "recovery_policies")
            if getattr(self, name)
        ]
        if unbuilt:
            raise PolicyNotImplemented(
                f"{', '.join(unbuilt)} is not implemented yet. Enabling it would "
                "report a defence as active while nothing happens, which would "
                "flatter the after-fix numbers. Implement it, then switch it on."
            )

    @classmethod
    def baseline(cls) -> "Policy":
        """Every defence off. The number the fixes are measured against."""
        return cls()

    @classmethod
    def hardened(cls) -> "Policy":
        """Every defence that exists, on."""
        return cls(quarantine_evidence=True, stage_allowlist=True)

    def with_(self, **changes: Any) -> "Policy":
        return replace(self, **changes)

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "quarantine_evidence",
                "stage_allowlist",
                "check_pass",
                "recovery_policies",
            )
            if getattr(self, name)
        )

    @property
    def name(self) -> str:
        if not self.enabled:
            return "baseline"
        if self == Policy.hardened():
            return "hardened"
        return "+".join(self.enabled)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quarantine_evidence": self.quarantine_evidence,
            "stage_allowlist": self.stage_allowlist,
            "check_pass": self.check_pass,
            "recovery_policies": self.recovery_policies,
            "max_parse_retries": self.max_parse_retries,
        }
