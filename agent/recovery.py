"""Recovery policy: what the loop does about a tool failure, per failure class.

Without this, every tool failure takes the same path — the error is rendered
into the message history and the model decides what to do. That is a defensible
baseline and it is what `docs/DESIGN.md` §6 says the baseline is. It is also
wasteful and unreliable in two specific ways the first baseline made visible:

**A transient failure costs a whole model turn.** A `timeout` on
`lookup_schedule` is not a question: the answer is to call it again. Routing
that through the model spends an iteration, a few thousand tokens of resent
history, and a round trip, to arrive at the decision the error code already
implied. At 25 % failure injection that is most of a run's budget.

**An unrecoverable failure gets retried anyway.** `source_mismatch` means the
pinned Gazette is not the document this tool was built against. No number of
retries fixes that, and a model told "the source hash does not match" will
usually try again.

So recovery is a table from error code to action:

    retry    re-issue the identical call, with backoff, without a model turn
    advise   hand the model the error plus guidance specific to that class
    abort    end the run; nothing downstream can be trusted

### Why this is a defence and lives behind a flag

`Policy.recovery_policies` is off in the baseline. The brief's whole shape is a
measured before and after, and DESIGN §6 named this in advance as one of the
things the baseline deliberately lacks. Turning it on is the "after"; the
before is already recorded in `results/baseline-clean.json`.

### Retries are bounded twice

By `max_attempts` per error class, and by the registry's existing
`max_calls_per_tool`. The second is what stops a recovery policy from becoming
the retry storm it exists to prevent — a policy that can retry without limit is
a worse failure than the one it handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: What to do about one failed call.
RETRY = "retry"
ADVISE = "advise"
ABORT = "abort"


@dataclass(frozen=True, slots=True)
class Recovery:
    """The decision, and why — the reason goes in the trace."""

    action: str
    reason: str
    #: Text appended to the tool result before the model sees it. Empty for
    #: `retry`, because the model never sees a retried failure.
    guidance: str = ""
    #: Only meaningful for `retry`.
    max_attempts: int = 0
    backoff: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "guidance": self.guidance,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class Rule:
    action: str
    reason: str
    guidance: str = ""
    max_attempts: int = 0
    backoff: bool = True


#: Error code -> what to do. Keyed on the closed vocabulary in
#: `agent/contract.py`, so a code with no rule here is a gap that shows up as a
#: KeyError in tests rather than as silent default behaviour in production.
RULES: dict[str, Rule] = {
    # --- transient: the identical call may succeed -----------------------
    "timeout": Rule(
        RETRY,
        "a timeout is not a question; the answer is to call again",
        max_attempts=2,
    ),
    "rate_limited": Rule(
        RETRY,
        "429 means wait, not think",
        max_attempts=2,
    ),
    "unavailable": Rule(
        RETRY,
        "an upstream blip resolves itself or it does not; the model cannot help",
        max_attempts=2,
    ),
    # --- the request was wrong; the model has to fix it ------------------
    "bad_argument": Rule(
        ADVISE,
        "only the caller can correct its own arguments",
        guidance=(
            "Fix the arguments and call the same tool again. Do not switch to a "
            "different tool because this one rejected your call — the tool is "
            "working, the call was malformed."
        ),
    ),
    "not_found": Rule(
        ADVISE,
        "a heading that is absent stays absent; widen or decline",
        guidance=(
            "This heading is not in the archived schedules. Do NOT infer a rate "
            "for it. Either propose a different candidate heading from "
            "propose_headings, or finish with terminal 'unanswerable' and "
            "reason 'heading-not-in-schedule'."
        ),
    ),
    "malformed": Rule(
        ADVISE,
        "the output surface rejected the object; the model must correct it",
        guidance=(
            "The object you submitted failed validation. Read the listed "
            "problems, correct exactly those fields, and call draft_opinion "
            "again. Do not change your determination to make validation pass — "
            "if the slab did not exist on the invoice date, the determination "
            "is what is wrong."
        ),
    ),
    # --- the ground truth is not trustworthy; stop -----------------------
    "source_missing": Rule(
        ABORT,
        "no amount of retrying conjures a missing Gazette",
    ),
    "source_mismatch": Rule(
        ABORT,
        "the pinned corpus is not the one this tool was built against; every "
        "rate it could return would be traceable to nothing",
    ),
    # --- our bug ---------------------------------------------------------
    "internal": Rule(
        ADVISE,
        "a handler raised; that is a bug here, not a decision for the model",
        guidance=(
            "That tool failed internally. Try a different approach or finish "
            "with what you have; calling it again with the same arguments will "
            "fail the same way."
        ),
    ),
}


@dataclass(slots=True)
class RecoveryPolicy:
    """Decides, and counts. One instance per run."""

    enabled: bool = True
    #: Retries taken, keyed by (idempotency key, error code). Keyed on the
    #: *call* rather than the tool, so two different lookups that both time out
    #: each get their own budget — and a single call that keeps timing out does
    #: not get an unbounded one.
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)
    #: Every decision taken, for the run record.
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def decide(self, call_key: str, result: Any) -> Recovery:
        """What to do about `result`. Never raises for an unknown code."""
        if result.ok:
            return Recovery(action="", reason="")

        code = result.error or "internal"
        rule = RULES.get(code)
        if rule is None:
            # An unlisted code is a gap in this table, not a licence to invent
            # behaviour. Advising is the conservative choice: it is what the
            # baseline does for everything.
            decision = Recovery(
                ADVISE,
                f"no recovery rule for {code!r}; falling back to the baseline path",
            )
            self._record(call_key, code, decision)
            return decision

        if rule.action == RETRY:
            seen = self.attempts.get((call_key, code), 0)
            if seen >= rule.max_attempts:
                # Exhausted. Hand it to the model rather than aborting: a tool
                # that keeps timing out may not be needed for this line at all.
                decision = Recovery(
                    ADVISE,
                    f"{code} persisted after {seen} retries",
                    guidance=(
                        f"That tool has failed {seen + 1} times with '{code}'. "
                        "Do not keep calling it. Continue with what you have, "
                        "or finish and say what could not be established."
                    ),
                )
                self._record(call_key, code, decision)
                return decision
            self.attempts[(call_key, code)] = seen + 1
            decision = Recovery(
                RETRY,
                rule.reason,
                max_attempts=rule.max_attempts,
                backoff=rule.backoff,
            )
            self._record(call_key, code, decision, attempt=seen + 1)
            return decision

        decision = Recovery(rule.action, rule.reason, guidance=rule.guidance)
        self._record(call_key, code, decision)
        return decision

    def _record(
        self, call_key: str, code: str, decision: Recovery, attempt: int = 0
    ) -> None:
        self.decisions.append(
            {
                "call_key": call_key,
                "error": code,
                "attempt": attempt,
                **decision.to_json(),
            }
        )

    @property
    def retries(self) -> int:
        return sum(self.attempts.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "retries": self.retries,
            "decisions": len(self.decisions),
            "by_action": {
                action: sum(1 for d in self.decisions if d["action"] == action)
                for action in (RETRY, ADVISE, ABORT)
            },
        }


def covered_codes() -> set[str]:
    """Error codes with an explicit rule. Used by a test against ERROR_CODES."""
    return set(RULES)
