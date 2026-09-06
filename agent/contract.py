"""What a tool returns, and the rules every tool obeys.

Four rules come from the project brief and one from this domain. All five are
enforced here rather than left to discipline, because each of them is a failure
class that would otherwise be discovered in week 5 at the cost of a rewrite.

1. **Structured errors, never exceptions.** A tool reports failure as data the
   agent can reason about. `ToolResult.err("timeout", retryable=True)` — not a
   traceback that unwinds the loop. `registry.invoke` converts any exception a
   handler leaks into `error="internal"`, so the guarantee holds even for a
   tool that breaks its own contract.

2. **Idempotent.** A tool is a pure function of `(name, arguments)`, or is made
   to behave like one by caching on `call_key`. Calling twice must be
   indistinguishable from calling once.

3. **Every call carries an id.** `ToolCall.call_id` is unique per invocation;
   `ToolCall.key` is the idempotency key, shared by repeat calls with the same
   arguments. Two different things, and conflating them is how deduplication
   quietly stops working.

4. **Bounded.** See `agent/budget.py`.

5. **Trusted and untrusted data are different fields.** `data` is structured
   output this repository's own code computed. `evidence` is verbatim text
   lifted out of a document — a Gazette entry, an advance-ruling excerpt.

   The split exists because in week 5 one of them is an attack surface and the
   other is not, and a defence that cannot tell them apart cannot be applied.
   It matters here before any attacker turns up: advance rulings carry the
   applicant's own rejected contention, argued in the first person and often
   wrong. Whether the agent follows that is measurable precisely because the
   text arrives in a field marked untrusted.

   Nothing in this module *acts* on the distinction. Rendering — delimiting,
   escaping, dropping — is `agent/render.py`, so the defence can be switched
   off and the baseline measured honestly.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

# --------------------------------------------------------------------------
# Error codes
# --------------------------------------------------------------------------

#: The closed set of machine-readable failure codes. Closed on purpose: the
#: agent's recovery policy dispatches on this string (week 6), and a policy
#: table cannot cover a code that a tool invented at runtime.
#:
#: `retryable` is the *default* for the code, not a promise. A tool may
#: override it — a 429 with no Retry-After is retryable, one on a hard quota is
#: not — but it may not invent a code.
ERROR_CODES: dict[str, bool] = {
    # transport / availability — retrying is meaningful
    "timeout": True,
    "rate_limited": True,
    "unavailable": True,
    # the request itself is wrong — retrying it unchanged cannot help
    "bad_argument": False,
    "not_found": False,
    # the tool's own source of truth is wrong or missing. Never retryable:
    # a hash mismatch does not resolve itself, and a tool that answers from a
    # document it cannot verify is worse than one that refuses.
    "source_missing": False,
    "source_mismatch": False,
    # the tool ran but its output could not be trusted
    "malformed": False,
    # the tool's handler raised. Always a bug in this repository.
    "internal": False,
}


class ContractError(RuntimeError):
    """A tool broke the contract in a way that must not be papered over.

    Raised for programming errors — an unknown error code, evidence that is not
    an `Evidence` — never for a tool's own runtime failure, which is data.
    """


# --------------------------------------------------------------------------
# Untrusted text
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """Verbatim text from a document. **Untrusted.**

    Carries its provenance so a citation in the final opinion can be traced to
    a document and a position, and so a defence can quote what it blocked.
    """

    #: The document this came out of, e.g. "09-2025-CTR.pdf".
    source: str
    #: Where in it, human-readable, e.g. "Schedule II, entry near 6810".
    locator: str
    #: The text itself, exactly as it appears. Never cleaned, never summarised
    #: — a defence that only sees tidied text is not tested against the real
    #: thing.
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"source": self.source, "locator": self.locator, "text": self.text}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool's answer. Success and failure are the same type.

    Construct with `ToolResult.ok_` or `ToolResult.err`, never directly — the
    constructors are what enforce the closed error-code set.
    """

    ok: bool
    #: Trusted, structured, computed by our own code.
    data: Mapping[str, Any] = field(default_factory=dict)
    #: Untrusted verbatim text. Empty for tools that read no documents.
    evidence: tuple[Evidence, ...] = ()
    #: A code from ERROR_CODES when ok is False, else None.
    error: str | None = None
    #: Human-readable detail. Goes in the trace and may reach the model.
    message: str | None = None
    #: Whether calling again unchanged could plausibly succeed.
    retryable: bool = False
    #: Set by the chaos middleware when it altered this result, so a trace can
    #: never be misread as an organic failure. Absent on real runs.
    chaos: str | None = None

    # -- constructors ----------------------------------------------------

    @classmethod
    def ok_(
        cls,
        data: Mapping[str, Any] | None = None,
        evidence: tuple[Evidence, ...] | list[Evidence] = (),
    ) -> "ToolResult":
        ev = tuple(evidence)
        for e in ev:
            if not isinstance(e, Evidence):
                raise ContractError(
                    f"evidence must be Evidence instances, got {type(e).__name__}. "
                    "Verbatim document text cannot be passed as a bare string — "
                    "the trusted/untrusted split is the point (see module docstring)."
                )
        return cls(ok=True, data=dict(data or {}), evidence=ev)

    @classmethod
    def err(
        cls,
        code: str,
        message: str | None = None,
        *,
        retryable: bool | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        if code not in ERROR_CODES:
            raise ContractError(
                f"unknown error code {code!r}; the set is closed because the "
                f"recovery policy dispatches on it. Known: {sorted(ERROR_CODES)}"
            )
        return cls(
            ok=False,
            data=dict(data or {}),
            error=code,
            message=message,
            retryable=ERROR_CODES[code] if retryable is None else retryable,
        )

    # -- serialisation ---------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "data": dict(self.data)}
        if self.evidence:
            out["evidence"] = [e.to_json() for e in self.evidence]
        if not self.ok:
            out["error"] = self.error
            out["retryable"] = self.retryable
        if self.message:
            out["message"] = self.message
        if self.chaos:
            out["chaos"] = self.chaos
        return out

    def with_chaos(self, label: str) -> "ToolResult":
        """Tag this result as having been altered by the chaos middleware.

        Kept here rather than in `chaos/` so that *every* path which fabricates
        a result is forced through one labelled constructor. An unlabelled
        injected failure in a trace is indistinguishable from a real one, which
        would make the failure taxonomy fiction.
        """
        return ToolResult(
            ok=self.ok,
            data=self.data,
            evidence=self.evidence,
            error=self.error,
            message=self.message,
            retryable=self.retryable,
            chaos=label,
        )


# --------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------


def call_key(name: str, arguments: Mapping[str, Any]) -> str:
    """The idempotency key for a call: a hash of `(name, arguments)`.

    Canonical JSON — sorted keys, no insignificant whitespace — so that
    `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the same call. They are the
    same call, and a cache that disagrees is a cache that misses on argument
    ordering the model chose arbitrarily.

    Truncated to 16 hex characters. That is 64 bits over a few hundred calls
    per run; the collision probability is not the risk worth engineering
    against, and a short key is readable in a trace.
    """
    blob = json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One invocation. Distinct from the *call* it repeats.

    `call_id` identifies this invocation and is unique. `key` identifies the
    (tool, arguments) pair and is shared with every repeat of it. Deduplication
    keys on `key`; the trace keys on `call_id`. Collapsing the two is how a
    duplicated response stops being visible in the trace at the exact moment
    you need to see it.
    """

    name: str
    arguments: Mapping[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: Which loop iteration issued this. Set by the loop.
    step: int = 0

    @property
    def key(self) -> str:
        return call_key(self.name, self.arguments)

    def to_json(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "step": self.step,
            "name": self.name,
            "arguments": dict(self.arguments),
            "key": self.key,
        }
