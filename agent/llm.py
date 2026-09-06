"""The model boundary. One method, one return shape.

A model here knows how to turn a system prompt and a message history into text
and report what it cost. It knows nothing about GST, tools or scoring — that
separation is what lets the chaos harness swap in `ScriptedModel` and test the
loop's recovery behaviour deterministically, with no key and no bill.

**Why `ScriptedModel` is not just a test fixture.** Most of the failure classes
this project is looking for are properties of the *loop*, not the model:
whether a duplicated tool response gets double-counted, whether a retry storm
is bounded, whether a `not_archived` error reaches a refusal. Those need a
model whose output is fixed so the loop is the only thing varying. Testing them
against a live model would measure the model's mood.

The live path exists for the runs that are about the model — the injection
tests especially, where the whole question is whether a real model complies.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from agent.config import agent_model


@dataclass(slots=True)
class Message:
    role: str  # "user" | "assistant"
    content: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class Completion:
    """One model call, with everything the trace needs to grade and price it."""

    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    stop_reason: str | None = None
    #: Set when the call failed. `text` is empty and the loop treats it as a
    #: recoverable step failure rather than a crash — a model that errors has
    #: not answered, and silently dropping the turn would inflate success.
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class Model(Protocol):
    provider: str
    model: str

    def complete(self, system: str, messages: Iterable[Message]) -> Completion: ...


class ModelError(RuntimeError):
    """A configuration problem — a missing key, an unknown model."""


class ScriptedModel:
    """Replays a fixed list of responses, in order.

    Exhausting the script is an error rather than a repeat of the last reply:
    a loop that runs longer than the script expected is exactly the bug these
    tests are looking for, and silently feeding it the last line again would
    hide it.
    """

    provider = "scripted"

    def __init__(self, responses: list[str], *, model: str = "scripted") -> None:
        self._responses = list(responses)
        self._i = 0
        self.model = model
        #: Every (system, messages) pair it was asked with, for assertions
        #: about what the loop actually put in front of the model — which is
        #: how the prompt-quarantine defence is tested.
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, system: str, messages: Iterable[Message]) -> Completion:
        msgs = list(messages)
        self.calls.append((system, msgs))
        if self._i >= len(self._responses):
            return Completion(
                text="",
                model=self.model,
                provider=self.provider,
                error=(
                    f"scripted model exhausted after {len(self._responses)} "
                    f"responses; the loop asked for one more"
                ),
            )
        text = self._responses[self._i]
        self._i += 1
        return Completion(
            text=text,
            model=self.model,
            provider=self.provider,
            tokens_in=sum(len(m.content) // 4 for m in msgs),
            tokens_out=len(text) // 4,
            stop_reason="end_turn",
        )


class AnthropicModel:
    """The live path. Pinned model id, recorded in every result file."""

    provider = "anthropic"

    def __init__(self, model: str | None = None, *, max_tokens: int = 2048) -> None:
        self.model = model or agent_model()
        self.max_tokens = max_tokens
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ModelError(
                "the live model path needs the SDK: "
                "pip install 'gst-resilient-agent[models]'"
            ) from exc
        # A zero-arg client also resolves ANTHROPIC_AUTH_TOKEN and a logged-in
        # profile, so an unset ANTHROPIC_API_KEY does not by itself mean there
        # are no credentials. Let the SDK decide and report its own error.
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise ModelError(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY in .env "
                "(see .env.example), or run the suite with --model scripted."
            )
        self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: Iterable[Message]) -> Completion:
        client = self._ensure_client()
        payload = [m.to_json() for m in messages]
        started = time.perf_counter()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=payload,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are data here
            return Completion(
                text="",
                model=self.model,
                provider=self.provider,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return Completion(
            text=text,
            model=self.model,
            provider=self.provider,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason=resp.stop_reason,
            extra={"request_id": getattr(resp, "_request_id", None)},
        )


def build(spec: str) -> Model:
    """`--model` resolution. 'scripted' is only reachable from tests."""
    if spec in ("anthropic", "live", ""):
        return AnthropicModel()
    if spec.startswith("claude"):
        return AnthropicModel(model=spec)
    raise ModelError(
        f"unknown model spec {spec!r}; use 'anthropic' or a claude model id"
    )
