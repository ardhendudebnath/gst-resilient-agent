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

### Providers

`OpenAICompatModel` covers NVIDIA's API catalog, a self-hosted NIM container,
OpenRouter and OpenAI, because all four serve the same chat-completions wire
format. It is **stdlib only**: the request is one POST with a JSON body, and an
SDK dependency here would buy nothing while breaking the promise that this
repository's core runs on a fresh clone.

The default is NVIDIA-hosted `nemotron-3-super-120b-a12b` — the same model
Project 01 benchmarked, which makes the two projects' numbers comparable, and
which has a published NIM container so "you could self-host this" is
demonstrable rather than asserted. That is also the bridge to Project 03.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from agent.config import agent_model

#: Chat-completions endpoints that share a wire format.
ENDPOINTS = {
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    # Filled in from NIM_BASE_URL at construction.
    "nim": "",
}

KEY_VARS = {
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    # A container on your own machine needs no key.
    "nim": "",
}

DEFAULT_NIM_BASE = "http://localhost:8000"

#: Reasoning switch per model family. The three mechanisms are incompatible and
#: sending the wrong one is silently ignored rather than rejected, so a run
#: would proceed with reasoning off while claiming it was on. Recorded per
#: model rather than guessed per provider — learned the hard way in Project 01.
REASONING_STYLES = {"chat_template", "system_toggle", "effort", ""}


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


#: Error prefixes that mean "the same call may succeed shortly". Everything
#: else — 401, 400, 404 — is a configuration problem that retrying cannot fix,
#: and retrying it would burn the run's wall clock to reach the same failure.
#:
#: 503 is on this list because it is the failure this project actually meets:
#: `nemotron-3-ultra-550b-a55b` returns "Service temporarily overloaded" often
#: enough that a single 8-step run hit it three times.
TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "http_408",
    "http_409",
    "http_429",
    "http_500",
    "http_502",
    "http_503",
    "http_504",
    "http_529",
    "TimeoutError",
    "timed out",
    "URLError",
    "ConnectionError",
    "ConnectionReset",
    "RemoteDisconnected",
    "IncompleteRead",
    "empty_response",
)


def is_transient(error: str | None) -> bool:
    """Could this identical call plausibly succeed if repeated?"""
    if not error:
        return False
    return any(marker in error for marker in TRANSIENT_ERROR_MARKERS)


#: Backoff between retries, seconds. Capped low on purpose: a run's wall clock
#: keeps ticking through a sleep, so a textbook exponential schedule would spend
#: the whole 120 s budget waiting rather than working.
BACKOFF_BASE_S = 0.75
BACKOFF_CAP_S = 6.0


def backoff_delay(attempt: int) -> float:
    """Exponential with jitter, for retry number `attempt` (1-based).

    Jitter matters more than it looks: a suite runs many tasks against one
    endpoint, and synchronised retries are how a temporarily overloaded service
    is kept overloaded.
    """
    ceiling = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
    return ceiling * (0.5 + random.random() / 2)


# --------------------------------------------------------------------------
# Scripted
# --------------------------------------------------------------------------


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
        #: Every (system, messages) pair it was asked with, for assertions about
        #: what the loop actually put in front of the model — which is how the
        #: prompt-quarantine defence is tested.
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
                    "responses; the loop asked for one more"
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


# --------------------------------------------------------------------------
# OpenAI-compatible: NVIDIA, NIM, OpenRouter, OpenAI
# --------------------------------------------------------------------------


class OpenAICompatModel:
    """Chat-completions over stdlib urllib.

    **Reasoning is off by default, and that is a budget decision.** Nemotron
    emits its chain into a separate `reasoning_content` field that still bills
    as output tokens, and Project 01 needed a 16k output budget to avoid
    truncating the answer behind it. This loop runs up to 12 turns against a
    60k token budget (`docs/DESIGN.md` §10), so a thinking budget sized for a
    single classification would exhaust the run before it reached an opinion.
    The loop's own reasoning is the tool sequence, and the `thought` field on
    each action carries the rationale into the trace.

    Turn it on with `thinking=True` and raise `Budget.max_tokens` to match.
    Either way it is recorded on every completion, so a result file can never
    imply reasoning was on when it was not.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str = "nvidia",
        max_tokens: int = 2048,
        thinking: bool = False,
        reasoning_style: str = "chat_template",
        timeout: float = 180.0,
    ) -> None:
        if provider not in ENDPOINTS:
            raise ModelError(
                f"no OpenAI-compatible endpoint for {provider!r}; "
                f"known: {sorted(ENDPOINTS)}"
            )
        if reasoning_style not in REASONING_STYLES:
            raise ModelError(f"unknown reasoning style {reasoning_style!r}")

        self.provider = provider
        self.model = model or agent_model()
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.reasoning_style = reasoning_style
        self._timeout = timeout

        key_var = KEY_VARS[provider]
        self._key = os.environ.get(key_var, "").strip() if key_var else ""
        if key_var and not self._key:
            raise ModelError(
                f"{key_var} is not set. Copy .env.example to .env and fill it "
                "in, or run the tests instead — they need no key."
            )

        if provider == "nim":
            base = os.environ.get("NIM_BASE_URL", DEFAULT_NIM_BASE).rstrip("/")
            self._url = f"{base}/v1/chat/completions"
        else:
            self._url = ENDPOINTS[provider]

    def _ensure_client(self) -> None:
        """No client to build; the key check happened in __init__.

        Kept so callers can probe configuration without making a call, which is
        what `agent/__main__.py` and `draft_opinion` both want.
        """
        return None

    def _apply_reasoning(self, payload: dict[str, Any]) -> None:
        if self.reasoning_style == "chat_template":
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.thinking)}
        elif self.reasoning_style == "system_toggle":
            payload["messages"] = [
                {
                    "role": "system",
                    "content": f"detailed thinking {'on' if self.thinking else 'off'}",
                },
                *payload["messages"],
            ]
        elif self.reasoning_style == "effort":
            payload["reasoning_effort"] = "max" if self.thinking else "low"

    def complete(self, system: str, messages: Iterable[Message]) -> Completion:
        wire = [{"role": "system", "content": system}] if system else []
        wire += [m.to_json() for m in messages]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire,
            # The action protocol is a single JSON object. Sampling variety
            # buys nothing and costs parse failures, so it is pinned low rather
            # than left at the provider's default.
            "temperature": 0.0,
        }
        self._apply_reasoning(payload)

        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        started = time.perf_counter()
        blank = Completion(text="", model=self.model, provider=self.provider)

        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            blank.error = f"http_{exc.code}: {detail}"
            blank.latency_ms = int((time.perf_counter() - started) * 1000)
            return blank
        except Exception as exc:  # noqa: BLE001 — a failed call is data, not a crash
            blank.error = f"{type(exc).__name__}: {exc}"
            blank.latency_ms = int((time.perf_counter() - started) * 1000)
            return blank

        completion = parse_chat_response(body, self.model, self.provider)
        completion.latency_ms = int((time.perf_counter() - started) * 1000)
        completion.extra["thinking"] = self.thinking
        return completion


def parse_chat_response(body: dict[str, Any], model: str, provider: str) -> Completion:
    """Turn a chat-completions body into a Completion.

    Separate from the request so it can be tested without a network call.

    **Reasoning is deliberately not merged into `text`.** Models that reason
    return it in `reasoning_content`, and folding it into the answer would hand
    the chain of thought to a parser looking for a JSON action — which would
    match whichever tool call the model *considered* rather than the one it
    settled on. Its length is recorded so the tokens are accounted for.
    """
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = body.get("usage") or {}

    extra: dict[str, Any] = {}
    if reasoning := (message.get("reasoning_content") or ""):
        extra["reasoning_chars"] = len(reasoning)

    return Completion(
        text=message.get("content") or "",
        model=body.get("model") or model,
        provider=provider,
        tokens_in=usage.get("prompt_tokens", 0) or 0,
        tokens_out=usage.get("completion_tokens", 0) or 0,
        stop_reason=(choices[0].get("finish_reason") if choices else None),
        error=None if choices else "empty_response: no choices returned",
        extra=extra,
    )


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicModel:
    """The first-party path. Optional — needs the SDK and a key."""

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
                "the Anthropic path needs the SDK: "
                "pip install 'gst-resilient-agent[anthropic]'"
            ) from exc
        if not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            raise ModelError(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY in .env, "
                "or use the NVIDIA path (the default)."
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
        except Exception as exc:  # noqa: BLE001 — provider errors are data here
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


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def provider_for(model_id: str) -> str:
    """Which provider serves this model id.

    Inferred from the id rather than configured separately, because a provider
    and a model that disagree is a class of misconfiguration that produces a
    confusing 404 rather than a clear error.
    """
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if os.environ.get("NIM_MODEL") == model_id:
        return "nim"
    if os.environ.get("OPENROUTER_MODEL") == model_id:
        return "openrouter"
    if lowered.startswith("gpt-") or lowered.startswith("o1"):
        return "openai"
    # Everything else is a catalog id like "nvidia/nemotron-3-super-120b-a12b".
    return "nvidia"


def build(model_id: str | None = None, **kwargs: Any) -> Model:
    """The model named by `--model`, or the configured default."""
    model_id = (model_id or agent_model()).strip()
    provider = provider_for(model_id)
    if provider == "anthropic":
        return AnthropicModel(model_id, **kwargs)
    return OpenAICompatModel(model_id, provider=provider, **kwargs)
