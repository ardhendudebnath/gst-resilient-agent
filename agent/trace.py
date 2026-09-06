"""The run trace. Written from the first commit, not added once it is needed.

You cannot build a failure taxonomy out of recollection. Every claim in
`FAILURES.md` has to point at a run, a step and a tool result, so the trace is
the primary artefact of this repository and the agent is the thing that
produces it.

**JSONL, one event per line, appended as the run proceeds.** Not a single JSON
document written at the end, and the reason is specific to this project: the
whole point of week 4 is to make runs die in the middle. A half-written JSON
array is unparseable and takes its evidence with it. A half-written JSONL file
is readable up to the last complete line, which is exactly the line before the
thing you are trying to diagnose.

**Chaos is an event, not an annotation.** Every fabricated failure emits its
own `chaos` event *and* tags the result it altered. An injected failure that
looked organic in the trace would make the taxonomy fiction, so the labelling
is done in two places on purpose.

What a run records, per `docs/DESIGN.md` §4: the full message history, every
tool call with arguments and result, timings, token counts, and the outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))

#: Schema version for the event stream. The viewer and any analysis script
#: check it. Bumped when an event's shape changes in a way that would silently
#: misread an older trace — a new *optional* field does not bump it.
TRACE_SCHEMA = 1

# Event types. Closed set: the viewer switches on these, and an unrecognised
# event would be dropped silently, which is the one thing a trace may not do.
EVENTS = frozenset(
    {
        "run_start",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "cache_hit",
        "chaos",  # a failure was injected
        "defence",  # a defence fired (week 5)
        "policy",  # a recovery policy fired (week 6)
        "note",  # free-text marker, for diagnosis
        "run_end",
    }
)


class TraceError(RuntimeError):
    pass


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _truncate(value: Any, limit: int) -> Any:
    """Shorten long strings, leaving proof of what was removed.

    Truncation is lossy, so it is never silent: the replacement carries the
    original length and a hash, which is enough to prove later that a full
    trace and a truncated one describe the same run.
    """
    if limit <= 0:
        return value
    if isinstance(value, str) and len(value) > limit:
        return {
            "__truncated__": True,
            "head": value[:limit],
            "chars": len(value),
            "sha256_16": _digest(value),
        }
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, limit) for v in value]
    return value


@dataclass(slots=True)
class Tracer:
    """Append-only event writer for one run.

    Not thread-safe, by choice: one run is one sequence of steps, and a trace
    whose event order depends on scheduling is not a trace of anything. Suite
    concurrency gets one tracer per task.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: Path | None = None
    #: 0 keeps every string whole. Set it only when trace size is genuinely a
    #: problem; the default is complete, because a trace that dropped the
    #: evidence is worth less than a large file.
    max_text_chars: int = 0
    _fh: Any = None
    _t0: float = field(default_factory=time.perf_counter)
    _seq: int = 0
    _events: list[dict[str, Any]] = field(default_factory=list)
    #: When True, events are held in memory and never written. For tests.
    memory_only: bool = False

    def __post_init__(self) -> None:
        if self.memory_only:
            return
        if self.path is None:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            self.path = TRACE_DIR / f"{self.run_id}.jsonl"
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "Tracer":
        if not self.memory_only and self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")  # type: ignore[union-attr]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- writing ---------------------------------------------------------

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        """Write one event. Returns it, so a caller can log and inspect at once."""
        if event not in EVENTS:
            raise TraceError(
                f"unknown trace event {event!r}; the set is closed because the "
                f"viewer switches on it. Known: {sorted(EVENTS)}"
            )
        self._seq += 1
        rec: dict[str, Any] = {
            "seq": self._seq,
            "t_ms": int((time.perf_counter() - self._t0) * 1000),
            "event": event,
            **_truncate(payload, self.max_text_chars),
        }
        self._events.append(rec)
        if not self.memory_only:
            if self._fh is None:
                self._fh = self.path.open("a", encoding="utf-8")  # type: ignore[union-attr]
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            # Flushed every event. A buffered trace loses precisely the tail
            # that explains a hang, and the tail is what week 4 is about.
            self._fh.flush()
        return rec

    # -- convenience, so callers never hand-build an event ---------------

    def run_start(self, **config: Any) -> None:
        self.emit(
            "run_start",
            run_id=self.run_id,
            schema=TRACE_SCHEMA,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **config,
        )

    def tool_call(self, call: Any) -> None:
        self.emit("tool_call", **call.to_json())

    def tool_result(self, call: Any, result: Any, *, duration_ms: int) -> None:
        self.emit(
            "tool_result",
            call_id=call.call_id,
            name=call.name,
            duration_ms=duration_ms,
            result=result.to_json(),
        )

    def cache_hit(self, call: Any) -> None:
        self.emit("cache_hit", call_id=call.call_id, name=call.name, key=call.key)

    def chaos(self, *, mode: str, target: str, call_id: str | None = None, **extra: Any) -> None:
        """Record an injected failure. Always paired with a tagged result."""
        self.emit("chaos", mode=mode, target=target, call_id=call_id, **extra)

    def run_end(self, *, terminal: str, **payload: Any) -> None:
        self.emit("run_end", terminal=terminal, **payload)

    # -- reading back ----------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Replay a trace file, tolerating a truncated final line.

    Tolerating it is the point. A run killed by a hang leaves a partial last
    line, and refusing to read the file because of it would discard every
    event before the interesting one.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Only the last line may be partial; anything earlier is
                # corruption and should be surfaced.
                if fh.read(1) == "":
                    return
                raise TraceError(f"{p}:{lineno}: malformed event, and not the last line")
