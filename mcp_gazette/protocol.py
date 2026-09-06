"""JSON-RPC 2.0 framing for MCP over stdio. Stdlib only.

MCP is JSON-RPC 2.0 with a fixed method vocabulary, carried over stdin/stdout as
**newline-delimited JSON** - one complete message per line, no Content-Length
headers. That is the whole transport.

Written out rather than imported from an SDK, for the same reason the agent loop
is hand-rolled: the protocol boundary is a thing this project needs to be able
to inspect, time and break on purpose, and a boundary you cannot see is one you
cannot measure. It also means the server has no dependencies at all, so
`python -m mcp_gazette` runs on a fresh clone.

### The one rule that breaks every stdio server at least once

**stdout is the protocol channel.** A stray `print()`, a warning from a library,
a progress bar - anything that reaches stdout - lands in the middle of the JSON
stream and the client's parser dies on it. Every diagnostic in this package goes
to **stderr**, and `log()` is the only way anything is written there.

### Requests, responses, notifications

A request has an `id` and must be answered exactly once. A **notification** has
no `id` and must never be answered - replying to one is a protocol violation
that some clients treat as fatal. `is_notification` is what keeps that straight.

Error codes are the JSON-RPC standard set. They describe *protocol* failures -
malformed JSON, unknown method, bad params. A tool that ran and failed is NOT a
protocol error: it returns a normal result with `isError: true`, because the
model needs to read it and reason about it. Conflating the two is how a tool
failure becomes a transport failure and takes the session with it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

#: The MCP revision this server implements. Sent in `initialize` and echoed by
#: the client; a mismatch is negotiated rather than fatal.
PROTOCOL_VERSION = "2024-11-05"

SERVER_NAME = "gazette"
SERVER_VERSION = "0.1.0"

# -- JSON-RPC standard error codes -----------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ProtocolError(Exception):
    """A JSON-RPC level failure, carrying the code to send back."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(slots=True)
class Request:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    #: None for a notification. Note that `0` is a valid id, so this must be
    #: tested with `is None` and never for truthiness.
    id: Any = None

    @property
    def is_notification(self) -> bool:
        return self.id is None


def parse_message(line: str) -> Request:
    """One line of stdin into a Request. Raises ProtocolError."""
    try:
        body = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(PARSE_ERROR, f"invalid JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise ProtocolError(INVALID_REQUEST, "message must be a JSON object")
    if body.get("jsonrpc") != "2.0":
        raise ProtocolError(
            INVALID_REQUEST, f"expected jsonrpc 2.0, got {body.get('jsonrpc')!r}"
        )
    method = body.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(INVALID_REQUEST, "missing method")

    params = body.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ProtocolError(INVALID_PARAMS, "params must be an object")

    return Request(method=method, params=params, id=body.get("id"))


def response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": err}


def write_message(payload: dict[str, Any], stream: TextIO | None = None) -> None:
    """Emit one message. Flushed immediately.

    Flushed every time because the client is blocking on a read: a buffered
    reply is a hang, and a hang looks exactly like a slow tool.
    """
    stream = stream or sys.stdout
    stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    stream.flush()


_STARTED = time.time()


def log(message: str, **fields: Any) -> None:
    """Structured diagnostics, to **stderr**.

    Never stdout. stdout is the protocol channel, and a diagnostic written
    there lands in the middle of the JSON stream and kills the client's parser.
    This is the single most common way a stdio MCP server breaks.
    """
    record = {
        "t": round(time.time() - _STARTED, 4),
        "level": "info",
        "msg": message,
        **fields,
    }
    sys.stderr.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    sys.stderr.flush()
