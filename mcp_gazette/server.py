"""The Gazette MCP server: tariff lookups over the Model Context Protocol.

Independently runnable, with no dependencies:

    python -m mcp_gazette

It speaks JSON-RPC 2.0 over stdin/stdout and exposes the two tools that read
the hash-pinned Indian GST notifications. In the brief's contract-review
framing these are `lookup_statute`: look a rule up in the primary source and
return something citable. In this domain the primary source is a Gazette
notification and the rule is a tariff heading's rate, so the tools are named for
what they actually do - `lookup_schedule` and `rate_history` - and
`lookup_statute` is accepted as an alias so a client written against the generic
name still works.

### Why this is the right tool to expose

`docs/DESIGN.md` §7: of the seven tools, this one has a genuinely rich return
type (resolved / ambiguous / chapter-only / absent are four different answers,
not one string), real error responses, and is **independently useful** - "look
an Indian tariff heading up in the archived Gazette and get a citable rate" is
a thing other people want, and this server can be published on its own.

### The handlers are the same functions the agent calls in-process

`agent.tools.lookup_schedule.lookup_schedule` is imported and called directly.
There is no second implementation of the lookup for the protocol path, because
a four-way return type maintained twice would diverge on the third case. What
the protocol adds is transport, framing and failure modes - which is exactly
what makes "what did MCP cost me" measurable rather than rhetorical.

### Errors: two kinds, kept apart

A **protocol** error (bad JSON, unknown method, missing params) is a JSON-RPC
error object. The session may not be able to continue.

A **tool** error (heading not found, source hash mismatch) is a perfectly good
JSON-RPC *result* with `isError: true` and the structured payload inside it.
The model has to be able to read it and reason about it, which it cannot do
with a transport failure. Collapsing the two is how a recoverable tool failure
takes the whole session down.
"""

from __future__ import annotations

import json
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, TextIO

from mcp_gazette import protocol as P

#: Hard ceiling on one tool call, seconds. The Gazette PDFs are parsed once and
#: cached, so a healthy call is milliseconds; this exists for the pathological
#: case, not the normal one.
#:
#: Honest limitation: a timed-out call is *abandoned*, not killed. Python cannot
#: safely terminate a running thread, so the worker keeps going and its result
#: is discarded. That is a real leak under sustained timeouts and it is stated
#: here rather than discovered later.
CALL_TIMEOUT_S = 30.0


def _tool_specs() -> list[Any]:
    """The tool specs this server exposes, imported lazily.

    Lazily because importing `agent.tools` parses PDFs and reads the corpus,
    and a client that only ever calls `initialize` and `tools/list` should not
    pay for that at process start.
    """
    from agent.tools.lookup_schedule import SPEC as LOOKUP
    from agent.tools.rate_history import SPEC as HISTORY

    return [LOOKUP, HISTORY]


#: Generic names a client may use instead of the domain names.
ALIASES = {"lookup_statute": "lookup_schedule"}


class GazetteServer:
    """One MCP session. Owns the lifecycle and the tool table."""

    def __init__(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        *,
        call_timeout_s: float = CALL_TIMEOUT_S,
    ) -> None:
        self._in = stdin or sys.stdin
        self._out = stdout or sys.stdout
        self._timeout = call_timeout_s
        self._specs: dict[str, Any] = {}
        self._initialized = False
        self._shutdown = False
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gazette")
        self.calls_served = 0

    # -- lifecycle -------------------------------------------------------

    def _load_tools(self) -> dict[str, Any]:
        if not self._specs:
            self._specs = {s.name: s for s in _tool_specs()}
        return self._specs

    def install_signal_handlers(self) -> None:
        """Terminate cleanly on Ctrl-C or SIGTERM rather than on a traceback.

        A stdio server is a child process, and the parent kills it by signal.
        Dying with a stack trace on stderr is noise at best and, if anything of
        it reached stdout, a corrupted stream at worst.
        """

        def handler(signum, _frame):  # pragma: no cover - signal path
            P.log("signal received; shutting down", signal=signum)
            self._shutdown = True

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # not the main thread, or unsupported
                    pass

    def close(self) -> None:
        """Graceful shutdown. Idempotent."""
        if self._pool is not None:
            # Do not wait on abandoned timed-out calls; they are already
            # discarded and joining them would hang the exit we are trying to
            # make graceful.
            self._pool.shutdown(wait=False, cancel_futures=True)
        P.log("server stopped", calls_served=self.calls_served)

    def __enter__(self) -> "GazetteServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- the read loop ---------------------------------------------------

    def serve_forever(self) -> int:
        """Read until stdin closes. Returns a process exit code.

        **EOF on stdin is the shutdown signal.** MCP has no `shutdown` method in
        this revision: the client closes the pipe and the server exits. A server
        that blocks forever on a closed stdin is a zombie the parent has to kill.
        """
        P.log("server ready", protocol=P.PROTOCOL_VERSION, transport="stdio")
        for line in self._in:
            if self._shutdown:
                break
            line = line.strip()
            if not line:
                continue
            self.handle_line(line)
        P.log("stdin closed")
        return 0

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """Parse, dispatch, reply. Returns what it sent, for tests."""
        try:
            request = P.parse_message(line)
        except P.ProtocolError as exc:
            # A message so malformed it has no id still gets an error, with a
            # null id, per JSON-RPC. Silence would leave the client waiting.
            payload = P.error_response(None, exc.code, exc.message, exc.data)
            P.write_message(payload, self._out)
            P.log("bad message", code=exc.code, error=exc.message)
            return payload

        try:
            result = self.dispatch(request)
        except P.ProtocolError as exc:
            if request.is_notification:
                P.log("notification failed", method=request.method, error=exc.message)
                return None
            payload = P.error_response(request.id, exc.code, exc.message, exc.data)
            P.write_message(payload, self._out)
            return payload
        except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the session
            P.log("handler raised", method=request.method, error=repr(exc))
            if request.is_notification:
                return None
            payload = P.error_response(
                request.id, P.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"
            )
            P.write_message(payload, self._out)
            return payload

        # A notification is never answered. Replying to one is a protocol
        # violation that some clients treat as fatal.
        if request.is_notification:
            return None

        payload = P.response(request.id, result)
        P.write_message(payload, self._out)
        return payload

    # -- dispatch --------------------------------------------------------

    def dispatch(self, request: P.Request) -> dict[str, Any]:
        method = request.method

        if method == "initialize":
            return self._initialize(request.params)
        if method in ("notifications/initialized", "initialized"):
            self._initialized = True
            P.log("client initialized")
            return {}
        if method == "ping":
            return {}

        # Everything below requires a completed handshake. The spec says the
        # client must initialize first, and enforcing it here turns a confusing
        # empty tool list into a message that names the problem.
        if not self._initialized and method.startswith("tools/"):
            raise P.ProtocolError(
                P.INVALID_REQUEST,
                f"{method} before initialize; send initialize and the "
                "notifications/initialized notification first",
            )

        if method == "tools/list":
            return self._tools_list()
        if method == "tools/call":
            return self._tools_call(request.params)

        raise P.ProtocolError(P.METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client = params.get("clientInfo") or {}
        requested = params.get("protocolVersion")
        P.log(
            "initialize",
            client=client.get("name"),
            client_version=client.get("version"),
            requested_protocol=requested,
        )
        # The handshake completes even on a version mismatch: the client is told
        # what this server speaks and decides for itself. Refusing outright would
        # make a newer client unable to talk to an older server at all.
        self._initialized = True
        return {
            "protocolVersion": P.PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": P.SERVER_NAME, "version": P.SERVER_VERSION},
        }

    def _tools_list(self) -> dict[str, Any]:
        specs = self._load_tools()
        tools = [
            {
                "name": spec.name,
                "description": spec.description,
                # The SAME schema object the in-process registry validates
                # against. One definition, so the protocol surface cannot drift
                # from the local one.
                "inputSchema": spec.parameters,
            }
            for spec in specs.values()
        ]
        P.log("tools/list", count=len(tools))
        return {"tools": tools}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise P.ProtocolError(P.INVALID_PARAMS, "tools/call needs a tool name")
        # `is None` rather than `or {}`. The `or` idiom silently coerces every
        # falsy value - `[]`, `0`, `""`, `false` - into an empty dict, which
        # turns a wrongly-typed argument list into a call with no arguments and
        # loses the protocol error the client needed to see.
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise P.ProtocolError(
                P.INVALID_PARAMS,
                f"arguments must be an object, got {type(arguments).__name__}",
            )

        resolved = ALIASES.get(name, name)
        specs = self._load_tools()
        spec = specs.get(resolved)
        if spec is None:
            # Not a protocol error: the client asked for a tool that does not
            # exist, which it can correct. Returned as an error *result* so the
            # model reads it rather than the session dying.
            return self._error_result(
                None,
                "not_found",
                f"no tool named {name!r}; available: {sorted(specs)}",
                retryable=False,
            )

        # MCP has no tool_call_id of its own - the JSON-RPC `id` plays that role
        # for the transport. The agent's contract wants one on the *result*, so
        # a client-supplied id is honoured and otherwise one is minted here.
        meta = params.get("_meta") or {}
        call_id = str(meta.get("tool_call_id") or uuid.uuid4().hex[:12])

        started = time.perf_counter()
        future = self._pool.submit(self._invoke, spec, arguments)
        try:
            result = future.result(timeout=self._timeout)
        except FutureTimeout:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            P.log("tool timeout", tool=resolved, call_id=call_id, latency_ms=elapsed)
            return self._error_result(
                call_id,
                "timeout",
                f"{resolved} exceeded {self._timeout:g}s",
                retryable=True,
                latency_ms=elapsed,
            )

        elapsed = round((time.perf_counter() - started) * 1000, 3)
        self.calls_served += 1
        P.log(
            "tools/call",
            tool=resolved,
            call_id=call_id,
            ok=result.ok,
            error=result.error,
            latency_ms=elapsed,
        )

        body = result.to_json()
        body["tool_call_id"] = call_id
        body["latency_ms"] = elapsed
        return {
            # Text content carrying the full structured envelope. The MCP
            # revision this targets has no typed result channel, so the payload
            # is JSON inside a text block - which is what every client of this
            # revision expects.
            "content": [
                {"type": "text", "text": json.dumps(body, ensure_ascii=False, default=str)}
            ],
            # A tool that ran and failed is a RESULT, flagged - not a JSON-RPC
            # error. The model has to read it to recover from it.
            "isError": not result.ok,
            "_meta": {
                "tool_call_id": call_id,
                "latency_ms": elapsed,
                "retryable": result.retryable,
                "error": result.error,
            },
        }

    @staticmethod
    def _invoke(spec: Any, arguments: dict[str, Any]) -> Any:
        """Call the handler, converting anything it leaks into a tool error.

        Argument validation happens here too, against the same schema the
        in-process registry uses, so a bad call over the protocol fails the same
        way and with the same message as a bad call locally.
        """
        from agent.contract import ToolResult
        from agent.jsonschema_lite import validate

        if problems := validate(dict(arguments), spec.parameters):
            return ToolResult.err("bad_argument", "; ".join(problems),
                                  data={"problems": problems})
        try:
            out = spec.handler(**arguments)
        except TypeError as exc:
            return ToolResult.err("bad_argument", str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.err("internal", f"{type(exc).__name__}: {exc}")
        if not isinstance(out, ToolResult):
            return ToolResult.err(
                "internal", f"{spec.name} returned {type(out).__name__}"
            )
        return out

    @staticmethod
    def _error_result(
        call_id: str | None,
        code: str,
        message: str,
        *,
        retryable: bool,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        body = {
            "ok": False,
            "data": {},
            "error": code,
            "retryable": retryable,
            "message": message,
            "tool_call_id": call_id,
            "latency_ms": latency_ms,
        }
        return {
            "content": [
                {"type": "text", "text": json.dumps(body, ensure_ascii=False)}
            ],
            "isError": True,
            "_meta": {
                "tool_call_id": call_id,
                "latency_ms": latency_ms,
                "retryable": retryable,
                "error": code,
            },
        }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mcp_gazette",
        description="Indian GST Gazette lookups over the Model Context Protocol (stdio).",
    )
    parser.add_argument(
        "--timeout", type=float, default=CALL_TIMEOUT_S, help="per-call timeout, seconds"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="print the tool schemas and exit, without speaking the protocol",
    )
    args = parser.parse_args(argv)

    if args.list_tools:
        # Diagnostics on stdout are safe here precisely because this path never
        # speaks the protocol.
        server = GazetteServer(call_timeout_s=args.timeout)
        print(json.dumps(server._tools_list(), indent=2, ensure_ascii=False))
        return 0

    with GazetteServer(call_timeout_s=args.timeout) as server:
        server.install_signal_handlers()
        try:
            return server.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            P.log("interrupted")
            return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
