"""An MCP client, and the bridge that lets the agent use the server as a tool.

Two pieces:

`McpClient` speaks the protocol to a server subprocess over stdio - handshake,
`tools/list`, `tools/call`, shutdown.

`mcp_tool_spec()` wraps one of those remote tools in a `ToolSpec` whose handler
returns a `ToolResult`, so `Registry` cannot tell the difference between a tool
that runs in this process and one that runs in another. That is what makes
`docs/DESIGN.md` §7's claim testable: the protocol boundary can be switched off,
so "what did MCP cost me in latency" is a measurement rather than a rhetorical
question.

### Every transport failure becomes a structured tool error

The agent's contract says a tool never raises. A subprocess that died, a pipe
that closed, a reply that never came - all of them come back as
`ToolResult.err(...)` with a code from the closed vocabulary and an honest
`retryable`. A dead server is `unavailable` and retryable; a malformed reply is
`malformed` and is not, because the same call will produce the same garbage.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

from agent.contract import Evidence, ToolResult
from agent.registry import ToolSpec
from mcp_gazette import protocol as P

DEFAULT_TIMEOUT_S = 60.0


class McpClientError(RuntimeError):
    """The client could not talk to a server at all."""


class McpClient:
    """Speaks MCP to a server subprocess over stdio."""

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cwd: str | None = None,
    ) -> None:
        self.command = command or [sys.executable, "-m", "mcp_gazette"]
        self.timeout_s = timeout_s
        self._cwd = cwd or os.getcwd()
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self.server_info: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "McpClient":
        if self._proc is not None:
            return self
        env = dict(os.environ)
        # The server imports `agent.*`, so the repo has to be importable in the
        # child. Inherited PYTHONPATH is not guaranteed under a test runner.
        env["PYTHONPATH"] = self._cwd + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr is the server's log channel. Left as a separate pipe so
                # it can never contaminate the protocol stream on stdout.
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line buffered: the transport is line delimited
                cwd=self._cwd,
                env=env,
            )
        except OSError as exc:
            raise McpClientError(f"could not start {self.command}: {exc}") from exc
        self.initialize()
        return self

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": P.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "gst-resilient-agent", "version": "0.1.0"},
            },
        )
        self.server_info = result.get("serverInfo", {})
        # A notification: no id, and no reply is expected or read.
        self.notify("notifications/initialized")
        return result

    def close(self) -> None:
        """Graceful shutdown: close stdin, let the server exit, then insist."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                # EOF on stdin is the shutdown signal in this MCP revision.
                proc.stdin.close()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):  # pragma: no cover
                pass
        finally:
            for pipe in (proc.stdout, proc.stderr):
                if pipe and not pipe.closed:
                    pipe.close()

    def __enter__(self) -> "McpClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- transport -------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpClientError("server is not running")
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpClientError(f"server closed the pipe: {exc}") from exc

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification. No id, and deliberately no reply read."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and wait for the matching reply."""
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            body = self._read_reply(request_id)

        if "error" in body:
            err = body["error"]
            raise McpClientError(
                f"{method} failed [{err.get('code')}]: {err.get('message')}"
            )
        return body.get("result") or {}

    def _read_reply(self, request_id: int) -> dict[str, Any]:
        """Read until the reply with this id arrives, or the server dies."""
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() > deadline:
                raise McpClientError(f"no reply within {self.timeout_s:g}s")
            line = self._proc.stdout.readline()
            if line == "":
                stderr = ""
                if self._proc.stderr and not self._proc.stderr.closed:
                    try:
                        stderr = self._proc.stderr.read() or ""
                    except (OSError, ValueError):  # pragma: no cover
                        pass
                raise McpClientError(
                    "server exited without replying"
                    + (f"; stderr tail: {stderr[-400:]}" if stderr else "")
                )
            line = line.strip()
            if not line:
                continue
            try:
                body = json.loads(line)
            except json.JSONDecodeError as exc:
                # Almost always a stray print() in the server contaminating the
                # protocol stream. Named explicitly because the generic
                # "invalid JSON" sends people looking in the wrong place.
                raise McpClientError(
                    f"non-JSON on the protocol stream (a stray write to stdout "
                    f"in the server?): {line[:200]!r}"
                ) from exc
            # Ignore anything that is not our reply - notifications, or replies
            # to a request that already timed out.
            if body.get("id") == request_id:
                return body

    # -- tools -----------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        self.tools = self.request("tools/list").get("tools", [])
        return self.tools

    def call_tool(
        self, name: str, arguments: dict[str, Any], *, tool_call_id: str | None = None
    ) -> ToolResult:
        """Call a remote tool and return it as a local `ToolResult`.

        Never raises: every transport failure is converted to a structured tool
        error, because the agent's contract says a tool cannot unwind the loop
        and a tool that happens to live in another process is no exception.
        """
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        if tool_call_id:
            params["_meta"] = {"tool_call_id": tool_call_id}

        started = time.perf_counter()
        try:
            result = self.request("tools/call", params)
        except McpClientError as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            dead = not self.alive
            return ToolResult.err(
                "unavailable" if dead else "timeout",
                f"MCP transport: {exc}",
                # A dead server may come back if the caller restarts it; a
                # timeout on a live one may simply have been slow.
                retryable=True,
                data={"transport": "mcp", "server_alive": self.alive},
            ).stamped(tool_call_id=tool_call_id or "mcp", latency_ms=elapsed)

        return self._decode(result, tool_call_id, started)

    @staticmethod
    def _decode(
        result: dict[str, Any], tool_call_id: str | None, started: float
    ) -> ToolResult:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        blocks = result.get("content") or []
        text = next(
            (b.get("text", "") for b in blocks if b.get("type") == "text"), ""
        )
        try:
            body = json.loads(text) if text else {}
        except json.JSONDecodeError:
            return ToolResult.err(
                "malformed",
                "the server returned a content block that is not JSON",
                # Not retryable: the same call will produce the same garbage.
                retryable=False,
                data={"raw": text[:400]},
            ).stamped(tool_call_id=tool_call_id or "mcp", latency_ms=elapsed)

        meta = result.get("_meta") or {}
        call_id = str(body.get("tool_call_id") or meta.get("tool_call_id") or tool_call_id or "mcp")

        if body.get("ok"):
            evidence = tuple(
                Evidence(
                    source=e.get("source", ""),
                    locator=e.get("locator", ""),
                    text=e.get("text", ""),
                )
                for e in (body.get("evidence") or [])
            )
            return ToolResult.ok_(body.get("data") or {}, evidence).stamped(
                tool_call_id=call_id, latency_ms=elapsed
            )

        return ToolResult.err(
            body.get("error") or meta.get("error") or "internal",
            body.get("message") or "the remote tool failed",
            retryable=bool(body.get("retryable", meta.get("retryable", False))),
            data=body.get("data") or {},
        ).stamped(tool_call_id=call_id, latency_ms=elapsed)


# --------------------------------------------------------------------------
# The bridge
# --------------------------------------------------------------------------


def mcp_tool_spec(client: McpClient, name: str, local: ToolSpec) -> ToolSpec:
    """A `ToolSpec` that runs `name` on the server instead of in this process.

    Takes the local spec for its schema and stage rather than trusting the
    remote description, so the registry validates arguments identically either
    way and the only difference between the two paths is the transport - which
    is the point.
    """

    def handler(**arguments: Any) -> ToolResult:
        return client.call_tool(name, arguments)

    return ToolSpec(
        name=local.name,
        description=local.description,
        parameters=local.parameters,
        handler=handler,
        stage=local.stage,
        returns_evidence=local.returns_evidence,
        # Not pure any more: the same arguments now depend on a subprocess being
        # alive. Idempotent only because the cache makes it so, which is exactly
        # the residual risk `IdempotencyCache` documents.
        pure=False,
    )


def build_registry_over_mcp(client: McpClient) -> Any:
    """The full seven-tool registry, with the Gazette tools served over MCP.

    Same tools, same schemas, same stages. Only `lookup_schedule` and
    `rate_history` cross a process boundary, because they are the two that read
    the pinned notifications.
    """
    from agent.tools import SPECS
    from agent.registry import Registry

    remote = {t["name"] for t in (client.tools or client.list_tools())}
    registry = Registry()
    for spec in SPECS:
        registry.register(
            mcp_tool_spec(client, spec.name, spec) if spec.name in remote else spec
        )
    return registry
