"""The MCP server and client: protocol conformance, and the boundary itself.

Most of this runs the server in-process with StringIO pipes, which is fast and
deterministic. A handful of tests spawn a real subprocess, because the failures
that actually bite a stdio server - a stray write to stdout, a process that
will not exit, a broken pipe - only exist when there is a real process.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from mcp_gazette import protocol as P
from mcp_gazette.server import GazetteServer

pytest.importorskip("pypdf", reason="the Gazette tools read the archived PDFs")


def make_server(**kw) -> tuple[GazetteServer, io.StringIO]:
    out = io.StringIO()
    return GazetteServer(stdin=io.StringIO(), stdout=out, **kw), out


def rpc(method: str, params: dict | None = None, id: int | None = 1) -> str:
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if id is not None:
        body["id"] = id
    return json.dumps(body)


def handshake(server: GazetteServer) -> None:
    server.handle_line(rpc("initialize", {"protocolVersion": P.PROTOCOL_VERSION}))
    server.handle_line(rpc("notifications/initialized", id=None))


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def test_a_request_and_a_notification_are_distinguished_by_id():
    """`0` is a valid id, so this must never be tested for truthiness."""
    assert P.parse_message(rpc("ping", id=0)).is_notification is False
    assert P.parse_message(rpc("ping", id=None)).is_notification is True


@pytest.mark.parametrize(
    "line,code",
    [
        ("not json at all", P.PARSE_ERROR),
        ('"a bare string"', P.INVALID_REQUEST),
        ('{"jsonrpc":"1.0","method":"ping"}', P.INVALID_REQUEST),
        ('{"jsonrpc":"2.0"}', P.INVALID_REQUEST),
        ('{"jsonrpc":"2.0","method":"ping","params":[]}', P.INVALID_PARAMS),
    ],
)
def test_malformed_messages_get_the_right_json_rpc_code(line, code):
    with pytest.raises(P.ProtocolError) as exc:
        P.parse_message(line)
    assert exc.value.code == code


def test_a_message_with_no_id_still_gets_an_error_reply():
    """Silence would leave the client blocked on a read forever."""
    server, out = make_server()
    server.handle_line("{ this is not json")
    reply = json.loads(out.getvalue())
    assert reply["id"] is None
    assert reply["error"]["code"] == P.PARSE_ERROR


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_initialize_returns_protocol_version_and_server_info():
    server, _ = make_server()
    reply = server.handle_line(rpc("initialize", {"protocolVersion": "2024-11-05"}))
    result = reply["result"]
    assert result["protocolVersion"] == P.PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "gazette"
    assert "tools" in result["capabilities"]


def test_a_version_mismatch_completes_the_handshake_rather_than_refusing():
    """The client is told what this server speaks and decides for itself.
    Refusing would make a newer client unable to talk to an older server."""
    server, _ = make_server()
    reply = server.handle_line(rpc("initialize", {"protocolVersion": "2099-01-01"}))
    assert reply["result"]["protocolVersion"] == P.PROTOCOL_VERSION


def test_a_notification_is_never_answered():
    """Replying to a notification is a protocol violation some clients treat
    as fatal."""
    server, out = make_server()
    assert server.handle_line(rpc("notifications/initialized", id=None)) is None
    assert out.getvalue() == ""


def test_tools_before_initialize_are_refused_by_name():
    server, _ = make_server()
    reply = server.handle_line(rpc("tools/list"))
    assert reply["error"]["code"] == P.INVALID_REQUEST
    assert "initialize" in reply["error"]["message"]


def test_ping_works_and_an_unknown_method_does_not():
    server, _ = make_server()
    handshake(server)
    assert server.handle_line(rpc("ping"))["result"] == {}
    reply = server.handle_line(rpc("tools/teleport"))
    assert reply["error"]["code"] == P.METHOD_NOT_FOUND


def test_close_is_idempotent():
    server, _ = make_server()
    server.close()
    server.close()


# --------------------------------------------------------------------------
# tools/list
# --------------------------------------------------------------------------


def test_tools_list_exposes_the_gazette_tools_with_their_schemas():
    server, _ = make_server()
    handshake(server)
    tools = server.handle_line(rpc("tools/list"))["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"lookup_schedule", "rate_history"} <= names
    for tool in tools:
        assert tool["description"].strip()
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert schema["required"]


def test_the_advertised_schema_is_the_same_object_the_registry_validates_with():
    """One definition. A protocol surface that drifted from the local one would
    accept calls the in-process tool rejects."""
    from agent.tools.lookup_schedule import SPEC

    server, _ = make_server()
    handshake(server)
    tools = server.handle_line(rpc("tools/list"))["result"]["tools"]
    advertised = next(t for t in tools if t["name"] == "lookup_schedule")
    assert advertised["inputSchema"] == SPEC.parameters


# --------------------------------------------------------------------------
# tools/call
# --------------------------------------------------------------------------


def call(server: GazetteServer, name: str, arguments: dict, **params) -> dict:
    reply = server.handle_line(
        rpc("tools/call", {"name": name, "arguments": arguments, **params})
    )
    return reply["result"]


def test_a_successful_call_carries_the_full_envelope():
    server, _ = make_server()
    handshake(server)
    result = call(server, "lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"})

    assert result["isError"] is False
    body = json.loads(result["content"][0]["text"])
    assert body["ok"] is True
    assert body["data"]["slab"] == "18"
    for key in ("tool_call_id", "latency_ms", "error", "retryable"):
        assert key in body
    assert result["_meta"]["tool_call_id"] == body["tool_call_id"]


def test_the_date_branch_survives_the_protocol():
    """2402 is 28% before the 2026 amendment and 40% after. If the boundary
    flattened that, the server would be answering a different question."""
    server, _ = make_server()
    handshake(server)
    before = json.loads(
        call(server, "lookup_schedule", {"heading": "2402", "on_date": "2025-11-01"})
        ["content"][0]["text"]
    )
    after = json.loads(
        call(server, "lookup_schedule", {"heading": "2402", "on_date": "2026-03-01"})
        ["content"][0]["text"]
    )
    assert before["data"]["slab"] == "28"
    assert after["data"]["slab"] == "40"


def test_lookup_statute_is_accepted_as_an_alias():
    """The brief's generic name for this tool. A client written against it
    works unchanged."""
    server, _ = make_server()
    handshake(server)
    result = call(server, "lookup_statute", {"heading": "6810", "on_date": "2026-03-14"})
    body = json.loads(result["content"][0]["text"])
    assert body["ok"] and body["data"]["heading"] == "6810"


def test_a_client_supplied_tool_call_id_is_honoured():
    server, _ = make_server()
    handshake(server)
    result = call(
        server, "lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"},
        _meta={"tool_call_id": "caller-chose-this"},
    )
    assert result["_meta"]["tool_call_id"] == "caller-chose-this"


def test_a_failed_tool_is_a_result_not_a_protocol_error():
    """The model has to read it to recover from it, which it cannot do with a
    transport failure. Collapsing the two takes the session down."""
    server, _ = make_server()
    handshake(server)
    reply = server.handle_line(
        rpc("tools/call", {"name": "lookup_schedule",
                           "arguments": {"heading": "zzz", "on_date": "2026-03-14"}})
    )
    assert "error" not in reply, "a tool failure must not be a JSON-RPC error"
    assert reply["result"]["isError"] is True
    body = json.loads(reply["result"]["content"][0]["text"])
    assert body["error"] == "bad_argument"
    assert body["retryable"] is False


def test_arguments_are_validated_against_the_schema_before_the_handler_runs():
    server, _ = make_server()
    handshake(server)
    body = json.loads(
        call(server, "lookup_schedule", {"heading": "6810"})["content"][0]["text"]
    )
    assert body["ok"] is False
    assert body["error"] == "bad_argument"
    assert "on_date" in body["message"]


def test_an_unknown_tool_is_a_result_the_caller_can_correct():
    server, _ = make_server()
    handshake(server)
    result = call(server, "no_such_tool", {})
    assert result["isError"] is True
    body = json.loads(result["content"][0]["text"])
    assert body["error"] == "not_found"
    assert "lookup_schedule" in body["message"]


@pytest.mark.parametrize(
    "params", [{}, {"name": ""}, {"name": "lookup_schedule", "arguments": []}]
)
def test_malformed_tools_call_params_are_protocol_errors(params):
    server, _ = make_server()
    handshake(server)
    reply = server.handle_line(rpc("tools/call", params))
    assert reply["error"]["code"] == P.INVALID_PARAMS


def test_a_slow_tool_times_out_as_a_retryable_error():
    server, _ = make_server(call_timeout_s=0.05)
    handshake(server)

    class Slow:
        name = "lookup_schedule"
        parameters = {"type": "object", "properties": {}, "additionalProperties": True}

        @staticmethod
        def handler(**_):
            import time

            time.sleep(2)

    server._specs = {"lookup_schedule": Slow()}
    result = call(server, "lookup_schedule", {})
    assert result["isError"] is True
    assert result["_meta"]["error"] == "timeout"
    assert result["_meta"]["retryable"] is True


def test_a_handler_that_raises_does_not_kill_the_session():
    server, _ = make_server()
    handshake(server)

    class Exploding:
        name = "lookup_schedule"
        parameters = {"type": "object", "properties": {}, "additionalProperties": True}

        @staticmethod
        def handler(**_):
            raise RuntimeError("boom")

    server._specs = {"lookup_schedule": Exploding()}
    body = json.loads(call(server, "lookup_schedule", {})["content"][0]["text"])
    assert body["error"] == "internal"
    # And the session is still usable.
    assert server.handle_line(rpc("ping", id=99))["result"] == {}


# --------------------------------------------------------------------------
# Nothing but protocol on stdout
# --------------------------------------------------------------------------


def test_stdout_carries_only_json_rpc():
    """A stray write to stdout lands in the middle of the stream and kills the
    client's parser. This is the single most common way a stdio server breaks."""
    server, out = make_server()
    handshake(server)
    call(server, "lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"})
    call(server, "nope", {})
    for line in out.getvalue().splitlines():
        if line.strip():
            body = json.loads(line)  # raises if anything else got in
            assert body["jsonrpc"] == "2.0"


def test_log_writes_to_stderr_not_stdout(capsys):
    P.log("hello", detail=1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["msg"] == "hello"


# --------------------------------------------------------------------------
# Client, against a real subprocess
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_client():
    from mcp_gazette.client import McpClient

    client = McpClient()
    try:
        client.start()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not start the server subprocess: {exc}")
    yield client
    client.close()


def test_client_handshake_and_tool_listing(live_client):
    assert live_client.server_info["name"] == "gazette"
    names = {t["name"] for t in live_client.list_tools()}
    assert {"lookup_schedule", "rate_history"} <= names


def test_a_remote_call_returns_a_normal_toolresult(live_client):
    result = live_client.call_tool(
        "lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"}
    )
    assert result.ok
    assert result.data["slab"] == "18"
    assert result.tool_call_id and result.latency_ms is not None


def test_evidence_survives_the_boundary(live_client):
    """Verbatim document text is the untrusted channel; losing it across the
    protocol would silently disarm every defence built on it."""
    result = live_client.call_tool(
        "lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"}
    )
    assert result.evidence
    assert all(e.source and e.text for e in result.evidence)


def test_a_remote_failure_is_a_structured_toolresult(live_client):
    result = live_client.call_tool(
        "lookup_schedule", {"heading": "6810", "on_date": "not-a-date"}
    )
    assert result.ok is False
    assert result.error == "bad_argument"


def test_a_dead_server_becomes_a_retryable_tool_error_not_an_exception():
    """The agent's contract says a tool never raises. A tool that happens to
    live in another process is no exception."""
    from mcp_gazette.client import McpClient

    client = McpClient()
    client.start()
    client._proc.kill()
    client._proc.wait(timeout=5)

    result = client.call_tool("lookup_schedule", {"heading": "6810", "on_date": "2026-03-14"})
    assert result.ok is False
    assert result.error in ("unavailable", "timeout")
    assert result.retryable is True
    client.close()


def test_the_registry_cannot_tell_which_side_of_the_boundary_a_tool_is_on(live_client):
    """`docs/DESIGN.md` §7: the local version is kept behind the same interface
    so the boundary can be switched off and its cost measured."""
    from agent.registry import build_call
    from agent.tools import build_registry
    from mcp_gazette.client import build_registry_over_mcp

    local = build_registry()
    remote = build_registry_over_mcp(live_client)
    assert local.names() == remote.names()

    args = {"heading": "7418", "on_date": "2026-03-14"}
    a = local.invoke(build_call("lookup_schedule", args))
    b = remote.invoke(build_call("lookup_schedule", args))

    assert a.ok == b.ok
    assert a.data["outcome"] == b.data["outcome"] == "ambiguous"
    assert a.data["distinct_rates"] == b.data["distinct_rates"]


def test_close_survives_being_called_twice(live_client):
    from mcp_gazette.client import McpClient

    client = McpClient().start()
    client.close()
    client.close()
    assert client.alive is False
