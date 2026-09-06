# The MCP protocol boundary

`mcp_gazette/` is a Model Context Protocol server exposing the tools that read
the hash-pinned Indian GST notifications. It runs on its own, with no
dependencies:

```bash
python -m mcp_gazette                # speak MCP over stdio
python -m mcp_gazette --list-tools   # print the schemas, speak nothing
```

The brief's contract-review framing calls this tool `lookup_statute`: look a
rule up in the primary source and return something citable. In this domain the
primary source is a Gazette notification and the rule is a tariff heading's
rate, so the tools are named for what they do — `lookup_schedule` and
`rate_history` — and **`lookup_statute` is accepted as an alias**, so a client
written against the generic name works unchanged.

---

## What MCP actually is

Strip the tooling away and it is **JSON-RPC 2.0 over newline-delimited stdio**.
One complete JSON object per line, no framing headers. That is the entire
transport.

```
client                                    server
  │                                         │
  │──▶ {"jsonrpc":"2.0","id":1,"method":"initialize",...}
  │◀── {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":...,"serverInfo":...}}
  │──▶ {"jsonrpc":"2.0","method":"notifications/initialized"}      (no id, no reply)
  │──▶ {"jsonrpc":"2.0","id":2,"method":"tools/list"}
  │◀── {"jsonrpc":"2.0","id":2,"result":{"tools":[{name,description,inputSchema}]}}
  │──▶ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":...,"arguments":{...}}}
  │◀── {"jsonrpc":"2.0","id":3,"result":{"content":[...],"isError":false}}
  │
  │  (client closes stdin)  ──────────────▶ EOF: server exits
```

Implemented directly in `protocol.py`, stdlib only. Not because an SDK would be
wrong, but because this project has to be able to **see, time and deliberately
break** the boundary, and an SDK that hides the framing hides exactly the seam
the chaos harness needs. It also means the server has no dependencies, so it
runs on a fresh clone.

---

## Five things that decide whether a stdio server works

### 1. stdout is the protocol channel

A stray `print()`, a library warning, a progress bar — anything reaching stdout
lands mid-stream and kills the client's JSON parser. Every diagnostic here goes
to **stderr** through `protocol.log()`, and a test asserts stdout carries
nothing but JSON-RPC.

The client names this failure explicitly when it sees it, because "invalid
JSON" sends people looking in the wrong place:

```
non-JSON on the protocol stream (a stray write to stdout in the server?)
```

### 2. A notification is never answered

A request has an `id` and must be answered exactly once. A **notification** has
none and must never be answered — replying is a protocol violation some clients
treat as fatal.

`0` is a valid id, so this is tested with `is None`, never for truthiness. That
is a real bug waiting in any implementation that writes `if request.id:`.

### 3. A tool failure is not a protocol failure

The distinction the whole design turns on:

| | mechanism | what it means |
|---|---|---|
| **protocol error** | JSON-RPC `error` object | malformed JSON, unknown method, bad params — the session may not continue |
| **tool error** | normal `result`, `isError: true` | the heading was not found, the source hash did not match — **the model must read this and recover** |

An agent cannot reason about a transport failure. Collapsing the two turns a
recoverable tool failure into a dead session, so `lookup_schedule` returning
`not_found` is a perfectly successful JSON-RPC call carrying a flagged result.

### 4. EOF is the shutdown signal

This MCP revision has no `shutdown` method. The client closes stdin and the
server exits. A server that blocks forever on a closed pipe is a zombie the
parent has to kill, so `serve_forever` reads until EOF, and SIGINT/SIGTERM are
handled so termination is clean rather than a stack trace — which, if any of it
reached stdout, would corrupt the stream on the way out.

### 5. Timeouts abandon, they do not kill

`tools/call` runs on a worker thread with a deadline. On expiry the client gets
a `timeout` tool error, flagged retryable.

**Stated plainly: the abandoned call keeps running.** Python cannot safely
terminate a thread, so the worker finishes and its result is discarded. That is
a real leak under sustained timeouts and it is documented rather than left to be
discovered.

---

## What crosses the boundary

`tools/list` advertises **the same schema object** the in-process registry
validates against — not a copy. A protocol surface that drifted from the local
one would accept calls the local tool rejects, and there is a test asserting
they are identical.

`tools/call` returns the full tool envelope as JSON inside a text block, since
this revision has no typed result channel:

```json
{"ok": true, "data": {...}, "evidence": [...], "error": null,
 "retryable": false, "tool_call_id": "c32d87d6cc17", "latency_ms": 4.2}
```

plus a `_meta` block carrying `tool_call_id`, `latency_ms`, `retryable` and
`error`, so a client can read them without parsing the text.

**`tool_call_id` is minted by the server** unless the client supplies one in
`_meta`. MCP has no such field of its own — the JSON-RPC `id` plays that role
for the transport — but the agent's contract wants one on the *result*, so the
mapping is made explicit here.

**`evidence` survives the crossing.** It is the untrusted channel every
injection defence is built on, and losing it across the protocol would silently
disarm all of them. Tested.

---

## The client, and why the boundary is switchable

`client.py` closes the loop. `build_registry_over_mcp()` returns the same
seven-tool registry with the two Gazette tools served across a process
boundary:

```python
from mcp_gazette.client import McpClient, build_registry_over_mcp

with McpClient() as client:
    registry = build_registry_over_mcp(client)   # same names, same schemas
    result = run_task(line, model=model, dispatcher=registry)
```

`Registry` cannot tell which side of the boundary a tool is on, and a test
asserts a local and a remote `lookup_schedule` return the same outcome for the
same arguments. That is what makes DESIGN §7's claim testable: **the boundary
can be switched off**, so "what did MCP cost me in latency" is a measurement
rather than a rhetorical question.

Every transport failure becomes a structured `ToolResult`, because the agent's
contract says a tool never raises and a tool that happens to live in another
process is no exception:

| what happened | code | retryable |
|---|---|---|
| server process died | `unavailable` | yes |
| no reply within the deadline | `timeout` | yes |
| reply was not JSON | `malformed` | **no** — the same call yields the same garbage |

One honest consequence: the remote spec is marked `pure=False`. The same
arguments now depend on a subprocess being alive, so it is idempotent only
because the cache makes it so — which is exactly the residual risk
`IdempotencyCache` documents.

---

## Why this tool

Of the seven, `lookup_schedule` is the right one to expose (DESIGN §7):

- **Its return type is genuinely rich.** `resolved` / `ambiguous` /
  `chapter_only` / `absent` are four different answers, so the schema exercises
  real design rather than returning a string.
- **It has real error responses** — source file missing, hash mismatch,
  malformed heading, a date outside the archive.
- **It is independently useful.** "Look an Indian tariff heading up in the
  archived Gazette and get a citable rate" is a thing other people want, and
  this server can be published on its own.

It also carries the branch the domain turns on, and the tests confirm it
survives the crossing: heading 2402 is **28%** on a November 2025 invoice and
**40%** on a March 2026 one, because Notification 19/2025 omitted Schedule VII
in between.

---

## Running it under another client

Any MCP client that speaks stdio can use it:

```json
{
  "mcpServers": {
    "gazette": {
      "command": "python",
      "args": ["-m", "mcp_gazette"],
      "cwd": "/path/to/gst-resilient-agent",
      "env": {"PYTHONPATH": "/path/to/gst-resilient-agent"}
    }
  }
}
```

`PYTHONPATH` matters: the server imports `agent.*` for the tool handlers, and a
client that spawns it from elsewhere will not have the repo importable
otherwise. `McpClient` sets it for exactly this reason.
