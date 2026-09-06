"""The Gazette MCP server: Indian GST tariff lookups over the protocol.

Independently runnable, with no dependencies at all:

    python -m mcp_gazette                # speak MCP over stdio
    python -m mcp_gazette --list-tools   # print the schemas and exit

Exposes the two tools that read the hash-pinned Gazette notifications. In the
brief's contract-review framing these are `lookup_statute` - look a rule up in
the primary source and return something citable - and `lookup_statute` is
accepted as an alias for `lookup_schedule` so a client written against the
generic name works unchanged.

The protocol is implemented directly: JSON-RPC 2.0 over newline-delimited
stdio, in `protocol.py`, stdlib only. Not because an SDK would be wrong, but
because this project has to be able to see, time and deliberately break the
boundary, and `docs/MCP.md` explains it on the assumption you can read the code
underneath it.

`client.py` closes the loop: `build_registry_over_mcp` returns the same
seven-tool registry with the two Gazette tools served across a process
boundary, so the agent runs identically either way and the cost of the boundary
becomes measurable.
"""

from mcp_gazette.protocol import PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION

__all__ = ["PROTOCOL_VERSION", "SERVER_NAME", "SERVER_VERSION"]
