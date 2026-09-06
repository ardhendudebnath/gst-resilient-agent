"""A multi-step GST rate-opinion agent, built to be broken on purpose.

The agent is the setup. `chaos/` is the punchline: a middleware layer that sits
between the loop and its tools and injects eleven failure modes on demand, and
`FAILURES.md` is the taxonomy of what that turned up.

Module map:

    contract.py         what a tool returns; trusted vs untrusted data
    registry.py         the one path a tool is ever called through
    budget.py           hard limits, set before the first chaos run
    trace.py            the run trace, written from the first commit
    jsonschema_lite.py  argument validation with no dependencies
    gst.py              the seam onto Project 01, and source integrity
    tools/              the seven tools

Read `docs/DESIGN.md` first: it pins the workflow, the success criteria and
what the baseline deliberately does not have, and it was written before any of
this so that nothing in it can be retrofitted to the results.
"""

__version__ = "0.1.0"
