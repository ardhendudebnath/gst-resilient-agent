"""Tool 3 — look a heading up in the archived Gazette. The one exposed over MCP.

This is a document lookup, not a judgement, and the distinction is the reason
the workflow branches at all. **It refuses to resolve an ambiguous heading.**
7418 splits on whether an article is a household article of copper; 8711 on
engine capacity; 2202 on added sugar. Each of those is a determination about
the goods, not about the document, and a lookup that picked one would be
inventing an answer with a citation attached — the most convincing kind of
wrong.

Four outcomes, and the agent has to handle all four:

| outcome | meaning | what the agent must do |
|---|---|---|
| `resolved` | exactly one rate | use it |
| `ambiguous` | the heading attracts more than one rate | call `check_conditions` |
| `chapter_only` | the heading is absent, its chapter is specified | widen, or decline |
| `absent` | not in the notification at all | reconsider the heading |

**The source is verified before it is read.** A tool that answers from a
document it cannot prove is the pinned one is worse than a tool that refuses,
because its answer carries a citation. A SHA-256 mismatch returns
`source_mismatch`, which is not retryable — a hash does not un-change itself.

### Why this reads the schedules *as of a date*

An earlier version of this tool took only a heading and answered from the
current table, delegating to Project 01's `schedule_lookup`. Two measured
problems retired that:

1. **It cannot answer the question this project asks.** Upstream applies
   Notification 19/2025 unconditionally and drops every Schedule VII entry, so
   for heading 2402 it reports 40 %. On an invoice dated 12 November 2025 the
   lawful rate was **28 %** — Schedule VII was live for another eleven weeks —
   and no combination of upstream calls surfaces it. The boundary scenarios
   DESIGN.md §9 requires were unreachable.
2. **Its failure mode was silent.** With Project 01 absent, upstream's
   amendment table returns empty, and a heading that did move reported
   `moved: false` rather than `unavailable`. A confident wrong fact is worse
   than a refusal, and this is the repository that exists to say so.

So the lookup reads the vendored, hash-pinned PDFs directly through
`agent.gazette`, scoped to `on_date`. That is not a second definition of the
label space — `agent.gst` still owns which slabs exist and is still checked
against upstream by `tests/test_upstream_parity.py`. It is a capability
upstream does not have, and `tests/test_gazette.py` pins the agreement between
the two on every date where both are defined.
"""

from __future__ import annotations

import re
from datetime import date

from agent import gazette
from agent.contract import Evidence, ToolResult
from agent.registry import ToolSpec

#: Gazette entry text is trimmed to something quotable, but a long entry still
#: runs to a few hundred characters and there can be several. Capped so one
#: lookup cannot dominate the message history — an unbounded evidence field is
#: a token-budget failure waiting to happen, and week 4 will find it.
MAX_EVIDENCE_ITEMS = 8

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _source_failure(exc: Exception) -> ToolResult:
    code = "source_missing" if isinstance(exc, gazette.SourceMissing) else "source_mismatch"
    return ToolResult.err(
        code,
        f"{exc} Refusing to quote a rate from a document that is not the one "
        "this tool was built against.",
        data={"primary_dir": str(gazette.gazette_dir())},
    )


def lookup_schedule(heading: str, on_date: str) -> ToolResult:
    """Find a 4-digit heading in the schedules as they stood on `on_date`."""
    heading = (heading or "").strip()
    if not (len(heading) >= 4 and heading[:4].isdigit()):
        return ToolResult.err(
            "bad_argument",
            f"heading {heading!r} is not a tariff heading; give at least 4 digits",
        )
    heading = heading[:4]

    if not _ISO_DATE.match(on_date or ""):
        return ToolResult.err(
            "bad_argument",
            f"on_date {on_date!r} is not yyyy-mm-dd. The schedules are read as "
            "they stood on the invoice date, so the date is required.",
        )
    when = date.fromisoformat(on_date)

    try:
        match = gazette.lookup(heading, when)
    except gazette.NotArchived as exc:
        return ToolResult.err(
            "not_found",
            f"{exc} Decline rather than reading a rate out of a notification "
            "that had not been issued on the invoice date.",
            data={
                "in_archive": False,
                "archive_starts": gazette.NINE_2025_IN_FORCE.isoformat(),
                "on_date": on_date,
            },
        )
    except (gazette.SourceMissing, gazette.SourceMismatch) as exc:
        return _source_failure(exc)

    common = {
        "heading": heading,
        "on_date": on_date,
        "notification": gazette.regime(when),
        "schedule_vii_live": gazette.schedule_vii_live(when),
    }

    evidence = [
        Evidence(
            source="09-2025-CTR.pdf",
            locator=f"Schedule {e.schedule}, entry near heading {heading}",
            text=e.text,
        )
        for e in match.entries[:MAX_EVIDENCE_ITEMS]
    ]
    evidence += [
        Evidence(source="10-2025-CTR.pdf", locator=f"exemption entry near {heading}", text=t)
        for t in match.exempt_entries[: max(0, MAX_EVIDENCE_ITEMS - len(evidence))]
    ]

    entries = [
        {"schedule": e.schedule, "slab": e.slab, "kind": "rated"} for e in match.entries
    ]
    entries += [{"schedule": None, "slab": "0", "kind": "exempt"} for _ in match.exempt_entries]

    if match.ambiguous:
        return ToolResult.ok_(
            {
                **common,
                "outcome": "ambiguous",
                "slab": None,
                "entries": entries,
                "distinct_rates": sorted(
                    {str(e["slab"]) for e in entries if e["slab"] is not None}
                ),
                "detail": (
                    f"heading {heading} attracts more than one rate on {on_date}. "
                    "Which applies is a determination about the goods, not about "
                    "the document, and this tool will not make it. Call "
                    "check_conditions."
                ),
            },
            evidence,
        )

    if match.chapter_only:
        return ToolResult.ok_(
            {
                **common,
                "outcome": "chapter_only",
                "slab": None,
                "chapter": heading[:2],
                "chapter_entries": [
                    {"schedule": e.schedule, "slab": e.slab} for e in match.chapter_entries
                ],
                "detail": (
                    f"heading {heading} is not itself listed; chapter {heading[:2]} "
                    "is. A chapter entry is a pointer, not a determination — it "
                    "does not resolve a slab for this heading."
                ),
            },
            [
                Evidence(
                    source="09-2025-CTR.pdf",
                    locator=f"Schedule {e.schedule}, chapter {heading[:2]} entry",
                    text=e.text,
                )
                for e in match.chapter_entries[:MAX_EVIDENCE_ITEMS]
            ],
        )

    if not match.found:
        return ToolResult.ok_(
            {
                **common,
                "outcome": "absent",
                "slab": None,
                "detail": (
                    f"heading {heading} appears in neither the rated schedules of "
                    "9/2025 nor the exemption list of 10/2025 as they stood on "
                    f"{on_date}. Reconsider the heading before concluding there "
                    "is no rate."
                ),
            }
        )

    slab = match.slab
    detail = (
        f"heading {heading} resolves to {slab}% under "
        f"{common['notification']}, as in force on {on_date}."
    )
    if match.schedule == "VII":
        # Live, and about to stop being live. Saying both halves matters: the
        # rate is correct for this invoice, and asserting it as *current* in the
        # final opinion would fail the domain safety property.
        detail += (
            f" This is a Schedule VII entry, which Notification 19/2025 omitted "
            f"with effect from {gazette.NINETEEN_2025_IN_FORCE.isoformat()}. It "
            "governs this invoice because the invoice predates that date — but "
            "do not describe 28% as a rate that is current today."
        )

    return ToolResult.ok_(
        {
            **common,
            "outcome": "resolved",
            "slab": slab,
            "schedule": match.schedule,
            "entries": entries,
            "detail": detail,
        },
        evidence,
    )


SPEC = ToolSpec(
    name="lookup_schedule",
    description=(
        "Look a 4-digit tariff heading up in the archived Gazette notifications "
        "(9/2025 rated schedules and 10/2025 exemptions) AS THEY STOOD on the "
        "invoice date, and return a citable rate. Outcomes: 'resolved' (one "
        "rate), 'ambiguous' (more than one rate — this tool will NOT choose; "
        "call check_conditions), 'chapter_only' (the heading is absent but its "
        "chapter is listed — not a determination), or 'absent'. Dates before "
        "2025-09-22 are outside the archive and return not_found. Sources are "
        "SHA-256 verified before they are read."
    ),
    parameters={
        "type": "object",
        "properties": {
            "heading": {
                "type": "string",
                "description": "Tariff heading. 4 digits; longer codes are truncated to 4.",
                "pattern": r"^\d{4}",
                "maxLength": 12,
            },
            "on_date": {
                "type": "string",
                "description": (
                    "Invoice date, yyyy-mm-dd. The schedules are read as they "
                    "stood on this date, not as they stand today."
                ),
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
        },
        "required": ["heading", "on_date"],
        "additionalProperties": False,
    },
    handler=lookup_schedule,
    stage="lookup",
    returns_evidence=True,
    # Pure with respect to its arguments *given a fixed document set*, which the
    # SHA-256 check is what guarantees. Without that check this would be a
    # function of a file on disk, and the cache would be memoising a moving
    # target.
    pure=True,
)
