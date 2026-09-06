"""Tool 5 — which notification governed this invoice, and did this heading move?

The branch the domain is actually about. India restructured GST twice in
seventeen months:

- **22 September 2025**, Notification 9/2025-CT(R) supersedes 1/2017 and the
  12 % slab ceases to exist.
- **1 February 2026**, Notification 19/2025-CT(R) omits Schedule VII and 28 %
  ceases to exist. It did not merely delete the schedule — it relocated every
  entry, and split one. Pan masala and most tobacco went to Schedule III at
  40 %; **biris went the other way, to Schedule II at 18 %.**

So the rate for heading 2403 on an invoice dated 15 January 2026 and on one
dated 15 March 2026 are different numbers, and for biris they are different in
the opposite direction from the rest of the chapter. No amount of prompting
gets that out of a model's weights; it comes from the amending notification or
it is a guess.

**Dates before 22 September 2025 are outside the archive, and the answer is a
refusal.** This is the one place where being unhelpful is the correct
behaviour. The superseded 1/2017 schedule is exactly the table models recite
from memory — it is the error Project 01 measures — so extrapolating backwards
from the documents we do have would be committing the failure this project
exists to detect. The tool says it cannot answer and means it.

### The rate on the date, not just the fact of a move

This tool reports `slab_on_date`: what the heading actually attracted on the
invoice date. That is not redundant with `lookup_schedule`, and leaving it out
was a measured hole rather than a hypothetical one. An earlier version reported
only *whether* a heading moved and what it moved *to*, which left the agent
told "the pre-amendment entry governs here" and never told what it was — the
28 % Schedule VII rate was unreachable through the whole tool set for exactly
the boundary invoices DESIGN.md §9 requires the suite to contain.

It also reported `moved: false` for a heading that did move, whenever the
amendment table was unavailable, because an empty table is indistinguishable
from no relocation. A confident wrong fact is worse than a refusal. The table
is now read from the vendored transcription in `agent.gazette`, which is
checked against the archived 19/2025 PDF by test and has no optional
dependency, so the case cannot arise.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from agent import gazette
from agent.contract import Evidence, ToolResult
from agent.registry import ToolSpec

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _slab_on(heading: str, when: date) -> tuple[str | None, str, tuple[str, ...]]:
    """(slab, outcome, distinct rates) for a heading on a date, without raising.

    The third element is what movement is actually detected on. Comparing only
    the resolved slab misses a heading that is ambiguous on *both* sides of the
    amendment: 2401 carries tobacco leaves at 5 % alongside unmanufactured
    tobacco, so it resolves to None before and after while the rates underneath
    go from {5, 28} to {5, 40}. On the slab alone that reads as "did not move",
    which is the precise class of confident wrong fact this tool exists not to
    produce.
    """
    try:
        match = gazette.lookup(heading, when)
    except gazette.NotArchived:
        return None, "not_archived", ()
    except (gazette.SourceMissing, gazette.SourceMismatch):
        return None, "source_unverified", ()

    rates = {e.slab for e in match.entries if e.slab is not None}
    if match.exempt_entries:
        rates.add("0")
    if not match.entries and not match.exempt_entries:
        rates = {e.slab for e in match.chapter_entries if e.slab is not None}
    ordered = tuple(sorted(rates, key=lambda s: float(s)))

    if match.ambiguous:
        return None, "ambiguous", ordered
    if match.chapter_only:
        return None, "chapter_only", ordered
    if not match.found:
        return None, "absent", ordered
    return match.slab, "resolved", ordered


def rate_history(heading: str, on_date: str) -> ToolResult:
    """Report which rate notification was in force on a date, what the heading
    attracted then, and whether the 1 February 2026 amendment moved it."""
    heading = (heading or "").strip()[:4]
    if not heading.isdigit() or len(heading) != 4:
        return ToolResult.err(
            "bad_argument", f"heading {heading!r} is not a 4-digit tariff heading"
        )
    if not _ISO_DATE.match(on_date or ""):
        return ToolResult.err("bad_argument", f"on_date {on_date!r} is not yyyy-mm-dd")

    when = date.fromisoformat(on_date)

    try:
        notification = gazette.regime(when)
    except gazette.NotArchived:
        return ToolResult.err(
            "not_found",
            f"invoice date {on_date} precedes "
            f"{gazette.NINE_2025_IN_FORCE.isoformat()}, when 9/2025-CT(R) came "
            "into force. The schedule that governed that date (1/2017-CT(R), as "
            "amended) is not archived in this repository. Do not infer the rate "
            "from a later notification — decline instead.",
            data={
                "in_archive": False,
                "archive_starts": gazette.NINE_2025_IN_FORCE.isoformat(),
                "on_date": on_date,
            },
        )

    try:
        gazette.verify_sources()
    except (gazette.SourceMissing, gazette.SourceMismatch) as exc:
        code = "source_missing" if isinstance(exc, gazette.SourceMissing) else "source_mismatch"
        return ToolResult.err(code, str(exc), data={"on_date": on_date})

    boundary = gazette.NINETEEN_2025_IN_FORCE
    amended = when >= boundary

    slab_now, outcome_now, rates_now = _slab_on(heading, when)
    slab_before, outcome_before, rates_before = _slab_on(heading, boundary - timedelta(days=1))
    slab_after, outcome_after, rates_after = _slab_on(heading, boundary)
    moved = (slab_before, outcome_before, rates_before) != (
        slab_after,
        outcome_after,
        rates_after,
    )

    moves = gazette.AMENDED_2026.get(heading, [])
    entries = [
        {
            "sub_heading": sub or heading,
            "schedule_from_2026_02_01": schedule,
            "slab_from_2026_02_01": gazette.SCHEDULE_SLAB.get(schedule),
        }
        for sub, schedule, _ in moves
    ]
    evidence = [
        Evidence(
            source="19-2025-CTR.pdf",
            locator=f"Schedule {schedule}, entry for {sub or heading}",
            text=text,
        )
        for sub, schedule, text in moves
    ]

    data = {
        "heading": heading,
        "on_date": on_date,
        "notification_in_force": notification,
        "in_archive": True,
        "amendment_applies": amended,
        # What the heading actually attracted on the invoice date. This is the
        # number the audit turns on.
        "slab_on_date": slab_now,
        "outcome_on_date": outcome_now,
        "schedule_vii_live": gazette.schedule_vii_live(when),
        "moved_on_2026_02_01": moved,
        "slab_before_amendment": slab_before,
        "slab_after_amendment": slab_after,
        # The rates available on each side. Populated even when the heading is
        # ambiguous and has no single slab, which is the case where the slab
        # fields alone say nothing.
        "rates_before_amendment": list(rates_before),
        "rates_after_amendment": list(rates_after),
        "rates_on_date": list(rates_now),
        "entries_after_amendment": entries,
        "slabs_abolished_by_this_date": sorted(
            s for s in gazette.SLAB_ABOLISHED_ON if gazette.slab_is_stale(s, when)
        ),
        "milestones": {
            "09-2025 in force": gazette.NINE_2025_IN_FORCE.isoformat(),
            "19-2025 in force": boundary.isoformat(),
        },
    }

    def _rate_phrase(slab: str | None, rates: tuple[str, ...]) -> str:
        """Describe a side of the boundary whether or not it resolves.

        A heading that is ambiguous on both sides still moved, and saying
        "it gives None%" there would be worse than saying nothing.
        """
        if slab is not None:
            return f"{slab}%"
        if rates:
            return "one of " + ", ".join(f"{r}%" for r in rates)
        return "no rated entry"

    if moved and not amended:
        # The heading moves, but not yet. Saying so explicitly matters: an agent
        # that sees a 2026 entry and applies it to a January invoice has made
        # the same error as one reciting an abolished rate, in the other
        # direction.
        data["note"] = (
            f"heading {heading} was relocated by 19/2025 with effect from "
            f"{boundary.isoformat()}, which is AFTER this invoice date. The "
            "pre-amendment entry governs here and it gives "
            f"{_rate_phrase(slab_before, rates_before)}. Do not apply the "
            f"{_rate_phrase(slab_after, rates_after)} that takes effect later."
        )
    elif moved and amended and len(moves) > 1:
        data["note"] = (
            f"19/2025 split heading {heading} across schedules — the sub-heading "
            "determines the rate. Settle which sub-heading applies with "
            "check_conditions before choosing a slab."
        )
    elif moved and amended:
        data["note"] = (
            f"heading {heading} moved at {boundary.isoformat()} "
            f"({_rate_phrase(slab_before, rates_before)} → "
            f"{_rate_phrase(slab_after, rates_after)}). This invoice is on or "
            "after that date, so the later entry governs."
        )

    return ToolResult.ok_(data, evidence)


SPEC = ToolSpec(
    name="rate_history",
    description=(
        "Report which GST rate notification was in force on the invoice date, "
        "what the heading attracted on that date (slab_on_date), and whether it "
        "was moved by the 1 February 2026 amendment (19/2025-CT(R), which "
        "abolished the 28% slab, sent tobacco and pan masala to 40%, and biris "
        "to 18%). Returns not_found for invoice dates before 2025-09-22, which "
        "are outside this repository's archive — when that happens, decline to "
        "give a rate rather than inferring one from a later notification."
    ),
    parameters={
        "type": "object",
        "properties": {
            "heading": {
                "type": "string",
                "description": "4-digit tariff heading, e.g. '2403'.",
                "pattern": r"^\d{4}$",
            },
            "on_date": {
                "type": "string",
                "description": "Invoice date, yyyy-mm-dd.",
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
        },
        "required": ["heading", "on_date"],
        "additionalProperties": False,
    },
    handler=rate_history,
    stage="history",
    returns_evidence=True,
    pure=True,
)
