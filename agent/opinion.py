"""The final object, its schema, and the validator the scorer shares.

Kept at the agent level rather than inside tool 7 because three things need the
same definition â€” the tool that emits it, the scorer that grades it, and the
trace viewer that renders it â€” and three copies of a schema is three schemas.

### The divergence from Project 01, stated plainly

Project 01 asks "what is the rate **now**", so its label space is the current
table and `ABOLISHED_SLABS = {12, 28}` is a flat constant. This project audits
invoices, and an invoice raised on 12 November 2025 was raised into a table
where Schedule VII was live and **28 % was a lawful rate**. Treating 28 % as
abolished for that line would mark a compliant supplier non-compliant.

So validity here is a function of the invoice date, not a constant:

    12 %  never valid on any date this archive covers â€” it had no successor
          schedule in 9/2025 at all.
    28 %  valid from 2025-09-22 to 2026-01-31, abolished from 2026-02-01.

That means Project 01's `find_abolished_citations` â€” which is date-blind, and
correct to be, for the question it asks â€” **cannot be reused unmodified** for
DESIGN.md Â§5 criterion 6. It is used for the date-independent half (12 %, and
the historical-language detection, which is the hard part and worth reusing)
and the 28 % case is decided here against the invoice date. Where the two
disagree, this module wins and the disagreement is recorded in the result row,
because a silent divergence between two definitions of "abolished" is the kind
of thing that makes a benchmark wrong in a way nobody notices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agent import gazette

#: Terminal states a run may report. `budget_exhausted` is set by the loop, not
#: by the model, and is never a valid `draft_opinion` argument.
TERMINALS: frozenset[str] = frozenset({"opinion", "out_of_scope", "unanswerable"})

#: Rates that exist in the archived schedules at some point in the covered
#: window. Whether a given one is valid on a given date is `slab_valid_on`.
KNOWN_SLABS: frozenset[str] = frozenset(
    {"0", "0.25", "1.5", "3", "5", "18", "28", "40"}
)

#: Mirrors Project 01's `harness.schema.UNANSWERABLE_REASONS`, plus the two
#: this workflow adds. `tests/test_scope_parity.py` asserts the shared five are
#: identical whenever Project 01 is installed.
UNANSWERABLE_REASONS: frozenset[str] = frozenset(
    {
        # --- shared with Project 01 -----------------------------------------
        "no-product-kind",
        "model-number-only",
        "rate-fact-absent",
        "packaging-indeterminate",
        "multi-good-no-dominant",
        # --- added by the audit framing --------------------------------------
        # The invoice predates the archived notifications.
        "date-outside-archive",
        # The heading resolved to nothing in the schedules.
        "heading-not-in-schedule",
    }
)

OUT_OF_SCOPE_REASONS: frozenset[str] = frozenset({"alcoholic-liquor"})

_HSN4 = re.compile(r"^\d{4}$")

#: Language that marks a rate as historical rather than asserted as current.
#: Borrowed in spirit from Project 01's `find_abolished_citations`, which
#: already knows that "the erstwhile 28 % rate no longer applies" is a correct
#: sentence and must not be scored as reciting a dead schedule.
_HISTORICAL = re.compile(
    r"\b(?:erstwhile|former|previous|prior|old|superseded|abolished|omitted|"
    r"repealed|no longer|until|before|up to|till|withdrawn|ceased|"
    r"pre-?amendment|then-?applicable|at the time)\b",
    re.I,
)

#: Verbs that frame a rate as *what the supplier did*, not as what is correct.
#: "The supplier declared 12%" is the audit finding itself â€” the whole point of
#: the workflow â€” and reading it as asserting 12 % to be current would score the
#: correct answer as a domain-safety violation.
#:
#: Found by the Â§1 worked example failing its own validator on the first run.
#: The window is tight (a few words, immediately before the number) because
#: `declared` also appears in sentences that really do assert a rate, such as
#: "the rate declared herein is 28%", and those must still be caught.
_DECLARED = re.compile(
    r"\b(?:declared|charged|collected|invoiced|levied|billed|paid|applied|"
    r"remitted|discharged)\b(?:\s+(?:at|a|an|the|rate|of|gst))*\s*$",
    re.I,
)

_RATE_MENTION = re.compile(r"(?<![\d.])(12|28)\s*(?:%|per ?cent)", re.I)


def slab_valid_on(slab: str, when: date) -> bool:
    """Was `slab` a lawful GST rate on `when`?"""
    return slab in KNOWN_SLABS and not gazette.slab_is_stale(slab, when)


def stale_rate_mentions(text: str, when: date) -> list[dict[str, Any]]:
    """Mentions of a rate that did not exist on `when`, minus historical ones.

    A sentence that names a dead rate *as dead* is correct and common in real
    tax writing, so flagging every occurrence of "28 %" would score the honest
    answer as the failure. The window is deliberately generous â€” 90 characters
    before the mention â€” because the hedge usually precedes the number and a
    tighter window misses "which was withdrawn with effect from 1 February
    2026, when the 28 % schedule was omitted".
    """
    findings: list[dict[str, Any]] = []
    for m in _RATE_MENTION.finditer(text or ""):
        rate = m.group(1)
        if not gazette.slab_is_stale(rate, when):
            continue
        window = text[max(0, m.start() - 90) : m.end() + 40]
        if _HISTORICAL.search(window):
            continue
        # Immediately-preceding declaration verb: "the supplier declared 12%".
        if _DECLARED.search(text[max(0, m.start() - 45) : m.start()]):
            continue
        findings.append(
            {
                "rate": rate,
                "abolished_on": gazette.SLAB_ABOLISHED_ON[rate].isoformat(),
                "excerpt": re.sub(r"\s+", " ", window).strip(),
            }
        )
    return findings


@dataclass(slots=True)
class Citation:
    notification: str
    schedule: str
    heading: str

    def to_json(self) -> dict[str, str]:
        return {
            "notification": self.notification,
            "schedule": self.schedule,
            "heading": self.heading,
        }


@dataclass(slots=True)
class Opinion:
    """The agent's final answer on one invoice line."""

    terminal: str
    line_id: str = ""
    invoice_date: str = ""
    hsn4: str | None = None
    slab: str | None = None
    answerable: bool = True
    declared_correct: bool | None = None
    differential_inr: str | None = None
    reason: str | None = None
    citations: list[Citation] = field(default_factory=list)
    justification: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "terminal": self.terminal,
            "line_id": self.line_id,
            "invoice_date": self.invoice_date,
            "hsn4": self.hsn4,
            "slab": self.slab,
            "answerable": self.answerable,
            "declared_correct": self.declared_correct,
            "differential_inr": self.differential_inr,
            "reason": self.reason,
            "citations": [c.to_json() for c in self.citations],
            "justification": self.justification,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Opinion":
        cites = [
            Citation(
                notification=str(c.get("notification", "")),
                schedule=str(c.get("schedule", "")),
                heading=str(c.get("heading", "")),
            )
            for c in (obj.get("citations") or [])
            if isinstance(c, dict)
        ]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in obj.items() if k in known and k != "citations"}
        return cls(**payload, citations=cites)


def validate(opinion: Opinion, *, invoice_date: date) -> list[str]:
    """Return a list of problems; empty means the object is well-formed.

    Well-formed is not the same as correct â€” this checks the shape and the
    domain invariants, not whether the slab is the right one. The scorer does
    that against gold.
    """
    errs: list[str] = []

    if opinion.terminal not in TERMINALS:
        errs.append(
            f"terminal {opinion.terminal!r} not in {sorted(TERMINALS)}"
        )
        return errs  # nothing else can be checked coherently

    if opinion.terminal == "opinion":
        if not opinion.answerable:
            errs.append("terminal is 'opinion' but answerable is false")
        if opinion.hsn4 is None or not _HSN4.match(str(opinion.hsn4)):
            errs.append(f"hsn4 {opinion.hsn4!r} is not a 4-digit heading")
        if opinion.slab is None:
            errs.append("terminal is 'opinion' but no slab was given")
        elif str(opinion.slab) not in KNOWN_SLABS:
            errs.append(
                f"slab {opinion.slab!r} is not a GST rate in {sorted(KNOWN_SLABS)}"
            )
        elif not slab_valid_on(str(opinion.slab), invoice_date):
            ceased = gazette.SLAB_ABOLISHED_ON[str(opinion.slab)]
            errs.append(
                f"slab {opinion.slab}% was abolished on {ceased.isoformat()}, "
                f"before the invoice date {invoice_date.isoformat()}"
            )
        if opinion.differential_inr is None:
            errs.append("terminal is 'opinion' but no differential was given")
        if not opinion.citations:
            errs.append("an opinion must cite at least one schedule entry")
        for c in opinion.citations:
            if not _HSN4.match(c.heading or ""):
                errs.append(f"citation heading {c.heading!r} is not a 4-digit heading")
            if not c.notification.strip():
                errs.append("citation is missing a notification")

    elif opinion.terminal == "out_of_scope":
        if opinion.answerable:
            errs.append("terminal is 'out_of_scope' but answerable is true")
        if opinion.reason not in OUT_OF_SCOPE_REASONS:
            errs.append(
                f"out_of_scope reason {opinion.reason!r} not in "
                f"{sorted(OUT_OF_SCOPE_REASONS)}"
            )
        if opinion.slab is not None:
            errs.append("out_of_scope lines have no slab; got " f"{opinion.slab!r}")

    elif opinion.terminal == "unanswerable":
        if opinion.answerable:
            errs.append("terminal is 'unanswerable' but answerable is true")
        if opinion.reason not in UNANSWERABLE_REASONS:
            errs.append(
                f"unanswerable reason {opinion.reason!r} not in "
                f"{sorted(UNANSWERABLE_REASONS)}"
            )
        if opinion.slab is not None:
            errs.append(f"unanswerable lines have no slab; got {opinion.slab!r}")

    # Domain safety property, checked on every terminal: DESIGN.md Â§5 (6).
    stale = stale_rate_mentions(opinion.justification, invoice_date)
    for finding in stale:
        errs.append(
            f"justification asserts {finding['rate']}% as current, a rate "
            f"abolished on {finding['abolished_on']}: â€œ{finding['excerpt']}â€"
        )

    return errs
