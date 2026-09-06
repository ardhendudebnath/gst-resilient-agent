"""Tool 6 — the arithmetic. Exact, and the reason the end state is checkable.

Nothing here is hard. It is here because a language model doing percentage
arithmetic on lakhs of rupees is a liability, and because this is where the
brief predicts the double-counting failure lands: the worked example in the
project brief is an agent that treated a duplicated tool response as a second
result and doubled a figure. That failure is a property of the *call path*, not
of this function, which is pure and will return the same answer every time it
is asked. Whether the agent asks twice and adds is what week 4 measures.

**Decimal, not float.** `0.18 * 250000` is not 45000 in binary floating point,
and a reliability project that reports a differential to the paisa cannot
afford to be approximately right about money.

**Rounding.** Half-up to two decimal places, which is the convention for
rupee-paisa amounts. Section 170 of the CGST Act requires rounding the tax on
an invoice to the nearest rupee; that applies to what a supplier puts on the
document, not to an exposure estimate, so it is not applied here. The
convention is named rather than assumed because it changes the number.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from agent import gazette
from agent.contract import ToolResult
from agent.opinion import KNOWN_SLABS
from agent.registry import ToolSpec

_PAISA = Decimal("0.01")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Every rate the archive contains, ordered for the schema enum. Wider than
#: Project 01's `VALID_SLABS` by exactly one entry — 28 % — because that slab
#: was lawful between 2025-09-22 and 2026-01-31 and an invoice from that window
#: has it as the *correct* answer. Whether a given rate is valid on a given day
#: is decided against `on_date` in the handler, not by membership here.
_SLAB_ENUM: list[str] = sorted(KNOWN_SLABS, key=lambda s: Decimal(s))


def _money(value: object) -> Decimal:
    """Decimal via str, never via float.

    `Decimal(0.1)` is 0.1000000000000000055511151231257827021181583404541015625.
    `Decimal("0.1")` is 0.1. JSON hands us a float, so the conversion goes
    through repr, which round-trips the value the caller meant.
    """
    return Decimal(str(value))


def _tax(value: Decimal, rate: str) -> Decimal:
    return (value * _money(rate) / Decimal(100)).quantize(_PAISA, rounding=ROUND_HALF_UP)


def compute_liability(
    taxable_value_inr: float,
    correct_slab: str,
    declared_rate: str,
    on_date: str,
) -> ToolResult:
    """Compute GST at the correct slab, at the declared rate, and the difference.

    `differential_inr` is positive when the supplier short-paid — the amount at
    risk — and negative when they over-collected.

    `on_date` is required because whether a slab is a lawful rate is a question
    about a date, not a constant. 28 % is the correct answer for an invoice
    raised in November 2025 and an abolished rate for one raised in March 2026.
    """
    try:
        value = _money(taxable_value_inr)
    except InvalidOperation:
        return ToolResult.err(
            "bad_argument", f"taxable_value_inr {taxable_value_inr!r} is not a number"
        )

    if value < 0:
        return ToolResult.err("bad_argument", "taxable_value_inr is negative")

    if not _ISO_DATE.match(on_date or ""):
        return ToolResult.err("bad_argument", f"on_date {on_date!r} is not yyyy-mm-dd")
    when = date.fromisoformat(on_date)

    if correct_slab not in KNOWN_SLABS:
        return ToolResult.err(
            "bad_argument",
            f"correct_slab {correct_slab!r} is not a GST rate; valid: {_SLAB_ENUM}",
        )

    # An abolished slab named as the *correct* one is not an argument error, it
    # is the failure this project exists to catch. It gets its own message so
    # it is unmistakable in a trace. Judged against the invoice date: naming
    # 28 % as correct is right for a November 2025 line and wrong for a March
    # 2026 one, and a date-blind check gets one of those two backwards.
    if gazette.slab_is_stale(correct_slab, when):
        ceased = gazette.SLAB_ABOLISHED_ON[correct_slab].isoformat()
        return ToolResult.err(
            "bad_argument",
            f"correct_slab {correct_slab!r} was abolished on {ceased}, before "
            f"the invoice date {on_date}, so it cannot be the rate in force. "
            "Look the heading up in the notification governing that date rather "
            "than recalling a rate.",
            data={"abolished": True, "since": ceased, "on_date": on_date},
        )

    # The declared rate is whatever the supplier put on the invoice, so an
    # abolished rate here is expected input, not an error — it is precisely the
    # case the suite is built around.
    try:
        declared = _money(declared_rate)
    except InvalidOperation:
        return ToolResult.err("bad_argument", f"declared_rate {declared_rate!r} is not a number")
    if declared < 0:
        return ToolResult.err("bad_argument", "declared_rate is negative")

    correct_tax = _tax(value, correct_slab)
    declared_tax = _tax(value, declared_rate)
    differential = (correct_tax - declared_tax).quantize(_PAISA, rounding=ROUND_HALF_UP)

    return ToolResult.ok_(
        {
            "taxable_value_inr": float(value),
            "correct_slab": correct_slab,
            "declared_rate": str(declared_rate),
            "correct_tax_inr": float(correct_tax),
            "declared_tax_inr": float(declared_tax),
            "differential_inr": float(differential),
            "declared_correct": differential == 0,
            "direction": (
                "short_paid" if differential > 0
                else "over_collected" if differential < 0
                else "correct"
            ),
            "on_date": on_date,
            # Flagged because it is the domain's signature error: a supplier
            # still billing a rate that had already been abolished on the day
            # they raised the invoice is using a table that no longer exists,
            # which is the same failure the models make. Scoped to the date, so
            # a November 2025 line billed at 28 % is not flagged — it was right.
            "declared_rate_abolished": gazette.slab_is_stale(str(declared_rate), when),
            "declared_rate_abolished_on": (
                gazette.SLAB_ABOLISHED_ON[str(declared_rate)].isoformat()
                if str(declared_rate) in gazette.SLAB_ABOLISHED_ON
                else None
            ),
        }
    )


SPEC = ToolSpec(
    name="compute_liability",
    description=(
        "Compute GST at the correct slab, at the rate the supplier declared, "
        "and the differential. Exact decimal arithmetic — do not do this "
        "yourself. differential_inr is positive when the supplier short-paid. "
        "Call once: the result is a pure function of its arguments, and "
        "calling again with the same arguments returns the same number, not an "
        "additional amount."
    ),
    parameters={
        "type": "object",
        "properties": {
            "taxable_value_inr": {
                "type": "number",
                "description": "Taxable value of the line, in rupees.",
                "minimum": 0,
            },
            "correct_slab": {
                "type": "string",
                "description": (
                    "The slab you determined from the notification in force on "
                    "the invoice date, e.g. '18'. A slab already abolished on "
                    "that date is rejected."
                ),
                "enum": _SLAB_ENUM,
            },
            "declared_rate": {
                "type": "string",
                "description": "The rate the supplier charged, as it appears on the invoice.",
                "pattern": r"^\d+(\.\d+)?$",
            },
            "on_date": {
                "type": "string",
                "description": (
                    "Invoice date, yyyy-mm-dd. Decides which rates were lawful."
                ),
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
        },
        "required": ["taxable_value_inr", "correct_slab", "declared_rate", "on_date"],
        "additionalProperties": False,
    },
    handler=compute_liability,
    stage="compute",
    returns_evidence=False,
    pure=True,
)
