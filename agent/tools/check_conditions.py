"""Tool 4 — settle an ambiguous heading from the description.

`lookup_schedule` refuses to resolve a heading that appears at more than one
rate, because which entry applies is a question about the goods. This tool asks
that question, and its most important answer is **"the description does not
say"** — `not_determinable`, reason `rate-fact-absent`, which is a correct
outcome rather than a failure.

### The rules are explicit, and the coverage is closed

There is no general condition-resolver here and there could not be: deciding
whether quartz slabs are "articles of artificial stone" is the classification
problem itself. What this covers is the small set of headings where the split
turns on **one stated fact** — a capacity, a sweetener, a use — and that fact is
either in the description or it is not.

Every rule was written by reading the entry text the lookup returns, and
`tests/test_conditions_match_gazette.py` asserts each rule's `describes` string
still appears in the archived notification. A rule that drifts from the
document it encodes is worse than no rule, because it keeps answering.

### Why it takes a date

The splits are not permanent features of a heading. 2403 has no split at all
before 1 February 2026 — the whole heading sat in Schedule VII at 28 % — and
offering the biris/other distinction for a 2025 invoice would resolve something
that did not yet exist, answering 18 % where the lawful rate was 28 %.

A heading with no rule gets one of two answers, and the difference matters. If
the lookup calls it ambiguous, `not_covered` means this tool cannot settle it
and the agent should decline rather than pick the cheaper rate. If the lookup
resolves it cleanly, `not_ambiguous` says so — the agent did not need this tool,
and telling it to decline there would turn a correct run into a wrong one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from agent import gazette
from agent.contract import Evidence, ToolResult
from agent.registry import ToolSpec

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class Condition:
    """One limb of a split heading, and how to tell whether it applies."""

    name: str
    slab: str
    schedule: str
    #: The entry text in the notification this limb corresponds to. Checked
    #: against the archived PDF by test, so it cannot drift.
    describes: str
    pattern: re.Pattern[str] | None = None
    #: True (applies), False (does not), or None (the fact is absent from the
    #: description) — and None is the answer that matters.
    predicate: Callable[[str], bool | None] | None = None

    def test(self, text: str) -> bool | None:
        if self.predicate is not None:
            return self.predicate(text)
        assert self.pattern is not None
        return bool(self.pattern.search(text))


_CC = re.compile(r"(\d{2,5})\s*(?:cc\b|cm3|cubic centimet|c\.c\.)", re.I)


def _engine_over_350(text: str) -> bool | None:
    """True over 350 cc, False at or under, None when no capacity is stated.

    The entries split on "not exceeding 350 cc" against "exceeding 350 cc", so
    exactly 350 falls in the lower limb. A line that never states a capacity is
    the `rate-fact-absent` case and must not be guessed at from words like
    "cruiser" or "sports bike".
    """
    m = _CC.search(text)
    if not m:
        return None
    return int(m.group(1)) > 350


#: Splits that turn on one stated fact, keyed by 4-digit heading. `2403` is
#: date-gated in `_rules_for`, not here, so this table stays a description of
#: the entries rather than of the calendar.
CONDITIONS: dict[str, tuple[Condition, ...]] = {
    "7418": (
        Condition(
            name="household_article_of_copper",
            slab="5",
            schedule="I",
            describes="Table, kitchen or other household articles of copper; Utensils",
            pattern=re.compile(
                r"\b(?:utensil|table ?ware|kitchen ?ware|household|cookware|"
                r"handi|kadai|kadhai|thali|tumbler|degchi|patila|lota|"
                r"cooking vessel|water bottle)\w*\b",
                re.I,
            ),
        ),
        Condition(
            name="other_articles_of_copper",
            slab="18",
            schedule="II",
            describes=(
                "All goods (other than table, kitchen or other household "
                "articles of copper; Utensils)"
            ),
            pattern=re.compile(
                r"\b(?:pipe|tube|fitting|busbar|bus ?bar|winding|conductor|"
                r"sheet|strip|foil|rod|wire|industrial|plumbing|roofing)\w*\b",
                re.I,
            ),
        ),
    ),
    "8711": (
        Condition(
            name="not_exceeding_350cc",
            slab="18",
            schedule="II",
            describes=(
                "Motorcycles (including mopeds) and cycles fitted with an "
                "auxiliary motor not exceeding 350 cc"
            ),
            predicate=lambda t: (None if (v := _engine_over_350(t)) is None else not v),
        ),
        Condition(
            name="exceeding_350cc",
            slab="40",
            schedule="III",
            describes="Motorcycles of engine capacity exceeding 350 cc",
            predicate=_engine_over_350,
        ),
    ),
    "2202": (
        Condition(
            name="plant_based_milk_drink",
            slab="5",
            schedule="I",
            describes="Plant-based milk drinks, ready for direct consumption as beverages",
            pattern=re.compile(
                r"\b(?:soya? milk|almond milk|oat milk|plant[- ]based milk|"
                r"coconut milk drink)\b",
                re.I,
            ),
        ),
        Condition(
            name="added_sugar_or_aerated",
            slab="40",
            schedule="III",
            describes=(
                "All goods (including aerated waters), containing added sugar "
                "or other sweetening matter or flavoured"
            ),
            pattern=re.compile(
                r"\b(?:aerated|carbonated|added sugar|sweeten\w*|flavour\w*|"
                r"flavor\w*|cola|soft drink|energy drink)\b",
                re.I,
            ),
        ),
    ),
    "9608": (
        Condition(
            name="pen",
            slab="18",
            schedule="II",
            describes=(
                "Ball point pens; felt tipped and other porous -tipped pens and "
                "markers; fountain pens"
            ),
            pattern=re.compile(
                r"\b(?:pen|pens|ball ?point|fountain pen|marker|stylo|"
                r"roller ?ball|gel pen)\b",
                re.I,
            ),
        ),
        Condition(
            name="pencil_or_crayon_exempt",
            slab="0",
            schedule="EXEMPT",
            describes=(
                "Pencils (including propelling or sliding pencils), crayons, "
                "pastels, drawing charcoals, writing or drawing chalks"
            ),
            pattern=re.compile(
                r"\b(?:pencil|crayon|pastel|charcoal|chalk|slate pencil)\w*\b",
                re.I,
            ),
        ),
    ),
    "2403": (
        Condition(
            name="biris",
            slab="18",
            schedule="II",
            describes="Biris",
            pattern=re.compile(r"\b(?:biri|beedi|bidi|biris|beedis|bidis)\b", re.I),
        ),
        Condition(
            name="other_manufactured_tobacco",
            slab="40",
            schedule="III",
            describes=(
                "Other manufactured tobacco and manufactured tobacco substitutes"
            ),
            pattern=re.compile(
                r"\b(?:chewing tobacco|zarda|khaini|hookah|hukka|snuff|"
                r"gutkha|smoking mixture|jarda|cigarette tobacco)\b",
                re.I,
            ),
        ),
    ),
}


def _rules_for(heading: str, when: date) -> tuple[Condition, ...]:
    """The conditions that exist on this heading on `when`.

    2403 only splits once 19/2025 is in force. Before that the whole heading
    sat in Schedule VII at 28 %, so offering the biris/other split for a 2025
    invoice would answer 18 % where the lawful rate was 28 %.
    """
    if heading == "2403" and when < gazette.NINETEEN_2025_IN_FORCE:
        return ()
    return CONDITIONS.get(heading, ())


def check_conditions(heading: str, description: str, on_date: str) -> ToolResult:
    """Settle a heading that lookup_schedule reported as ambiguous."""
    heading = (heading or "").strip()[:4]
    if not heading.isdigit() or len(heading) != 4:
        return ToolResult.err(
            "bad_argument", f"heading {heading!r} is not a 4-digit tariff heading"
        )
    if not (description or "").strip():
        return ToolResult.err("bad_argument", "description is required")
    if not _ISO_DATE.match(on_date or ""):
        return ToolResult.err("bad_argument", f"on_date {on_date!r} is not yyyy-mm-dd")

    when = date.fromisoformat(on_date)
    rules = _rules_for(heading, when)

    if not rules:
        try:
            match = gazette.lookup(heading, when)
            ambiguous, resolved = match.ambiguous, match.slab
        except Exception:  # noqa: BLE001 — lookup_schedule reports this properly
            ambiguous, resolved = True, None

        if not ambiguous:
            return ToolResult.ok_(
                {
                    "heading": heading,
                    "on_date": on_date,
                    "outcome": "not_ambiguous",
                    "resolved": False,
                    "slab_from_lookup": resolved,
                    "detail": (
                        f"heading {heading} is not ambiguous on {on_date} — "
                        f"lookup_schedule resolves it to {resolved}%. Nothing "
                        "needs settling; use that result and continue."
                    ),
                }
            )

        return ToolResult.ok_(
            {
                "heading": heading,
                "on_date": on_date,
                "outcome": "not_covered",
                "resolved": False,
                "covered_headings": sorted(CONDITIONS),
                "detail": (
                    f"heading {heading} is ambiguous on {on_date} but no "
                    "condition rule is encoded for it, so this tool cannot "
                    "settle it. Decline — terminal 'unanswerable', reason "
                    "'rate-fact-absent' — rather than choosing between the rates."
                ),
            }
        )

    matched = [c for c in rules if c.test(description) is True]

    if len(matched) == 1:
        winner = matched[0]
        return ToolResult.ok_(
            {
                "heading": heading,
                "on_date": on_date,
                "outcome": "resolved",
                "resolved": True,
                "slab": winner.slab,
                "schedule": winner.schedule,
                "condition": winner.name,
                "detail": (
                    f"the description satisfies the {winner.name!r} limb of "
                    f"heading {heading}, which is rated {winner.slab}%."
                ),
            },
            [
                Evidence(
                    source="09-2025-CTR.pdf",
                    locator=f"Schedule {winner.schedule}, heading {heading}",
                    text=winner.describes,
                )
            ],
        )

    if len(matched) > 1:
        return ToolResult.ok_(
            {
                "heading": heading,
                "on_date": on_date,
                "outcome": "not_determinable",
                "resolved": False,
                "reason": "rate-fact-absent",
                "conflicting_conditions": [
                    {"condition": c.name, "slab": c.slab} for c in matched
                ],
                "detail": (
                    "the description satisfies more than one limb of this "
                    "heading, so it does not determine a single rate. Terminal "
                    "state 'unanswerable', reason 'rate-fact-absent'."
                ),
            }
        )

    return ToolResult.ok_(
        {
            "heading": heading,
            "on_date": on_date,
            "outcome": "not_determinable",
            "resolved": False,
            "reason": "rate-fact-absent",
            "conditions_tested": [
                {
                    "condition": c.name,
                    "slab": c.slab,
                    "fact_absent": c.test(description) is None,
                }
                for c in rules
            ],
            "detail": (
                "the description does not state the fact this heading splits "
                "on. Terminal state 'unanswerable', reason 'rate-fact-absent'. "
                "Do not pick a rate."
            ),
        }
    )


SPEC = ToolSpec(
    name="check_conditions",
    description=(
        "Settle a heading that lookup_schedule reported as ambiguous, by testing "
        "the goods description against the condition each schedule entry turns "
        "on (engine capacity for 8711, household use for 7418, added sugar for "
        "2202, pens vs pencils for 9608, biris for 2403). Returns 'resolved' "
        "with a slab, 'not_determinable' when the description does not state the "
        "deciding fact (terminal: unanswerable, reason 'rate-fact-absent'), "
        "'not_covered' when no rule is encoded, or 'not_ambiguous' when the "
        "heading did not need settling. Never guess a rate from this tool's "
        "silence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "heading": {
                "type": "string",
                "description": "The 4-digit heading to settle, e.g. '8711'.",
                "pattern": r"^\d{4}$",
            },
            "description": {
                "type": "string",
                "description": "The goods description to test the conditions against.",
                "maxLength": 20000,
            },
            "on_date": {
                "type": "string",
                "description": (
                    "Invoice date, yyyy-mm-dd. Some splits only exist from "
                    "2026-02-01."
                ),
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
        },
        "required": ["heading", "description", "on_date"],
        "additionalProperties": False,
    },
    handler=check_conditions,
    stage="conditions",
    returns_evidence=True,
    pure=True,
)
