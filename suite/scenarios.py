"""The task suite. See docs/DESIGN.md §9.

Two populations, scored in separate columns and never averaged into one number.

**Derived scenarios** turn each of Project 01's 28 golden rows into an invoice
line by attaching four fields the golden set does not have: a declared heading,
a declared rate, a taxable value and an invoice date. Those four are
*constructed, not observed*, and the construction is deterministic from the row
id so the suite is reproducible without storing it.

**Synthetic scenarios** exist to reach branches the golden set cannot: an
out-of-scope line, an under-specified line for each reason code, an invoice
predating the archive, the 1 February 2026 boundary, and the conditional
headings. Marked `synthetic: true`.

A success rate that mixed 28 real classification problems with 18 scenarios
written to exercise the author's own branches is not one number, and is not
reported as one.

### One trap avoided, and it is worth naming

Project 01 labelled its rows against the **current** table — the slab that
applies today, after Notification 19/2025. So a derived scenario may only carry
a **post-amendment invoice date**, or the gold slab would be wrong for it: on a
November 2025 invoice, heading 2402 is 28 %, and the golden set says 40 %.

Every scenario whose date sits before 2026-02-01 is therefore synthetic, and
its expected slab is read from the archived Gazette at that date rather than
inherited from the golden label. That is a lookup rather than a judgement, and
it is disclosed here rather than left for someone to discover in the numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterator

from agent import gazette
from agent.config import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "data" / "golden.jsonl"

PAISA = Decimal("0.01")

#: Declared rates a supplier might have used. Weighted toward the stale table
#: on purpose: `docs/DESIGN.md` §1 says the suite is built from lines where a
#: supplier used the pre-2025 rates, because that is the error the domain
#: actually has and the one Project 01 measured models reproducing.
DECLARED_RATES: tuple[str, ...] = ("12", "12", "28", "18", "5", "18")

#: Taxable values, in rupees. Spread across magnitudes so a rounding error at
#: one scale cannot hide.
TAXABLE_VALUES: tuple[str, ...] = (
    "25000.00", "118500.50", "250000.00", "999999.99", "4750.25", "1875000.00",
)

#: Invoice dates for derived scenarios. All on or after 2026-02-01, so the
#: golden label — which describes the current table — is the correct answer.
POST_AMENDMENT_DATES: tuple[str, ...] = (
    "2026-02-01", "2026-02-17", "2026-03-14", "2026-04-30", "2026-06-09",
)


def _pick(seed: str, options: tuple[str, ...]) -> str:
    """Deterministic choice from a row id. Reproducible without a stored file."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _differential(taxable: str, declared: str, correct: str) -> str:
    value = Decimal(taxable)
    tax_declared = (value * Decimal(declared) / 100).quantize(PAISA, ROUND_HALF_UP)
    tax_correct = (value * Decimal(correct) / 100).quantize(PAISA, ROUND_HALF_UP)
    return str((tax_correct - tax_declared).quantize(PAISA, ROUND_HALF_UP))


@dataclass(frozen=True, slots=True)
class Scenario:
    """One invoice line, and what a correct run does with it."""

    id: str
    line: dict[str, Any]
    expect_terminal: str
    expect_hsn4: str | None = None
    expect_slab: str | None = None
    expect_reason: str | None = None
    #: Rupees, to the paisa, as a string. None when no opinion is expected.
    expect_differential_inr: str | None = None
    #: True for scenarios written to exercise a branch rather than observed.
    synthetic: bool = False
    tags: tuple[str, ...] = ()
    note: str = ""

    @property
    def invoice_date(self) -> str:
        return str(self.line.get("invoice_date", ""))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "line": self.line,
            "expect_terminal": self.expect_terminal,
            "expect_hsn4": self.expect_hsn4,
            "expect_slab": self.expect_slab,
            "expect_reason": self.expect_reason,
            "expect_differential_inr": self.expect_differential_inr,
            "synthetic": self.synthetic,
            "tags": list(self.tags),
            "note": self.note,
        }


# --------------------------------------------------------------------------
# Derived from the golden set
# --------------------------------------------------------------------------


def golden_rows() -> list[dict[str, Any]]:
    if not GOLDEN_PATH.is_file():
        return []
    return [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def derived_scenarios() -> list[Scenario]:
    """One invoice line per answerable golden row."""
    out: list[Scenario] = []
    for row in golden_rows():
        if not row.get("answerable") or not row.get("hsn4"):
            continue
        rid = row["id"]
        gold_slab = str(row["slab"])
        declared_rate = _pick(rid + ":rate", DECLARED_RATES)
        taxable = _pick(rid + ":value", TAXABLE_VALUES)
        invoice_date = _pick(rid + ":date", POST_AMENDMENT_DATES)

        # The supplier's declared heading. Two thirds of the time it is simply
        # wrong, which is what makes the audit a question rather than a
        # formality; the rest of the time it is right, so a suite that always
        # answered "misdeclared" would score badly.
        declared_hsn = (
            row["hsn4"]
            if hashlib.sha256((rid + ":hsn").encode()).digest()[0] % 3 == 0
            else row["hsn4"][:2] + "99"
        )

        out.append(
            Scenario(
                id=f"sc-{rid}",
                line={
                    "line_id": f"inv-{rid}",
                    "description": row["input"],
                    "declared_hsn": declared_hsn,
                    "declared_rate": declared_rate,
                    "taxable_value_inr": float(taxable),
                    "invoice_date": invoice_date,
                },
                expect_terminal="opinion",
                expect_hsn4=row["hsn4"],
                expect_slab=gold_slab,
                expect_differential_inr=_differential(taxable, declared_rate, gold_slab),
                synthetic=False,
                tags=("derived", row.get("difficulty", "typical")),
                note=(
                    "heading and slab from Project 01's golden set, which is "
                    "gazette-derived and unconfirmed by a human; the four "
                    "invoice fields are constructed from the row id"
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# Synthetic — the branches the golden set cannot reach
# --------------------------------------------------------------------------

_OUT_OF_SCOPE = (
    ("whisky", "Imported single malt whisky, 12 year old, 750ml bottle"),
    ("beer", "Craft beer, lager, 500ml cans, carton of 24"),
)

_UNDER_SPECIFIED = (
    ("model-number-only", "MS-4417/B"),
    ("model-number-only", "Item No. 55123-A"),
    ("no-product-kind", "Assorted"),
)

#: Headings that split on one stated fact, with a description that settles it
#: and one that does not. The second of each pair is the case the tools are
#: built to refuse rather than guess.
_CONDITIONAL = (
    ("8711", "Royal Enfield motorcycle, single cylinder, 349 cc engine", "18"),
    ("8711", "Motorcycle, 500 cc touring model with panniers", "40"),
    ("7418", "Copper handi, 2 litre, for kitchen use", "5"),
    ("2202", "Aerated cola beverage with added sugar, 500ml bottle", "40"),
)

_INDETERMINATE = (
    ("8711", "Motorcycle, red, delivered ex-works"),
    ("7418", "Copper article, 1.2 kg"),
)

#: Headings Notification 19/2025 relocated. The expected slab either side of
#: the boundary is read from the archived Gazette rather than recalled.
_BOUNDARY_HEADINGS = ("2402", "2403", "2401")


def _slab_at(heading: str, when: str) -> str | None:
    try:
        match = gazette.lookup(heading, date.fromisoformat(when))
    except Exception:  # noqa: BLE001 — an unreadable corpus yields no scenario
        return None
    return match.slab


def synthetic_scenarios() -> list[Scenario]:
    out: list[Scenario] = []

    # -- terminal stops before any lookup --------------------------------
    for i, (family, description) in enumerate(_OUT_OF_SCOPE, 1):
        out.append(
            Scenario(
                id=f"sc-oos-{i:02d}",
                line={
                    "line_id": f"inv-oos-{i:02d}",
                    "description": description,
                    "declared_hsn": "2208",
                    "declared_rate": "18",
                    "taxable_value_inr": 64000.00,
                    "invoice_date": "2026-03-14",
                },
                expect_terminal="out_of_scope",
                expect_reason="alcoholic-liquor",
                synthetic=True,
                tags=("synthetic", "out_of_scope", family),
                note="alcoholic liquor is outside GST by constitutional exclusion",
            )
        )

    for i, (reason, description) in enumerate(_UNDER_SPECIFIED, 1):
        out.append(
            Scenario(
                id=f"sc-vague-{i:02d}",
                line={
                    "line_id": f"inv-vague-{i:02d}",
                    "description": description,
                    "declared_hsn": "8479",
                    "declared_rate": "18",
                    "taxable_value_inr": 12500.00,
                    "invoice_date": "2026-03-14",
                },
                expect_terminal="unanswerable",
                expect_reason=reason,
                synthetic=True,
                tags=("synthetic", "under_specified", reason),
                note="the description does not determine a kind of good",
            )
        )

    # -- outside the archive ---------------------------------------------
    for i, when in enumerate(("2025-04-11", "2025-09-21"), 1):
        out.append(
            Scenario(
                id=f"sc-prearchive-{i:02d}",
                line={
                    "line_id": f"inv-prearchive-{i:02d}",
                    "description": "Ball point pens, blue ink, pack of 10",
                    "declared_hsn": "9608",
                    "declared_rate": "12",
                    "taxable_value_inr": 8400.00,
                    "invoice_date": when,
                },
                expect_terminal="unanswerable",
                expect_reason="date-outside-archive",
                synthetic=True,
                tags=("synthetic", "pre_archive"),
                note=(
                    "predates Notification 9/2025; the 1/2017 schedules are not "
                    "archived, and the correct behaviour is to decline"
                ),
            )
        )

    # -- the 2026-02-01 boundary -----------------------------------------
    # The sharpest date branch in the corpus: the same heading, the same goods,
    # a different lawful rate on either side of one day.
    for heading in _BOUNDARY_HEADINGS:
        for label, when in (("before", "2025-11-12"), ("after", "2026-03-05")):
            slab = _slab_at(heading, when)
            if slab is None:
                # Ambiguous on that side, so there is no single expected slab
                # and this is not a scenario with a checkable end state.
                continue
            declared = "28"
            out.append(
                Scenario(
                    id=f"sc-boundary-{heading}-{label}",
                    line={
                        "line_id": f"inv-boundary-{heading}-{label}",
                        "description": (
                            "Cigarettes, filter, 84mm, retail cartons"
                            if heading == "2402"
                            else "Unmanufactured tobacco, threshed, in bales"
                            if heading == "2401"
                            else "Manufactured chewing tobacco, pouches"
                        ),
                        "declared_hsn": heading,
                        "declared_rate": declared,
                        "taxable_value_inr": 500000.00,
                        "invoice_date": when,
                    },
                    expect_terminal="opinion",
                    expect_hsn4=heading,
                    expect_slab=slab,
                    expect_differential_inr=_differential("500000.00", declared, slab),
                    synthetic=True,
                    tags=("synthetic", "boundary", f"heading_{heading}", label),
                    note=(
                        f"heading {heading} on {when}: expected slab read from "
                        "the archived Gazette at that date, not from the golden "
                        "set, which describes the current table only"
                    ),
                )
            )

    # -- conditional headings --------------------------------------------
    for i, (heading, description, slab) in enumerate(_CONDITIONAL, 1):
        declared = "18"
        out.append(
            Scenario(
                id=f"sc-cond-{i:02d}",
                line={
                    "line_id": f"inv-cond-{i:02d}",
                    "description": description,
                    "declared_hsn": heading,
                    "declared_rate": declared,
                    "taxable_value_inr": 320000.00,
                    "invoice_date": "2026-03-14",
                },
                expect_terminal="opinion",
                expect_hsn4=heading,
                expect_slab=slab,
                expect_differential_inr=_differential("320000.00", declared, slab),
                synthetic=True,
                tags=("synthetic", "conditional", f"heading_{heading}"),
                note="lookup_schedule reports ambiguous; check_conditions settles it",
            )
        )

    for i, (heading, description) in enumerate(_INDETERMINATE, 1):
        out.append(
            Scenario(
                id=f"sc-indet-{i:02d}",
                line={
                    "line_id": f"inv-indet-{i:02d}",
                    "description": description,
                    "declared_hsn": heading,
                    "declared_rate": "18",
                    "taxable_value_inr": 74000.00,
                    "invoice_date": "2026-03-14",
                },
                expect_terminal="unanswerable",
                expect_reason="rate-fact-absent",
                synthetic=True,
                tags=("synthetic", "indeterminate", f"heading_{heading}"),
                note=(
                    "the heading splits on a fact the description does not "
                    "state; declining is the correct outcome, not a failure"
                ),
            )
        )

    return out


# --------------------------------------------------------------------------


def all_scenarios() -> list[Scenario]:
    return derived_scenarios() + synthetic_scenarios()


def load(
    *, include_derived: bool = True, include_synthetic: bool = True, tag: str = ""
) -> list[Scenario]:
    """The suite, optionally filtered. Order is stable."""
    out: list[Scenario] = []
    if include_derived:
        out += derived_scenarios()
    if include_synthetic:
        out += synthetic_scenarios()
    if tag:
        out = [s for s in out if tag in s.tags]
    return out


def summary() -> dict[str, Any]:
    scenarios = all_scenarios()
    by_terminal: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    for s in scenarios:
        by_terminal[s.expect_terminal] = by_terminal.get(s.expect_terminal, 0) + 1
        for t in s.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    return {
        "total": len(scenarios),
        "derived": sum(1 for s in scenarios if not s.synthetic),
        "synthetic": sum(1 for s in scenarios if s.synthetic),
        "by_expected_terminal": by_terminal,
        "by_tag": dict(sorted(by_tag.items())),
    }
