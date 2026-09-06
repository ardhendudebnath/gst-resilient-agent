"""The date branch, end to end through the tools, plus four fixed regressions.

Every test here failed at some point during the build. They are kept because
each one is a way the tool set could go back to answering a date-blind question
confidently, which is the single failure mode this domain punishes hardest: the
answer still looks like a citable rate.

The boundary under test is 1 February 2026, when Notification 19/2025 omitted
Schedule VII. Cigarettes (heading 2402) are the cleanest probe — 28 % before,
40 % after, and both are correct answers on their own side of the line.
"""

from __future__ import annotations

import pytest

from agent.registry import build_call
from agent.tools import build_registry

pytest.importorskip("pypdf", reason="reading the archived notifications needs pypdf")

BEFORE = "2025-11-01"  # Schedule VII live, 28 % lawful
AFTER = "2026-03-01"   # 19/2025 in force, 28 % abolished


@pytest.fixture(scope="module")
def registry():
    return build_registry()


def call(registry, name, **args):
    return registry.invoke(build_call(name, args))


# --------------------------------------------------------------------------
# The gap: a pre-amendment rate must be reachable at all
# --------------------------------------------------------------------------


def test_lookup_returns_the_rate_in_force_on_the_invoice_date(registry):
    """28 % on a November 2025 invoice is the correct answer, not a stale one.

    Regression: the lookup delegated to a date-blind upstream that dropped
    every Schedule VII entry, so this returned 40 % — marking a compliant
    supplier as having short-paid by twelve points.
    """
    before = call(registry, "lookup_schedule", heading="2402", on_date=BEFORE)
    assert before.ok
    assert before.data["outcome"] == "resolved"
    assert before.data["slab"] == "28"
    assert before.data["schedule"] == "VII"
    assert before.data["schedule_vii_live"] is True

    after = call(registry, "lookup_schedule", heading="2402", on_date=AFTER)
    assert after.ok
    assert after.data["slab"] == "40"
    assert after.data["schedule"] == "III"
    assert after.data["schedule_vii_live"] is False


def test_rate_history_reports_the_rate_on_the_date_not_just_that_it_moved(registry):
    """Regression: the agent was told "the pre-amendment entry governs here"
    and never told what it was, so 28 % was unreachable through the tool set."""
    r = call(registry, "rate_history", heading="2402", on_date=BEFORE)
    assert r.ok
    assert r.data["slab_on_date"] == "28"
    assert r.data["moved_on_2026_02_01"] is True
    assert r.data["slab_before_amendment"] == "28"
    assert r.data["slab_after_amendment"] == "40"
    assert r.data["amendment_applies"] is False
    assert "28" in r.data["note"]


def test_rate_history_never_claims_a_moved_heading_did_not_move(registry):
    """Regression: with the amendment table unavailable, an empty table read as
    "no relocation" and the tool asserted `moved: false` for a heading that
    moved. A confident wrong fact is worse than a refusal."""
    for heading in ("2401", "2402", "2403", "2404", "2106"):
        r = call(registry, "rate_history", heading=heading, on_date=BEFORE)
        assert r.ok, f"{heading}: {r.message}"
        assert r.data["moved_on_2026_02_01"] is True, (
            f"heading {heading} was relocated by 19/2025 and the tool says it was not"
        )


def test_dates_before_the_archive_are_refused_not_extrapolated(registry):
    r = call(registry, "rate_history", heading="6810", on_date="2025-01-15")
    assert not r.ok
    assert r.error == "not_found"
    assert r.data["in_archive"] is False


# --------------------------------------------------------------------------
# Arithmetic: validity is a question about a date
# --------------------------------------------------------------------------


def test_compute_liability_accepts_28_before_and_rejects_it_after(registry):
    ok = call(
        registry, "compute_liability",
        taxable_value_inr=100000, correct_slab="28", declared_rate="28", on_date=BEFORE,
    )
    assert ok.ok, ok.message
    assert ok.data["differential_inr"] == 0.0
    assert ok.data["declared_correct"] is True
    # Billing 28 % in November 2025 was right, so it must not be flagged.
    assert ok.data["declared_rate_abolished"] is False

    bad = call(
        registry, "compute_liability",
        taxable_value_inr=100000, correct_slab="28", declared_rate="28", on_date=AFTER,
    )
    assert not bad.ok
    assert bad.error == "bad_argument"
    assert bad.data["abolished"] is True


def test_declared_12_percent_is_flagged_abolished_on_every_archived_date(registry):
    """12 % had no successor schedule in 9/2025, so it is stale throughout."""
    for on_date in (BEFORE, AFTER):
        r = call(
            registry, "compute_liability",
            taxable_value_inr=250000, correct_slab="18", declared_rate="12", on_date=on_date,
        )
        assert r.ok
        assert r.data["declared_rate_abolished"] is True
        assert r.data["differential_inr"] == 15000.0
        assert r.data["direction"] == "short_paid"


# --------------------------------------------------------------------------
# Conditions only exist when the split exists
# --------------------------------------------------------------------------


def test_2403_splits_only_after_the_amendment(registry):
    """Before 2026-02-01 the whole heading sat in Schedule VII at 28 %.

    Offering the biris/other split for a 2025 invoice would answer 18 % where
    the lawful rate was 28 % — a wrong answer produced by a tool being helpful.
    """
    early = call(
        registry, "check_conditions",
        heading="2403", description="Biris, hand-rolled", on_date=BEFORE,
    )
    assert early.ok
    assert early.data["outcome"] == "not_ambiguous"
    assert early.data["resolved"] is False
    assert early.data["slab_from_lookup"] == "28"

    late = call(
        registry, "check_conditions",
        heading="2403", description="Biris, hand-rolled", on_date=AFTER,
    )
    assert late.ok
    assert late.data["outcome"] == "resolved"
    assert late.data["slab"] == "18"


def test_an_absent_deciding_fact_is_a_refusal_not_a_guess(registry):
    r = call(
        registry, "check_conditions",
        heading="8711", description="Motorcycle, red", on_date=AFTER,
    )
    assert r.ok
    assert r.data["outcome"] == "not_determinable"
    assert r.data["reason"] == "rate-fact-absent"
    assert "slab" not in r.data


# --------------------------------------------------------------------------
# Candidate proposal
# --------------------------------------------------------------------------


def test_worked_example_yields_candidates(registry):
    """Regression: requiring two overlapping words returned zero candidates for
    the seven-word description in DESIGN.md §1, which is the main path."""
    r = call(
        registry, "propose_headings",
        description="Quartz slabs, 92% crushed quartz bonded with 8% polyester resin, polished",
    )
    assert r.ok
    assert r.data["count"] > 0
    assert "2506" in r.data["candidates"]


def test_singular_invoice_language_matches_plural_tariff_language(registry):
    """Regression: the schedules say "Motorcycles", invoices say "motorcycle",
    and without folding this returned nothing for a conditional heading."""
    r = call(registry, "propose_headings", description="Royal Enfield motorcycle 349 cc")
    assert r.ok
    assert "8711" in r.data["candidates"]


def test_discriminating_words_past_a_line_break_still_count(registry):
    """Regression: the index kept only the first line of each entry, so "copper"
    in 7418's entry was truncated away and a copper utensil lost to glassware."""
    r = call(registry, "propose_headings", description="Copper handi, 2 litre, kitchen use")
    assert r.ok
    assert "7418" in r.data["candidates"]
    top = r.data["keyword"][0]
    assert top["heading"] == "7418"
    assert set(top["matched_words"]) >= {"copper", "kitchen"}


def test_the_declared_heading_is_always_a_candidate(registry):
    """The declaration is the claim under audit and must be testable.

    Regression from the first live run: the tool took no declared_hsn, the
    model reported "the proposed headings do not include 6802", and having
    nowhere to take that it adopted the retriever's top keyword hit and
    answered 5% on a line whose declared heading it never checked.
    """
    r = call(
        registry, "propose_headings",
        description="Quartz slabs, 92% crushed quartz bonded with 8% polyester resin",
        declared_hsn="6802",
    )
    assert r.ok
    assert "6802" in r.data["candidates"]
    # The search does not independently support it, and saying so is the point:
    # that makes it more worth checking, not less.
    assert r.data["declared_hsn_found_independently"] is False
    assert r.data["declared_hsn"] == "6802"


def test_a_declared_heading_the_search_also_found_is_not_duplicated(registry):
    r = call(
        registry, "propose_headings",
        description="Royal Enfield motorcycle 349 cc",
        declared_hsn="8711",
    )
    assert r.ok
    assert r.data["candidates"].count("8711") == 1
    assert r.data["declared_hsn_found_independently"] is True


def test_advocacy_channel_fires_on_a_heading_the_document_argues_for(registry):
    """Regression: `_CODE` consumed the cue word, so the prefix window ended one
    token early, `_CUE` never matched and `mentioned` was always empty — which
    silently disabled the measurement this tool exists to make."""
    r = call(
        registry, "propose_headings",
        description=(
            "Quartz slabs. The applicant is of the opinion that the correct "
            "classification is under HSN 6810, attracting GST @ 18%."
        ),
    )
    assert r.ok
    assert r.data["advocacy"] is True
    assert "6810" in {c["heading"] for c in r.data["mentioned"]}
    assert any(e.source == "invoice-line:description" for e in r.evidence)
