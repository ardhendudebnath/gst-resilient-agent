"""The corpus layer: hashes, the date regimes, and the four lookup outcomes.

These are the tests that would catch the archive being swapped, the amendment
being mis-transcribed, or the date-scoping silently reverting to "read today's
table" — which is the failure that would make every number in this repository
wrong in the same direction at once.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent import gazette

pytest.importorskip("pypdf", reason="reading the archived notifications needs pypdf")

BEFORE = date(2025, 11, 1)   # Schedule VII live, 28 % lawful
AFTER = date(2026, 3, 14)    # 19/2025 in force, Schedule VII omitted


def test_archived_sources_match_their_hashes():
    digests = gazette.verify_sources()
    assert set(digests) == {"09-2025-CTR.pdf", "10-2025-CTR.pdf", "19-2025-CTR.pdf"}


def test_regime_boundaries():
    with pytest.raises(gazette.NotArchived):
        gazette.regime(date(2025, 9, 21))

    assert gazette.regime(gazette.NINE_2025_IN_FORCE) == "9/2025"
    assert gazette.regime(date(2026, 1, 31)) == "9/2025"
    assert gazette.regime(gazette.NINETEEN_2025_IN_FORCE) == "9/2025 as amended by 19/2025"


def test_schedule_vii_is_live_only_in_the_middle_window():
    assert gazette.schedule_vii_live(BEFORE)
    assert gazette.schedule_vii_live(date(2026, 1, 31))
    assert not gazette.schedule_vii_live(AFTER)


def test_28_percent_is_stale_only_after_the_amendment():
    # The whole reason this project cannot reuse Project 01's flat
    # ABOLISHED_SLABS: 28 % was a lawful rate for four months.
    assert not gazette.slab_is_stale("28", BEFORE)
    assert gazette.slab_is_stale("28", AFTER)
    # 12 % had no successor schedule in 9/2025 at all.
    assert gazette.slab_is_stale("12", BEFORE)
    assert gazette.slab_is_stale("12", AFTER)


def test_resolved_heading():
    m = gazette.lookup("6810", AFTER)
    assert m.slab == "18"
    assert m.schedule == "II"
    assert not m.ambiguous and not m.chapter_only


@pytest.mark.parametrize(
    "heading,rates",
    [
        ("7418", {"5", "18"}),    # household articles of copper vs other
        ("8711", {"18", "40"}),   # 350 cc
        ("2202", {"5", "40"}),    # added sugar
    ],
)
def test_conditional_headings_are_reported_ambiguous_never_resolved(heading, rates):
    m = gazette.lookup(heading, AFTER)
    assert m.ambiguous, f"{heading} should not resolve to one rate"
    assert m.slab is None, "an ambiguous heading must not offer a slab"
    assert rates <= {e.slab for e in m.entries}


def test_2403_moves_from_resolved_to_ambiguous_across_the_amendment():
    """The sharpest date branch in the corpus.

    Before the amendment the whole heading sat in Schedule VII at 28 %. After
    it, 19/2025 split biris (18 %) from the rest of manufactured tobacco
    (40 %), so the same heading and the same goods become a question the
    lookup refuses to answer.
    """
    before = gazette.lookup("2403", BEFORE)
    assert before.slab == "28"
    assert before.schedule == "VII"

    after = gazette.lookup("2403", AFTER)
    assert after.ambiguous
    assert {e.slab for e in after.entries} == {"18", "40"}


def test_2402_rate_changes_across_the_amendment():
    assert gazette.lookup("2402", BEFORE).slab == "28"
    assert gazette.lookup("2402", AFTER).slab == "40"


def test_schedule_vii_entries_are_withheld_after_the_amendment():
    """A rate that no longer exists must not be offered, even at slab None."""
    after = gazette.lookup("2402", AFTER)
    assert all(e.schedule != "VII" for e in after.entries)


def test_lookup_before_the_archive_raises_rather_than_guessing():
    with pytest.raises(gazette.NotArchived):
        gazette.lookup("6810", date(2025, 1, 15))


def test_search_returns_candidates_but_does_not_rank_the_answer_first():
    """Documents the retriever's weakness rather than papering over it.

    The quartz-slab line's correct heading is 6810; bag-of-words overlap puts
    the mineral headings above it because the description says "quartz" twice.
    An agent that takes rank 1 is wrong, which is the branch the workflow
    exists to exercise. If this test ever starts failing because 6810 ranks
    first, the suite got easier and the README should say so.
    """
    hits = gazette.search(
        "Quartz slabs, 92% crushed quartz bonded with 8% polyester resin", limit=8
    )
    assert hits, "the search should return something"
    assert hits[0][0] != "6810"
