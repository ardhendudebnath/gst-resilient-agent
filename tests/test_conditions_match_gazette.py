"""Every condition rule must still describe an entry that is in the document.

`check_conditions` encodes, in Python, splits that exist in the notification —
350 cc, added sugar, household articles of copper. If the schedules are ever
re-vendored and an entry has moved or been reworded, those rules become fiction
that still returns confident answers. This is the test that turns that from a
silent wrong answer into a red build.

Comparison is on a normalised form: lowercase, punctuation and hyphens
flattened, whitespace collapsed. PDF text extraction inserts breaks at
arbitrary points ("porous -tipped", "cement , of concrete"), so an exact
substring check would fail on text that is plainly present. Normalising is the
honest way to compare; matching a paraphrase would not be.
"""

from __future__ import annotations

import re

import pytest

from agent import gazette
from agent.tools.check_conditions import CONDITIONS

pytest.importorskip("pypdf", reason="reading the archived notifications needs pypdf")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@pytest.fixture(scope="module")
def corpus() -> str:
    rated = gazette._text(str(gazette.rated_path()))
    exempt = gazette._text(str(gazette.exempt_path()))
    amending = gazette._text(str(gazette.gazette_dir() / gazette.AMENDING_FILE))
    return norm(rated + "\n" + exempt + "\n" + amending)


@pytest.mark.parametrize(
    "heading,condition",
    [(h, c) for h, conds in CONDITIONS.items() for c in conds],
    ids=[f"{h}:{c.name}" for h, conds in CONDITIONS.items() for c in conds],
)
def test_condition_describes_text_present_in_archive(heading, condition, corpus):
    assert norm(condition.describes) in corpus, (
        f"condition {heading}:{condition.name} claims the notification says "
        f"{condition.describes!r}, and it does not. Either the archive changed "
        "or the rule was written from memory."
    )


@pytest.mark.parametrize("heading", sorted(CONDITIONS))
def test_condition_rates_match_what_the_lookup_reports(heading):
    """A rule may not offer a rate the schedules do not put on that heading."""
    from datetime import date

    match = gazette.lookup(heading, date(2026, 3, 14))
    available = {e.slab for e in match.entries}
    if match.exempt_entries:
        available.add("0")

    for condition in CONDITIONS[heading]:
        assert condition.slab in available, (
            f"{heading}:{condition.name} resolves to {condition.slab}%, which "
            f"the lookup does not list for this heading (it has {sorted(available)})"
        )


def test_every_conditional_heading_is_actually_ambiguous():
    """No rule should exist for a heading that resolves on its own.

    A condition table entry for an unambiguous heading is dead code at best,
    and at worst it overrides a clean lookup with a pattern match.
    """
    from datetime import date

    for heading in CONDITIONS:
        match = gazette.lookup(heading, date(2026, 3, 14))
        assert match.ambiguous, (
            f"{heading} has condition rules but resolves cleanly to "
            f"{match.slab}% — the rules are unreachable"
        )
