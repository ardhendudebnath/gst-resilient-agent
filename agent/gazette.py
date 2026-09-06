"""Read the archived Gazette notifications, as they stood on a given date.

The parsing here â€” schedule bounds, entry extraction, chapter-level fallback,
collapsing sub-headings by schedule â€” is adapted from `schedule_lookup` in the
GST eval harness (project 01), which is the same author's and carries the same
refusal to resolve an ambiguous heading. What is new is the date.

**Why the date matters.** Project 01 asks "what is the rate now", so it applies
Notification 19/2025 unconditionally and treats Schedule VII as gone. An
invoice audit cannot do that. An invoice raised on 12 November 2025 was raised
into a rate table where Schedule VII was live and 28 % was a lawful rate, and
scoring it against today's table would report a compliant supplier as having
under-collected. There are three regimes:

    before 2025-09-22   Notification 1/2017 and its amendments. Not archived
                        here, so the honest answer is `not_archived` rather
                        than a rate read from the wrong document.
    2025-09-22 .. 2026-01-31
                        9/2025 and 10/2025 as issued. Schedules I-VII.
                        28 % is live. Tobacco sits in VII.
    from 2026-02-01     9/2025 as amended by 19/2025. Schedule VII omitted,
                        tobacco to III (40 %), biris to II (18 %).

So the same heading, the same goods and the same corpus give three different
answers depending on one field of the input, and a model answering from recall
gets the middle window wrong in both directions. That is the branch this
workflow is built around.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from agent.config import gazette_dir

RATED_FILE = "09-2025-CTR.pdf"
EXEMPT_FILE = "10-2025-CTR.pdf"
AMENDING_FILE = "19-2025-CTR.pdf"


def rated_path() -> Path:
    return gazette_dir() / RATED_FILE


def exempt_path() -> Path:
    return gazette_dir() / EXEMPT_FILE

#: The two dates that partition the archive. Both are `in_force_from` values
#: read out of MANIFEST.json, not recalled.
NINE_2025_IN_FORCE = date(2025, 9, 22)
NINETEEN_2025_IN_FORCE = date(2026, 2, 1)

#: Schedule -> combined GST rate. The notification states CGST; combined is
#: double. Schedule VII is the one that moves: 28 % until 19/2025 omitted it.
SCHEDULE_SLAB: dict[str, str] = {
    "I": "5",
    "II": "18",
    "III": "40",
    "IV": "3",
    "V": "0.25",
    "VI": "1.5",
    "VII": "28",
}

#: Slabs that do not exist under the current table, with the date each ceased.
#: 12 % had no successor schedule in 9/2025 at all, so it is stale for every
#: date this corpus covers; 28 % is stale only from 2026-02-01.
SLAB_ABOLISHED_ON: dict[str, date] = {
    "12": date(2025, 9, 22),
    "28": date(2026, 2, 1),
}

#: Notification 19/2025, in force 2026-02-01, transcribed from the archived
#: copy. It did not merely delete Schedule VII â€” it moved every entry, and
#: split one, which is why biris end up at 18 % and the rest of tobacco at
#: 40 %. Transcribed rather than regex-parsed: the amending text is three
#: sentences of legal drafting and a pattern over it would be far more fragile.
#: `tests/test_gazette_amendment.py` checks each row against the archived text.
AMENDED_2026: dict[str, list[tuple[str, str, str]]] = {
    "2106": [("2106 90 20", "III", "Pan masala")],
    "2401": [("", "III", "Unmanufactured tobacco; tobacco refuse "
                         "[other than tobacco leaves]")],
    "2402": [("", "III", "Cigars, cheroots, cigarillos and cigarettes, of "
                         "tobacco or of tobacco substitutes")],
    "2403": [
        ("2403 19 21, 2403 19 29", "II", "Biris"),
        ("2403 (other than 2403 19 21, 2403 19 29)", "III",
         "Other manufactured tobacco and manufactured tobacco substitutes; "
         "homogenised or reconstituted tobacco; tobacco extracts and essences "
         "[other than biris]"),
    ],
    "2404": [
        ("2404 11 00", "III", "Products containing tobacco or reconstituted "
                              "tobacco and intended for inhalation without "
                              "combustion"),
        ("2404 19 00", "III", "Products containing tobacco or nicotine "
                              "substitutes and intended for inhalation "
                              "without combustion"),
    ],
}


class NotArchived(Exception):
    """The date falls outside the archived notifications."""


class SourceMissing(Exception):
    """A pinned notification is not where it should be."""


class SourceMismatch(Exception):
    """A pinned notification is not the document it was built against."""


@lru_cache(maxsize=1)
def verify_sources() -> dict[str, str]:
    """Check every archived notification against MANIFEST.json. See DESIGN.md §8.

    A tool whose answers depend on a document must fail loudly when the
    document is not the one it was built against. Silently reading a different
    copy would produce a citable-looking rate traceable to nothing, which is
    worse than no answer at all — so this raises, and the raise surfaces to the
    agent as `source_mismatch`, an error it has to handle rather than a crash.

    Cached: hashing 2.3 MB on every one of 20 tool calls per run, across a
    600-run chaos suite, is four minutes of SHA-256 for a file that cannot
    change mid-suite.
    """
    manifest_path = gazette_dir() / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SourceMissing(
            f"no MANIFEST.json in {gazette_dir()}. The Gazette corpus is "
            "vendored in data/reference/primary/; set GAZETTE_PRIMARY_DIR if "
            "it lives elsewhere."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digests: dict[str, str] = {}
    for entry in manifest["documents"]:
        path = gazette_dir() / entry["file"]
        if not path.is_file():
            raise SourceMissing(f"pinned notification missing: {path}")

        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()

        if digest != entry["sha256"]:
            raise SourceMismatch(
                f"{entry['file']} is not the archived copy: expected "
                f"{entry['sha256'][:16]}…, found {digest[:16]}…. Every rate "
                "this repository cites traces to that document."
            )
        digests[entry["file"]] = digest
    return digests


@dataclass(slots=True)
class Entry:
    schedule: str
    slab: str | None
    text: str

    def to_json(self) -> dict:
        return {"schedule": self.schedule, "slab": self.slab, "text": self.text}


@dataclass(slots=True)
class Match:
    heading: str
    as_of: date
    entries: list[Entry] = field(default_factory=list)
    exempt_entries: list[str] = field(default_factory=list)
    #: Chapter-level entries, found only when the heading itself is unlisted.
    #: Never resolve a slab: a chapter entry carries exclusions, and whether
    #: these goods fall inside them is a reading, not a lookup.
    chapter_entries: list[Entry] = field(default_factory=list)

    @property
    def _outcomes(self) -> set[str | None]:
        """The distinct *rates* this heading could attract.

        Counting outcomes rather than entries matters: 2404 has two
        sub-headings and both land at 40 %, so which applies changes nothing
        about the rate and reporting it as ambiguous would send the agent off
        to resolve a distinction with no consequence.
        """
        found: set[str | None] = {e.slab for e in self.entries}
        if self.exempt_entries:
            found.add("0")
        if not self.entries and not self.exempt_entries:
            found |= {e.slab for e in self.chapter_entries}
        return found

    @property
    def ambiguous(self) -> bool:
        return len(self._outcomes) > 1

    @property
    def chapter_only(self) -> bool:
        return not self.entries and not self.exempt_entries and bool(self.chapter_entries)

    @property
    def found(self) -> bool:
        return bool(self.entries or self.exempt_entries or self.chapter_entries)

    @property
    def slab(self) -> str | None:
        """The rate, only when every entry agrees and none is chapter-level."""
        if self.ambiguous or self.chapter_only:
            return None
        outcomes = self._outcomes
        return next(iter(outcomes)) if len(outcomes) == 1 else None

    @property
    def schedule(self) -> str | None:
        if self.ambiguous or not self.entries:
            return None
        schedules = {e.schedule for e in self.entries}
        return next(iter(schedules)) if len(schedules) == 1 else None


def regime(as_of: date) -> str:
    """Which notification governs on `as_of`. Raises for the pre-archive era."""
    if as_of < NINE_2025_IN_FORCE:
        raise NotArchived(
            f"{as_of.isoformat()} precedes Notification 9/2025 "
            f"({NINE_2025_IN_FORCE.isoformat()}); the 1/2017 schedules that "
            "governed then are not archived in this repository"
        )
    return "9/2025" if as_of < NINETEEN_2025_IN_FORCE else "9/2025 as amended by 19/2025"


def schedule_vii_live(as_of: date) -> bool:
    """True while 28 % is a lawful rate."""
    return NINE_2025_IN_FORCE <= as_of < NINETEEN_2025_IN_FORCE


def slab_is_stale(slab: str, as_of: date) -> bool:
    """True when `slab` names a rate that did not exist on `as_of`."""
    ceased = SLAB_ABOLISHED_ON.get(slab.strip())
    return ceased is not None and as_of >= ceased


_SCHED_HEAD = re.compile(
    r"Schedule\s+(VII|VI|V|IV|III|II|I)\s*[-–—]{0,2}\s*([\d.]+)\s*%", re.I
)

#: A chapter number carrying goods is always introduced by its serial number in
#: these schedules â€” "390. 63 [other than ...] Other made up textile articles".
#: Requiring the serial is what separates a real chapter entry from a two-digit
#: fragment of a longer code; without it the "29" in "0101 29 Live horses"
#: reads as chapter 29 and files live horses under organic chemicals.
_SERIAL = r"(?:^|[^\d])\d{1,3}\s*\.\s*"


@lru_cache(maxsize=4)
def _text(path: str) -> str:
    import logging

    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SourceMissing(
            "reading the archived notifications needs pypdf: "
            "pip install 'gst-resilient-agent[gazette]'"
        ) from exc

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = pypdf.PdfReader(path)
    raw = "\n".join((p.extract_text() or "") for p in reader.pages)
    return re.sub(r"[ \t]+", " ", raw)


@lru_cache(maxsize=4)
def _bounds(path: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (m.start(), m.group(1).upper()) for m in _SCHED_HEAD.finditer(_text(path))
    )


def _schedule_at(bounds: tuple[tuple[int, str], ...], pos: int) -> str | None:
    current = None
    for start, roman in bounds:
        if start <= pos:
            current = roman
        else:
            break
    return current


def _entry_text(text: str, pos: int, width: int = 240) -> str:
    """The entry around a match, trimmed to something quotable."""
    start = text.rfind(". ", max(0, pos - 160), pos)
    start = start + 2 if start != -1 else max(0, pos - 60)
    return re.sub(r"\s+", " ", text[start : pos + width]).strip()


def _chapter_entries(
    text: str, bounds: tuple[tuple[int, str], ...], chapter: str
) -> dict[str, str]:
    pattern = re.compile(
        rf"{_SERIAL}({re.escape(chapter)})\s*(?:\[[^\]]*\]\s*)?(?=[A-Z])"
    )
    by_schedule: dict[str, str] = {}
    for m in pattern.finditer(text):
        sched = _schedule_at(bounds, m.start(1))
        if sched is None or sched in by_schedule:
            continue
        by_schedule[sched] = _entry_text(text, m.start(1))
    return by_schedule


def lookup(heading: str, as_of: date) -> Match:
    """Find `heading` in the schedules as they stood on `as_of`.

    Raises `NotArchived` for a date before 9/2025, and `FileNotFoundError` if
    the pinned corpus is missing â€” both loudly, because silently returning
    "not found" for a document we never opened is the failure that reads as
    "the notification is silent" when nothing of the sort has been checked.
    """
    verify_sources()  # raises SourceMissing / SourceMismatch
    regime(as_of)  # raises NotArchived for the pre-archive era

    heading = heading.strip()[:4]
    match = Match(heading=heading, as_of=as_of)

    rated = _text(str(rated_path()))
    bounds = _bounds(str(rated_path()))

    # Collapse by schedule. A heading's sub-headings each match separately
    # ("3306", "3306 10 10"), and a neighbouring entry can name the heading too
    # ("8714 Parts and accessories of vehicles of heading 8711"). Counting
    # those as distinct entries reports a heading as ambiguous when every match
    # sits in one schedule at one rate.
    by_schedule: dict[str, str] = {}
    for m in re.finditer(rf"\b{re.escape(heading)}\b", rated):
        sched = _schedule_at(bounds, m.start())
        if sched is None:
            continue
        text = _entry_text(rated, m.start())
        # Prefer the entry that opens with the heading itself.
        if sched not in by_schedule or (
            text.startswith(heading) and not by_schedule[sched].startswith(heading)
        ):
            by_schedule[sched] = text

    amended = as_of >= NINETEEN_2025_IN_FORCE
    for sched, text in by_schedule.items():
        if sched == "VII" and amended:
            # Omitted outright on 2026-02-01. Reporting its entries as live
            # would offer a rate that no longer exists; the replacements come
            # from AMENDED_2026 below.
            continue
        match.entries.append(Entry(sched, SCHEDULE_SLAB.get(sched), text))

    if amended:
        for sub, sched, description in AMENDED_2026.get(heading, []):
            label = f"{sub or heading} {description}"
            match.entries.append(
                Entry(
                    sched,
                    SCHEDULE_SLAB.get(sched),
                    f"[Notification 19/2025, in force 2026-02-01] {label}",
                )
            )

    # Only when the heading itself is absent. A heading with its own entry is
    # governed by it, and dragging in the chapter would invent competition.
    if not match.entries:
        for sched, text in _chapter_entries(rated, bounds, heading[:2]).items():
            match.chapter_entries.append(Entry(sched, SCHEDULE_SLAB.get(sched), text))

    if exempt_path().exists():
        exempt = _text(str(exempt_path()))
        seen: set[str] = set()
        for m in re.finditer(rf"\b{re.escape(heading)}\b", exempt):
            text = _entry_text(exempt, m.start())
            if text[:80] in seen:
                continue
            seen.add(text[:80])
            match.exempt_entries.append(text)

    return match


#: Words too common in the schedules to discriminate between headings. Searching
#: for "other" returns most of the tariff, which is worse than returning nothing
#: because it looks like a result.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "other", "than", "and", "or", "the", "of", "for", "with", "without",
        "not", "any", "all", "such", "whether", "including", "goods", "articles",
        "products", "kind", "used", "similar", "form", "forms", "put", "up",
        "per", "cent", "nil", "schedule",
    }
)

_HEADING_AT_ENTRY = re.compile(rf"{_SERIAL}(\d{{4}})\b")


@lru_cache(maxsize=1)
def _heading_index() -> tuple[tuple[str, str, str], ...]:
    """Every (heading, schedule, entry text) the rated notification lists.

    Built once and cached: the PDF is 52 pages and re-scanning it per search
    turned a 40-scenario suite into a two-minute wait.
    """
    verify_sources()
    rated = _text(str(rated_path()))
    bounds = _bounds(str(rated_path()))
    index: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _HEADING_AT_ENTRY.finditer(rated):
        heading = m.group(1)
        sched = _schedule_at(bounds, m.start(1))
        if sched is None or (heading, sched) in seen:
            continue
        seen.add((heading, sched))
        index.append((heading, sched, _entry_text(rated, m.start(1))))
    return tuple(index)


def search(keywords: str, limit: int = 8) -> list[tuple[str, str, str, int]]:
    """Rank tariff headings by keyword overlap with their entry text.

    Deliberately a dumb bag-of-words score, not an embedding. The point of this
    tool is to hand the agent candidates it must then *verify* through
    `lookup_schedule`; a retriever good enough to be trusted blindly would hide
    the branch this workflow exists to exercise.
    """
    terms = [
        t for t in re.findall(r"[a-z0-9]+", keywords.lower())
        if len(t) > 2 and t not in _STOPWORDS
    ]
    if not terms:
        return []

    scored: list[tuple[str, str, str, int]] = []
    for heading, sched, text in _heading_index():
        low = text.lower()
        score = sum(3 if f" {t} " in f" {low} " else 1 for t in terms if t in low)
        if score:
            scored.append((heading, sched, text, score))

    scored.sort(key=lambda row: (-row[3], row[0]))
    return scored[:limit]
