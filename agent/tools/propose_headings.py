"""Tool 2 — candidate tariff headings, with their provenance kept separate.

Candidates come from two places, and the difference between them is the most
important thing this tool returns.

**`mentioned`** — headings named *in the description itself*. For an advance
ruling excerpt that is the applicant's own contention, argued in the first
person and frequently the reason the matter went to an authority at all:

    "The applicant is of the opinion that correct classification … is under
    HSN 6810 … attracting GST @ 18%."

That text is advocacy. It is often wrong — an authority disagreeing with it is
what produced the document — and it is inside the input the agent must read.
It is returned as **untrusted evidence**, flagged `advocacy: true`, because
whether the agent defers to it is a measurable behaviour and week 5 measures
it. No attacker put it there; the corpus is adversarial before anything is
injected into it.

**`keyword`** — headings whose Gazette entry text overlaps the description.
Derived from the pinned notification, independent of anything the document
argues for.

An agent that only ever picks `mentioned` candidates has learned to trust the
input, and that is visible from the trace without any further instrumentation.

**Zero, one, or several candidates are three different paths**, which is the
second reason this tool exists. Zero means the description does not name goods
this tool can place. Several means a General Rules of Interpretation tie-break,
which is a judgement, and — like `lookup_schedule` — this tool declines to make
it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from agent.contract import Evidence, ToolResult
from agent.gst import PRIMARY_DIR
from agent.registry import ToolSpec

#: Tariff codes as they appear in prose: "HSN 6810", "heading 68029900",
#: "Chapter Heading 8704", "CTH 3926", "6802.99.00". Only the first four digits
#: are a heading; the rest is sub-heading detail this workflow does not use.
_CODE = re.compile(
    r"(?:\b(?:hsn|hs|cth|cta|chapter\s+heading|sub[-\s]?heading|heading|tariff\s+item)\b"
    r"[\s:.]*)?"
    r"\b(\d{4})(?:[\s.]?\d{2})?(?:[\s.]?\d{2})?\b",
    re.I,
)

#: A bare 4-digit number is a year, a quantity or a case citation far more often
#: than it is a tariff heading. Requiring a cue word in front removes almost all
#: of that, at the cost of missing headings written bare — a trade recorded here
#: because it biases the tool toward *fewer* candidates, which is the safe
#: direction for a tool whose job is to widen the search rather than settle it.
_CUE = re.compile(
    r"\b(?:hsn|hs|cth|cta|chapter\s+heading|sub[-\s]?heading|heading|tariff\s+item)\b"
    r"[\s:.]*$",
    re.I,
)

_STOP = frozenset(
    """a an the and or of for to in on at by with from as is are was were be been being
    that this these those it its which such other others any all not no nor but if then
    than so per under over into out up down more most less least same different applicant
    applicants submitted submission opinion view contention ruling authority advance
    section rule rules read together case cases held decision order dated para paragraph
    goods good product products item items supply supplies classification classifiable
    classified attract attracts attracting rate rates gst tax taxable value amount
    shall may can will would should must also further however therefore hence thus
    whether question questions answer answered raised sought seeks""".split()
)

_WORD = re.compile(r"[a-z]{3,}")


def _fold(word: str) -> str:
    """Crudely singularise, so "motorcycle" matches the tariff's "Motorcycles".

    The schedules are written in the plural throughout — "Motorcycles",
    "Articles", "Pencils" — and invoice lines are written in the singular. That
    mismatch cost every candidate for "Royal Enfield motorcycle 349 cc", which
    is one of the conditional headings the suite is built around.

    Deliberately not a stemmer. Chopping a trailing "s" over-matches in ways
    that are visible and cheap to reason about; a real stemmer would conflate
    "resin"/"resins" correctly and also "glass"/"glas", and debugging its
    surprises is not worth the recall on a seven-word description.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _content_words(text: str) -> set[str]:
    return {_fold(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


# --------------------------------------------------------------------------
# Gazette keyword index
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gazette_index() -> dict[str, str]:
    """`{heading: entry text}` from the rated schedule. Empty when unreadable.

    Delegates the extraction to `agent.gazette`, which anchors entries on their
    serial number and collapses whitespace before slicing. An earlier version
    matched `[^\\n]{15,220}` against the raw PDF text and so kept only the
    *first line* of each entry — which is where the discriminating words live
    for exactly the headings this workflow cares about. Heading 7418's entry
    reads "All goods (other than table, kitchen or other household articles of
    copper; Utensils)", and the word "copper" fell past the line break, so a
    copper utensil scored one point for "kitchen" and lost to heading 7013.

    Empty is a legitimate answer — pypdf absent, or the corpus unverifiable —
    and the caller reports it explicitly rather than silently returning fewer
    candidates.
    """
    try:
        from agent import gazette
    except ImportError:  # pragma: no cover
        return {}
    try:
        rows = gazette._heading_index()
    except Exception:  # noqa: BLE001 — an unreadable corpus is a missing index
        return {}

    index: dict[str, str] = {}
    for heading, _schedule, entry in rows:
        # Keep the longest entry seen for a heading: a heading can appear in
        # more than one schedule, and sub-heading rows repeat the same four
        # digits with progressively less description.
        if len(entry) > len(index.get(heading, "")):
            index[heading] = entry
    return index


#: Above this many content words, a description is long enough that a single
#: overlapping word is noise and two should be required.
_LONG_DESCRIPTION_WORDS = 8


def _min_overlap(n_words: int) -> int:
    """How many overlapping words to require, given the description's length.

    A fixed threshold of two gets this wrong at both ends, and the short end is
    the one that matters. "Quartz slabs, 92% crushed quartz bonded with 8%
    polyester resin, polished" has seven content words, and the entry for
    heading 2506 — "Quartz (other than natural sands); quartzite …" — overlaps
    on exactly one of them. Requiring two returned *zero* candidates for the
    worked example in DESIGN.md §1, which is the main path.

    A catalogue line carries few words and each one is doing work; a 200-word
    advance-ruling excerpt carries many and a single coincidental match between
    one of them and a tariff entry means nothing. So the bar scales.
    """
    return 1 if n_words < _LONG_DESCRIPTION_WORDS else 2


def _keyword_candidates(description: str, limit: int) -> list[dict[str, Any]]:
    index = _gazette_index()
    if not index:
        return []
    words = _content_words(description)
    if not words:
        return []
    threshold = _min_overlap(len(words))
    scored: list[tuple[int, int, str, str, list[str]]] = []
    for heading, entry in index.items():
        entry_words = _content_words(entry)
        hits = sorted(words & entry_words)
        if len(hits) < threshold:
            continue
        # Tie-break on how much of the *entry* the match accounts for. Between
        # two headings matching one word each, the one whose entry is mostly
        # that word is the better candidate: "Quartz (other than natural
        # sands); quartzite" beats a sixty-word residual entry that happens to
        # contain "quartz" once. Sorted ascending, so the shorter entry wins.
        scored.append((len(hits), len(entry_words), heading, entry, hits))
    scored.sort(key=lambda r: (-r[0], r[1], r[2]))
    return [
        {"heading": h, "matched_words": hits, "score": n, "entry": entry}
        for n, _, h, entry, hits in scored[:limit]
    ]


# --------------------------------------------------------------------------


def propose_headings(description: str, max_candidates: int = 5) -> ToolResult:
    """Propose candidate 4-digit tariff headings for a goods description."""
    text = (description or "").strip()
    if not text:
        return ToolResult.err("bad_argument", "description is empty")

    # -- headings named in the document (untrusted) -----------------------
    mentioned: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for m in _CODE.finditer(text):
        heading = m.group(1)
        # Anchored on the digits, not on the whole match. `_CODE` optionally
        # consumes the cue word itself, so `m.start()` sits *before* "HSN" and
        # the prefix window then ends one token too early — `_CUE` never
        # matched, `mentioned` was always empty, and the advocacy channel this
        # tool exists to expose was silently dead. Found by a description
        # reading "...is under HSN 6810, attracting GST @ 18%" returning
        # advocacy=False.
        digits_at = m.start(1)
        prefix = text[max(0, digits_at - 40) : digits_at]
        if not _CUE.search(prefix):
            continue
        if heading in seen:
            continue
        seen.add(heading)
        window = " ".join(text[max(0, digits_at - 160) : m.end() + 160].split())
        mentioned.append({"heading": heading, "position": digits_at})
        evidence.append(
            Evidence(
                source="invoice-line:description",
                locator=f"heading {heading} named at char {digits_at}",
                text=window,
            )
        )
        if len(mentioned) >= max_candidates:
            break

    # -- headings the Gazette's own text suggests -------------------------
    index_available = bool(_gazette_index())
    keyword = _keyword_candidates(text, max_candidates) if index_available else []
    for cand in keyword:
        evidence.append(
            Evidence(
                source="09-2025-CTR.pdf",
                locator=f"rated-schedule entry for heading {cand['heading']}",
                text=cand.pop("entry"),
            )
        )

    all_headings = sorted({c["heading"] for c in mentioned} | {c["heading"] for c in keyword})

    return ToolResult.ok_(
        {
            "candidates": all_headings,
            "count": len(all_headings),
            "mentioned": mentioned,
            "keyword": keyword,
            # The flag that makes the trace legible: candidates that came only
            # from the document are the ones the description argued for.
            "advocacy": bool(mentioned),
            "advocacy_only": sorted(
                {c["heading"] for c in mentioned} - {c["heading"] for c in keyword}
            ),
            # Never left implicit. A degraded search returning three candidates
            # instead of five looks identical to a complete one, and that is
            # how a partial result becomes a wrong answer.
            "gazette_search": "ok" if index_available else "unavailable",
            "sources_searched": ["description"] + (["09-2025-CTR.pdf"] if index_available else []),
            "detail": (
                "Headings under 'mentioned' were named in the description itself. "
                "In an advance-ruling excerpt that is the applicant's contention "
                "— often the contention an authority rejected. Treat it as a "
                "claim to be checked, not as an answer. Where several candidates "
                "remain, choosing between them is a General Rules of "
                "Interpretation judgement and this tool does not make it."
            ),
        },
        evidence,
    )


SPEC = ToolSpec(
    name="propose_headings",
    description=(
        "Propose candidate 4-digit tariff headings for a goods description. "
        "Returns two kinds of candidate, kept separate: 'mentioned' (headings "
        "named in the description itself — for an advance ruling this is the "
        "applicant's own contention and may well be the one the authority "
        "rejected) and 'keyword' (headings whose Gazette entry text overlaps "
        "the description). Zero, one and several candidates are three different "
        "situations. This tool does not choose between candidates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The goods description from the invoice line.",
                "maxLength": 20000,
            },
            "max_candidates": {
                "type": "integer",
                "description": "Cap on candidates returned per source. Default 5.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    },
    handler=propose_headings,
    stage="propose",
    returns_evidence=True,
    pure=True,
)
