"""Candidate retrieval over the tariff schedules, keyword and semantic.

Two backends behind one interface, and the choice is configuration rather than
a rewrite. That is deliberate, and it is a measurement decision before it is a
design one.

### Why both, rather than just the better one

Keyword overlap has a measured ceiling in this corpus. The worked example in
`docs/DESIGN.md` §1 is a quartz slab whose correct heading is 6810, "Articles of
cement, of concrete or of artificial stone" — which shares no word with "quartz
slabs, 92% crushed quartz bonded with 8% polyester resin". Bag-of-words cannot
reach it from there, and five of the twenty-eight golden rows are in that
position, capping the suite at roughly 82 % before the agent reasons at all.

That is exactly the gap a semantic index closes. But replacing the retriever
outright would move the baseline the chaos results are measured against, so the
before/after table would silently become "keyword versus embeddings" instead of
"no defences versus defences". Both backends stay, the mode is recorded on every
run, and retrieval gets its own row in the results rather than contaminating
someone else's.

### Degradation

Semantic retrieval needs a key and a built index. Without either it reports
unavailable and the caller falls back to keyword — it does not raise, and it
does not silently return fewer candidates while looking healthy. `make test`
still passes on a fresh clone with no key.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from agent import gazette
from agent.embed import (
    EmbeddingClient,
    EmbeddingError,
    VectorIndex,
    build_index,
    embed_query,
)

#: How candidates are proposed. `hybrid` runs both and merges by rank.
MODES = ("keyword", "semantic", "hybrid")
DEFAULT_MODE = "keyword"

#: Name of the cached index over the rated schedule.
INDEX_NAME = "gazette-headings"


def retrieval_mode() -> str:
    """Configured backend. Keyword by default, so the baseline is unchanged
    unless a run explicitly asks for something else."""
    mode = os.environ.get("RETRIEVAL_MODE", "").strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"RETRIEVAL_MODE={mode!r} not in {MODES}")
    return mode


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def gazette_entries() -> dict[str, str]:
    """`{heading: entry text}` from the rated schedule.

    One entry per heading, longest kept: a heading can appear in more than one
    schedule, and sub-heading rows repeat the same four digits with
    progressively less description.

    **This is the chunking, and it is structural rather than fixed-size.** The
    Gazette is not prose. Each row is a serial number, a heading, a description
    and a rate, and `docs/DESIGN.md` §5 requires every citation to resolve to a
    specific schedule entry. Splitting on a token count would cut entries in
    half and make that impossible; one entry per vector keeps a retrieved chunk
    and a citable unit the same object.
    """
    try:
        rows = gazette._heading_index()
    except Exception:  # noqa: BLE001 — an unreadable corpus is an empty index
        return {}
    index: dict[str, str] = {}
    for heading, _schedule, entry in rows:
        if len(entry) > len(index.get(heading, "")):
            index[heading] = entry
    return index


def corpus() -> tuple[list[str], list[str]]:
    """(headings, entry texts), in a stable order."""
    entries = gazette_entries()
    ids = sorted(entries)
    return ids, [entries[h] for h in ids]


# --------------------------------------------------------------------------
# Keyword
# --------------------------------------------------------------------------

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

_LONG_DESCRIPTION_WORDS = 8


def fold(word: str) -> str:
    """Crudely singularise, so "motorcycle" matches the tariff's "Motorcycles".

    The schedules are written in the plural throughout and invoice lines in the
    singular. Deliberately not a stemmer: chopping a trailing "s" over-matches
    in ways that are visible and cheap to reason about.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def content_words(text: str) -> set[str]:
    return {fold(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def min_overlap(n_words: int) -> int:
    """How many overlapping words to require, given the description's length.

    A fixed threshold of two gets this wrong at the short end, which is the end
    that matters: a seven-word catalogue line carries few words and each is
    doing work, while a 200-word ruling excerpt carries many and one
    coincidental match means nothing.
    """
    return 1 if n_words < _LONG_DESCRIPTION_WORDS else 2


def keyword_search(description: str, limit: int = 5) -> list[dict[str, Any]]:
    entries = gazette_entries()
    if not entries:
        return []
    words = content_words(description)
    if not words:
        return []
    threshold = min_overlap(len(words))

    scored: list[tuple[int, int, str, str, list[str]]] = []
    for heading, entry in entries.items():
        entry_words = content_words(entry)
        hits = sorted(words & entry_words)
        if len(hits) < threshold:
            continue
        # Between two headings matching one word each, prefer the one whose
        # entry is mostly that word: "Quartz; quartzite" beats a sixty-word
        # residual entry that happens to contain "quartz" once.
        scored.append((len(hits), len(entry_words), heading, entry, hits))

    scored.sort(key=lambda r: (-r[0], r[1], r[2]))
    return [
        {
            "heading": h,
            "score": n,
            "matched_words": hits,
            "entry": entry,
            "via": "keyword",
        }
        for n, _spec, h, entry, hits in scored[:limit]
    ]


# --------------------------------------------------------------------------
# Semantic
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _index() -> VectorIndex | None:
    ids, texts = corpus()
    if not ids:
        return None
    client = EmbeddingClient()
    if not client.available():
        return None
    try:
        return build_index(ids, texts, name=INDEX_NAME, client=client)
    except EmbeddingError:
        return None


def semantic_available() -> bool:
    """True when a query could actually be answered semantically."""
    return _index() is not None


def semantic_search(description: str, limit: int = 5) -> list[dict[str, Any]]:
    """Nearest schedule entries by cosine similarity. Empty when unavailable."""
    index = _index()
    if index is None:
        return []
    try:
        vector = embed_query(description)
    except EmbeddingError:
        return []

    entries = gazette_entries()
    return [
        {
            "heading": heading,
            "score": round(score, 4),
            "matched_words": [],
            "entry": entries.get(heading, ""),
            "via": "semantic",
        }
        for heading, score in index.search(vector, limit=limit)
    ]


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------


def _rrf(ranked: list[list[dict[str, Any]]], limit: int, k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal rank fusion.

    Chosen over score averaging because the two backends' scores are not
    comparable — one is a count of shared words, the other a cosine — and
    normalising them against each other would invent a relationship that does
    not exist. RRF only uses rank, which both produce honestly.
    """
    merged: dict[str, dict[str, Any]] = {}
    for results in ranked:
        for rank, item in enumerate(results, 1):
            row = merged.setdefault(
                item["heading"],
                {**item, "via": item["via"], "fused": 0.0, "sources": []},
            )
            row["fused"] += 1.0 / (k + rank)
            row["sources"].append(item["via"])
            if item.get("matched_words"):
                row["matched_words"] = item["matched_words"]
    out = sorted(merged.values(), key=lambda r: (-r["fused"], r["heading"]))
    for row in out:
        row["via"] = "+".join(sorted(set(row["sources"])))
        row["score"] = round(row.pop("fused"), 5)
        row.pop("sources", None)
    return out[:limit]


def search(
    description: str, limit: int = 5, *, mode: str | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Candidates and the mode that actually produced them.

    The returned mode is not always the requested one: `semantic` degrades to
    `keyword` when there is no key or no index. Reporting what happened rather
    than what was asked for is the difference between a result you can read and
    one you cannot.
    """
    mode = mode or retrieval_mode()

    if mode == "keyword":
        return keyword_search(description, limit), "keyword"

    if mode == "semantic":
        hits = semantic_search(description, limit)
        if hits:
            return hits, "semantic"
        return keyword_search(description, limit), "keyword (semantic unavailable)"

    kw = keyword_search(description, limit * 2)
    sem = semantic_search(description, limit * 2)
    if not sem:
        return kw[:limit], "keyword (semantic unavailable)"
    return _rrf([kw, sem], limit), "hybrid"
