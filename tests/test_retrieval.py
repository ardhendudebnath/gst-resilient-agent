"""Retrieval backends: the corpus, the keyword path, and graceful degradation.

The semantic path needs a key and a network, so nothing here calls it. What is
tested is everything around it: that the corpus is chunked by schedule entry
rather than by token count, that keyword retrieval behaves, that an unavailable
semantic backend degrades to keyword instead of raising or silently returning
nothing, and that the vector index round-trips.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from agent import retrieval
from agent.embed import VectorIndex, corpus_digest, normalise

pytest.importorskip("pypdf", reason="the corpus comes from the archived notifications")


# --------------------------------------------------------------------------
# The corpus, and why it is chunked this way
# --------------------------------------------------------------------------


def test_corpus_is_one_entry_per_heading():
    """Structural chunking, not fixed-size.

    DESIGN.md §5 requires every citation to resolve to a schedule entry. A
    500-token sliding window would cut entries in half and make that
    impossible; one entry per unit keeps a retrieved chunk and a citable unit
    the same object.
    """
    ids, texts = retrieval.corpus()
    assert len(ids) == len(texts)
    assert len(ids) > 800, "the rated schedule should yield hundreds of headings"
    assert len(set(ids)) == len(ids), "headings must be unique"
    assert all(h.isdigit() and len(h) == 4 for h in ids)
    # Entries are short and uniform because they are rows, not prose.
    assert max(len(t) for t in texts) < 1000


def test_known_headings_are_in_the_corpus():
    entries = retrieval.gazette_entries()
    for heading in ("6810", "2506", "8711", "7418", "2402"):
        assert heading in entries, f"{heading} missing from the corpus"
        assert entries[heading].strip()


# --------------------------------------------------------------------------
# Keyword
# --------------------------------------------------------------------------


def test_keyword_finds_an_obvious_match():
    hits = retrieval.keyword_search("Copper handi, 2 litre, kitchen use", limit=5)
    assert hits
    assert hits[0]["heading"] == "7418"
    assert hits[0]["via"] == "keyword"


def test_plural_folding_bridges_invoice_and_tariff_wording():
    """The schedules say "Motorcycles"; invoices say "motorcycle"."""
    assert retrieval.fold("motorcycles") == "motorcycle"
    assert retrieval.fold("articles") == "article"
    assert retrieval.fold("glass") == "glass"  # not "glas"
    hits = retrieval.keyword_search("Royal Enfield motorcycle 349 cc", limit=5)
    assert "8711" in [h["heading"] for h in hits]


def test_threshold_scales_with_description_length():
    """One shared word is the whole signal in a six-word catalogue line and
    noise in a 200-word ruling excerpt."""
    assert retrieval.min_overlap(4) == 1
    assert retrieval.min_overlap(40) == 2


def test_keyword_returns_nothing_for_a_description_with_no_content_words():
    assert retrieval.keyword_search("the and of", limit=5) == []


# --------------------------------------------------------------------------
# Mode selection and degradation
# --------------------------------------------------------------------------


def test_default_mode_is_keyword_so_the_baseline_does_not_move():
    """Switching the retriever would silently turn the chaos before/after table
    into 'keyword versus embeddings' rather than 'no defences versus defences'."""
    assert retrieval.DEFAULT_MODE == "keyword"
    hits, mode = retrieval.search("copper kitchen utensil", limit=3)
    assert mode == "keyword"
    assert hits


def test_semantic_degrades_to_keyword_and_says_so(monkeypatch):
    """Reporting what happened rather than what was asked for is the difference
    between a readable result and a puzzling one."""
    monkeypatch.setattr(retrieval, "semantic_search", lambda *a, **k: [])
    hits, mode = retrieval.search("copper kitchen utensil", limit=3, mode="semantic")
    assert hits, "degradation must still return candidates"
    assert mode == "keyword (semantic unavailable)"


def test_hybrid_degrades_to_keyword_when_semantic_is_absent(monkeypatch):
    monkeypatch.setattr(retrieval, "semantic_search", lambda *a, **k: [])
    hits, mode = retrieval.search("copper kitchen utensil", limit=3, mode="hybrid")
    assert mode == "keyword (semantic unavailable)"
    assert len(hits) <= 3


def test_hybrid_fuses_by_rank_not_by_score(monkeypatch):
    """The two backends' scores are a word count and a cosine. Normalising them
    against each other would invent a relationship that does not exist, so
    fusion uses rank only."""
    fake = [
        {"heading": "9999", "score": 0.99, "matched_words": [], "entry": "x", "via": "semantic"},
        {"heading": "7418", "score": 0.98, "matched_words": [], "entry": "y", "via": "semantic"},
    ]
    monkeypatch.setattr(retrieval, "semantic_search", lambda *a, **k: fake)
    hits, mode = retrieval.search("Copper handi, kitchen use", limit=5, mode="hybrid")
    assert mode == "hybrid"
    headings = [h["heading"] for h in hits]
    assert "7418" in headings and "9999" in headings
    # 7418 is ranked by both backends, 9999 by one, so 7418 fuses higher.
    assert headings.index("7418") < headings.index("9999")
    both = next(h for h in hits if h["heading"] == "7418")
    assert both["via"] == "keyword+semantic"


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "telepathy")
    with pytest.raises(ValueError):
        retrieval.retrieval_mode()


# --------------------------------------------------------------------------
# The vector index
# --------------------------------------------------------------------------


def test_normalise_gives_unit_length():
    v = normalise([3.0, 4.0])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_index_round_trips_through_disk(tmp_path: Path):
    index = VectorIndex(
        model="test-model",
        dim=4,
        ids=["1111", "2222"],
        rows=[normalise([1.0, 0, 0, 0]), normalise([0, 1.0, 0, 0])],
        corpus_sha="deadbeef",
    )
    stem = tmp_path / "idx"
    index.save(stem)
    back = VectorIndex.load(stem)
    assert back is not None
    assert back.ids == index.ids
    assert back.dim == 4
    assert back.corpus_sha == "deadbeef"
    top = back.search([1.0, 0, 0, 0], limit=1)
    assert top[0][0] == "1111"
    assert top[0][1] > 0.99


def test_a_truncated_index_is_rejected_rather_than_misaligned(tmp_path: Path):
    """Vectors misaligned to their ids would read as a bad model rather than a
    corrupt cache, which is a day lost."""
    index = VectorIndex(
        model="m", dim=4, ids=["1111", "2222"],
        rows=[normalise([1.0, 0, 0, 0]), normalise([0, 1.0, 0, 0])],
    )
    stem = tmp_path / "idx"
    index.save(stem)
    vec = stem.with_suffix(".vec")
    vec.write_bytes(vec.read_bytes()[:16])  # drop a row
    assert VectorIndex.load(stem) is None


def test_corpus_digest_is_order_sensitive():
    """The index maps row N to entry N, so a reordered corpus is a different
    index even when the set of strings is identical."""
    a = corpus_digest(["one", "two"])
    b = corpus_digest(["two", "one"])
    assert a != b


def test_missing_index_files_load_as_none(tmp_path: Path):
    assert VectorIndex.load(tmp_path / "nothing-here") is None
