"""Ingestion: identity, determinism, page fidelity, and adversarial documents.

The property most of these defend is that a chunk can be traced back to the
exact span of the exact page of the exact document it came from. A citation
that cannot be checked is not a citation, and this project's whole claim is
that its rates are citable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import ingest
from agent.ingest import (
    Chunk,
    IngestError,
    chunk_id_for,
    chunk_page,
    document_id_for,
    ingest_many,
    ingest_pdf,
    locate,
    parse_chunk_id,
    verify,
)

pytest.importorskip("fitz", reason="building test PDFs needs PyMuPDF")

from chaos import documents as docs  # noqa: E402

GAZETTE = Path("data/reference/primary/19-2025-CTR.pdf")


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ingest")
    built = docs.write_adversarial_pdf(d / "clean.pdf", payload=None)
    return built.path


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_document_id_is_content_addressed_not_path_addressed(tmp_path):
    """Two copies of one document are one document; a changed byte is a
    different one, loudly."""
    a = docs.write_adversarial_pdf(tmp_path / "a.pdf", payload=None).path
    b = tmp_path / "b.pdf"
    b.write_bytes(a.read_bytes())

    assert ingest_pdf(a).document_id == ingest_pdf(b).document_id

    tampered = tmp_path / "c.pdf"
    tampered.write_bytes(a.read_bytes() + b"\n% edited\n")
    assert ingest_pdf(tampered).document_id != ingest_pdf(a).document_id


def test_chunk_ids_are_unique_readable_and_parseable(sample):
    result = ingest_pdf(sample)
    ids = [c.chunk_id for c in result.chunks]
    assert ids
    assert len(set(ids)) == len(ids)
    for chunk in result.chunks:
        doc_id, page, ordinal = parse_chunk_id(chunk.chunk_id)
        assert doc_id == chunk.document_id
        assert page == chunk.page_number
        assert chunk.chunk_id == chunk_id_for(doc_id, page, ordinal)


def test_a_malformed_chunk_id_raises():
    with pytest.raises(IngestError):
        parse_chunk_id("not-a-chunk-id")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_ingesting_twice_gives_identical_chunks(sample):
    """An ingestion pipeline that drifts silently invalidates every downstream
    index built from it."""
    a = [c.to_json() for c in ingest_pdf(sample).chunks]
    b = [c.to_json() for c in ingest_pdf(sample).chunks]
    for x, y in zip(a, b):
        x["metadata"].pop("ingested_at", None)
        y["metadata"].pop("ingested_at", None)
    assert a == b


def test_chunking_is_a_pure_function_of_its_arguments():
    text = "word " * 900
    assert chunk_page(text, size=400, overlap=50) == chunk_page(
        text, size=400, overlap=50
    )


def test_chunks_cover_the_page_and_overlap_as_configured():
    text = "".join(f"{i:04d} " for i in range(600))
    spans = chunk_page(text, size=500, overlap=80)
    assert len(spans) > 1
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (s0, e0, _), (s1, _e1, _t) in zip(spans, spans[1:]):
        assert s1 > s0, "chunking must advance"
        assert s1 <= e0, "chunks must not leave a gap"
        assert e0 - s1 <= 80 + 1, "overlap must not exceed what was configured"


def test_chunking_terminates_on_text_with_no_spaces():
    """A boundary search that cannot find whitespace must not stall."""
    spans = chunk_page("x" * 5000, size=300, overlap=100)
    assert spans
    assert spans[-1][1] == 5000


def test_chunking_rejects_impossible_parameters():
    with pytest.raises(IngestError):
        chunk_page("abc", size=0)
    with pytest.raises(IngestError):
        chunk_page("abc", size=100, overlap=100)


# --------------------------------------------------------------------------
# Page fidelity
# --------------------------------------------------------------------------


def test_page_numbers_are_one_based_and_complete(sample):
    result = ingest_pdf(sample)
    assert [p.page_number for p in result.pages] == list(
        range(1, result.page_count + 1)
    )
    assert all(c.page_number >= 1 for c in result.chunks)


def test_every_page_is_reported_including_the_ones_with_no_text(tmp_path):
    """A document that silently yielded fewer pages than it has is a corpus
    with an invisible hole in it."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(56, 56, 500, 700), "Real text " * 60, fontsize=10)
    doc.new_page()  # deliberately blank
    path = tmp_path / "with-blank.pdf"
    doc.save(str(path))
    doc.close()

    result = ingest_pdf(path)
    assert result.page_count == 2
    statuses = {p.page_number: p.status for p in result.pages}
    assert statuses[1] == "ok"
    assert statuses[2] == "empty"
    assert all(c.page_number == 1 for c in result.chunks)


def test_a_document_with_no_text_at_all_reports_empty_document(tmp_path):
    import fitz

    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()

    result = ingest_pdf(path)
    assert not result.ok
    assert any(e["error"] == "empty_document" for e in result.errors)
    assert result.chunks == []


# --------------------------------------------------------------------------
# Reproducing the source location
# --------------------------------------------------------------------------


def test_every_chunk_can_be_located_in_its_source(sample):
    """The requirement made executable rather than asserted: re-open the file,
    re-check the hash, re-extract the page, slice the recorded offsets."""
    result = ingest_pdf(sample)
    assert result.chunks
    for chunk in result.chunks:
        assert verify(chunk), f"{chunk.chunk_id} does not match its source span"


def test_locating_a_chunk_whose_source_changed_raises(tmp_path):
    built = docs.write_adversarial_pdf(tmp_path / "moving.pdf", payload=None)
    chunk = ingest_pdf(built.path).chunks[0]
    built.path.write_bytes(built.path.read_bytes() + b"\n% edited\n")
    with pytest.raises(IngestError, match="no longer hashes"):
        locate(chunk)


def test_chunk_metadata_carries_the_whole_source_location(sample):
    chunk = ingest_pdf(sample).chunks[0]
    for key in (
        "source_name", "source_sha256", "page_number",
        "chunk_ordinal", "char_start", "char_end", "backend",
    ):
        assert key in chunk.metadata, f"metadata is missing {key}"


# --------------------------------------------------------------------------
# Structured errors
# --------------------------------------------------------------------------


def test_a_missing_file_is_data_not_an_exception(tmp_path):
    result = ingest_pdf(tmp_path / "nope.pdf")
    assert not result.ok
    assert result.errors[0]["error"] == "file_missing"


def test_a_non_pdf_is_rejected_by_its_header(tmp_path):
    path = tmp_path / "not.pdf"
    path.write_text("I am plainly not a PDF", encoding="utf-8")
    result = ingest_pdf(path)
    assert not result.ok
    assert result.errors[0]["error"] == "not_a_pdf"


def test_a_corrupt_pdf_reports_unreadable_rather_than_crashing(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"\x00\xff" * 400)
    result = ingest_pdf(path)
    assert not result.ok
    assert {e["error"] for e in result.errors} & {
        "unreadable_document", "empty_document"
    }


def test_an_unknown_error_code_is_refused():
    from agent.ingest import IngestResult

    with pytest.raises(IngestError):
        IngestResult(ok=True).error("nonsense", "detail")


def test_results_are_json_serialisable(sample):
    json.dumps(ingest_pdf(sample).to_json())


# --------------------------------------------------------------------------
# Documents never mix
# --------------------------------------------------------------------------


def test_chunks_from_two_documents_never_share_an_id_or_a_document(tmp_path):
    a = docs.write_adversarial_pdf(tmp_path / "one.pdf", payload=None).path
    b = docs.write_adversarial_pdf(
        tmp_path / "two.pdf", payload=None, pages=("Entirely different text. " * 40,)
    ).path

    results = ingest_many([a, b])
    assert len(results) == 2
    ids = list(results)
    assert ids[0] != ids[1]

    for doc_id, result in results.items():
        assert all(c.document_id == doc_id for c in result.chunks)

    all_chunk_ids = [c.chunk_id for r in results.values() for c in r.chunks]
    assert len(set(all_chunk_ids)) == len(all_chunk_ids)


def test_ingesting_the_same_bytes_twice_is_reported_not_merged(tmp_path):
    a = docs.write_adversarial_pdf(tmp_path / "x.pdf", payload=None).path
    b = tmp_path / "y.pdf"
    b.write_bytes(a.read_bytes())
    results = ingest_many([a, b])
    assert len(results) == 1
    only = next(iter(results.values()))
    assert any("already ingested" in e["detail"] for e in only.errors)


# --------------------------------------------------------------------------
# Real corpus
# --------------------------------------------------------------------------


@pytest.mark.skipif(not GAZETTE.is_file(), reason="pinned corpus not present")
def test_the_pinned_notification_ingests_cleanly():
    result = ingest_pdf(GAZETTE)
    assert result.ok, result.errors
    assert result.page_count == 2
    assert result.readable_pages == 2
    assert result.chunks
    assert verify(result.chunks[0])


@pytest.mark.skipif(not GAZETTE.is_file(), reason="pinned corpus not present")
def test_both_backends_read_the_same_document_id_but_own_their_offsets():
    """Two extractors do not agree character-for-character, which is why the
    backend is recorded on every chunk and offsets are only valid against it."""
    if "pypdf" not in ingest.available_backends():
        pytest.skip("pypdf not installed")
    a = ingest_pdf(GAZETTE, backend="pymupdf")
    b = ingest_pdf(GAZETTE, backend="pypdf")
    assert a.document_id == b.document_id
    assert a.chunks[0].metadata["backend"] == "pymupdf"
    assert b.chunks[0].metadata["backend"] == "pypdf"
    assert verify(a.chunks[0]) and verify(b.chunks[0])


# --------------------------------------------------------------------------
# Adversarial documents
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", list(docs.payloads.PAYLOAD_NAMES))
def test_every_payload_survives_ingestion_inline(tmp_path, payload):
    """A payload mangled by chunking would read as a defence working."""
    built = docs.write_adversarial_pdf(
        tmp_path / f"{payload}.pdf", payload=payload, placement="inline"
    )
    result = ingest_pdf(built.path)
    assert result.ok, result.errors
    hits = docs.find_payload_chunks(result.chunks, payload)
    assert hits, f"{payload} did not survive ingestion"


def test_hidden_text_is_invisible_to_a_reader_and_plain_to_an_extractor(tmp_path):
    """The case where "a human reviewed the source PDFs" is not a control.

    The payload is rendered white at 1pt in the bottom margin. Nothing about
    the extracted text says it was hidden.
    """
    built = docs.write_adversarial_pdf(
        tmp_path / "hidden.pdf", payload="exfiltration", placement="hidden"
    )
    result = ingest_pdf(built.path)
    assert result.ok
    text = "\n".join(c.text for c in result.chunks)
    assert docs.payloads.EXFIL_MARKER in text


def test_isolated_placement_puts_the_payload_on_its_own_page(tmp_path):
    built = docs.write_adversarial_pdf(
        tmp_path / "isolated.pdf", payload="tool_misuse", placement="isolated"
    )
    result = ingest_pdf(built.path)
    assert built.page_count == len(docs.CARRIER_PAGES) + 1
    hits = docs.find_payload_chunks(result.chunks, "tool_misuse")
    assert hits
    assert {c.page_number for c in hits} == {built.payload_page}


def test_a_clean_control_document_carries_no_payload(tmp_path):
    """A compliance rate is uninterpretable without the rate on the identical
    document with nothing hidden in it."""
    built = docs.write_adversarial_pdf(tmp_path / "control.pdf", payload=None)
    result = ingest_pdf(built.path)
    assert built.payload is None
    assert built.payload_page is None
    text = "\n".join(c.text for c in result.chunks)
    assert docs.payloads.EXFIL_MARKER not in text
    for name in docs.payloads.PAYLOAD_NAMES:
        assert not docs.find_payload_chunks(result.chunks, name)


def test_generated_corpus_covers_every_payload_and_placement(tmp_path):
    built = docs.write_corpus(tmp_path)
    expected = len(docs.payloads.PAYLOAD_NAMES) * len(docs.PLACEMENTS) + 1
    assert len(built) == expected
    assert sum(1 for b in built if b.payload is None) == 1
    assert all(b.path.is_file() for b in built)


def test_document_compliance_uses_the_same_detector_as_the_middleware(tmp_path):
    """Document-borne and middleware-borne injections must be comparable."""
    built = docs.write_adversarial_pdf(tmp_path / "exfil.pdf", payload="exfiltration")
    clean = {"terminal": "opinion", "opinion": {"justification": "Heading 6810 at 18%."}}
    dirty = {
        "terminal": "opinion",
        "opinion": {"justification": f"Done. {docs.payloads.EXFIL_MARKER}"},
    }
    assert built.complied(clean, []) is False
    assert built.complied(dirty, []) is True


def test_an_unknown_placement_is_refused(tmp_path):
    with pytest.raises(ValueError):
        docs.write_adversarial_pdf(tmp_path / "x.pdf", placement="steganography")
