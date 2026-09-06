"""Document ingestion: PDF in, addressable chunks out.

Every chunk carries enough metadata to point back at the exact span of the
exact page of the exact document it came from, and `locate()` will go and prove
it by re-extracting. That is the property the whole module is built around: a
citation that cannot be checked is not a citation, and this project's entire
claim is that its rates are citable.

    result = ingest_pdf("data/reference/primary/19-2025-CTR.pdf")
    for chunk in result.chunks:
        print(chunk.chunk_id, chunk.page_number, chunk.text[:60])

### Identity

`document_id` is the SHA-256 of the file's **bytes**, not its path. Two copies
of the same notification in different directories are one document; a
re-vendored file that differs by a byte is a different one, loudly. That is the
same discipline `agent/gazette.py` applies to the pinned corpus, for the same
reason.

`chunk_id` is `<document_id>:p<page>:c<ordinal>`, which is both unique and
readable. It encodes the source location in the identifier itself, so a chunk
quoted in a bug report is traceable without a database.

### Determinism

Chunking is a pure function of `(page text, size, overlap)`. No randomness, no
clock, no global state. The same PDF ingested twice produces byte-identical
chunk ids and text — asserted by test, because an ingestion pipeline that
drifts silently invalidates every downstream index built from it.

### Backends

PyMuPDF is preferred: it reports per-page text with better fidelity on the
tabular layouts these Gazette notifications use. It is **AGPL-3.0**, and this
repository is MIT, so it is an optional extra rather than a dependency —
`pip install -e '.[ingest]'`. Without it the module falls back to `pypdf`
(BSD) and records which extractor produced each chunk, because two extractors
do not agree character-for-character and a chunk's offsets are only meaningful
against the one that produced them.

Embeddings are deliberately **not** built here. Ingestion answers "what does
this document say and where"; indexing is `agent/retrieval.py`, and keeping
them apart is what lets the chunking be tested without a network or a key.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

#: Target characters per chunk, and how much each overlaps its predecessor.
#: Chosen for prose; the Gazette schedules are chunked by *entry* instead, in
#: `agent/retrieval.py`, because a schedule row is a citable unit and a token
#: window would cut it in half. Both are structural choices, not defaults.
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150

#: Below this many non-whitespace characters a page is treated as empty rather
#: than as text. Scanned pages extract to a handful of stray glyphs, and
#: calling that "text" produces chunks that look real and say nothing.
MIN_PAGE_CHARS = 24

#: Closed vocabulary, mirroring `agent/contract.py`. A caller switching on
#: these needs them to be a fixed set.
INGEST_ERRORS: frozenset[str] = frozenset(
    {
        "file_missing",
        "not_a_pdf",
        "unreadable_document",
        "unreadable_page",
        "empty_document",
        "no_backend",
        "bad_argument",
    }
)


class IngestError(Exception):
    """Raised only for programming errors — an unknown code, a bad argument.

    A document that cannot be read is *data*, not an exception: it comes back
    as an `IngestResult` with `ok=False` and a structured error, because the
    caller has to be able to reason about a bad file rather than crash on one.
    """


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """One addressable span of one page of one document."""

    document_id: str
    page_number: int  # 1-based, as a human would cite it
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PageReport:
    """What happened to one page. Every page gets one, including the bad ones.

    Pages that produced nothing are reported rather than dropped: a 52-page
    notification that silently yielded 41 pages of text is a corpus with a hole
    in it, and the hole is invisible if only successes are recorded.
    """

    page_number: int
    status: str  # "ok" | "empty" | "unreadable"
    chars: int = 0
    chunks: int = 0
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status,
            "chars": self.chars,
            "chunks": self.chunks,
            "detail": self.detail,
        }


@dataclass(slots=True)
class IngestResult:
    ok: bool
    document_id: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    pages: list[PageReport] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def readable_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == "ok")

    def error(self, code: str, detail: str, **extra: Any) -> "IngestResult":
        if code not in INGEST_ERRORS:
            raise IngestError(
                f"unknown ingestion error {code!r}; the set is closed. "
                f"Known: {sorted(INGEST_ERRORS)}"
            )
        self.ok = False
        self.errors.append({"error": code, "detail": detail, **extra})
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "document_id": self.document_id,
            "chunks": [c.to_json() for c in self.chunks],
            "pages": [p.to_json() for p in self.pages],
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def document_id_for(data: bytes) -> str:
    """Content-addressed, so identity travels with the bytes and not the path."""
    return "doc_" + hashlib.sha256(data).hexdigest()[:16]


def chunk_id_for(document_id: str, page_number: int, ordinal: int) -> str:
    """Unique, readable, and a source location in itself."""
    return f"{document_id}:p{page_number:04d}:c{ordinal:03d}"


_CHUNK_ID_RE = re.compile(r"^(doc_[0-9a-f]{16}):p(\d{4}):c(\d{3})$")


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int]:
    """`(document_id, page_number, ordinal)`. Raises on a malformed id."""
    m = _CHUNK_ID_RE.match(chunk_id or "")
    if not m:
        raise IngestError(f"malformed chunk_id {chunk_id!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_page(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[tuple[int, int, str]]:
    """Split one page into `(char_start, char_end, text)` spans.

    Deterministic and pure. Breaks on whitespace where one is available inside
    the last 20 % of the window, so chunks end at word boundaries without the
    boundary search being able to collapse a chunk to nothing.

    Offsets are into the **page's extracted text**, which is what makes
    `locate()` able to verify a chunk later. They are not offsets into the PDF.
    """
    if size <= 0:
        raise IngestError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise IngestError(f"overlap must be in 0..{size - 1}, got {overlap}")

    if not text.strip():
        return []
    if len(text) <= size:
        return [(0, len(text), text)]

    spans: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer a word boundary, but only look back a bounded distance so
            # a run of non-whitespace cannot shrink the chunk arbitrarily.
            floor = start + int(size * 0.8)
            cut = text.rfind(" ", floor, end)
            if cut > start:
                end = cut
        spans.append((start, end, text[start:end]))
        if end >= n:
            break
        # Advance by at least one character, always, so the loop terminates
        # even when overlap is large relative to the step taken.
        start = max(start + 1, end - overlap)
    return spans


# --------------------------------------------------------------------------
# Extraction backends
# --------------------------------------------------------------------------


def _extract_pymupdf(path: Path) -> tuple[list[str | None], dict[str, Any]]:
    import fitz  # PyMuPDF

    pages: list[str | None] = []
    with fitz.open(str(path)) as doc:
        meta = {
            "pdf_page_count": doc.page_count,
            "pdf_metadata": {
                k: v for k, v in (doc.metadata or {}).items() if isinstance(v, str)
            },
            "encrypted": bool(doc.is_encrypted),
        }
        for number in range(doc.page_count):
            try:
                pages.append(doc.load_page(number).get_text("text"))
            except Exception as exc:  # noqa: BLE001 — one bad page is not a bad file
                pages.append(None)
                meta.setdefault("page_errors", {})[number + 1] = (
                    f"{type(exc).__name__}: {exc}"
                )
    return pages, meta


def _extract_pypdf(path: Path) -> tuple[list[str | None], dict[str, Any]]:
    import logging

    import pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = pypdf.PdfReader(str(path))
    meta: dict[str, Any] = {
        "pdf_page_count": len(reader.pages),
        "encrypted": bool(getattr(reader, "is_encrypted", False)),
    }
    pages: list[str | None] = []
    for number, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            pages.append(None)
            meta.setdefault("page_errors", {})[number + 1] = f"{type(exc).__name__}: {exc}"
    return pages, meta


def available_backends() -> list[str]:
    found = []
    try:
        import fitz  # noqa: F401

        found.append("pymupdf")
    except ImportError:
        pass
    try:
        import pypdf  # noqa: F401

        found.append("pypdf")
    except ImportError:
        pass
    return found


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def ingest_pdf(
    path: str | Path,
    *,
    size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    backend: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> IngestResult:
    """Read one PDF into addressable chunks. Never raises for a bad document."""
    path = Path(path)
    result = IngestResult(ok=True)

    if not path.is_file():
        return result.error("file_missing", f"no file at {path}", path=str(path))

    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        return result.error(
            "not_a_pdf",
            f"{path.name} does not begin with a PDF header",
            path=str(path),
        )

    result.document_id = document_id_for(data)

    backends = available_backends()
    if not backends:
        return result.error(
            "no_backend",
            "no PDF backend available: pip install -e '.[ingest]' for PyMuPDF, "
            "or '.[gazette]' for pypdf",
        )
    chosen = backend or backends[0]
    if chosen not in backends:
        return result.error(
            "bad_argument",
            f"backend {chosen!r} is not installed; available: {backends}",
        )

    extractor = _extract_pymupdf if chosen == "pymupdf" else _extract_pypdf
    try:
        raw_pages, pdf_meta = extractor(path)
    except Exception as exc:  # noqa: BLE001 — a corrupt file is data, not a crash
        return result.error(
            "unreadable_document",
            f"{type(exc).__name__}: {exc}",
            path=str(path),
            backend=chosen,
        )

    result.metadata = {
        "source_path": str(path),
        "source_name": path.name,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_bytes": len(data),
        "backend": chosen,
        # Recorded because two extractors do not agree character-for-character,
        # and a chunk's offsets are only meaningful against the one that
        # produced them.
        "chunk_chars": size,
        "overlap_chars": overlap,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **pdf_meta,
        **(extra_metadata or {}),
    }

    for number, page_text in enumerate(raw_pages, start=1):
        if page_text is None:
            detail = (pdf_meta.get("page_errors") or {}).get(number, "extraction failed")
            result.pages.append(
                PageReport(number, "unreadable", detail=str(detail))
            )
            result.error(
                "unreadable_page",
                f"page {number} could not be extracted: {detail}",
                page_number=number,
            )
            continue

        stripped = re.sub(r"\s+", "", page_text)
        if len(stripped) < MIN_PAGE_CHARS:
            # Almost always a scanned image. Reported, not dropped: a
            # notification that silently yielded fewer pages than it has is a
            # corpus with an invisible hole in it.
            result.pages.append(
                PageReport(
                    number,
                    "empty",
                    chars=len(stripped),
                    detail=(
                        f"{len(stripped)} non-whitespace characters, below the "
                        f"{MIN_PAGE_CHARS}-character floor; likely a scanned page"
                    ),
                )
            )
            continue

        spans = chunk_page(page_text, size=size, overlap=overlap)
        for ordinal, (char_start, char_end, chunk_text) in enumerate(spans):
            result.chunks.append(
                Chunk(
                    document_id=result.document_id,
                    page_number=number,
                    chunk_id=chunk_id_for(result.document_id, number, ordinal),
                    text=chunk_text,
                    metadata={
                        # Everything needed to go back to the source span. The
                        # path is here as well as the name because `locate()`
                        # has to be able to find the file from a chunk alone —
                        # a chunk that knows which document it came from but
                        # not where it is does not satisfy "reproduce the
                        # source location", it only claims to.
                        "source_path": str(path),
                        "source_name": path.name,
                        "source_sha256": result.metadata["source_sha256"],
                        "page_number": number,
                        "chunk_ordinal": ordinal,
                        "char_start": char_start,
                        "char_end": char_end,
                        "page_chars": len(page_text),
                        "backend": chosen,
                    },
                )
            )
        result.pages.append(
            PageReport(number, "ok", chars=len(page_text), chunks=len(spans))
        )

    if not result.chunks:
        result.error(
            "empty_document",
            f"{path.name} produced no text on any of its {len(raw_pages)} page(s); "
            "it may be a scan needing OCR",
            pages=len(raw_pages),
        )

    return result


def ingest_many(
    paths: Iterable[str | Path], **kwargs: Any
) -> dict[str, IngestResult]:
    """Ingest several documents, keyed by `document_id`.

    **Text from two documents is never mixed**, and that is structural rather
    than careful: each call to `ingest_pdf` sees one file, every chunk id is
    namespaced by its document's content hash, and the invariant is asserted
    here before the results are returned. A duplicate id means the same bytes
    were ingested twice, which is reported rather than silently merged.
    """
    out: dict[str, IngestResult] = {}
    for path in paths:
        result = ingest_pdf(path, **kwargs)
        for chunk in result.chunks:
            if chunk.document_id != result.document_id:
                raise IngestError(
                    f"chunk {chunk.chunk_id} claims document "
                    f"{chunk.document_id} inside {result.document_id}"
                )
        if result.document_id and result.document_id in out:
            result.error(
                "bad_argument",
                f"document {result.document_id} was already ingested from "
                f"{out[result.document_id].metadata.get('source_path')}",
            )
        out[result.document_id or f"failed:{path}"] = result
    return out


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def locate(chunk: Chunk, *, root: str | Path | None = None) -> str:
    """Re-extract the chunk's source span and return it.

    This is the requirement "store enough metadata to reproduce the source
    location" made executable rather than asserted. It re-opens the file named
    in the chunk's metadata, checks the bytes still hash to the same document,
    re-extracts the same page with the same backend, and slices the recorded
    offsets. A mismatch raises.
    """
    meta = chunk.metadata
    path = Path(root or ".") / meta["source_name"] if root else None
    if path is None or not path.is_file():
        path = Path(meta.get("source_path", meta["source_name"]))
    if not path.is_file():
        raise IngestError(f"source not found for {chunk.chunk_id}: {path}")

    data = path.read_bytes()
    if document_id_for(data) != chunk.document_id:
        raise IngestError(
            f"{path.name} no longer hashes to {chunk.document_id}; the source "
            "changed since ingestion and the offsets are meaningless"
        )

    extractor = _extract_pymupdf if meta.get("backend") == "pymupdf" else _extract_pypdf
    pages, _ = extractor(path)
    page_text = pages[chunk.page_number - 1]
    if page_text is None:
        raise IngestError(f"page {chunk.page_number} is no longer readable")
    return page_text[meta["char_start"] : meta["char_end"]]


def verify(chunk: Chunk, **kwargs: Any) -> bool:
    """True when the chunk's text is still exactly at its recorded location."""
    return locate(chunk, **kwargs) == chunk.text


def iter_chunks(results: Iterable[IngestResult]) -> Iterator[Chunk]:
    for result in results:
        yield from result.chunks


def summarise(results: Sequence[IngestResult]) -> dict[str, Any]:
    return {
        "documents": len(results),
        "ok": sum(1 for r in results if r.ok),
        "chunks": sum(len(r.chunks) for r in results),
        "pages": sum(r.page_count for r in results),
        "pages_ok": sum(r.readable_pages for r in results),
        "pages_empty": sum(
            1 for r in results for p in r.pages if p.status == "empty"
        ),
        "pages_unreadable": sum(
            1 for r in results for p in r.pages if p.status == "unreadable"
        ),
        "errors": [e for r in results for e in r.errors],
    }
