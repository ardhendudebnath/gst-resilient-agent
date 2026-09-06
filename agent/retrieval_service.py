"""The retrieval seam: one interface, two backends, no Supabase in the agent.

The agent depends on `RetrievalService` and on nothing else. Whether the
vectors live in a process-local array or in Postgres is a deployment decision
made in `build_service()`, and no tool, no loop and no test needs to know which
it got.

    service = build_service()          # in-memory, or pgvector if configured
    for hit in service.search("quartz slabs bonded with resin", limit=5):
        print(hit.chunk_id, hit.page_number, round(hit.score, 3))

### Which corpus this is

**Document chunks, not schedule entries.** `agent/retrieval.py` searches the
961 tariff headings of the rated schedule and is keyed by heading; that corpus
is small, structured and stays in memory. This one is keyed by `chunk_id` and
carries a page number, because it searches *ingested documents* — the advance
rulings, which run to thousands of pages and are where a vector database
eventually earns its place.

The two are deliberately separate. Collapsing them would force the schedule
lookup, which must be exact and citable to a specific entry, to go through an
approximate nearest-neighbour search.

### Determinism

`exact=True` (the default) scans every embedded chunk and is reproducible: the
same query against the same corpus returns the same rows in the same order,
because ties are broken on `chunk_id` rather than left to whatever order the
storage engine happens to yield.

`exact=False` uses an ANN index and is **not** reproducible — approximate
search may return different neighbours after a reindex, a vacuum, or a change
in `hnsw.ef_search`. That is the trade, it is not hidden behind a default, and
`SearchHit.exact` records which one produced a given row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk, in the shape every backend must return."""

    chunk_id: str
    document_id: str
    page_number: int
    text: str
    #: Cosine similarity in 0..1, where 1 is identical. Normalised the same way
    #: by both backends so a score is comparable across them — pgvector's `<=>`
    #: is cosine *distance*, and returning that raw would invert the ordering
    #: relative to the in-memory backend.
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    #: False when this row came from an approximate index.
    exact: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
            "exact": self.exact,
        }


@runtime_checkable
class RetrievalService(Protocol):
    """What the agent may depend on. Implementations must not leak a driver."""

    name: str

    def available(self) -> bool: ...

    def upsert(self, chunks: Sequence[Any], *, embeddings: Sequence[Sequence[float]]) -> int: ...

    def search(
        self, query: str, *, limit: int = 5, exact: bool = True,
        document_id: str | None = None,
    ) -> list[SearchHit]: ...


# --------------------------------------------------------------------------
# In-memory
# --------------------------------------------------------------------------


class InMemoryRetrievalService:
    """Brute-force cosine over an in-process array. The default.

    Right for the corpus this project actually has today, and saying why
    matters because the instinct is to reach for a database: a few thousand
    chunks at 2048 dimensions is a handful of megabytes, one contiguous scan,
    and a few million multiply-adds. A Postgres round trip per query would cost
    more in latency than the scan costs in CPU, and it would put a network
    dependency inside the tool path.

    It is also exact, and therefore deterministic, at no extra cost.
    """

    name = "in-memory"

    def __init__(self) -> None:
        from array import array

        self._array = array
        self._ids: list[str] = []
        self._rows: list[Any] = []
        self._meta: dict[str, dict[str, Any]] = {}

    def available(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._ids)

    def upsert(
        self, chunks: Sequence[Any], *, embeddings: Sequence[Sequence[float]]
    ) -> int:
        from agent.embed import normalise

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks against {len(embeddings)} embeddings"
            )
        written = 0
        for chunk, vector in zip(chunks, embeddings):
            key = chunk.chunk_id
            payload = {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
            }
            if key in self._meta:
                self._rows[self._ids.index(key)] = normalise(vector)
            else:
                self._ids.append(key)
                self._rows.append(normalise(vector))
            self._meta[key] = payload
            written += 1
        return written

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        exact: bool = True,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        from agent.embed import dot, embed_query, normalise

        if not self._ids:
            return []
        q = normalise(embed_query(query))
        scored: list[tuple[float, str]] = []
        for key, row in zip(self._ids, self._rows):
            if document_id and self._meta[key]["document_id"] != document_id:
                continue
            scored.append((dot(q, row), key))
        # Ties broken on chunk_id, so the order is reproducible rather than
        # dependent on insertion order.
        scored.sort(key=lambda r: (-r[0], r[1]))
        return [
            SearchHit(
                chunk_id=key,
                document_id=self._meta[key]["document_id"],
                page_number=self._meta[key]["page_number"],
                text=self._meta[key]["text"],
                score=round(float(score), 6),
                metadata=self._meta[key]["metadata"],
                exact=True,
            )
            for score, key in scored[:limit]
        ]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def build_service(prefer: str | None = None) -> RetrievalService:
    """The configured backend, or the in-memory one.

    Falls back rather than raising, and the returned object's `name` says which
    one arrived — a caller that silently got a different backend than it asked
    for is how a benchmark ends up comparing two systems it thinks are one.
    """
    import os

    choice = (prefer or os.environ.get("RETRIEVAL_BACKEND", "") or "auto").lower()

    if choice in ("memory", "in-memory"):
        return InMemoryRetrievalService()

    if choice in ("auto", "pgvector", "postgres", "supabase"):
        try:
            from storage.vectors import PgVectorRetrievalService

            service = PgVectorRetrievalService()
            if service.available():
                return service
        except Exception:  # noqa: BLE001 — an unconfigured database is normal
            pass
        if choice != "auto":
            raise RuntimeError(
                f"retrieval backend {choice!r} was requested but is not "
                "available; set DATABASE_URL and pip install -e '.[storage]'"
            )
    return InMemoryRetrievalService()
