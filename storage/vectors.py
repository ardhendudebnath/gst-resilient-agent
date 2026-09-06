"""pgvector-backed retrieval. Implements `agent.retrieval_service.RetrievalService`.

The agent never imports this module. It depends on the protocol; `build_service`
picks the implementation. That is the whole point of the seam — Supabase is a
deployment decision, not an architectural one.

### Exact against approximate, and why the default is exact

    exact=True   ORDER BY embedding <=> query        full scan, deterministic
    exact=False  ORDER BY embedding::halfvec <=> …   HNSW index, approximate

Requirement "make retrieval deterministic where possible" and requirement "add
indexes appropriate for vector search" pull in opposite directions, and it is
worth being plain about it rather than shipping one and implying the other. An
ANN index returns *approximate* neighbours: the same query can return different
rows after a reindex, after a vacuum changes the visibility map, or when
`hnsw.ef_search` differs between sessions. That is what makes it fast.

So both exist, exact is the default, ties break on `chunk_id` so even exact
ordering is stable, and every row records which path produced it.

### The 2048-dimension problem

`nemotron-3-embed-1b` emits 2048 dimensions. pgvector stores that fine, but its
ivfflat and hnsw indexes stop at 2000, so the obvious index cannot be created
at all. The column is `vector(2048)` for exact search and the index is built on
a `halfvec(2048)` expression, which hnsw supports to 4000. Half precision costs
a little recall on the approximate path and nothing on the exact one. See the
comment block at the head of `migrations/0001_initial.sql`.
"""

from __future__ import annotations

from typing import Any, Sequence

from agent.retrieval_service import SearchHit
from storage import config


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input format: `[1,2,3]`.

    Built by hand rather than with a driver adapter so that psycopg is the only
    dependency and `pgvector-python` is not required. Values are floats that
    came from our own embedding call, so there is nothing here to escape.
    """
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


class PgVectorRetrievalService:
    """Chunks and their embeddings in Postgres."""

    name = "pgvector"

    def __init__(self, url: str | None = None, *, dim: int = 2048) -> None:
        self._url = url
        self._dim = dim
        self._conn: Any = None

    # -- lifecycle -------------------------------------------------------

    def _connection(self):
        if self._conn is None:
            self._conn = config.connect(self._url)
        return self._conn

    def available(self) -> bool:
        """True when a database is configured and reachable.

        Actually connects. `configured()` alone would report a typo'd DSN as
        available and fail later inside a tool call, where it is far more
        expensive to diagnose.
        """
        if not (self._url or config.configured()):
            return False
        try:
            self._connection().execute("SELECT 1")
            return True
        except config.StorageUnavailable:
            return False

    def close(self) -> None:
        if self._conn is not None:
            self._conn.raw.close()
            self._conn = None

    # -- writes ----------------------------------------------------------

    def upsert_document(self, result: Any) -> None:
        """Record an ingested document. Idempotent on `document_id`."""
        conn = self._connection()
        meta = config.redact(dict(result.metadata))
        conn.execute(
            """
            INSERT INTO documents (document_id, source_name, source_sha256,
                                   source_bytes, page_count, pages_ok,
                                   pages_empty, pages_unreadable, extractor,
                                   metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                page_count = EXCLUDED.page_count,
                pages_ok = EXCLUDED.pages_ok,
                metadata = EXCLUDED.metadata
            """,
            (
                result.document_id,
                meta.get("source_name", ""),
                meta.get("source_sha256", ""),
                int(meta.get("source_bytes", 0)),
                result.page_count,
                result.readable_pages,
                sum(1 for p in result.pages if p.status == "empty"),
                sum(1 for p in result.pages if p.status == "unreadable"),
                meta.get("backend", "unknown"),
                _json(meta),
            ),
        )
        conn.commit()

    def upsert(
        self, chunks: Sequence[Any], *, embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Write chunks with their vectors. Idempotent on `chunk_id`."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks against {len(embeddings)} embeddings")
        if not chunks:
            return 0

        from agent.embed import embed_model

        conn = self._connection()
        model = embed_model()
        rows = []
        for chunk, vector in zip(chunks, embeddings):
            if len(vector) != self._dim:
                raise ValueError(
                    f"{chunk.chunk_id}: expected {self._dim} dimensions, got "
                    f"{len(vector)}. An index built at one width cannot answer "
                    "queries at another."
                )
            meta = config.redact(dict(chunk.metadata))
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.page_number,
                    int(meta.get("chunk_ordinal", 0)),
                    int(meta.get("char_start", 0)),
                    int(meta.get("char_end", 0)),
                    chunk.text,
                    _vector_literal(vector),
                    model,
                    _json(meta),
                )
            )

        conn.executemany(
            """
            INSERT INTO document_chunks (chunk_id, document_id, page_number,
                                         chunk_ordinal, char_start, char_end,
                                         text, embedding, embedding_model,
                                         metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                metadata = EXCLUDED.metadata
            """,
            rows,
        )
        conn.commit()
        return len(rows)

    # -- reads -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        exact: bool = True,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        from agent.embed import embed_query

        return self.search_vector(
            embed_query(query), limit=limit, exact=exact, document_id=document_id
        )

    def search_vector(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        exact: bool = True,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Nearest chunks. Separated from `search` so a caller with a vector
        already in hand does not pay for a second embedding call."""
        conn = self._connection()
        literal = _vector_literal(vector)

        # `<=>` is cosine *distance*; similarity is 1 - distance. Converting
        # here keeps a score comparable with the in-memory backend, where a
        # higher number is a better match.
        if exact:
            order = "embedding <=> %(q)s::vector"
            score = "1 - (embedding <=> %(q)s::vector)"
        else:
            order = "embedding::halfvec(2048) <=> %(q)s::halfvec(2048)"
            score = "1 - (embedding::halfvec(2048) <=> %(q)s::halfvec(2048))"

        where = ["embedding IS NOT NULL"]
        params: dict[str, Any] = {"q": literal, "limit": int(limit)}
        if document_id:
            where.append("document_id = %(doc)s")
            params["doc"] = document_id

        cur = conn.execute(
            f"""
            SELECT chunk_id, document_id, page_number, text, metadata,
                   {score} AS score
            FROM document_chunks
            WHERE {' AND '.join(where)}
            -- chunk_id breaks ties, so equal scores come back in a stable
            -- order instead of whatever the executor happens to emit.
            ORDER BY {order}, chunk_id
            LIMIT %(limit)s
            """,
            params,
        )
        return [
            SearchHit(
                chunk_id=row[0],
                document_id=row[1],
                page_number=row[2],
                text=row[3],
                score=round(float(row[5]), 6),
                metadata=row[4] or {},
                exact=exact,
            )
            for row in cur.fetchall()
        ]

    def count(self) -> int:
        cur = self._connection().execute(
            "SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL"
        )
        return int(cur.fetchone()[0])


def _json(value: Any) -> str:
    import json

    return json.dumps(config.redact(value), ensure_ascii=False, default=str)
