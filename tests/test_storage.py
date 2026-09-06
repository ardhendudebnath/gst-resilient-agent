"""Storage: secret hygiene, the retrieval seam, and integration tests that skip.

Two populations of test here, and the split is the honest part.

**Unit tests** run everywhere and cover the logic that has no database in it:
redaction, DSN safety, the in-memory retrieval service, backend selection, and
the SQL text itself.

**Integration tests** need a live Postgres with pgvector and are skipped
without `DATABASE_URL`. They are marked rather than mocked: a mocked database
tests the mock, and the failure this project would actually hit is a schema
that does not apply or an index pgvector refuses to build. Until someone runs
these against a real instance, the SQL in `storage/migrations/` is
**unexercised**, and that is stated in the README rather than implied away.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agent.retrieval_service import (
    InMemoryRetrievalService,
    RetrievalService,
    SearchHit,
    build_service,
)
from storage import config

MIGRATIONS = Path(__file__).resolve().parent.parent / "storage" / "migrations"


def _sql_only() -> str:
    """Every migration with `--` comments stripped.

    The migrations explain themselves at length, and the header of 0001 quotes
    the index form that does *not* work as an illustration. A test that greps
    the raw file therefore matches prose about SQL rather than SQL, which is
    how both of these assertions first passed against the wrong thing.
    """
    body = "\n".join(p.read_text(encoding="utf-8") for p in sorted(MIGRATIONS.glob("*.sql")))
    return "\n".join(re.sub(r"--.*$", "", line) for line in body.splitlines())

integration = pytest.mark.skipif(
    not config.configured(),
    reason="needs a live Postgres: set DATABASE_URL",
)


# --------------------------------------------------------------------------
# Secrets never reach a row, a log, or a file
# --------------------------------------------------------------------------


def test_a_dsn_password_is_never_echoed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin:hunter2@db.example:5432/postgres")
    safe = config.safe_dsn()
    assert "hunter2" not in safe
    assert "[redacted]" in safe
    assert "db.example" in safe, "the host is useful and not secret"


def test_an_unparseable_dsn_is_not_echoed_verbatim(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://[not-a-url:::")
    assert "not-a-url" not in config.safe_dsn()


def test_redaction_matches_on_value_not_on_field_name(monkeypatch):
    """The realistic mistake is a key landing in a field nobody blocklisted."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-SUPERSECRETVALUE0123456789")
    payload = {
        "note": "called with nvapi-SUPERSECRETVALUE0123456789 and it worked",
        "nested": [{"deep": "nvapi-SUPERSECRETVALUE0123456789"}],
        "count": 3,
    }
    out = config.redact(payload)
    blob = repr(out)
    assert "SUPERSECRET" not in blob
    assert out["count"] == 3, "non-strings pass through untouched"
    assert "[redacted]" in out["note"]
    assert "[redacted]" in out["nested"][0]["deep"]


def test_token_shaped_strings_are_redacted_even_when_not_in_the_environment():
    """Backstop for a secret belonging to a different process."""
    out = config.redact({"header": "Bearer sk-abcdefghijklmnopqrstuvwxyz012345"})
    assert "abcdefghij" not in out["header"]


def test_no_migration_contains_a_credential():
    """A migration is committed to git; a credential in one is public."""
    for sql in MIGRATIONS.glob("*.sql"):
        body = sql.read_text(encoding="utf-8")
        assert "postgresql://" not in body
        assert not re.search(r"\b(password|secret|api_key)\s*=\s*'", body, re.I)


# --------------------------------------------------------------------------
# The schema, as text
# --------------------------------------------------------------------------


def test_every_required_table_is_created():
    body = _sql_only()
    for table in (
        "documents", "document_chunks", "agent_runs", "tool_calls",
        "failures", "chaos_runs", "evaluation_results",
    ):
        assert re.search(rf"CREATE TABLE IF NOT EXISTS {table}\b", body), table


def test_migrations_are_idempotent_by_construction():
    """Applying twice must be a no-op. A migration you are afraid to run twice
    is one nobody runs."""
    body = _sql_only()
    creates = re.findall(r"CREATE (TABLE|INDEX|EXTENSION|VIEW)([^;]*)", body)
    for kind, rest in creates:
        guarded = "IF NOT EXISTS" in rest or "OR REPLACE" in rest
        assert guarded, f"unguarded CREATE {kind}: {rest[:60]}"


def test_the_vector_index_works_around_the_2048_dimension_limit():
    """nemotron-3-embed-1b is 2048-dimensional and pgvector's ANN indexes stop
    at 2000, so a plain hnsw index on vector(2048) cannot be created at all.
    The index must go through halfvec, which hnsw supports to 4000."""
    body = _sql_only()
    assert "vector(2048)" in body, "the column stores full precision"
    index = re.search(r"CREATE INDEX[^;]*hnsw[^;]*", body, re.S)
    assert index, "no hnsw index found"
    assert "halfvec(2048)" in index.group(0)
    assert "halfvec_cosine_ops" in index.group(0)


def test_the_chaos_column_is_nullable_and_never_defaulted():
    """An injected failure that could be mistaken for an organic one would make
    the whole failure taxonomy fiction."""
    body = _sql_only()
    chaos_line = re.search(r"^\s*chaos\s+text.*$", body, re.M)
    assert chaos_line
    assert "NOT NULL" not in chaos_line.group(0)
    assert "DEFAULT" not in chaos_line.group(0)


# --------------------------------------------------------------------------
# The retrieval seam
# --------------------------------------------------------------------------


class _Chunk:
    def __init__(self, i: int, doc: str = "doc_0000000000000001"):
        self.chunk_id = f"{doc}:p0001:c{i:03d}"
        self.document_id = doc
        self.page_number = 1
        self.text = f"chunk number {i}"
        self.metadata = {"chunk_ordinal": i, "char_start": 0, "char_end": 10}


def test_in_memory_service_satisfies_the_protocol():
    assert isinstance(InMemoryRetrievalService(), RetrievalService)


def test_build_service_falls_back_to_memory_without_a_database(monkeypatch):
    for var in config.DSN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    service = build_service()
    assert service.name == "in-memory"


def test_asking_for_pgvector_without_one_is_an_error_not_a_silent_fallback(monkeypatch):
    """A caller that silently got a different backend than it asked for is how
    a benchmark ends up comparing two systems it thinks are one."""
    for var in config.DSN_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="not.*available"):
        build_service("pgvector")


def test_upsert_rejects_mismatched_lengths():
    service = InMemoryRetrievalService()
    with pytest.raises(ValueError):
        service.upsert([_Chunk(0)], embeddings=[])


def test_search_is_deterministic_and_breaks_ties_on_chunk_id(monkeypatch):
    """Equal scores must come back in a stable order rather than in insertion
    order, or 'deterministic retrieval' is only true until someone reindexes."""
    service = InMemoryRetrievalService()
    # Three identical vectors: every score ties exactly.
    chunks = [_Chunk(i) for i in (3, 1, 2)]
    service.upsert(chunks, embeddings=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr("agent.embed.embed_query", lambda *a, **k: [1.0, 0.0])

    first = [h.chunk_id for h in service.search("anything", limit=3)]
    second = [h.chunk_id for h in service.search("anything", limit=3)]
    assert first == second
    assert first == sorted(first), "ties must break on chunk_id"


def test_search_returns_the_required_shape(monkeypatch):
    service = InMemoryRetrievalService()
    service.upsert([_Chunk(1)], embeddings=[[0.0, 1.0]])
    monkeypatch.setattr("agent.embed.embed_query", lambda *a, **k: [0.0, 1.0])
    hit = service.search("q", limit=1)[0]
    assert isinstance(hit, SearchHit)
    body = hit.to_json()
    for key in ("chunk_id", "document_id", "page_number", "text", "score", "metadata"):
        assert key in body
    assert 0.0 <= hit.score <= 1.0001
    assert hit.exact is True


def test_search_can_be_scoped_to_one_document(monkeypatch):
    service = InMemoryRetrievalService()
    a, b = _Chunk(1, "doc_aaaaaaaaaaaaaaaa"), _Chunk(1, "doc_bbbbbbbbbbbbbbbb")
    service.upsert([a, b], embeddings=[[1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr("agent.embed.embed_query", lambda *a, **k: [1.0, 0.0])
    hits = service.search("q", limit=5, document_id="doc_bbbbbbbbbbbbbbbb")
    assert [h.document_id for h in hits] == ["doc_bbbbbbbbbbbbbbbb"]


def test_upsert_is_idempotent_on_chunk_id(monkeypatch):
    service = InMemoryRetrievalService()
    chunk = _Chunk(1)
    service.upsert([chunk], embeddings=[[1.0, 0.0]])
    service.upsert([chunk], embeddings=[[0.0, 1.0]])
    assert len(service) == 1


def test_an_empty_index_returns_nothing_rather_than_raising():
    assert InMemoryRetrievalService().search("anything") == []


# --------------------------------------------------------------------------
# Integration â€” skipped without a database
# --------------------------------------------------------------------------


@integration
def test_migrations_apply_and_are_idempotent():
    from storage import migrate

    migrate.migrate()
    again = migrate.migrate()
    assert again == [], "a second apply must be a no-op"
    assert all(migrate.status().values())


@integration
def test_pgvector_round_trip():
    """Write chunks with vectors, read the nearest one back."""
    from storage import migrate
    from storage.vectors import PgVectorRetrievalService

    migrate.migrate()
    service = PgVectorRetrievalService(dim=2)
    assert service.available()

    conn = config.connect()
    conn.execute(
        """
        INSERT INTO documents (document_id, source_name, source_sha256,
                               source_bytes, page_count, extractor)
        VALUES ('doc_test0000000001','t.pdf','sha-test',1,1,'pymupdf')
        ON CONFLICT (document_id) DO NOTHING
        """
    )
    conn.commit()

    chunks = [_Chunk(1, "doc_test0000000001"), _Chunk(2, "doc_test0000000001")]
    service.upsert(chunks, embeddings=[[1.0, 0.0], [0.0, 1.0]])
    hits = service.search_vector([1.0, 0.0], limit=1)
    assert hits[0].chunk_id == chunks[0].chunk_id
    assert hits[0].score > 0.9
    assert hits[0].exact is True


@integration
def test_no_api_key_is_ever_written_to_a_row(monkeypatch):
    """Requirement 10, checked against the database rather than the code."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-MUSTNOTBESTORED0123456789")
    from storage import migrate
    from storage.repository import PostgresRepository

    migrate.migrate()
    repo = PostgresRepository(strict=True)
    repo.record_failure(
        failure_id="FAILURE-TEST",
        klass="test",
        trigger="calling with nvapi-MUSTNOTBESTORED0123456789",
        symptom="the key appears in a field",
    )
    cur = config.connect().execute(
        "SELECT trigger FROM failures WHERE failure_id = 'FAILURE-TEST'"
    )
    assert "MUSTNOTBESTORED" not in str(cur.fetchone())
