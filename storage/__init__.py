"""Optional Postgres/pgvector backend. The agent never imports this package.

Two halves, and they earn their keep at different times.

**Telemetry** — `chaos_runs`, `agent_runs`, `tool_calls`, `evaluation_results`,
`failures` — is useful now. FAILURES.md has to say "17 of 200 adversarial runs
(8.5 %)", the before/after table aggregates across four chaos levels and two
policies, and the trace viewer queries runs and their calls. Those are GROUP BY
queries, and computing them by globbing JSON files is how a taxonomy ends up
with numbers nobody can reproduce.

**Vectors** — `documents`, `document_chunks` — earn their keep later. The
tariff schedule is 961 entries and belongs in memory; the advance-ruling
archive is thousands of pages and belongs here. The seam is
`agent.retrieval_service.RetrievalService`, so which one is in use is a
deployment decision rather than an architectural one.

    export DATABASE_URL=postgresql://...
    pip install -e '.[storage]'
    python -m storage.migrate

Everything degrades: with no `DATABASE_URL` the repository is a no-op, the
retrieval service falls back to the in-memory index, and the test suite still
passes on a fresh clone with no network.
"""

from storage.config import (
    Connection,
    StorageUnavailable,
    configured,
    connect,
    dsn,
    redact,
    safe_dsn,
)
from storage.repository import NullRepository, PostgresRepository, build_repository

__all__ = [
    "Connection",
    "NullRepository",
    "PostgresRepository",
    "StorageUnavailable",
    "build_repository",
    "configured",
    "connect",
    "dsn",
    "redact",
    "safe_dsn",
]
