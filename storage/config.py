"""Database connection, from the environment, with secrets kept out of rows.

Configuration is read from environment variables and never written anywhere.
The DSN itself is a credential — a Supabase connection string carries the
database password in userinfo — so it is redacted on the way into logs, errors
and metadata, and there is a test asserting that.

    DATABASE_URL=postgresql://user:pass@host:5432/postgres   # preferred
    SUPABASE_DB_URL=...                                      # also accepted

Nothing here is required to run the agent. Storage is an optional backend:
without a URL, `configured()` is False, the repository becomes a no-op recorder
and retrieval falls back to the in-memory index. `make test` still passes on a
fresh clone with no database, which is the same rule the rest of the project
follows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

#: Read in order; the first one set wins.
DSN_VARS = ("DATABASE_URL", "SUPABASE_DB_URL", "POSTGRES_URL")

#: Environment variables whose *values* must never reach a database row, a log
#: line or a results file. Requirement: never store API keys in the database.
#: Enforced by `redact`, which scrubs by value rather than by key name — a
#: caller that copies a key into a field called `note` is exactly the case a
#: key-name blocklist misses.
SECRET_VARS = (
    "NVIDIA_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_ANON_KEY",
    *DSN_VARS,
)

REDACTED = "[redacted]"

#: Anything shaped like a bearer token or a Supabase key, as a backstop for a
#: secret that is not in the environment of the process doing the writing.
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|nvapi-[A-Za-z0-9_\-]{16,}|eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-.]{20,})"
)


def dsn() -> str | None:
    for var in DSN_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def configured() -> bool:
    return dsn() is not None


def safe_dsn(value: str | None = None) -> str:
    """A DSN with its password removed, for logs and error messages.

    Returned even when parsing fails: an unparseable DSN must not be echoed
    back verbatim on the assumption that it contained nothing sensitive.
    """
    value = value or dsn()
    if not value:
        return "(unset)"
    try:
        parts = urlsplit(value)
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}", ":" + REDACTED)
            parts = parts._replace(netloc=netloc)
        return urlunsplit(parts)
    except ValueError:
        return "(unparseable dsn)"


def _secret_values() -> list[str]:
    return [v for v in (os.environ.get(k, "").strip() for k in SECRET_VARS) if len(v) > 8]


def redact(value: Any) -> Any:
    """Scrub secrets out of anything on its way to a row, a log or a file.

    Recurses through dicts and lists. Matches on the *value* of every known
    secret environment variable, not on field names, because the realistic
    mistake is a key landing in a field nobody thought to blocklist.
    """
    secrets = _secret_values()
    if isinstance(value, Mapping):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        out = value
        for secret in secrets:
            if secret in out:
                out = out.replace(secret, REDACTED)
        return _TOKEN_RE.sub(REDACTED, out)
    return value


class StorageUnavailable(RuntimeError):
    """No database is configured, or the driver is not installed."""


@dataclass(frozen=True, slots=True)
class Connection:
    """Thin wrapper so callers never import psycopg directly."""

    raw: Any

    def execute(self, sql: str, params: Any = None) -> Any:
        cur = self.raw.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, rows: Any) -> None:
        cur = self.raw.cursor()
        cur.executemany(sql, rows)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()


def connect(url: str | None = None) -> Connection:
    """Open a connection. Raises `StorageUnavailable` rather than ImportError.

    The driver is imported here rather than at module scope so that importing
    `storage` costs nothing and works with no database installed — the
    repository and the migration runner both need to be importable in order to
    report that they are unavailable.
    """
    url = url or dsn()
    if not url:
        raise StorageUnavailable(
            "no database configured. Set DATABASE_URL (or SUPABASE_DB_URL). "
            "Storage is optional: without it the repository is a no-op and "
            "retrieval uses the in-memory index."
        )
    try:
        import psycopg
    except ImportError as exc:
        raise StorageUnavailable(
            "the Postgres backend needs psycopg: pip install -e '.[storage]'"
        ) from exc
    try:
        return Connection(psycopg.connect(url))
    except Exception as exc:  # noqa: BLE001 — never echo the DSN
        raise StorageUnavailable(
            f"could not connect to {safe_dsn(url)}: {type(exc).__name__}: "
            f"{redact(str(exc))}"
        ) from exc


def iter_migrations() -> Iterator[tuple[str, str]]:
    """`(name, sql)` for each migration, in lexical order."""
    from pathlib import Path

    directory = Path(__file__).resolve().parent / "migrations"
    for path in sorted(directory.glob("*.sql")):
        yield path.name, path.read_text(encoding="utf-8")
