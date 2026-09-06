"""Apply the migrations.

    python -m storage.migrate            # apply anything not yet applied
    python -m storage.migrate --status   # report, change nothing
    python -m storage.migrate --dry-run  # print the SQL

Migrations are recorded in `schema_migrations` and every one is written to be
idempotent, so applying twice is a no-op rather than an error. A migration you
are afraid to run is one nobody runs.
"""

from __future__ import annotations

import argparse
import sys

from storage import config

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


def applied(conn: config.Connection) -> set[str]:
    conn.execute(LEDGER)
    conn.commit()
    cur = conn.execute("SELECT name FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def migrate(*, dry_run: bool = False) -> list[str]:
    """Apply pending migrations. Returns the names applied."""
    if dry_run:
        for name, sql in config.iter_migrations():
            print(f"-- {name}\n{sql}")
        return []

    conn = config.connect()
    done = applied(conn)
    ran: list[str] = []
    for name, sql in config.iter_migrations():
        if name in done:
            continue
        # One transaction per migration, so a failure leaves the ledger and the
        # schema agreeing with each other rather than half-applied.
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (name) VALUES (%s) "
                "ON CONFLICT (name) DO NOTHING",
                (name,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        ran.append(name)
    return ran


def status() -> dict[str, bool]:
    conn = config.connect()
    done = applied(conn)
    return {name: name in done for name, _ in config.iter_migrations()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="storage.migrate", description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        migrate(dry_run=True)
        return 0

    if not config.configured():
        print(
            "no database configured. Set DATABASE_URL or SUPABASE_DB_URL.\n"
            "Storage is optional: the agent runs without it.",
            file=sys.stderr,
        )
        return 2

    print(f"database: {config.safe_dsn()}")
    try:
        if args.status:
            for name, done in status().items():
                print(f"  [{'x' if done else ' '}] {name}")
            return 0
        ran = migrate()
    except config.StorageUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"applied {len(ran)} migration(s): {', '.join(ran) or 'none pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
