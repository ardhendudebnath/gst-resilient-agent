"""Run telemetry: suites, runs, tool calls, scores, and the failure taxonomy.

This is the half of the schema that earns its keep first, and it has nothing to
do with vectors. `FAILURES.md` has to be able to say "17 of 200 adversarial
runs (8.5 %)"; the before/after table aggregates across four chaos levels and
two policies; the trace viewer queries runs and their calls. Every one of those
is a GROUP BY, and computing them by globbing JSON files is how a taxonomy ends
up with numbers nobody can reproduce.

**Recording is optional and never fatal.** `NullRepository` satisfies the same
interface and does nothing, so a suite runs identically with no database. A
telemetry layer that can fail a run is worse than no telemetry layer — the
measurement must not be able to break the thing being measured.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Protocol

from storage import config


class Repository(Protocol):
    def record_suite(self, run: Any) -> str | None: ...
    def record_run(self, chaos_run_id: str | None, score: Any, result: Any) -> str | None: ...
    def record_failure(self, **kwargs: Any) -> None: ...
    def available(self) -> bool: ...


class NullRepository:
    """Records nothing. The default when no database is configured."""

    name = "null"

    def available(self) -> bool:
        return False

    def record_suite(self, run: Any) -> str | None:
        return None

    def record_run(self, chaos_run_id: str | None, score: Any, result: Any) -> str | None:
        return None

    def record_failure(self, **kwargs: Any) -> None:
        return None


def _json(value: Any) -> str:
    # Redacted on the way in, every time. Requirement: never store API keys in
    # the database. Enforced here rather than trusted to call sites, because
    # the realistic mistake is a key reaching a field nobody thought about.
    return json.dumps(config.redact(value), ensure_ascii=False, default=str)


class PostgresRepository:
    """Writes telemetry to Postgres. Never raises into a run."""

    name = "postgres"

    def __init__(self, url: str | None = None, *, strict: bool = False) -> None:
        self._url = url
        self._conn: Any = None
        #: When False (the default), a write failure is swallowed and reported
        #: on `errors` rather than propagated. Tests set strict=True.
        self._strict = strict
        self.errors: list[str] = []

    def _connection(self):
        if self._conn is None:
            self._conn = config.connect(self._url)
        return self._conn

    def available(self) -> bool:
        if not (self._url or config.configured()):
            return False
        try:
            self._connection().execute("SELECT 1")
            return True
        except config.StorageUnavailable:
            return False

    def _guard(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — telemetry must not break a run
            if self._strict:
                raise
            self.errors.append(f"{type(exc).__name__}: {config.redact(str(exc))}")
            try:
                self._connection().rollback()
            except Exception:  # noqa: BLE001
                pass
            return None

    # -- writes ----------------------------------------------------------

    def record_suite(self, run: Any) -> str | None:
        return self._guard(self._record_suite, run)

    def _record_suite(self, run: Any) -> str:
        body = run.to_json()
        chaos = body.get("chaos") or {}
        chaos_run_id = f"{body['name']}-{body['started_at']}"
        conn = self._connection()
        conn.execute(
            """
            INSERT INTO chaos_runs (chaos_run_id, name, model, policy,
                                    retrieval_mode, chaos_rate, chaos_modes,
                                    payload, seed, budget, started_at,
                                    finished_at, aborted, metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (chaos_run_id) DO UPDATE SET
                finished_at = EXCLUDED.finished_at,
                aborted = EXCLUDED.aborted
            """,
            (
                chaos_run_id,
                body["name"],
                body["model"],
                (body.get("policy") or {}).get("name", "baseline"),
                body.get("retrieval_mode", ""),
                float(chaos.get("rate", 0.0)),
                list(chaos.get("modes") or []),
                chaos.get("payload"),
                int(chaos.get("seed", 0)),
                _json(body.get("budget") or {}),
                body["started_at"],
                body.get("finished_at"),
                body.get("aborted"),
                _json({"skipped": body.get("skipped") or []}),
            ),
        )
        conn.commit()
        return chaos_run_id

    def record_run(
        self, chaos_run_id: str | None, score: Any, result: Any
    ) -> str | None:
        return self._guard(self._record_run, chaos_run_id, score, result)

    def _record_run(self, chaos_run_id: str | None, score: Any, result: Any) -> str:
        body = result if isinstance(result, dict) else result.to_json()
        ledger = body.get("ledger") or {}
        run_id = body["run_id"]
        conn = self._connection()

        conn.execute(
            """
            INSERT INTO agent_runs (run_id, chaos_run_id, scenario_id, line_id,
                                    model, policy, terminal, reason, steps,
                                    tool_calls, cache_hits, tokens_in,
                                    tokens_out, elapsed_s, model_retries,
                                    parse_retries, justification_source,
                                    opinion, trace_path)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                run_id,
                chaos_run_id,
                getattr(score, "scenario_id", None),
                body.get("line_id"),
                body.get("model", ""),
                (body.get("policy") or {}).get("name", "baseline"),
                body.get("terminal", ""),
                body.get("reason"),
                int(body.get("steps") or 0),
                int(ledger.get("tool_calls") or 0),
                int(ledger.get("cache_hits") or 0),
                int(ledger.get("tokens_in") or 0),
                int(ledger.get("tokens_out") or 0),
                float(ledger.get("elapsed_s") or 0.0),
                int(body.get("model_retries") or 0),
                int(body.get("parse_retries") or 0),
                body.get("justification_source"),
                _json(body.get("opinion")),
                body.get("trace_path"),
            ),
        )

        if score is not None:
            s = score.to_json() if hasattr(score, "to_json") else score
            conn.execute(
                """
                INSERT INTO evaluation_results (run_id, scenario_id, synthetic,
                    tags, passed, terminal_ok, schema_ok, within_budget,
                    hsn4_ok, slab_ok, differential_ok, reason_ok,
                    asserted_abolished, chapter_ok, expected, actual,
                    failure_reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, scenario_id) DO NOTHING
                """,
                (
                    run_id,
                    s["scenario_id"],
                    bool(s["synthetic"]),
                    list(s.get("tags") or []),
                    bool(s["passed"]),
                    bool(s["terminal_ok"]),
                    bool(s["schema_ok"]),
                    bool(s["within_budget"]),
                    s.get("hsn4_ok"),
                    s.get("slab_ok"),
                    s.get("differential_ok"),
                    s.get("reason_ok"),
                    bool(s.get("asserted_abolished")),
                    s.get("chapter_ok"),
                    _json(s.get("expected") or {}),
                    _json(s.get("actual") or {}),
                    s.get("failure_reason"),
                ),
            )

        conn.commit()
        return run_id

    def record_tool_calls(self, run_id: str, calls: Iterable[dict[str, Any]]) -> None:
        self._guard(self._record_tool_calls, run_id, list(calls))

    def _record_tool_calls(self, run_id: str, calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        conn = self._connection()
        conn.executemany(
            """
            INSERT INTO tool_calls (tool_call_id, run_id, step, name, call_key,
                                    arguments, ok, error, retryable,
                                    latency_ms, from_cache, chaos)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tool_call_id) DO NOTHING
            """,
            [
                (
                    c.get("tool_call_id") or c.get("call_id"),
                    run_id,
                    int(c.get("step") or 0),
                    c.get("name", ""),
                    c.get("key", ""),
                    _json(c.get("arguments") or {}),
                    bool(c.get("ok")),
                    c.get("error"),
                    bool(c.get("retryable")),
                    c.get("latency_ms"),
                    bool(c.get("from_cache")),
                    c.get("chaos"),
                )
                for c in calls
            ],
        )
        conn.commit()

    def record_failure(
        self,
        *,
        failure_id: str,
        klass: str,
        trigger: str,
        symptom: str,
        root_cause: str | None = None,
        fix: str | None = None,
        residual_risk: str | None = None,
        anticipated: bool = False,
        owasp: str | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._guard(
            self._record_failure, failure_id, klass, trigger, symptom,
            root_cause, fix, residual_risk, anticipated, owasp,
            run_id, tool_call_id, detail,
        )

    def _record_failure(
        self, failure_id, klass, trigger, symptom, root_cause, fix,
        residual_risk, anticipated, owasp, run_id, tool_call_id, detail,
    ) -> None:
        conn = self._connection()
        conn.execute(
            """
            INSERT INTO failures (failure_id, class, trigger, symptom,
                                  root_cause, fix, residual_risk, anticipated,
                                  owasp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (failure_id) DO UPDATE SET
                root_cause = COALESCE(EXCLUDED.root_cause, failures.root_cause),
                fix = COALESCE(EXCLUDED.fix, failures.fix),
                residual_risk = COALESCE(EXCLUDED.residual_risk, failures.residual_risk)
            """,
            (failure_id, klass, trigger, symptom, root_cause, fix,
             residual_risk, anticipated, owasp),
        )
        if run_id:
            conn.execute(
                """
                INSERT INTO failure_observations (failure_id, run_id,
                                                  tool_call_id, detail)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (failure_id, run_id, tool_call_id) DO NOTHING
                """,
                (failure_id, run_id, tool_call_id, detail),
            )
        conn.commit()

    # -- reads -----------------------------------------------------------

    def pass_rates(self) -> list[dict[str, Any]]:
        """The before/after table, as a query rather than a script."""
        cur = self._connection().execute(
            """
            SELECT c.name, c.chaos_rate, c.policy,
                   count(*) FILTER (WHERE NOT e.synthetic)                   AS derived_n,
                   count(*) FILTER (WHERE NOT e.synthetic AND e.passed)      AS derived_pass,
                   count(*) FILTER (WHERE e.synthetic)                       AS synthetic_n,
                   count(*) FILTER (WHERE e.synthetic AND e.passed)          AS synthetic_pass,
                   count(*) FILTER (WHERE e.asserted_abolished)              AS stale,
                   count(*) FILTER (WHERE NOT e.within_budget)               AS budget
            FROM chaos_runs c
            JOIN agent_runs r        ON r.chaos_run_id = c.chaos_run_id
            JOIN evaluation_results e ON e.run_id = r.run_id
            GROUP BY c.name, c.chaos_rate, c.policy
            ORDER BY c.policy, c.chaos_rate
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def failure_frequency(self) -> list[dict[str, Any]]:
        cur = self._connection().execute(
            "SELECT * FROM failure_frequency WHERE runs_affected > 0 "
            "ORDER BY failure_id, chaos_rate"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def build_repository(url: str | None = None, *, strict: bool = False) -> Repository:
    """A Postgres repository when one is configured, else a no-op."""
    if url or config.configured():
        repo = PostgresRepository(url, strict=strict)
        if repo.available():
            return repo
    return NullRepository()
