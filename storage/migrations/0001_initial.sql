-- 0001_initial — documents, chunks, and the run telemetry the taxonomy needs.
--
-- Apply with:  python -m storage.migrate
--
-- Idempotent throughout: every statement is IF NOT EXISTS or CREATE OR REPLACE,
-- so re-applying is a no-op rather than an error. A migration you are afraid to
-- run twice is one nobody runs at all.
--
-- ---------------------------------------------------------------------------
-- THE 2048-DIMENSION PROBLEM, AND WHY THE INDEX LOOKS ODD
-- ---------------------------------------------------------------------------
-- `nvidia/nemotron-3-embed-1b` returns **2048** dimensions. pgvector stores a
-- `vector` of up to 16,000 dimensions happily, but its ANN indexes — both
-- ivfflat and hnsw — are capped at **2,000**. So a plain
--
--     CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
--
-- on a vector(2048) column fails outright. This is not a tuning detail; it
-- makes the obvious schema unindexable with the chosen model.
--
-- The documented way through is `halfvec`, which hnsw supports to 4,000
-- dimensions. The embedding is therefore stored at full float32 precision for
-- exact search, and indexed through a `halfvec` expression, so:
--
--   * exact search reads the full-precision column and is deterministic;
--   * approximate search uses the half-precision index and is not.
--
-- Both are available, the choice is explicit, and §12 below says which is which.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Documents and chunks
-- ---------------------------------------------------------------------------

-- `document_id` is content-addressed (SHA-256 of the file's bytes), which is
-- why it is the primary key rather than a serial: two copies of one document
-- are one row, and a changed byte is a different row, loudly.
CREATE TABLE IF NOT EXISTS documents (
    document_id     text PRIMARY KEY,
    source_name     text        NOT NULL,
    source_sha256   text        NOT NULL UNIQUE,
    source_bytes    bigint      NOT NULL CHECK (source_bytes >= 0),
    page_count      integer     NOT NULL CHECK (page_count >= 0),
    pages_ok        integer     NOT NULL DEFAULT 0,
    pages_empty     integer     NOT NULL DEFAULT 0,
    pages_unreadable integer    NOT NULL DEFAULT 0,
    extractor       text        NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        text PRIMARY KEY,
    document_id     text        NOT NULL
                        REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number     integer     NOT NULL CHECK (page_number >= 1),
    chunk_ordinal   integer     NOT NULL CHECK (chunk_ordinal >= 0),
    -- Offsets into the page's extracted text. With the document hash and the
    -- extractor name these are enough to go back and re-read the exact span,
    -- which is what makes a citation checkable rather than decorative.
    char_start      integer     NOT NULL CHECK (char_start >= 0),
    char_end        integer     NOT NULL CHECK (char_end >= char_start),
    text            text        NOT NULL,
    embedding       vector(2048),
    embedding_model text,
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    -- A chunk is identified twice: by its readable id, and by its position.
    -- Both are unique, and disagreeing would mean the ingester is broken.
    UNIQUE (document_id, page_number, chunk_ordinal)
);

CREATE INDEX IF NOT EXISTS document_chunks_document_idx
    ON document_chunks (document_id, page_number, chunk_ordinal);

-- Only rows that actually carry a vector. A partial index keeps un-embedded
-- chunks — which is every chunk until the embedder has run — out of the index
-- entirely rather than occupying it as nulls.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE embedding IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Run telemetry
--
-- This half of the schema has nothing to do with vectors and is the half that
-- earns its keep first. FAILURES.md has to say "17 of 200 adversarial runs
-- (8.5 %)", the before/after table aggregates across four chaos levels and two
-- policies, and the trace viewer queries runs and their calls. All of that is
-- a GROUP BY, and doing it by globbing JSON files is how a taxonomy ends up
-- with numbers nobody can reproduce.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chaos_runs (
    chaos_run_id    text PRIMARY KEY,
    name            text        NOT NULL,
    model           text        NOT NULL,
    policy          text        NOT NULL,
    retrieval_mode  text        NOT NULL,
    chaos_rate      double precision NOT NULL CHECK (chaos_rate BETWEEN 0 AND 1),
    chaos_modes     text[]      NOT NULL DEFAULT '{}',
    payload         text,
    seed            bigint      NOT NULL,
    budget          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz,
    aborted         text,
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id          text PRIMARY KEY,
    chaos_run_id    text        REFERENCES chaos_runs(chaos_run_id) ON DELETE CASCADE,
    scenario_id     text,
    line_id         text,
    model           text        NOT NULL,
    policy          text        NOT NULL,
    terminal        text        NOT NULL,
    reason          text,
    steps           integer     NOT NULL DEFAULT 0,
    tool_calls      integer     NOT NULL DEFAULT 0,
    cache_hits      integer     NOT NULL DEFAULT 0,
    tokens_in       integer     NOT NULL DEFAULT 0,
    tokens_out      integer     NOT NULL DEFAULT 0,
    elapsed_s       double precision NOT NULL DEFAULT 0,
    -- Provider failures survived, kept apart from anything the agent did. A
    -- suite quietly absorbing 503s is measuring the endpoint's mood.
    model_retries   integer     NOT NULL DEFAULT 0,
    parse_retries   integer     NOT NULL DEFAULT 0,
    justification_source text,
    opinion         jsonb,
    trace_path      text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_runs_chaos_run_idx ON agent_runs (chaos_run_id);
CREATE INDEX IF NOT EXISTS agent_runs_scenario_idx  ON agent_runs (scenario_id);
CREATE INDEX IF NOT EXISTS agent_runs_terminal_idx  ON agent_runs (terminal);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id    text PRIMARY KEY,
    run_id          text        NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step            integer     NOT NULL DEFAULT 0,
    name            text        NOT NULL,
    -- The idempotency key. Two rows sharing it are the same call repeated,
    -- which is how a retry storm is counted without reading a trace.
    call_key        text        NOT NULL,
    arguments       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    ok              boolean     NOT NULL,
    error           text,
    retryable       boolean     NOT NULL DEFAULT false,
    latency_ms      double precision,
    from_cache      boolean     NOT NULL DEFAULT false,
    -- Which chaos mode fabricated this result, or NULL when it is organic.
    -- Nullable on purpose and never defaulted: an injected failure that could
    -- be mistaken for a real one would make the taxonomy fiction.
    chaos           text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tool_calls_run_idx   ON tool_calls (run_id, step);
CREATE INDEX IF NOT EXISTS tool_calls_name_idx  ON tool_calls (name, ok);
CREATE INDEX IF NOT EXISTS tool_calls_key_idx   ON tool_calls (run_id, call_key);
CREATE INDEX IF NOT EXISTS tool_calls_chaos_idx ON tool_calls (chaos) WHERE chaos IS NOT NULL;

CREATE TABLE IF NOT EXISTS evaluation_results (
    id              bigserial PRIMARY KEY,
    run_id          text        NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    scenario_id     text        NOT NULL,
    synthetic       boolean     NOT NULL,
    tags            text[]      NOT NULL DEFAULT '{}',
    passed          boolean     NOT NULL,
    terminal_ok     boolean     NOT NULL,
    schema_ok       boolean     NOT NULL,
    within_budget   boolean     NOT NULL,
    hsn4_ok         boolean,
    slab_ok         boolean,
    differential_ok boolean,
    reason_ok       boolean,
    -- The domain safety property, in its own column because it is not an
    -- accuracy metric: a run can be right in every field and still assert a
    -- rate that no longer exists.
    asserted_abolished boolean  NOT NULL DEFAULT false,
    chapter_ok      boolean,
    expected        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    actual          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    failure_reason  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, scenario_id)
);

CREATE INDEX IF NOT EXISTS evaluation_results_scenario_idx
    ON evaluation_results (scenario_id, passed);

-- The taxonomy itself. One row per observed failure class, and one row per
-- observation linking it to the run that produced it — so a class can carry a
-- frequency instead of an anecdote.
CREATE TABLE IF NOT EXISTS failures (
    failure_id      text PRIMARY KEY,          -- FAILURE-014
    class           text        NOT NULL,
    trigger         text        NOT NULL,
    symptom         text        NOT NULL,
    root_cause      text,
    fix             text,
    residual_risk   text,
    -- Whether this was predicted in DESIGN.md §6 before any measurement, or
    -- found by testing. The brief asks for that distinction to be marked, and
    -- a column is harder to fudge than prose.
    anticipated     boolean     NOT NULL DEFAULT false,
    owasp           text,
    first_seen_at   timestamptz NOT NULL DEFAULT now(),
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS failure_observations (
    id              bigserial PRIMARY KEY,
    failure_id      text        NOT NULL REFERENCES failures(failure_id) ON DELETE CASCADE,
    run_id          text        NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    tool_call_id    text,
    detail          text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (failure_id, run_id, tool_call_id)
);

CREATE INDEX IF NOT EXISTS failure_observations_failure_idx
    ON failure_observations (failure_id);

-- Frequency per failure class, per condition. This view is the reason the
-- telemetry half of this schema exists: it is the "17 of 200 adversarial runs
-- (8.5 %)" line in FAILURES.md, computed rather than tallied by hand.
CREATE OR REPLACE VIEW failure_frequency AS
SELECT
    f.failure_id,
    f.class,
    c.name              AS condition,
    c.chaos_rate,
    c.policy,
    count(DISTINCT o.run_id)                        AS runs_affected,
    count(DISTINCT r.run_id)                        AS runs_total,
    round(
        count(DISTINCT o.run_id)::numeric
        / NULLIF(count(DISTINCT r.run_id), 0) * 100, 2
    )                                               AS pct
FROM failures f
CROSS JOIN chaos_runs c
LEFT JOIN agent_runs r          ON r.chaos_run_id = c.chaos_run_id
LEFT JOIN failure_observations o ON o.failure_id = f.failure_id AND o.run_id = r.run_id
GROUP BY f.failure_id, f.class, c.name, c.chaos_rate, c.policy;
