-- Studio state in PostgreSQL: documents, jobs, and runs for every workspace.
--
-- Compared with the SQLite schema (one file per workspace, see
-- SqliteStudioRepository._migrate) four things change, and each is a decision
-- you should be able to defend:
--
-- 1. One database, many workspaces. Every table carries workspace_id and it is
--    the first column of every primary key, unique constraint, and index. Two
--    workspaces may contain the same PDF (same sha256) or the same paper_id;
--    within one workspace they may not. There is no workspaces table: a
--    workspace is still a folder with epilens.toml, so a foreign key would have
--    to be kept in sync with the filesystem.
--
-- 2. Real types. Timestamps are timestamptz (stored as UTC, compared and
--    sorted as instants) instead of ISO strings in TEXT. JSON is jsonb (parsed,
--    indexable, queryable with -> and ->>) instead of TEXT that only Python can
--    read.
--
-- 3. Run results live in the database (studio_runs.result) instead of a JSON
--    file next to the database. This is what makes run history queryable in SQL
--    and lets the API run on several machines that share no disk.
--
-- 4. The database enforces invariants the Python code assumed: CHECK
--    constraints on status values and counters, and foreign keys between
--    documents, jobs, and runs.
--
-- Ids stay text (uuid4().hex, or the caller's document id) so both backends
-- return identical records. The uuid type would save space; it is not worth a
-- second id format.

CREATE TABLE studio_documents (
    workspace_id     text        NOT NULL,
    id               text        NOT NULL,
    original_name    text        NOT NULL,
    stored_name      text        NOT NULL,
    sha256           text        NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size             bigint      NOT NULL CHECK (size >= 0),
    paper_id         text        NOT NULL,
    status           text        NOT NULL,
    title            text,
    section_count    integer,
    reference_count  integer,
    last_job_id      text,
    error            text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, id),
    -- Deduplicates uploads: find_document_by_hash relies on this being unique.
    UNIQUE (workspace_id, sha256),
    UNIQUE (workspace_id, paper_id)
);

CREATE TABLE studio_runs (
    workspace_id  text        NOT NULL,
    id            text        NOT NULL,
    kind          text        NOT NULL,   -- index | classification | precision_miner | explore
    task_key      text,                   -- NULL for index and explore runs
    status        text        NOT NULL,
    summary       text        NOT NULL DEFAULT '',
    request       jsonb       NOT NULL,
    config        jsonb       NOT NULL,
    -- Can be large (one entry per paper with evidence). Postgres moves large
    -- values out of line (TOAST), so list queries that skip this column stay
    -- cheap: never SELECT * when you do not need the result.
    result        jsonb       NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, id)
);

CREATE TABLE studio_jobs (
    workspace_id  text        NOT NULL,
    id            text        NOT NULL,
    kind          text        NOT NULL,
    status        text        NOT NULL CHECK (status IN (
                      'queued', 'running', 'cancel_requested', 'cancelled',
                      'completed', 'completed_with_errors', 'failed', 'interrupted')),
    current       integer     NOT NULL DEFAULT 0 CHECK (current >= 0),
    total         integer     NOT NULL DEFAULT 0 CHECK (total >= 0),
    message       text        NOT NULL DEFAULT '',
    payload       jsonb       NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz,
    error         text,
    result_id     text,
    retry_parent  text,
    PRIMARY KEY (workspace_id, id),
    -- Composite foreign keys: the referenced row must be in the same workspace.
    -- With the default MATCH SIMPLE a NULL result_id or retry_parent skips the
    -- check, which is what we want for jobs that have none.
    FOREIGN KEY (workspace_id, result_id)    REFERENCES studio_runs (workspace_id, id),
    FOREIGN KEY (workspace_id, retry_parent) REFERENCES studio_jobs (workspace_id, id)
);

-- Deliberately no foreign key from studio_documents.last_job_id to studio_jobs:
-- `epilens index` from the CLI records the pseudo-job id 'cli-index', which
-- never exists as a row. A constraint the application cannot honour is worse
-- than none, because it turns a working command into an IntegrityError.

-- list_jobs / list_runs: "newest first in one workspace". The index matches the
-- WHERE and ORDER BY, so Postgres reads the first LIMIT rows and stops.
CREATE INDEX studio_jobs_recent ON studio_jobs (workspace_id, created_at DESC);
CREATE INDEX studio_runs_recent ON studio_runs (workspace_id, created_at DESC);

-- Partial index: interrupt_active_jobs only ever touches the few unfinished
-- jobs, so index just those rows instead of the whole history.
CREATE INDEX studio_jobs_active ON studio_jobs (workspace_id)
    WHERE status IN ('queued', 'running', 'cancel_requested');
