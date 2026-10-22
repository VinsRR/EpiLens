"""PostgreSQL backend for Studio state.

Selected by ``EPILENS_DATABASE_URL=postgresql://...`` (see
:func:`epilens.services.studio_repository.open_studio_repository`). The schema
lives in ``migrations/postgres/*.sql``; read ``0001_studio_initial.sql`` first.

Conventions every method follows:

* Parameters go through ``%s`` placeholders, never string formatting. Column
  names that vary (``update_*``) go through :mod:`psycopg.sql` identifiers.
* Every statement filters on ``workspace_id = %s``: one database holds every
  workspace, and a query without that filter leaks rows across workspaces.
* jsonb comes back from psycopg already decoded; writing it needs the
  :class:`~psycopg.types.json.Jsonb` wrapper around ``json_ready(value)``.
* ``RETURNING`` gives back the written row in the same round trip, so writes do
  not need a follow-up ``SELECT`` (the SQLite backend does one).
* Rows go through :meth:`PostgresStudioRepository._row` before leaving the
  class, so callers get the same dicts the SQLite backend returns.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - exercised only without the extra
    psycopg = None  # type: ignore

from epilens.services.studio_repository import (
    DOCUMENT_UPDATE_FIELDS,
    JOB_STATUSES,
    JOB_UPDATE_FIELDS,
    StudioRepository,
    json_ready,
)
from epilens.workspace import WorkspaceConfig


MIGRATIONS_DIR = Path(__file__).with_name("migrations") / "postgres"
# Any fixed 64-bit number works; every EpiLens process must use the same one.
_MIGRATION_LOCK_ID = 7_240_118_001
# Everything a run record has except the (possibly large) result document.
_RUN_SUMMARY_COLUMNS = (
    "id", "kind", "task_key", "status", "summary", "request", "config", "created_at",
)
_migrated_urls: set[str] = set()
_migrated_guard = threading.Lock()


def _require_psycopg() -> None:
    if psycopg is None:
        raise ImportError(
            "PostgreSQL support needs psycopg: pip install 'epilens[postgres]'."
        )


def migrate(database_url: str) -> list[str]:
    """Apply pending ``migrations/postgres/NNNN_*.sql`` files; return their names.

    Safe to call from several processes at once: the transaction takes a
    Postgres advisory lock first, so a second API replica waits for the first
    to finish migrating and then finds nothing left to do. Each file runs in
    the same transaction as its bookkeeping row, so a failing migration leaves
    neither half-applied tables nor a version marked as done.
    """
    _require_psycopg()
    applied: list[str] = []
    with psycopg.connect(database_url) as connection, connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        connection.execute(
            """CREATE TABLE IF NOT EXISTS epilens_schema_migrations (
                   version    integer     PRIMARY KEY,
                   name       text        NOT NULL,
                   applied_at timestamptz NOT NULL DEFAULT now()
               )"""
        )
        done = {
            row[0]
            for row in connection.execute("SELECT version FROM epilens_schema_migrations")
        }
        for path in sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql")):
            version = int(path.name[:4])
            if version in done:
                continue
            # No parameters, so psycopg sends the file as one multi-statement
            # query. Bytes, because psycopg only types literal strings as queries.
            connection.execute(path.read_bytes())
            connection.execute(
                "INSERT INTO epilens_schema_migrations (version, name) VALUES (%s, %s)",
                (version, path.name),
            )
            applied.append(path.name)
    return applied


def ensure_migrated(database_url: str) -> None:
    """Run :func:`migrate` once per URL per process (repositories are short-lived)."""
    with _migrated_guard:
        if database_url not in _migrated_urls:
            migrate(database_url)
            _migrated_urls.add(database_url)


class PostgresStudioRepository(StudioRepository):
    """Studio documents, jobs, and runs for one workspace in a shared database."""

    def __init__(self, workspace: WorkspaceConfig, database_url: str) -> None:
        _require_psycopg()
        self.workspace = workspace
        self.workspace_id = workspace.root.name
        self.database_url = database_url
        ensure_migrated(database_url)

    def _connect(self) -> "psycopg.Connection[Dict[str, Any]]":
        # One connection per call, like the SQLite backend. Opening a Postgres
        # connection costs a few milliseconds; if that ever shows up in API
        # latency, the fix is a psycopg_pool.ConnectionPool shared per URL.
        # Used as a context manager it commits on success and rolls back on error.
        return psycopg.connect(self.database_url, row_factory=dict_row)  # type: ignore

    @staticmethod
    def _row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Turn a database row into the backend-neutral record callers expect."""
        if row is None:
            return None
        data = {key: value for key, value in row.items() if key != "workspace_id"}
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @staticmethod
    def _assignments(values: Iterable[str]) -> "list[sql.Composable]":
        """``column = %s`` fragments with each column quoted as an identifier."""
        return [sql.SQL("{} = %s").format(sql.Identifier(column)) for column in values]

    @staticmethod
    def _run_columns(include_result: bool) -> "sql.Composable":
        columns = _RUN_SUMMARY_COLUMNS + (("result",) if include_result else ())
        return sql.SQL(", ").join(sql.Identifier(column) for column in columns)

    def _run_record(self, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # The API model carries result_path; this backend has no result file.
        return {**(self._row(row) or {}), "result_path": None}

    # Documents ----------------------------------------------------------

    def document_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT count(*) AS n FROM studio_documents WHERE workspace_id = %s",
                (self.workspace_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def create_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # created_at / updated_at come from the column defaults (now()).
        with self._connect() as connection:
            row = connection.execute(
                """INSERT INTO studio_documents
                   (workspace_id, id, original_name, stored_name, sha256, size, paper_id, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (
                    self.workspace_id, record["id"], record["original_name"],
                    record["stored_name"], record["sha256"], record["size"],
                    record["paper_id"], record.get("status", "uploaded"),
                ),
            ).fetchone()
        return self._row(row) or {}

    def get_document(self, document_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studio_documents WHERE workspace_id = %s AND id = %s",
                (self.workspace_id, document_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown document {document_id!r}.")
        return self._row(row) or {}

    def find_document_by_hash(self, digest: str) -> Optional[Dict[str, Any]]:
        # Served by the UNIQUE (workspace_id, sha256) index.
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studio_documents WHERE workspace_id = %s AND sha256 = %s",
                (self.workspace_id, digest),
            ).fetchone()
        return self._row(row)

    def list_documents(self) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM studio_documents
                   WHERE workspace_id = %s
                   ORDER BY created_at DESC""",
                (self.workspace_id,),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def update_document(self, document_id: str, **changes: Any) -> Dict[str, Any]:
        values = {key: value for key, value in changes.items() if key in DOCUMENT_UPDATE_FIELDS}
        # updated_at is always bumped, so the SET list is never empty.
        assignments = self._assignments(values) + [sql.SQL("updated_at = now()")]
        query = sql.SQL(
            "UPDATE studio_documents SET {} WHERE workspace_id = %s AND id = %s RETURNING *"
        ).format(sql.SQL(", ").join(assignments))
        with self._connect() as connection:
            row = connection.execute(
                query, (*values.values(), self.workspace_id, document_id)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown document {document_id!r}.")
        return self._row(row) or {}

    # Jobs ---------------------------------------------------------------

    def create_job(
        self, kind: str, payload: Dict[str, Any], *, total: int, retry_parent: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """INSERT INTO studio_jobs
                   (workspace_id, id, kind, status, current, total, message, payload, retry_parent)
                   VALUES (%s, %s, %s, 'queued', 0, %s, 'Queued', %s, %s)
                   RETURNING *""",
                (
                    self.workspace_id, uuid.uuid4().hex, kind, total,
                    Jsonb(json_ready(payload)), retry_parent,
                ),
            ).fetchone()
        return self._row(row) or {}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE workspace_id = %s AND id = %s",
                (self.workspace_id, job_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job {job_id!r}.")
        return self._row(row) or {}

    def list_jobs(self, *, limit: int = 100) -> list[Dict[str, Any]]:
        # Matches the studio_jobs_recent index: read the first `limit` entries, no sort.
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM studio_jobs
                   WHERE workspace_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (self.workspace_id, limit),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def update_job(self, job_id: str, **changes: Any) -> Dict[str, Any]:
        values = {key: value for key, value in changes.items() if key in JOB_UPDATE_FIELDS}
        # Checked here as well as by the CHECK constraint, for a readable error.
        if "status" in values and values["status"] not in JOB_STATUSES:
            raise ValueError(f"Invalid job status {values['status']!r}.")
        if not values:
            return self.get_job(job_id)
        # started_at / finished_at arrive as ISO strings; Postgres casts them.
        query = sql.SQL(
            "UPDATE studio_jobs SET {} WHERE workspace_id = %s AND id = %s RETURNING *"
        ).format(sql.SQL(", ").join(self._assignments(values)))
        with self._connect() as connection:
            row = connection.execute(query, (*values.values(), self.workspace_id, job_id)).fetchone()
        if row is None:
            raise KeyError(f"Unknown job {job_id!r}.")
        return self._row(row) or {}

    def interrupt_active_jobs(self) -> None:
        # Keep the literal IN list: the planner uses the studio_jobs_active
        # partial index only if it can prove this WHERE implies the index
        # predicate. A constant list always allows that; a bound parameter
        # (status = ANY(%s)) does not once Postgres switches to a generic plan.
        with self._connect() as connection:
            connection.execute(
                """UPDATE studio_jobs
                   SET status = 'interrupted', finished_at = now()
                   WHERE workspace_id = %s
                     AND status IN ('queued', 'running', 'cancel_requested')""",
                (self.workspace_id,),
            )

    # Runs ---------------------------------------------------------------

    def create_run(
        self,
        *,
        kind: str,
        task_key: Optional[str],
        status: str,
        summary: str,
        request: Dict[str, Any],
        config: Dict[str, Any],
        result: Any,
    ) -> Dict[str, Any]:
        query = sql.SQL(
            """INSERT INTO studio_runs
               (workspace_id, id, kind, task_key, status, summary, request, config, result)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING {}"""
        ).format(self._run_columns(include_result=False))
        with self._connect() as connection:
            row = connection.execute(
                query,
                (
                    self.workspace_id, uuid.uuid4().hex, kind, task_key, status, summary,
                    Jsonb(json_ready(request)), Jsonb(json_ready(config)),
                    Jsonb(json_ready(result)),
                ),
            ).fetchone()
        return self._run_record(row)

    def get_run(self, run_id: str, *, include_result: bool = True) -> Dict[str, Any]:
        query = sql.SQL(
            "SELECT {} FROM studio_runs WHERE workspace_id = %s AND id = %s"
        ).format(self._run_columns(include_result))
        with self._connect() as connection:
            row = connection.execute(query, (self.workspace_id, run_id)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run {run_id!r}.")
        return self._run_record(row)

    def list_runs(self, *, limit: int = 100) -> list[Dict[str, Any]]:
        query = sql.SQL(
            """SELECT {} FROM studio_runs
               WHERE workspace_id = %s
               ORDER BY created_at DESC
               LIMIT %s"""
        ).format(self._run_columns(include_result=False))
        with self._connect() as connection:
            rows = connection.execute(query, (self.workspace_id, limit)).fetchall()
        return [self._run_record(row) for row in rows]
