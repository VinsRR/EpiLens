"""Persistence for Studio documents, jobs, and runs.

Studio state sits behind :class:`StudioRepository` so the services never know
which database holds it. :func:`open_studio_repository` picks the backend:

* SQLite (default): one ``outputs/studio.sqlite3`` per workspace, run results as
  JSON files. Needs nothing installed and suits a single local process.
* PostgreSQL: set ``EPILENS_DATABASE_URL=postgresql://...``. One database holds
  every workspace, rows are scoped by ``workspace_id``, and run results are
  stored as ``jsonb`` so they can be queried. Needs ``pip install epilens[postgres]``.

Every backend returns plain dicts with the same keys, timestamps as ISO-8601
strings, and ``payload`` / ``request`` / ``config`` already decoded from JSON.
``tests/integration/studio/test_repository_contract.py`` pins that contract.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from epilens.settings import env
from epilens.workspace import WorkspaceConfig


DATABASE_URL_ENV = "EPILENS_DATABASE_URL"

JOB_STATUSES = {
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "completed_with_errors",
    "failed",
    "interrupted",
}

DOCUMENT_UPDATE_FIELDS = frozenset(
    {"status", "title", "section_count", "reference_count", "last_job_id", "error"}
)
JOB_UPDATE_FIELDS = frozenset(
    {"status", "current", "total", "message", "started_at", "finished_at", "error", "result_id"}
)
ACTIVE_JOB_STATUSES = ("queued", "running", "cancel_requested")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_ready(value.model_dump())
    if is_dataclass(value):
        return {field.name: json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return getattr(value, "name")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(json_ready(value), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


class StudioRepository(ABC):
    """Storage contract for one workspace's Studio documents, jobs, and runs.

    Lookups of an unknown id raise ``KeyError``. ``update_*`` methods silently
    drop keys outside ``DOCUMENT_UPDATE_FIELDS`` / ``JOB_UPDATE_FIELDS`` and
    return the updated record. ``list_*`` methods return newest first.
    """

    # Documents ----------------------------------------------------------
    @abstractmethod
    def document_count(self) -> int: ...

    @abstractmethod
    def create_document(self, record: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def get_document(self, document_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def find_document_by_hash(self, digest: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_documents(self) -> list[Dict[str, Any]]: ...

    @abstractmethod
    def update_document(self, document_id: str, **changes: Any) -> Dict[str, Any]: ...

    # Jobs ---------------------------------------------------------------
    @abstractmethod
    def create_job(
        self, kind: str, payload: Dict[str, Any], *, total: int, retry_parent: Optional[str] = None
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def get_job(self, job_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def list_jobs(self, *, limit: int = 100) -> list[Dict[str, Any]]: ...

    @abstractmethod
    def update_job(self, job_id: str, **changes: Any) -> Dict[str, Any]: ...

    @abstractmethod
    def interrupt_active_jobs(self) -> None:
        """Mark queued/running/cancel_requested jobs as interrupted (process restart)."""

    # Runs ---------------------------------------------------------------
    @abstractmethod
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
        """Store a finished run; returns it without the (possibly large) ``result``."""

    @abstractmethod
    def get_run(self, run_id: str, *, include_result: bool = True) -> Dict[str, Any]: ...

    @abstractmethod
    def list_runs(self, *, limit: int = 100) -> list[Dict[str, Any]]:
        """Run summaries without ``result``."""


class SqliteStudioRepository(StudioRepository):
    """Per-workspace SQLite repository; run results live in JSON files beside it."""

    def __init__(self, workspace: WorkspaceConfig) -> None:
        self.workspace = workspace
        self.path = workspace.resolve_path(workspace.outputs_dir) / "studio.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / "runs").mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        original_name TEXT NOT NULL,
                        stored_name TEXT NOT NULL,
                        sha256 TEXT NOT NULL UNIQUE,
                        size INTEGER NOT NULL,
                        paper_id TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL,
                        title TEXT,
                        section_count INTEGER,
                        reference_count INTEGER,
                        last_job_id TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current INTEGER NOT NULL DEFAULT 0,
                        total INTEGER NOT NULL DEFAULT 0,
                        message TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT,
                        result_id TEXT,
                        retry_parent TEXT
                    );
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        task_key TEXT,
                        status TEXT NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        request_json TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        result_path TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )

    def interrupt_active_jobs(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='interrupted', finished_at=? "
                "WHERE status IN ('queued', 'running', 'cancel_requested')",
                (utc_now(),),
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        if "payload_json" in data:
            data["payload"] = json.loads(str(data.pop("payload_json")))
        if "request_json" in data:
            data["request"] = json.loads(str(data.pop("request_json")))
        if "config_json" in data:
            data["config"] = json.loads(str(data.pop("config_json")))
        return data

    def document_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def create_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, original_name, stored_name, sha256, size, paper_id, status,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"], record["original_name"], record["stored_name"],
                    record["sha256"], record["size"], record["paper_id"],
                    record.get("status", "uploaded"), now, now,
                ),
            )
        return self.get_document(record["id"])

    def get_document(self, document_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown document {document_id!r}.")
        return dict(row)

    def find_document_by_hash(self, digest: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE sha256=?", (digest,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_document(self, document_id: str, **changes: Any) -> Dict[str, Any]:
        values = {key: value for key, value in changes.items() if key in DOCUMENT_UPDATE_FIELDS}
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE documents SET {assignments} WHERE id=?",
                (*values.values(), document_id),
            )
        return self.get_document(document_id)

    def create_job(
        self, kind: str, payload: Dict[str, Any], *, total: int, retry_parent: Optional[str] = None
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                (id, kind, status, current, total, message, payload_json, created_at, retry_parent)
                VALUES (?, ?, 'queued', 0, ?, 'Queued', ?, ?, ?)""",
                (job_id, kind, total, json.dumps(json_ready(payload)), utc_now(), retry_parent),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown job {job_id!r}.")
        return self._row(row) or {}

    def list_jobs(self, *, limit: int = 100) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def update_job(self, job_id: str, **changes: Any) -> Dict[str, Any]:
        values = {key: value for key, value in changes.items() if key in JOB_UPDATE_FIELDS}
        if "status" in values and values["status"] not in JOB_STATUSES:
            raise ValueError(f"Invalid job status {values['status']!r}.")
        if not values:
            return self.get_job(job_id)
        assignments = ", ".join(f"{key}=?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), job_id)
            )
        return self.get_job(job_id)

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
        run_id = uuid.uuid4().hex
        result_path = self.path.parent / "runs" / f"{run_id}.json"
        _atomic_json(result_path, result)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO runs
                (id, kind, task_key, status, summary, request_json, config_json,
                 result_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, kind, task_key, status, summary,
                    json.dumps(json_ready(request)), json.dumps(json_ready(config)),
                    str(result_path), utc_now(),
                ),
            )
        return self.get_run(run_id, include_result=False)

    def get_run(self, run_id: str, *, include_result: bool = True) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run {run_id!r}.")
        data = self._row(row) or {}
        if include_result:
            data["result"] = json.loads(Path(data["result_path"]).read_text(encoding="utf-8"))
        return data

    def list_runs(self, *, limit: int = 100) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) or {} for row in rows]


def open_studio_repository(
    workspace: WorkspaceConfig, *, database_url: Optional[str] = None
) -> StudioRepository:
    """Return the repository for ``workspace``, honouring ``EPILENS_DATABASE_URL``."""
    url = database_url or env(DATABASE_URL_ENV)
    if url and url.startswith(("postgresql://", "postgres://")):
        from epilens.services.studio_postgres import PostgresStudioRepository

        return PostgresStudioRepository(workspace, url)
    if url:
        raise ValueError(
            f"Unsupported {DATABASE_URL_ENV} scheme; expected postgresql://. "
            "Leave it unset to use the per-workspace SQLite file."
        )
    return SqliteStudioRepository(workspace)
