"""Behaviour every StudioRepository backend must share.

Each test runs once per backend. SQLite always runs. PostgreSQL runs only when
``EPILENS_TEST_DATABASE_URL`` points at a database the tests may write to, for
example::

    createdb epilens_test
    EPILENS_TEST_DATABASE_URL=postgresql:///epilens_test \\
        pytest tests/integration/studio -q

Rows are scoped to a fresh workspace per test and deleted afterwards, so the
database can be reused between runs.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import pytest

from epilens.services.studio_repository import (
    SqliteStudioRepository,
    StudioRepository,
    open_studio_repository,
)
from epilens.workspace import WorkspaceConfig, create_workspace

POSTGRES_URL = os.environ.get("EPILENS_TEST_DATABASE_URL")

RepositoryFactory = Callable[[], StudioRepository]


def _delete_workspace_rows(database_url: str, workspace_id: str) -> None:
    import psycopg

    with psycopg.connect(database_url) as connection:
        # Children before parents: jobs reference runs and other jobs.
        for table in ("studio_documents", "studio_jobs", "studio_runs"):
            connection.execute(
                f"DELETE FROM {table} WHERE workspace_id = %s", (workspace_id,)
            )


@pytest.fixture(params=["sqlite", "postgres"])
def new_repository(request, tmp_path: Path):
    """Factory: each call makes a repository for a brand-new workspace."""
    backend = request.param
    if backend == "postgres":
        if not POSTGRES_URL:
            pytest.skip("set EPILENS_TEST_DATABASE_URL to run the PostgreSQL contract")
        pytest.importorskip("psycopg")
    created: list[WorkspaceConfig] = []

    def make() -> StudioRepository:
        workspace = create_workspace(tmp_path / f"ws-{uuid.uuid4().hex[:12]}")
        created.append(workspace)
        if backend == "postgres":
            return open_studio_repository(workspace, database_url=POSTGRES_URL)
        return open_studio_repository(workspace)

    yield make
    if backend == "postgres":
        for workspace in created:
            _delete_workspace_rows(POSTGRES_URL or "", workspace.root.name)


@pytest.fixture
def repository(new_repository: RepositoryFactory) -> StudioRepository:
    return new_repository()


def _document(name: str = "paper.pdf", *, paper_id: str | None = None) -> dict:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "id": uuid.uuid4().hex,
        "original_name": name,
        "stored_name": f"{digest[:12]}.pdf",
        "sha256": digest,
        "size": 1234,
        "paper_id": paper_id or Path(name).stem,
    }


def _is_iso_timestamp(value: object) -> bool:
    return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None


# Documents ----------------------------------------------------------------


def test_document_round_trip(repository: StudioRepository) -> None:
    record = _document()
    created = repository.create_document(record)

    assert created["id"] == record["id"]
    assert created["status"] == "uploaded"
    assert created["size"] == 1234
    assert created["title"] is None
    assert _is_iso_timestamp(created["created_at"])
    assert "workspace_id" not in created
    assert repository.get_document(record["id"]) == created
    assert repository.document_count() == 1


def test_find_document_by_hash(repository: StudioRepository) -> None:
    record = _document()
    repository.create_document(record)

    assert repository.find_document_by_hash(record["sha256"])["id"] == record["id"]
    assert repository.find_document_by_hash("0" * 64) is None


def test_list_documents_newest_first(repository: StudioRepository) -> None:
    first = repository.create_document(_document("a.pdf"))
    second = repository.create_document(_document("b.pdf"))

    assert [doc["id"] for doc in repository.list_documents()] == [second["id"], first["id"]]


def test_update_document_keeps_only_allowed_fields(repository: StudioRepository) -> None:
    record = repository.create_document(_document())
    job = repository.create_job("index", {"document_ids": [record["id"]]}, total=1)

    updated = repository.update_document(
        record["id"],
        status="indexed",
        title="A paper",
        section_count=4,
        last_job_id=job["id"],
        sha256="f" * 64,  # not updatable: must be ignored
    )

    assert updated["status"] == "indexed"
    assert updated["title"] == "A paper"
    assert updated["section_count"] == 4
    assert updated["last_job_id"] == job["id"]
    assert updated["sha256"] == record["sha256"]
    assert updated["updated_at"] >= record["updated_at"]


def test_update_document_accepts_a_job_id_that_is_not_a_row(
    repository: StudioRepository,
) -> None:
    # `epilens index` records the pseudo-job id "cli-index".
    record = repository.create_document(_document())
    assert repository.update_document(record["id"], last_job_id="cli-index")["last_job_id"] == (
        "cli-index"
    )


def test_unknown_ids_raise_key_error(repository: StudioRepository) -> None:
    with pytest.raises(KeyError):
        repository.get_document("missing")
    with pytest.raises(KeyError):
        repository.get_job("missing")
    with pytest.raises(KeyError):
        repository.get_run("missing")


# Jobs ---------------------------------------------------------------------


def test_job_round_trip(repository: StudioRepository) -> None:
    payload = {"paper_ids": ["a", "b"], "task_key": "data_accessibility", "detailed": True}
    job = repository.create_job("classification", payload, total=2)

    assert job["status"] == "queued"
    assert job["current"] == 0
    assert job["total"] == 2
    assert job["message"] == "Queued"
    assert job["payload"] == payload
    assert job["retry_parent"] is None
    assert job["started_at"] is None
    assert _is_iso_timestamp(job["created_at"])
    assert repository.get_job(job["id"]) == job


def test_update_job_progress_and_completion(repository: StudioRepository) -> None:
    job = repository.create_job("index", {"document_ids": []}, total=3)

    repository.update_job(job["id"], status="running", started_at="2026-10-22T09:00:00+00:00")
    repository.update_job(job["id"], current=2, message="Processed 2 of 3")
    done = repository.update_job(
        job["id"], status="completed", current=3, finished_at="2026-10-22T09:05:00+00:00"
    )

    assert done["status"] == "completed"
    assert done["current"] == 3
    assert done["message"] == "Processed 2 of 3"
    assert datetime.fromisoformat(done["started_at"]) == datetime.fromisoformat(
        "2026-10-22T09:00:00+00:00"
    )
    assert _is_iso_timestamp(done["finished_at"])


def test_update_job_rejects_unknown_status(repository: StudioRepository) -> None:
    job = repository.create_job("index", {}, total=0)
    with pytest.raises(ValueError, match="Invalid job status"):
        repository.update_job(job["id"], status="exploded")
    assert repository.get_job(job["id"])["status"] == "queued"


def test_update_job_without_allowed_fields_is_a_read(repository: StudioRepository) -> None:
    job = repository.create_job("index", {}, total=0)
    assert repository.update_job(job["id"], kind="explore") == job


def test_retry_parent_links_jobs(repository: StudioRepository) -> None:
    parent = repository.create_job("index", {}, total=1)
    child = repository.create_job("index", {}, total=1, retry_parent=parent["id"])
    assert repository.get_job(child["id"])["retry_parent"] == parent["id"]


def test_list_jobs_newest_first_with_limit(repository: StudioRepository) -> None:
    ids = [repository.create_job("index", {"n": n}, total=0)["id"] for n in range(3)]

    assert [job["id"] for job in repository.list_jobs()] == ids[::-1]
    assert [job["id"] for job in repository.list_jobs(limit=2)] == ids[:0:-1]


def test_interrupt_active_jobs_only_touches_unfinished_jobs(
    repository: StudioRepository,
) -> None:
    queued = repository.create_job("index", {}, total=0)
    running = repository.create_job("index", {}, total=0)
    cancelling = repository.create_job("index", {}, total=0)
    finished = repository.create_job("index", {}, total=0)
    repository.update_job(running["id"], status="running")
    repository.update_job(cancelling["id"], status="cancel_requested")
    repository.update_job(finished["id"], status="completed")

    repository.interrupt_active_jobs()

    for job in (queued, running, cancelling):
        reloaded = repository.get_job(job["id"])
        assert reloaded["status"] == "interrupted"
        assert _is_iso_timestamp(reloaded["finished_at"])
    reloaded = repository.get_job(finished["id"])
    assert reloaded["status"] == "completed"
    assert reloaded["finished_at"] is None


# Runs ---------------------------------------------------------------------


def _create_run(repository: StudioRepository, **overrides) -> dict:
    arguments = {
        "kind": "classification",
        "task_key": "data_accessibility",
        "status": "completed",
        "summary": "Completed 1 paper(s)",
        "request": {"paper_ids": ["a"], "task_key": "data_accessibility"},
        "config": {"llm_provider": "ollama", "top_k": 5},
        "result": {"items": [{"paper_id": "a", "result": {"label": "open"}}], "errors": []},
    }
    arguments.update(overrides)
    return repository.create_run(**arguments)


def test_run_round_trip(repository: StudioRepository) -> None:
    run = _create_run(repository)

    assert "result" not in run
    assert run["kind"] == "classification"
    assert run["task_key"] == "data_accessibility"
    assert run["request"] == {"paper_ids": ["a"], "task_key": "data_accessibility"}
    assert run["config"] == {"llm_provider": "ollama", "top_k": 5}
    assert "result_path" in run
    assert _is_iso_timestamp(run["created_at"])

    full = repository.get_run(run["id"])
    assert full["result"] == {"items": [{"paper_id": "a", "result": {"label": "open"}}], "errors": []}
    assert "result" not in repository.get_run(run["id"], include_result=False)


def test_run_without_task_key(repository: StudioRepository) -> None:
    run = _create_run(repository, kind="explore", task_key=None, result={"answer": "x"})
    assert repository.get_run(run["id"])["task_key"] is None


def test_list_runs_newest_first_without_results(repository: StudioRepository) -> None:
    ids = [_create_run(repository, summary=f"run {n}")["id"] for n in range(3)]

    runs = repository.list_runs()
    assert [run["id"] for run in runs] == ids[::-1]
    assert all("result" not in run for run in runs)
    assert len(repository.list_runs(limit=1)) == 1


def test_job_can_point_at_its_run(repository: StudioRepository) -> None:
    job = repository.create_job("classification", {}, total=1)
    run = _create_run(repository)
    assert repository.update_job(job["id"], result_id=run["id"])["result_id"] == run["id"]


# Workspaces ---------------------------------------------------------------


def test_workspaces_do_not_see_each_other(new_repository: RepositoryFactory) -> None:
    alpha, beta = new_repository(), new_repository()
    record = _document()
    alpha.create_document(record)
    job = alpha.create_job("index", {}, total=0)
    run = _create_run(alpha)

    # Same file uploaded to another workspace is allowed.
    beta.create_document({**record, "id": uuid.uuid4().hex})

    assert beta.document_count() == 1
    assert beta.list_jobs() == []
    assert beta.list_runs() == []
    with pytest.raises(KeyError):
        beta.get_job(job["id"])
    with pytest.raises(KeyError):
        beta.get_run(run["id"])
    beta.interrupt_active_jobs()
    assert alpha.get_job(job["id"])["status"] == "queued"


# Backend selection --------------------------------------------------------


def test_factory_defaults_to_sqlite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EPILENS_DATABASE_URL", raising=False)
    workspace = create_workspace(tmp_path / "ws")
    assert isinstance(open_studio_repository(workspace), SqliteStudioRepository)


def test_factory_rejects_unknown_scheme(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EPILENS_DATABASE_URL", "mysql://localhost/epilens")
    workspace = create_workspace(tmp_path / "ws")
    with pytest.raises(ValueError, match="postgresql://"):
        open_studio_repository(workspace)
