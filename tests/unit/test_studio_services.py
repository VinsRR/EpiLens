from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from typer.testing import CliRunner

from epilens.cli import app
from epilens.schemas import PaperMetadata, StructuredSection
from epilens.services.studio import (
    CorpusService,
    JobManager,
    StudioSettings,
    TaskService,
    WorkspaceService,
)
from epilens.services.studio_repository import open_studio_repository
from epilens.workflows.registry import classifier_catalog


class _FakeEmbedder:
    model_name = "fake/embedder"
    dim = 4

    def embed_texts(self, texts):
        return [[1.0, float(index), 0.0, 0.0] for index, _ in enumerate(texts)]


class _FakeLoader:
    def load(self, path):
        text = Path(path).read_text(encoding="utf-8")
        return (
            [StructuredSection(title="Methods", content=text, section_type="Methods")],
            PaperMetadata(title="A paper", abstract="A sufficiently useful abstract."),
            [],
        )


@pytest.fixture()
def workspace_service(tmp_path) -> WorkspaceService:
    return WorkspaceService(StudioSettings(workspaces_root=tmp_path / "workspaces"))


def test_workspace_managed_root_and_settings_round_trip(workspace_service) -> None:
    created = workspace_service.create("review-one", name="Review One")
    assert created["id"] == "review-one"
    assert workspace_service.list()[0]["name"] == "Review One"

    updated = workspace_service.update_settings(
        "review-one",
        {"llm_provider": "ollama", "llm_model": "qwen", "workflow_top_k": 17},
    )
    assert updated["settings"]["llm_provider"] == "ollama"
    assert updated["settings"]["workflow_top_k"] == 17
    assert workspace_service.get("review-one").llm_model == "qwen"


@pytest.mark.parametrize("workspace_id", ["../escape", "/absolute", "", "a/b"])
def test_workspace_ids_cannot_escape_root(workspace_service, workspace_id) -> None:
    with pytest.raises(ValueError):
        workspace_service.workspace_path(workspace_id)


def test_upload_validation_deduplication_and_paper_id_collision(workspace_service) -> None:
    workspace_service.create("review")
    corpus = CorpusService(workspace_service.get("review"), upload_limit_mb=1)
    first = corpus.upload("paper.txt", b"A useful paper body.", paper_id="paper-1")
    duplicate = corpus.upload("paper-copy.txt", b"A useful paper body.", paper_id="other")
    assert duplicate["id"] == first["id"]
    assert duplicate["duplicate"] is True

    with pytest.raises(ValueError, match="already assigned"):
        corpus.upload("different.txt", b"Different body.", paper_id="paper-1")
    with pytest.raises(ValueError, match="Unsupported"):
        corpus.upload("paper.exe", b"data", paper_id="bad")
    with pytest.raises(ValueError, match="path"):
        corpus.upload("../paper.txt", b"data", paper_id="bad")


def test_task_service_is_workspace_scoped_and_validates_prompts(workspace_service) -> None:
    workspace_service.create("one")
    workspace_service.create("two")
    task = {
        "key": "study_design",
        "kind": "classifier",
        "labels": [{"code": "cohort"}, {"code": "unclear"}],
        "default_label": "unclear",
        "multi_label": False,
    }
    service = TaskService(workspace_service.get("one"))
    service.save(task)
    assert service.get("study_design").key == "study_design"
    assert TaskService(workspace_service.get("two")).list()["classifiers"]
    with pytest.raises(KeyError):
        TaskService(workspace_service.get("two")).get("study_design")

    invalid = {**task, "key": "bad_prompt", "user_prompt_template": "Bad {missing}"}
    with pytest.raises(ValueError, match="unrecognised placeholder"):
        service.validate(invalid)


def test_repository_marks_active_jobs_interrupted_only_when_requested(workspace_service) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    repository = open_studio_repository(workspace)
    job = repository.create_job("index", {"document_ids": []}, total=0)
    assert open_studio_repository(workspace).get_job(job["id"])["status"] == "queued"
    repository.interrupt_active_jobs()
    assert repository.get_job(job["id"])["status"] == "interrupted"


def test_local_indexing_persists_paper_and_rolls_back_on_failure(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    corpus = CorpusService(workspace)
    document = corpus.upload(
        "paper.txt",
        b"This methods paragraph contains enough words to become one useful indexed chunk.",
        paper_id="paper-1",
    )
    monkeypatch.setattr(
        "epilens.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: _FakeLoader(),
    )
    monkeypatch.setattr(
        "epilens.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: _FakeEmbedder(),
    )
    result = corpus.index_documents(
        [document["id"]],
        replace_existing=False,
        job_id="job-one",
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    assert result["errors"] == []
    assert corpus.list_papers()[0]["paper_id"] == "paper-1"
    cli_result = CliRunner().invoke(
        app,
        ["papers", "--workspace", str(workspace.root), "--format", "json"],
    )
    assert cli_result.exit_code == 0
    assert json.loads(cli_result.stdout)["paper_ids"] == ["paper-1"]
    index_metadata = workspace.resolve_path(workspace.index_dir) / "metadata.json"
    before = index_metadata.read_bytes()

    monkeypatch.setattr(
        "epilens.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("embedder failed")),
    )
    with pytest.raises(RuntimeError, match="embedder failed"):
        corpus.index_documents(
            [document["id"]],
            replace_existing=True,
            job_id="job-two",
            progress=lambda *_: None,
            cancelled=lambda: False,
        )
    assert index_metadata.read_bytes() == before


def test_cli_workspace_index_uses_managed_corpus_service(
    workspace_service, monkeypatch, tmp_path
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    paper = tmp_path / "managed.txt"
    paper.write_text("managed corpus body", encoding="utf-8")
    _install_fake_indexing(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["index", str(paper), "--workspace", str(workspace.root), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["papers"][0]["paper_id"] == "managed"
    documents = open_studio_repository(workspace).list_documents()
    assert documents[0]["original_name"] == "managed.txt"
    assert documents[0]["status"] == "indexed"


def test_cli_task_listing_does_not_mutate_global_registry(
    workspace_service
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    TaskService(workspace).save(
        {
            "key": "isolated_cli_task",
            "kind": "classifier",
            "labels": [{"code": "yes"}, {"code": "no"}],
            "default_label": "no",
        }
    )
    before = {entry["key"] for entry in classifier_catalog()}
    result = CliRunner().invoke(app, ["tasks", "--workspace", str(workspace.root)])
    assert result.exit_code == 0, result.output
    after = {entry["key"] for entry in classifier_catalog()}
    assert before == after
    assert "isolated_cli_task" not in after


def _install_fake_indexing(monkeypatch) -> None:
    monkeypatch.setattr(
        "epilens.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: _FakeLoader(),
    )
    monkeypatch.setattr(
        "epilens.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: _FakeEmbedder(),
    )


def test_workspace_lock_contention_is_reported(workspace_service) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    held = FileLock(str(workspace.root / ".epilens.lock"))
    with held:
        with pytest.raises(Timeout):
            CorpusService(workspace, lock_timeout=0.01).list_papers()


def test_indexing_keeps_successes_when_one_document_fails(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    corpus = CorpusService(workspace)
    good = corpus.upload("good.txt", b"good content", paper_id="good")
    bad = corpus.upload("bad.txt", b"bad content", paper_id="bad")

    class PartialLoader(_FakeLoader):
        def load(self, path):
            if "bad content" in Path(path).read_text(encoding="utf-8"):
                raise RuntimeError("parser rejected document")
            return super().load(path)

    monkeypatch.setattr(
        "epilens.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: PartialLoader(),
    )
    monkeypatch.setattr(
        "epilens.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: _FakeEmbedder(),
    )
    result = corpus.index_documents(
        [good["id"], bad["id"]],
        replace_existing=False,
        job_id="partial",
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    assert [item["paper_id"] for item in result["items"]] == ["good"]
    assert result["errors"][0]["document_id"] == bad["id"]
    assert corpus.repository.get_document(good["id"])["status"] == "indexed"
    assert corpus.repository.get_document(bad["id"])["status"] == "failed"


def test_indexing_cancellation_takes_effect_between_documents(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    corpus = CorpusService(workspace)
    documents = [
        corpus.upload(f"paper-{index}.txt", f"paper {index}".encode(), paper_id=f"p{index}")
        for index in range(2)
    ]
    _install_fake_indexing(monkeypatch)
    parsed = 0

    class CountingLoader(_FakeLoader):
        def load(self, path):
            nonlocal parsed
            parsed += 1
            return super().load(path)

    monkeypatch.setattr(
        "epilens.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: CountingLoader(),
    )
    result = corpus.index_documents(
        [item["id"] for item in documents],
        replace_existing=False,
        job_id="cancel",
        progress=lambda *_: None,
        cancelled=lambda: parsed == 1,
    )
    assert result["cancelled"] is True
    assert parsed == 1
    assert len(result["items"]) == 1


def test_commit_failure_restores_previous_searchable_snapshot(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    corpus = CorpusService(workspace)
    first = corpus.upload("first.txt", b"first body", paper_id="first")
    _install_fake_indexing(monkeypatch)
    corpus.index_documents(
        [first["id"]],
        replace_existing=False,
        job_id="first-job",
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    live_index = workspace.resolve_path(workspace.index_dir)
    live_metadata = workspace.resolve_path(workspace.metadata_path)
    old_index_metadata = (live_index / "metadata.json").read_bytes()
    old_academic_metadata = live_metadata.read_bytes()
    second = corpus.upload("second.txt", b"second body", paper_id="second")
    real_replace = os.replace

    def fail_metadata_commit(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name == "metadata.json"
            and source_path.parent.name.startswith(".epilens-stage-")
            and destination_path == live_metadata
        ):
            raise OSError("simulated commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr("epilens.services.studio.os.replace", fail_metadata_commit)
    with pytest.raises(OSError, match="simulated commit failure"):
        corpus.index_documents(
            [second["id"]],
            replace_existing=False,
            job_id="second-job",
            progress=lambda *_: None,
            cancelled=lambda: False,
        )
    assert (live_index / "metadata.json").read_bytes() == old_index_metadata
    assert live_metadata.read_bytes() == old_academic_metadata
    assert [paper["paper_id"] for paper in corpus.list_papers()] == ["first"]


def test_job_manager_marks_restart_jobs_interrupted_and_retries(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    repository = open_studio_repository(workspace)
    original = repository.create_job("index", {"document_ids": []}, total=0)
    manager = JobManager(workspace_service)
    try:
        assert repository.get_job(original["id"])["status"] == "interrupted"
        monkeypatch.setattr(manager._executor, "submit", lambda *_args, **_kwargs: None)
        retried = manager.retry("review", original["id"])
        assert retried["status"] == "queued"
        assert retried["retry_parent"] == original["id"]
    finally:
        manager.shutdown()


def test_job_manager_persists_partial_batch_results(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    manager = JobManager(workspace_service)
    monkeypatch.setattr(
        "epilens.services.studio.CorpusService.list_papers",
        lambda _self: [{"paper_id": "good"}, {"paper_id": "bad"}],
    )
    monkeypatch.setattr(
        manager,
        "_run_workflow",
        lambda *_args, **_kwargs: {
            "items": [{"paper_id": "good", "result": {"classification": ["yes"]}}],
            "errors": [{"paper_id": "bad", "error": "LLM failed"}],
            "cancelled": False,
        },
    )
    try:
        job = manager.submit_workflow(
            "review",
            kind="classification",
            paper_ids=["good", "bad"],
            task_key="data_accessibility",
        )
        repository = open_studio_repository(workspace)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = repository.get_job(job["id"])
            if current["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert current["status"] == "completed_with_errors"
        run = repository.get_run(current["result_id"])
        assert run["result"]["errors"][0]["paper_id"] == "bad"
    finally:
        manager.shutdown()
