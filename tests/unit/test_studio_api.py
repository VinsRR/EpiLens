from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import epilens.api as api_module  # noqa: E402
from epilens.services.studio import (  # noqa: E402
    StudioSettings,
    WorkspaceService,
)
from epilens.services.studio_repository import open_studio_repository  # noqa: E402
from epilens.schemas import PaperMetadata, StructuredSection  # noqa: E402


class _AcceptanceLoader:
    def load(self, path):
        text = Path(path).read_text(encoding="utf-8")
        return (
            [StructuredSection(title="Methods", content=text, section_type="Methods")],
            PaperMetadata(title=Path(path).stem, abstract="Acceptance abstract"),
            [],
        )


class _AcceptanceEmbedder:
    model_name = "acceptance/embedder"
    dim = 4

    def embed_texts(self, texts):
        return [[1.0, float(index), 0.0, 0.0] for index, _ in enumerate(texts)]


def _wait_for_job(client: TestClient, workspace_id: str, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/workspaces/{workspace_id}/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running", "cancel_requested"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not finish")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    service = WorkspaceService(StudioSettings(workspaces_root=tmp_path / "workspaces"))
    monkeypatch.setattr(api_module, "_studio_workspaces", service)
    monkeypatch.setattr(api_module, "_studio_jobs", None)
    with TestClient(api_module.app) as test_client:
        yield test_client
    if api_module._studio_jobs is not None:
        api_module._studio_jobs.shutdown()
    monkeypatch.setattr(api_module, "_studio_jobs", None)


def test_workspace_document_and_task_api(client) -> None:
    response = client.post("/workspaces", json={"id": "review", "name": "Review"})
    assert response.status_code == 201
    assert client.get("/workspaces").json()["items"][0]["id"] == "review"

    upload = client.post(
        "/workspaces/review/documents",
        files={"file": ("paper.txt", b"Paper contents")},
        data={"paper_id": "paper-1"},
    )
    assert upload.status_code == 201
    assert upload.json()["paper_id"] == "paper-1"

    task = {
        "key": "study_design",
        "kind": "classifier",
        "labels": [{"code": "cohort"}, {"code": "unclear"}],
        "default_label": "unclear",
        "multi_label": False,
    }
    assert client.post("/workspaces/review/tasks/validate", json=task).status_code == 200
    assert client.post("/workspaces/review/tasks", json=task).status_code == 201
    assert client.get("/workspaces/review/tasks/study_design").json()["key"] == "study_design"


def test_workspace_api_rejects_traversal_and_never_exposes_secrets(client, monkeypatch) -> None:
    monkeypatch.setenv("MONGO_URI", "mongodb://user:secret@example.invalid")
    monkeypatch.setenv("QDRANT_URL", "https://token@vectors.example.invalid")
    assert client.post("/workspaces", json={"id": "../escape"}).status_code == 400
    payload = client.get("/health").json()
    assert "secret" not in str(payload)
    assert "token" not in str(payload)
    assert payload["defaults"]["mongo_uri"] == "***configured***"
    assert payload["defaults"]["qdrant_url"] == "***configured***"
    round_trip = api_module.BackendConfig.model_validate(payload["defaults"])
    assert round_trip.to_runtime_config().mongo_uri == "mongodb://user:secret@example.invalid"


def test_jobs_are_persisted_and_can_be_cancelled(client, monkeypatch) -> None:
    client.post("/workspaces", json={"id": "review"})
    upload = client.post(
        "/workspaces/review/documents",
        files={"file": ("paper.txt", b"paper")},
        data={"paper_id": "paper"},
    )
    monkeypatch.setattr(api_module._jobs()._executor, "submit", lambda *_args, **_kwargs: None)
    response = client.post(
        "/workspaces/review/index-jobs",
        json={"document_ids": [upload.json()["id"]], "replace_existing": False},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    listed = client.get("/workspaces/review/jobs").json()["items"]
    assert any(item["id"] == job_id for item in listed)
    assert client.post(f"/workspaces/review/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert client.post(
        "/workspaces/review/index-jobs", json={"document_ids": []}
    ).status_code == 400


def test_studio_openapi_exposes_typed_resource_schemas(client) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    assert "WorkspaceResource" in components
    assert "DocumentResource" in components
    assert "JobResource" in components
    assert "RunResource" in components
    response_schema = schema["paths"]["/workspaces/{workspace_id}/jobs/{job_id}"][
        "get"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/JobResource")


def test_upload_limits_and_retry_contract(client, monkeypatch) -> None:
    client.post("/workspaces", json={"id": "review"})
    unsupported = client.post(
        "/workspaces/review/documents",
        files={"file": ("paper.exe", b"not a paper")},
    )
    assert unsupported.status_code == 400

    workspace = api_module._workspaces().get("review")
    repository = open_studio_repository(workspace)
    interrupted = repository.create_job("index", {"document_ids": []}, total=0)
    repository.interrupt_active_jobs()
    monkeypatch.setattr(api_module._jobs()._executor, "submit", lambda *_args, **_kwargs: None)
    retried = client.post(f"/workspaces/review/jobs/{interrupted['id']}/retry")
    assert retried.status_code == 202
    assert retried.json()["retry_parent"] == interrupted["id"]


def test_complete_local_api_acceptance_survives_restart(tmp_path, monkeypatch) -> None:
    service = WorkspaceService(StudioSettings(workspaces_root=tmp_path / "workspaces"))
    monkeypatch.setattr(api_module, "_studio_workspaces", service)
    monkeypatch.setattr(api_module, "_studio_jobs", None)
    monkeypatch.setattr(
        "epilens.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: _AcceptanceLoader(),
    )
    monkeypatch.setattr(
        "epilens.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: _AcceptanceEmbedder(),
    )

    with TestClient(api_module.app) as first:
        assert first.post("/workspaces", json={"id": "review", "name": "Review"}).status_code == 201
        document_ids = []
        for index in (1, 2):
            response = first.post(
                "/workspaces/review/documents",
                files={"file": (f"paper-{index}.txt", f"evidence {index}".encode())},
                data={"paper_id": f"paper-{index}"},
            )
            assert response.status_code == 201
            document_ids.append(response.json()["id"])
        queued = first.post(
            "/workspaces/review/index-jobs",
            json={"document_ids": document_ids, "replace_existing": False},
        )
        assert queued.status_code == 202
        indexed = _wait_for_job(first, "review", queued.json()["id"])
        assert indexed["status"] == "completed"
        assert len(first.get("/workspaces/review/papers").json()["items"]) == 2

        classifier = {
            "key": "acceptance_classifier",
            "kind": "classifier",
            "labels": [{"code": "yes"}, {"code": "unclear"}],
            "default_label": "unclear",
            "multi_label": False,
        }
        miner = {
            "key": "acceptance_miner",
            "kind": "miner",
            "retrieval_templates": ["Where is the evidence?"],
        }
        assert first.post("/workspaces/review/tasks/validate", json=classifier).status_code == 200
        assert first.post("/workspaces/review/tasks", json=classifier).status_code == 201
        assert first.post("/workspaces/review/tasks", json=miner).status_code == 201

        def classify(_self, paper_id, **_kwargs):
            if paper_id == "paper-2":
                raise RuntimeError("simulated per-paper LLM failure")
            return {"classification": ["yes"], "evidence": ["evidence 1"]}

        monkeypatch.setattr("epilens.services.studio.EpiLensRuntime.classify", classify)
        monkeypatch.setattr(
            "epilens.services.studio.EpiLensRuntime.precision_mine",
            lambda _self, paper_id, **_kwargs: {"items": [{"name": paper_id}]},
        )
        monkeypatch.setattr(
            "epilens.services.studio.EpiLensRuntime.explore",
            lambda _self, query, **_kwargs: {
                "query": query,
                "retrieval_count": 1,
                "retrieved_chunks": [{"paper_id": "paper-1", "text": "evidence"}],
            },
        )
        explored = first.post(
            "/workspaces/review/runs/explore",
            json={"query": "Where is the evidence?", "generate_answer": False},
        )
        assert explored.status_code == 200
        explore_run_id = explored.json()["run_id"]

        classification = first.post(
            "/workspaces/review/runs/classification",
            json={
                "paper_ids": ["paper-1", "paper-2"],
                "task_key": "acceptance_classifier",
                "detailed": True,
            },
        )
        classification_job = _wait_for_job(first, "review", classification.json()["id"])
        assert classification_job["status"] == "completed_with_errors"

        mining = first.post(
            "/workspaces/review/runs/precision-miner",
            json={
                "paper_ids": ["paper-1"],
                "task_key": "acceptance_miner",
                "detailed": True,
            },
        )
        assert _wait_for_job(first, "review", mining.json()["id"])["status"] == "completed"
        json_export = first.get(
            f"/workspaces/review/runs/{explore_run_id}/download?format=json"
        )
        assert json_export.status_code == 200
        assert json.loads(json_export.content)["query"] == "Where is the evidence?"

    monkeypatch.setattr(api_module, "_studio_jobs", None)
    with TestClient(api_module.app) as restarted:
        summary = restarted.get("/workspaces/review").json()
        assert summary["document_count"] == 2
        assert summary["paper_count"] == 2
        assert len(restarted.get("/workspaces/review/runs").json()["items"]) == 4
