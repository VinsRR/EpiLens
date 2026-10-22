from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

from epilens import __version__
from epilens.services import (
    CorpusService,
    EpiLensRuntime,
    JobManager,
    RuntimeConfig,
    TaskService,
    WorkspaceService,
    open_studio_repository,
    run_export,
    runtime_config_for_workspace,
)
from epilens.services.errors import humanize_error
from epilens.workflows.registry import (
    TaskSpec,
    build_classifier_config_from_spec,
    build_miner_config_from_spec,
    classifier_catalog,
    miner_catalog,
)
from epilens.workspace import workspace_lock_path

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _studio_jobs
    _jobs()
    try:
        yield
    finally:
        if _studio_jobs is not None:
            _studio_jobs.shutdown()
            _studio_jobs = None


app = FastAPI(title="EpiLens API", version=__version__, lifespan=_lifespan)


def _path_str(value: Optional[Path]) -> Optional[str]:
    return str(value) if value else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {str(key): _json_ready(item) for key, item in value.model_dump().items()}
    if is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


class BackendConfig(BaseModel):
    strategy_name: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().strategy_name
    )
    mongo_uri: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().mongo_uri
    )
    mongo_db_name: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().mongo_db_name
    )
    qdrant_url: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().qdrant_url
    )
    qdrant_collection: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().qdrant_collection
    )
    llm_provider: Literal["gemini", "openai", "openrouter", "anthropic", "ollama"] = (
        Field(default_factory=lambda: RuntimeConfig.from_settings().llm_provider)
    )
    llm_model: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().llm_model
    )
    llm_temperature: float = 0.0
    workflow_top_k: int = 10
    retrieval_mode: Literal[
        "dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"
    ] = "hybrid"
    evidence_reranker_kind: Literal[
        "none", "global_cross_encoder", "within_label_cross_encoder"
    ] = "none"
    cross_encoder_model: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().cross_encoder_model
    )
    cross_encoder_top_k: Optional[int] = 15
    # Local-first fallback paths, used only when qdrant_url / mongo_uri are
    # unset (see RuntimeConfig). Exposed as str over HTTP; converted back to
    # Path in to_runtime_config().
    index_dir: Optional[str] = Field(
        default_factory=lambda: _path_str(RuntimeConfig.from_settings().index_dir)
    )
    metadata_backup: Optional[str] = Field(
        default_factory=lambda: _path_str(
            RuntimeConfig.from_settings().metadata_backup
        )
    )

    def to_runtime_config(self) -> RuntimeConfig:
        data = self.model_dump()
        defaults = RuntimeConfig.from_settings()
        if data["mongo_uri"] == "***configured***":
            data["mongo_uri"] = defaults.mongo_uri
        if data["qdrant_url"] == "***configured***":
            data["qdrant_url"] = defaults.qdrant_url
        data["index_dir"] = Path(data["index_dir"]) if data["index_dir"] else None
        data["metadata_backup"] = (
            Path(data["metadata_backup"]) if data["metadata_backup"] else None
        )
        return RuntimeConfig(**data)


class ClassificationRequest(BaseModel):
    paper_id: str
    classifier_kind: str = "data_accessibility"
    task: Optional[TaskSpec] = None
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)
    workspace_id: Optional[str] = None


class PrecisionMinerRequest(BaseModel):
    paper_id: str
    miner_kind: str = "find_data_sources"
    task: Optional[TaskSpec] = None
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)
    workspace_id: Optional[str] = None


class ExplorerRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    generate_answer: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)
    config: BackendConfig = Field(default_factory=BackendConfig)
    workspace_id: Optional[str] = None


class WorkspaceCreateRequest(BaseModel):
    id: str
    name: Optional[str] = None


class WorkspaceSettingsRequest(BaseModel):
    changes: Dict[str, Any]


class IndexJobRequest(BaseModel):
    document_ids: list[str]
    replace_existing: bool = False


class WorkflowRunRequest(BaseModel):
    paper_ids: list[str]
    task_key: str
    detailed: bool = True


class StudioExploreRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=100)
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0)
    generate_answer: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)


class StudioResource(BaseModel):
    """Typed public resource while allowing additive, backwards-compatible fields."""

    model_config = ConfigDict(extra="allow")


class WorkspaceResource(StudioResource):
    id: str
    name: str
    path: str
    active: bool
    indexed: bool
    document_count: int
    paper_count: int
    task_count: int
    settings: Dict[str, Any]


class WorkspaceListResource(StudioResource):
    root: str
    active_workspace_id: Optional[str]
    items: list[WorkspaceResource]


class DocumentResource(StudioResource):
    id: str
    original_name: str
    stored_name: str
    sha256: str
    size: int
    paper_id: str
    status: str
    duplicate: bool = False
    title: Optional[str] = None
    section_count: Optional[int] = None
    reference_count: Optional[int] = None
    last_job_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentListResource(StudioResource):
    items: list[DocumentResource]


class PaperResource(StudioResource):
    paper_id: str
    metadata: Dict[str, Any]


class PaperListResource(StudioResource):
    items: list[PaperResource]


class JobResource(StudioResource):
    id: str
    kind: str
    status: str
    current: int
    total: int
    message: str
    payload: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result_id: Optional[str] = None
    retry_parent: Optional[str] = None


class JobListResource(StudioResource):
    items: list[JobResource]


class RunResource(StudioResource):
    id: str
    kind: str
    task_key: Optional[str] = None
    status: str
    summary: str
    request: Dict[str, Any]
    config: Dict[str, Any]
    # None when the backend stores results in the database (PostgreSQL).
    result_path: Optional[str] = None
    created_at: str
    result: Optional[Any] = None


class RunListResource(StudioResource):
    items: list[RunResource]


class TaskCatalogResource(StudioResource):
    classifiers: list[Dict[str, Any]]
    miners: list[Dict[str, Any]]


_studio_workspaces: Optional[WorkspaceService] = None
_studio_jobs: Optional[JobManager] = None


def _workspaces() -> WorkspaceService:
    global _studio_workspaces
    if _studio_workspaces is None:
        _studio_workspaces = WorkspaceService()
    return _studio_workspaces


def _jobs() -> JobManager:
    global _studio_jobs
    if _studio_jobs is None:
        _studio_jobs = JobManager(_workspaces())
    return _studio_jobs


def _studio_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=humanize_error(exc))
    if isinstance(exc, Timeout):
        return HTTPException(
            status_code=423,
            detail="The workspace is busy with another corpus operation. Try again shortly.",
        )
    return HTTPException(status_code=500, detail=humanize_error(exc))


def _workspace_runtime(workspace_id: str) -> tuple[Any, EpiLensRuntime]:
    workspace = _workspaces().get(workspace_id)
    return workspace, EpiLensRuntime(runtime_config_for_workspace(workspace))


def _redact_secrets(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Never expose connection strings (which may carry credentials) over HTTP.

    ``checks.mongo_uri_configured`` already reports presence as a boolean.
    """
    safe = dict(defaults)
    if safe.get("mongo_uri"):
        safe["mongo_uri"] = "***configured***"
    if safe.get("qdrant_url"):
        safe["qdrant_url"] = "***configured***"
    return safe


@app.get("/health")
def health() -> Dict[str, Any]:
    runtime_health = EpiLensRuntime().health()
    return {
        "status": "ok",
        "service": "epilens-api",
        "defaults": _redact_secrets(_json_ready(runtime_health.defaults)),
        "checks": runtime_health.checks,
        "kinds": {
            "classifiers": classifier_catalog(),
            "miners": miner_catalog(),
        },
    }


@app.post("/classify")
def classify(request: ClassificationRequest) -> Dict[str, Any]:
    try:
        workspace = None
        if request.workspace_id:
            workspace, runtime = _workspace_runtime(request.workspace_id)
        else:
            runtime = EpiLensRuntime(request.config.to_runtime_config())
        inline_config = None
        if request.task is not None:
            if request.task.kind != "classifier":
                raise ValueError("task.kind must be 'classifier' for /classify.")
            inline_config = build_classifier_config_from_spec(request.task)
        elif workspace is not None:
            spec = TaskService(workspace).get(request.classifier_kind)
            if spec is not None:
                inline_config = build_classifier_config_from_spec(spec)
        runtime_kwargs: Dict[str, Any] = {}
        if workspace is not None:
            with FileLock(str(workspace_lock_path(workspace)), timeout=60):
                runtime_kwargs["retriever"] = runtime.build_retriever()
                runtime_kwargs["academic_db"] = runtime.build_db()
            runtime_kwargs["generator"] = runtime.build_generator()
        result = runtime.classify(
            request.paper_id,
            classifier_kind=request.classifier_kind,
            config=inline_config,
            detailed=request.detailed,
            **runtime_kwargs,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Classification failed: {humanize_error(exc)}"
        ) from exc


@app.post("/precision-miner")
def precision_miner(request: PrecisionMinerRequest) -> Dict[str, Any]:
    try:
        workspace = None
        if request.workspace_id:
            workspace, runtime = _workspace_runtime(request.workspace_id)
        else:
            runtime = EpiLensRuntime(request.config.to_runtime_config())
        inline_config = None
        if request.task is not None:
            if request.task.kind != "miner":
                raise ValueError("task.kind must be 'miner' for /precision-miner.")
            inline_config = build_miner_config_from_spec(request.task)
        elif workspace is not None:
            spec = TaskService(workspace).get(request.miner_kind)
            if spec is not None:
                inline_config = build_miner_config_from_spec(spec)
        runtime_kwargs: Dict[str, Any] = {}
        if workspace is not None:
            with FileLock(str(workspace_lock_path(workspace)), timeout=60):
                runtime_kwargs["retriever"] = runtime.build_retriever()
                runtime_kwargs["academic_db"] = runtime.build_db()
            runtime_kwargs["generator"] = runtime.build_generator()
        result = runtime.precision_mine(
            request.paper_id,
            miner_kind=request.miner_kind,
            config=inline_config,
            detailed=request.detailed,
            **runtime_kwargs,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Precision miner failed: {humanize_error(exc)}"
        ) from exc


@app.post("/explore")
def explore(request: ExplorerRequest) -> Dict[str, Any]:
    try:
        workspace = None
        if request.workspace_id:
            workspace, runtime = _workspace_runtime(request.workspace_id)
        else:
            runtime = EpiLensRuntime(request.config.to_runtime_config())
        runtime_kwargs: Dict[str, Any] = {}
        if workspace is not None:
            with FileLock(str(workspace_lock_path(workspace)), timeout=60):
                runtime_kwargs["retriever"] = runtime.build_retriever()
        result = runtime.explore(
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            generate_answer=request.generate_answer,
            filters=request.filters,
            **runtime_kwargs,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explorer failed: {humanize_error(exc)}") from exc


# ---------------------------------------------------------------------------
# Workspace-scoped Studio API
# ---------------------------------------------------------------------------


@app.get("/workspaces", response_model=WorkspaceListResource)
def list_workspaces() -> Dict[str, Any]:
    return {
        "root": str(_workspaces().root),
        "active_workspace_id": _workspaces().settings.active_workspace_id,
        "items": _workspaces().list(),
    }


@app.post(
    "/workspaces",
    status_code=status.HTTP_201_CREATED,
    response_model=WorkspaceResource,
)
def create_studio_workspace(request: WorkspaceCreateRequest) -> Dict[str, Any]:
    try:
        return _workspaces().create(request.id, name=request.name)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}", response_model=WorkspaceResource)
def get_studio_workspace(workspace_id: str) -> Dict[str, Any]:
    try:
        return _workspaces().summary(_workspaces().get(workspace_id))
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.patch(
    "/workspaces/{workspace_id}/settings", response_model=WorkspaceResource
)
def update_studio_workspace(
    workspace_id: str, request: WorkspaceSettingsRequest
) -> Dict[str, Any]:
    try:
        return _workspaces().update_settings(workspace_id, request.changes)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get(
    "/workspaces/{workspace_id}/documents", response_model=DocumentListResource
)
def list_documents(workspace_id: str) -> Dict[str, Any]:
    try:
        workspace = _workspaces().get(workspace_id)
        return {"items": CorpusService(workspace).list_documents()}
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentResource,
)
async def upload_document(
    workspace_id: str,
    file: UploadFile = File(...),
    paper_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        workspace = _workspaces().get(workspace_id)
        content = await file.read()
        return CorpusService(
            workspace, upload_limit_mb=_workspaces().settings.upload_limit_mb
        ).upload(file.filename or "", content, paper_id=paper_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/papers", response_model=PaperListResource)
def list_papers(workspace_id: str) -> Dict[str, Any]:
    try:
        return {"items": CorpusService(_workspaces().get(workspace_id)).list_papers()}
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get(
    "/workspaces/{workspace_id}/papers/{paper_id}", response_model=PaperResource
)
def get_paper(workspace_id: str, paper_id: str) -> Dict[str, Any]:
    try:
        return CorpusService(_workspaces().get(workspace_id)).get_paper(paper_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/index-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResource,
)
def create_index_job(workspace_id: str, request: IndexJobRequest) -> Dict[str, Any]:
    try:
        return _jobs().submit_index(
            workspace_id,
            request.document_ids,
            replace_existing=request.replace_existing,
        )
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/tasks", response_model=TaskCatalogResource)
def list_tasks(workspace_id: str) -> Dict[str, Any]:
    try:
        return TaskService(_workspaces().get(workspace_id)).list()
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post("/workspaces/{workspace_id}/tasks/validate")
def validate_task(workspace_id: str, task: TaskSpec) -> Dict[str, Any]:
    try:
        spec = TaskService(_workspaces().get(workspace_id)).validate(task.model_dump())
        return {"valid": True, "task": spec.model_dump()}
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/tasks", status_code=status.HTTP_201_CREATED
)
def create_task(workspace_id: str, task: TaskSpec) -> Dict[str, Any]:
    try:
        service = TaskService(_workspaces().get(workspace_id))
        if (service.directory / f"{task.key}.json").exists():
            raise ValueError(f"Task {task.key!r} already exists; use PUT to edit it.")
        return service.save(task.model_dump()).model_dump()
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.put("/workspaces/{workspace_id}/tasks/{task_key}")
def update_task(workspace_id: str, task_key: str, task: TaskSpec) -> Dict[str, Any]:
    try:
        return TaskService(_workspaces().get(workspace_id)).save(
            task.model_dump(), expected_key=task_key
        ).model_dump()
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/tasks/{task_key}")
def get_workspace_task(workspace_id: str, task_key: str) -> Dict[str, Any]:
    try:
        spec = TaskService(_workspaces().get(workspace_id)).get(task_key)
        if spec is None:
            raise ValueError("Built-in tasks are read-only and do not have an editable spec.")
        return spec.model_dump()
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post("/workspaces/{workspace_id}/runs/explore")
def studio_explore(workspace_id: str, request: StudioExploreRequest) -> Dict[str, Any]:
    try:
        return _jobs().explore(workspace_id, request.model_dump())
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/runs/classification",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResource,
)
def studio_classification(
    workspace_id: str, request: WorkflowRunRequest
) -> Dict[str, Any]:
    try:
        return _jobs().submit_workflow(
            workspace_id,
            kind="classification",
            paper_ids=request.paper_ids,
            task_key=request.task_key,
            detailed=request.detailed,
        )
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/runs/precision-miner",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResource,
)
def studio_precision_miner(
    workspace_id: str, request: WorkflowRunRequest
) -> Dict[str, Any]:
    try:
        return _jobs().submit_workflow(
            workspace_id,
            kind="precision_miner",
            paper_ids=request.paper_ids,
            task_key=request.task_key,
            detailed=request.detailed,
        )
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/jobs", response_model=JobListResource)
def list_jobs(workspace_id: str) -> Dict[str, Any]:
    try:
        repository = open_studio_repository(_workspaces().get(workspace_id))
        return {"items": repository.list_jobs()}
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get(
    "/workspaces/{workspace_id}/jobs/{job_id}", response_model=JobResource
)
def get_job(workspace_id: str, job_id: str) -> Dict[str, Any]:
    try:
        return open_studio_repository(_workspaces().get(workspace_id)).get_job(job_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/jobs/{job_id}/cancel",
    response_model=JobResource,
)
def cancel_job(workspace_id: str, job_id: str) -> Dict[str, Any]:
    try:
        return _jobs().cancel(workspace_id, job_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.post(
    "/workspaces/{workspace_id}/jobs/{job_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResource,
)
def retry_job(workspace_id: str, job_id: str) -> Dict[str, Any]:
    try:
        return _jobs().retry(workspace_id, job_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/runs", response_model=RunListResource)
def list_runs(workspace_id: str) -> Dict[str, Any]:
    try:
        return {"items": open_studio_repository(_workspaces().get(workspace_id)).list_runs()}
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/runs/{run_id}", response_model=RunResource)
def get_run(workspace_id: str, run_id: str) -> Dict[str, Any]:
    try:
        return open_studio_repository(_workspaces().get(workspace_id)).get_run(run_id)
    except Exception as exc:
        raise _studio_http_error(exc) from exc


@app.get("/workspaces/{workspace_id}/runs/{run_id}/download")
def download_run(workspace_id: str, run_id: str, format: str = "json") -> Response:
    try:
        run = open_studio_repository(_workspaces().get(workspace_id)).get_run(run_id)
        content, media_type, filename = run_export(run, format)
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise _studio_http_error(exc) from exc
