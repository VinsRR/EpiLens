"""Local-first application services used by the Studio API, UI, and CLI.

The module deliberately contains no FastAPI, Streamlit, or Typer imports.  It
keeps browser state thin and makes workspace, corpus, task, job, and run data
durable inside an ordinary EpiLens workspace.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

from filelock import FileLock

from epilens.db import InMemoryAcademicDB
from epilens.rag.embeddings.factory import EmbedderFactory
from epilens.rag.indexing.chunking import (
    FixedSizeChunker,
    NoChunker,
    ParagraphChunker,
    SentenceChunker,
)
from epilens.rag.indexing.indexer import Indexer
from epilens.rag.ingestion.document_loader import DocumentLoaderFactory
from epilens.schemas import PaperMetadata
from epilens.services.studio_repository import (
    _atomic_json,
    json_ready,
    open_studio_repository,
    utc_now,
)
from epilens.services.runtime import (
    EvidenceRerankerKind,
    EpiLensRuntime,
    LlmProvider,
    RetrievalMode,
    RuntimeConfig,
)
from epilens.settings import env
from epilens.vectordb.file import FileDB
from epilens.workspace import (
    WORKSPACE_FILE,
    WorkspaceConfig,
    create_workspace,
    load_workspace,
    state_root,
    workspace_lock_path,
    write_workspace_config,
)
from epilens.workflows.classification.prompting import ClassificationPromptBuilder
from epilens.workflows.precision_miner.prompting import PrecisionMinerPromptBuilder
from epilens.workflows.registry import (
    BUILTIN_CLASSIFIER_KEYS,
    BUILTIN_MINER_KEYS,
    TaskSpec,
    build_classifier_config_from_spec,
    build_miner_config_from_spec,
    classifier_catalog,
    humanize_task_spec_error,
    miner_catalog,
)


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md", ".text"}
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class StudioSettings:
    workspaces_root: Path
    active_workspace_id: Optional[str] = None
    upload_limit_mb: int = 100

    @classmethod
    def from_env(cls) -> "StudioSettings":
        default_root = state_root(Path.home()) / "workspaces"
        root = Path(
            env("EPILENS_WORKSPACES_ROOT", str(default_root)) or str(default_root)
        ).expanduser()
        raw_limit = env("EPILENS_UPLOAD_LIMIT_MB", "100") or "100"
        return cls(
            workspaces_root=root,
            active_workspace_id=env("EPILENS_ACTIVE_WORKSPACE"),
            upload_limit_mb=max(1, int(raw_limit)),
        )


class WorkspaceService:
    """Resolve workspace IDs beneath one managed root."""

    def __init__(self, settings: Optional[StudioSettings] = None) -> None:
        self.settings = settings or StudioSettings.from_env()
        self.root = self.settings.workspaces_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_id(workspace_id: str) -> str:
        if not _WORKSPACE_ID_RE.fullmatch(workspace_id) or workspace_id in {".", ".."}:
            raise ValueError(
                "Workspace ID must start with a letter or digit and contain only "
                "letters, digits, dots, underscores, or hyphens."
            )
        return workspace_id

    def workspace_path(self, workspace_id: str) -> Path:
        workspace_id = self.validate_id(workspace_id)
        candidate = (self.root / workspace_id).resolve()
        if candidate.parent != self.root:
            raise ValueError("Workspace path escapes the managed workspace root.")
        return candidate

    def get(self, workspace_id: str) -> WorkspaceConfig:
        return load_workspace(self.workspace_path(workspace_id))

    def create(self, workspace_id: str, *, name: Optional[str] = None) -> Dict[str, Any]:
        path = self.workspace_path(workspace_id)
        config = create_workspace(path, name=name or workspace_id)
        open_studio_repository(config)
        return self.summary(config)

    def list(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        if not self.root.exists():
            return summaries
        for child in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir() and (child / WORKSPACE_FILE).is_file():
                try:
                    summaries.append(self.summary(load_workspace(child)))
                except Exception:
                    continue
        return summaries

    def summary(self, config: WorkspaceConfig) -> Dict[str, Any]:
        repository = open_studio_repository(config)
        index_dir = config.resolve_path(config.index_dir)
        indexed = (index_dir / "config.json").is_file()
        return {
            "id": config.root.name,
            "name": config.name,
            "path": str(config.root),
            "active": config.root.name == self.settings.active_workspace_id,
            "indexed": indexed,
            "document_count": repository.document_count(),
            "paper_count": len(self._paper_ids(config)),
            "task_count": len(list((config.root / "tasks").glob("*.json"))),
            "settings": self.public_settings(config),
        }

    def _paper_ids(self, config: WorkspaceConfig) -> List[str]:
        try:
            db = InMemoryAcademicDB(str(config.resolve_path(config.metadata_path)))
            return db.list_docs(config.strategy_name)
        except Exception:
            return []

    @staticmethod
    def public_settings(config: WorkspaceConfig) -> Dict[str, Any]:
        data = asdict(config)
        data.pop("root", None)
        return json_ready(data)

    def update_settings(self, workspace_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get(workspace_id)
        indexed = (config.resolve_path(config.index_dir) / "config.json").exists()
        indexing_fields = {
            "loader",
            "chunker",
            "min_chunk_size",
            "chunk_size",
            "chunk_overlap",
            "embed_model",
            "embed_provider",
        }
        runtime_fields = {
            "name",
            "llm_provider",
            "llm_model",
            "llm_temperature",
            "workflow_top_k",
            "retrieval_mode",
            "evidence_reranker_kind",
            "cross_encoder_model",
            "cross_encoder_top_k",
        }
        unknown = set(changes) - indexing_fields - runtime_fields
        if unknown:
            raise ValueError(f"Unsupported workspace setting(s): {', '.join(sorted(unknown))}.")
        if indexed and indexing_fields.intersection(changes):
            raise ValueError(
                "Indexing settings are read-only after the first index. Create a new "
                "workspace or use the CLI for an explicit rebuild."
            )

        normalized = dict(changes)
        if "llm_provider" in normalized and normalized["llm_provider"] not in {
            "gemini", "openai", "openrouter", "anthropic", "ollama"
        }:
            raise ValueError("Unsupported LLM provider.")
        if "retrieval_mode" in normalized and config.index_backend == "file":
            if normalized["retrieval_mode"] != "dense_only":
                raise ValueError("Local file indexes support dense_only retrieval.")
        for key in ("min_chunk_size", "chunk_size", "workflow_top_k", "cross_encoder_top_k"):
            if key in normalized and int(normalized[key]) < 1:
                raise ValueError(f"{key} must be at least 1.")
            if key in normalized:
                normalized[key] = int(normalized[key])
        if "chunk_overlap" in normalized:
            normalized["chunk_overlap"] = int(normalized["chunk_overlap"])
            if normalized["chunk_overlap"] < 0:
                raise ValueError("chunk_overlap cannot be negative.")
        if "llm_temperature" in normalized:
            normalized["llm_temperature"] = float(normalized["llm_temperature"])
            if not 0.0 <= normalized["llm_temperature"] <= 2.0:
                raise ValueError("llm_temperature must be between 0 and 2.")
        resolved_reranker = normalized.get(
            "evidence_reranker_kind", config.evidence_reranker_kind
        )
        resolved_cross_encoder = normalized.get(
            "cross_encoder_model", config.cross_encoder_model
        )
        if resolved_reranker != "none" and not resolved_cross_encoder:
            raise ValueError(
                "cross_encoder_model is required when an evidence reranker is enabled."
            )
        resolved_chunker = normalized.get("chunker", config.chunker)
        resolved_chunk_size = int(normalized.get("chunk_size", config.chunk_size))
        resolved_overlap = int(normalized.get("chunk_overlap", config.chunk_overlap))
        if resolved_chunker == "fixed_size" and resolved_overlap >= resolved_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if "embed_model" in normalized and not str(normalized["embed_model"]).strip():
            raise ValueError("embed_model must not be empty.")
        updated = replace(config, **normalized)
        write_workspace_config(updated)
        return self.summary(updated)


class TaskService:
    def __init__(self, workspace: WorkspaceConfig) -> None:
        self.workspace = workspace
        self.directory = workspace.root / "tasks"
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> Dict[str, List[Dict[str, Any]]]:
        classifiers = [entry for entry in classifier_catalog() if entry["source"] == "builtin"]
        miners = [entry for entry in miner_catalog() if entry["source"] == "builtin"]
        for spec in self.specs():
            entry = {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "source": "workspace",
            }
            (classifiers if spec.kind == "classifier" else miners).append(entry)
        return {"classifiers": classifiers, "miners": miners}

    def specs(self) -> List[TaskSpec]:
        specs: List[TaskSpec] = []
        for path in sorted(self.directory.glob("*.json")):
            specs.append(self.validate(json.loads(path.read_text(encoding="utf-8"))))
        return specs

    def get(self, key: str) -> Optional[TaskSpec]:
        if key in BUILTIN_CLASSIFIER_KEYS or key in BUILTIN_MINER_KEYS:
            return None
        path = self.directory / f"{key}.json"
        if not path.is_file():
            raise KeyError(f"Unknown workspace task {key!r}.")
        return self.validate(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def validate(data: Dict[str, Any]) -> TaskSpec:
        try:
            spec = TaskSpec.model_validate(data)
            metadata = PaperMetadata(title="Validation paper", abstract="Validation abstract")
            if spec.kind == "classifier":
                config = build_classifier_config_from_spec(spec)
                ClassificationPromptBuilder(config).build_initial_prompt(metadata, [])
            else:
                config = build_miner_config_from_spec(spec)
                PrecisionMinerPromptBuilder(config).build_messages(metadata, [])
            return spec
        except Exception as exc:
            message = humanize_task_spec_error(exc)
            raise ValueError(message) from exc

    def save(self, data: Dict[str, Any], *, expected_key: Optional[str] = None) -> TaskSpec:
        spec = self.validate(data)
        if spec.key in BUILTIN_CLASSIFIER_KEYS or spec.key in BUILTIN_MINER_KEYS:
            raise ValueError(f"Task key {spec.key!r} is built in and cannot be overwritten.")
        if expected_key is not None and expected_key != spec.key:
            raise ValueError("A task key cannot be changed while editing; duplicate it instead.")
        _atomic_json(self.directory / f"{spec.key}.json", spec.model_dump())
        return spec


class CorpusService:
    def __init__(
        self,
        workspace: WorkspaceConfig,
        *,
        upload_limit_mb: int = 100,
        lock_timeout: float = 60,
    ) -> None:
        self.workspace = workspace
        self.repository = open_studio_repository(workspace)
        self.upload_limit = upload_limit_mb * 1024 * 1024
        self.lock_timeout = lock_timeout
        self.papers_dir = workspace.resolve_path(workspace.papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, filename: str, content: bytes, *, paper_id: Optional[str] = None) -> Dict[str, Any]:
        clean_name = Path(filename).name
        if not clean_name or clean_name != filename:
            raise ValueError("Uploaded filename must not contain a path.")
        suffix = Path(clean_name).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(
                f"Unsupported document type {suffix or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))}."
            )
        if len(content) > self.upload_limit:
            raise ValueError(f"Document exceeds the {self.upload_limit // (1024 * 1024)} MB limit.")
        if not content:
            raise ValueError("Uploaded document is empty.")
        digest = hashlib.sha256(content).hexdigest()
        existing = self.repository.find_document_by_hash(digest)
        if existing:
            return {**existing, "duplicate": True}

        resolved_paper_id = (paper_id or Path(clean_name).stem).strip()
        if not resolved_paper_id:
            raise ValueError("paper_id must not be empty.")
        if any(item["paper_id"] == resolved_paper_id for item in self.repository.list_documents()):
            raise ValueError(
                f"Paper ID {resolved_paper_id!r} is already assigned to another upload."
            )
        document_id = uuid.uuid4().hex
        stored_name = f"{document_id}{suffix}"
        destination = self.papers_dir / stored_name
        with tempfile.NamedTemporaryFile(dir=self.papers_dir, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return self.repository.create_document(
            {
                "id": document_id,
                "original_name": clean_name,
                "stored_name": stored_name,
                "sha256": digest,
                "size": len(content),
                "paper_id": resolved_paper_id,
                "status": "uploaded",
            }
        )

    def list_documents(self) -> list[Dict[str, Any]]:
        return self.repository.list_documents()

    def import_files(
        self, paths: Iterable[Path], *, paper_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Copy CLI-selected files into managed storage using upload semantics."""
        files = list(paths)
        if paper_id and len(files) != 1:
            raise ValueError("paper_id can only be used when importing a single file.")
        imported: list[Dict[str, Any]] = []
        for path in files:
            imported.append(
                self.upload(
                    path.name,
                    path.read_bytes(),
                    paper_id=paper_id or path.stem,
                )
            )
        return imported

    def list_papers(self) -> list[Dict[str, Any]]:
        lock = FileLock(
            str(workspace_lock_path(self.workspace)), timeout=self.lock_timeout
        )
        with lock:
            db = InMemoryAcademicDB(
                str(self.workspace.resolve_path(self.workspace.metadata_path))
            )
        papers: list[Dict[str, Any]] = []
        documents_by_paper = {item["paper_id"]: item for item in self.list_documents()}
        for paper_id in db.list_docs(self.workspace.strategy_name):
            metadata = db.get_paper_metadata(paper_id, self.workspace.strategy_name)
            sections = db.retrieve(paper_id, "sections", self.workspace.strategy_name) or []
            references = db.retrieve(paper_id, "references", self.workspace.strategy_name) or []
            document = documents_by_paper.get(paper_id)
            papers.append(
                {
                    "paper_id": paper_id,
                    "title": getattr(metadata, "title", paper_id),
                    "metadata": json_ready(metadata),
                    "section_count": len(sections),
                    "reference_count": len(references),
                    "document_id": document["id"] if document else None,
                    "source": document["original_name"] if document else getattr(metadata, "file_path", None),
                    "status": document["status"] if document else "indexed",
                }
            )
        return papers

    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        lock = FileLock(
            str(workspace_lock_path(self.workspace)), timeout=self.lock_timeout
        )
        with lock:
            db = InMemoryAcademicDB(
                str(self.workspace.resolve_path(self.workspace.metadata_path))
            )
        metadata = db.get_paper_metadata(paper_id, self.workspace.strategy_name)
        if metadata is None:
            raise KeyError(f"Unknown paper {paper_id!r}.")
        sections = db.retrieve(paper_id, "sections", self.workspace.strategy_name) or []
        references = db.retrieve(paper_id, "references", self.workspace.strategy_name) or []
        return {
            "paper_id": paper_id,
            "metadata": json_ready(metadata),
            "sections": json_ready(sections),
            "references": json_ready(references),
        }

    def _chunker(self):
        kind = self.workspace.chunker
        if kind == "none":
            return NoChunker()
        if kind == "sentence":
            return SentenceChunker()
        if kind == "paragraph":
            return ParagraphChunker(min_chunk_size=self.workspace.min_chunk_size)
        if kind == "fixed_size":
            return FixedSizeChunker(
                chunk_size=self.workspace.chunk_size,
                chunk_overlap=self.workspace.chunk_overlap,
            )
        raise ValueError(f"Unsupported chunker {kind!r}.")

    def index_documents(
        self,
        document_ids: Iterable[str],
        *,
        replace_existing: bool,
        job_id: str,
        progress: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> Dict[str, Any]:
        documents = [self.repository.get_document(document_id) for document_id in document_ids]
        total = len(documents)
        if not documents:
            raise ValueError("Select at least one uploaded document to index.")

        lock = FileLock(
            str(workspace_lock_path(self.workspace)), timeout=self.lock_timeout
        )
        stage_root = Path(tempfile.mkdtemp(prefix=".epilens-stage-", dir=self.workspace.root))
        stage_index = stage_root / "index"
        stage_metadata = stage_root / "metadata.json"
        live_index = self.workspace.resolve_path(self.workspace.index_dir)
        live_metadata = self.workspace.resolve_path(self.workspace.metadata_path)
        successes: list[Dict[str, Any]] = []
        errors: list[Dict[str, str]] = []
        was_cancelled = False

        try:
            with lock:
                if live_index.exists():
                    shutil.copytree(live_index, stage_index, dirs_exist_ok=True)
                else:
                    stage_index.mkdir(parents=True, exist_ok=True)
                if live_metadata.exists():
                    shutil.copy2(live_metadata, stage_metadata)

                vectordb = FileDB(str(stage_index))
                academic_db = InMemoryAcademicDB(str(stage_metadata))
                embedder = EmbedderFactory.get_embedder(
                    self.workspace.embed_model, provider=self.workspace.embed_provider
                )
                indexer = Indexer(vectordb, embedder=embedder, chunker=self._chunker())
                loader = DocumentLoaderFactory.get_loader(self.workspace.loader)

                for position, document in enumerate(documents, start=1):
                    if cancelled():
                        was_cancelled = True
                        break
                    progress(position - 1, total, f"Parsing {document['original_name']}")
                    self.repository.update_document(
                        document["id"], status="indexing", last_job_id=job_id, error=None
                    )
                    vector_snapshot = (
                        vectordb._embeddings.copy(),
                        copy.deepcopy(vectordb._metadata),
                        vectordb._model,
                        copy.deepcopy(vectordb._chunking_config),
                        set(vectordb._payload_keys),
                        vectordb._dirty,
                    )
                    metadata_snapshot = copy.deepcopy(academic_db._store)
                    try:
                        paper_id = document["paper_id"]
                        already_indexed = paper_id in academic_db.list_docs(
                            self.workspace.strategy_name
                        )
                        if already_indexed and not replace_existing:
                            raise ValueError(
                                f"Paper {paper_id!r} is already indexed; confirm replacement to re-index it."
                            )
                        if already_indexed:
                            vectordb.delete(paper_id)
                            academic_db.delete_doc(paper_id, self.workspace.strategy_name)

                        source_path = self.papers_dir / document["stored_name"]
                        sections, metadata, references = loader.load(source_path)
                        if not metadata.title or metadata.title == source_path.stem:
                            metadata.title = Path(document["original_name"]).stem
                        metadata.file_path = document["original_name"]
                        progress(position - 1, total, f"Embedding {document['original_name']}")
                        indexer.index_paper(sections, metadata, paper_id)
                        if not vectordb.get_points(namespace=paper_id):
                            raise ValueError(
                                "Parsing succeeded but produced no indexable text chunks. "
                                "Inspect the document or choose a different parser/chunker."
                            )
                        academic_db.insert(
                            paper_id, "sections", self.workspace.strategy_name,
                            [section.to_dict() for section in sections],
                        )
                        academic_db.insert(
                            paper_id, "metadata", self.workspace.strategy_name, metadata.to_dict()
                        )
                        academic_db.insert(
                            paper_id, "references", self.workspace.strategy_name,
                            [reference.to_dict() for reference in references],
                        )
                        successes.append(
                            {
                                "document_id": document["id"],
                                "paper_id": paper_id,
                                "title": metadata.title,
                                "section_count": len(sections),
                                "reference_count": len(references),
                            }
                        )
                    except Exception as exc:
                        (
                            vectordb._embeddings,
                            vectordb._metadata,
                            vectordb._model,
                            vectordb._chunking_config,
                            vectordb._payload_keys,
                            vectordb._dirty,
                        ) = vector_snapshot
                        academic_db._store = metadata_snapshot
                        academic_db._flush_backup()
                        errors.append({"document_id": document["id"], "error": str(exc)})
                        self.repository.update_document(
                            document["id"], status="failed", last_job_id=job_id, error=str(exc)
                        )
                    progress(position, total, f"Processed {position} of {total}")

                if successes:
                    vectordb.save()
                    validation = FileDB(str(stage_index), strict=True)
                    if validation._embeddings.size:
                        expected_dim = getattr(embedder, "dim", None)
                        actual_dim = int(validation._embeddings.shape[1])
                        if expected_dim is not None and actual_dim != int(expected_dim):
                            raise ValueError(
                                "Staged index validation failed: embedding dimension "
                                f"{actual_dim} does not match the configured embedder "
                                f"dimension {expected_dim}."
                            )
                    self._commit_stage(stage_index, stage_metadata, live_index, live_metadata)
                    for item in successes:
                        self.repository.update_document(
                            item["document_id"],
                            status="indexed",
                            title=item["title"],
                            section_count=item["section_count"],
                            reference_count=item["reference_count"],
                            last_job_id=job_id,
                            error=None,
                        )
        except Exception as exc:
            for item in successes:
                self.repository.update_document(
                    item["document_id"],
                    status="failed",
                    last_job_id=job_id,
                    error=f"Index commit failed: {exc}",
                )
            raise
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

        return {"items": successes, "errors": errors, "cancelled": was_cancelled}

    def _commit_stage(
        self, stage_index: Path, stage_metadata: Path, live_index: Path, live_metadata: Path
    ) -> None:
        backup = self.workspace.root / ".epilens-backup"
        shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir(parents=True, exist_ok=True)
        old_index = backup / "index"
        old_metadata = backup / "metadata.json"
        moved_index = False
        moved_metadata = False
        try:
            if live_index.exists():
                os.replace(live_index, old_index)
                moved_index = True
            if live_metadata.exists():
                os.replace(live_metadata, old_metadata)
                moved_metadata = True
            live_index.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_index, live_index)
            if stage_metadata.exists():
                os.replace(stage_metadata, live_metadata)
        except Exception:
            shutil.rmtree(live_index, ignore_errors=True)
            if live_metadata.exists():
                live_metadata.unlink()
            if moved_index and old_index.exists():
                os.replace(old_index, live_index)
            if moved_metadata and old_metadata.exists():
                os.replace(old_metadata, live_metadata)
            raise


def runtime_config_for_workspace(workspace: WorkspaceConfig) -> RuntimeConfig:
    use_qdrant = workspace.index_backend == "qdrant"
    use_mongo = workspace.metadata_backend == "mongo"
    return RuntimeConfig(
        strategy_name=workspace.strategy_name,
        mongo_uri=env("MONGO_URI") if use_mongo else None,
        mongo_db_name=workspace.mongo_db_name,
        qdrant_url=workspace.qdrant_url if use_qdrant else None,
        qdrant_collection=workspace.qdrant_collection,
        llm_provider=cast(LlmProvider, workspace.llm_provider),
        llm_model=workspace.llm_model,
        llm_temperature=workspace.llm_temperature,
        workflow_top_k=workspace.workflow_top_k,
        retrieval_mode=cast(
            RetrievalMode, "dense_only" if not use_qdrant else workspace.retrieval_mode
        ),
        evidence_reranker_kind=cast(
            EvidenceRerankerKind, workspace.evidence_reranker_kind
        ),
        cross_encoder_model=workspace.cross_encoder_model,
        cross_encoder_top_k=workspace.cross_encoder_top_k,
        index_dir=workspace.resolve_path(workspace.index_dir),
        metadata_backup=workspace.resolve_path(workspace.metadata_path),
    )


class JobManager:
    """Single-worker, persistent local job runner."""

    def __init__(self, workspaces: WorkspaceService) -> None:
        self.workspaces = workspaces
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="epilens-studio")
        self._lock = threading.Lock()
        for summary in self.workspaces.list():
            open_studio_repository(self.workspaces.get(summary["id"])).interrupt_active_jobs()

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def submit_index(
        self, workspace_id: str, document_ids: list[str], *, replace_existing: bool = False
    ) -> Dict[str, Any]:
        if not document_ids:
            raise ValueError("Select at least one uploaded document to index.")
        workspace = self.workspaces.get(workspace_id)
        repository = open_studio_repository(workspace)
        for document_id in document_ids:
            repository.get_document(document_id)
        payload = {"document_ids": document_ids, "replace_existing": replace_existing}
        return self._submit(workspace_id, "index", payload, total=len(document_ids))

    def submit_workflow(
        self,
        workspace_id: str,
        *,
        kind: str,
        paper_ids: list[str],
        task_key: str,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        if kind not in {"classification", "precision_miner"}:
            raise ValueError("Unknown workflow job kind.")
        if not paper_ids:
            raise ValueError("Select at least one indexed paper.")
        workspace = self.workspaces.get(workspace_id)
        available_papers = {
            paper["paper_id"] for paper in CorpusService(workspace).list_papers()
        }
        missing = [paper_id for paper_id in paper_ids if paper_id not in available_papers]
        if missing:
            raise KeyError(f"Unknown indexed paper(s): {', '.join(missing)}.")
        catalog = TaskService(workspace).list()
        family = "classifiers" if kind == "classification" else "miners"
        if task_key not in {entry["key"] for entry in catalog[family]}:
            raise KeyError(f"Unknown {kind} task {task_key!r}.")
        payload = {"paper_ids": paper_ids, "task_key": task_key, "detailed": detailed}
        return self._submit(workspace_id, kind, payload, total=len(paper_ids))

    def _submit(
        self,
        workspace_id: str,
        kind: str,
        payload: Dict[str, Any],
        *,
        total: int,
        retry_parent: Optional[str] = None,
    ) -> Dict[str, Any]:
        workspace = self.workspaces.get(workspace_id)
        repository = open_studio_repository(workspace)
        job = repository.create_job(kind, payload, total=total, retry_parent=retry_parent)
        self._executor.submit(self._run, workspace_id, job["id"])
        return job

    def _run(self, workspace_id: str, job_id: str) -> None:
        workspace = self.workspaces.get(workspace_id)
        repository = open_studio_repository(workspace)
        existing = repository.get_job(job_id)
        if existing["status"] == "cancelled":
            return
        job = repository.update_job(
            job_id, status="running", started_at=utc_now(), message="Starting"
        )

        def progress(current: int, total: int, message: str) -> None:
            repository.update_job(job_id, current=current, total=total, message=message)

        def cancelled() -> bool:
            return repository.get_job(job_id)["status"] == "cancel_requested"

        try:
            if job["kind"] == "index":
                corpus = CorpusService(workspace, upload_limit_mb=self.workspaces.settings.upload_limit_mb)
                result = corpus.index_documents(
                    job["payload"]["document_ids"],
                    replace_existing=bool(job["payload"].get("replace_existing")),
                    job_id=job_id,
                    progress=progress,
                    cancelled=cancelled,
                )
                status = self._result_status(result)
                summary = f"Indexed {len(result['items'])} document(s)"
                task_key = None
            else:
                result = self._run_workflow(
                    workspace, job_id, job["kind"], job["payload"], progress, cancelled
                )
                status = self._result_status(result)
                summary = f"Completed {len(result['items'])} paper(s)"
                task_key = job["payload"]["task_key"]

            run = repository.create_run(
                kind=job["kind"],
                task_key=task_key,
                status=status,
                summary=summary,
                request=job["payload"],
                config=WorkspaceService.public_settings(workspace),
                result=result,
            )
            repository.update_job(
                job_id,
                status=status,
                current=job["total"] if not result.get("cancelled") else len(result["items"]),
                message=summary,
                finished_at=utc_now(),
                result_id=run["id"],
            )
        except Exception as exc:
            repository.update_job(
                job_id,
                status="failed",
                message="Job failed",
                error=str(exc),
                finished_at=utc_now(),
            )

    @staticmethod
    def _result_status(result: Dict[str, Any]) -> str:
        if result.get("cancelled"):
            return "cancelled"
        if result.get("errors"):
            return "completed_with_errors" if result.get("items") else "failed"
        return "completed"

    def _run_workflow(
        self,
        workspace: WorkspaceConfig,
        job_id: str,
        kind: str,
        payload: Dict[str, Any],
        progress: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> Dict[str, Any]:
        runtime = EpiLensRuntime(runtime_config_for_workspace(workspace))
        snapshot_lock = FileLock(str(workspace_lock_path(workspace)), timeout=60)
        with snapshot_lock:
            retriever = runtime.build_retriever()
            academic_db = runtime.build_db()
        task_service = TaskService(workspace)
        task_key = payload["task_key"]
        spec = task_service.get(task_key)
        items: list[Dict[str, Any]] = []
        errors: list[Dict[str, str]] = []
        paper_ids = payload["paper_ids"]
        was_cancelled = False
        for position, paper_id in enumerate(paper_ids, start=1):
            if cancelled():
                was_cancelled = True
                break
            progress(position - 1, len(paper_ids), f"Processing {paper_id}")
            try:
                if kind == "classification":
                    config = build_classifier_config_from_spec(spec) if spec else None
                    result = runtime.classify(
                        paper_id,
                        classifier_kind=task_key,
                        config=config,
                        detailed=bool(payload.get("detailed", True)),
                        retriever=retriever,
                        generator=None,
                        academic_db=academic_db,
                    )
                else:
                    config = build_miner_config_from_spec(spec) if spec else None
                    result = runtime.precision_mine(
                        paper_id,
                        miner_kind=task_key,
                        config=config,
                        detailed=bool(payload.get("detailed", True)),
                        retriever=retriever,
                        generator=None,
                        academic_db=academic_db,
                    )
                items.append({"paper_id": paper_id, "result": json_ready(result)})
            except Exception as exc:
                errors.append({"paper_id": paper_id, "error": str(exc)})
            progress(position, len(paper_ids), f"Processed {position} of {len(paper_ids)}")
        return {"items": items, "errors": errors, "cancelled": was_cancelled}

    def cancel(self, workspace_id: str, job_id: str) -> Dict[str, Any]:
        repository = open_studio_repository(self.workspaces.get(workspace_id))
        job = repository.get_job(job_id)
        if job["status"] == "queued":
            return repository.update_job(
                job_id, status="cancelled", message="Cancelled", finished_at=utc_now()
            )
        if job["status"] == "running":
            return repository.update_job(
                job_id, status="cancel_requested", message="Cancellation requested"
            )
        return job

    def retry(self, workspace_id: str, job_id: str) -> Dict[str, Any]:
        repository = open_studio_repository(self.workspaces.get(workspace_id))
        job = repository.get_job(job_id)
        if job["status"] not in {"failed", "interrupted", "cancelled", "completed_with_errors"}:
            raise ValueError("Only failed, interrupted, cancelled, or partial jobs can be retried.")
        return self._submit(
            workspace_id,
            job["kind"],
            job["payload"],
            total=job["total"],
            retry_parent=job_id,
        )

    def explore(self, workspace_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        workspace = self.workspaces.get(workspace_id)
        runtime = EpiLensRuntime(runtime_config_for_workspace(workspace))
        snapshot_lock = FileLock(str(workspace_lock_path(workspace)), timeout=60)
        with snapshot_lock:
            retriever = runtime.build_retriever()
        result = runtime.explore(
            request["query"],
            top_k=int(request.get("top_k", 5)),
            similarity_threshold=float(request.get("similarity_threshold", 0.0)),
            generate_answer=bool(request.get("generate_answer", False)),
            filters=request.get("filters") or {},
            retriever=retriever,
        )
        ready = json_ready(result)
        repository = open_studio_repository(workspace)
        run = repository.create_run(
            kind="explore",
            task_key=None,
            status="completed",
            summary=request["query"][:160],
            request=request,
            config=WorkspaceService.public_settings(workspace),
            result=ready,
        )
        return {**ready, "run_id": run["id"]}


def run_export(run: Dict[str, Any], output_format: str) -> tuple[bytes, str, str]:
    if output_format == "json":
        return (
            json.dumps(run["result"], indent=2, ensure_ascii=False).encode("utf-8"),
            "application/json",
            f"{run['id']}.json",
        )
    if output_format != "csv":
        raise ValueError("Export format must be json or csv.")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["paper_id", "status", "result"])
    writer.writeheader()
    result = run["result"]
    if isinstance(result, dict) and "items" in result:
        for item in result.get("items", []):
            writer.writerow(
                {
                    "paper_id": item.get("paper_id", ""),
                    "status": "completed",
                    "result": json.dumps(item.get("result", item), ensure_ascii=False),
                }
            )
        for item in result.get("errors", []):
            writer.writerow(
                {
                    "paper_id": item.get("paper_id", item.get("document_id", "")),
                    "status": "failed",
                    "result": item.get("error", ""),
                }
            )
    else:
        writer.writerow(
            {"paper_id": "", "status": run["status"], "result": json.dumps(result, ensure_ascii=False)}
        )
    return buffer.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{run['id']}.csv"
