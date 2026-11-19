from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir, load_json


@dataclass
class PathsConfig:
    competition_dir: str
    docs_dir: str
    work_dir: str

    @property
    def index_dir(self) -> str:
        return str(Path(self.work_dir) / "index")

    @property
    def output_dir(self) -> str:
        return str(Path(self.work_dir) / "outputs")

    @property
    def debug_dir(self) -> str:
        return str(Path(self.work_dir) / "debug")


@dataclass
class DocumentsConfig:
    download_if_missing: bool = False
    request_timeout_seconds: int = 30
    user_agent: str = "wattbot-baseline/0.1"


@dataclass
class RetrievalConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200
    enable_dense: bool = True
    dense_model_name: str = "BAAI/bge-base-en-v1.5"
    enable_reranker: bool = True
    reranker_model_name: Optional[str] = "BAAI/bge-reranker-base"
    bm25_top_k: int = 12
    dense_top_k: int = 12
    final_top_k: int = 8
    rrf_k: int = 60
    blank_relevance_threshold: float = 0.01


@dataclass
class GenerationConfig:
    backend: str = "openai_compatible"
    model: str = "gpt-4.1-mini"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class RuntimeConfig:
    seed: int = 42
    log_level: str = "INFO"


@dataclass
class AppConfig:
    paths: PathsConfig
    documents: DocumentsConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig

    def ensure_dirs(self) -> None:
        ensure_dir(self.paths.work_dir)
        ensure_dir(self.paths.docs_dir)
        ensure_dir(self.paths.index_dir)
        ensure_dir(self.paths.output_dir)
        ensure_dir(self.paths.debug_dir)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> AppConfig:
    data = load_json(path)
    cfg = AppConfig(
        paths=PathsConfig(**data["paths"]),
        documents=DocumentsConfig(**data.get("documents", {})),
        retrieval=RetrievalConfig(**data.get("retrieval", {})),
        generation=GenerationConfig(**data.get("generation", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
    )
    cfg.ensure_dirs()
    return cfg
