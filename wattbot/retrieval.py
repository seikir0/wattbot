from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from .config import RetrievalConfig
from .schema import Chunk, RetrievalHit
from .utils import keyword_tokens, overlap_score, tokenize, unique_keep_order


LOGGER = logging.getLogger("wattbot.retrieval")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency behavior
    CrossEncoder = None
    SentenceTransformer = None


def build_query_variants(question: str) -> List[str]:
    variants = [question.strip()]
    
    # 关键词查询
    keyword_query = " ".join(keyword_tokens(question))
    if keyword_query and keyword_query not in variants:
        variants.append(keyword_query)
    
    # 去掉问号
    if "?" in question:
        bare = question.rstrip("?").strip()
        if bare and bare not in variants:
            variants.append(bare)
    
    # True/False 题提取核心陈述
    lowered = question.lower().strip()
    if lowered.startswith("true or false"):
        if ":" in question:
            stripped = question.split(":", 1)[-1].strip()
        else:
            stripped = re.sub(r"(?i)^true or false\s*", "", question).strip()
        if stripped and stripped not in variants:
            variants.append(stripped)
    
    # 提取数字相关关键词（年份、模型名等）
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
    capitalized = re.findall(r'\b[A-Z][A-Za-z0-9\-]+\b', question)
    if numbers or capitalized:
        extra = " ".join(capitalized + numbers)
        if extra.strip() and extra.strip() not in variants:
            variants.append(extra.strip())
    # 提取 "Author et al. (year)" 格式
    author_year = re.findall(r'[A-Z][a-z]+ et al\.?\s*\(?(\d{4})\)?', question)
    if author_year:
        extra = " ".join(author_year)
        if extra not in variants:
            variants.append(extra)
    return unique_keep_order([v for v in variants if v])

class BM25Index:
    """Minimal BM25 implementation to avoid an extra runtime dependency."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize(chunk.search_text) for chunk in self.chunks]
        self.doc_lens = np.array([len(tokens) for tokens in self.corpus_tokens], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.term_freqs: List[Dict[str, int]] = []
        doc_freqs: Dict[str, int] = {}
        for tokens in self.corpus_tokens:
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            for tok in tf:
                doc_freqs[tok] = doc_freqs.get(tok, 0) + 1
        n_docs = max(len(self.corpus_tokens), 1)
        self.idf: Dict[str, float] = {
            tok: np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for tok, df in doc_freqs.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if not self.corpus_tokens:
            return np.array([])
        scores = np.zeros(len(self.corpus_tokens), dtype=float)
        if self.avgdl == 0.0:
            return scores
        for tok in query_tokens:
            if tok not in self.idf:
                continue
            freqs = np.array([tf.get(tok, 0) for tf in self.term_freqs], dtype=float)
            numer = freqs * (self.k1 + 1.0)
            denom = freqs + self.k1 * (1.0 - self.b + self.b * (self.doc_lens / self.avgdl))
            denom = np.where(denom == 0.0, 1e-12, denom)
            scores += self.idf[tok] * (numer / denom)
        return scores

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        scores = self.get_scores(q_tokens)
        if len(scores) == 0:
            return []
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in order]


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], model_name: str, embeddings: Optional[np.ndarray] = None):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.model = None
        self.embeddings = embeddings

        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers is unavailable; dense retrieval disabled")
            return

        try:
            self.model = SentenceTransformer(model_name)
            if self.embeddings is None:
                texts = [chunk.search_text for chunk in self.chunks]
                self.embeddings = self.model.encode(
                    texts,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
        except Exception as exc:
            LOGGER.warning("Failed to initialize dense model %s: %s", model_name, exc)
            self.model = None
            self.embeddings = None

    @property
    def enabled(self) -> bool:
        return self.model is not None and self.embeddings is not None

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self.enabled:
            return []
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_emb)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in order]


class OptionalReranker:
    def __init__(self, model_name: Optional[str]):
        self.model_name = model_name
        self.model = None
        if not model_name:
            return
        if CrossEncoder is None:
            LOGGER.warning("CrossEncoder unavailable; falling back to lexical reranking")
            return
        try:
            self.model = CrossEncoder(model_name)
        except Exception as exc:
            LOGGER.warning("Failed to load reranker %s: %s", model_name, exc)
            self.model = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> List[float]:
        if not chunks:
            return []
        if self.enabled:
            pairs = [(query, chunk.search_text) for chunk in chunks]
            scores = self.model.predict(pairs)
            probabilities = 1 / (1 + np.exp(-np.clip(scores, -100, 100)))
            return [float(x) for x in probabilities]
        return [overlap_score(query, chunk.search_text) for chunk in chunks]


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        retrieval_cfg: RetrievalConfig,
        dense_embeddings: Optional[np.ndarray] = None,
    ):
        self.chunks = list(chunks)
        self.chunk_by_id: Dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.cfg = retrieval_cfg
        self.bm25 = BM25Index(self.chunks)
        self.dense = DenseIndex(self.chunks, retrieval_cfg.dense_model_name, embeddings=dense_embeddings) if retrieval_cfg.enable_dense else None
        self.reranker = OptionalReranker(retrieval_cfg.reranker_model_name) if retrieval_cfg.enable_reranker else OptionalReranker(None)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        if self.dense is not None and self.dense.embeddings is not None:
            np.save(index_dir / "dense_embeddings.npy", self.dense.embeddings)
        meta = {"dense_model_name": self.dense.model_name if self.dense is not None else None}
        (index_dir / "index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: str | Path, retrieval_cfg: RetrievalConfig) -> Optional["HybridRetriever"]:
        index_dir = Path(index_dir)
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            return None
        chunks: List[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))
        embeddings = None
        dense_path = index_dir / "dense_embeddings.npy"
        meta_path = index_dir / "index_meta.json"
        if dense_path.exists():
            saved_model = None
            if meta_path.exists():
                saved_model = json.loads(meta_path.read_text(encoding="utf-8")).get("dense_model_name")
            if saved_model != retrieval_cfg.dense_model_name:
                LOGGER.warning(
                    "Dense model changed (%s → %s); discarding cached embeddings",
                    saved_model,
                    retrieval_cfg.dense_model_name,
                )
            else:
                embeddings = np.load(dense_path)
        return cls(chunks=chunks, retrieval_cfg=retrieval_cfg, dense_embeddings=embeddings)

    def _rrf_accumulate(
        self,
        accumulator: Dict[str, Dict[str, float]],
        results: Sequence[Tuple[int, float]],
        source: str,
    ) -> None:
        for rank, (idx, score) in enumerate(results, start=1):
            chunk = self.chunks[idx]
            entry = accumulator[chunk.chunk_id]
            entry["fusion"] += 1.0 / (self.cfg.rrf_k + rank)
            entry[source] = max(entry[source], float(score))

    def search(self, question: str) -> List[RetrievalHit]:
        variants = build_query_variants(question)
        accumulator: Dict[str, Dict[str, float]] = defaultdict(lambda: {"bm25": 0.0, "dense": 0.0, "fusion": 0.0})

        for query in variants:
            bm25_results = self.bm25.search(query, top_k=self.cfg.bm25_top_k)
            self._rrf_accumulate(accumulator, bm25_results, "bm25")
            if self.dense is not None:
                dense_results = self.dense.search(query, top_k=self.cfg.dense_top_k)
                self._rrf_accumulate(accumulator, dense_results, "dense")

        candidates: List[RetrievalHit] = []
        for chunk_id, scores in accumulator.items():
            candidates.append(
                RetrievalHit(
                    chunk=self.chunk_by_id[chunk_id],
                    bm25_score=scores["bm25"],
                    dense_score=scores["dense"],
                    fusion_score=scores["fusion"],
                )
            )

        candidates.sort(key=lambda hit: hit.fusion_score, reverse=True)
        candidates = candidates[: self.cfg.final_top_k * 5]
        if self.cfg.enable_reranker:
            rerank_scores = self.reranker.rerank(question, [hit.chunk for hit in candidates])
            for hit, score in zip(candidates, rerank_scores):
                hit.rerank_score = float(score)
                hit.rerank_applied = True
        candidates.sort(key=lambda hit: (hit.final_score, hit.fusion_score), reverse=True)
        return candidates[: self.cfg.final_top_k]

    def unique_doc_ids(self, hits: Sequence[RetrievalHit]) -> List[str]:
        return unique_keep_order([hit.chunk.doc_id for hit in hits])
