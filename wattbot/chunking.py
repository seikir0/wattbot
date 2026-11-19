from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

from .config import RetrievalConfig
from .schema import Chunk, DocumentMeta
from .utils import normalize_whitespace


def _page_paragraphs(pages: Sequence[str]) -> List[Tuple[int, str]]:
    units: List[Tuple[int, str]] = []
    for page_no, page_text in enumerate(pages, start=1):
        cleaned = normalize_whitespace(page_text)
        if not cleaned:
            continue
        paragraphs = [normalize_whitespace(p) for p in cleaned.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned]
        for paragraph in paragraphs:
            units.append((page_no, paragraph))
    return units


def chunk_document(meta: DocumentMeta, pages: Sequence[str], retrieval_cfg: RetrievalConfig) -> List[Chunk]:
    units = _page_paragraphs(pages)
    if not units:
        return []

    chunk_size = retrieval_cfg.chunk_size
    overlap = retrieval_cfg.chunk_overlap
    chunks: List[Chunk] = []

    window: List[Tuple[int, str]] = []
    window_chars = 0
    chunk_index = 0

    def emit(items: List[Tuple[int, str]]) -> None:
        nonlocal chunk_index
        if not items:
            return
        pages_here = [page for page, _ in items]
        text = "\n\n".join(text for _, text in items)
        chunks.append(
            Chunk(
                chunk_id=f"{meta.doc_id}::c{chunk_index:04d}",
                doc_id=meta.doc_id,
                text=text,
                page_start=min(pages_here),
                page_end=max(pages_here),
                title=meta.title,
                url=meta.url,
                metadata={"year": meta.year, "type": meta.doc_type},
            )
        )
        chunk_index += 1

    for item in units:
        page_no, paragraph = item
        para_len = len(paragraph)
        if window and window_chars + para_len > chunk_size:
            emit(window)
            if overlap > 0:
                kept: List[Tuple[int, str]] = []
                kept_chars = 0
                for old in reversed(window):
                    kept.insert(0, old)
                    kept_chars += len(old[1])
                    if kept_chars >= overlap:
                        break
                window = kept
                window_chars = sum(len(text) for _, text in window)
            else:
                window = []
                window_chars = 0
        window.append((page_no, paragraph))
        window_chars += para_len

    if window:
        emit(window)

    return chunks


def chunk_records_to_dicts(chunks: Iterable[Chunk]) -> List[dict]:
    return [asdict(chunk) for chunk in chunks]
