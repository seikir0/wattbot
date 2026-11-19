from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from .config import AppConfig
from .schema import DocumentMeta
from .utils import ensure_dir


LOGGER = logging.getLogger("wattbot.io")


def read_competition_frames(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(config.paths.competition_dir)
    train_df = pd.read_csv(base / "train_QA.csv", encoding='utf-8-sig')
    test_df = pd.read_csv(base / "test_Q.csv", encoding='utf-8-sig')
    metadata_df = pd.read_csv(base / "metadata.csv", encoding='latin1')
    return train_df, test_df, metadata_df


def build_metadata_map(metadata_df: pd.DataFrame) -> Dict[str, DocumentMeta]:
    output: Dict[str, DocumentMeta] = {}
    for _, row in metadata_df.iterrows():
        doc_id = str(row["id"]).strip()
        output[doc_id] = DocumentMeta(
            doc_id=doc_id,
            title=str(row.get("title", "") or "").strip(),
            url=str(row.get("url", "") or "").strip(),
            doc_type=str(row.get("type", "paper") or "paper").strip(),
            year=None if pd.isna(row.get("year")) else str(row.get("year")).strip(),
            citation=str(row.get("citation", "") or "").strip() or None,
        )
    return output


def find_local_document(doc_id: str, docs_dir: str | Path) -> Optional[Path]:
    docs_dir = Path(docs_dir)
    candidates = [
        docs_dir / f"{doc_id}.pdf",
        docs_dir / f"{doc_id}.txt",
        docs_dir / f"{doc_id}.md",
        docs_dir / doc_id / "document.pdf",
        docs_dir / doc_id / "document.txt",
        docs_dir / doc_id / "document.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def download_document(meta: DocumentMeta, docs_dir: str | Path, timeout_seconds: int = 30, user_agent: str = "wattbot-baseline/0.1") -> Optional[Path]:
    docs_dir = ensure_dir(docs_dir)
    target = docs_dir / f"{meta.doc_id}.pdf"
    if target.exists():
        return target
    if not meta.url:
        return None

    headers = {"User-Agent": user_agent}
    try:
        response = requests.get(meta.url, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            LOGGER.warning("URL for %s returned HTML instead of a document; skipping: %s", meta.doc_id, meta.url)
            return None
        if "pdf" not in content_type and not meta.url.lower().endswith(".pdf"):
            LOGGER.warning("URL for %s may not be a PDF (Content-Type: %s): %s", meta.doc_id, content_type, meta.url)
        target.write_bytes(response.content)
        LOGGER.info("Downloaded %s -> %s", meta.doc_id, target)
        return target
    except Exception as exc:
        LOGGER.warning("Failed to download %s from %s: %s", meta.doc_id, meta.url, exc)
        return None
