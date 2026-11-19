from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .utils import normalize_whitespace

LOGGER = logging.getLogger("wattbot.pdf")


def read_document_pages(path: str | Path) -> List[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [normalize_whitespace(text)]
    if suffix != ".pdf":
        raise ValueError(f"Unsupported document type: {path}")
    return _read_pdf(path)


def _read_pdf(path: Path) -> List[str]:
    # 首选 pymupdf4llm (极速，自带高质量的 Markdown 表格解析)
    try:
        return _read_pdf_pymupdf4llm(path)
    except Exception as exc:
        LOGGER.warning("pymupdf4llm failed for %s: %s, falling back to pdfplumber", path, exc)
        
    # 降级到 pdfplumber
    try:
        return _read_pdf_pdfplumber(path)
    except Exception as exc:
        LOGGER.warning("pdfplumber failed for %s: %s, falling back to pypdf", path, exc)
        
    # 最后降级到 pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(normalize_whitespace(page.extract_text() or ""))
        return pages
    except Exception as exc:
        LOGGER.warning("pypdf also failed for %s: %s", path, exc)
        return []


def _read_pdf_pymupdf4llm(path: Path) -> List[str]:
    import pymupdf4llm
    
    # 使用 page_chunks=True 提取，保留物理页码对应关系
    md_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    pages = []
    
    for page_data in md_pages:
        text = page_data.get("text", "")
        # 这里不使用完全无脑的 normalize_whitespace，以防止破坏 Markdown 表格的换行和管道符
        # 仅去除多余的前后空格
        if text.strip():
            pages.append(text.strip())
            
    return pages


def _read_pdf_pdfplumber(path: Path) -> List[str]:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts = []
            text = page.extract_text() or ""
            if text.strip():
                parts.append(normalize_whitespace(text))
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                table_text = "\n".join(rows)
                if table_text.strip():
                    parts.append("[TABLE]\n" + table_text)
            pages.append("\n\n".join(parts))
    return pages