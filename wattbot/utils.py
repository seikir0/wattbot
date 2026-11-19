from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np


LOGGER = logging.getLogger("wattbot")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "s",
    "since",
    "that",
    "the",
    "their",
    "this",
    "to",
    "true",
    "false",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_literal(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def serialize_list(values: Sequence[str] | str | None) -> str:
    if values is None:
        return "is_blank"
    if isinstance(values, str):
        if not values.strip():
            return "is_blank"
        return values
    cleaned = [v for v in values if str(v).strip()]
    if not cleaned:
        return "is_blank"
    return repr(list(cleaned))


def deserialize_list(value: Any) -> List[str]:
    parsed = safe_literal(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed if str(v).strip() and str(v).strip() != "is_blank"]
    text = str(parsed).strip()
    if not text or text == "is_blank":
        return []
    return [text]


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text.lower())]


def keyword_tokens(text: str) -> List[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def overlap_score(a: str, b: str) -> float:
    aa = set(keyword_tokens(a))
    bb = set(keyword_tokens(b))
    if not aa or not bb:
        return 0.0
    inter = len(aa & bb)
    union = len(aa | bb)
    return inter / max(union, 1)


def clamp_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def find_first_number(text: str) -> str | None:
    match = NUMBER_RE.search(text or "")
    if not match:
        return None
    return match.group(0).replace(",", "")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, data: Any, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def strip_code_fence(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        return json.loads(snippet)
    raise ValueError("Could not parse JSON object from model output")


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
