from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import pandas as pd

from .utils import deserialize_list


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"true", "yes"}:
        return "1"
    if text.lower() in {"false", "no"}:
        return "0"
    return text.replace(",", "").strip().lower()


def _parse_range(value: str) -> tuple[float, float] | None:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return None
    try:
        left, right = value[1:-1].split(",", 1)
        return float(left), float(right)
    except Exception:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def answer_value_score(gt: Any, pred: Any, rel_tol: float = 1e-3) -> float:
    gt_norm = _normalize_text(gt)
    pred_norm = _normalize_text(pred)

    if gt_norm == "is_blank":
        return 1.0 if pred_norm == "is_blank" else 0.0
    if pred_norm == "is_blank":
        return 0.0

    # Parse ranges before _normalize_text strips commas, otherwise "[10,20]" becomes "[1020]".
    gt_range = _parse_range(str(gt).strip())
    pred_range = _parse_range(str(pred).strip())
    if gt_range is not None and pred_range is not None:
        return 1.0 if all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12) for a, b in zip(gt_range, pred_range)) else 0.0

    gt_float = _to_float(gt_norm)
    pred_float = _to_float(pred_norm)
    if gt_float is not None and pred_float is not None:
        return 1.0 if math.isclose(gt_float, pred_float, rel_tol=rel_tol, abs_tol=1e-12) else 0.0

    return 1.0 if gt_norm == pred_norm else 0.0


def ref_id_score(gt: Any, pred: Any) -> float:
    gt_list = set(deserialize_list(gt))
    pred_list = set(deserialize_list(pred))
    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0
    return len(gt_list & pred_list) / len(gt_list | pred_list)


def is_na_score(gt_answer_value: Any, pred_answer_value: Any) -> float:
    gt_blank = _normalize_text(gt_answer_value) == "is_blank"
    pred_blank = _normalize_text(pred_answer_value) == "is_blank"
    return 1.0 if gt_blank == pred_blank else 0.0


def compute_approx_wattbot_score(gt_row: pd.Series, pred_row: pd.Series) -> Dict[str, float]:
    ans = answer_value_score(gt_row.get("answer_value"), pred_row.get("answer_value"))
    ref = ref_id_score(gt_row.get("ref_id"), pred_row.get("ref_id"))
    na = is_na_score(gt_row.get("answer_value"), pred_row.get("answer_value"))
    total = 0.75 * ans + 0.15 * ref + 0.10 * na
    return {
        "answer_value": ans,
        "ref_id": ref,
        "is_na": na,
        "score": total,
    }


def evaluate_predictions(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, float]:
    pred_map = {str(row["id"]): row for _, row in pred_df.iterrows()}
    rows: List[Dict[str, float]] = []
    for _, gt_row in gt_df.iterrows():
        qid = str(gt_row["id"])
        pred_row = pred_map.get(qid)
        if pred_row is None:
            pred_row = pd.Series({"answer_value": "is_blank", "ref_id": "is_blank"})
        rows.append(compute_approx_wattbot_score(gt_row, pred_row))

    metrics = {
        "n": float(len(rows)),
        "answer_value": float(sum(r["answer_value"] for r in rows) / max(len(rows), 1)),
        "ref_id": float(sum(r["ref_id"] for r in rows) / max(len(rows), 1)),
        "is_na": float(sum(r["is_na"] for r in rows) / max(len(rows), 1)),
        "score": float(sum(r["score"] for r in rows) / max(len(rows), 1)),
    }
    return metrics


def evaluate_retrieval_recall(gt_df: pd.DataFrame, debug_records: Iterable[Dict[str, Any]], k: int) -> Dict[str, float]:
    gt_map = {str(row["id"]): row for _, row in gt_df.iterrows()}
    total = 0
    hits = 0
    for record in debug_records:
        qid = str(record["id"])
        gt_row = gt_map.get(qid)
        if gt_row is None:
            continue
        gt_refs = set(deserialize_list(gt_row.get("ref_id")))
        if not gt_refs:
            continue
        pred_docs = list(record.get("top_doc_ids", []))[:k]
        total += 1
        if gt_refs.intersection(pred_docs):
            hits += 1
    return {
        f"doc_recall@{k}": (hits / total) if total else 0.0,
        "retrieval_eval_questions": float(total),
    }
