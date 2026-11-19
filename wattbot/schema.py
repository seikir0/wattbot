from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .utils import deserialize_list, serialize_list, unique_keep_order


ABSTAIN_ANSWER = "Unable to answer with confidence based on the provided documents."


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    url: str
    doc_type: str = "paper"
    year: str | None = None
    citation: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    title: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        year = self.metadata.get("year")
        if year:
            parts.append(f"Year: {year}")
        parts.append(self.text)
        return "\n".join(parts).strip()


@dataclass
class RetrievalHit:
    chunk: Chunk
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    rerank_applied: bool = False

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_applied else self.fusion_score


@dataclass
class Prediction:
    question_id: str
    question: str
    answer: str
    answer_value: str
    answer_unit: str
    ref_id: List[str] = field(default_factory=list)
    ref_url: List[str] = field(default_factory=list)
    supporting_materials: str = "is_blank"
    explanation: str = "is_blank"
    raw_output: str | None = None

    @classmethod
    def blank(cls, question_id: str, question: str, answer_unit: str) -> "Prediction":
        return cls(
            question_id=question_id,
            question=question,
            answer=ABSTAIN_ANSWER,
            answer_value="is_blank",
            answer_unit=answer_unit or "is_blank",
            ref_id=[],
            ref_url=[],
            supporting_materials="is_blank",
            explanation="is_blank",
        )

    @classmethod
    def from_model_dict(
        cls,
        question_id: str,
        question: str,
        answer_unit: str,
        data: Dict[str, Any],
        raw_output: str | None = None,
    ) -> "Prediction":
        ref_id = unique_keep_order(deserialize_list(data.get("ref_id")))
        ref_url = unique_keep_order(deserialize_list(data.get("ref_url")))
        answer = str(data.get("answer", "")).strip() or ABSTAIN_ANSWER
        answer_value = str(data.get("answer_value", "")).strip() or "is_blank"
        if answer_value.lower() in {"is_blank", "none", "n/a", ""}:
            answer_value = "is_blank"
        # 归一化范围答案格式，去掉空格
        if answer_value.startswith("[") and answer_value.endswith("]"):
            import re
            answer_value = re.sub(r",\s+", ",", answer_value)
        # 归一化 2.0 -> 2
        if answer_value != "is_blank":
            try:
                f = float(answer_value)
                if f == int(f):
                    answer_value = str(int(f))
            except:
                pass
        answer_unit_out = str(data.get("answer_unit", "")).strip() or answer_unit or "is_blank"
        supporting_materials = str(data.get("supporting_materials", "")).strip() or "is_blank"
        explanation = str(data.get("explanation", "")).strip() or "is_blank"
        return cls(
            question_id=question_id,
            question=question,
            answer=answer,
            answer_value=answer_value,
            answer_unit=answer_unit_out,
            ref_id=ref_id,
            ref_url=ref_url,
            supporting_materials=supporting_materials,
            explanation=explanation,
            raw_output=raw_output,
        )

    def as_submission_row(self) -> Dict[str, Any]:
        return {
            "id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "answer_value": self.answer_value,
            "answer_unit": self.answer_unit,
            "ref_id": serialize_list(self.ref_id),
            "ref_url": serialize_list(self.ref_url),
            "supporting_materials": self.supporting_materials,
            "explanation": self.explanation,
        }
