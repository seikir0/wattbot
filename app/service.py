"""Answer a single free-form question with the WattBot pipeline.

The generator is chosen per request: the visitor's own API key if they give one,
otherwise the free rule-based baseline. Keys are used for that request only and
never stored or logged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from wattbot.generation import OpenAICompatibleGenerator, RuleBasedGenerator
from wattbot.pipeline import WattBotPipeline
from wattbot.utils import clamp_text


def answer_question(
    pipeline: WattBotPipeline,
    question: str,
    answer_unit: str = "",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if api_key:
        cfg = copy.copy(pipeline.config.generation)
        if base_url:
            cfg.base_url = base_url
        if model:
            cfg.model = model
        generator = OpenAICompatibleGenerator(cfg, api_key=api_key)
        mode = "llm"
    else:
        generator = RuleBasedGenerator()
        mode = "rule_based"

    unit = answer_unit.strip() or "is_blank"
    retriever = pipeline.build_or_load_index()
    hits = retriever.search(question)
    top_score = hits[0].final_score if hits else 0.0
    prediction = generator.generate("live", question, unit, hits, top_score=top_score)
    prediction = pipeline._normalize_prediction(prediction, unit, hits)

    return {
        "mode": mode,
        "question": question,
        "answer": prediction.answer,
        "answer_value": prediction.answer_value,
        "answer_unit": prediction.answer_unit,
        "ref_id": prediction.ref_id,
        "ref_url": prediction.ref_url,
        "explanation": prediction.explanation,
        "supporting_materials": prediction.supporting_materials,
        "top_score": top_score,
        "evidence": [
            {
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "url": hit.chunk.url,
                "pages": [hit.chunk.page_start, hit.chunk.page_end],
                "score": hit.final_score,
                "text": clamp_text(hit.chunk.text, 600),
            }
            for hit in hits[:5]
        ],
    }
