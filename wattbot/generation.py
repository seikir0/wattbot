from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

from .config import GenerationConfig
from .schema import ABSTAIN_ANSWER, Prediction, RetrievalHit
from .utils import (
    clamp_text,
    env_or_none,
    extract_json_object,
    find_first_number,
    keyword_tokens,
    overlap_score,
    unique_keep_order,
)


LOGGER = logging.getLogger("wattbot.generation")


def format_context_blocks(hits: Sequence[RetrievalHit], max_chars_per_hit: int = 2000) -> str:
    blocks: List[str] = []
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        text = clamp_text(chunk.text, max_chars_per_hit)
        block = (
            f"[EVIDENCE {idx}]\n"
            f"doc_id: {chunk.doc_id}\n"
            f"title: {chunk.title}\n"
            f"pages: {chunk.page_start}-{chunk.page_end}\n"
            f"url: {chunk.url}\n"
            f"score: {hit.final_score:.6f}\n"
            f"text:\n{text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(question: str, answer_unit: str, hits: Sequence[RetrievalHit], top_score: float = 0.0) -> str:
    context = format_context_blocks(hits)
    unit_hint = "no specific unit" if not answer_unit or answer_unit == "is_blank" else answer_unit
    if top_score >= 0.7:
        confidence_hint = "HIGH — the evidence is highly relevant to this question. You MUST attempt an answer. Do NOT abstain."
    elif top_score >= 0.3:
        confidence_hint = "MEDIUM — attempt an answer if you can find relevant information."
    else:
        confidence_hint = "LOW — abstain if you are not confident."

    return f"""You are a precise scientific question-answering assistant. Answer using ONLY the evidence below.

Question: {question}
Expected answer unit: {unit_hint}
Retrieval confidence: {confidence_hint}

STRICT RULES:
1. Use ONLY information from the evidence. No outside knowledge.
2. "answer_value" must be a bare number or text only (e.g. "4.3" not "4.3 tCO2e").
3. "answer" is the full human-readable answer (e.g. "4.3 tCO2e").
4. "ref_id" must be a JSON array of doc_id strings from the evidence (e.g. ["patterson2021"]).
5. "ref_url" must be a JSON array of URL strings (e.g. ["https://..."]).
6. For True/False questions: answer_value must be exactly "1" for TRUE or "0" for FALSE.
7. "supporting_materials" should be a short direct quote or table reference from the evidence.
8. "explanation" must explain how the evidence supports the answer (never use "is_blank" here).
9. For range answers, format answer_value as [low, high] (e.g. "[0.02, 0.1]" or "[80, 90]").
10. Pay close attention to the exact unit in the question. If the question asks for "liters" give liters not millions of liters. If it asks for "million liters", answer_value must be in million liters (e.g. 5.439 not 5439000). If it asks for "USD" give the full number (e.g. 2510000 not 2.51). Never silently convert units.
11. If the question asks "what percentage", "how many", "what is the value" — always give a number, never TRUE/FALSE.
11b. For "which of the following" questions, answer_value must be the exact name/term, never a number.
12. For True/False questions, read VERY carefully. If the statement says X "outperforms" Y but evidence shows Y outperforms X, answer FALSE. Double-check before deciding.
13. Pay special attention to tables marked with [TABLE] — they often contain the exact numbers needed.
14. If the answer requires combining info from multiple evidence blocks, do so carefully.
15. When retrieval confidence is HIGH, you MUST provide an answer — only abstain if the evidence truly contains nothing related to the question.
16. If evidence is truly insufficient, abstain: answer="Unable to answer with confidence based on the provided documents", answer_value="is_blank", ref_id=[], ref_url=[], supporting_materials="is_blank", explanation="is_blank".
17. Return valid JSON only with exactly these fields in this order: reasoning, answer, answer_value, answer_unit, ref_id, ref_url, supporting_materials, explanation.
18. You MUST write step-by-step thinking in "reasoning" BEFORE answer_value. If calculation is needed, show the math in "reasoning" first.

Before answering, explicitly check: (1) what unit does the question ask for? (2) is my answer_value in that exact unit? (3) for True/False, does the evidence support or contradict the statement?
Evidence:
{context}""".strip()


class BaseGenerator:
    def generate(
        self,
        question_id: str,
        question: str,
        answer_unit: str,
        hits: Sequence[RetrievalHit],
        top_score: float = 0.0,
    ) -> Prediction:
        raise NotImplementedError


@dataclass
class RuleBasedGenerator(BaseGenerator):
    def _best_snippet(self, question: str, hits: Sequence[RetrievalHit]) -> tuple[str, RetrievalHit | None]:
        best_sentence = ""
        best_hit: RetrievalHit | None = None
        best_score = -1.0
        for hit in hits:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", hit.chunk.text)
            for sent in sentences:
                sentence = sent.strip()
                if len(sentence) < 20:
                    continue
                score = overlap_score(question, sentence)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_hit = hit
        return best_sentence, best_hit

    def generate(
        self,
        question_id: str,
        question: str,
        answer_unit: str,
        hits: Sequence[RetrievalHit],
        top_score: float = 0.0,
    ) -> Prediction:
        if not hits:
            return Prediction.blank(question_id, question, answer_unit)

        snippet, best_hit = self._best_snippet(question, hits)
        lowered_q = question.lower()
        if not snippet or best_hit is None:
            return Prediction.blank(question_id, question, answer_unit)

        if "true or false" in lowered_q or lowered_q.startswith("is ") or lowered_q.startswith("are "):
            s = snippet.lower()
            if re.search(r"\btrue\b|\byes\b", s):
                return Prediction(
                    question_id=question_id,
                    question=question,
                    answer="TRUE",
                    answer_value="1",
                    answer_unit=answer_unit or "is_blank",
                    ref_id=[best_hit.chunk.doc_id],
                    ref_url=[best_hit.chunk.url] if best_hit.chunk.url else [],
                    supporting_materials=clamp_text(snippet, 280),
                    explanation="Rule-based baseline found an explicit positive statement in the top evidence.",
                )
            if re.search(r"\bfalse\b|\bno\b", s):
                return Prediction(
                    question_id=question_id,
                    question=question,
                    answer="FALSE",
                    answer_value="0",
                    answer_unit=answer_unit or "is_blank",
                    ref_id=[best_hit.chunk.doc_id],
                    ref_url=[best_hit.chunk.url] if best_hit.chunk.url else [],
                    supporting_materials=clamp_text(snippet, 280),
                    explanation="Rule-based baseline found an explicit negative statement in the top evidence.",
                )
            return Prediction.blank(question_id, question, answer_unit)

        number = find_first_number(snippet)
        if number is None:
            return Prediction.blank(question_id, question, answer_unit)
        answer = f"{number} {answer_unit}".strip()
        return Prediction(
            question_id=question_id,
            question=question,
            answer=answer,
            answer_value=number,
            answer_unit=answer_unit or "is_blank",
            ref_id=[best_hit.chunk.doc_id],
            ref_url=[best_hit.chunk.url] if best_hit.chunk.url else [],
            supporting_materials=clamp_text(snippet, 280),
            explanation="Rule-based baseline selected the highest-overlap sentence and extracted the first number.",
        )


class OpenAICompatibleGenerator(BaseGenerator):
    def __init__(self, config: GenerationConfig, api_key: str | None = None):
        self.config = config
        api_key = api_key or env_or_none(config.api_key_env)
        if api_key is None:
            raise ValueError(f"Environment variable {config.api_key_env} is not set")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is not installed") from exc

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a careful evidence-grounded WattBot assistant. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "response_format" not in exc_str and "json" not in exc_str:
                raise
            LOGGER.warning("json_object mode rejected by endpoint, retrying without: %s", exc)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or "{}"

    def generate(
        self,
        question_id: str,
        question: str,
        answer_unit: str,
        hits: Sequence[RetrievalHit],
        top_score: float = 0.0,
    ) -> Prediction:
        if not hits:
            return Prediction.blank(question_id, question, answer_unit)
        prompt = build_prompt(question, answer_unit, hits, top_score=top_score)
        raw = self._call_llm(prompt)
        try:
            data = extract_json_object(raw)
            return Prediction.from_model_dict(
                question_id=question_id,
                question=question,
                answer_unit=answer_unit,
                data=data,
                raw_output=raw,
            )
        except Exception as exc:
            LOGGER.warning("Failed to parse LLM JSON for %s: %s", question_id, exc)
            fallback = RuleBasedGenerator().generate(question_id, question, answer_unit, hits)
            fallback.raw_output = raw
            return fallback


def build_generator(config: GenerationConfig) -> BaseGenerator:
    backend = (config.backend or "rule_based").lower()
    if backend == "openai_compatible":
        try:
            return OpenAICompatibleGenerator(config)
        except Exception as exc:
            LOGGER.warning("Falling back to rule-based generator: %s", exc)
            return RuleBasedGenerator()
    return RuleBasedGenerator()