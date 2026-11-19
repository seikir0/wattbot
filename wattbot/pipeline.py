from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .chunking import chunk_document
from .config import AppConfig
from .evaluation import evaluate_predictions, evaluate_retrieval_recall
from .generation import build_generator
from .io_utils import build_metadata_map, download_document, find_local_document, read_competition_frames
from .pdf_utils import read_document_pages
from .retrieval import HybridRetriever
from .schema import Prediction, RetrievalHit
from .utils import clamp_text, dump_json, setup_logging, set_seed, unique_keep_order


LOGGER = logging.getLogger("wattbot.pipeline")


class WattBotPipeline:
    def __init__(self, config: AppConfig):
        load_dotenv(override=False)
        self.config = config
        setup_logging(config.runtime.log_level)
        set_seed(config.runtime.seed)
        self.config.ensure_dirs()
        self.train_df, self.test_df, metadata_df = read_competition_frames(config)
        self.metadata_map = build_metadata_map(metadata_df)
        self.retriever: Optional[HybridRetriever] = None
        self.generator = build_generator(config.generation)

    def _build_chunks(self) -> List[Any]:
        chunks = []
        for doc_id, meta in tqdm(self.metadata_map.items(), desc="Parsing documents"):
            doc_path = find_local_document(doc_id, self.config.paths.docs_dir)
            if doc_path is None and self.config.documents.download_if_missing:
                doc_path = download_document(
                    meta,
                    self.config.paths.docs_dir,
                    timeout_seconds=self.config.documents.request_timeout_seconds,
                    user_agent=self.config.documents.user_agent,
                )
            if doc_path is None:
                LOGGER.warning("Missing local document for %s", doc_id)
                continue
            try:
                pages = read_document_pages(doc_path)
                doc_chunks = chunk_document(meta, pages, self.config.retrieval)
                chunks.extend(doc_chunks)
            except Exception as exc:
                LOGGER.warning("Failed to process %s (%s): %s", doc_id, doc_path, exc)
        LOGGER.info("Built %d chunks from %d documents", len(chunks), len(self.metadata_map))
        return chunks

    def build_or_load_index(self, force_rebuild: bool = False) -> HybridRetriever:
        if self.retriever is not None and not force_rebuild:
            return self.retriever
        if not force_rebuild:
            loaded = HybridRetriever.load(self.config.paths.index_dir, self.config.retrieval)
            if loaded is not None:
                self.retriever = loaded
                LOGGER.info("Loaded cached index from %s", self.config.paths.index_dir)
                return loaded
        chunks = self._build_chunks()
        self.retriever = HybridRetriever(chunks=chunks, retrieval_cfg=self.config.retrieval)
        self.retriever.save(self.config.paths.index_dir)
        LOGGER.info("Saved index to %s", self.config.paths.index_dir)
        return self.retriever

    def _normalize_prediction(
        self,
        prediction: Prediction,
        answer_unit: str,
        hits: Sequence[RetrievalHit],
    ) -> Prediction:
        if prediction.answer_value.strip().lower() == "is_blank":
            return Prediction.blank(prediction.question_id, prediction.question, answer_unit)

        if hits and hits[0].final_score < self.config.retrieval.blank_relevance_threshold:
            return Prediction.blank(prediction.question_id, prediction.question, answer_unit)

        valid_doc_ids = unique_keep_order([hit.chunk.doc_id for hit in hits])
        cleaned_ref_ids = [doc_id for doc_id in prediction.ref_id if doc_id in valid_doc_ids]
        if not cleaned_ref_ids and valid_doc_ids:
            cleaned_ref_ids = [valid_doc_ids[0]]
        prediction.ref_id = cleaned_ref_ids

        ref_urls = []
        for doc_id in prediction.ref_id:
            meta = self.metadata_map.get(doc_id)
            if meta and meta.url:
                ref_urls.append(meta.url)
        prediction.ref_url = unique_keep_order(ref_urls)

        if not prediction.answer_unit or prediction.answer_unit == "is_blank":
            prediction.answer_unit = answer_unit or "is_blank"

        if not prediction.supporting_materials or prediction.supporting_materials == "is_blank":
            if hits:
                prediction.supporting_materials = clamp_text(hits[0].chunk.text, 280)
            else:
                prediction.supporting_materials = "is_blank"

        if not prediction.explanation or prediction.explanation == "is_blank":
            prediction.explanation = "Grounded in the top retrieved evidence chunk(s)." if hits else "is_blank"

        return prediction

    def predict_row(self, row: pd.Series) -> Tuple[Prediction, List[RetrievalHit], Dict[str, Any]]:
        retriever = self.build_or_load_index()
        question_id = str(row["id"])
        question = str(row["question"])
        answer_unit = str(row.get("answer_unit", "") or "").strip() or "is_blank"
        hits = retriever.search(question)
        top_score = hits[0].final_score if hits else 0.0
        prediction = self.generator.generate(question_id, question, answer_unit, hits, top_score=top_score)
        prediction = self._normalize_prediction(prediction, answer_unit, hits)
        debug = {
            "id": question_id,
            "question": question,
            "top_doc_ids": retriever.unique_doc_ids(hits),
            "hits": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_id,
                    "pages": [hit.chunk.page_start, hit.chunk.page_end],
                    "title": hit.chunk.title,
                    "url": hit.chunk.url,
                    "fusion_score": hit.fusion_score,
                    "rerank_score": hit.rerank_score,
                    "final_score": hit.final_score,
                    "text_preview": clamp_text(hit.chunk.text, 500),
                }
                for hit in hits
            ],
            "raw_output": prediction.raw_output,
        }
        return prediction, hits, debug

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        split_name: str,
        limit: Optional[int] = None,
        save_outputs: bool = True,
        output_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        work_df = df.copy()
        if limit is not None:
            work_df = work_df.head(limit)

        predictions: List[Dict[str, Any]] = []
        debug_records: List[Dict[str, Any]] = []

        for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc=f"Predicting {split_name}"):
            pred, _, debug = self.predict_row(row)
            predictions.append(pred.as_submission_row())
            debug_records.append(debug)

        pred_df = pd.DataFrame(predictions)
        if save_outputs:
            out_name = output_name or f"{split_name}_predictions.csv"
            out_path = Path(self.config.paths.output_dir) / out_name
            pred_df.to_csv(out_path, index=False)
            debug_path = Path(self.config.paths.debug_dir) / f"{split_name}_debug.jsonl"
            with debug_path.open("w", encoding="utf-8") as f:
                for record in debug_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            LOGGER.info("Wrote predictions to %s", out_path)
            LOGGER.info("Wrote debug traces to %s", debug_path)
        return pred_df, debug_records

    def predict_split(
        self,
        split: str,
        limit: Optional[int] = None,
        save_outputs: bool = True,
        output_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        split = split.lower()
        if split == "train":
            return self.predict_dataframe(self.train_df, split_name="train", limit=limit, save_outputs=save_outputs, output_name=output_name)
        if split == "test":
            return self.predict_dataframe(self.test_df, split_name="test", limit=limit, save_outputs=save_outputs, output_name=output_name)
        raise ValueError(f"Unknown split: {split}")

    def evaluate(self, limit: Optional[int] = None, save_outputs: bool = True) -> Dict[str, Any]:
        pred_df, debug_records = self.predict_split("train", limit=limit, save_outputs=save_outputs)
        gt_df = self.train_df.head(limit) if limit is not None else self.train_df
        metrics = evaluate_predictions(gt_df, pred_df)
        metrics.update(evaluate_retrieval_recall(gt_df, debug_records, k=self.config.retrieval.final_top_k))
        metrics["limit"] = float(limit) if limit is not None else float(len(gt_df))
        if save_outputs:
            dump_json(Path(self.config.paths.output_dir) / "train_eval_metrics.json", metrics)
            LOGGER.info("Wrote evaluation metrics to %s", Path(self.config.paths.output_dir) / "train_eval_metrics.json")
        return metrics
