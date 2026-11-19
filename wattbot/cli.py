from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io_utils import build_metadata_map, download_document, find_local_document, read_competition_frames
from .pipeline import WattBotPipeline


def cmd_build_index(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pipeline = WattBotPipeline(cfg)
    pipeline.build_or_load_index(force_rebuild=args.force)


def cmd_download_docs(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    _, _, metadata_df = read_competition_frames(cfg)
    metadata_map = build_metadata_map(metadata_df)
    downloaded = 0
    skipped = 0
    for meta in metadata_map.values():
        existing = find_local_document(meta.doc_id, cfg.paths.docs_dir)
        if existing is not None and not args.force:
            skipped += 1
            continue
        result = download_document(
            meta,
            cfg.paths.docs_dir,
            timeout_seconds=cfg.documents.request_timeout_seconds,
            user_agent=cfg.documents.user_agent,
        )
        if result is not None:
            downloaded += 1
    print(json.dumps({"downloaded": downloaded, "skipped": skipped}, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pipeline = WattBotPipeline(cfg)
    pipeline.build_or_load_index(force_rebuild=args.force_rebuild)
    pred_df, _ = pipeline.predict_split(
        split=args.split,
        limit=args.limit,
        save_outputs=True,
        output_name=args.output,
    )
    print(pred_df.head().to_string(index=False))


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pipeline = WattBotPipeline(cfg)
    pipeline.build_or_load_index(force_rebuild=args.force_rebuild)
    metrics = pipeline.evaluate(limit=args.limit, save_outputs=True)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WattBot RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-index", help="Parse documents and build retrieval index")
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--force", action="store_true", help="Rebuild even if cached index exists")
    p_build.set_defaults(func=cmd_build_index)

    p_download = subparsers.add_parser("download-docs", help="Download missing PDFs listed in metadata.csv")
    p_download.add_argument("--config", required=True)
    p_download.add_argument("--force", action="store_true", help="Redownload documents even if present locally")
    p_download.set_defaults(func=cmd_download_docs)

    p_predict = subparsers.add_parser("predict", help="Generate a submission-style CSV for train or test")
    p_predict.add_argument("--config", required=True)
    p_predict.add_argument("--split", choices=["train", "test"], default="test")
    p_predict.add_argument("--limit", type=int, default=None)
    p_predict.add_argument("--output", default=None, help="Custom output CSV filename")
    p_predict.add_argument("--force-rebuild", action="store_true")
    p_predict.set_defaults(func=cmd_predict)

    p_eval = subparsers.add_parser("evaluate", help="Run approximate local evaluation on train_QA.csv")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--force-rebuild", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
