"""Precompute answers for the demo's example list.

Uses whatever backend config.json specifies. With the LLM backend this is the
only step that costs API credits, and it runs once.

    python scripts/build_demo_cache.py --config config.json --split train --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.service import answer_question  # noqa: E402
from wattbot.config import load_config  # noqa: E402
from wattbot.generation import OpenAICompatibleGenerator  # noqa: E402
from wattbot.pipeline import WattBotPipeline  # noqa: E402
from wattbot.utils import env_or_none  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default="demo/cached_answers.json")
    args = parser.parse_args()

    pipeline = WattBotPipeline(load_config(args.config))
    api_key = None
    if isinstance(pipeline.generator, OpenAICompatibleGenerator):
        api_key = env_or_none(pipeline.config.generation.api_key_env)

    df = pipeline.train_df if args.split == "train" else pipeline.test_df
    rows = []
    for _, row in df.head(args.limit).iterrows():
        result = answer_question(
            pipeline,
            str(row["question"]),
            answer_unit=str(row.get("answer_unit", "") or ""),
            api_key=api_key,
        )
        if args.split == "train":
            result["expected_answer_value"] = str(row.get("answer_value", ""))
        rows.append(result)
        print(f"{len(rows):>3}  {result['answer_value']:<20} {row['question'][:70]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} answers to {out}")


if __name__ == "__main__":
    main()
