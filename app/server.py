"""Web demo for WattBot RAG.

Run from the repo root:
    uvicorn app.server:app --port 8000

Environment variables:
    WATTBOT_CONFIG      path to config.json (default: config.json)
    WATTBOT_DEMO_CACHE  precomputed answers shown as examples (default: demo/cached_answers.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from wattbot.config import load_config
from wattbot.pipeline import WattBotPipeline

from .service import answer_question

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = os.environ.get("WATTBOT_CONFIG", "config.json")
CACHE_PATH = Path(os.environ.get("WATTBOT_DEMO_CACHE", "demo/cached_answers.json"))

app = FastAPI(title="WattBot RAG demo")
_pipeline: Optional[WattBotPipeline] = None


def get_pipeline() -> WattBotPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = WattBotPipeline(load_config(CONFIG_PATH))
        _pipeline.build_or_load_index()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    answer_unit: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/examples")
def examples() -> list:
    if not CACHE_PATH.exists():
        return []
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return answer_question(
            get_pipeline(),
            req.question,
            answer_unit=req.answer_unit,
            api_key=(req.api_key or "").strip() or None,
            base_url=(req.base_url or "").strip() or None,
            model=(req.model or "").strip() or None,
        )
    except Exception as exc:  # surface a readable message, never echo the key
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from None
