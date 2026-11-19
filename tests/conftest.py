from types import SimpleNamespace

import pytest

from wattbot.pipeline import WattBotPipeline
from wattbot.schema import Chunk, DocumentMeta, RetrievalHit


def make_hit(doc_id: str, score: float, text: str = "evidence text") -> RetrievalHit:
    chunk = Chunk(chunk_id=f"{doc_id}-0", doc_id=doc_id, text=text, page_start=1, page_end=1)
    return RetrievalHit(chunk=chunk, fusion_score=score)


@pytest.fixture
def pipeline():
    """A pipeline with only what post-processing needs: no data files, models or API key."""
    p = object.__new__(WattBotPipeline)
    p.config = SimpleNamespace(retrieval=SimpleNamespace(blank_relevance_threshold=0.03))
    p.metadata_map = {
        "doc1": DocumentMeta(doc_id="doc1", title="Doc 1", url="https://example.org/doc1.pdf"),
        "doc2": DocumentMeta(doc_id="doc2", title="Doc 2", url="https://example.org/doc2.pdf"),
    }
    return p
