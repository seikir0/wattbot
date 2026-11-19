"""Abstention safety net and citation hardening in WattBotPipeline._normalize_prediction."""

from wattbot.schema import Prediction

from conftest import make_hit


def make_pred(answer_value="24", ref_id=None, ref_url=None):
    return Prediction(
        question_id="q1",
        question="How many?",
        answer=f"{answer_value} data centers",
        answer_value=answer_value,
        answer_unit="data centers",
        ref_id=ref_id if ref_id is not None else ["doc1"],
        ref_url=ref_url if ref_url is not None else [],
        supporting_materials="some quote",
        explanation="some reasoning",
    )


def test_low_retrieval_score_forces_blank(pipeline):
    hits = [make_hit("doc1", score=0.01)]  # below 0.03 threshold
    out = pipeline._normalize_prediction(make_pred(), "data centers", hits)
    assert out.answer_value == "is_blank"
    assert out.ref_id == [] and out.ref_url == []
    assert out.supporting_materials == "is_blank" and out.explanation == "is_blank"


def test_llm_blank_clears_all_citation_fields(pipeline):
    pred = make_pred(answer_value="is_blank", ref_id=["doc1"])
    out = pipeline._normalize_prediction(pred, "data centers", [make_hit("doc1", 0.9)])
    assert out.ref_id == [] and out.supporting_materials == "is_blank"


def test_hallucinated_ref_id_is_dropped(pipeline):
    pred = make_pred(ref_id=["doc1", "fake_doc"])
    out = pipeline._normalize_prediction(pred, "data centers", [make_hit("doc1", 0.9)])
    assert out.ref_id == ["doc1"]


def test_falls_back_to_top_retrieved_doc_when_no_valid_ref(pipeline):
    pred = make_pred(ref_id=["fake_doc"])
    hits = [make_hit("doc2", 0.9), make_hit("doc1", 0.5)]
    out = pipeline._normalize_prediction(pred, "data centers", hits)
    assert out.ref_id == ["doc2"]


def test_ref_url_rebuilt_from_metadata_not_model(pipeline):
    pred = make_pred(ref_id=["doc1"], ref_url=["https://made-up.example.com"])
    out = pipeline._normalize_prediction(pred, "data centers", [make_hit("doc1", 0.9)])
    assert out.ref_url == ["https://example.org/doc1.pdf"]


def test_missing_supporting_materials_backfilled_from_top_hit(pipeline):
    pred = make_pred()
    pred.supporting_materials = "is_blank"
    out = pipeline._normalize_prediction(pred, "data centers", [make_hit("doc1", 0.9, text="24 data centers")])
    assert out.supporting_materials == "24 data centers"
