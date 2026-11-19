from wattbot.retrieval import build_query_variants


def test_original_question_is_first():
    q = "How much electricity did GPT-3 training consume?"
    assert build_query_variants(q)[0] == q


def test_true_false_core_statement_extracted():
    variants = build_query_variants("True or false: GPT-3 consumed 1287 MWh")
    assert "GPT-3 consumed 1287 MWh" in variants


def test_entities_and_numbers_variant():
    variants = build_query_variants("How many V100 GPUs did BERT use in 2019?")
    assert any("V100" in v and "2019" in v and "BERT" in v for v in variants[1:])


def test_variants_are_unique_and_deterministic():
    q = "What is the PUE of Google data centers in 2022?"
    first = build_query_variants(q)
    assert first == build_query_variants(q)
    assert len(first) == len(set(first))
