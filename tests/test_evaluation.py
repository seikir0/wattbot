from wattbot.evaluation import answer_value_score, is_na_score, ref_id_score


def test_numeric_answers_compared_as_numbers():
    assert answer_value_score("462.5", "462.50") == 1.0
    assert answer_value_score("24", "23") == 0.0


def test_range_answers():
    assert answer_value_score("[10,20]", "[10, 20]") == 1.0


def test_blank_only_matches_blank():
    assert answer_value_score("is_blank", "is_blank") == 1.0
    assert answer_value_score("is_blank", "5") == 0.0
    assert answer_value_score("5", "is_blank") == 0.0


def test_ref_id_is_jaccard():
    assert ref_id_score("['a','b']", "['a']") == 0.5
    assert ref_id_score("['a']", "['a']") == 1.0


def test_is_na():
    assert is_na_score("is_blank", "is_blank") == 1.0
    assert is_na_score("is_blank", "3") == 0.0
    assert is_na_score("3", "4") == 1.0


def test_different_ranges_do_not_collide():
    # Regression: stripping commas used to make these identical ("[123]").
    assert answer_value_score("[1,23]", "[12,3]") == 0.0
