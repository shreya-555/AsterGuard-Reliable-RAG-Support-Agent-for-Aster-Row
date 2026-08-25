from evaluation.assertions import concept_matches


def test_concept_patterns_support_alternative_phrasings():
    assert concept_matches(
        "Aster & Row currently ships internationally only to Canada.",
        "Canada is supported",
    )


def test_compound_concept_requires_both_parts():
    assert not concept_matches(
        "Drinkware has a one-year warranty.",
        "drinkware and travel accessories have 1 year",
    )
    assert concept_matches(
        "Drinkware has 1 year and travel accessories have 1 year.",
        "drinkware and travel accessories have 1 year",
    )


def test_human_review_before_approval_accepts_required_word_order():
    assert concept_matches(
        "Human review before approval is required.",
        "human review before approval",
    )
    assert not concept_matches(
        "Human review is available.",
        "human review before approval",
    )
