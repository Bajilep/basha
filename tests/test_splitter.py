from basha.longtext.splitter import split_sentences


def test_splits_two_sentences():
    assert split_sentences("The camera is terrible. The battery is great.") == [
        "The camera is terrible.",
        "The battery is great.",
    ]


def test_abbreviation_is_not_a_sentence_end():
    assert split_sentences("Dr. Smith liked it.") == ["Dr. Smith liked it."]


def test_decimal_number_is_not_a_sentence_end():
    assert split_sentences("It costs 3.5 dollars. Fine.") == [
        "It costs 3.5 dollars.",
        "Fine.",
    ]


def test_exclamation_and_question_marks_split():
    assert split_sentences("Great! Awful?") == ["Great!", "Awful?"]


def test_single_sentence_without_punctuation():
    assert split_sentences("one") == ["one"]