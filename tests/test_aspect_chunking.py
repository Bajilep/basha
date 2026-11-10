"""
Tests for EnglishAspectModel text handling.
A fake pipeline replaces the real model, so these run instantly
and need no downloads.
"""

import pytest

from basha.aspects.aspect_model import EnglishAspectModel


class FakePipeline:
    """Records every piece it receives; finds one aspect per call."""

    def __init__(self, word="food", label="pos"):
        self.calls = []
        self.word = word
        self.label = label

    def __call__(self, piece):
        self.calls.append(piece)
        return [{
            "word": self.word,
            "entity_group": self.label,
            "score": 0.9,
            "start": 0,
            "end": len(self.word),
        }]


def make_model(fake, sentence_level=False):
    model = object.__new__(EnglishAspectModel)  # skip loading real weights
    model.pipeline = fake
    model.sentence_level = sentence_level
    return model


def test_short_text_is_one_pass():
    fake = FakePipeline()
    result = make_model(fake).predict("The food is great. The room is clean.")
    assert len(fake.calls) == 1
    assert len(result) == 1


def test_long_text_is_analyzed_sentence_by_sentence():
    fake = FakePipeline()
    text = "The food was great. " * 40          # about 800 characters
    result = make_model(fake).predict(text)
    assert len(fake.calls) == 40
    assert len(result) == 40


def test_long_text_offsets_increase():
    fake = FakePipeline()
    result = make_model(fake).predict("The food was great. " * 40)
    starts = [item["start"] for item in result]
    assert starts == sorted(starts)
    assert starts[1] == len("The food was great.") + 1


def test_sentence_level_flag_forces_sentence_mode_on_short_text():
    fake = FakePipeline()
    make_model(fake, sentence_level=True).predict(
        "The food is great. The room is clean."
    )
    assert len(fake.calls) == 2


def test_oversized_sentence_is_split_under_limit():
    fake = FakePipeline()
    text = " ".join(["word"] * 1500)             # one sentence, no punctuation
    make_model(fake).predict(text)
    assert len(fake.calls) >= 3
    assert all(len(piece) <= EnglishAspectModel.MAX_PIECE_CHARS
               for piece in fake.calls)


def test_punctuation_only_aspects_are_removed():
    result = make_model(FakePipeline(word=".")).predict("Fine.")
    assert result == []


@pytest.mark.parametrize("raw, expected", [
    ("pos", "positive"), ("neg", "negative"), ("neu", "neutral"),
    ("con", "neutral"), ("conflict", "neutral"),
])
def test_label_mapping(raw, expected):
    result = make_model(FakePipeline(label=raw)).predict("The food is great.")
    assert result[0]["sentiment"] == expected


def test_rejects_non_string_and_empty_input():
    model = make_model(FakePipeline())
    with pytest.raises(TypeError):
        model.predict(123)
    with pytest.raises(ValueError):
        model.predict("   ")