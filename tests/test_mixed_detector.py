from basha.fusion.mixed_detector import MixedDetector


def aspect(name, sentiment, confidence):
    return {"aspect": name, "sentiment": sentiment, "confidence": confidence}


def sentence(text, label, confidence):
    return {"text": text, "label": label, "confidence": confidence}


def test_opposite_aspects_are_mixed():
    result = MixedDetector().detect(
        aspects=[aspect("battery", "positive", 0.91),
                 aspect("camera", "negative", 0.93)],
        sentences=[],
        contrast=None,
    )
    assert result["mixed"] is True
    assert "opposite_aspect_sentiments" in result["reasons"]


def test_only_positive_aspects_are_not_mixed():
    result = MixedDetector().detect(
        aspects=[aspect("camera", "positive", 0.90),
                 aspect("battery", "positive", 0.88)],
        sentences=[],
        contrast=None,
    )
    assert result["mixed"] is False
    assert result["confidence"] == 0.0


def test_opposite_sentences_are_mixed():
    result = MixedDetector().detect(
        aspects=[],
        sentences=[sentence("The display is beautiful.", "positive", 0.96),
                   sentence("The speaker is awful.", "negative", 0.95)],
        contrast=None,
    )
    assert result["mixed"] is True
    assert "opposite_sentence_sentiments" in result["reasons"]


def test_low_confidence_aspect_is_ignored():
    # 0.30 is below the 0.50 aspect threshold, so no positive evidence.
    result = MixedDetector().detect(
        aspects=[aspect("battery", "positive", 0.30),
                 aspect("camera", "negative", 0.93)],
        sentences=[],
        contrast=None,
    )
    assert result["mixed"] is False


def test_empty_input_is_not_mixed():
    result = MixedDetector().detect(aspects=[], sentences=[], contrast=None)
    assert result["mixed"] is False