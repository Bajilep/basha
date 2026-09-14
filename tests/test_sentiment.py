"""Tests for the Bajil Sentiment library."""

import pytest

from basha import bashaanalyser


analyzer = bashaanalyser()


# --------------------------------------------------
# Basic sentiment tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "This product is excellent",
        "This is amazing",
        "I love this product",
        "The service was wonderful",
        "This application is useful",
    ],
)
def test_positive_sentences(text):
    result = analyzer.predict(text)

    assert result["label"] == "positive"


@pytest.mark.parametrize(
    "text",
    [
        "This product is terrible",
        "This is awful",
        "I hate this product",
        "The service was horrible",
        "This application is useless",
    ],
)
def test_negative_sentences(text):
    result = analyzer.predict(text)

    assert result["label"] == "negative"


@pytest.mark.parametrize(
    "text",
    [
        "The parcel arrived today",
        "The product is on the table",
        "The meeting starts tomorrow",
    ],
)
def test_neutral_sentences(text):
    result = analyzer.predict(text)

    assert result["label"] == "neutral"


# --------------------------------------------------
# Negation tests
# --------------------------------------------------

def test_not_good():
    result = analyzer.predict(
        "This product is not good"
    )

    assert result["label"] == "negative"


def test_not_bad():
    result = analyzer.predict(
        "This product is not bad"
    )

    assert result["label"] == "positive"


def test_contracted_negation():
    result = analyzer.predict(
        "This product isn't good"
    )

    assert result["label"] == "negative"


def test_curly_apostrophe_negation():
    result = analyzer.predict(
        "This product isn’t good"
    )

    assert result["label"] == "negative"


def test_was_not_bad():
    result = analyzer.predict(
        "The movie wasn't bad"
    )

    assert result["label"] == "positive"


def test_negation_stops_after_but():
    result = analyzer.predict(
        "The camera is not good, "
        "but the battery is excellent"
    )

    assert result["label"] == "positive"


# --------------------------------------------------
# Intensity tests
# --------------------------------------------------

def test_intensifier():
    normal = analyzer.predict("good")
    intense = analyzer.predict("very good")

    assert (
        intense["scores"]["compound"]
        > normal["scores"]["compound"]
    )


def test_strong_intensifier():
    normal = analyzer.predict("good")

    intense = analyzer.predict(
        "extremely good"
    )

    assert (
        intense["scores"]["compound"]
        > normal["scores"]["compound"]
    )


def test_diminisher():
    normal = analyzer.predict("good")

    diminished = analyzer.predict(
        "slightly good"
    )

    assert (
        normal["scores"]["compound"]
        > diminished["scores"]["compound"]
    )


# --------------------------------------------------
# Capital letters and punctuation
# --------------------------------------------------

def test_capital_letters():
    normal = analyzer.predict("great")
    capital = analyzer.predict("GREAT")

    assert (
        capital["scores"]["compound"]
        > normal["scores"]["compound"]
    )


def test_punctuation():
    normal = analyzer.predict("great.")

    emphasized = analyzer.predict(
        "great!!!"
    )

    assert (
        emphasized["scores"]["compound"]
        > normal["scores"]["compound"]
    )


def test_caps_and_punctuation():
    normal = analyzer.predict("great")

    emphasized = analyzer.predict(
        "GREAT!!!"
    )

    assert (
        emphasized["scores"]["compound"]
        > normal["scores"]["compound"]
    )


# --------------------------------------------------
# Contrast tests
# --------------------------------------------------

def test_negative_after_but():
    result = analyzer.predict(
        "The camera is good, "
        "but the battery is terrible"
    )

    assert result["label"] == "negative"


def test_positive_after_but():
    result = analyzer.predict(
        "The delivery was terrible, "
        "but the product is excellent"
    )

    assert result["label"] == "positive"


# --------------------------------------------------
# Phrase tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "This product is highly recommended",
        "This is a must buy",
        "It is worth the money",
        "This product works perfectly",
        "This is value for money",
        "I would recommend this",
    ],
)
def test_positive_phrases(text):
    result = analyzer.predict(text)

    assert result["label"] == "positive"


@pytest.mark.parametrize(
    "text",
    [
        "This is a waste of money",
        "This is not worth it",
        "This product is a piece of junk",
        "The product does not work",
        "I would not recommend this",
        "This was the worst purchase",
    ],
)
def test_negative_phrases(text):
    result = analyzer.predict(text)

    assert result["label"] == "negative"


def test_phrase_appears_in_explanation():
    result = analyzer.predict(
        "This product is highly recommended"
    )

    tokens = [
        item["token"]
        for item in result["explanation"]
    ]

    assert "highly recommended" in tokens


def test_phrase_not_counted_twice():
    result = analyzer.predict(
        "This is highly recommended"
    )

    sentiment_items = [
        item
        for item in result["explanation"]
        if item["base_score"] != 0
    ]

    assert len(sentiment_items) == 1


# --------------------------------------------------
# Context tests
# --------------------------------------------------

def test_positive_sick_context():
    result = analyzer.predict(
        "The movie was sick"
    )

    assert result["label"] == "positive"


def test_negative_sick_context():
    result = analyzer.predict(
        "I feel sick"
    )

    assert result["label"] == "negative"


def test_positive_killer_context():
    result = analyzer.predict(
        "This application has a killer feature"
    )

    assert result["label"] == "positive"


def test_negative_killer_context():
    result = analyzer.predict(
        "The killer was arrested by police"
    )

    assert result["label"] == "negative"


def test_positive_fire_context():
    result = analyzer.predict(
        "This song is fire"
    )

    assert result["label"] == "positive"


def test_negative_fire_context():
    result = analyzer.predict(
        "The house is on fire"
    )

    assert result["label"] == "negative"


# --------------------------------------------------
# Emoji and emoticon tests
# --------------------------------------------------

def test_positive_emoji():
    result = analyzer.predict(
        "I love this phone 😍"
    )

    assert result["label"] == "positive"


def test_positive_emoji_only():
    result = analyzer.predict("😍")

    assert result["label"] == "positive"


def test_negative_emoji():
    result = analyzer.predict(
        "I hate this product 😡"
    )

    assert result["label"] == "negative"


def test_positive_emoticon():
    result = analyzer.predict(
        "This is good :)"
    )

    assert result["label"] == "positive"


def test_negative_emoticon():
    result = analyzer.predict(
        "This is bad :("
    )

    assert result["label"] == "negative"


def test_repeated_emoji():
    single = analyzer.predict("😍")
    repeated = analyzer.predict("😍😍")

    assert (
        repeated["scores"]["compound"]
        > single["scores"]["compound"]
    )


# --------------------------------------------------
# Spelling tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "incorrect_word",
    [
        "graet",
        "excelent",
        "amazng",
        "awsome",
        "fantstic",
    ],
)
def test_positive_spelling_correction(
    incorrect_word
):
    result = analyzer.predict(
        f"This product is {incorrect_word}"
    )

    assert result["label"] == "positive"


@pytest.mark.parametrize(
    "incorrect_word",
    [
        "terible",
        "horrble",
        "dissapointed",
        "useess",
    ],
)
def test_negative_spelling_correction(
    incorrect_word
):
    result = analyzer.predict(
        f"This product is {incorrect_word}"
    )

    assert result["label"] == "negative"


def test_repeated_letter_correction():
    result = analyzer.predict(
        "This product is goooood"
    )

    assert result["label"] == "positive"


def test_spelling_rule_in_explanation():
    result = analyzer.predict(
        "This product is graet"
    )

    assert any(
        "spelling" in rule
        for item in result["explanation"]
        for rule in item["rules"]
    )


def test_unknown_word_not_forcefully_corrected():
    result = analyzer.predict(
        "The product is zxqvbn"
    )

    assert result["label"] == "neutral"


# --------------------------------------------------
# Sarcasm tests
# --------------------------------------------------

def test_sarcasm_with_negative_event():
    result = analyzer.predict(
        "Great! The application crashed again."
    )

    assert result["label"] == "negative"
    assert result["flags"]["sarcasm"] is True


def test_explicit_sarcasm_marker():
    result = analyzer.predict(
        "Excellent service, yeah right."
    )

    assert result["label"] == "negative"
    assert result["flags"]["sarcasm"] is True


def test_sarcasm_emoji():
    result = analyzer.predict(
        "Wonderful service 🙄"
    )

    assert result["label"] == "negative"
    assert result["flags"]["sarcasm"] is True


def test_sarcasm_confidence():
    result = analyzer.predict(
        "Great! The application crashed again."
    )

    confidence = result["flags"][
        "sarcasm_confidence"
    ]

    assert 0 <= confidence <= 1
    assert confidence >= 0.60


def test_normal_positive_not_sarcastic():
    result = analyzer.predict(
        "This is genuinely excellent."
    )

    assert result["label"] == "positive"
    assert result["flags"]["sarcasm"] is False


def test_great_job_not_always_sarcastic():
    result = analyzer.predict(
        "Great job on the presentation"
    )

    assert result["label"] == "positive"
    assert result["flags"]["sarcasm"] is False


def test_totally_not_always_sarcastic():
    result = analyzer.predict(
        "I totally love this product"
    )

    assert result["label"] == "positive"
    assert result["flags"]["sarcasm"] is False


# --------------------------------------------------
# Output-format tests
# --------------------------------------------------

def test_standard_output_keys():
    result = analyzer.predict(
        "This is excellent"
    )

    assert set(result) == {
        "text",
        "label",
        "emotion",
        "intensity",
        "emoji",
        "coverage",
        "scores",
        "flags",
        "explanation",
    }


def test_score_keys():
    result = analyzer.predict(
        "This is excellent"
    )

    assert set(result["scores"]) == {
        "compound",
        "positive",
        "negative",
        "neutral",
    }


def test_flag_keys():
    result = analyzer.predict(
        "This is excellent"
    )

    assert "sarcasm" in result["flags"]
    assert "sarcasm_confidence" in result["flags"]
    assert "sarcasm_reasons" in result["flags"]


def test_explanation_keys():
    result = analyzer.predict(
        "This product is very good"
    )

    first_item = result["explanation"][0]

    assert set(first_item) == {
        "token",
        "base_score",
        "final_score",
        "rules",
    }


def test_compound_range():
    result = analyzer.predict(
        "This is absolutely amazing!!!"
    )

    compound = result["scores"]["compound"]

    assert -1 <= compound <= 1


def test_score_proportions():
    result = analyzer.predict(
        "The camera is good but delivery is bad"
    )

    scores = result["scores"]

    total = (
        scores["positive"]
        + scores["negative"]
        + scores["neutral"]
    )

    assert abs(total - 1.0) < 0.01


def test_original_text_returned():
    text = "This product is excellent"

    result = analyzer.predict(text)

    assert result["text"] == text


# --------------------------------------------------
# Custom-word tests
# --------------------------------------------------

def test_custom_word():
    custom_analyzer = bashaanalyser()

    custom_analyzer.add_word(
        "supercalifragilistic",
        3.5,
    )

    result = custom_analyzer.predict(
        "This is supercalifragilistic"
    )

    assert result["label"] == "positive"


def test_custom_word_does_not_change_other_analyzer():
    first_analyzer = bashaanalyser()
    second_analyzer = bashaanalyser()

    first_analyzer.add_word(
        "custompositiveword",
        3.0,
    )

    first_result = first_analyzer.predict(
        "custompositiveword"
    )

    second_result = second_analyzer.predict(
        "custompositiveword"
    )

    assert first_result["label"] == "positive"
    assert second_result["label"] == "neutral"


def test_custom_word_invalid_score():
    with pytest.raises(ValueError):
        analyzer.add_word(
            "example",
            5.0,
        )


def test_custom_word_zero_score():
    with pytest.raises(ValueError):
        analyzer.add_word(
            "example",
            0,
        )


# --------------------------------------------------
# Batch tests
# --------------------------------------------------

def test_batch():
    results = analyzer.predict_batch([
        "This is excellent",
        "This is terrible",
        "The parcel arrived",
    ])

    assert len(results) == 3
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "neutral"


def test_invalid_batch_input():
    with pytest.raises(TypeError):
        analyzer.predict_batch(
            "This is not a collection"
        )


# --------------------------------------------------
# Input-validation tests
# --------------------------------------------------

def test_empty_text():
    with pytest.raises(ValueError):
        analyzer.predict("")


def test_whitespace_text():
    with pytest.raises(ValueError):
        analyzer.predict("   ")


def test_non_string_input():
    with pytest.raises(TypeError):
        analyzer.predict(123)


def test_none_input():
    with pytest.raises(TypeError):
        analyzer.predict(None)

def test_what_a_product_phrase():
    result = analyzer.predict(
        "What a product!"
    )

    assert result["label"] == "positive"

    assert any(
        item["token"] == "what a product"
        for item in result["explanation"]
    )


def test_informal_product_expression():
    result = analyzer.predict(
        "Ooww, what a product...?"
    )

    assert result["label"] == "positive"

def test_joy_emotion():
    result = analyzer.predict(
        "I am extremely happy"
    )

    assert result["emotion"]["label"] == "joy"


def test_sadness_emotion():
    result = analyzer.predict(
        "I feel very sad and lonely"
    )

    assert result["emotion"]["label"] == "sadness"


def test_anger_emotion():
    result = analyzer.predict(
        "I am absolutely furious"
    )

    assert result["emotion"]["label"] == "anger"


def test_fear_emotion():
    result = analyzer.predict(
        "I am scared and worried"
    )

    assert result["emotion"]["label"] == "fear"


def test_surprise_emotion():
    result = analyzer.predict(
        "I am completely shocked"
    )

    assert result["emotion"]["label"] == "surprise"


def test_disgust_emotion():
    result = analyzer.predict(
        "This food is absolutely disgusting"
    )

    assert result["emotion"]["label"] == "disgust"


def test_neutral_emotion():
    result = analyzer.predict(
        "The parcel arrived today"
    )

    assert result["emotion"]["label"] == "neutral"


def test_emotion_confidence_range():
    result = analyzer.predict(
        "I am very happy"
    )

    confidence = result["emotion"]["confidence"]

    assert 0 <= confidence <= 1


def test_emotion_scores_present():
    result = analyzer.predict(
        "I am happy"
    )

    assert "joy" in result["emotion"]["scores"]
    assert "sadness" in result["emotion"]["scores"]
    assert "anger" in result["emotion"]["scores"]
    assert "fear" in result["emotion"]["scores"]
    assert "surprise" in result["emotion"]["scores"]
    assert "disgust" in result["emotion"]["scores"]

def test_negation_stops_at_full_stop():
    result = analyzer.predict(
        "This is not good. Excellent design."
    )

    assert result["label"] == "positive"


def test_sarcastic_joy_becomes_uncertain():
    result = analyzer.predict(
        "Excellent service, yeah right."
    )

    assert result["label"] == "negative"
    assert result["flags"]["sarcasm"] is True
    assert result["emotion"]["label"] == "uncertain"
    assert (
        result["emotion"]["sarcasm_adjusted"]
        is True
    )

def test_social_luv():
    result = analyzer.predict(
        "I luv this product"
    )

    assert result["label"] == "positive"


def test_social_gr8():
    result = analyzer.predict(
        "This product is gr8"
    )

    assert result["label"] == "positive"


def test_positive_hashtag():
    result = analyzer.predict(
        "#AmazingProduct"
    )

    assert result["label"] == "positive"


def test_negative_hashtag():
    result = analyzer.predict(
        "#WorstProduct"
    )

    assert result["label"] == "negative"


def test_mention_is_ignored():
    result = analyzer.predict(
        "@company this is excellent"
    )

    assert result["label"] == "positive"


def test_url_is_ignored():
    result = analyzer.predict(
        "Visit https://example.com today"
    )

    assert result["label"] == "neutral"


def test_social_changes_returned():
    result = analyzer.predict(
        "I luv this #AmazingProduct"
    )

    assert (
        len(
            result["flags"][
                "social_normalizations"
            ]
        )
        > 0
    )


def test_complete_waste_phrase():
    result = analyzer.predict(
        "This is a complete waste of money"
    )

    assert result["label"] == "negative"


def test_my_money_phrase():
    result = analyzer.predict(
        "This is a waste of my money"
    )

    assert result["label"] == "negative"


def test_sharp_image_context():
    result = analyzer.predict(
        "The camera produces a sharp image"
    )

    assert result["label"] == "positive"


def test_sharp_pain_context():
    result = analyzer.predict(
        "I have a sharp pain"
    )

    assert result["label"] == "negative"


def test_hot_design_context():
    result = analyzer.predict(
        "This is a hot design"
    )

    assert result["label"] == "positive"


def test_hot_weather_context():
    result = analyzer.predict(
        "The weather is hot"
    )

    assert result["label"] == "neutral"


def test_sweet_person_context():
    result = analyzer.predict(
        "She is a sweet person"
    )

    assert result["label"] == "positive"


def test_sweet_food_context():
    result = analyzer.predict(
        "This food is sweet"
    )

    assert result["label"] == "neutral"


# --------------------------------------------------
# Master-merge features
# --------------------------------------------------

def test_master_intensity_metadata():
    result = analyzer.predict("WOW!!! This is very good")
    assert result["intensity"]["score"] > 0.5
    assert result["intensity"]["factors"]["exclamations"] == 3


def test_master_emoji_combination():
    result = analyzer.predict("This product is amazing 😍🔥")
    combinations = result["emoji"]["combinations"]
    assert combinations[0]["sequence"] == "😍🔥"


def test_master_mixed_polarity():
    result = analyzer.predict("Great camera but terrible battery")
    assert result["flags"]["mixed_polarity"] is True


def test_master_coverage_information():
    result = analyzer.predict("excellent flibbertigibbet")
    assert "excellent" in result["coverage"]["known_words"]
    assert "flibbertigibbet" in result["coverage"]["unknown_words"]


def test_master_convenience_methods():
    assert analyzer.sentiment("excellent") == "positive"
    assert analyzer.polarity("terrible") < 0
    assert analyzer.analyze("excellent")["label"] == "positive"
