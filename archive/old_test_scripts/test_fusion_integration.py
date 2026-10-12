from pprint import pprint

from src.basha.sentiment.english_model import (
    EnglishSentimentModel,
)

from src.basha.aspects.aspect_model import (
    EnglishAspectModel,
)

from src.basha.aspects.contrast import (
    ContrastAnalyzer,
)

from src.basha.longtext.splitter import (
    split_sentences,
)

from src.basha.fusion.mixed_detector import (
    MixedDetector,
)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 80)
print("LOADING BASHA COMPONENTS")
print("=" * 80)


sentiment_model = (
    EnglishSentimentModel()
)

aspect_model = (
    EnglishAspectModel()
)

contrast_analyzer = (
    ContrastAnalyzer(
        sentiment_model
    )
)

mixed_detector = (
    MixedDetector()
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_text(
    text: str,
):

    print(
        "\n"
        + "=" * 80
    )

    print("TEXT:")
    print(text)

    print("=" * 80)

    # --------------------------------------------------------
    # 1. Overall sentiment
    # --------------------------------------------------------

    overall = (
        sentiment_model.predict(
            text
        )
    )

    # --------------------------------------------------------
    # 2. Sentence splitting
    # --------------------------------------------------------

    sentence_texts = (
        split_sentences(
            text
        )
    )

    # --------------------------------------------------------
    # 3. Sentence sentiment
    # --------------------------------------------------------

    sentence_results = []

    for sentence in sentence_texts:

        result = (
            sentiment_model.predict(
                sentence
            )
        )

        sentence_results.append(
            {
                "text": sentence,
                **result,
            }
        )

    # --------------------------------------------------------
    # 4. Aspect sentiment
    # --------------------------------------------------------

    aspects = (
        aspect_model.predict(
            text
        )
    )

    # --------------------------------------------------------
    # 5. Contrast
    # --------------------------------------------------------

    contrast = (
        contrast_analyzer.analyze(
            text
        )
    )

    # --------------------------------------------------------
    # 6. Mixed detection
    # --------------------------------------------------------

    mixed = (
        mixed_detector.detect(
            aspects=aspects,
            sentences=sentence_results,
            contrast=contrast,
        )
    )

    # --------------------------------------------------------
    # 7. Temporary final label
    # --------------------------------------------------------

    if mixed["mixed"]:

        final_label = "mixed"

        final_confidence = (
            mixed["confidence"]
        )

    else:

        final_label = (
            overall["label"]
        )

        final_confidence = (
            overall["confidence"]
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_result = {
        "text": text,

        "final": {
            "label":
                final_label,

            "confidence":
                final_confidence,
        },

        "overall_sentiment":
            overall,

        "sentences":
            sentence_results,

        "aspects":
            aspects,

        "contrast":
            contrast,

        "mixed_analysis":
            mixed,
    }

    pprint(
        final_result,
        sort_dicts=False,
    )


# ============================================================
# TEST CASES
# ============================================================

TESTS = [

    # --------------------------------------------------------
    # Simple positive
    # --------------------------------------------------------

    "The camera is excellent.",

    # --------------------------------------------------------
    # Simple negative
    # --------------------------------------------------------

    "The camera is terrible.",

    # --------------------------------------------------------
    # Same-sentence mixed aspects
    # --------------------------------------------------------

    (
        "The battery is excellent "
        "but the camera is terrible."
    ),

    # --------------------------------------------------------
    # Multiple positive aspects
    # --------------------------------------------------------

    (
        "The camera is excellent "
        "and the battery is great."
    ),

    # --------------------------------------------------------
    # Opposite sentiment across sentences
    # --------------------------------------------------------

    (
        "The display is beautiful. "
        "The battery lasts all day. "
        "The camera is terrible. "
        "The software crashes constantly."
    ),

    # --------------------------------------------------------
    # IMPORTANT:
    # expectation → actual outcome
    #
    # We do NOT automatically want this to become Mixed.
    # --------------------------------------------------------

    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    # --------------------------------------------------------
    # Neutral factual statement
    # --------------------------------------------------------

    "The phone has a 6.5 inch display.",

    # --------------------------------------------------------
    # Larger realistic mixed review
    # --------------------------------------------------------

    (
        "The display is fantastic and "
        "the battery lasts all day. "
        "However, the camera performs poorly "
        "at night and the software crashes "
        "frequently."
    ),
]


# ============================================================
# RUN
# ============================================================

for text in TESTS:

    analyze_text(
        text
    )