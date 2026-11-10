from pprint import pprint

from basha.fusion.sarcasm_resolver import (
    SarcasmResolver,
)


resolver = SarcasmResolver()


def run_test(
    name,
    sarcasm,
    overall,
    aspects,
    sentences,
    discourse=None,
):

    if discourse is None:
        discourse = {
            "resolved": False
        }

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    result = resolver.resolve(
        sarcasm=sarcasm,
        overall=overall,
        aspects=aspects,
        sentences=sentences,
        mixed={},
        discourse=discourse,
    )

    pprint(
        result,
        sort_dicts=False,
    )


# ============================================================
# TEST 1
#
# Sarcasm + negative aspect
# Should resolve NEGATIVE.
# ============================================================

run_test(

    "Sarcasm + negative aspect",

    sarcasm={
        "detected": True,
        "sarcasm": 0.9803,
    },

    overall={
        "label": "positive",
        "positive": 0.4905,
        "neutral": 0.2576,
        "negative": 0.2519,
    },

    aspects=[
        {
            "aspect": "battery",
            "sentiment": "negative",
            "confidence": 0.5216,
        }
    ],

    sentences=[
        {
            "text":
                "Wonderful, the battery died "
                "again after two hours.",

            "label": "positive",
            "confidence": 0.4905,
        }
    ],
)


# ============================================================
# TEST 2
#
# Sarcasm + negative sentence
# Should resolve NEGATIVE.
# ============================================================

run_test(

    "Sarcasm + negative sentence",

    sarcasm={
        "detected": True,
        "sarcasm": 0.9848,
    },

    overall={
        "label": "negative",
        "positive": 0.0628,
        "neutral": 0.1121,
        "negative": 0.8251,
    },

    aspects=[],

    sentences=[
        {
            "text":
                "Great, another software crash.",

            "label": "negative",
            "confidence": 0.8726,
        },

        {
            "text":
                "Exactly what I needed.",

            "label": "positive",
            "confidence": 0.9486,
        },
    ],
)


# ============================================================
# TEST 3
#
# Strong sarcasm but NO negative evidence.
#
# Must NOT blindly flip sentiment.
# ============================================================

run_test(

    "Sarcasm without negative evidence",

    sarcasm={
        "detected": True,
        "sarcasm": 0.95,
    },

    overall={
        "label": "positive",
        "positive": 0.90,
        "neutral": 0.07,
        "negative": 0.03,
    },

    aspects=[
        {
            "aspect": "camera",
            "sentiment": "positive",
            "confidence": 0.90,
        }
    ],

    sentences=[
        {
            "text":
                "The camera is excellent.",

            "label": "positive",
            "confidence": 0.90,
        }
    ],
)


# ============================================================
# TEST 4
#
# Normal negative review.
#
# Sarcasm FALSE.
# Must NOT resolve.
# ============================================================

run_test(

    "Normal negative review",

    sarcasm={
        "detected": False,
        "sarcasm": 0.05,
    },

    overall={
        "label": "negative",
        "positive": 0.02,
        "neutral": 0.06,
        "negative": 0.92,
    },

    aspects=[
        {
            "aspect": "camera",
            "sentiment": "negative",
            "confidence": 0.81,
        }
    ],

    sentences=[
        {
            "text":
                "The camera is terrible.",

            "label": "negative",
            "confidence": 0.92,
        }
    ],
)


# ============================================================
# TEST 5
#
# Discourse already solved the sentence.
#
# Sarcasm resolver should leave it alone.
# ============================================================

run_test(

    "Discourse has priority",

    sarcasm={
        "detected": True,
        "sarcasm": 0.90,
    },

    overall={
        "label": "negative",
        "positive": 0.28,
        "neutral": 0.19,
        "negative": 0.53,
    },

    aspects=[],

    sentences=[],

    discourse={
        "resolved": True,
        "label": "positive",
        "confidence": 0.97,
        "reason":
            "expectation_vs_actual",
    },
)