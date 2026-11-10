from basha.fusion.mixed_detector import (
    MixedDetector,
)


detector = MixedDetector()


tests = [
    {
        "name": "Opposite aspects",

        "aspects": [
            {
                "aspect": "battery",
                "sentiment": "positive",
                "confidence": 0.91,
            },
            {
                "aspect": "camera",
                "sentiment": "negative",
                "confidence": 0.93,
            },
        ],

        "sentences": [],
        "contrast": None,
    },

    {
        "name": "Only positive aspects",

        "aspects": [
            {
                "aspect": "camera",
                "sentiment": "positive",
                "confidence": 0.90,
            },
            {
                "aspect": "battery",
                "sentiment": "positive",
                "confidence": 0.88,
            },
        ],

        "sentences": [],
        "contrast": None,
    },

    {
        "name": "Opposite sentences",

        "aspects": [],

        "sentences": [
            {
                "text":
                    "The display is beautiful.",
                "label": "positive",
                "confidence": 0.96,
            },
            {
                "text":
                    "The camera is terrible.",
                "label": "negative",
                "confidence": 0.91,
            },
        ],

        "contrast": None,
    },

    {
        "name": "Weak opposite aspect",

        "aspects": [
            {
                "aspect": "battery",
                "sentiment": "positive",
                "confidence": 0.92,
            },
            {
                "aspect": "camera",
                "sentiment": "negative",
                "confidence": 0.21,
            },
        ],

        "sentences": [],
        "contrast": None,
    },
]


for test in tests:

    print(
        "\n"
        + "=" * 80
    )

    print(
        test["name"]
    )

    print(
        "=" * 80
    )

    result = detector.detect(
        aspects=test["aspects"],
        sentences=test["sentences"],
        contrast=test["contrast"],
    )

    print(
        result
    )