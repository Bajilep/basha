from src.basha.sentiment.english_model import (
    EnglishSentimentModel,
)

from src.basha.aspects.contrast import (
    ContrastAnalyzer,
)


sentiment_model = (
    EnglishSentimentModel()
)

analyzer = ContrastAnalyzer(
    sentiment_model
)


tests = [

    "The battery is excellent but the camera is terrible.",

    "I love the design, but I hate the battery life.",

    "The performance is great although the software is buggy.",

    "I expected it to be terrible, but it is actually excellent.",

    "The camera is excellent and the battery is great.",

    "The phone is expensive but still excellent.",

]


for text in tests:

    result = analyzer.analyze(
        text
    )

    print("\n" + "=" * 80)

    print(
        "TEXT:",
        text
    )

    print(
        "CONTRAST:",
        result[
            "contrast_detected"
        ]
    )

    print(
        "MARKER:",
        result[
            "marker"
        ]
    )

    print(
        "FINAL LABEL:",
        result[
            "label"
        ]
    )

    print(
        "CONFIDENCE:",
        result[
            "confidence"
        ]
    )

    print(
        "\nORIGINAL WHOLE-SENTENCE:"
    )

    print(
        result[
            "original_sentiment"
        ]
    )


    if result["clauses"]:

        print(
            "\nCLAUSES:"
        )

        for clause in result[
            "clauses"
        ]:

            print(
                f"  {clause['text']}"
            )

            print(
                f"    → "
                f"{clause['label']} "
                f"({clause['confidence']})"
            )