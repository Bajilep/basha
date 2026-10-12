from src.basha.sentiment.english_model import (
    EnglishSentimentModel,
)

from src.basha.aspects.contrast import (
    ContrastAnalyzer,
)

from src.basha.fusion.discourse_resolver import (
    DiscourseResolver,
)


sentiment_model = EnglishSentimentModel()

contrast_analyzer = ContrastAnalyzer(
    sentiment_model
)

resolver = DiscourseResolver()


tests = [
    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    (
        "I thought it would be excellent, "
        "but it is actually terrible."
    ),

    (
        "I assumed the camera would be bad, "
        "but in reality it is excellent."
    ),

    (
        "The battery is excellent "
        "but the camera is terrible."
    ),

    (
        "The screen is beautiful "
        "but the speakers are awful."
    ),
]


for text in tests:

    print(
        "\n"
        + "=" * 80
    )

    print(text)

    contrast = (
        contrast_analyzer.analyze(
            text
        )
    )

    result = resolver.resolve(
        contrast
    )

    print(
        "\nContrast:"
    )

    print(
        contrast
    )

    print(
        "\nDiscourse resolution:"
    )

    print(
        result
    )