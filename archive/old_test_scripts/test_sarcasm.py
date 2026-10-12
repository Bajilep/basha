from pprint import pprint

from src.basha.sarcasm.english_sarcasm import (
    EnglishSarcasmModel,
)


model = EnglishSarcasmModel()


tests = [

    # --------------------------------------------------------
    # Normal positive
    # --------------------------------------------------------

    "I absolutely love this phone.",

    # --------------------------------------------------------
    # Normal negative
    # --------------------------------------------------------

    "This phone is terrible.",

    # --------------------------------------------------------
    # Obvious sarcastic praise
    # --------------------------------------------------------

    (
        "Great, another software crash. "
        "Exactly what I needed."
    ),

    # --------------------------------------------------------
    # Sarcastic positive wording
    # --------------------------------------------------------

    (
        "Wonderful, the battery died "
        "again after two hours."
    ),

    # --------------------------------------------------------
    # Another sarcastic expression
    # --------------------------------------------------------

    (
        "Oh fantastic, the app deleted "
        "all my files."
    ),

    # --------------------------------------------------------
    # Genuine praise
    # --------------------------------------------------------

    (
        "Fantastic! The camera quality "
        "is excellent."
    ),

    # --------------------------------------------------------
    # Mixed, but NOT necessarily sarcasm
    # --------------------------------------------------------

    (
        "The display is excellent, "
        "but the camera is terrible."
    ),

    # --------------------------------------------------------
    # Expectation vs actual
    # --------------------------------------------------------

    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    # --------------------------------------------------------
    # Factual / neutral
    # --------------------------------------------------------

    "The phone has a 6.5 inch display.",

    # --------------------------------------------------------
    # Common sarcastic structure
    # --------------------------------------------------------

    (
        "Yeah, because crashing every "
        "five minutes is exactly what "
        "a good app should do."
    ),
]


for text in tests:

    print(
        "\n"
        + "=" * 80
    )

    print(text)

    print("=" * 80)

    result = model.predict(
        text
    )

    pprint(
        result,
        sort_dicts=False,
    )