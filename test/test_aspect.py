from src.basha.aspects.aspect_model import (
    EnglishAspectModel,
)


model = EnglishAspectModel()


tests = [

    # ------------------------------------------------
    # Clear mixed aspects
    # ------------------------------------------------

    (
        "The battery is excellent "
        "but the camera is terrible."
    ),

    (
        "I love the design, "
        "but I hate the battery life."
    ),

    (
        "The performance is great "
        "although the software is buggy."
    ),

    # ------------------------------------------------
    # Multiple positive aspects
    # ------------------------------------------------

    (
        "The camera is excellent "
        "and the battery is great."
    ),

    # ------------------------------------------------
    # Multiple aspects
    # ------------------------------------------------

    (
        "The screen is beautiful, "
        "the processor is fast, "
        "but the speakers are awful."
    ),

    # ------------------------------------------------
    # Neutral / factual
    # ------------------------------------------------

    "The phone has a 6.5 inch display.",

    # ------------------------------------------------
    # Expectation case
    # ------------------------------------------------

    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    # ------------------------------------------------
    # Longer review
    # ------------------------------------------------

    (
        "The display is fantastic and "
        "the battery lasts all day. "
        "However, the camera performs "
        "poorly at night and the software "
        "crashes frequently."
    ),
]


for number, text in enumerate(
    tests,
    start=1,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TEST {number}"
    )

    print(
        "=" * 80
    )

    print(
        "TEXT:",
        text,
    )

    aspects = model.predict(
        text
    )

    print(
        "\nASPECTS:"
    )

    if not aspects:

        print(
            "  No aspects detected."
        )

        continue

    for aspect in aspects:

        print(
            f"  Aspect:     "
            f"{aspect['aspect']}"
        )

        print(
            f"  Sentiment:  "
            f"{aspect['sentiment']}"
        )

        print(
            f"  Confidence: "
            f"{aspect['confidence']}"
        )

        print(
            f"  Position:   "
            f"{aspect['start']}"
            f"-"
            f"{aspect['end']}"
        )

        print()