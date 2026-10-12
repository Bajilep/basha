from pprint import pprint

from src.basha.fusion.english_fusion import (
    EnglishFusion,
)


basha = EnglishFusion()


tests = [

    # --------------------------------------------------
    # NORMAL
    # --------------------------------------------------

    (
        "Normal positive",
        "The camera is excellent.",
        "positive",
    ),

    (
        "Normal negative",
        "The camera is terrible.",
        "negative",
    ),

    (
        "Factual neutral",
        "The phone has a 6.5 inch display.",
        "neutral",
    ),

    # --------------------------------------------------
    # MIXED
    # --------------------------------------------------

    (
        "Genuine mixed",
        (
            "The battery is excellent "
            "but the camera is terrible."
        ),
        "mixed",
    ),

    (
        "Long mixed",
        (
            "The display is fantastic and "
            "the battery lasts all day. "
            "However, the camera performs "
            "poorly at night and the "
            "software crashes frequently."
        ),
        "mixed",
    ),

    # --------------------------------------------------
    # DISCOURSE
    # --------------------------------------------------

    (
        "Expectation becomes positive",
        (
            "I expected it to be terrible, "
            "but it is actually excellent."
        ),
        "positive",
    ),

    (
        "Expectation becomes negative",
        (
            "I thought it would be excellent, "
            "but it is actually terrible."
        ),
        "negative",
    ),

    # --------------------------------------------------
    # SARCASM
    # --------------------------------------------------

    (
        "Sarcastic crash",
        (
            "Great, another software crash. "
            "Exactly what I needed."
        ),
        "negative",
    ),

    (
        "Sarcastic battery",
        (
            "Wonderful, the battery died "
            "again after two hours."
        ),
        "negative",
    ),

    (
        "Sarcastic laptop",
        (
            "Perfect, my laptop just crashed "
            "before I saved my work."
        ),
        "negative",
    ),

    # --------------------------------------------------
    # Known difficult cases
    #
    # Don't enforce expected labels yet.
    # We want to inspect these.
    # --------------------------------------------------

    (
        "Borderline sarcasm",
        (
            "The battery lasted an entire "
            "two hours. Impressive."
        ),
        None,
    ),

    (
        "Missed sarcasm",
        (
            "Customer support replied after "
            "five days. Very helpful."
        ),
        None,
    ),
]


correct = 0
scored = 0


for name, text, expected in tests:

    print(
        "\n"
        + "=" * 100
    )

    print(
        name
    )

    print(
        text
    )

    print(
        "=" * 100
    )

    result = basha.analyze(
        text
    )

    final = result[
        "sentiment"
    ]

    print(
        "\nFINAL:"
    )

    pprint(
        final,
        sort_dicts=False,
    )

    print(
        "\nSOURCE:"
    )

    print(
        result[
            "decision_source"
        ]
    )

    print(
        "\nSARCASM:"
    )

    pprint(
        result[
            "sarcasm"
        ],
        sort_dicts=False,
    )

    print(
        "\nSARCASM RESOLUTION:"
    )

    pprint(
        result[
            "sarcasm_resolution"
        ],
        sort_dicts=False,
    )

    print(
        "\nASPECTS:"
    )

    pprint(
        result[
            "aspects"
        ],
        sort_dicts=False,
    )

    print(
        "\nMIXED:"
    )

    pprint(
        result[
            "mixed_analysis"
        ],
        sort_dicts=False,
    )

    print(
        "\nDISCOURSE:"
    )

    pprint(
        result[
            "discourse"
        ],
        sort_dicts=False,
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    if expected is not None:

        scored += 1

        predicted = (
            final["label"]
        )

        passed = (
            predicted
            == expected
        )

        if passed:
            correct += 1

        print(
            "\nEXPECTED:",
            expected,
        )

        print(
            "RESULT:",
            "PASS"
            if passed
            else "FAIL",
        )

    else:

        print(
            "\nEXPECTED:"
            " inspection only"
        )


print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY"
)

print(
    "=" * 100
)

print(
    f"\nScored tests: "
    f"{scored}"
)

print(
    f"Passed: "
    f"{correct}/{scored}"
)

if scored:

    print(
        f"Accuracy: "
        f"{correct / scored * 100:.2f}%"
    )