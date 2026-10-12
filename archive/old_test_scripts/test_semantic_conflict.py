from pprint import pprint

from src.basha.semantics.semantic_conflict import (
    SemanticConflictDetector,
)


detector = SemanticConflictDetector()


tests = [

    # ==============================================
    # Sarcastic contradiction
    # ==============================================

    (
        "Laptop crash",
        (
            "Perfect, my laptop just crashed "
            "before I saved my work."
        ),
    ),

    (
        "Battery died",
        (
            "Wonderful, the battery died "
            "again after two hours."
        ),
    ),

    (
        "Software crash",
        (
            "Great, the software crashed again."
        ),
    ),

    (
        "Deleted files",
        (
            "Fantastic, the app deleted "
            "all my files."
        ),
    ),

    # ==============================================
    # Normal positive
    # Must NOT become conflict
    # ==============================================

    (
        "Normal camera",
        "The camera is excellent.",
    ),

    (
        "Normal battery",
        (
            "The battery performance "
            "is excellent."
        ),
    ),

    (
        "Normal display",
        "The display is beautiful.",
    ),

    # ==============================================
    # Normal negative
    # No positive evaluation word
    # ==============================================

    (
        "Bad camera",
        "The camera is terrible.",
    ),

    # ==============================================
    # Difficult contextual sarcasm
    # ==============================================

    (
        "Two-hour battery",
        (
            "The battery lasted an entire "
            "two hours. Impressive."
        ),
    ),

    (
        "Slow support",
        (
            "Customer support replied after "
            "five days. Very helpful."
        ),
    ),
]


for name, text in tests:

    print(
        "\n"
        + "=" * 90
    )

    print(name)

    print(text)

    print(
        "=" * 90
    )

    result = detector.analyze(
        text
    )

    pprint(
        result,
        sort_dicts=False,
    )