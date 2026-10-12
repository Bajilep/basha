from pprint import pprint

from src.basha.semantics.nli_conflict import (
    NLIConflictDetector,
)


nli = NLIConflictDetector()


tests = [

    # ==================================================
    # Sarcastic situations
    # ==================================================

    (
        "Laptop crash + perfect",
        (
            "My laptop crashed before "
            "I saved my work."
        ),
        "This was a perfect experience.",
    ),

    (
        "Battery died + wonderful",
        (
            "The battery died again "
            "after two hours."
        ),
        "This was wonderful.",
    ),

    (
        "Software crash + great",
        "The software crashed again.",
        "This was great.",
    ),

    (
        "Deleted files + fantastic",
        "The app deleted all my files.",
        "This was fantastic.",
    ),

    (
        "Broken phone + best",
        (
            "The phone breaks after "
            "two days."
        ),
        "This is the best phone ever.",
    ),

    # ==================================================
    # Normal positive agreement
    # ==================================================

    (
        "Excellent camera",
        (
            "The camera produces "
            "excellent photos."
        ),
        "The camera is excellent.",
    ),

    (
        "Good battery",
        (
            "The battery lasts all day."
        ),
        "The battery performance is good.",
    ),

    # ==================================================
    # Normal negative agreement
    # ==================================================

    (
        "Bad camera",
        (
            "The camera produces "
            "very poor photos."
        ),
        "The camera is terrible.",
    ),

    # ==================================================
    # Neutral/factual
    # ==================================================

    (
        "Display fact",
        (
            "The phone has a "
            "6.5 inch display."
        ),
        "The phone has a display.",
    ),

    # ==================================================
    # IMPORTANT difficult examples
    # ==================================================

    (
        "Two hour battery + impressive",
        (
            "The battery lasted "
            "only two hours."
        ),
        "The battery performance was impressive.",
    ),

    (
        "Five day support + helpful",
        (
            "Customer support took "
            "five days to reply."
        ),
        "Customer support was very helpful.",
    ),
]


for name, premise, hypothesis in tests:

    print(
        "\n"
        + "=" * 90
    )

    print(name)

    print(
        "=" * 90
    )

    print(
        "\nPREMISE:"
    )

    print(
        premise
    )

    print(
        "\nHYPOTHESIS:"
    )

    print(
        hypothesis
    )

    result = nli.predict(
        premise=premise,
        hypothesis=hypothesis,
    )

    print(
        "\nNLI:"
    )

    pprint(
        result,
        sort_dicts=False,
    )