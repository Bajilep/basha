from pprint import pprint

from basha.fusion.english_fusion import (
    EnglishFusion,
)


basha = EnglishFusion()


tests = [

    # Normal positive
    "The camera is excellent.",

    # Normal negative
    "The camera is terrible.",

    # Genuine mixed
    (
        "The battery is excellent "
        "but the camera is terrible."
    ),

    # Expectation -> actual
    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    # Obvious sarcasm
    (
        "Great, another software crash. "
        "Exactly what I needed."
    ),

    # Sarcastic positive wording
    (
        "Wonderful, the battery died "
        "again after two hours."
    ),

    # Sarcasm with failure
    (
        "Perfect, my laptop just crashed "
        "before I saved my work."
    ),

    # Strong sarcasm
    (
        "I just love waiting three hours "
        "for customer support."
    ),

    # More difficult sarcasm
    (
        "The battery lasted an entire "
        "two hours. Impressive."
    ),

    # Known difficult BESSTIE example
    (
        "Customer support replied after "
        "five days. Very helpful."
    ),

    # Factual
    "The phone has a 6.5 inch display.",

    # Longer mixed review
    (
        "The display is fantastic and the "
        "battery lasts all day. However, "
        "the camera performs poorly at night "
        "and the software crashes frequently."
    ),
]


for text in tests:

    print(
        "\n"
        + "=" * 100
    )

    print(text)

    print(
        "=" * 100
    )

    result = basha.analyze(
        text
    )

    print(
        "\nFINAL SENTIMENT:"
    )

    pprint(
        result["sentiment"],
        sort_dicts=False,
    )

    print(
        "\nDECISION SOURCE:"
    )

    print(
        result["decision_source"]
    )

    print(
        "\nSARCASM:"
    )

    pprint(
        result["sarcasm"],
        sort_dicts=False,
    )

    print(
        "\nASPECTS:"
    )

    pprint(
        result["aspects"],
        sort_dicts=False,
    )

    print(
        "\nSENTENCES:"
    )

    pprint(
        result["sentences"],
        sort_dicts=False,
    )

    print(
        "\nMIXED:"
    )

    pprint(
        result["mixed_analysis"],
        sort_dicts=False,
    )

    print(
        "\nDISCOURSE:"
    )

    pprint(
        result["discourse"],
        sort_dicts=False,
    )