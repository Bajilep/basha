from pprint import pprint

from basha.fusion.english_fusion import (
    EnglishFusion,
)


basha = EnglishFusion()


tests = [
    "The camera is excellent.",

    "The camera is terrible.",

    (
        "The battery is excellent "
        "but the camera is terrible."
    ),

    (
        "The camera is excellent "
        "and the battery is great."
    ),

    (
        "The display is beautiful. "
        "The battery lasts all day. "
        "The camera is terrible. "
        "The software crashes constantly."
    ),

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

    "The phone has a 6.5 inch display.",

    (
        "The display is fantastic and "
        "the battery lasts all day. "
        "However, the camera performs poorly "
        "at night and the software crashes "
        "frequently."
    ),
]


for text in tests:

    print(
        "\n"
        + "=" * 80
    )

    print(text)

    print("=" * 80)

    result = basha.analyze(
        text
    )

    print(
        "\nFINAL:"
    )

    pprint(
        result["sentiment"],
        sort_dicts=False,
    )

    print(
        "\nSOURCE:",
        result["decision_source"],
    )

    print(
        "\nASPECTS:"
    )

    pprint(
        result["aspects"],
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