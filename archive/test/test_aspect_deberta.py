import time

from src.basha.aspects.aspect_model_deberta import (
    DebertaAspectModel,
)


model = DebertaAspectModel()


tests = [

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

    (
        "The camera is excellent "
        "and the battery is great."
    ),

    (
        "The screen is beautiful, "
        "the processor is fast, "
        "but the speakers are awful."
    ),

    "The phone has a 6.5 inch display.",

    (
        "I expected it to be terrible, "
        "but it is actually excellent."
    ),

    (
        "The display is fantastic and "
        "the battery lasts all day. "
        "However, the camera performs "
        "poorly at night and the software "
        "crashes frequently."
    ),
]


total_start = time.perf_counter()


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

    start = time.perf_counter()

    aspects = model.predict(
        text
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nInference time: "
        f"{elapsed * 1000:.2f} ms"
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


total_elapsed = (
    time.perf_counter()
    - total_start
)


print(
    "\n"
    + "=" * 80
)

print(
    "TOTAL INFERENCE TIME"
)

print(
    "=" * 80
)

print(
    f"{total_elapsed:.2f} seconds"
)