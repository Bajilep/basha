from src.basha.sentiment.english_model import (
    EnglishSentimentModel,
)


model = EnglishSentimentModel()


TESTS = {

    # ========================================================
    # 1. BASIC SENTIMENT
    # ========================================================

    "basic": [
        (
            "I absolutely love this phone.",
            "positive",
        ),
        (
            "This product is horrible.",
            "negative",
        ),
        (
            "The package arrived on Monday.",
            "neutral",
        ),
    ],

    # ========================================================
    # 2. WEAK SENTIMENT
    # ========================================================

    "weak": [
        (
            "The phone is okay.",
            "neutral",
        ),
        (
            "It could be better.",
            "negative",
        ),
        (
            "It's pretty decent.",
            "positive",
        ),
    ],

    # ========================================================
    # 3. NEGATION
    # ========================================================

    "negation": [
        (
            "This phone is not good.",
            "negative",
        ),
        (
            "The camera is not bad.",
            "positive",
        ),
        (
            "I don't hate it.",
            "positive",
        ),
        (
            "I wouldn't recommend this product.",
            "negative",
        ),
    ],

    # ========================================================
    # 4. INTENSITY
    # ========================================================

    "intensity": [
        (
            "The camera is extremely good.",
            "positive",
        ),
        (
            "The battery is absolutely terrible.",
            "negative",
        ),
        (
            "This is the worst phone I have ever owned.",
            "negative",
        ),
    ],

    # ========================================================
    # 5. MIXED SENTIMENT
    #
    # Current model has no Mixed class.
    # We expect these to expose its limitation.
    # ========================================================

    "mixed": [
        (
            "The display is excellent but the camera is terrible.",
            "mixed",
        ),
        (
            "I love the design, but I hate the battery life.",
            "mixed",
        ),
        (
            "The performance is great although the software is buggy.",
            "mixed",
        ),
    ],

    # ========================================================
    # 6. SARCASM
    #
    # We do NOT expect the basic sentiment model to reliably
    # solve these. These test why Basha needs sarcasm detection.
    # ========================================================

    "sarcasm": [
        (
            "Great, another software crash. Exactly what I needed.",
            "negative",
        ),
        (
            "Wonderful, the battery died again.",
            "negative",
        ),
        (
            "Amazing customer service. They only ignored me for a week.",
            "negative",
        ),
    ],

    # ========================================================
    # 7. FACTUAL / NEUTRAL
    # ========================================================

    "neutral": [
        (
            "The phone has a 6.5 inch display.",
            "neutral",
        ),
        (
            "The device comes in three colors.",
            "neutral",
        ),
        (
            "The battery capacity is 5000 mAh.",
            "neutral",
        ),
    ],

    # ========================================================
    # 8. CONTRAST
    # ========================================================

    "contrast": [
        (
            "I expected it to be terrible, but it is actually excellent.",
            "positive",
        ),
        (
            "It looked promising, but unfortunately it was disappointing.",
            "negative",
        ),
    ],
}


total = 0
correct = 0
mixed_cases = 0


for category, tests in TESTS.items():

    print("\n")
    print("=" * 80)
    print(category.upper())
    print("=" * 80)

    for text, expected in tests:

        result = model.predict(text)

        predicted = result["label"]

        total += 1

        # Mixed is intentionally unsupported
        # by our current 3-class model.
        if expected == "mixed":

            status = "LIMITATION"
            mixed_cases += 1

        elif predicted == expected:

            status = "PASS"
            correct += 1

        else:

            status = "FAIL"

        print()

        print(
            "TEXT:",
            text
        )

        print(
            "EXPECTED:",
            expected
        )

        print(
            "PREDICTED:",
            predicted
        )

        print(
            "CONFIDENCE:",
            result["confidence"]
        )

        print(
            "SCORES:",
            {
                "negative": result["negative"],
                "neutral": result["neutral"],
                "positive": result["positive"],
            }
        )

        print(
            "STATUS:",
            status
        )


# ============================================================
# SUMMARY
# ============================================================

gradable = total - mixed_cases

accuracy = (
    correct / gradable * 100
    if gradable
    else 0
)


print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Total tests:",
    total
)

print(
    "3-class tests:",
    gradable
)

print(
    "Correct:",
    correct
)

print(
    "Incorrect:",
    gradable - correct
)

print(
    "Mixed limitation tests:",
    mixed_cases
)

print(
    f"Simple test accuracy: "
    f"{accuracy:.2f}%"
)