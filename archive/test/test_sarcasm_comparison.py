from src.basha.sarcasm.english_sarcasm import (
    EnglishSarcasmModel,
)

from src.basha.sarcasm.besstie_sarcasm import (
    BesstieSarcasmModel,
)


CARDIFF_THRESHOLD = 0.40
BESSTIE_THRESHOLD = 0.80


tests = [

    # --------------------------------------------------
    # Genuine positive
    # --------------------------------------------------

    ("I absolutely love this phone.", False),

    ("The camera quality is fantastic.", False),

    ("The battery life is excellent and lasts all day.", False),

    ("This is one of the best laptops I have ever used.", False),

    # --------------------------------------------------
    # Genuine negative
    # --------------------------------------------------

    ("This phone is terrible.", False),

    ("The camera quality is extremely poor.", False),

    ("The software crashes every few minutes.", False),

    ("Customer support was very disappointing.", False),

    # --------------------------------------------------
    # Factual / neutral
    # --------------------------------------------------

    ("The phone has a 6.5 inch display.", False),

    ("The laptop contains 16 GB of RAM.", False),

    ("The battery capacity is 5000 mAh.", False),

    ("The device was released in 2025.", False),

    # --------------------------------------------------
    # Genuine mixed sentiment
    # --------------------------------------------------

    (
        "The display is excellent, but the camera is terrible.",
        False,
    ),

    (
        "I love the battery life, although the speakers are poor.",
        False,
    ),

    (
        "The processor is fast, but the software is buggy.",
        False,
    ),

    # --------------------------------------------------
    # Expectation vs actual
    # --------------------------------------------------

    (
        "I expected it to be terrible, but it is actually excellent.",
        False,
    ),

    (
        "I thought the camera would be excellent, but it is actually terrible.",
        False,
    ),

    # --------------------------------------------------
    # Obvious sarcasm
    # --------------------------------------------------

    (
        "Great, another software crash. Exactly what I needed.",
        True,
    ),

    (
        "Wonderful, the battery died again after two hours.",
        True,
    ),

    (
        "Oh fantastic, the app deleted all my files.",
        True,
    ),

    (
        "Amazing, the camera stopped working on the first day.",
        True,
    ),

    (
        "Perfect, my laptop just crashed before I saved my work.",
        True,
    ),

    # --------------------------------------------------
    # Sarcastic constructions
    # --------------------------------------------------

    (
        "Yeah, because crashing every five minutes is exactly what a good app should do.",
        True,
    ),

    (
        "I just love waiting three hours for customer support.",
        True,
    ),

    (
        "Nothing says premium quality like a phone that breaks in two days.",
        True,
    ),

    (
        "What a fantastic update, now the app doesn't even open.",
        True,
    ),

    (
        "Best phone ever. It only crashes ten times a day.",
        True,
    ),

    # --------------------------------------------------
    # More subtle sarcasm
    # --------------------------------------------------

    (
        "The battery lasted an entire two hours. Impressive.",
        True,
    ),

    (
        "Customer support replied after five days. Very helpful.",
        True,
    ),

    (
        "The camera produces blurry photos every time. Truly professional.",
        True,
    ),

    (
        "A phone without working Wi-Fi. Brilliant design.",
        True,
    ),

    (
        "It froze during the presentation. Couldn't have happened at a better time.",
        True,
    ),
]


cardiff = EnglishSarcasmModel()
besstie = BesstieSarcasmModel()


cardiff_correct = 0
besstie_correct = 0


print()
print("=" * 110)
print("BASHA SARCASM CHALLENGE")
print("=" * 110)


for number, (text, expected) in enumerate(
    tests,
    start=1,
):

    cardiff_result = cardiff.predict(text)
    besstie_result = besstie.predict(text)

    cardiff_probability = cardiff_result["irony"]
    besstie_probability = besstie_result["sarcasm"]

    cardiff_prediction = (
        cardiff_probability >= CARDIFF_THRESHOLD
    )

    besstie_prediction = (
        besstie_probability >= BESSTIE_THRESHOLD
    )

    if cardiff_prediction == expected:
        cardiff_correct += 1

    if besstie_prediction == expected:
        besstie_correct += 1

    print()
    print("-" * 110)

    print(
        f"{number}. {text}"
    )

    print(
        f"Expected : "
        f"{'SARCASM' if expected else 'NORMAL'}"
    )

    print()

    print(
        f"Cardiff  : "
        f"{'SARCASM' if cardiff_prediction else 'NORMAL':<8} "
        f"prob={cardiff_probability:.4f} "
        f"{'✓' if cardiff_prediction == expected else '✗'}"
    )

    print(
        f"BESSTIE  : "
        f"{'SARCASM' if besstie_prediction else 'NORMAL':<8} "
        f"prob={besstie_probability:.4f} "
        f"{'✓' if besstie_prediction == expected else '✗'}"
    )


total = len(tests)


print()
print("=" * 110)
print("FINAL RESULTS")
print("=" * 110)


print(
    f"\nTotal examples: {total}"
)

print(
    f"\nCardiff:"
    f"\n  Correct = {cardiff_correct}/{total}"
    f"\n  Accuracy = {(cardiff_correct / total) * 100:.2f}%"
)

print(
    f"\nBESSTIE:"
    f"\n  Correct = {besstie_correct}/{total}"
    f"\n  Accuracy = {(besstie_correct / total) * 100:.2f}%"
)


if besstie_correct > cardiff_correct:

    print(
        "\nWINNER: BESSTIE DeBERTa-v3"
    )

elif cardiff_correct > besstie_correct:

    print(
        "\nWINNER: CardiffNLP RoBERTa"
    )

else:

    print(
        "\nRESULT: TIE"
    )