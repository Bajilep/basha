"""Example for using Basha."""

from basha import bashaanalyser

analyzer = bashaanalyser()

print("=" * 40)
print("Basha Analyser")
print("=" * 40)

sentence = input("Enter a sentence: ")

try:
    result = analyzer.predict(sentence)

    scores = result["scores"]
    flags = result["flags"]
    emotion = result["emotion"]
    intensity = result["intensity"]

    print("\nAnalysis Result")
    print("------------------------")
    print("Text:", result["text"])
    print("Sentiment:", result["label"].upper())
    print("Compound:", scores["compound"])

    print("\nSentiment Proportions")
    print("------------------------")
    print("Positive:", f"{scores['positive'] * 100:.1f}%")
    print("Negative:", f"{scores['negative'] * 100:.1f}%")
    print("Neutral:", f"{scores['neutral'] * 100:.1f}%")

    print("\nEmotion Information")
    print("------------------------")
    print("Emotion:", emotion["label"].upper())
    print("Confidence:", emotion["confidence"])

    print("\nExpression Intensity")
    print("------------------------")
    print("Score:", intensity["score"])
    print("Multiplier:", intensity["multiplier"])

    print("Emotion scores:")

    for emotion_name, score in emotion["scores"].items():
        print(
            f"  {emotion_name.title()}: "
            f"{score * 100:.1f}%"
        )

    print("\nSarcasm Information")
    print("------------------------")
    print("Detected:", flags["sarcasm"])
    print("Confidence:", flags["sarcasm_confidence"])

    if flags["sarcasm_reasons"]:
        print("Reasons:")

        for reason in flags["sarcasm_reasons"]:
            print("-", reason)

    print("Mixed polarity:", flags["mixed_polarity"])

    print("\nExplanation")
    print("------------------------")

    if not result["explanation"]:
        print("No sentiment words or symbols found.")

    for item in result["explanation"]:
        print("Token:", item["token"])
        print("Base score:", item["base_score"])
        print("Final score:", item["final_score"])

        if item["rules"]:
            print("Rules:", ", ".join(item["rules"]))
        else:
            print("Rules: None")

        print()

except (TypeError, ValueError) as error:
    print("Error:", error)
