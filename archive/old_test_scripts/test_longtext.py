from src.basha.sentiment.english_model import (
    EnglishSentimentModel,
)

from src.basha.longtext.splitter import (
    SentenceSplitter,
)

from src.basha.longtext.aggregator import (
    SentimentAggregator,
)


sentiment_model = (
    EnglishSentimentModel()
)

splitter = (
    SentenceSplitter()
)

aggregator = (
    SentimentAggregator()
)


text = """
The display is absolutely beautiful.
The colors are vibrant and the screen is very sharp.
The battery easily lasts all day.
However, the camera is terrible in low light.
The software crashes whenever I open the gallery.
Customer support was also extremely disappointing.
"""


# ============================================================
# SPLIT
# ============================================================

sentences = splitter.split(
    text
)


print("\n" + "=" * 70)
print("SENTENCES")
print("=" * 70)


for index, sentence in enumerate(
    sentences,
    start=1,
):

    print(
        f"{index}. {sentence}"
    )


# ============================================================
# SENTIMENT PER SENTENCE
# ============================================================

results = []


print("\n" + "=" * 70)
print("SENTENCE SENTIMENT")
print("=" * 70)


for sentence in sentences:

    result = sentiment_model.predict(
        sentence
    )

    results.append(
        result
    )

    print()

    print(
        "TEXT:",
        sentence
    )

    print(
        "LABEL:",
        result["label"]
    )

    print(
        "CONFIDENCE:",
        result["confidence"]
    )

    print(
        "NEG:",
        result["negative"]
    )

    print(
        "NEU:",
        result["neutral"]
    )

    print(
        "POS:",
        result["positive"]
    )


# ============================================================
# DOCUMENT AGGREGATION
# ============================================================

document_result = (
    aggregator.aggregate(
        results
    )
)


print("\n" + "=" * 70)
print("DOCUMENT RESULT")
print("=" * 70)


print(
    document_result
)