from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from basha.sentiment.english_model import (
    EnglishSentimentModel,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
MAX_SAMPLES = 1000

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING TWEETEVAL SENTIMENT")
print("=" * 70)

dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "sentiment",
)

full_test_data = dataset["test"]

print(
    "Full test samples:",
    len(full_test_data),
)


# ============================================================
# RANDOM REPRODUCIBLE SAMPLE
# ============================================================

test_data = (
    full_test_data
    .shuffle(seed=SEED)
    .select(
        range(
            min(
                MAX_SAMPLES,
                len(full_test_data),
            )
        )
    )
)

samples_to_test = len(test_data)

print(
    "Selected random samples:",
    samples_to_test,
)

print(
    "Random seed:",
    SEED,
)


# ============================================================
# LOAD BASHA MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = EnglishSentimentModel()


# ============================================================
# PREDICTION
# ============================================================

y_true = []
y_pred = []

print("\n" + "=" * 70)

print(
    f"BENCHMARKING {samples_to_test} SAMPLES"
)

print("=" * 70)


for index in range(samples_to_test):

    example = test_data[index]

    text = example["text"]

    true_label = int(
        example["label"]
    )

    result = model.predict(
        text
    )

    predicted_label = LABEL_TO_ID[
        result["label"]
    ]

    y_true.append(
        true_label
    )

    y_pred.append(
        predicted_label
    )

    if (index + 1) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1}/"
            f"{samples_to_test}"
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nCLASSIFICATION REPORT:\n"
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
        target_names=[
            "Negative",
            "Neutral",
            "Positive",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nCONFUSION MATRIX:"
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1,
        2,
    ],
)

print(matrix)

print(
    "\nRows    = Actual"
)

print(
    "Columns = Predicted"
)

print(
    """
                 PREDICTED
              NEG   NEU   POS

ACTUAL NEG
ACTUAL NEU
ACTUAL POS
"""
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(
    f"Dataset       : TweetEval Sentiment"
)

print(
    f"Samples       : {samples_to_test}"
)

print(
    f"Seed          : {SEED}"
)

print(
    f"Accuracy      : {accuracy * 100:.2f}%"
)