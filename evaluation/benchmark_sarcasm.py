from __future__ import annotations

import time

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.basha.sarcasm.english_sarcasm import (
    EnglishSarcasmModel,
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# LOAD DATASET
# ============================================================

print(
    "=" * 80
)

print(
    "LOADING TWEETEVAL IRONY DATASET"
)

print(
    "=" * 80
)


dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "irony",
)


test_data = dataset["test"]


print(
    f"\nTest examples: {len(test_data)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LOADING SARCASM MODEL"
)

print(
    "=" * 80
)


model = EnglishSarcasmModel()


# ============================================================
# INFERENCE
# ============================================================

true_labels = []

irony_probabilities = []


print(
    "\n"
    + "=" * 80
)

print(
    "RUNNING INFERENCE"
)

print(
    "=" * 80
)


start_time = time.perf_counter()


for index, item in enumerate(
    test_data,
    start=1,
):

    text = item["text"]

    true_label = int(
        item["label"]
    )

    result = model.predict(
        text
    )

    true_labels.append(
        true_label
    )

    irony_probabilities.append(
        result["irony"]
    )

    if (
        index % 100 == 0
        or index == len(test_data)
    ):

        print(
            f"Processed "
            f"{index}/{len(test_data)}"
        )


elapsed = (
    time.perf_counter()
    - start_time
)


ms_per_sample = (
    elapsed
    / len(test_data)
    * 1000
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "THRESHOLD COMPARISON"
)

print(
    "=" * 80
)


print(
    f"\n{'THRESHOLD':<12}"
    f"{'ACC':>10}"
    f"{'PRECISION':>12}"
    f"{'RECALL':>10}"
    f"{'F1':>10}"
    f"{'MACRO F1':>12}"
)


print(
    "-" * 66
)


results = {}


for threshold in THRESHOLDS:

    predictions = [
        1 if probability >= threshold else 0
        for probability
        in irony_probabilities
    ]

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    results[threshold] = {
        "predictions":
            predictions,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "macro_f1":
            macro_f1,
    }

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:>9.2f}%"
        f"{precision * 100:>11.2f}%"
        f"{recall * 100:>9.2f}%"
        f"{f1 * 100:>9.2f}%"
        f"{macro_f1 * 100:>11.2f}%"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

best_threshold = max(
    results,
    key=lambda threshold:
        results[threshold]["macro_f1"],
)


best = results[
    best_threshold
]


print(
    "\n"
    + "=" * 80
)

print(
    "BEST THRESHOLD"
)

print(
    "=" * 80
)


print(
    f"\nThreshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy: "
    f"{best['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{best['f1'] * 100:.2f}%"
)

print(
    f"Macro-F1: "
    f"{best['macro_f1'] * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nCLASSIFICATION REPORT:\n"
)


print(
    classification_report(
        true_labels,
        best["predictions"],
        target_names=[
            "non_irony",
            "irony",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "CONFUSION MATRIX:"
)


print(
    confusion_matrix(
        true_labels,
        best["predictions"],
    )
)


# ============================================================
# SPEED
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "INFERENCE SPEED"
)

print(
    "=" * 80
)


print(
    f"\nTotal time: "
    f"{elapsed:.2f} sec"
)

print(
    f"Average: "
    f"{ms_per_sample:.2f} ms/sample"
)


# ============================================================
# FALSE POSITIVES
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "SAMPLE FALSE POSITIVES"
)

print(
    "=" * 80
)


false_positive_count = 0


for item, true_label, probability in zip(
    test_data,
    true_labels,
    irony_probabilities,
):

    prediction = (
        1
        if probability >= best_threshold
        else 0
    )

    if (
        true_label == 0
        and prediction == 1
    ):

        print(
            f"\nIrony probability: "
            f"{probability:.4f}"
        )

        print(
            item["text"]
        )

        false_positive_count += 1

        if false_positive_count >= 10:
            break


# ============================================================
# FALSE NEGATIVES
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "SAMPLE FALSE NEGATIVES"
)

print(
    "=" * 80
)


false_negative_count = 0


for item, true_label, probability in zip(
    test_data,
    true_labels,
    irony_probabilities,
):

    prediction = (
        1
        if probability >= best_threshold
        else 0
    )

    if (
        true_label == 1
        and prediction == 0
    ):

        print(
            f"\nIrony probability: "
            f"{probability:.4f}"
        )

        print(
            item["text"]
        )

        false_negative_count += 1

        if false_negative_count >= 10:
            break