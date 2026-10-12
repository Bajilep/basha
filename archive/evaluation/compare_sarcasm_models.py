from __future__ import annotations

import time

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.basha.sarcasm.english_sarcasm import (
    EnglishSarcasmModel,
)

from src.basha.sarcasm.besstie_sarcasm import (
    BesstieSarcasmModel,
)


THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# DATASET
# ============================================================

print(
    "=" * 80
)

print(
    "LOADING TWEETEVAL IRONY"
)

print(
    "=" * 80
)


dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "irony",
)


test_data = dataset["test"]


texts = [
    item["text"]
    for item in test_data
]


true_labels = [
    int(
        item["label"]
    )
    for item in test_data
]


print(
    f"\nExamples: {len(texts)}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    probability_key,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"EVALUATING: {name}"
    )

    print(
        "=" * 80
    )

    probabilities = []

    start = time.perf_counter()

    for index, text in enumerate(
        texts,
        start=1,
    ):

        result = model.predict(
            text
        )

        probabilities.append(
            float(
                result[
                    probability_key
                ]
            )
        )

        if (
            index % 100 == 0
            or index == len(texts)
        ):

            print(
                f"Processed "
                f"{index}/{len(texts)}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    speed = (
        elapsed
        / len(texts)
        * 1000
    )

    print(
        "\nTHRESHOLD RESULTS"
    )

    print(
        "-" * 78
    )

    print(
        f"{'THRESHOLD':<12}"
        f"{'ACC':>10}"
        f"{'PREC':>10}"
        f"{'RECALL':>10}"
        f"{'IRONY F1':>12}"
        f"{'MACRO F1':>12}"
    )

    print(
        "-" * 78
    )

    threshold_results = {}

    for threshold in THRESHOLDS:

        predictions = [
            1
            if probability >= threshold
            else 0

            for probability
            in probabilities
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

        irony_f1 = f1_score(
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

        threshold_results[
            threshold
        ] = {
            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "irony_f1":
                irony_f1,

            "macro_f1":
                macro_f1,

            "predictions":
                predictions,
        }

        print(
            f"{threshold:<12.2f}"
            f"{accuracy * 100:>9.2f}%"
            f"{precision * 100:>9.2f}%"
            f"{recall * 100:>9.2f}%"
            f"{irony_f1 * 100:>11.2f}%"
            f"{macro_f1 * 100:>11.2f}%"
        )

    best_threshold = max(
        threshold_results,
        key=lambda threshold:
            threshold_results[
                threshold
            ]["macro_f1"],
    )

    best = threshold_results[
        best_threshold
    ]

    matrix = confusion_matrix(
        true_labels,
        best["predictions"],
    )

    print(
        "\nBEST RESULT"
    )

    print(
        "-" * 50
    )

    print(
        f"Threshold:       "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:        "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:       "
        f"{best['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:          "
        f"{best['recall'] * 100:.2f}%"
    )

    print(
        f"Irony F1:        "
        f"{best['irony_f1'] * 100:.2f}%"
    )

    print(
        f"Macro-F1:        "
        f"{best['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Speed:           "
        f"{speed:.2f} ms/sample"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        matrix
    )

    return {
        "name":
            name,

        "threshold":
            best_threshold,

        "accuracy":
            best["accuracy"],

        "precision":
            best["precision"],

        "recall":
            best["recall"],

        "irony_f1":
            best["irony_f1"],

        "macro_f1":
            best["macro_f1"],

        "speed":
            speed,

        "probabilities":
            probabilities,
    }


# ============================================================
# CARDIFF
# ============================================================

cardiff = (
    EnglishSarcasmModel()
)


cardiff_result = evaluate_model(
    name="CardiffNLP RoBERTa Irony",
    model=cardiff,
    probability_key="irony",
)


# Free memory before loading second model.

del cardiff


# ============================================================
# BESSTIE
# ============================================================

besstie = (
    BesstieSarcasmModel()
)


besstie_result = evaluate_model(
    name="BESSTIE DeBERTa-v3",
    model=besstie,
    probability_key="sarcasm",
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FINAL SARCASM MODEL COMPARISON"
)

print(
    "=" * 100
)


print(
    f"\n{'MODEL':<30}"
    f"{'THRESH':>9}"
    f"{'ACC':>10}"
    f"{'PREC':>10}"
    f"{'RECALL':>10}"
    f"{'IRONY F1':>12}"
    f"{'MACRO F1':>12}"
    f"{'MS/SAMPLE':>12}"
)


print(
    "-" * 105
)


for result in [
    cardiff_result,
    besstie_result,
]:

    print(
        f"{result['name']:<30}"
        f"{result['threshold']:>9.2f}"
        f"{result['accuracy'] * 100:>9.2f}%"
        f"{result['precision'] * 100:>9.2f}%"
        f"{result['recall'] * 100:>9.2f}%"
        f"{result['irony_f1'] * 100:>11.2f}%"
        f"{result['macro_f1'] * 100:>11.2f}%"
        f"{result['speed']:>12.2f}"
    )


winner = max(
    [
        cardiff_result,
        besstie_result,
    ],
    key=lambda result:
        result["macro_f1"],
)


print(
    "\n"
    + "=" * 100
)

print(
    "BEST TWEETEVAL RESULT"
)

print(
    "=" * 100
)


print(
    f"\n{winner['name']}"
)

print(
    f"Best threshold: "
    f"{winner['threshold']:.2f}"
)

print(
    f"Macro-F1: "
    f"{winner['macro_f1'] * 100:.2f}%"
)

print(
    f"Irony F1: "
    f"{winner['irony_f1'] * 100:.2f}%"
)