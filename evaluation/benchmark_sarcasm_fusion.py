from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.basha.sarcasm.english_sarcasm import EnglishSarcasmModel


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "sarcasm_fusion_subset.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "sarcasm_fusion_results.json"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Dataset must contain a JSON list."
        )

    return data


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 80)
    print("BASHA SARCASM BENCHMARK")
    print("PRE-FINE-TUNING BASELINE")
    print("=" * 80)

    dataset = load_dataset()

    print(f"\nTotal examples: {len(dataset)}")

    model = EnglishSarcasmModel()

    y_true = []
    y_pred = []

    detailed_results = []

    total_time = 0.0

    false_positives = []
    false_negatives = []

    # ======================================================
    # RUN
    # ======================================================

    print("\nRunning sarcasm benchmark...\n")

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        text = item["text"]

        # --------------------------------------------------
        # Gold sarcasm label
        #
        # Dataset format:
        #
        # {
        #     "text": "...",
        #     "sarcasm": true/false,
        #     "phenomenon": "...",
        #     "source": "tweet_eval_irony"
        # }
        # --------------------------------------------------

        if "sarcasm" not in item:
            raise ValueError(
                f"Example {index} has no 'sarcasm' field."
            )

        raw_label = item["sarcasm"]

        if not isinstance(raw_label, bool):
            raise ValueError(
                f"Example {index} has invalid sarcasm label: "
                f"{raw_label!r}"
            )

        true_label = (
            1
            if raw_label
            else 0
        )

        start = time.perf_counter()

        result = model.predict(text)

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        predicted_label = (
            1
            if result["detected"]
            else 0
        )

        y_true.append(true_label)
        y_pred.append(predicted_label)

        correct = (
            predicted_label
            == true_label
        )

        status = (
            "PASS"
            if correct
            else "FAIL"
        )

        true_name = (
            "sarcasm"
            if true_label == 1
            else "non_sarcasm"
        )

        predicted_name = (
            "sarcasm"
            if predicted_label == 1
            else "non_sarcasm"
        )

        record = {
            "text": text,
            "true_label": true_name,
            "predicted_label": predicted_name,
            "correct": correct,
            "sarcasm_probability":
                result["sarcasm"],
            "non_sarcasm_probability":
                result["non_sarcasm"],
            "threshold":
                result["threshold"],
            "time_seconds":
                elapsed,
        }

        detailed_results.append(record)

        # ----------------------------------------------
        # Error categories
        # ----------------------------------------------

        if (
            true_label == 0
            and predicted_label == 1
        ):
            false_positives.append(record)

        elif (
            true_label == 1
            and predicted_label == 0
        ):
            false_negatives.append(record)

        print(
            f"[{index:>3}/{len(dataset)}] "
            f"{status:<4} | "
            f"TRUE={true_name:<12} | "
            f"PRED={predicted_name:<12} | "
            f"SARCASM={result['sarcasm']:.4f}"
        )

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sarcasm_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # ======================================================
    # DISPLAY
    # ======================================================

    print("\n" + "=" * 80)
    print("FINAL SARCASM RESULTS")
    print("=" * 80)

    print(
        f"\nAccuracy:          "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision:         "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall:            "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Sarcasm F1:        "
        f"{sarcasm_f1 * 100:.2f}%"
    )

    print(
        f"Macro-F1:          "
        f"{macro_f1 * 100:.2f}%"
    )

    print("\nCLASSIFICATION REPORT\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "non_sarcasm",
                "sarcasm",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print(
        "[[TN FP]\n"
        " [FN TP]]"
    )
    print(matrix)

    print("\nERROR ANALYSIS")

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    # ======================================================
    # SPEED
    # ======================================================

    average_time = (
        total_time
        / len(dataset)
    )

    print("\nPERFORMANCE")

    print(
        f"Total inference time: "
        f"{total_time:.2f} sec"
    )

    print(
        f"Average time/sample: "
        f"{average_time * 1000:.2f} ms"
    )

    # ======================================================
    # SAVE
    # ======================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "benchmark": {
            "name":
                "Basha Sarcasm "
                "Pre-Fine-Tuning Baseline",

            "examples":
                len(dataset),

            "threshold":
                model.threshold,
        },

        "metrics": {
            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "sarcasm_f1":
                sarcasm_f1,

            "macro_f1":
                macro_f1,
        },

        "confusion_matrix": {
            "tn":
                int(matrix[0][0]),

            "fp":
                int(matrix[0][1]),

            "fn":
                int(matrix[1][0]),

            "tp":
                int(matrix[1][1]),
        },

        "errors": {
            "false_positive_count":
                len(false_positives),

            "false_negative_count":
                len(false_negatives),

            "false_positives":
                false_positives,

            "false_negatives":
                false_negatives,
        },

        "performance": {
            "total_seconds":
                total_time,

            "average_seconds_per_sample":
                average_time,
        },

        "results":
            detailed_results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("SARCASM BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()