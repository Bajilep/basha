
from __future__ import annotations

import json

import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent



from basha.aspects.aspect_model import EnglishAspectModel
from basha.fusion.english_fusion import EnglishFusion


DATASET_PATH = (
    EVALUATION_DIR
    / "data"
    / "english_fusion_benchmark.json"
)

FINETUNED_MODEL_PATH = (
    EVALUATION_DIR
    / "models"
    / "gauneg_basha_v1"
    / "best_model"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "results"
    / "english_fusion_finetuned_comparison.json"
)

LABELS = ["negative", "neutral", "positive", "mixed"]

ORIGINAL_MODEL_NAME = EnglishAspectModel.MODEL_NAME


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Benchmark dataset must be a list.")

    if len(dataset) != 400:
        raise ValueError(
            f"Expected 400 frozen examples, found {len(dataset)}."
        )

    counts = Counter(
        str(item["label"]).strip().lower()
        for item in dataset
    )

    if counts["mixed"] != 100:
        raise ValueError(
            "Expected exactly 100 Mixed examples."
        )

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid example {index}")

        if not isinstance(item.get("text"), str):
            raise ValueError(f"Missing text at example {index}")

        if str(item.get("label", "")).lower() not in LABELS:
            raise ValueError(f"Invalid label at example {index}")

    return dataset


def create_fusion(model_path: str):
    """
    Construct an EnglishFusion instance using the selected
    Gauneg checkpoint.

    The original class attribute is restored even if model
    initialization fails.
    """
    previous_model = EnglishAspectModel.MODEL_NAME

    try:
        EnglishAspectModel.MODEL_NAME = model_path
        fusion = EnglishFusion()
    finally:
        EnglishAspectModel.MODEL_NAME = previous_model

    return fusion



def extract_final_label(output):
    """
    Extract the final sentiment from Basha EnglishFusion.

    Supports a string sentiment or a dictionary containing
    a label, while rejecting unexpected output formats.
    """
    if not isinstance(output, dict):
        raise TypeError(
            f"Expected dictionary, got {type(output).__name__}"
        )

    sentiment = output.get("sentiment")

    if isinstance(sentiment, str):
        label = sentiment.strip().lower()

    elif isinstance(sentiment, dict):
        value = sentiment.get("label")

        if not isinstance(value, str):
            raise ValueError(
                "Sentiment dictionary has no valid label. "
                f"Contents: {sentiment}"
            )

        label = value.strip().lower()

    else:
        raise ValueError(
            "Unexpected sentiment output format: "
            f"{type(sentiment).__name__}. "
            f"Value: {sentiment}"
        )

    if label not in LABELS:
        raise ValueError(
            f"Unknown sentiment label: {label!r}"
        )

    return label

def evaluate_model(name, fusion, dataset):
    print("\n" + "=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    y_true = []
    y_pred = []
    details = []

    start_time = time.perf_counter()

    for index, item in enumerate(dataset, start=1):
        text = item["text"]
        gold = item["label"].strip().lower()

        output = fusion.analyze(text)
        predicted = extract_final_label(output)

        y_true.append(gold)
        y_pred.append(predicted)

        details.append({
            "index": index,
            "text": text,
            "gold": gold,
            "predicted": predicted,
            "correct": gold == predicted,
            "phenomenon": item.get("phenomenon"),
        })

        if index % 25 == 0:
            print(f"{index}/400 complete")

    elapsed = time.perf_counter() - start_time

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    mixed_precision, mixed_recall, mixed_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["mixed"],
            zero_division=0,
        )
    )

    normal_indices = [
        i for i, gold in enumerate(y_true)
        if gold != "mixed"
    ]

    normal_accuracy = accuracy_score(
        [y_true[i] for i in normal_indices],
        [y_pred[i] for i in normal_indices],
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mixed_precision": float(mixed_precision[0]),
        "mixed_recall": float(mixed_recall[0]),
        "mixed_f1": float(mixed_f1[0]),
        "normal_accuracy": float(normal_accuracy),
        "mixed_correct": sum(
            gold == "mixed" and pred == "mixed"
            for gold, pred in zip(y_true, y_pred)
        ),
        "mixed_predicted": y_pred.count("mixed"),
        "elapsed_seconds": elapsed,
        "mean_inference_ms": elapsed / len(dataset) * 1000,
    }

    print("\nRESULTS")
    for key in (
        "accuracy",
        "macro_f1",
        "mixed_precision",
        "mixed_recall",
        "mixed_f1",
        "normal_accuracy",
    ):
        print(f"{key:<22}: {metrics[key] * 100:.2f}%")

    print(
        "\nClassification report:\n",
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        ),
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
        "details": details,
    }


def print_comparison(original, finetuned):
    print("\n" + "=" * 85)
    print("BASHA FROZEN 400 - FULL FUSION COMPARISON")
    print("=" * 85)

    print(
        f"{'Metric':<24}"
        f"{'Original':>15}"
        f"{'Fine-tuned':>15}"
        f"{'Change (pp)':>16}"
    )
    print("-" * 85)

    for key in (
        "accuracy",
        "macro_f1",
        "mixed_precision",
        "mixed_recall",
        "mixed_f1",
        "normal_accuracy",
    ):
        old = original["metrics"][key] * 100
        new = finetuned["metrics"][key] * 100

        print(
            f"{key:<24}"
            f"{old:>14.2f}%"
            f"{new:>14.2f}%"
            f"{new - old:>+15.2f}"
        )


def main():
    print("=" * 80)
    print("BASHA - EXPERIMENTAL FULL FUSION COMPARISON")
    print("=" * 80)

    if not FINETUNED_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint missing: {FINETUNED_MODEL_PATH}"
        )

    dataset = load_dataset()

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Examples: {len(dataset)}")
    print(f"Original model: {ORIGINAL_MODEL_NAME}")
    print(f"Fine-tuned model: {FINETUNED_MODEL_PATH}")

    # Original production configuration
    original_fusion = create_fusion(ORIGINAL_MODEL_NAME)
    original_results = evaluate_model(
        "Original Basha",
        original_fusion,
        dataset,
    )

    # Release unused model references before the next run.
    del original_fusion

    # Experimental configuration
    finetuned_fusion = create_fusion(
        str(FINETUNED_MODEL_PATH)
    )

    finetuned_results = evaluate_model(
        "Fine-tuned Basha",
        finetuned_fusion,
        dataset,
    )

    print_comparison(original_results, finetuned_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": str(DATASET_PATH),
        "original_checkpoint": ORIGINAL_MODEL_NAME,
        "finetuned_checkpoint": str(FINETUNED_MODEL_PATH),
        "labels": LABELS,
        "original": original_results,
        "finetuned": finetuned_results,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("Production source files remain unchanged.")


if __name__ == "__main__":
    main()
