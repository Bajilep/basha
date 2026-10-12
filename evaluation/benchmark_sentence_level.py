"""
Benchmark: aspect model sentence_level OFF vs ON
on the frozen 400-example English benchmark.

Run from the project root:
    python evaluation/benchmark_sentence_level.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from basha.fusion.english_fusion import EnglishFusion

EVALUATION_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVALUATION_DIR / "data" / "english_fusion_benchmark.json"
MODEL_PATH = EVALUATION_DIR / "models" / "gauneg_basha_v1" / "best_model"
OUTPUT_PATH = EVALUATION_DIR / "results" / "sentence_level_comparison.json"

LABELS = ["negative", "neutral", "positive", "mixed"]


def evaluate(fusion, dataset):
    predictions = []
    started = time.perf_counter()

    for index, item in enumerate(dataset, start=1):
        result = fusion.analyze(item["text"])
        predictions.append(result["sentiment"]["label"].strip().lower())

        if index % 50 == 0:
            print(f"  {index}/{len(dataset)}", flush=True)

    seconds = time.perf_counter() - started
    gold = [item["label"].strip().lower() for item in dataset]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predictions, labels=LABELS, zero_division=0
    )

    metrics = {
        "accuracy": round(accuracy_score(gold, predictions), 4),
        "macro_f1": round(
            f1_score(gold, predictions, labels=LABELS,
                     average="macro", zero_division=0), 4),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
            }
            for i, label in enumerate(LABELS)
        },
        "seconds": round(seconds, 1),
        "seconds_per_example": round(seconds / len(dataset), 3),
    }
    return predictions, gold, metrics


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    assert len(dataset) == 400, f"Expected 400 examples, found {len(dataset)}"

    fusion = EnglishFusion(str(MODEL_PATH))

    runs = {}
    all_predictions = {}

    for name, flag in (("sentence_level_off", False),
                       ("sentence_level_on", True)):
        print(f"\nRunning {name}...")
        fusion.aspect_model.sentence_level = flag
        predictions, gold, metrics = evaluate(fusion, dataset)
        runs[name] = metrics
        all_predictions[name] = predictions

    off = all_predictions["sentence_level_off"]
    on = all_predictions["sentence_level_on"]

    fixed = sum(1 for g, a, b in zip(gold, off, on) if a != g and b == g)
    broken = sum(1 for g, a, b in zip(gold, off, on) if a == g and b != g)

    runs["changes"] = {
        "wrong_to_correct": fixed,
        "correct_to_wrong": broken,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for name in ("sentence_level_off", "sentence_level_on"):
        m = runs[name]
        mixed = m["per_class"]["mixed"]
        print(f"{name}")
        print(f"  accuracy        {m['accuracy']:.4f}")
        print(f"  macro F1        {m['macro_f1']:.4f}")
        print(f"  mixed precision {mixed['precision']:.4f}")
        print(f"  mixed recall    {mixed['recall']:.4f}")
        print(f"  sec / example   {m['seconds_per_example']}")
    print(f"\nwrong -> correct: {fixed}")
    print(f"correct -> wrong: {broken}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()