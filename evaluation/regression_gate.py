"""
Regression gate for Basha English fusion.

Runs the frozen 400-example benchmark and exits with code 1 if
  - accuracy < 0.7025, or
  - Mixed precision < 0.85.

Run from the project root before every release:
    python evaluation/regression_gate.py

Optional:
    --sentence-level     run with the aspect model's sentence_level=True
    --min-accuracy 0.70  override thresholds
    --min-mixed-precision 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from basha.fusion.english_fusion import EnglishFusion

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "data" / "english_fusion_benchmark.json"
MODEL_PATH = EVALUATION_DIR / "models" / "gauneg_basha_v1" / "best_model"
OUTPUT_PATH = EVALUATION_DIR / "results" / "regression_gate.json"

LABELS = ["negative", "neutral", "positive", "mixed"]


def compute_metrics(gold, predicted):
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, predicted))

    per_class = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.7025)
    parser.add_argument("--min-mixed-precision", type=float, default=0.85)
    parser.add_argument("--sentence-level", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if len(dataset) != 400:
        print(f"FAIL: expected 400 examples, found {len(dataset)}")
        return 1

    fusion = EnglishFusion(str(MODEL_PATH))
    fusion.aspect_model.sentence_level = args.sentence_level

    gold, predicted = [], []
    started = time.perf_counter()
    for i, item in enumerate(dataset, start=1):
        result = fusion.analyze(item["text"])
        predicted.append(result["sentiment"]["label"].strip().lower())
        gold.append(item["label"].strip().lower())
        if i % 50 == 0:
            print(f"  {i}/{len(dataset)}", flush=True)
    seconds = time.perf_counter() - started

    metrics = compute_metrics(gold, predicted)
    metrics["seconds"] = round(seconds, 1)
    metrics["sentence_level"] = args.sentence_level
    mixed_precision = metrics["per_class"]["mixed"]["precision"]

    # Compare on raw counts so float rounding cannot cause a false failure.
    # 0.7025 * 400 = 281 correct.
    min_correct = round(args.min_accuracy * metrics["n"])
    failures = []
    if metrics["correct"] < min_correct:
        failures.append(
            f"accuracy {metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['n']}) < {args.min_accuracy}"
        )
    if mixed_precision < args.min_mixed_precision:
        failures.append(
            f"mixed precision {mixed_precision:.4f} < {args.min_mixed_precision}"
        )

    metrics["passed"] = not failures
    metrics["failures"] = failures
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"accuracy        {metrics['accuracy']:.4f}  (min {args.min_accuracy})")
    print(f"macro F1        {metrics['macro_f1']:.4f}")
    print(f"mixed precision {mixed_precision:.4f}  (min {args.min_mixed_precision})")
    print(f"mixed recall    {metrics['per_class']['mixed']['recall']:.4f}")
    print(f"time            {seconds:.1f}s")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())