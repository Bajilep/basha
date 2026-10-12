from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.basha.fusion.english_fusion import EnglishFusion

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "english_fusion_benchmark.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "english_fusion_results.json"
)

LABELS = [
    "negative",
    "neutral",
    "positive",
    "mixed",
]

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset_file() -> List[Dict]:

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "\nBenchmark dataset not found:\n"
            f"{DATASET_PATH}\n"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Benchmark JSON must contain a list."
        )

    required = {
        "text",
        "label",
        "phenomenon",
    }

    for index, item in enumerate(data):

        missing = required - set(item)

        if missing:
            raise ValueError(
                f"Example {index} missing fields: "
                f"{missing}"
            )

        if item["label"] not in LABELS:
            raise ValueError(
                f"Invalid label at example {index}: "
                f"{item['label']}"
            )

    return data


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print("=" * 90)
    print("BASHA ENGLISH FUSION BENCHMARK")
    print("PRE-FINE-TUNING BASELINE")
    print("=" * 90)

    dataset = load_dataset_file()

    print(
        f"\nTotal examples: {len(dataset)}"
    )

    label_counts = Counter(
        item["label"]
        for item in dataset
    )

    phenomenon_counts = Counter(
        item["phenomenon"]
        for item in dataset
    )

    print("\nLABEL DISTRIBUTION")

    for label in LABELS:
        print(
            f"{label:<10}: "
            f"{label_counts[label]}"
        )

    print("\nPHENOMENON DISTRIBUTION")

    for phenomenon, count in sorted(
        phenomenon_counts.items()
    ):
        print(
            f"{phenomenon:<25}: {count}"
        )

    # ======================================================
    # LOAD BASHA
    # ======================================================

    print("\nInitializing Basha English Fusion...")

    basha = EnglishFusion()

    # ======================================================
    # STORAGE
    # ======================================================

    y_true = []

    # Final Basha predictions
    y_fusion = []

    # Raw Cardiff predictions
    y_core = []

    results = []

    phenomenon_results = defaultdict(
        lambda: {
            "true": [],
            "fusion": [],
            "core": [],
        }
    )

    decision_sources = Counter()

    fusion_fixed_core = 0
    fusion_broke_core = 0
    both_correct = 0
    both_wrong = 0

    total_time = 0.0

    # ======================================================
    # RUN BENCHMARK
    # ======================================================

    print("\nRunning benchmark...\n")

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        text = item["text"]
        true_label = item["label"]
        phenomenon = item["phenomenon"]

        start_time = time.perf_counter()

        output = basha.analyze(text)

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        fusion_label = (
            output["sentiment"]["label"]
        )

        fusion_confidence = (
            output["sentiment"]["confidence"]
        )

        core_label = (
            output["core_sentiment"]["label"]
        )

        core_confidence = (
            output["core_sentiment"]["confidence"]
        )

        source = output["decision_source"]

        fusion_correct = (
            fusion_label == true_label
        )

        core_correct = (
            core_label == true_label
        )

        # --------------------------------------------------
        # Compare fusion vs raw Cardiff
        # --------------------------------------------------

        if (
            not core_correct
            and fusion_correct
        ):
            fusion_fixed_core += 1

        elif (
            core_correct
            and not fusion_correct
        ):
            fusion_broke_core += 1

        elif (
            core_correct
            and fusion_correct
        ):
            both_correct += 1

        else:
            both_wrong += 1

        # --------------------------------------------------
        # Save predictions
        # --------------------------------------------------

        y_true.append(
            true_label
        )

        y_fusion.append(
            fusion_label
        )

        y_core.append(
            core_label
        )

        decision_sources[
            source
        ] += 1

        phenomenon_results[
            phenomenon
        ]["true"].append(
            true_label
        )

        phenomenon_results[
            phenomenon
        ]["fusion"].append(
            fusion_label
        )

        phenomenon_results[
            phenomenon
        ]["core"].append(
            core_label
        )

        result = {
            "text":
                text,

            "true_label":
                true_label,

            "fusion_label":
                fusion_label,

            "fusion_confidence":
                fusion_confidence,

            "core_label":
                core_label,

            "core_confidence":
                core_confidence,

            "decision_source":
                source,

            "phenomenon":
                phenomenon,

            "fusion_correct":
                fusion_correct,

            "core_correct":
                core_correct,

            "time_seconds":
                elapsed,
        }

        # Keep useful evidence for later error analysis.

        if "sarcasm" in output:
            result["sarcasm"] = (
                output["sarcasm"]
            )

        if "mixed_analysis" in output:
            result["mixed_analysis"] = (
                output["mixed_analysis"]
            )

        if "discourse" in output:
            result["discourse"] = (
                output["discourse"]
            )

        if "semantic_conflict" in output:
            result["semantic_conflict"] = (
                output["semantic_conflict"]
            )

        results.append(result)

        status = (
            "PASS"
            if fusion_correct
            else "FAIL"
        )

        print(
            f"[{index:>3}/{len(dataset)}] "
            f"{status:<4} | "
            f"TRUE={true_label:<8} | "
            f"CORE={core_label:<8} | "
            f"FINAL={fusion_label:<8} | "
            f"SOURCE={source}"
        )

    # ======================================================
    # FINAL BASHA METRICS
    # ======================================================

    fusion_accuracy = accuracy_score(
        y_true,
        y_fusion,
    )

    fusion_macro_f1 = f1_score(
        y_true,
        y_fusion,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    fusion_weighted_f1 = f1_score(
        y_true,
        y_fusion,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    # ======================================================
    # CORE CARDIFF METRICS
    # ======================================================

    core_accuracy = accuracy_score(
        y_true,
        y_core,
    )

    core_macro_f1 = f1_score(
        y_true,
        y_core,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    core_weighted_f1 = f1_score(
        y_true,
        y_core,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    # ======================================================
    # DISPLAY OVERALL RESULTS
    # ======================================================

    print("\n" + "=" * 90)
    print("FINAL BASHA FUSION RESULTS")
    print("=" * 90)

    print(
        f"\nAccuracy:      "
        f"{fusion_accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1:      "
        f"{fusion_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1:   "
        f"{fusion_weighted_f1 * 100:.2f}%"
    )

    print("\nCLASSIFICATION REPORT\n")

    print(
        classification_report(
            y_true,
            y_fusion,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")

    print(
        "Order:",
        LABELS,
    )

    print(
        confusion_matrix(
            y_true,
            y_fusion,
            labels=LABELS,
        )
    )

    # ======================================================
    # CORE CARDIFF VS BASHA
    # ======================================================

    print("\n" + "=" * 90)
    print("CORE CARDIFF VS BASHA FUSION")
    print("=" * 90)

    print("\nCORE CARDIFF")

    print(
        f"Accuracy:      "
        f"{core_accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1:      "
        f"{core_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1:   "
        f"{core_weighted_f1 * 100:.2f}%"
    )

    print("\nBASHA FUSION")

    print(
        f"Accuracy:      "
        f"{fusion_accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1:      "
        f"{fusion_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1:   "
        f"{fusion_weighted_f1 * 100:.2f}%"
    )

    accuracy_difference = (
        fusion_accuracy
        - core_accuracy
    )

    macro_difference = (
        fusion_macro_f1
        - core_macro_f1
    )

    print("\nIMPROVEMENT")

    print(
        f"Accuracy difference: "
        f"{accuracy_difference * 100:+.2f} pp"
    )

    print(
        f"Macro-F1 difference: "
        f"{macro_difference * 100:+.2f} pp"
    )

    print("\nCORRECTION ANALYSIS")

    print(
        f"Fusion fixed core errors:       "
        f"{fusion_fixed_core}"
    )

    print(
        f"Fusion broke correct core:      "
        f"{fusion_broke_core}"
    )

    print(
        f"Both correct:                   "
        f"{both_correct}"
    )

    print(
        f"Both wrong:                     "
        f"{both_wrong}"
    )

    net_corrections = (
        fusion_fixed_core
        - fusion_broke_core
    )

    print(
        f"Net corrections:                "
        f"{net_corrections:+d}"
    )

    # ======================================================
    # DECISION SOURCES
    # ======================================================

    print("\n" + "=" * 90)
    print("DECISION SOURCES")
    print("=" * 90)

    for source, count in (
        decision_sources.most_common()
    ):

        percentage = (
            count
            / len(dataset)
            * 100
        )

        print(
            f"{source:<30}"
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    # ======================================================
    # PHENOMENON ANALYSIS
    # ======================================================

    print("\n" + "=" * 90)
    print("RESULTS BY PHENOMENON")
    print("=" * 90)

    phenomenon_summary = {}

    for phenomenon in sorted(
        phenomenon_results
    ):

        values = phenomenon_results[
            phenomenon
        ]

        true_values = values["true"]
        fusion_values = values["fusion"]
        core_values = values["core"]

        fusion_acc = accuracy_score(
            true_values,
            fusion_values,
        )

        core_acc = accuracy_score(
            true_values,
            core_values,
        )

        # Macro-F1 only across labels actually
        # represented by this phenomenon subset.
        subset_labels = sorted(
            set(true_values)
            | set(fusion_values)
            | set(core_values)
        )

        fusion_f1 = f1_score(
            true_values,
            fusion_values,
            labels=subset_labels,
            average="macro",
            zero_division=0,
        )

        core_f1 = f1_score(
            true_values,
            core_values,
            labels=subset_labels,
            average="macro",
            zero_division=0,
        )

        phenomenon_summary[
            phenomenon
        ] = {
            "count":
                len(true_values),

            "fusion_accuracy":
                fusion_acc,

            "fusion_macro_f1":
                fusion_f1,

            "core_accuracy":
                core_acc,

            "core_macro_f1":
                core_f1,
        }

        print(
            f"\n{phenomenon}"
        )

        print(
            f"  Examples:          "
            f"{len(true_values)}"
        )

        print(
            f"  Core accuracy:     "
            f"{core_acc * 100:.2f}%"
        )

        print(
            f"  Fusion accuracy:   "
            f"{fusion_acc * 100:.2f}%"
        )

        print(
            f"  Core Macro-F1:     "
            f"{core_f1 * 100:.2f}%"
        )

        print(
            f"  Fusion Macro-F1:   "
            f"{fusion_f1 * 100:.2f}%"
        )

    # ======================================================
    # PERFORMANCE
    # ======================================================

    average_time = (
        total_time
        / len(dataset)
    )

    print("\n" + "=" * 90)
    print("PERFORMANCE")
    print("=" * 90)

    print(
        f"\nTotal inference time: "
        f"{total_time:.2f} sec"
    )

    print(
        f"Average time/sample:  "
        f"{average_time * 1000:.2f} ms"
    )

    # ======================================================
    # SAVE COMPLETE RESULTS
    # ======================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {

        "benchmark": {
            "name":
                "Basha English Fusion "
                "Pre-Fine-Tuning Baseline",

            "examples":
                len(dataset),

            "labels":
                LABELS,
        },

        "fusion": {
            "accuracy":
                fusion_accuracy,

            "macro_f1":
                fusion_macro_f1,

            "weighted_f1":
                fusion_weighted_f1,
        },

        "core": {
            "accuracy":
                core_accuracy,

            "macro_f1":
                core_macro_f1,

            "weighted_f1":
                core_weighted_f1,
        },

        "improvement": {
            "accuracy_difference":
                accuracy_difference,

            "macro_f1_difference":
                macro_difference,

            "fusion_fixed_core":
                fusion_fixed_core,

            "fusion_broke_core":
                fusion_broke_core,

            "net_corrections":
                net_corrections,
        },

        "decision_sources":
            dict(decision_sources),

        "phenomena":
            phenomenon_summary,

        "performance": {
            "total_seconds":
                total_time,

            "average_seconds_per_sample":
                average_time,
        },

        "results":
            results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to:")

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()