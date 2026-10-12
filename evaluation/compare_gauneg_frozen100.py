
from __future__ import annotations

import json
from pathlib import Path

from benchmark_ate_models import (
    BENCHMARK_FILE,
    GAUNEG_ROBERTA,
    SEQUENCE_THRESHOLD,
    TOKEN_OVERLAP_THRESHOLD,
    GaunegModel,
    build_gold_aspects,
    load_mixed_examples,
    print_comparison,
    run_candidate,
)


BASE_DIR = Path(__file__).resolve().parent

FINETUNED_MODEL = (
    BASE_DIR
    / "models"
    / "gauneg_basha_v1"
    / "best_model"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "gauneg_frozen100_comparison.json"
)


def main():

    print("=" * 80)
    print("BASHA - FROZEN 100 MIXED BENCHMARK")
    print("=" * 80)

    print("\nBenchmark dataset:")
    print(BENCHMARK_FILE)

    if not FINETUNED_MODEL.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {FINETUNED_MODEL}"
        )

    examples = load_mixed_examples()

    total_gold = sum(
        len(build_gold_aspects(example))
        for example in examples
    )

    print(f"\nMixed examples: {len(examples)}")
    print(f"Gold aspects: {total_gold}")

    if len(examples) != 100:
        raise RuntimeError(
            "Expected exactly 100 frozen Mixed examples. "
            f"Found {len(examples)}. Benchmark aborted."
        )

    results = {}

    # Original model: unchanged baseline
    results["Gauneg Original"] = run_candidate(
        "Gauneg Original",
        lambda: GaunegModel(GAUNEG_ROBERTA),
        examples,
    )

    # Experimental fine-tuned model
    results["Gauneg Fine-tuned"] = run_candidate(
        "Gauneg Fine-tuned",
        lambda: GaunegModel(str(FINETUNED_MODEL)),
        examples,
    )

    print_comparison(results)

    # Do not silently accept failed model runs.
    failed = [
        name
        for name, result in results.items()
        if result.get("status") != "success"
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "benchmark": {
            "dataset": str(BENCHMARK_FILE),
            "mixed_examples": len(examples),
            "gold_aspects": total_gold,
            "sequence_threshold": SEQUENCE_THRESHOLD,
            "token_overlap_threshold": TOKEN_OVERLAP_THRESHOLD,
            "scoring_source": "benchmark_ate_models.py",
        },
        "models": {
            "original": GAUNEG_ROBERTA,
            "finetuned": str(FINETUNED_MODEL),
        },
        "results": results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    if failed:
        raise RuntimeError(
            "Benchmark failed for: " + ", ".join(failed)
        )

    print("\nFROZEN 100 COMPARISON COMPLETE")
    print("Production model remains unchanged.")


if __name__ == "__main__":
    main()
