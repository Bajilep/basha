
from __future__ import annotations

import gc
import json

import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"



from basha.aspects.aspect_model import EnglishAspectModel
from basha.fusion.english_fusion import EnglishFusion


COMPARISON_PATH = (
    EVAL / "results"
    / "english_fusion_finetuned_comparison.json"
)

CHECKPOINT_PATH = (
    EVAL / "models"
    / "gauneg_basha_v1"
    / "best_model"
)

OUTPUT_PATH = (
    EVAL / "results"
    / "fusion_diagnostic_v1.json"
)

# Diagnose every corrected case and regression,
# plus representative remaining Mixed examples.
MAX_REMAINING_MIXED = 30
MAX_CORRECT_MIXED = 10

ORIGINAL_CHECKPOINT = EnglishAspectModel.MODEL_NAME


def to_jsonable(value: Any) -> Any:
    """Convert model outputs into JSON-compatible values."""

    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def load_comparison():
    with COMPARISON_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        report = json.load(file)

    original = {
        item["index"]: item
        for item in report["original"]["details"]
    }

    finetuned = {
        item["index"]: item
        for item in report["finetuned"]["details"]
    }

    if set(original) != set(finetuned):
        raise ValueError(
            "Original and fine-tuned example IDs differ."
        )

    return original, finetuned


def select_examples(original, finetuned):
    groups = {
        "corrected": [],
        "regressions": [],
        "remaining_mixed": [],
        "correct_mixed": [],
    }

    for index in sorted(original):
        old = original[index]
        new = finetuned[index]

        if (
            old["text"] != new["text"]
            or old["gold"] != new["gold"]
        ):
            raise ValueError(
                f"Example mismatch at index {index}"
            )

        gold = old["gold"]
        old_correct = old["predicted"] == gold
        new_correct = new["predicted"] == gold

        if not old_correct and new_correct:
            groups["corrected"].append(index)

        elif old_correct and not new_correct:
            groups["regressions"].append(index)

        elif gold == "mixed" and not new_correct:
            groups["remaining_mixed"].append(index)

        elif gold == "mixed" and new_correct:
            groups["correct_mixed"].append(index)

    groups["remaining_mixed"] = (
        groups["remaining_mixed"][
            :MAX_REMAINING_MIXED
        ]
    )

    groups["correct_mixed"] = (
        groups["correct_mixed"][
            :MAX_CORRECT_MIXED
        ]
    )

    return groups


def create_fusion(checkpoint: str):
    """
    Temporarily select a checkpoint while constructing
    EnglishFusion. Restore the production default.
    """
    previous = EnglishAspectModel.MODEL_NAME

    try:
        EnglishAspectModel.MODEL_NAME = checkpoint
        fusion = EnglishFusion()
    finally:
        EnglishAspectModel.MODEL_NAME = previous

    return fusion


def extract_label(output):
    sentiment = output.get("sentiment")

    if isinstance(sentiment, str):
        return sentiment.strip().lower()

    if isinstance(sentiment, dict):
        label = sentiment.get("label")

        if isinstance(label, str):
            return label.strip().lower()

    raise ValueError(
        f"Unknown final sentiment structure: {sentiment}"
    )


def run_diagnostics(
    name,
    checkpoint,
    selected_ids,
    original,
    finetuned,
):
    print("\n" + "=" * 75)
    print(f"RUNNING: {name}")
    print("=" * 75)

    fusion = create_fusion(checkpoint)
    records = {}

    try:
        for position, index in enumerate(
            selected_ids, start=1
        ):
            example = original[index]
            text = example["text"]

            start = time.perf_counter()
            output = fusion.analyze(text)
            elapsed = time.perf_counter() - start

            label = extract_label(output)

            expected = (
                example["predicted"]
                if name == "original"
                else finetuned[index]["predicted"]
            )

            if label != expected:
                raise RuntimeError(
                    f"Prediction changed at #{index}: "
                    f"expected {expected}, got {label}. "
                    "Check model configuration and "
                    "inference determinism."
                )

            records[str(index)] = {
                "index": index,
                "text": text,
                "gold": example["gold"],
                "predicted": label,
                "correct": label == example["gold"],
                "latency_ms": round(
                    elapsed * 1000, 2
                ),
                "decision_source": to_jsonable(
                    output.get("decision_source")
                ),
                "sentiment": to_jsonable(
                    output.get("sentiment")
                ),
                "core_sentiment": to_jsonable(
                    output.get("core_sentiment")
                ),
                "aspects": to_jsonable(
                    output.get("aspects")
                ),
                "sentences": to_jsonable(
                    output.get("sentences")
                ),
                "contrast": to_jsonable(
                    output.get("contrast")
                ),
                "mixed_analysis": to_jsonable(
                    output.get("mixed_analysis")
                ),
                "discourse": to_jsonable(
                    output.get("discourse")
                ),
                "sarcasm": to_jsonable(
                    output.get("sarcasm")
                ),
                "semantic_conflict": to_jsonable(
                    output.get("semantic_conflict")
                ),
                "sarcasm_resolution": to_jsonable(
                    output.get("sarcasm_resolution")
                ),
            }

            if (
                position % 10 == 0
                or position == len(selected_ids)
            ):
                print(
                    f"{position}/{len(selected_ids)} complete"
                )

    finally:
        del fusion
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def main():
    print("=" * 75)
    print("BASHA - FUSION ERROR DIAGNOSTICS")
    print("=" * 75)

    if not COMPARISON_PATH.is_file():
        raise FileNotFoundError(COMPARISON_PATH)

    if not CHECKPOINT_PATH.is_dir():
        raise FileNotFoundError(CHECKPOINT_PATH)

    original, finetuned = load_comparison()
    groups = select_examples(original, finetuned)

    selected_ids = sorted({
        index
        for indices in groups.values()
        for index in indices
    })

    print("\nSelected diagnostic examples:")
    for name, indices in groups.items():
        print(f"{name:<22}: {len(indices)}")

    print(f"{'Total unique':<22}: {len(selected_ids)}")

    original_records = run_diagnostics(
        "original",
        ORIGINAL_CHECKPOINT,
        selected_ids,
        original,
        finetuned,
    )

    finetuned_records = run_diagnostics(
        "finetuned",
        str(CHECKPOINT_PATH),
        selected_ids,
        original,
        finetuned,
    )

    transitions = Counter()
    decision_changes = Counter()

    for index in selected_ids:
        key = str(index)

        old = original_records[key]
        new = finetuned_records[key]

        transitions[
            f"{old['predicted']} -> {new['predicted']}"
        ] += 1

        decision_changes[
            (
                str(old["decision_source"]),
                str(new["decision_source"]),
            )
        ] += 1

    report = {
        "description": (
            "Read-only original vs fine-tuned "
            "Basha internal fusion diagnostics"
        ),
        "original_checkpoint": ORIGINAL_CHECKPOINT,
        "finetuned_checkpoint": str(CHECKPOINT_PATH),
        "selection_groups": groups,
        "selected_count": len(selected_ids),
        "prediction_transitions": dict(transitions),
        "decision_source_transitions": [
            {
                "original": old,
                "finetuned": new,
                "count": count,
            }
            for (old, new), count
            in decision_changes.most_common()
        ],
        "original": original_records,
        "finetuned": finetuned_records,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 75)
    print("DIAGNOSTICS COMPLETED")
    print("=" * 75)

    print("\nPrediction transitions:")
    for transition, count in transitions.most_common():
        print(f"{transition:<30}: {count}")

    print(f"\nSaved: {OUTPUT_PATH}")
    print("No production source files were modified.")


if __name__ == "__main__":
    main()
