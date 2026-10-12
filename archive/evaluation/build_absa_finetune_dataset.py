from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
VALIDATION_RATIO = 0.10

BASE_DIR = Path(__file__).resolve().parent

FROZEN_BENCHMARK = (
    BASE_DIR
    / "data"
    / "english_fusion_benchmark.json"
)

TRAIN_OUTPUT = (
    BASE_DIR
    / "data"
    / "absa_finetune_train.json"
)

VALIDATION_OUTPUT = (
    BASE_DIR
    / "data"
    / "absa_finetune_validation.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize only for leakage checking.

    We preserve the ORIGINAL sentence in the saved dataset.
    """
    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


# ============================================================
# LOAD FROZEN BASHA BENCHMARK
# ============================================================

def load_frozen_texts() -> set[str]:
    if not FROZEN_BENCHMARK.exists():
        raise FileNotFoundError(
            f"Frozen benchmark not found:\n"
            f"{FROZEN_BENCHMARK}"
        )

    with FROZEN_BENCHMARK.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "examples" in data:
            data = data["examples"]
        elif "data" in data:
            data = data["data"]

    if not isinstance(data, list):
        raise TypeError(
            "english_fusion_benchmark.json "
            "must contain a list of examples."
        )

    frozen = set()

    for item in data:
        text = item.get("text")

        if text:
            frozen.add(
                normalize_text(text)
            )

    return frozen


# ============================================================
# LOAD SEMEVAL
# ============================================================

def load_semeval_rows():
    print("\nLoading SemEval restaurant dataset...")

    dataset = load_dataset(
        "tomaarsen/setfit-absa-semeval-restaurants"
    )

    # We use ONLY the original training split.
    rows = dataset["train"]

    print(
        f"Original annotation rows: {len(rows)}"
    )

    return rows


# ============================================================
# GROUP ASPECTS BY SENTENCE
# ============================================================

def group_by_sentence(rows):
    """
    SemEval contains one row per aspect.

    Example:

        The food is good but service is bad.
        food    -> positive
        service -> negative

    We group those rows back into one sentence.
    """

    grouped = defaultdict(list)
    original_text = {}

    for row in rows:
        text = str(row["text"])
        normalized = normalize_text(text)

        original_text[normalized] = text

        aspect = str(row["span"]).strip()
        label = str(row["label"]).strip().lower()

        if label not in {
            "positive",
            "negative",
            "neutral",
            "conflict",
        }:
            continue

        grouped[normalized].append(
            {
                "aspect": aspect,
                "label": label,
            }
        )

    examples = []

    for normalized, aspects in grouped.items():
        examples.append(
            {
                "text": original_text[normalized],
                "aspects": aspects,
            }
        )

    return examples


# ============================================================
# REMOVE BENCHMARK LEAKAGE
# ============================================================

def remove_frozen_examples(
    examples,
    frozen_texts,
):
    safe_examples = []
    removed_examples = []

    for example in examples:
        normalized = normalize_text(
            example["text"]
        )

        if normalized in frozen_texts:
            removed_examples.append(example)
        else:
            safe_examples.append(example)

    return safe_examples, removed_examples


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(examples):
    rng = random.Random(SEED)

    examples = list(examples)

    # Deterministic starting order.
    examples.sort(
        key=lambda item: normalize_text(
            item["text"]
        )
    )

    rng.shuffle(examples)

    validation_size = max(
        1,
        round(
            len(examples)
            * VALIDATION_RATIO
        ),
    )

    validation = examples[
        :validation_size
    ]

    train = examples[
        validation_size:
    ]

    return train, validation


# ============================================================
# SAFETY CHECKS
# ============================================================

def verify_no_overlap(
    train,
    validation,
    frozen_texts,
):
    train_texts = {
        normalize_text(item["text"])
        for item in train
    }

    validation_texts = {
        normalize_text(item["text"])
        for item in validation
    }

    train_validation_overlap = (
        train_texts
        & validation_texts
    )

    train_frozen_overlap = (
        train_texts
        & frozen_texts
    )

    validation_frozen_overlap = (
        validation_texts
        & frozen_texts
    )

    print("\nLEAKAGE CHECK")
    print("-" * 50)

    print(
        "Train ↔ Validation overlap:",
        len(train_validation_overlap),
    )

    print(
        "Train ↔ Frozen benchmark overlap:",
        len(train_frozen_overlap),
    )

    print(
        "Validation ↔ Frozen benchmark overlap:",
        len(validation_frozen_overlap),
    )

    if train_validation_overlap:
        raise RuntimeError(
            "ERROR: Train and validation overlap!"
        )

    if train_frozen_overlap:
        raise RuntimeError(
            "ERROR: Frozen benchmark leaked "
            "into training!"
        )

    if validation_frozen_overlap:
        raise RuntimeError(
            "ERROR: Frozen benchmark leaked "
            "into validation!"
        )

    print("\n✓ No leakage detected.")


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(name, examples):
    labels = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "conflict": 0,
    }

    total_aspects = 0

    mixed_sentences = 0

    for example in examples:
        sentiments = set()

        for aspect in example["aspects"]:
            label = aspect["label"]

            labels[label] += 1
            total_aspects += 1

            sentiments.add(label)

        if (
            "positive" in sentiments
            and "negative" in sentiments
        ):
            mixed_sentences += 1

    print(f"\n{name}")
    print("-" * 50)

    print(
        f"Sentences:       {len(examples)}"
    )

    print(
        f"Aspect labels:   {total_aspects}"
    )

    print(
        f"Positive:        {labels['positive']}"
    )

    print(
        f"Negative:        {labels['negative']}"
    )

    print(
        f"Neutral:         {labels['neutral']}"
    )

    print(
        f"Conflict:        {labels['conflict']}"
    )

    print(
        f"Mixed sentences: {mixed_sentences}"
    )


# ============================================================
# SAVE
# ============================================================

def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(
        "BASHA ABSA FINE-TUNING DATASET BUILDER"
    )
    print("=" * 70)

    print(
        "\nFrozen benchmark:"
    )
    print(FROZEN_BENCHMARK)

    frozen_texts = load_frozen_texts()

    print(
        f"\nFrozen benchmark sentences: "
        f"{len(frozen_texts)}"
    )

    rows = load_semeval_rows()

    examples = group_by_sentence(rows)

    print(
        f"\nUnique SemEval sentences: "
        f"{len(examples)}"
    )

    safe_examples, removed = (
        remove_frozen_examples(
            examples,
            frozen_texts,
        )
    )

    print(
        f"Removed because they occur "
        f"in frozen benchmark: "
        f"{len(removed)}"
    )

    print(
        f"Remaining safe sentences: "
        f"{len(safe_examples)}"
    )

    train, validation = split_dataset(
        safe_examples
    )

    verify_no_overlap(
        train,
        validation,
        frozen_texts,
    )

    print_statistics(
        "TRAINING DATA",
        train,
    )

    print_statistics(
        "VALIDATION DATA",
        validation,
    )

    save_json(
        TRAIN_OUTPUT,
        train,
    )

    save_json(
        VALIDATION_OUTPUT,
        validation,
    )

    print("\nSaved:")
    print(TRAIN_OUTPUT)
    print(VALIDATION_OUTPUT)

    print()
    print("=" * 70)
    print(
        "DATASET BUILD COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()