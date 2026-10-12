from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path(
    "evaluation/data/sarcasm_fusion_subset.json"
)

SEED = 42

SARCASM_COUNT = 100
NON_SARCASM_COUNT = 100


def main():

    print("=" * 80)
    print("BUILD BASHA SARCASM FUSION SUBSET")
    print("=" * 80)

    print("\nLoading TweetEval irony...")

    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "irony",
    )

    test = dataset["test"]

    sarcasm_examples = []
    normal_examples = []

    for row in test:

        text = row["text"].strip()

        if not text:
            continue

        label = int(row["label"])

        item = {
            "text": text,
            "sarcasm": bool(label == 1),
            "phenomenon": (
                "sarcasm"
                if label == 1
                else "non_sarcasm"
            ),
            "source": "tweet_eval_irony",
        }

        if label == 1:
            sarcasm_examples.append(item)

        elif label == 0:
            normal_examples.append(item)

    print(
        f"\nSarcasm available:     "
        f"{len(sarcasm_examples)}"
    )

    print(
        f"Non-sarcasm available: "
        f"{len(normal_examples)}"
    )

    # ------------------------------------------------------
    # Deterministic ordering
    # ------------------------------------------------------

    sarcasm_examples = sorted(
        sarcasm_examples,
        key=lambda x: x["text"],
    )

    normal_examples = sorted(
        normal_examples,
        key=lambda x: x["text"],
    )

    rng = random.Random(SEED)

    sarcasm_selected = rng.sample(
        sarcasm_examples,
        min(
            SARCASM_COUNT,
            len(sarcasm_examples),
        ),
    )

    normal_selected = rng.sample(
        normal_examples,
        min(
            NON_SARCASM_COUNT,
            len(normal_examples),
        ),
    )

    examples = (
        sarcasm_selected
        + normal_selected
    )

    rng.shuffle(examples)

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    texts = [
        item["text"]
        for item in examples
    ]

    duplicate_count = (
        len(texts)
        - len(set(texts))
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate texts."
        )

    distribution = Counter(
        item["sarcasm"]
        for item in examples
    )

    print("\nSelected distribution:")

    print(
        f"Sarcasm:     "
        f"{distribution[True]}"
    )

    print(
        f"Non-sarcasm: "
        f"{distribution[False]}"
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            examples,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nValidation passed.")

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 80)
    print("STEP 2C COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal sarcasm benchmark examples: "
        f"{len(examples)}"
    )


if __name__ == "__main__":
    main()