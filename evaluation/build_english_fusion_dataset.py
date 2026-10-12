from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_PATH = Path(
    "evaluation/data/english_fusion_benchmark.json"
)

SEED = 42

NORMAL_PER_CLASS = 100
MIXED_TARGET = 100


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ==========================================================
# STEP 2A — NORMAL SENTIMENT
# ==========================================================

def build_normal_examples() -> List[Dict]:

    print(
        "\nLoading TweetEval sentiment..."
    )

    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment",
    )

    grouped = {
        "negative": [],
        "neutral": [],
        "positive": [],
    }

    for row in dataset["test"]:

        label_id = int(
            row["label"]
        )

        if label_id not in LABEL_MAP:
            continue

        text = row["text"].strip()

        if not text:
            continue

        label = LABEL_MAP[
            label_id
        ]

        grouped[label].append(
            {
                "text": text,
                "label": label,
                "phenomenon":
                    "normal_sentiment",
                "source":
                    "tweet_eval_sentiment",
            }
        )

    rng = random.Random(
        SEED
    )

    selected = []

    for label in [
        "negative",
        "neutral",
        "positive",
    ]:

        candidates = sorted(
            grouped[label],
            key=lambda x: x["text"],
        )

        chosen = rng.sample(
            candidates,
            NORMAL_PER_CLASS,
        )

        selected.extend(
            chosen
        )

    return selected


# ==========================================================
# NORMALIZE ABSA LABEL
# ==========================================================

def normalize_absa_label(
    label,
) -> str:

    """
    Convert the dataset's ABSA label into a lowercase
    string.

    The SetFit ABSA dataset normally exposes labels such as:

        positive
        negative
        neutral
        conflict

    This function also safely handles integer labels if
    the dataset representation changes.
    """

    if isinstance(label, str):

        return (
            label
            .strip()
            .lower()
        )

    # Fallback mapping based on the dataset feature names.

    return str(label).strip().lower()


# ==========================================================
# STEP 2B — REAL MIXED SENTENCES
# ==========================================================

def build_mixed_examples() -> List[Dict]:

    print(
        "\nLoading SemEval ABSA data..."
    )

    dataset = load_dataset(
        "tomaarsen/setfit-absa-semeval-restaurants"
    )

    train = dataset["train"]

    # ------------------------------------------------------
    # Group all aspect annotations belonging to the same
    # sentence.
    #
    # Example:
    #
    # "The battery is excellent but the camera is terrible."
    #
    # sentence
    #    ├── battery → positive
    #    └── camera  → negative
    #
    # therefore document/sentence sentiment = MIXED
    # ------------------------------------------------------

    grouped = defaultdict(
        list
    )

    for row in train:

        text = (
            row["text"]
            .strip()
        )

        if not text:
            continue

        label = normalize_absa_label(
            row["label"]
        )

        aspect = (
            row["span"]
            .strip()
        )

        grouped[text].append(
            {
                "aspect":
                    aspect,

                "label":
                    label,
            }
        )

    mixed_examples = []

    for text, annotations in (
        grouped.items()
    ):

        labels = {
            item["label"]
            for item in annotations
        }

        # --------------------------------------------------
        # Basha's definition of genuine Mixed:
        #
        # At least one gold positive aspect
        # AND
        # at least one gold negative aspect
        #
        # Neutral does not create Mixed by itself.
        # --------------------------------------------------

        has_positive = (
            "positive" in labels
        )

        has_negative = (
            "negative" in labels
        )

        if not (
            has_positive
            and has_negative
        ):
            continue

        positive_aspects = [
            item["aspect"]
            for item in annotations
            if item["label"]
            == "positive"
        ]

        negative_aspects = [
            item["aspect"]
            for item in annotations
            if item["label"]
            == "negative"
        ]

        mixed_examples.append(
            {
                "text":
                    text,

                "label":
                    "mixed",

                "phenomenon":
                    "aspect_mixed",

                "source":
                    "semeval_restaurants_absa",

                "gold_evidence": {
                    "positive_aspects":
                        positive_aspects,

                    "negative_aspects":
                        negative_aspects,
                },
            }
        )

    print(
        "Genuine positive+negative "
        f"sentences available: "
        f"{len(mixed_examples)}"
    )

    # ------------------------------------------------------
    # Freeze selection deterministically.
    # ------------------------------------------------------

    mixed_examples = sorted(
        mixed_examples,
        key=lambda x: x["text"],
    )

    rng = random.Random(
        SEED
    )

    if (
        len(mixed_examples)
        > MIXED_TARGET
    ):

        mixed_examples = rng.sample(
            mixed_examples,
            MIXED_TARGET,
        )

    return mixed_examples


# ==========================================================
# VALIDATION
# ==========================================================

def validate(
    examples: List[Dict],
) -> None:

    print(
        "\nValidating complete dataset..."
    )

    texts = [
        item["text"]
        for item in examples
    ]

    duplicate_count = (
        len(texts)
        - len(set(texts))
    )

    print(
        f"Duplicate texts: "
        f"{duplicate_count}"
    )

    if duplicate_count:

        raise ValueError(
            "Duplicate texts detected."
        )

    labels = Counter(
        item["label"]
        for item in examples
    )

    print(
        "\nLabel distribution:"
    )

    for label in [
        "negative",
        "neutral",
        "positive",
        "mixed",
    ]:

        print(
            f"{label:<10}: "
            f"{labels[label]}"
        )

    phenomena = Counter(
        item["phenomenon"]
        for item in examples
    )

    print(
        "\nPhenomenon distribution:"
    )

    for name, count in sorted(
        phenomena.items()
    ):

        print(
            f"{name:<20}: "
            f"{count}"
        )

    # ------------------------------------------------------
    # Verify every Mixed example really contains both
    # positive and negative gold aspect evidence.
    # ------------------------------------------------------

    for item in examples:

        if (
            item["label"]
            != "mixed"
        ):
            continue

        evidence = item.get(
            "gold_evidence",
            {}
        )

        positive = evidence.get(
            "positive_aspects",
            [],
        )

        negative = evidence.get(
            "negative_aspects",
            [],
        )

        if (
            not positive
            or not negative
        ):

            raise ValueError(
                "Invalid Mixed example:\n"
                f"{item['text']}"
            )

    print(
        "\nValidation passed."
    )


# ==========================================================
# SAVE
# ==========================================================

def save_dataset(
    examples: List[Dict],
) -> None:

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

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "BUILD BASHA ENGLISH FUSION DATASET"
    )

    print(
        "=" * 80
    )

    # ======================================================
    # STEP 2A
    # ======================================================

    normal_examples = (
        build_normal_examples()
    )

    print(
        f"\nFrozen normal examples: "
        f"{len(normal_examples)}"
    )

    # ======================================================
    # STEP 2B
    # ======================================================

    mixed_examples = (
        build_mixed_examples()
    )

    print(
        f"Selected Mixed examples: "
        f"{len(mixed_examples)}"
    )

    # ======================================================
    # COMBINE
    # ======================================================

    examples = (
        normal_examples
        + mixed_examples
    )

    # ------------------------------------------------------
    # Deterministic final shuffle
    # ------------------------------------------------------

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        examples
    )

    # ======================================================
    # VALIDATE
    # ======================================================

    validate(
        examples
    )

    # ======================================================
    # SAVE
    # ======================================================

    save_dataset(
        examples
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2B COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTotal benchmark examples: "
        f"{len(examples)}"
    )

    print(
        f"Normal sentiment: "
        f"{len(normal_examples)}"
    )

    print(
        f"Mixed sentiment:  "
        f"{len(mixed_examples)}"
    )


if __name__ == "__main__":
    main()