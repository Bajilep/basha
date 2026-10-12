from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FROZEN_BENCHMARK = (
    DATA_DIR
    / "english_fusion_benchmark.json"
)

SEMEVAL_TRAIN = (
    DATA_DIR
    / "absa_finetune_train.json"
)

SEMEVAL_VALIDATION = (
    DATA_DIR
    / "absa_finetune_validation.json"
)

MAMS_RAW = (
    DATA_DIR
    / "MAMS-ATSA"
    / "raw"
)

MAMS_TRAIN = (
    MAMS_RAW
    / "train.xml"
)

MAMS_VALIDATION = (
    MAMS_RAW
    / "val.xml"
)

MAMS_TEST = (
    MAMS_RAW
    / "test.xml"
)

FINAL_TRAIN = (
    DATA_DIR
    / "absa_final_train.json"
)

FINAL_VALIDATION = (
    DATA_DIR
    / "absa_final_validation.json"
)

FINAL_MAMS_TEST = (
    DATA_DIR
    / "absa_mams_test.json"
)

REPORT_FILE = (
    BASE_DIR
    / "results"
    / "absa_final_dataset_report.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


def normalize_polarity(
    polarity: str | None,
) -> str | None:

    if polarity is None:
        return None

    value = (
        str(polarity)
        .strip()
        .lower()
    )

    mapping = {
        "positive": "positive",
        "pos": "positive",

        "negative": "negative",
        "neg": "negative",

        "neutral": "neutral",
        "neu": "neutral",

        "conflict": "conflict",
    }

    return mapping.get(
        value,
        value,
    )


# ============================================================
# FILE CHECK
# ============================================================

def check_required_files():

    files = [
        FROZEN_BENCHMARK,
        SEMEVAL_TRAIN,
        SEMEVAL_VALIDATION,
        MAMS_TRAIN,
        MAMS_VALIDATION,
        MAMS_TEST,
    ]

    print("\nChecking required files...")
    print("-" * 70)

    missing = []

    for path in files:

        if path.exists():
            print(
                f"✓ {path}"
            )
        else:
            print(
                f"✗ {path}"
            )
            missing.append(path)

    if missing:

        raise FileNotFoundError(
            "One or more required files "
            "are missing."
        )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    path: Path,
    data,
):

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
# FROZEN BENCHMARK
# ============================================================

def load_frozen_texts():

    data = load_json(
        FROZEN_BENCHMARK
    )

    if isinstance(data, dict):

        if "examples" in data:
            data = data["examples"]

        elif "data" in data:
            data = data["data"]

    frozen = set()

    for item in data:

        text = item.get("text")

        if text:

            frozen.add(
                normalize_text(text)
            )

    return frozen


# ============================================================
# SEMEVAL
# ============================================================

def convert_semeval_examples(
    path: Path,
    split_name: str,
):

    data = load_json(path)

    converted = []

    for example in data:

        aspects = []

        for aspect in example.get(
            "aspects",
            []
        ):

            term = str(
                aspect.get(
                    "aspect",
                    ""
                )
            ).strip()

            polarity = (
                normalize_polarity(
                    aspect.get("label")
                )
            )

            if not term:
                continue

            if polarity not in {
                "positive",
                "negative",
                "neutral",
                "conflict",
            }:
                continue

            aspects.append(
                {
                    "aspect": term,
                    "polarity": polarity,
                    "from": None,
                    "to": None,
                }
            )

        if not aspects:
            continue

        converted.append(
            {
                "text":
                    example["text"],

                "aspects":
                    aspects,

                "source":
                    "semeval",

                "original_split":
                    split_name,
            }
        )

    return converted


# ============================================================
# MAMS XML
# ============================================================

def read_mams(
    path: Path,
    split_name: str,
):

    tree = ET.parse(path)

    root = tree.getroot()

    output = []

    stats = Counter()

    for sentence in root.findall(
        ".//sentence"
    ):

        text_element = (
            sentence.find("text")
        )

        if (
            text_element is None
            or not text_element.text
        ):
            continue

        text = (
            text_element.text.strip()
        )

        aspects = []

        for element in (
            sentence.findall(
                ".//aspectTerm"
            )
        ):

            term = (
                element.attrib.get(
                    "term",
                    ""
                )
                .strip()
            )

            polarity = (
                normalize_polarity(
                    element.attrib.get(
                        "polarity"
                    )
                )
            )

            start_raw = (
                element.attrib.get(
                    "from"
                )
            )

            end_raw = (
                element.attrib.get(
                    "to"
                )
            )

            if not term:
                continue

            if polarity not in {
                "positive",
                "negative",
                "neutral",
            }:
                continue

            try:
                start = int(start_raw)
                end = int(end_raw)

            except (
                TypeError,
                ValueError,
            ):

                stats[
                    "invalid_span"
                ] += 1

                continue

            # --------------------------------------------
            # Validate character span
            # --------------------------------------------

            actual = text[
                start:end
            ]

            if (
                normalize_text(actual)
                !=
                normalize_text(term)
            ):

                stats[
                    "invalid_span"
                ] += 1

                continue

            aspects.append(
                {
                    "aspect": term,
                    "polarity":
                        polarity,
                    "from": start,
                    "to": end,
                }
            )

            stats[
                "valid_aspects"
            ] += 1

        if not aspects:

            stats[
                "sentences_removed_no_valid_aspects"
            ] += 1

            continue

        output.append(
            {
                "text": text,
                "aspects": aspects,
                "source": "mams",
                "original_split":
                    split_name,
            }
        )

    stats[
        "sentences"
    ] = len(output)

    return output, dict(stats)


# ============================================================
# REMOVE FROZEN BENCHMARK
# ============================================================

def remove_frozen_overlap(
    examples,
    frozen_texts,
):

    safe = []
    removed = []

    for example in examples:

        normalized = normalize_text(
            example["text"]
        )

        if normalized in frozen_texts:

            removed.append(
                example
            )

        else:

            safe.append(
                example
            )

    return safe, removed


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    examples,
):

    seen = set()

    output = []

    duplicates = []

    for example in examples:

        normalized = normalize_text(
            example["text"]
        )

        if normalized in seen:

            duplicates.append(
                example
            )

            continue

        seen.add(normalized)

        output.append(example)

    return output, duplicates


# ============================================================
# CROSS-SPLIT SAFETY
# ============================================================

def remove_validation_train_overlap(
    train,
    validation,
):

    train_texts = {
        normalize_text(
            item["text"]
        )
        for item in train
    }

    safe_validation = []

    removed = []

    for example in validation:

        normalized = normalize_text(
            example["text"]
        )

        if normalized in train_texts:

            removed.append(
                example
            )

        else:

            safe_validation.append(
                example
            )

    return (
        safe_validation,
        removed,
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    examples,
):

    polarity = Counter()

    source = Counter()

    mixed = 0

    total_aspects = 0

    for example in examples:

        source[
            example["source"]
        ] += 1

        sentiments = set()

        for aspect in example[
            "aspects"
        ]:

            label = aspect[
                "polarity"
            ]

            polarity[label] += 1

            sentiments.add(label)

            total_aspects += 1

        if (
            "positive" in sentiments
            and
            "negative" in sentiments
        ):
            mixed += 1

    return {
        "sentences":
            len(examples),

        "aspects":
            total_aspects,

        "positive":
            polarity["positive"],

        "negative":
            polarity["negative"],

        "neutral":
            polarity["neutral"],

        "conflict":
            polarity["conflict"],

        "mixed_sentences":
            mixed,

        "source_counts":
            dict(source),
    }


# ============================================================
# VERIFY LEAKAGE
# ============================================================

def verify_safety(
    train,
    validation,
    mams_test,
    frozen_texts,
):

    train_texts = {
        normalize_text(x["text"])
        for x in train
    }

    validation_texts = {
        normalize_text(x["text"])
        for x in validation
    }

    test_texts = {
        normalize_text(x["text"])
        for x in mams_test
    }

    checks = {
        "train_vs_validation":
            len(
                train_texts
                & validation_texts
            ),

        "train_vs_mams_test":
            len(
                train_texts
                & test_texts
            ),

        "validation_vs_mams_test":
            len(
                validation_texts
                & test_texts
            ),

        "train_vs_frozen":
            len(
                train_texts
                & frozen_texts
            ),

        "validation_vs_frozen":
            len(
                validation_texts
                & frozen_texts
            ),

        "mams_test_vs_frozen":
            len(
                test_texts
                & frozen_texts
            ),
    }

    return checks


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(
    title,
    stats,
):

    print(
        "\n" + title
    )

    print("-" * 60)

    print(
        f"Sentences:       "
        f"{stats['sentences']}"
    )

    print(
        f"Aspects:         "
        f"{stats['aspects']}"
    )

    print(
        f"Positive:        "
        f"{stats['positive']}"
    )

    print(
        f"Negative:        "
        f"{stats['negative']}"
    )

    print(
        f"Neutral:         "
        f"{stats['neutral']}"
    )

    print(
        f"Conflict:        "
        f"{stats['conflict']}"
    )

    print(
        f"Mixed sentences: "
        f"{stats['mixed_sentences']}"
    )

    print(
        f"Sources:         "
        f"{stats['source_counts']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "BASHA — FINAL ABSA "
        "FINE-TUNING DATASET BUILDER"
    )

    print("=" * 75)

    check_required_files()

    # ========================================================
    # Frozen benchmark
    # ========================================================

    frozen_texts = (
        load_frozen_texts()
    )

    print(
        f"\nFrozen Basha benchmark: "
        f"{len(frozen_texts)} sentences"
    )

    # ========================================================
    # Safe SemEval
    # ========================================================

    semeval_train = (
        convert_semeval_examples(
            SEMEVAL_TRAIN,
            "train",
        )
    )

    semeval_validation = (
        convert_semeval_examples(
            SEMEVAL_VALIDATION,
            "validation",
        )
    )

    print(
        f"\nSafe SemEval train: "
        f"{len(semeval_train)}"
    )

    print(
        f"Safe SemEval validation: "
        f"{len(semeval_validation)}"
    )

    # ========================================================
    # MAMS
    # ========================================================

    mams_train, mams_train_info = (
        read_mams(
            MAMS_TRAIN,
            "train",
        )
    )

    (
        mams_validation,
        mams_validation_info,
    ) = read_mams(
        MAMS_VALIDATION,
        "validation",
    )

    mams_test, mams_test_info = (
        read_mams(
            MAMS_TEST,
            "test",
        )
    )

    print(
        "\nMAMS valid data loaded:"
    )

    print(
        "Train:",
        mams_train_info,
    )

    print(
        "Validation:",
        mams_validation_info,
    )

    print(
        "Test:",
        mams_test_info,
    )

    # ========================================================
    # Remove frozen benchmark from ALL MAMS data
    # ========================================================

    (
        mams_train,
        removed_train_frozen,
    ) = remove_frozen_overlap(
        mams_train,
        frozen_texts,
    )

    (
        mams_validation,
        removed_validation_frozen,
    ) = remove_frozen_overlap(
        mams_validation,
        frozen_texts,
    )

    (
        mams_test,
        removed_test_frozen,
    ) = remove_frozen_overlap(
        mams_test,
        frozen_texts,
    )

    print(
        "\nMAMS ↔ Basha benchmark overlaps removed:"
    )

    print(
        "Train:",
        len(removed_train_frozen),
    )

    print(
        "Validation:",
        len(
            removed_validation_frozen
        ),
    )

    print(
        "Test:",
        len(removed_test_frozen),
    )

    # ========================================================
    # Combine
    # ========================================================

    final_train = (
        semeval_train
        + mams_train
    )

    final_validation = (
        semeval_validation
        + mams_validation
    )

    # ========================================================
    # Remove internal duplicates
    # ========================================================

    (
        final_train,
        train_duplicates,
    ) = remove_duplicates(
        final_train
    )

    (
        final_validation,
        validation_duplicates,
    ) = remove_duplicates(
        final_validation
    )

    (
        mams_test,
        test_duplicates,
    ) = remove_duplicates(
        mams_test
    )

    # ========================================================
    # Ensure validation not in training
    # ========================================================

    (
        final_validation,
        validation_train_overlap,
    ) = remove_validation_train_overlap(
        final_train,
        final_validation,
    )

    # ========================================================
    # Keep test isolated from train/validation
    # ========================================================

    train_texts = {
        normalize_text(x["text"])
        for x in final_train
    }

    validation_texts = {
        normalize_text(x["text"])
        for x in final_validation
    }

    safe_test = []

    test_cross_overlap = []

    for example in mams_test:

        normalized = normalize_text(
            example["text"]
        )

        if (
            normalized in train_texts
            or
            normalized in validation_texts
        ):

            test_cross_overlap.append(
                example
            )

        else:

            safe_test.append(
                example
            )

    mams_test = safe_test

    # ========================================================
    # Stats
    # ========================================================

    train_stats = calculate_stats(
        final_train
    )

    validation_stats = calculate_stats(
        final_validation
    )

    test_stats = calculate_stats(
        mams_test
    )

    print_stats(
        "FINAL TRAINING DATA",
        train_stats,
    )

    print_stats(
        "FINAL VALIDATION DATA",
        validation_stats,
    )

    print_stats(
        "UNTOUCHED MAMS TEST DATA",
        test_stats,
    )

    # ========================================================
    # Safety
    # ========================================================

    checks = verify_safety(
        final_train,
        final_validation,
        mams_test,
        frozen_texts,
    )

    print(
        "\nFINAL LEAKAGE CHECK"
    )

    print("-" * 60)

    for name, value in (
        checks.items()
    ):

        print(
            f"{name:<30} {value}"
        )

    # Training and validation MUST be completely
    # isolated from frozen benchmark and each other.
    required_zero = [
        "train_vs_validation",
        "train_vs_mams_test",
        "validation_vs_mams_test",
        "train_vs_frozen",
        "validation_vs_frozen",
    ]

    failed = [
        key
        for key in required_zero
        if checks[key] != 0
    ]

    if failed:

        raise RuntimeError(
            "Dataset leakage remains: "
            + ", ".join(failed)
        )

    print(
        "\n✓ Training, validation and "
        "test datasets are safely isolated."
    )

    # ========================================================
    # Save
    # ========================================================

    save_json(
        FINAL_TRAIN,
        final_train,
    )

    save_json(
        FINAL_VALIDATION,
        final_validation,
    )

    save_json(
        FINAL_MAMS_TEST,
        mams_test,
    )

    report = {
        "train":
            train_stats,

        "validation":
            validation_stats,

        "mams_test":
            test_stats,

        "mams_invalid_span_info": {
            "train":
                mams_train_info,

            "validation":
                mams_validation_info,

            "test":
                mams_test_info,
        },

        "removed_frozen_overlap": {
            "mams_train":
                len(
                    removed_train_frozen
                ),

            "mams_validation":
                len(
                    removed_validation_frozen
                ),

            "mams_test":
                len(
                    removed_test_frozen
                ),
        },

        "removed_duplicates": {
            "train":
                len(train_duplicates),

            "validation":
                len(
                    validation_duplicates
                ),

            "test":
                len(test_duplicates),

            "validation_vs_train":
                len(
                    validation_train_overlap
                ),

            "test_cross_split":
                len(
                    test_cross_overlap
                ),
        },

        "leakage_checks":
            checks,
    }

    save_json(
        REPORT_FILE,
        report,
    )

    print(
        "\nSaved:"
    )

    print(FINAL_TRAIN)
    print(FINAL_VALIDATION)
    print(FINAL_MAMS_TEST)
    print(REPORT_FILE)

    print(
        "\n" + "=" * 75
    )

    print(
        "FINAL DATASET BUILD COMPLETE"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()