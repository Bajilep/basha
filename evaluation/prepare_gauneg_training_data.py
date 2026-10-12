from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gauneg/roberta-base-absa-ate-sentiment"
MAX_LENGTH = 512

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "absa_final_train.json"
VALIDATION_FILE = DATA_DIR / "absa_final_validation.json"

OUTPUT_TRAIN = DATA_DIR / "gauneg_tokenized_train.json"
OUTPUT_VALIDATION = DATA_DIR / "gauneg_tokenized_validation.json"

REPORT_FILE = RESULTS_DIR / "gauneg_alignment_report.json"
PROBLEM_FILE = RESULTS_DIR / "gauneg_alignment_problems.json"


# ============================================================
# EXACT GAUNEG LABELS
# ============================================================

LABEL2ID = {
    "O": 0,
    "B-neu": 1,
    "I-neu": 2,
    "B-neg": 3,
    "B-con": 4,
    "I-pos": 5,
    "B-pos": 6,
    "I-con": 7,
    "I-neg": 8,
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}

IGNORE_INDEX = -100

POLARITY_TO_TAG = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
    "conflict": "con",
}


# ============================================================
# TOKENIZER
# ============================================================

print("Loading Gauneg tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if not tokenizer.is_fast:
    raise RuntimeError(
        "Fast tokenizer required for offset mapping."
    )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path):

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# TEXT NORMALIZATION FOR COMPARISON
# ============================================================

def normalize_term(value: str) -> str:

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


# ============================================================
# BOUNDARY-AWARE TERM SEARCH
# ============================================================

def find_term_occurrences(
    text: str,
    term: str,
) -> list[tuple[int, int]]:

    """
    Find aspect term occurrences without matching inside
    larger words.

    Examples:

        "bar" will NOT match "bartender"
        "food" will NOT match "seafood"
        "salad" will NOT match "salade"

    Matching remains case-insensitive.
    """

    term = term.strip()

    if not term:
        return []

    pattern = (
        r"(?<!\w)"
        + re.escape(term)
        + r"(?!\w)"
    )

    matches = re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    return [
        (
            match.start(),
            match.end(),
        )
        for match in matches
    ]


# ============================================================
# CHECK ORIGINAL SPAN
# ============================================================

def validate_original_span(
    text: str,
    term: str,
    start,
    end,
) -> Optional[tuple[int, int]]:

    if not (
        isinstance(start, int)
        and isinstance(end, int)
    ):
        return None

    if not (
        0 <= start < end <= len(text)
    ):
        return None

    actual = text[start:end]

    if (
        normalize_term(actual)
        !=
        normalize_term(term)
    ):
        return None

    return start, end


# ============================================================
# REMOVE EXACT DUPLICATE ANNOTATIONS
# ============================================================

def deduplicate_annotations(
    aspects: list[dict],
) -> tuple[list[dict], int]:

    unique = []
    seen = set()
    duplicates = 0

    for aspect in aspects:

        key = (
            normalize_term(
                aspect.get(
                    "aspect",
                    ""
                )
            ),
            str(
                aspect.get(
                    "polarity",
                    ""
                )
            ).strip().lower(),
            aspect.get("from"),
            aspect.get("to"),
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        unique.append(aspect)

    return unique, duplicates


# ============================================================
# RESOLVE ASPECT SPANS
# ============================================================

def resolve_aspects(
    example: dict,
):

    text = example["text"]

    raw_aspects = example.get(
        "aspects",
        []
    )

    raw_aspects, duplicate_count = (
        deduplicate_annotations(
            raw_aspects
        )
    )

    stats = Counter()

    stats[
        "duplicate_annotations_removed"
    ] += duplicate_count

    resolved = []
    problems = []

    # Used occurrence is tracked only for annotations
    # requiring reconstructed spans.
    recovered_used = set()

    for aspect in raw_aspects:

        term = str(
            aspect.get(
                "aspect",
                ""
            )
        ).strip()

        polarity = str(
            aspect.get(
                "polarity",
                ""
            )
        ).strip().lower()

        if not term:

            stats["empty_term"] += 1

            problems.append({
                "type": "empty_term",
                "text": text,
                "aspect": aspect,
            })

            continue

        if polarity not in POLARITY_TO_TAG:

            stats[
                "unsupported_polarity"
            ] += 1

            problems.append({
                "type":
                    "unsupported_polarity",

                "text":
                    text,

                "aspect":
                    aspect,
            })

            continue

        # ----------------------------------------------------
        # First trust a VALID original character span.
        # ----------------------------------------------------

        span = validate_original_span(
            text=text,
            term=term,
            start=aspect.get("from"),
            end=aspect.get("to"),
        )

        span_source = "original"

        # ----------------------------------------------------
        # Otherwise reconstruct carefully.
        # ----------------------------------------------------

        if span is None:

            occurrences = (
                find_term_occurrences(
                    text,
                    term,
                )
            )

            available = [
                candidate
                for candidate in occurrences
                if candidate not in recovered_used
            ]

            if not available:

                stats[
                    "span_not_found"
                ] += 1

                problems.append({
                    "type":
                        "span_not_found",

                    "text":
                        text,

                    "aspect":
                        term,

                    "polarity":
                        polarity,

                    "source":
                        example.get(
                            "source"
                        ),

                    "occurrences":
                        occurrences,
                })

                continue

            # If the same aspect appears multiple times
            # and no original span tells us which one,
            # assign the first still-unused exact occurrence.
            span = available[0]

            recovered_used.add(span)

            span_source = "recovered"

            stats[
                "recovered_span"
            ] += 1

        else:

            stats[
                "original_span"
            ] += 1

        resolved.append({
            "aspect":
                term,

            "polarity":
                polarity,

            "from":
                span[0],

            "to":
                span[1],

            "span_source":
                span_source,
        })

    return resolved, stats, problems


# ============================================================
# OVERLAP
# ============================================================

def spans_overlap(
    first: dict,
    second: dict,
) -> bool:

    return (
        first["from"] < second["to"]
        and
        second["from"] < first["to"]
    )


# ============================================================
# RESOLVE NESTED / CONFLICTING ASPECTS
# ============================================================

def resolve_overlaps(
    aspects: list[dict],
    text: str,
):

    """
    Token classification cannot represent overlapping
    BIO aspect spans.

    Policy:

    1. Exact same span + same polarity:
       keep one.

    2. Exact same span + different polarity:
       drop both from token training.

    3. Nested/overlapping spans:
       prefer the longer/more specific span.

    Nothing is changed in the original dataset.
    """

    stats = Counter()
    problems = []

    if len(aspects) <= 1:
        return aspects, stats, problems

    # Longer aspects first.
    ordered = sorted(
        aspects,
        key=lambda x: (
            -(x["to"] - x["from"]),
            x["from"],
            x["to"],
        ),
    )

    kept = []

    for candidate in ordered:

        reject_candidate = False
        remove_existing = []

        for existing in kept:

            if not spans_overlap(
                candidate,
                existing,
            ):
                continue

            same_span = (
                candidate["from"]
                == existing["from"]
                and
                candidate["to"]
                == existing["to"]
            )

            # ------------------------------------------------
            # Same span
            # ------------------------------------------------

            if same_span:

                if (
                    candidate["polarity"]
                    ==
                    existing["polarity"]
                ):

                    stats[
                        "same_span_duplicate_removed"
                    ] += 1

                    reject_candidate = True

                    break

                # Same characters but incompatible sentiment.
                stats[
                    "same_span_polarity_conflict"
                ] += 1

                problems.append({
                    "type":
                        "same_span_polarity_conflict",

                    "text":
                        text,

                    "first":
                        existing,

                    "second":
                        candidate,
                })

                remove_existing.append(
                    existing
                )

                reject_candidate = True

                break

            # ------------------------------------------------
            # Nested/partial overlap
            # ------------------------------------------------

            existing_length = (
                existing["to"]
                - existing["from"]
            )

            candidate_length = (
                candidate["to"]
                - candidate["from"]
            )

            if candidate_length > existing_length:

                remove_existing.append(
                    existing
                )

                stats[
                    "overlap_shorter_removed"
                ] += 1

                problems.append({
                    "type":
                        "overlap_shorter_removed",

                    "text":
                        text,

                    "kept":
                        candidate,

                    "removed":
                        existing,
                })

            else:

                stats[
                    "overlap_shorter_removed"
                ] += 1

                problems.append({
                    "type":
                        "overlap_shorter_removed",

                    "text":
                        text,

                    "kept":
                        existing,

                    "removed":
                        candidate,
                })

                reject_candidate = True

                break

        for item in remove_existing:

            if item in kept:
                kept.remove(item)

        if not reject_candidate:
            kept.append(candidate)

    kept.sort(
        key=lambda x: (
            x["from"],
            x["to"],
        )
    )

    return kept, stats, problems


# ============================================================
# TOKEN OVERLAP
# ============================================================

def token_overlaps_span(
    token_start: int,
    token_end: int,
    aspect_start: int,
    aspect_end: int,
) -> bool:

    return (
        token_start < aspect_end
        and
        token_end > aspect_start
    )


# ============================================================
# ALIGN ONE EXAMPLE
# ============================================================

def align_example(
    example: dict,
):

    stats = Counter()
    problems = []

    text = example["text"]

    (
        aspects,
        span_stats,
        span_problems,
    ) = resolve_aspects(
        example
    )

    stats.update(span_stats)
    problems.extend(span_problems)

    (
        aspects,
        overlap_stats,
        overlap_problems,
    ) = resolve_overlaps(
        aspects,
        text,
    )

    stats.update(overlap_stats)
    problems.extend(
        overlap_problems
    )

    if not aspects:

        stats[
            "example_no_valid_aspects"
        ] += 1

        return (
            None,
            stats,
            problems,
        )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids = encoded[
        "input_ids"
    ]

    attention_mask = encoded[
        "attention_mask"
    ]

    offsets = encoded[
        "offset_mapping"
    ]

    tokens = (
        tokenizer.convert_ids_to_tokens(
            input_ids
        )
    )

    labels = [
        LABEL2ID["O"]
        for _ in input_ids
    ]

    # Special tokens ignored by CrossEntropyLoss.
    for index, (
        token_start,
        token_end,
    ) in enumerate(offsets):

        if (
            token_start == 0
            and token_end == 0
        ):

            labels[index] = (
                IGNORE_INDEX
            )

    aligned_aspects = []

    for aspect in aspects:

        token_indices = []

        for index, (
            token_start,
            token_end,
        ) in enumerate(offsets):

            if (
                token_start == 0
                and token_end == 0
            ):
                continue

            if token_overlaps_span(
                token_start,
                token_end,
                aspect["from"],
                aspect["to"],
            ):

                token_indices.append(
                    index
                )

        if not token_indices:

            stats[
                "aspect_not_aligned"
            ] += 1

            problems.append({
                "type":
                    "aspect_not_aligned",

                "text":
                    text,

                "aspect":
                    aspect,
            })

            continue

        # This should now be zero after overlap cleanup.
        collision = any(
            labels[index]
            != LABEL2ID["O"]

            for index in token_indices
        )

        if collision:

            stats[
                "token_collision"
            ] += 1

            problems.append({
                "type":
                    "token_collision",

                "text":
                    text,

                "aspect":
                    aspect,
            })

            continue

        short_tag = (
            POLARITY_TO_TAG[
                aspect["polarity"]
            ]
        )

        labels[
            token_indices[0]
        ] = LABEL2ID[
            f"B-{short_tag}"
        ]

        for token_index in (
            token_indices[1:]
        ):

            labels[
                token_index
            ] = LABEL2ID[
                f"I-{short_tag}"
            ]

        aligned_aspects.append(
            aspect
        )

        stats[
            "aspect_aligned"
        ] += 1

    if not aligned_aspects:

        stats[
            "example_no_aligned_aspects"
        ] += 1

        return (
            None,
            stats,
            problems,
        )

    output = {
        "text":
            text,

        "source":
            example.get("source"),

        "original_split":
            example.get(
                "original_split"
            ),

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "labels":
            labels,

        "tokens":
            tokens,

        "offset_mapping": [
            [start, end]
            for start, end
            in offsets
        ],

        "aspects":
            aligned_aspects,
    }

    return (
        output,
        stats,
        problems,
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    examples,
    split_name,
):

    outputs = []
    stats = Counter()
    problems = []

    print(
        f"\nProcessing {split_name}..."
    )

    print("-" * 60)

    for index, example in enumerate(
        examples,
        start=1,
    ):

        (
            output,
            example_stats,
            example_problems,
        ) = align_example(
            example
        )

        stats.update(
            example_stats
        )

        problems.extend(
            example_problems
        )

        if output is not None:
            outputs.append(output)

        if (
            index % 500 == 0
            or index == len(examples)
        ):

            print(
                f"{index}/"
                f"{len(examples)}"
            )

    stats[
        "input_examples"
    ] = len(examples)

    stats[
        "output_examples"
    ] = len(outputs)

    return (
        outputs,
        dict(stats),
        problems,
    )


# ============================================================
# LABEL COUNTS
# ============================================================

def count_labels(
    examples,
):

    counts = Counter()

    for example in examples:

        for label_id in (
            example["labels"]
        ):

            if label_id == IGNORE_INDEX:
                label = "X"
            else:
                label = ID2LABEL[
                    label_id
                ]

            counts[label] += 1

    return dict(counts)


# ============================================================
# MANUAL DISPLAY
# ============================================================

def print_example(
    example,
    number,
):

    print(
        "\n" + "=" * 75
    )

    print(
        f"EXAMPLE {number}"
    )

    print("=" * 75)

    print("\nTEXT:")
    print(example["text"])

    print("\nASPECTS:")

    for aspect in example[
        "aspects"
    ]:

        print(
            f"  "
            f"{aspect['aspect']:<30}"
            f" → "
            f"{aspect['polarity']:<10}"
            f" "
            f"[{aspect['from']}:"
            f"{aspect['to']}] "
            f"({aspect['span_source']})"
        )

    print("\nTOKEN ALIGNMENT")
    print("-" * 75)

    print(
        f"{'TOKEN':<30}"
        f"{'OFFSET':<18}"
        f"LABEL"
    )

    print("-" * 75)

    for (
        token,
        offset,
        label_id,
    ) in zip(
        example["tokens"],
        example["offset_mapping"],
        example["labels"],
    ):

        if label_id == IGNORE_INDEX:
            label = "X"
        else:
            label = ID2LABEL[
                label_id
            ]

        print(
            f"{token:<30}"
            f"{str(tuple(offset)):<18}"
            f"{label}"
        )


# ============================================================
# SELECT INSPECTION EXAMPLES
# ============================================================

def select_inspection_examples(
    examples,
):

    selected = []

    # Mixed example
    for example in examples:

        polarities = {
            a["polarity"]
            for a in example[
                "aspects"
            ]
        }

        if (
            "positive" in polarities
            and
            "negative" in polarities
        ):

            selected.append(example)
            break

    # Multi-token aspect
    for example in examples:

        if any(
            " " in aspect["aspect"]
            for aspect in example[
                "aspects"
            ]
        ):

            selected.append(example)
            break

    # Recovered SemEval span
    for example in examples:

        if any(
            aspect["span_source"]
            == "recovered"

            for aspect in example[
                "aspects"
            ]
        ):

            selected.append(example)
            break

    unique = []
    seen = set()

    for example in selected:

        if example["text"] in seen:
            continue

        seen.add(
            example["text"]
        )

        unique.append(example)

    return unique


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "BASHA — FINAL GAUNEG "
        "TRAINING DATA PREPARATION"
    )

    print("=" * 75)

    train = load_json(
        TRAIN_FILE
    )

    validation = load_json(
        VALIDATION_FILE
    )

    print(
        "\nOriginal datasets are READ ONLY."
    )

    print(
        f"Training examples: "
        f"{len(train)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation)}"
    )

    (
        tokenized_train,
        train_stats,
        train_problems,
    ) = process_dataset(
        train,
        "TRAIN",
    )

    (
        tokenized_validation,
        validation_stats,
        validation_problems,
    ) = process_dataset(
        validation,
        "VALIDATION",
    )

    train_labels = count_labels(
        tokenized_train
    )

    validation_labels = (
        count_labels(
            tokenized_validation
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(
        OUTPUT_TRAIN,
        tokenized_train,
    )

    save_json(
        OUTPUT_VALIDATION,
        tokenized_validation,
    )

    report = {
        "model":
            MODEL_NAME,

        "max_length":
            MAX_LENGTH,

        "label2id":
            LABEL2ID,

        "train_stats":
            train_stats,

        "validation_stats":
            validation_stats,

        "train_label_counts":
            train_labels,

        "validation_label_counts":
            validation_labels,
    }

    save_json(
        REPORT_FILE,
        report,
    )

    save_json(
        PROBLEM_FILE,
        {
            "train":
                train_problems,

            "validation":
                validation_problems,
        },
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print("TRAIN REPORT")

    print("=" * 75)

    for key, value in (
        train_stats.items()
    ):

        print(
            f"{key:<40}"
            f"{value}"
        )

    print(
        "\nTRAIN LABEL COUNTS"
    )

    for key, value in (
        train_labels.items()
    ):

        print(
            f"{key:<15}"
            f"{value}"
        )

    print(
        "\n" + "=" * 75
    )

    print(
        "VALIDATION REPORT"
    )

    print("=" * 75)

    for key, value in (
        validation_stats.items()
    ):

        print(
            f"{key:<40}"
            f"{value}"
        )

    print(
        "\nVALIDATION LABEL COUNTS"
    )

    for key, value in (
        validation_labels.items()
    ):

        print(
            f"{key:<15}"
            f"{value}"
        )

    # --------------------------------------------------------
    # CRITICAL CHECK
    # --------------------------------------------------------

    train_collisions = (
        train_stats.get(
            "token_collision",
            0,
        )
    )

    validation_collisions = (
        validation_stats.get(
            "token_collision",
            0,
        )
    )

    print(
        "\n" + "=" * 75
    )

    print("CRITICAL CHECK")

    print("=" * 75)

    print(
        "Train token collisions:",
        train_collisions,
    )

    print(
        "Validation token collisions:",
        validation_collisions,
    )

    if (
        train_collisions == 0
        and
        validation_collisions == 0
    ):

        print(
            "\nPASS: No token-label "
            "collisions remain."
        )

    else:

        print(
            "\nFAIL: Token collisions "
            "still exist."
        )

    # --------------------------------------------------------
    # MANUAL EXAMPLES
    # --------------------------------------------------------

    examples = (
        select_inspection_examples(
            tokenized_train
        )
    )

    print(
        "\nMANUAL INSPECTION"
    )

    for number, example in enumerate(
        examples,
        start=1,
    ):

        print_example(
            example,
            number,
        )

    print(
        "\n" + "=" * 75
    )

    print("FILES SAVED")

    print("=" * 75)

    print(OUTPUT_TRAIN)
    print(OUTPUT_VALIDATION)
    print(REPORT_FILE)
    print(PROBLEM_FILE)

    print(
        "\nDo NOT start training yet."
    )

    print(
        "Send the complete console "
        "output for final verification."
    )


if __name__ == "__main__":
    main()