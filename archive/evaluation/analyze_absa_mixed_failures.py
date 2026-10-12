from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "mixed_error_analysis.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "absa_mixed_failure_analysis.json"
)


# ============================================================
# SIMPLE LANGUAGE PATTERNS
# ============================================================

CONTRAST_MARKERS = [
    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
    "while",
    "despite",
]

COMPARATIVE_PATTERNS = [
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bmore than\b",
    r"\bless than\b",
    r"\btoo\b",
    r"\bworth\b",
    r"\bprice\b",
    r"\bcheap\b",
    r"\bexpensive\b",
]

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bn't\b",
    r"\bbarely\b",
    r"\bhardly\b",
]


# ============================================================
# LOAD
# ============================================================

def load_errors():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    errors = data.get("errors")

    if not isinstance(errors, list):
        raise ValueError(
            "Input JSON does not contain "
            "an 'errors' list."
        )

    return errors


# ============================================================
# HELPERS
# ============================================================

def contains_any_word(
    text: str,
    words: list[str],
) -> bool:

    text = text.lower()

    for word in words:

        if re.search(
            rf"\b{re.escape(word)}\b",
            text,
        ):
            return True

    return False


def contains_pattern(
    text: str,
    patterns: list[str],
) -> bool:

    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def classify_surface_pattern(text: str):

    """
    This does NOT claim the true cause of the
    ABSA failure.

    It only describes useful linguistic
    patterns present in the sentence.
    """

    patterns = []

    if contains_any_word(
        text,
        CONTRAST_MARKERS,
    ):
        patterns.append(
            "contrast_language"
        )

    if contains_pattern(
        text,
        NEGATION_PATTERNS,
    ):
        patterns.append(
            "negation_language"
        )

    if contains_pattern(
        text,
        COMPARATIVE_PATTERNS,
    ):
        patterns.append(
            "comparative_or_value_language"
        )

    if not patterns:
        patterns.append(
            "no_simple_surface_pattern"
        )

    return patterns


def get_evidence(item):

    mixed_analysis = (
        item.get("mixed_analysis")
        or {}
    )

    evidence = (
        mixed_analysis.get("evidence")
        or {}
    )

    positive_aspects = (
        evidence.get("positive_aspects")
        or []
    )

    negative_aspects = (
        evidence.get("negative_aspects")
        or []
    )

    positive_sentences = (
        evidence.get("positive_sentences")
        or []
    )

    negative_sentences = (
        evidence.get("negative_sentences")
        or []
    )

    contrast = (
        evidence.get("contrast")
        or {}
    )

    return {
        "positive_aspects":
            positive_aspects,

        "negative_aspects":
            negative_aspects,

        "positive_sentences":
            positive_sentences,

        "negative_sentences":
            negative_sentences,

        "contrast":
            contrast,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 80)
    print("BASHA ABSA MIXED FAILURE ANALYSIS")
    print("=" * 80)

    errors = load_errors()

    print(
        f"\nMixed misses loaded: "
        f"{len(errors)}"
    )

    failure_counts = Counter()
    prediction_counts = Counter()
    surface_pattern_counts = Counter()

    contrast_support_count = 0

    sentence_support_same_side = 0
    sentence_support_opposite_side = 0
    no_sentence_support = 0

    detailed = []

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    for index, item in enumerate(
        errors,
        start=1,
    ):

        text = str(
            item.get("text", "")
        ).strip()

        predicted_label = str(
            item.get(
                "predicted_label",
                "unknown",
            )
        ).lower()

        failure_category = str(
            item.get(
                "failure_category",
                "unknown",
            )
        )

        prediction_counts[
            predicted_label
        ] += 1

        failure_counts[
            failure_category
        ] += 1

        evidence = get_evidence(item)

        positive_aspects = (
            evidence[
                "positive_aspects"
            ]
        )

        negative_aspects = (
            evidence[
                "negative_aspects"
            ]
        )

        positive_sentences = (
            evidence[
                "positive_sentences"
            ]
        )

        negative_sentences = (
            evidence[
                "negative_sentences"
            ]
        )

        contrast = evidence[
            "contrast"
        ]

        # ----------------------------------------------------
        # Surface patterns
        # ----------------------------------------------------

        patterns = (
            classify_surface_pattern(
                text
            )
        )

        for pattern in patterns:
            surface_pattern_counts[
                pattern
            ] += 1

        # ----------------------------------------------------
        # Contrast support
        # ----------------------------------------------------

        contrast_supports_mixed = bool(
            contrast.get(
                "supports_mixed",
                False,
            )
        )

        if contrast_supports_mixed:
            contrast_support_count += 1

        # ----------------------------------------------------
        # Compare ABSA evidence with sentence evidence
        # ----------------------------------------------------

        sentence_relationship = (
            "none"
        )

        if (
            positive_aspects
            and positive_sentences
        ):
            sentence_relationship = (
                "same_positive"
            )

            sentence_support_same_side += 1

        elif (
            negative_aspects
            and negative_sentences
        ):
            sentence_relationship = (
                "same_negative"
            )

            sentence_support_same_side += 1

        elif (
            positive_aspects
            and negative_sentences
        ):
            sentence_relationship = (
                "opposite_negative"
            )

            sentence_support_opposite_side += 1

        elif (
            negative_aspects
            and positive_sentences
        ):
            sentence_relationship = (
                "opposite_positive"
            )

            sentence_support_opposite_side += 1

        else:
            no_sentence_support += 1

        # ----------------------------------------------------
        # Aspect summary
        # ----------------------------------------------------

        if (
            positive_aspects
            and not negative_aspects
        ):
            aspect_state = (
                "positive_only"
            )

        elif (
            negative_aspects
            and not positive_aspects
        ):
            aspect_state = (
                "negative_only"
            )

        elif (
            positive_aspects
            and negative_aspects
        ):
            aspect_state = (
                "both"
            )

        else:
            aspect_state = (
                "none"
            )

        detailed.append(
            {
                "index": index,

                "text": text,

                "true_label":
                    item.get(
                        "true_label"
                    ),

                "predicted_label":
                    predicted_label,

                "decision_source":
                    item.get(
                        "decision_source"
                    ),

                "failure_category":
                    failure_category,

                "aspect_state":
                    aspect_state,

                "positive_aspects":
                    positive_aspects,

                "negative_aspects":
                    negative_aspects,

                "positive_sentences":
                    positive_sentences,

                "negative_sentences":
                    negative_sentences,

                "sentence_relationship":
                    sentence_relationship,

                "contrast_supports_mixed":
                    contrast_supports_mixed,

                "contrast":
                    contrast,

                "surface_patterns":
                    patterns,
            }
        )

    # ========================================================
    # REPORT
    # ========================================================

    total = len(errors)

    print("\n" + "=" * 80)
    print("ABSA EVIDENCE FAILURES")
    print("=" * 80)

    for category, count in (
        failure_counts.most_common()
    ):

        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        print(
            f"{category:<55}"
            f"{count:>3} "
            f"({percentage:6.2f}%)"
        )

    print("\n" + "=" * 80)
    print("PREDICTED LABELS")
    print("=" * 80)

    for label, count in (
        prediction_counts.most_common()
    ):
        print(
            f"{label:<15}{count}"
        )

    print("\n" + "=" * 80)
    print("LINGUISTIC PATTERNS")
    print("=" * 80)

    for pattern, count in (
        surface_pattern_counts
        .most_common()
    ):

        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        print(
            f"{pattern:<40}"
            f"{count:>3} "
            f"({percentage:6.2f}%)"
        )

    print("\n" + "=" * 80)
    print("OTHER FUSION EVIDENCE")
    print("=" * 80)

    print(
        f"\nContrast analyzer supports Mixed: "
        f"{contrast_support_count}"
    )

    print(
        f"Sentence evidence supports "
        f"same detected polarity: "
        f"{sentence_support_same_side}"
    )

    print(
        f"Sentence evidence provides "
        f"opposite polarity: "
        f"{sentence_support_opposite_side}"
    )

    print(
        f"No useful sentence support: "
        f"{no_sentence_support}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "summary": {
            "total_mixed_misses":
                total,

            "failure_categories":
                dict(
                    failure_counts
                ),

            "predicted_labels":
                dict(
                    prediction_counts
                ),

            "surface_patterns":
                dict(
                    surface_pattern_counts
                ),

            "contrast_supports_mixed":
                contrast_support_count,

            "sentence_support_same_side":
                sentence_support_same_side,

            "sentence_support_opposite_side":
                sentence_support_opposite_side,

            "no_sentence_support":
                no_sentence_support,
        },

        "errors": detailed,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDetailed analysis saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print(
        "ABSA MIXED FAILURE ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()