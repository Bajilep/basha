from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from basha.sentiment.english_model import EnglishSentimentModel


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ERROR_FILE = (
    BASE_DIR
    / "results"
    / "mixed_error_analysis.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "clause_mixed_recovery.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

# Keep this conservative.
CLAUSE_CONFIDENCE_THRESHOLD = 0.65

CONTRAST_MARKERS = [
    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
    "while",
]


# ==========================================================
# HELPERS
# ==========================================================

def load_errors():

    if not ERROR_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{ERROR_FILE}"
        )

    with ERROR_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    errors = data.get("errors")

    if not isinstance(errors, list):
        raise ValueError(
            "mixed_error_analysis.json "
            "does not contain an 'errors' list."
        )

    return errors


def split_on_contrast(text: str):

    """
    Find the first supported contrast marker
    and split the sentence around it.

    Returns:
        {
            "marker": "...",
            "left": "...",
            "right": "..."
        }

    or None.
    """

    if not text:
        return None

    markers = "|".join(
        re.escape(marker)
        for marker in CONTRAST_MARKERS
    )

    pattern = re.compile(
        rf"\b({markers})\b",
        flags=re.IGNORECASE,
    )

    match = pattern.search(text)

    if not match:
        return None

    left = text[:match.start()].strip(
        " ,;:-"
    )

    right = text[match.end():].strip(
        " ,;:-"
    )

    if not left or not right:
        return None

    return {
        "marker": match.group(1).lower(),
        "left": left,
        "right": right,
    }


def normalize_sentiment(result):

    """
    Converts the Cardiff result into a small
    structure needed for this experiment.
    """

    if not isinstance(result, dict):
        raise ValueError(
            f"Unexpected sentiment result: {result!r}"
        )

    label = str(
        result.get("label", "")
    ).strip().lower()

    confidence = result.get(
        "confidence",
        0.0,
    )

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "label": label,
        "confidence": confidence,
    }


def is_confident_opposite(left, right):

    """
    Recovery requires:

        POSITIVE vs NEGATIVE

    or:

        NEGATIVE vs POSITIVE

    Both clauses must also pass the confidence
    threshold.

    Neutral does NOT count.
    """

    if (
        left["confidence"]
        < CLAUSE_CONFIDENCE_THRESHOLD
    ):
        return False

    if (
        right["confidence"]
        < CLAUSE_CONFIDENCE_THRESHOLD
    ):
        return False

    pair = {
        left["label"],
        right["label"],
    }

    return pair == {
        "positive",
        "negative",
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 80)
    print("BASHA CLAUSE-LEVEL MIXED RECOVERY EXPERIMENT")
    print("=" * 80)

    errors = load_errors()

    print(
        f"\nMissed Mixed examples loaded: "
        f"{len(errors)}"
    )

    print(
        f"Clause confidence threshold: "
        f"{CLAUSE_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        "\nLoading Cardiff sentiment model..."
    )

    sentiment_model = EnglishSentimentModel()

    print("Model loaded.")

    # ------------------------------------------------------
    # Counters
    # ------------------------------------------------------

    total_errors = len(errors)

    contrast_examples = 0
    no_contrast_examples = 0

    recoverable = 0
    not_recoverable = 0

    marker_counts = Counter()
    recovery_marker_counts = Counter()

    recovered_by_original_prediction = Counter()

    detailed_results = []

    # ------------------------------------------------------
    # Analyze each missed Mixed example
    # ------------------------------------------------------

    print("\nRunning clause analysis...\n")

    for index, item in enumerate(
        errors,
        start=1,
    ):

        text = str(
            item.get("text", "")
        ).strip()

        original_prediction = str(
            item.get(
                "predicted_label",
                "unknown",
            )
        ).lower()

        split = split_on_contrast(text)

        # ----------------------------------------------
        # No supported contrast marker
        # ----------------------------------------------

        if split is None:

            no_contrast_examples += 1

            detailed_results.append(
                {
                    "index": index,
                    "text": text,
                    "original_prediction":
                        original_prediction,
                    "has_contrast": False,
                    "recoverable": False,
                    "reason":
                        "no_supported_contrast_marker",
                }
            )

            continue

        # ----------------------------------------------
        # Contrast found
        # ----------------------------------------------

        contrast_examples += 1

        marker = split["marker"]

        marker_counts[marker] += 1

        left_text = split["left"]
        right_text = split["right"]

        left_raw = sentiment_model.predict(
            left_text
        )

        right_raw = sentiment_model.predict(
            right_text
        )

        left = normalize_sentiment(
            left_raw
        )

        right = normalize_sentiment(
            right_raw
        )

        recovered = is_confident_opposite(
            left,
            right,
        )

        if recovered:

            recoverable += 1

            recovery_marker_counts[
                marker
            ] += 1

            recovered_by_original_prediction[
                original_prediction
            ] += 1

        else:

            not_recoverable += 1

        detailed_results.append(
            {
                "index": index,
                "text": text,

                "original_prediction":
                    original_prediction,

                "failure_category":
                    item.get(
                        "failure_category"
                    ),

                "has_contrast": True,

                "marker": marker,

                "left": {
                    "text": left_text,
                    "sentiment":
                        left["label"],
                    "confidence":
                        left["confidence"],
                },

                "right": {
                    "text": right_text,
                    "sentiment":
                        right["label"],
                    "confidence":
                        right["confidence"],
                },

                "recoverable":
                    recovered,
            }
        )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    recovery_rate_all = (
        recoverable / total_errors
        if total_errors
        else 0.0
    )

    recovery_rate_contrast = (
        recoverable / contrast_examples
        if contrast_examples
        else 0.0
    )

    potential_new_correct_mixed = (
        41 + recoverable
    )

    potential_recall = (
        potential_new_correct_mixed / 100
    )

    # ------------------------------------------------------
    # Console report
    # ------------------------------------------------------

    print("=" * 80)
    print("CLAUSE RECOVERY RESULTS")
    print("=" * 80)

    print(
        f"\nTotal missed Mixed:       "
        f"{total_errors}"
    )

    print(
        f"Contrast examples:        "
        f"{contrast_examples}"
    )

    print(
        f"No contrast marker:       "
        f"{no_contrast_examples}"
    )

    print(
        f"\nRecoverable by clauses:   "
        f"{recoverable}"
    )

    print(
        f"Not recoverable:          "
        f"{not_recoverable}"
    )

    print(
        f"\nRecovery rate (all):      "
        f"{recovery_rate_all * 100:.2f}%"
    )

    print(
        f"Recovery rate (contrast): "
        f"{recovery_rate_contrast * 100:.2f}%"
    )

    print("\nCONTRAST MARKERS")

    if marker_counts:

        for marker, count in (
            marker_counts.most_common()
        ):

            recovered_count = (
                recovery_marker_counts[
                    marker
                ]
            )

            print(
                f"{marker:<12} "
                f"total={count:<3} "
                f"recovered={recovered_count}"
            )

    else:
        print("No contrast markers found.")

    print(
        "\nRECOVERED BY ORIGINAL PREDICTION"
    )

    if recovered_by_original_prediction:

        for label, count in (
            recovered_by_original_prediction
            .most_common()
        ):

            print(
                f"{label:<12} "
                f"{count}"
            )

    else:
        print("None")

    print("\n" + "-" * 80)

    print(
        "Current Mixed recall:     "
        "41.00%"
    )

    print(
        f"Theoretical recall if every "
        f"clause recovery were accepted: "
        f"{potential_recall * 100:.2f}%"
    )

    print(
        "\nIMPORTANT: this is only a "
        "diagnostic upper-bound."
    )

    print(
        "We must test false positives on "
        "non-Mixed examples before adding "
        "this rule to production."
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    output = {
        "settings": {
            "clause_confidence_threshold":
                CLAUSE_CONFIDENCE_THRESHOLD,

            "contrast_markers":
                CONTRAST_MARKERS,
        },

        "summary": {
            "total_missed_mixed":
                total_errors,

            "contrast_examples":
                contrast_examples,

            "no_contrast_examples":
                no_contrast_examples,

            "recoverable":
                recoverable,

            "not_recoverable":
                not_recoverable,

            "recovery_rate_all":
                recovery_rate_all,

            "recovery_rate_contrast":
                recovery_rate_contrast,

            "current_correct_mixed":
                41,

            "potential_correct_mixed":
                potential_new_correct_mixed,

            "current_mixed_recall":
                0.41,

            "potential_mixed_recall":
                potential_recall,
        },

        "marker_counts":
            dict(marker_counts),

        "recovery_marker_counts":
            dict(
                recovery_marker_counts
            ),

        "recovered_by_original_prediction":
            dict(
                recovered_by_original_prediction
            ),

        "results":
            detailed_results,
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

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print(
        "CLAUSE MIXED RECOVERY EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()