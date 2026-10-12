from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BENCHMARK_FILE = (
    BASE_DIR
    / "data"
    / "english_fusion_benchmark.json"
)

ERROR_FILE = (
    BASE_DIR
    / "results"
    / "mixed_error_analysis.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "absa_gold_failure_analysis.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Conservative normalization used only for matching
    aspect strings.

    We intentionally do NOT do stemming, lemmatization,
    fuzzy matching, or semantic matching here.
    """

    text = str(text).lower().strip()

    # Normalize apostrophes.
    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Remove punctuation at the boundaries only.
    text = text.strip(
        " \t\n\r.,!?;:'\"()[]{}"
    )

    return text


def aspect_matches(
    gold_aspect: str,
    predicted_aspect: str,
) -> bool:
    """
    Strict normalized string matching.

    This first diagnostic deliberately avoids fuzzy matching
    because we want trustworthy error counts.
    """

    gold = normalize_text(
        gold_aspect
    )

    predicted = normalize_text(
        predicted_aspect
    )

    if not gold or not predicted:
        return False

    return gold == predicted


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_benchmark():

    data = load_json(
        BENCHMARK_FILE
    )

    if not isinstance(data, list):
        raise ValueError(
            "english_fusion_benchmark.json "
            "must contain a list."
        )

    return data


def load_errors():

    data = load_json(
        ERROR_FILE
    )

    errors = data.get(
        "errors"
    )

    if not isinstance(errors, list):
        raise ValueError(
            "mixed_error_analysis.json "
            "does not contain an 'errors' list."
        )

    return errors


# ============================================================
# BUILD GOLD LOOKUP
# ============================================================

def build_gold_lookup(
    benchmark,
):

    lookup = {}

    for item in benchmark:

        if (
            item.get("label")
            != "mixed"
        ):
            continue

        text = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        gold_evidence = (
            item.get(
                "gold_evidence"
            )
            or {}
        )

        positive_aspects = (
            gold_evidence.get(
                "positive_aspects"
            )
            or []
        )

        negative_aspects = (
            gold_evidence.get(
                "negative_aspects"
            )
            or []
        )

        lookup[text] = {
            "positive_aspects":
                positive_aspects,

            "negative_aspects":
                negative_aspects,
        }

    return lookup


# ============================================================
# GET GAUNEG PREDICTIONS
# ============================================================

def get_predicted_aspects(
    error_item,
):

    mixed_analysis = (
        error_item.get(
            "mixed_analysis"
        )
        or {}
    )

    evidence = (
        mixed_analysis.get(
            "evidence"
        )
        or {}
    )

    positive = (
        evidence.get(
            "positive_aspects"
        )
        or []
    )

    negative = (
        evidence.get(
            "negative_aspects"
        )
        or []
    )

    predictions = []

    for item in positive:

        predictions.append(
            {
                "aspect":
                    item.get(
                        "aspect",
                        "",
                    ),

                "sentiment":
                    "positive",

                "confidence":
                    item.get(
                        "confidence",
                        0.0,
                    ),
            }
        )

    for item in negative:

        predictions.append(
            {
                "aspect":
                    item.get(
                        "aspect",
                        "",
                    ),

                "sentiment":
                    "negative",

                "confidence":
                    item.get(
                        "confidence",
                        0.0,
                    ),
            }
        )

    return predictions


# ============================================================
# GOLD ASPECT ANALYSIS
# ============================================================

def analyze_gold_aspect(
    gold_aspect: str,
    gold_sentiment: str,
    predictions,
):

    matching_predictions = []

    for prediction in predictions:

        if aspect_matches(
            gold_aspect,
            prediction.get(
                "aspect",
                "",
            ),
        ):
            matching_predictions.append(
                prediction
            )

    # --------------------------------------------------------
    # Nothing matching this gold aspect was extracted.
    # --------------------------------------------------------

    if not matching_predictions:

        return {
            "gold_aspect":
                gold_aspect,

            "gold_sentiment":
                gold_sentiment,

            "status":
                "extraction_failure",

            "matched_prediction":
                None,
        }

    # --------------------------------------------------------
    # Correct extraction + correct polarity
    # --------------------------------------------------------

    for prediction in (
        matching_predictions
    ):

        if (
            prediction.get(
                "sentiment"
            )
            == gold_sentiment
        ):

            return {
                "gold_aspect":
                    gold_aspect,

                "gold_sentiment":
                    gold_sentiment,

                "status":
                    "correct",

                "matched_prediction":
                    prediction,
            }

    # --------------------------------------------------------
    # Aspect was found, but polarity disagrees with gold.
    # --------------------------------------------------------

    best_prediction = max(
        matching_predictions,
        key=lambda x: float(
            x.get(
                "confidence",
                0.0,
            )
            or 0.0
        ),
    )

    return {
        "gold_aspect":
            gold_aspect,

        "gold_sentiment":
            gold_sentiment,

        "status":
            "polarity_failure",

        "matched_prediction":
            best_prediction,
    }


# ============================================================
# SENTENCE FAILURE TYPE
# ============================================================

def determine_sentence_failure(
    aspect_results,
):

    statuses = {
        item["status"]
        for item in aspect_results
    }

    has_extraction_failure = (
        "extraction_failure"
        in statuses
    )

    has_polarity_failure = (
        "polarity_failure"
        in statuses
    )

    if (
        has_extraction_failure
        and has_polarity_failure
    ):
        return (
            "both_extraction_and_polarity"
        )

    if has_extraction_failure:
        return (
            "extraction_failure"
        )

    if has_polarity_failure:
        return (
            "polarity_failure"
        )

    return (
        "no_gold_aspect_error_found"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "BASHA GOLD-vs-GAUNEG ABSA FAILURE ANALYSIS"
    )
    print("=" * 80)

    benchmark = load_benchmark()

    errors = load_errors()

    gold_lookup = (
        build_gold_lookup(
            benchmark
        )
    )

    print(
        f"\nFrozen benchmark examples: "
        f"{len(benchmark)}"
    )

    print(
        f"Frozen gold Mixed examples: "
        f"{len(gold_lookup)}"
    )

    print(
        f"Missed Mixed examples: "
        f"{len(errors)}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    sentence_failure_counts = Counter()

    gold_aspect_status_counts = Counter()

    gold_polarity_counts = Counter()

    extraction_failure_by_polarity = (
        Counter()
    )

    polarity_failure_by_gold = (
        Counter()
    )

    missing_from_benchmark = []

    detailed_results = []

    total_gold_aspects = 0

    # ========================================================
    # ANALYZE MISSED MIXED
    # ========================================================

    for index, error in enumerate(
        errors,
        start=1,
    ):

        text = str(
            error.get(
                "text",
                "",
            )
        ).strip()

        gold = gold_lookup.get(
            text
        )

        if gold is None:

            missing_from_benchmark.append(
                text
            )

            continue

        predictions = (
            get_predicted_aspects(
                error
            )
        )

        gold_positive = (
            gold.get(
                "positive_aspects"
            )
            or []
        )

        gold_negative = (
            gold.get(
                "negative_aspects"
            )
            or []
        )

        aspect_results = []

        # ----------------------------------------------------
        # GOLD POSITIVE
        # ----------------------------------------------------

        for aspect in gold_positive:

            total_gold_aspects += 1

            gold_polarity_counts[
                "positive"
            ] += 1

            result = (
                analyze_gold_aspect(
                    gold_aspect=aspect,
                    gold_sentiment="positive",
                    predictions=predictions,
                )
            )

            aspect_results.append(
                result
            )

            gold_aspect_status_counts[
                result["status"]
            ] += 1

            if (
                result["status"]
                == "extraction_failure"
            ):
                extraction_failure_by_polarity[
                    "positive"
                ] += 1

            elif (
                result["status"]
                == "polarity_failure"
            ):
                polarity_failure_by_gold[
                    "positive"
                ] += 1

        # ----------------------------------------------------
        # GOLD NEGATIVE
        # ----------------------------------------------------

        for aspect in gold_negative:

            total_gold_aspects += 1

            gold_polarity_counts[
                "negative"
            ] += 1

            result = (
                analyze_gold_aspect(
                    gold_aspect=aspect,
                    gold_sentiment="negative",
                    predictions=predictions,
                )
            )

            aspect_results.append(
                result
            )

            gold_aspect_status_counts[
                result["status"]
            ] += 1

            if (
                result["status"]
                == "extraction_failure"
            ):
                extraction_failure_by_polarity[
                    "negative"
                ] += 1

            elif (
                result["status"]
                == "polarity_failure"
            ):
                polarity_failure_by_gold[
                    "negative"
                ] += 1

        sentence_failure = (
            determine_sentence_failure(
                aspect_results
            )
        )

        sentence_failure_counts[
            sentence_failure
        ] += 1

        detailed_results.append(
            {
                "index":
                    index,

                "text":
                    text,

                "basha_prediction":
                    error.get(
                        "predicted_label"
                    ),

                "basha_failure_category":
                    error.get(
                        "failure_category"
                    ),

                "gold": {
                    "positive_aspects":
                        gold_positive,

                    "negative_aspects":
                        gold_negative,
                },

                "gauneg_predictions":
                    predictions,

                "gold_aspect_results":
                    aspect_results,

                "sentence_failure_type":
                    sentence_failure,
            }
        )

    # ========================================================
    # METRICS
    # ========================================================

    matched_sentences = (
        len(detailed_results)
    )

    correct_aspects = (
        gold_aspect_status_counts[
            "correct"
        ]
    )

    extraction_failures = (
        gold_aspect_status_counts[
            "extraction_failure"
        ]
    )

    polarity_failures = (
        gold_aspect_status_counts[
            "polarity_failure"
        ]
    )

    correct_rate = (
        correct_aspects
        / total_gold_aspects
        if total_gold_aspects
        else 0.0
    )

    extraction_failure_rate = (
        extraction_failures
        / total_gold_aspects
        if total_gold_aspects
        else 0.0
    )

    polarity_failure_rate = (
        polarity_failures
        / total_gold_aspects
        if total_gold_aspects
        else 0.0
    )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print("\n" + "=" * 80)
    print("MATCHING")
    print("=" * 80)

    print(
        f"\nMatched missed-Mixed sentences: "
        f"{matched_sentences}"
    )

    print(
        f"Not found in benchmark:         "
        f"{len(missing_from_benchmark)}"
    )

    print("\n" + "=" * 80)
    print("GOLD ASPECT-LEVEL RESULTS")
    print("=" * 80)

    print(
        f"\nTotal gold aspects:       "
        f"{total_gold_aspects}"
    )

    print(
        f"Correct:                  "
        f"{correct_aspects} "
        f"({correct_rate * 100:.2f}%)"
    )

    print(
        f"Extraction failures:      "
        f"{extraction_failures} "
        f"({extraction_failure_rate * 100:.2f}%)"
    )

    print(
        f"Polarity failures:        "
        f"{polarity_failures} "
        f"({polarity_failure_rate * 100:.2f}%)"
    )

    print("\nEXTRACTION FAILURES BY GOLD POLARITY")

    print(
        f"positive: "
        f"{extraction_failure_by_polarity['positive']}"
    )

    print(
        f"negative: "
        f"{extraction_failure_by_polarity['negative']}"
    )

    print("\nPOLARITY FAILURES BY GOLD POLARITY")

    print(
        f"positive: "
        f"{polarity_failure_by_gold['positive']}"
    )

    print(
        f"negative: "
        f"{polarity_failure_by_gold['negative']}"
    )

    print("\n" + "=" * 80)
    print("SENTENCE-LEVEL FAILURE TYPES")
    print("=" * 80)

    for failure_type, count in (
        sentence_failure_counts
        .most_common()
    ):

        percentage = (
            count
            / matched_sentences
            * 100
            if matched_sentences
            else 0.0
        )

        print(
            f"{failure_type:<40}"
            f"{count:>3} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "summary": {
            "matched_sentences":
                matched_sentences,

            "missing_from_benchmark":
                len(
                    missing_from_benchmark
                ),

            "total_gold_aspects":
                total_gold_aspects,

            "correct_aspects":
                correct_aspects,

            "correct_rate":
                correct_rate,

            "extraction_failures":
                extraction_failures,

            "extraction_failure_rate":
                extraction_failure_rate,

            "polarity_failures":
                polarity_failures,

            "polarity_failure_rate":
                polarity_failure_rate,

            "extraction_failure_by_polarity":
                dict(
                    extraction_failure_by_polarity
                ),

            "polarity_failure_by_gold":
                dict(
                    polarity_failure_by_gold
                ),

            "sentence_failure_types":
                dict(
                    sentence_failure_counts
                ),
        },

        "missing_sentences":
            missing_from_benchmark,

        "errors":
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

    print("\nDetailed analysis saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print(
        "GOLD-vs-GAUNEG ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()