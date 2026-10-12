from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "english_fusion_results.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "mixed_error_analysis.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

ASPECT_THRESHOLD = 0.50
SENTENCE_THRESHOLD = 0.65


# ==========================================================
# HELPERS
# ==========================================================

def load_results():

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Fusion results not found:\n{RESULTS_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_label(value):

    if value is None:
        return None

    return str(value).strip().lower()


def extract_probability(item, label):

    """
    Tries common probability formats used by Basha.
    """

    if not isinstance(item, dict):
        return 0.0

    label = normalize_label(label)

    # Direct field:
    # {"positive": 0.9}
    value = item.get(label)

    if isinstance(value, (int, float)):
        return float(value)

    # Nested probabilities:
    # {"probabilities": {"positive": 0.9}}
    probabilities = item.get("probabilities")

    if isinstance(probabilities, dict):

        value = probabilities.get(label)

        if isinstance(value, (int, float)):
            return float(value)

    # Generic confidence if predicted label matches
    predicted_label = normalize_label(
        item.get("label")
    )

    confidence = item.get("confidence")

    if (
        predicted_label == label
        and isinstance(confidence, (int, float))
    ):
        return float(confidence)

    return 0.0


def get_aspect_label(aspect):

    if not isinstance(aspect, dict):
        return None

    for key in (
        "sentiment",
        "label",
        "polarity",
    ):

        value = aspect.get(key)

        if isinstance(value, str):
            return normalize_label(value)

        if isinstance(value, dict):
            nested_label = value.get("label")

            if isinstance(nested_label, str):
                return normalize_label(nested_label)

    return None


def get_aspect_confidence(aspect):

    if not isinstance(aspect, dict):
        return 0.0

    for key in (
        "confidence",
        "score",
        "sentiment_score",
    ):

        value = aspect.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    sentiment = aspect.get("sentiment")

    if isinstance(sentiment, dict):

        for key in (
            "confidence",
            "score",
        ):

            value = sentiment.get(key)

            if isinstance(value, (int, float)):
                return float(value)

    return 0.0


def analyze_aspects(aspects):

    positive = []
    negative = []

    if not isinstance(aspects, list):
        return {
            "positive": positive,
            "negative": negative,
            "has_both": False,
        }

    for aspect in aspects:

        label = get_aspect_label(aspect)
        confidence = get_aspect_confidence(aspect)

        if confidence < ASPECT_THRESHOLD:
            continue

        if label == "positive":
            positive.append(aspect)

        elif label == "negative":
            negative.append(aspect)

    return {
        "positive": positive,
        "negative": negative,
        "has_both": bool(
            positive and negative
        ),
    }


def analyze_sentences(sentences):

    positive = []
    negative = []

    if not isinstance(sentences, list):
        return {
            "positive": positive,
            "negative": negative,
            "has_both": False,
        }

    for sentence in sentences:

        positive_probability = (
            extract_probability(
                sentence,
                "positive",
            )
        )

        negative_probability = (
            extract_probability(
                sentence,
                "negative",
            )
        )

        if (
            positive_probability
            >= SENTENCE_THRESHOLD
        ):
            positive.append(sentence)

        if (
            negative_probability
            >= SENTENCE_THRESHOLD
        ):
            negative.append(sentence)

    return {
        "positive": positive,
        "negative": negative,
        "has_both": bool(
            positive and negative
        ),
    }


def get_result_list(data):

    """
    Supports either:

    {
        "results": [...]
    }

    or a raw [...]
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        results = data.get("results")

        if isinstance(results, list):
            return results

    raise ValueError(
        "Could not find result list in "
        "english_fusion_results.json"
    )


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def main():

    print("=" * 80)
    print("BASHA MIXED ERROR ANALYSIS")
    print("=" * 80)

    data = load_results()
    results = get_result_list(data)

    print(
        f"\nTotal benchmark records: "
        f"{len(results)}"
    )

    gold_mixed = []
    correct_mixed = []
    missed_mixed = []

    predicted_as = Counter()
    failure_categories = Counter()
    decision_sources = Counter()

    detailed_errors = []

    # ======================================================
    # FIND GOLD MIXED EXAMPLES
    # ======================================================

    for record in results:

        true_label = normalize_label(
            record.get(
                "true_label",
                record.get("gold_label"),
            )
        )

        predicted_label = normalize_label(
            record.get(
                "predicted_label",
                record.get("fusion_label"),
            )
        )

        if true_label != "mixed":
            continue

        gold_mixed.append(record)

        if predicted_label == "mixed":
            correct_mixed.append(record)
            continue

        missed_mixed.append(record)

        predicted_as[predicted_label] += 1

        decision_source = record.get(
            "decision_source",
            "unknown",
        )

        decision_sources[
            str(decision_source)
        ] += 1

        # ----------------------------------------------
        # Component evidence
        # ----------------------------------------------

        mixed_analysis = (
                record.get("mixed_analysis")
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

        aspect_evidence = {
            "positive": positive_aspects,
            "negative": negative_aspects,
            "has_both": bool(
                positive_aspects
                and negative_aspects
            ),
        }

        sentence_evidence = {
            "positive": positive_sentences,
            "negative": negative_sentences,
            "has_both": bool(
                positive_sentences
                and negative_sentences
            ),
        }

        # ----------------------------------------------
        # Failure classification
        # ----------------------------------------------

        if aspect_evidence["has_both"]:

            category = (
                "mixed_detector_or_fusion_failed_"
                "despite_opposite_aspects"
            )

        elif sentence_evidence["has_both"]:

            category = (
                "mixed_detector_or_fusion_failed_"
                "despite_opposite_sentences"
            )

        elif (
            aspect_evidence["positive"]
            and not aspect_evidence["negative"]
        ):

            category = (
                "missing_negative_aspect_evidence"
            )

        elif (
            aspect_evidence["negative"]
            and not aspect_evidence["positive"]
        ):

            category = (
                "missing_positive_aspect_evidence"
            )

        elif not (
            aspect_evidence["positive"]
            or aspect_evidence["negative"]
        ):

            category = (
                "no_confident_aspect_evidence"
            )

        else:

            category = "other"

        failure_categories[category] += 1

        detailed_errors.append(
            {
                "text":
                    record.get("text"),

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "decision_source":
                    decision_source,

                "failure_category":
                    category,

                "aspect_evidence": {
                    "positive_count":
                        len(
                            aspect_evidence[
                                "positive"
                            ]
                        ),

                    "negative_count":
                        len(
                            aspect_evidence[
                                "negative"
                            ]
                        ),

                    "has_both":
                        aspect_evidence[
                            "has_both"
                        ],
                },

                "sentence_evidence": {
                    "positive_count":
                        len(
                            sentence_evidence[
                                "positive"
                            ]
                        ),

                    "negative_count":
                        len(
                            sentence_evidence[
                                "negative"
                            ]
                        ),

                    "has_both":
                        sentence_evidence[
                            "has_both"
                        ],
                },

                "aspects": {
                    "positive":
                        positive_aspects,

                    "negative":
                        negative_aspects,
                },

                "sentences": {
                    "positive":
                        positive_sentences,

                    "negative":
                        negative_sentences,
                },

                "contrast":
                    contrast,

                "mixed_analysis":
                    mixed_analysis,

                "core_sentiment":
                    record.get(
                        "core_sentiment"
                    ),

                "sarcasm":
                    record.get(
                        "sarcasm"
                    ),

                "semantic_conflict":
                    record.get(
                        "semantic_conflict"
                    ),

                "discourse":
                    record.get(
                        "discourse"
                    ),
            }
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    total_gold = len(gold_mixed)
    correct_count = len(correct_mixed)
    missed_count = len(missed_mixed)

    recall = (
        correct_count / total_gold
        if total_gold
        else 0.0
    )

    print("\n" + "=" * 80)
    print("MIXED SUMMARY")
    print("=" * 80)

    print(
        f"\nGold Mixed examples: "
        f"{total_gold}"
    )

    print(
        f"Correctly detected:   "
        f"{correct_count}"
    )

    print(
        f"Missed Mixed:         "
        f"{missed_count}"
    )

    print(
        f"Mixed recall:         "
        f"{recall * 100:.2f}%"
    )

    # ======================================================
    # WHAT MIXED BECAME
    # ======================================================

    print("\nMISSED MIXED PREDICTED AS")

    for label, count in (
        predicted_as.most_common()
    ):

        print(
            f"{label:<15} "
            f"{count}"
        )

    # ======================================================
    # FAILURE CATEGORIES
    # ======================================================

    print("\nFAILURE CATEGORIES")

    for category, count in (
        failure_categories.most_common()
    ):

        percentage = (
            count / missed_count * 100
            if missed_count
            else 0.0
        )

        print(
            f"{category:<60} "
            f"{count:>3} "
            f"({percentage:>6.2f}%)"
        )

    # ======================================================
    # DECISION SOURCES
    # ======================================================

    print("\nFINAL DECISION SOURCES FOR MISSED MIXED")

    for source, count in (
        decision_sources.most_common()
    ):

        print(
            f"{source:<30} "
            f"{count}"
        )

    # ======================================================
    # SAVE
    # ======================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "summary": {
            "gold_mixed":
                total_gold,

            "correct_mixed":
                correct_count,

            "missed_mixed":
                missed_count,

            "mixed_recall":
                recall,
        },

        "predicted_as":
            dict(predicted_as),

        "failure_categories":
            dict(failure_categories),

        "decision_sources":
            dict(decision_sources),

        "errors":
            detailed_errors,
    }

    with OUTPUT_PATH.open(
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
    print(OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("MIXED ERROR ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()