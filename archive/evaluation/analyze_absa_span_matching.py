from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "absa_gold_failure_analysis.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "absa_span_matching_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_THRESHOLD = 0.80
TOKEN_OVERLAP_THRESHOLD = 0.50


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text: str) -> str:

    text = str(text).lower().strip()

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip(
        " \t\n\r.,!?;:'\"()[]{}"
    )

    return text


def tokens(text: str):

    return set(
        re.findall(
            r"[a-z0-9]+",
            normalize(text),
        )
    )


# ============================================================
# SIMILARITY
# ============================================================

def sequence_similarity(
    gold: str,
    predicted: str,
) -> float:

    return SequenceMatcher(
        None,
        normalize(gold),
        normalize(predicted),
    ).ratio()


def token_overlap(
    gold: str,
    predicted: str,
) -> float:

    gold_tokens = tokens(gold)
    predicted_tokens = tokens(predicted)

    if (
        not gold_tokens
        or not predicted_tokens
    ):
        return 0.0

    intersection = (
        gold_tokens
        & predicted_tokens
    )

    # Coverage relative to the smaller span.
    return (
        len(intersection)
        / min(
            len(gold_tokens),
            len(predicted_tokens),
        )
    )


def containment_match(
    gold: str,
    predicted: str,
) -> bool:

    gold_norm = normalize(gold)
    pred_norm = normalize(predicted)

    if (
        not gold_norm
        or not pred_norm
    ):
        return False

    # Avoid accepting extremely tiny fragments.
    if min(
        len(gold_norm),
        len(pred_norm),
    ) < 4:
        return False

    return (
        gold_norm in pred_norm
        or pred_norm in gold_norm
    )


# ============================================================
# CLASSIFY MATCH
# ============================================================

def classify_match(
    gold_aspect: str,
    predicted_aspect: str,
):

    gold_norm = normalize(
        gold_aspect
    )

    pred_norm = normalize(
        predicted_aspect
    )

    if (
        gold_norm
        == pred_norm
    ):
        return {
            "match_type": "exact",
            "sequence_similarity": 1.0,
            "token_overlap": 1.0,
        }

    sequence = (
        sequence_similarity(
            gold_aspect,
            predicted_aspect,
        )
    )

    overlap = (
        token_overlap(
            gold_aspect,
            predicted_aspect,
        )
    )

    contained = (
        containment_match(
            gold_aspect,
            predicted_aspect,
        )
    )

    if (
        contained
        or sequence >= SEQUENCE_THRESHOLD
        or overlap >= TOKEN_OVERLAP_THRESHOLD
    ):
        return {
            "match_type": "near_match",
            "sequence_similarity":
                sequence,
            "token_overlap":
                overlap,
        }

    return {
        "match_type": "no_match",
        "sequence_similarity":
            sequence,
        "token_overlap":
            overlap,
    }


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# FIND BEST PREDICTED SPAN
# ============================================================

def find_best_prediction(
    gold_aspect,
    predictions,
):

    candidates = []

    for prediction in predictions:

        predicted_aspect = (
            prediction.get(
                "aspect",
                "",
            )
        )

        comparison = (
            classify_match(
                gold_aspect,
                predicted_aspect,
            )
        )

        if (
            comparison["match_type"]
            == "no_match"
        ):
            continue

        # Ranking:
        #
        # exact > near
        #
        # then sequence/token similarity.

        rank = (
            2
            if comparison[
                "match_type"
            ] == "exact"
            else 1
        )

        score = max(
            comparison[
                "sequence_similarity"
            ],
            comparison[
                "token_overlap"
            ],
        )

        candidates.append(
            (
                rank,
                score,
                prediction,
                comparison,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, _, prediction, comparison = (
        candidates[0]
    )

    return {
        "prediction":
            prediction,

        "comparison":
            comparison,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "BASHA ABSA SPAN-MATCHING ANALYSIS"
    )
    print("=" * 80)

    data = load_data()

    errors = (
        data.get("errors")
        or []
    )

    counters = Counter()

    near_match_polarity = Counter()

    detailed = []

    original_extraction_failures = 0

    # ========================================================
    # ANALYZE
    # ========================================================

    for error in errors:

        text = error.get(
            "text",
            "",
        )

        predictions = (
            error.get(
                "gauneg_predictions"
            )
            or []
        )

        aspect_results = (
            error.get(
                "gold_aspect_results"
            )
            or []
        )

        for aspect_result in (
            aspect_results
        ):

            if (
                aspect_result.get(
                    "status"
                )
                != "extraction_failure"
            ):
                continue

            original_extraction_failures += 1

            gold_aspect = (
                aspect_result.get(
                    "gold_aspect",
                    "",
                )
            )

            gold_sentiment = (
                aspect_result.get(
                    "gold_sentiment",
                    "",
                )
            )

            best = (
                find_best_prediction(
                    gold_aspect,
                    predictions,
                )
            )

            # --------------------------------------------
            # Truly missing
            # --------------------------------------------

            if best is None:

                counters[
                    "true_extraction_failure"
                ] += 1

                detailed.append(
                    {
                        "text":
                            text,

                        "gold_aspect":
                            gold_aspect,

                        "gold_sentiment":
                            gold_sentiment,

                        "result":
                            "true_extraction_failure",

                        "matched_prediction":
                            None,
                    }
                )

                continue

            prediction = (
                best["prediction"]
            )

            comparison = (
                best["comparison"]
            )

            predicted_sentiment = (
                prediction.get(
                    "sentiment"
                )
            )

            # --------------------------------------------
            # Near span + correct sentiment
            # --------------------------------------------

            if (
                predicted_sentiment
                == gold_sentiment
            ):

                counters[
                    "near_match_correct_polarity"
                ] += 1

                near_match_polarity[
                    gold_sentiment
                ] += 1

                result_type = (
                    "near_match_correct_polarity"
                )

            # --------------------------------------------
            # Near span + wrong sentiment
            # --------------------------------------------

            else:

                counters[
                    "near_match_wrong_polarity"
                ] += 1

                result_type = (
                    "near_match_wrong_polarity"
                )

            detailed.append(
                {
                    "text":
                        text,

                    "gold_aspect":
                        gold_aspect,

                    "gold_sentiment":
                        gold_sentiment,

                    "result":
                        result_type,

                    "matched_prediction":
                        prediction,

                    "match_type":
                        comparison[
                            "match_type"
                        ],

                    "sequence_similarity":
                        comparison[
                            "sequence_similarity"
                        ],

                    "token_overlap":
                        comparison[
                            "token_overlap"
                        ],
                }
            )

    # ========================================================
    # RESULTS
    # ========================================================

    near_correct = (
        counters[
            "near_match_correct_polarity"
        ]
    )

    near_wrong = (
        counters[
            "near_match_wrong_polarity"
        ]
    )

    true_missing = (
        counters[
            "true_extraction_failure"
        ]
    )

    near_total = (
        near_correct
        + near_wrong
    )

    print("\nOriginal exact-match extraction failures:")
    print(
        original_extraction_failures
    )

    print("\n" + "=" * 80)
    print("SPAN-MATCHING RESULTS")
    print("=" * 80)

    print(
        f"\nNear/partial span matches:       "
        f"{near_total}"
    )

    print(
        f"  Correct polarity:             "
        f"{near_correct}"
    )

    print(
        f"  Wrong polarity:               "
        f"{near_wrong}"
    )

    print(
        f"\nTrue extraction failures:       "
        f"{true_missing}"
    )

    if original_extraction_failures:

        print(
            "\nNear-match share of original "
            "extraction failures: "
            f"{near_total / original_extraction_failures * 100:.2f}%"
        )

        print(
            "True-missing share: "
            f"{true_missing / original_extraction_failures * 100:.2f}%"
        )

    print("\nNEAR MATCHES BY GOLD POLARITY")

    print(
        "positive:",
        near_match_polarity[
            "positive"
        ],
    )

    print(
        "negative:",
        near_match_polarity[
            "negative"
        ],
    )

    # ========================================================
    # CORRECTED TOTALS
    # ========================================================

    original_summary = (
        data.get("summary")
        or {}
    )

    original_correct = int(
        original_summary.get(
            "correct_aspects",
            0,
        )
    )

    original_polarity_failures = int(
        original_summary.get(
            "polarity_failures",
            0,
        )
    )

    total_gold = int(
        original_summary.get(
            "total_gold_aspects",
            0,
        )
    )

    corrected_correct = (
        original_correct
        + near_correct
    )

    corrected_polarity = (
        original_polarity_failures
        + near_wrong
    )

    corrected_extraction = (
        true_missing
    )

    print("\n" + "=" * 80)
    print("CORRECTED GOLD-ASPECT VIEW")
    print("=" * 80)

    print(
        f"\nTotal gold aspects:       "
        f"{total_gold}"
    )

    print(
        f"Correct/near-correct:     "
        f"{corrected_correct}"
    )

    print(
        f"True extraction failures: "
        f"{corrected_extraction}"
    )

    print(
        f"Polarity failures:        "
        f"{corrected_polarity}"
    )

    if total_gold:

        print(
            f"\nCorrect rate:             "
            f"{corrected_correct / total_gold * 100:.2f}%"
        )

        print(
            f"True extraction failure:  "
            f"{corrected_extraction / total_gold * 100:.2f}%"
        )

        print(
            f"Polarity failure:         "
            f"{corrected_polarity / total_gold * 100:.2f}%"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "settings": {
            "sequence_threshold":
                SEQUENCE_THRESHOLD,

            "token_overlap_threshold":
                TOKEN_OVERLAP_THRESHOLD,
        },

        "summary": {
            "original_extraction_failures":
                original_extraction_failures,

            "near_matches":
                near_total,

            "near_match_correct_polarity":
                near_correct,

            "near_match_wrong_polarity":
                near_wrong,

            "true_extraction_failures":
                true_missing,

            "corrected_correct_aspects":
                corrected_correct,

            "corrected_polarity_failures":
                corrected_polarity,

            "total_gold_aspects":
                total_gold,
        },

        "details":
            detailed,
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

    print("\nDetailed results saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print(
        "ABSA SPAN-MATCHING ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()