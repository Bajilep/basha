from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.basha.aspects.aspect_model import (
    EnglishAspectModel,
)

from src.basha.aspects.aspect_model_deberta import (
    DebertaAspectModel,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = (
    "tomaarsen/"
    "setfit-absa-semeval-restaurants"
)

SEED = 42
TEST_RATIO = 0.20

VALID_LABELS = {
    "negative",
    "neutral",
    "positive",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_aspect(
    text: str,
) -> str:
    """
    Normalize aspect strings for matching.

    We intentionally keep this conservative.
    """

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip(
        " \t\n\r.,!?;:'\"`()[]{}"
    )

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("LOADING SEMEVAL ABSA")
print("=" * 80)


dataset = load_dataset(
    DATASET_NAME
)

data = dataset["train"]


data = data.filter(
    lambda sample:
    sample["label"]
    in VALID_LABELS
)


# ============================================================
# RECREATE FROZEN SENTENCE SPLIT
# ============================================================

sentence_groups = defaultdict(
    list
)


for index, sample in enumerate(data):

    text = sample["text"].strip()

    sentence_groups[text].append(
        index
    )


unique_sentences = list(
    sentence_groups.keys()
)


rng = random.Random(
    SEED
)

rng.shuffle(
    unique_sentences
)


benchmark_sentence_count = round(
    len(unique_sentences)
    * TEST_RATIO
)


benchmark_sentences = set(
    unique_sentences[
        :benchmark_sentence_count
    ]
)


benchmark_indices = []


for text, indices in sentence_groups.items():

    if text in benchmark_sentences:

        benchmark_indices.extend(
            indices
        )


benchmark_data = data.select(
    benchmark_indices
)


print(
    f"Benchmark sentences: "
    f"{len(benchmark_sentences)}"
)

print(
    f"Benchmark aspects: "
    f"{len(benchmark_data)}"
)


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

ground_truth = defaultdict(
    list
)


for sample in benchmark_data:

    text = sample["text"].strip()

    ground_truth[text].append(
        {
            "aspect": sample["span"],
            "aspect_normalized":
                normalize_aspect(
                    sample["span"]
                ),
            "sentiment":
                sample["label"],
        }
    )


# ============================================================
# MATCH PREDICTED ASPECT
# ============================================================

def find_prediction(
    true_aspect: str,
    predictions: List[Dict],
) -> Optional[Dict]:

    true_normalized = (
        normalize_aspect(
            true_aspect
        )
    )

    # --------------------------------------------------------
    # First: exact normalized match
    # --------------------------------------------------------

    for prediction in predictions:

        predicted_normalized = (
            normalize_aspect(
                prediction["aspect"]
            )
        )

        if (
            predicted_normalized
            == true_normalized
        ):

            return prediction

    # --------------------------------------------------------
    # Second: conservative containment match
    #
    # Example:
    # true      = "battery life"
    # predicted = "battery life"
    #
    # Also tolerates minor span differences.
    # --------------------------------------------------------

    candidates = []

    for prediction in predictions:

        predicted_normalized = (
            normalize_aspect(
                prediction["aspect"]
            )
        )

        if not predicted_normalized:
            continue

        if (
            true_normalized
            in predicted_normalized
            or predicted_normalized
            in true_normalized
        ):

            # Avoid matching tiny garbage fragments
            if (
                len(predicted_normalized)
                >= 3
            ):

                candidates.append(
                    prediction
                )

    if not candidates:
        return None

    # Prefer the closest length match
    candidates.sort(
        key=lambda item:
        abs(
            len(
                normalize_aspect(
                    item["aspect"]
                )
            )
            - len(true_normalized)
        )
    )

    return candidates[0]


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model,
) -> Dict:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"EVALUATING: {model_name}"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Run inference ONCE per sentence
    # --------------------------------------------------------

    predictions_by_sentence = {}

    total_inference_time = 0.0


    sentences = list(
        ground_truth.keys()
    )


    for number, text in enumerate(
        sentences,
        start=1,
    ):

        start = time.perf_counter()

        predictions = model.predict(
            text
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        total_inference_time += (
            elapsed
        )

        predictions_by_sentence[
            text
        ] = predictions


        if (
            number % 50 == 0
            or number == len(sentences)
        ):

            print(
                f"Processed "
                f"{number}/"
                f"{len(sentences)} "
                f"sentences"
            )

    # --------------------------------------------------------
    # Evaluate aspect annotations
    # --------------------------------------------------------

    total_true_aspects = 0

    matched_aspects = 0

    correct_end_to_end = 0

    y_true_sentiment = []

    y_pred_sentiment = []

    missed_examples = []

    wrong_sentiment_examples = []


    for text, true_aspects in (
        ground_truth.items()
    ):

        predictions = (
            predictions_by_sentence[
                text
            ]
        )

        for true_item in true_aspects:

            total_true_aspects += 1

            prediction = (
                find_prediction(
                    true_item[
                        "aspect"
                    ],
                    predictions,
                )
            )

            if prediction is None:

                missed_examples.append(
                    {
                        "text": text,
                        "aspect":
                            true_item[
                                "aspect"
                            ],
                        "sentiment":
                            true_item[
                                "sentiment"
                            ],
                    }
                )

                continue

            matched_aspects += 1

            true_sentiment = (
                true_item[
                    "sentiment"
                ]
            )

            predicted_sentiment = (
                prediction[
                    "sentiment"
                ]
            )

            y_true_sentiment.append(
                true_sentiment
            )

            y_pred_sentiment.append(
                predicted_sentiment
            )

            if (
                true_sentiment
                == predicted_sentiment
            ):

                correct_end_to_end += 1

            else:

                wrong_sentiment_examples.append(
                    {
                        "text": text,

                        "aspect":
                            true_item[
                                "aspect"
                            ],

                        "true":
                            true_sentiment,

                        "predicted":
                            predicted_sentiment,

                        "confidence":
                            prediction.get(
                                "confidence"
                            ),
                    }
                )

    # --------------------------------------------------------
    # EXTRACTION RECALL
    # --------------------------------------------------------

    extraction_recall = (
        matched_aspects
        / total_true_aspects
        if total_true_aspects
        else 0.0
    )

    # --------------------------------------------------------
    # SENTIMENT METRICS
    #
    # Only for correctly located/matched aspects.
    # --------------------------------------------------------

    if y_true_sentiment:

        sentiment_accuracy = (
            accuracy_score(
                y_true_sentiment,
                y_pred_sentiment,
            )
        )

        sentiment_macro_f1 = (
            f1_score(
                y_true_sentiment,
                y_pred_sentiment,
                labels=[
                    "negative",
                    "neutral",
                    "positive",
                ],
                average="macro",
                zero_division=0,
            )
        )

    else:

        sentiment_accuracy = 0.0
        sentiment_macro_f1 = 0.0

    # --------------------------------------------------------
    # END-TO-END
    # --------------------------------------------------------

    end_to_end_accuracy = (
        correct_end_to_end
        / total_true_aspects
        if total_true_aspects
        else 0.0
    )

    average_ms = (
        total_inference_time
        / len(sentences)
        * 1000
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\nRESULTS"
    )

    print(
        "-" * 80
    )

    print(
        f"True aspects:       "
        f"{total_true_aspects}"
    )

    print(
        f"Matched aspects:    "
        f"{matched_aspects}"
    )

    print(
        f"Missed aspects:     "
        f"{total_true_aspects - matched_aspects}"
    )

    print(
        f"Extraction recall:  "
        f"{extraction_recall * 100:.2f}%"
    )

    print(
        f"\nSentiment accuracy: "
        f"{sentiment_accuracy * 100:.2f}%"
    )

    print(
        f"Sentiment Macro-F1: "
        f"{sentiment_macro_f1 * 100:.2f}%"
    )

    print(
        f"\nEnd-to-end accuracy:"
        f" {end_to_end_accuracy * 100:.2f}%"
    )

    print(
        f"\nInference time:     "
        f"{total_inference_time:.2f} sec"
    )

    print(
        f"Average speed:      "
        f"{average_ms:.2f} ms/sentence"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    if y_true_sentiment:

        print(
            "\nSENTIMENT CLASSIFICATION REPORT"
        )

        print(
            classification_report(
                y_true_sentiment,
                y_pred_sentiment,
                labels=[
                    "negative",
                    "neutral",
                    "positive",
                ],
                digits=4,
                zero_division=0,
            )
        )

    # --------------------------------------------------------
    # SAMPLE ERRORS
    # --------------------------------------------------------

    print(
        "\nFIRST 10 MISSED ASPECTS"
    )

    for item in (
        missed_examples[:10]
    ):

        print(
            "\nText:",
            item["text"]
        )

        print(
            "Missed aspect:",
            item["aspect"]
        )

        print(
            "True sentiment:",
            item["sentiment"]
        )


    print(
        "\nFIRST 10 WRONG SENTIMENTS"
    )

    for item in (
        wrong_sentiment_examples[:10]
    ):

        print(
            "\nText:",
            item["text"]
        )

        print(
            "Aspect:",
            item["aspect"]
        )

        print(
            "True:",
            item["true"]
        )

        print(
            "Predicted:",
            item["predicted"]
        )

        print(
            "Confidence:",
            item["confidence"]
        )


    return {
        "model": model_name,

        "extraction_recall":
            extraction_recall,

        "sentiment_accuracy":
            sentiment_accuracy,

        "sentiment_macro_f1":
            sentiment_macro_f1,

        "end_to_end_accuracy":
            end_to_end_accuracy,

        "seconds":
            total_inference_time,

        "ms_per_sentence":
            average_ms,

        "matched":
            matched_aspects,

        "missed":
            total_true_aspects
            - matched_aspects,
    }


# ============================================================
# CANDIDATE 1
# ============================================================

roberta_model = (
    EnglishAspectModel()
)


roberta_results = (
    evaluate_model(
        "Gauneg-RoBERTa",
        roberta_model,
    )
)


# Free reference before loading second model
del roberta_model


# ============================================================
# CANDIDATE 2
# ============================================================

deberta_model = (
    DebertaAspectModel()
)


deberta_results = (
    evaluate_model(
        "Yangheng-DeBERTa",
        deberta_model,
    )
)


del deberta_model


# ============================================================
# FINAL COMPARISON
# ============================================================

results = [
    roberta_results,
    deberta_results,
]


print(
    "\n"
    + "=" * 100
)

print(
    "FINAL ABSA COMPARISON"
)

print(
    "=" * 100
)


header = (
    f"{'MODEL':22s}"
    f"{'EXTRACT':>12s}"
    f"{'SENT ACC':>12s}"
    f"{'MACRO F1':>12s}"
    f"{'E2E ACC':>12s}"
    f"{'MS/SENT':>12s}"
)


print(
    header
)

print(
    "-" * 100
)


for result in results:

    print(
        f"{result['model']:22s}"
        f"{result['extraction_recall'] * 100:11.2f}%"
        f"{result['sentiment_accuracy'] * 100:11.2f}%"
        f"{result['sentiment_macro_f1'] * 100:11.2f}%"
        f"{result['end_to_end_accuracy'] * 100:11.2f}%"
        f"{result['ms_per_sentence']:12.2f}"
    )


print(
    "\n"
    + "=" * 100
)

print(
    "BENCHMARK COMPLETE"
)

print(
    "=" * 100
)