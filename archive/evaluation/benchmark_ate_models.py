from __future__ import annotations

import json
import re
import time
import traceback
from collections import Counter
from difflib import SequenceMatcher
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

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "ate_model_benchmark.json"
)


# ============================================================
# MODELS
# ============================================================

GAUNEG_ROBERTA = (
    "gauneg/roberta-base-absa-ate-sentiment"
)

GAUNEG_DEBERTA = (
    "gauneg/deberta-v3-base-absa-ate-sentiment"
)

SETFIT_ASPECT = (
    "tomaarsen/"
    "setfit-absa-bge-small-en-v1.5-restaurants-aspect"
)

SETFIT_POLARITY = (
    "tomaarsen/"
    "setfit-absa-bge-small-en-v1.5-restaurants-polarity"
)

SPACY_MODEL = "en_core_web_lg"


# ============================================================
# MATCHING SETTINGS
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


def token_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            normalize(text),
        )
    )


# ============================================================
# SPAN MATCHING
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
    gold_tokens = token_set(gold)
    predicted_tokens = token_set(predicted)

    if not gold_tokens or not predicted_tokens:
        return 0.0

    intersection = (
        gold_tokens
        & predicted_tokens
    )

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
    predicted_norm = normalize(predicted)

    if not gold_norm or not predicted_norm:
        return False

    # Avoid accepting tiny fragments such as:
    #
    # gold = "ambience"
    # prediction = "amb"
    #
    # as automatically correct merely because one is contained
    # inside the other.
    if min(
        len(gold_norm),
        len(predicted_norm),
    ) < 4:
        return False

    return (
        gold_norm in predicted_norm
        or predicted_norm in gold_norm
    )


def span_match_score(
    gold: str,
    predicted: str,
) -> tuple[bool, float, str]:
    gold_norm = normalize(gold)
    pred_norm = normalize(predicted)

    if gold_norm == pred_norm:
        return True, 1.0, "exact"

    sequence = sequence_similarity(
        gold,
        predicted,
    )

    overlap = token_overlap(
        gold,
        predicted,
    )

    contained = containment_match(
        gold,
        predicted,
    )

    score = max(
        sequence,
        overlap,
    )

    if (
        contained
        or sequence >= SEQUENCE_THRESHOLD
        or overlap >= TOKEN_OVERLAP_THRESHOLD
    ):
        return True, score, "near"

    return False, score, "none"


# ============================================================
# SENTIMENT NORMALIZATION
# ============================================================

def normalize_sentiment(
    label: Any,
) -> str | None:
    if label is None:
        return None

    value = str(label).strip().lower()

    mapping = {
        "pos": "positive",
        "positive": "positive",

        "neg": "negative",
        "negative": "negative",

        "neu": "neutral",
        "neutral": "neutral",
    }

    return mapping.get(value)


# ============================================================
# DATASET
# ============================================================

def load_mixed_examples() -> list[dict]:
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark not found:\n"
            f"{BENCHMARK_FILE}"
        )

    with BENCHMARK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "examples" in data:
            data = data["examples"]
        elif "data" in data:
            data = data["data"]

    if not isinstance(data, list):
        raise TypeError(
            "Frozen benchmark JSON must contain "
            "a list of examples."
        )

    mixed = []

    for item in data:
        label = str(
            item.get("label", "")
        ).lower()

        if label != "mixed":
            continue

        gold = (
            item.get("gold_evidence")
            or {}
        )

        positive = (
            gold.get("positive_aspects")
            or []
        )

        negative = (
            gold.get("negative_aspects")
            or []
        )

        mixed.append(
            {
                "text": item["text"],
                "positive_aspects": positive,
                "negative_aspects": negative,
            }
        )

    if not mixed:
        raise RuntimeError(
            "No Mixed examples were found "
            "in the frozen benchmark."
        )

    return mixed


# ============================================================
# GOLD ASPECTS
# ============================================================

def build_gold_aspects(
    example: dict,
) -> list[dict]:
    gold = []

    for aspect in example[
        "positive_aspects"
    ]:
        gold.append(
            {
                "aspect": aspect,
                "sentiment": "positive",
            }
        )

    for aspect in example[
        "negative_aspects"
    ]:
        gold.append(
            {
                "aspect": aspect,
                "sentiment": "negative",
            }
        )

    return gold


# ============================================================
# GAUNEG MODEL
# ============================================================

class GaunegModel:
    def __init__(
        self,
        model_name: str,
    ):
        from transformers import pipeline

        self.model_name = model_name

        self.pipe = pipeline(
            task="token-classification",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=-1,
        )

    def predict(
        self,
        text: str,
    ) -> list[dict]:
        raw = self.pipe(text)

        predictions = []

        for item in raw:
            aspect = str(
                item.get("word", "")
            ).strip()

            sentiment = (
                normalize_sentiment(
                    item.get(
                        "entity_group"
                    )
                )
            )

            if (
                not aspect
                or sentiment is None
            ):
                continue

            # Remove punctuation-only predictions.
            if not re.search(
                r"[A-Za-z0-9]",
                aspect,
            ):
                continue

            predictions.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "confidence": float(
                        item.get(
                            "score",
                            0.0,
                        )
                    ),
                }
            )

        return predictions


# ============================================================
# SETFIT ABSA
# ============================================================

class SetFitABSAModel:
    def __init__(self):
        from setfit import AbsaModel

        self.model = (
            AbsaModel.from_pretrained(
                SETFIT_ASPECT,
                SETFIT_POLARITY,
                spacy_model=SPACY_MODEL,
            )
        )

    def predict(
        self,
        text: str,
    ) -> list[dict]:
        raw = self.model.predict(text)

        # Current SetFit returns a list of dicts
        # for one input sentence.
        if raw is None:
            return []

        # Defensive handling in case a version returns
        # one nested list for a single sentence.
        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], list)
        ):
            raw = raw[0]

        predictions = []

        for item in raw:
            if not isinstance(
                item,
                dict,
            ):
                continue

            aspect = str(
                item.get(
                    "span",
                    item.get(
                        "aspect",
                        "",
                    ),
                )
            ).strip()

            sentiment = (
                normalize_sentiment(
                    item.get(
                        "polarity",
                        item.get(
                            "sentiment"
                        ),
                    )
                )
            )

            if (
                not aspect
                or sentiment is None
            ):
                continue

            if not re.search(
                r"[A-Za-z0-9]",
                aspect,
            ):
                continue

            predictions.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,

                    # SetFit ABSA's high-level predict()
                    # does not necessarily expose a probability.
                    "confidence": None,
                }
            )

        return predictions


# ============================================================
# MATCH GOLD ↔ PREDICTIONS
# ============================================================

def match_predictions(
    gold_aspects: list[dict],
    predictions: list[dict],
) -> dict:
    """
    One-to-one greedy matching.

    Important:
    A predicted aspect may only match one gold annotation.

    This prevents one prediction from artificially matching
    several different gold aspects.
    """

    candidate_pairs = []

    for gold_index, gold in enumerate(
        gold_aspects
    ):
        for pred_index, prediction in enumerate(
            predictions
        ):
            matched, score, match_type = (
                span_match_score(
                    gold["aspect"],
                    prediction["aspect"],
                )
            )

            if not matched:
                continue

            exact_bonus = (
                1
                if match_type == "exact"
                else 0
            )

            candidate_pairs.append(
                {
                    "gold_index": gold_index,
                    "pred_index": pred_index,
                    "score": score,
                    "match_type": match_type,
                    "exact_bonus": exact_bonus,
                }
            )

    candidate_pairs.sort(
        key=lambda x: (
            x["exact_bonus"],
            x["score"],
        ),
        reverse=True,
    )

    used_gold = set()
    used_predictions = set()

    matches = []

    for pair in candidate_pairs:
        gold_index = pair[
            "gold_index"
        ]

        pred_index = pair[
            "pred_index"
        ]

        if gold_index in used_gold:
            continue

        if pred_index in used_predictions:
            continue

        used_gold.add(
            gold_index
        )

        used_predictions.add(
            pred_index
        )

        gold = gold_aspects[
            gold_index
        ]

        prediction = predictions[
            pred_index
        ]

        polarity_correct = (
            gold["sentiment"]
            == prediction["sentiment"]
        )

        matches.append(
            {
                "gold": gold,
                "prediction": prediction,
                "match_type":
                    pair["match_type"],
                "span_score":
                    pair["score"],
                "polarity_correct":
                    polarity_correct,
            }
        )

    unmatched_gold = [
        gold
        for index, gold
        in enumerate(gold_aspects)
        if index not in used_gold
    ]

    unmatched_predictions = [
        prediction
        for index, prediction
        in enumerate(predictions)
        if index not in used_predictions
    ]

    return {
        "matches": matches,
        "unmatched_gold":
            unmatched_gold,
        "unmatched_predictions":
            unmatched_predictions,
    }


# ============================================================
# METRICS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(
    counts: Counter,
) -> dict:
    gold_total = counts[
        "gold_aspects"
    ]

    predicted_total = counts[
        "predicted_aspects"
    ]

    matched = counts[
        "matched_aspects"
    ]

    correct_polarity = counts[
        "correct_polarity"
    ]

    precision = safe_divide(
        matched,
        predicted_total,
    )

    recall = safe_divide(
        matched,
        gold_total,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    polarity_accuracy = safe_divide(
        correct_polarity,
        matched,
    )

    end_to_end_accuracy = safe_divide(
        correct_polarity,
        gold_total,
    )

    mixed_recoverable_rate = safe_divide(
        counts[
            "sentences_both_polarities"
        ],
        counts[
            "sentences"
        ],
    )

    return {
        "gold_aspects":
            gold_total,

        "predicted_aspects":
            predicted_total,

        "matched_aspects":
            matched,

        "correct_polarity":
            correct_polarity,

        "aspect_precision":
            precision,

        "aspect_recall":
            recall,

        "aspect_f1":
            f1,

        "polarity_accuracy_on_matched":
            polarity_accuracy,

        "end_to_end_aspect_polarity_accuracy":
            end_to_end_accuracy,

        "sentences":
            counts[
                "sentences"
            ],

        "sentences_with_both_polarities":
            counts[
                "sentences_both_polarities"
            ],

        "mixed_recovery_potential":
            mixed_recoverable_rate,

        "exact_matches":
            counts[
                "exact_matches"
            ],

        "near_matches":
            counts[
                "near_matches"
            ],

        "unmatched_gold":
            counts[
                "unmatched_gold"
            ],

        "unmatched_predictions":
            counts[
                "unmatched_predictions"
            ],
    }


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(
    model_name: str,
    model,
    examples: list[dict],
) -> dict:
    print()
    print("=" * 80)
    print(
        f"BENCHMARKING: {model_name}"
    )
    print("=" * 80)

    counts = Counter()
    details = []

    inference_times = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        text = example["text"]

        gold_aspects = (
            build_gold_aspects(
                example
            )
        )

        start = time.perf_counter()

        predictions = (
            model.predict(text)
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_times.append(
            elapsed
        )

        matching = (
            match_predictions(
                gold_aspects,
                predictions,
            )
        )

        matches = matching[
            "matches"
        ]

        unmatched_gold = matching[
            "unmatched_gold"
        ]

        unmatched_predictions = (
            matching[
                "unmatched_predictions"
            ]
        )

        counts[
            "sentences"
        ] += 1

        counts[
            "gold_aspects"
        ] += len(
            gold_aspects
        )

        counts[
            "predicted_aspects"
        ] += len(
            predictions
        )

        counts[
            "matched_aspects"
        ] += len(
            matches
        )

        counts[
            "unmatched_gold"
        ] += len(
            unmatched_gold
        )

        counts[
            "unmatched_predictions"
        ] += len(
            unmatched_predictions
        )

        for match in matches:
            if (
                match["match_type"]
                == "exact"
            ):
                counts[
                    "exact_matches"
                ] += 1
            else:
                counts[
                    "near_matches"
                ] += 1

            if match[
                "polarity_correct"
            ]:
                counts[
                    "correct_polarity"
                ] += 1

        # --------------------------------------------
        # Mixed recovery potential
        # --------------------------------------------

        found_positive = False
        found_negative = False

        for match in matches:
            if not match[
                "polarity_correct"
            ]:
                continue

            sentiment = (
                match["gold"][
                    "sentiment"
                ]
            )

            if sentiment == "positive":
                found_positive = True

            elif sentiment == "negative":
                found_negative = True

        if (
            found_positive
            and found_negative
        ):
            counts[
                "sentences_both_polarities"
            ] += 1

        details.append(
            {
                "index": index,
                "text": text,
                "gold_aspects":
                    gold_aspects,
                "predictions":
                    predictions,
                "matches":
                    matches,
                "unmatched_gold":
                    unmatched_gold,
                "unmatched_predictions":
                    unmatched_predictions,
                "inference_seconds":
                    elapsed,
            }
        )

        if (
            index % 10 == 0
            or index == len(examples)
        ):
            print(
                f"{index:3d}/"
                f"{len(examples)} "
                f"sentences complete"
            )

    metrics = calculate_metrics(
        counts
    )

    total_time = sum(
        inference_times
    )

    mean_time = safe_divide(
        total_time,
        len(inference_times),
    )

    metrics[
        "total_inference_seconds"
    ] = total_time

    metrics[
        "mean_inference_ms"
    ] = (
        mean_time * 1000
    )

    print()
    print(
        f"Aspect Precision: "
        f"{metrics['aspect_precision'] * 100:.2f}%"
    )

    print(
        f"Aspect Recall:    "
        f"{metrics['aspect_recall'] * 100:.2f}%"
    )

    print(
        f"Aspect F1:        "
        f"{metrics['aspect_f1'] * 100:.2f}%"
    )

    print(
        f"Polarity Acc:     "
        f"{metrics['polarity_accuracy_on_matched'] * 100:.2f}%"
    )

    print(
        f"End-to-End:       "
        f"{metrics['end_to_end_aspect_polarity_accuracy'] * 100:.2f}%"
    )

    print(
        f"Mixed potential:  "
        f"{metrics['mixed_recovery_potential'] * 100:.2f}%"
    )

    print(
        f"Mean CPU time:    "
        f"{metrics['mean_inference_ms']:.2f} ms"
    )

    return {
        "status": "success",
        "model": model_name,
        "metrics": metrics,
        "details": details,
    }


# ============================================================
# SAFE RUNNER
# ============================================================

def run_candidate(
    display_name: str,
    loader,
    examples: list[dict],
) -> dict:
    print()
    print("#" * 80)
    print(
        f"LOADING: {display_name}"
    )
    print("#" * 80)

    try:
        load_start = (
            time.perf_counter()
        )

        model = loader()

        load_seconds = (
            time.perf_counter()
            - load_start
        )

        print(
            f"Loaded in "
            f"{load_seconds:.2f} seconds"
        )

        result = benchmark_model(
            display_name,
            model,
            examples,
        )

        result[
            "load_seconds"
        ] = load_seconds

        del model

        # Free PyTorch CPU/GPU references where possible.
        try:
            import gc
            import torch

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass

        return result

    except Exception as exc:
        print()
        print(
            f"FAILED: {display_name}"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return {
            "status": "failed",
            "model": display_name,
            "error_type":
                type(exc).__name__,
            "error":
                str(exc),
            "traceback":
                traceback.format_exc(),
        }


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(
    results: dict,
) -> None:
    print()
    print("=" * 115)
    print(
        "ATE MODEL COMPARISON"
    )
    print("=" * 115)

    header = (
        f"{'Model':<28}"
        f"{'Prec':>9}"
        f"{'Recall':>9}"
        f"{'F1':>9}"
        f"{'PolAcc':>10}"
        f"{'E2E':>9}"
        f"{'Mixed':>9}"
        f"{'ms/text':>11}"
    )

    print(header)
    print("-" * 115)

    for name, result in (
        results.items()
    ):
        if (
            result.get("status")
            != "success"
        ):
            print(
                f"{name:<28}"
                f"{'FAILED':>9}"
            )
            continue

        metrics = result[
            "metrics"
        ]

        print(
            f"{name:<28}"
            f"{metrics['aspect_precision'] * 100:>8.2f}%"
            f"{metrics['aspect_recall'] * 100:>8.2f}%"
            f"{metrics['aspect_f1'] * 100:>8.2f}%"
            f"{metrics['polarity_accuracy_on_matched'] * 100:>9.2f}%"
            f"{metrics['end_to_end_aspect_polarity_accuracy'] * 100:>8.2f}%"
            f"{metrics['mixed_recovery_potential'] * 100:>8.2f}%"
            f"{metrics['mean_inference_ms']:>11.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print(
        "BASHA PRETRAINED ATE MODEL BENCHMARK"
    )
    print("=" * 80)

    print(
        "\nFrozen benchmark:"
    )
    print(BENCHMARK_FILE)

    examples = (
        load_mixed_examples()
    )

    print(
        f"\nMixed examples loaded: "
        f"{len(examples)}"
    )

    total_gold = sum(
        len(
            build_gold_aspects(
                example
            )
        )
        for example in examples
    )

    print(
        f"Total gold aspects: "
        f"{total_gold}"
    )

    results = {}

    # ========================================================
    # 1. CURRENT BASELINE
    # ========================================================

    results[
        "Gauneg RoBERTa"
    ] = run_candidate(
        "Gauneg RoBERTa",
        lambda: GaunegModel(
            GAUNEG_ROBERTA
        ),
        examples,
    )

    # ========================================================
    # 2. GAUNEG DEBERTA
    # ========================================================

    results[
        "Gauneg DeBERTa"
    ] = run_candidate(
        "Gauneg DeBERTa",
        lambda: GaunegModel(
            GAUNEG_DEBERTA
        ),
        examples,
    )

    # ========================================================
    # 3. SETFIT ABSA
    # ========================================================

    results[
        "SetFit ABSA"
    ] = run_candidate(
        "SetFit ABSA",
        lambda: SetFitABSAModel(),
        examples,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print_comparison(
        results
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "benchmark": {
            "dataset":
                str(BENCHMARK_FILE),

            "mixed_examples":
                len(examples),

            "gold_aspects":
                total_gold,

            "sequence_threshold":
                SEQUENCE_THRESHOLD,

            "token_overlap_threshold":
                TOKEN_OVERLAP_THRESHOLD,
        },

        "results":
            results,
    }

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

    print()
    print(
        "Detailed benchmark saved to:"
    )
    print(OUTPUT_FILE)

    print()
    print("=" * 80)
    print(
        "ATE MODEL BENCHMARK COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()