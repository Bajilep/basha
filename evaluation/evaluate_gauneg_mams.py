
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TEST_FILE = DATA_DIR / "absa_mams_test.json"

ORIGINAL_MODEL = "gauneg/roberta-base-absa-ate-sentiment"

FINETUNED_MODEL = (
    BASE_DIR / "models" / "gauneg_basha_v1" / "best_model"
)

OUTPUT_FILE = RESULTS_DIR / "gauneg_mams_comparison.json"

BATCH_SIZE = 8
MAX_LENGTH = 512
DEVICE = torch.device("cpu")

LABELS = {
    0: "O",
    1: "B-neu",
    2: "I-neu",
    3: "B-neg",
    4: "B-con",
    5: "I-pos",
    6: "B-pos",
    7: "I-con",
    8: "I-neg",
    9: "UNUSED",
}

POLARITY = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "con": "conflict",
    "conflict": "conflict",
}


# ============================================================
# DATASET
# ============================================================

def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    with TEST_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of test examples.")

    if not data:
        raise ValueError("MAMS test dataset is empty.")

    return data


def normalize_polarity(value):

    value = str(value).strip().lower()

    if value not in POLARITY:
        raise ValueError(f"Unknown polarity: {value}")

    return POLARITY[value]


def get_gold_aspects(example):

    text = example["text"]
    aspects = []

    for aspect in example["aspects"]:

        term = str(
            aspect.get("term", aspect.get("aspect", ""))
        ).strip()

        polarity = normalize_polarity(
            aspect.get("polarity", aspect.get("sentiment", ""))
        )

        start = aspect.get("from")
        end = aspect.get("to")

        if start is None:
            start = aspect.get("start")

        if end is None:
            end = aspect.get("end")

        if start is not None and end is not None:
            start, end = int(start), int(end)

            if (
                0 <= start < end <= len(text)
                and text[start:end].casefold() == term.casefold()
            ):
                aspects.append({
                    "start": start,
                    "end": end,
                    "term": text[start:end],
                    "polarity": polarity,
                })
                continue

        # Recover only whole-term matches.
        pattern = re.compile(
            r"(?<!\w)" + re.escape(term) + r"(?!\w)",
            re.IGNORECASE,
        )

        matches = list(pattern.finditer(text))

        if not matches:
            raise ValueError(
                f"Cannot locate gold aspect {term!r} in {text!r}"
            )

        match = matches[0]

        aspects.append({
            "start": match.start(),
            "end": match.end(),
            "term": text[match.start():match.end()],
            "polarity": polarity,
        })

    return aspects


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_all(model_path, examples):

    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path)
    ).to(DEVICE)

    if model.config.num_labels != 10:
        raise RuntimeError("Expected a 10-output Gauneg model.")

    model.eval()

    all_predictions = []
    output9_count = 0

    start_time = time.perf_counter()

    with torch.inference_mode():

        for batch_start in range(0, len(examples), BATCH_SIZE):

            batch = examples[
                batch_start:batch_start + BATCH_SIZE
            ]

            texts = [item["text"] for item in batch]

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offsets = encoded.pop("offset_mapping")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=attention_mask.to(DEVICE),
            )

            predictions = outputs.logits.argmax(
                dim=-1
            ).cpu()

            for row_index, text in enumerate(texts):

                spans = []
                current = None

                for token_index in range(input_ids.shape[1]):

                    if attention_mask[row_index, token_index] == 0:
                        continue

                    char_start, char_end = offsets[
                        row_index, token_index
                    ].tolist()

                    if char_start == char_end:
                        continue

                    label_id = int(
                        predictions[row_index, token_index]
                    )

                    if label_id == 9:
                        output9_count += 1

                    label = LABELS[label_id]

                    if label == "O" or label == "UNUSED":

                        if current is not None:
                            spans.append(current)
                            current = None

                        continue

                    prefix, sentiment = label.split("-", 1)
                    polarity = normalize_polarity(sentiment)

                    # Continue an aspect only when both polarity
                    # and character continuity are compatible.
                    can_continue = (
                        prefix == "I"
                        and current is not None
                        and current["polarity"] == polarity
                        and text[current["end"]:char_start].strip() == ""
                    )

                    if can_continue:
                        current["end"] = char_end
                    else:
                        if current is not None:
                            spans.append(current)

                        current = {
                            "start": char_start,
                            "end": char_end,
                            "polarity": polarity,
                        }

                if current is not None:
                    spans.append(current)

                for span in spans:
                    span["term"] = text[
                        span["start"]:span["end"]
                    ]

                all_predictions.append(spans)

            processed = min(
                batch_start + BATCH_SIZE,
                len(examples),
            )

            if processed % 100 < BATCH_SIZE or processed == len(examples):
                print(f"Processed {processed}/{len(examples)}")

    elapsed = time.perf_counter() - start_time

    return {
        "predictions": all_predictions,
        "seconds": elapsed,
        "output9_count": output9_count,
    }


# ============================================================
# METRICS
# ============================================================

def evaluate(examples, predictions):

    total_gold = 0
    total_pred = 0

    matched_spans = 0
    matched_with_polarity = 0

    correct_polarities = 0
    polarity_comparisons = 0

    mixed_total = 0
    mixed_detected = 0

    for example, predicted in zip(examples, predictions):

        gold = get_gold_aspects(example)

        gold_spans = {
            (a["start"], a["end"]) for a in gold
        }

        pred_spans = {
            (a["start"], a["end"]) for a in predicted
        }

        gold_full = {
            (a["start"], a["end"], a["polarity"])
            for a in gold
        }

        pred_full = {
            (a["start"], a["end"], a["polarity"])
            for a in predicted
        }

        total_gold += len(gold_spans)
        total_pred += len(pred_spans)

        matched_spans += len(
            gold_spans & pred_spans
        )

        matched_with_polarity += len(
            gold_full & pred_full
        )

        gold_by_span = {
            (a["start"], a["end"]): a["polarity"]
            for a in gold
        }

        pred_by_span = {
            (a["start"], a["end"]): a["polarity"]
            for a in predicted
        }

        for span in gold_spans & pred_spans:

            polarity_comparisons += 1

            if gold_by_span[span] == pred_by_span[span]:
                correct_polarities += 1

        gold_polarities = {
            a["polarity"] for a in gold
        }

        pred_polarities = {
            a["polarity"] for a in predicted
        }

        is_gold_mixed = (
            "positive" in gold_polarities
            and "negative" in gold_polarities
        )

        is_pred_mixed = (
            "positive" in pred_polarities
            and "negative" in pred_polarities
        )

        if is_gold_mixed:
            mixed_total += 1

            if is_pred_mixed:
                mixed_detected += 1

    def precision_recall_f1(correct, predicted_count, gold_count):

        precision = (
            correct / predicted_count
            if predicted_count else 0.0
        )

        recall = (
            correct / gold_count
            if gold_count else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    aspect_metrics = precision_recall_f1(
        matched_spans,
        total_pred,
        total_gold,
    )

    end_to_end_metrics = precision_recall_f1(
        matched_with_polarity,
        total_pred,
        total_gold,
    )

    return {
        "gold_aspects": total_gold,
        "predicted_aspects": total_pred,
        "matched_aspects": matched_spans,
        "aspect": aspect_metrics,
        "polarity_accuracy_on_matched_spans": (
            correct_polarities / polarity_comparisons
            if polarity_comparisons else 0.0
        ),
        "end_to_end": end_to_end_metrics,
        "gold_mixed_sentences": mixed_total,
        "mixed_detected": mixed_detected,
        "mixed_detection_rate": (
            mixed_detected / mixed_total
            if mixed_total else 0.0
        ),
    }


def print_metrics(name, result):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Aspect Precision: {result['aspect']['precision']:.2%}"
    )
    print(
        f"Aspect Recall:    {result['aspect']['recall']:.2%}"
    )
    print(
        f"Aspect F1:        {result['aspect']['f1']:.2%}"
    )

    print(
        "Polarity Accuracy (matched aspects): "
        f"{result['polarity_accuracy_on_matched_spans']:.2%}"
    )

    print(
        f"End-to-End F1:    {result['end_to_end']['f1']:.2%}"
    )

    print(
        f"Mixed Detection:  {result['mixed_detected']}/"
        f"{result['gold_mixed_sentences']} "
        f"({result['mixed_detection_rate']:.2%})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASHA — INDEPENDENT MAMS TEST")
    print("=" * 70)

    examples = load_test_data()

    print(f"\nTest sentences: {len(examples)}")

    # Validate all gold data before expensive inference.
    gold_count = sum(
        len(get_gold_aspects(example))
        for example in examples
    )

    print(f"Gold annotations: {gold_count}")

    if not FINETUNED_MODEL.exists():
        raise FileNotFoundError(
            f"Fine-tuned model missing: {FINETUNED_MODEL}"
        )

    original_run = predict_all(
        ORIGINAL_MODEL,
        examples,
    )

    finetuned_run = predict_all(
        FINETUNED_MODEL,
        examples,
    )

    original_metrics = evaluate(
        examples,
        original_run["predictions"],
    )

    finetuned_metrics = evaluate(
        examples,
        finetuned_run["predictions"],
    )

    print_metrics("ORIGINAL GAUNEG", original_metrics)
    print_metrics("FINE-TUNED GAUNEG", finetuned_metrics)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for metric_name, original_value, finetuned_value in [
        (
            "Aspect F1",
            original_metrics["aspect"]["f1"],
            finetuned_metrics["aspect"]["f1"],
        ),
        (
            "Polarity Accuracy",
            original_metrics[
                "polarity_accuracy_on_matched_spans"
            ],
            finetuned_metrics[
                "polarity_accuracy_on_matched_spans"
            ],
        ),
        (
            "End-to-End F1",
            original_metrics["end_to_end"]["f1"],
            finetuned_metrics["end_to_end"]["f1"],
        ),
        (
            "Mixed Detection",
            original_metrics["mixed_detection_rate"],
            finetuned_metrics["mixed_detection_rate"],
        ),
    ]:

        change = (finetuned_value - original_value) * 100

        print(
            f"{metric_name:<25} "
            f"{original_value:>8.2%} → "
            f"{finetuned_value:>8.2%} "
            f"({change:+.2f} pp)"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "test_file": str(TEST_FILE),
        "test_sentences": len(examples),
        "gold_annotations": gold_count,
        "original": {
            "model": ORIGINAL_MODEL,
            "metrics": original_metrics,
            "inference_seconds": original_run["seconds"],
            "output9_predictions": original_run["output9_count"],
        },
        "finetuned": {
            "model": str(FINETUNED_MODEL),
            "metrics": finetuned_metrics,
            "inference_seconds": finetuned_run["seconds"],
            "output9_predictions": finetuned_run["output9_count"],
        },
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nReport saved:")
    print(OUTPUT_FILE)

    print("\nDo not replace the production model yet.")


if __name__ == "__main__":
    main()
