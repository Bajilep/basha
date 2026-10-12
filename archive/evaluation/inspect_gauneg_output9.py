import json
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)


MODEL_NAME = "gauneg/roberta-base-absa-ate-sentiment"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "absa_final_train.json"
VALIDATION_FILE = DATA_DIR / "absa_final_validation.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
MAX_LENGTH = 512


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print("BASHA — GAUNEG OUTPUT 9 CHECK")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print("Loading model...")

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME
        )
        .to(DEVICE)
    )

    model.eval()

    train = load_json(TRAIN_FILE)
    validation = load_json(
        VALIDATION_FILE
    )

    # Use both datasets to get a fairly large,
    # realistic sample of sentences.
    examples = train + validation

    texts = [
        example["text"]
        for example in examples
    ]

    print(
        "\nSentences to inspect:",
        len(texts),
    )

    predicted_counts = Counter()

    output9_examples = []

    total_real_tokens = 0
    total_output9 = 0

    highest_output9_probability = 0.0
    highest_output9_info = None

    print("\nRunning original Gauneg model...")

    with torch.inference_mode():

        for batch_start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):

            batch_texts = texts[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            offsets = encoded.pop(
                "offset_mapping"
            )

            model_inputs = {
                key: value.to(DEVICE)
                for key, value
                in encoded.items()
            }

            outputs = model(
                **model_inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            predictions = (
                probabilities.argmax(
                    dim=-1
                )
            )

            predictions = (
                predictions.cpu()
            )

            probabilities = (
                probabilities.cpu()
            )

            input_ids = (
                encoded["input_ids"]
            )

            attention_mask = (
                encoded["attention_mask"]
            )

            for batch_index, text in enumerate(
                batch_texts
            ):

                tokens = (
                    tokenizer.convert_ids_to_tokens(
                        input_ids[
                            batch_index
                        ].tolist()
                    )
                )

                for token_index, token in enumerate(
                    tokens
                ):

                    if (
                        attention_mask[
                            batch_index,
                            token_index
                        ].item()
                        == 0
                    ):
                        continue

                    start, end = offsets[
                        batch_index,
                        token_index
                    ].tolist()

                    # Ignore special tokens.
                    if start == 0 and end == 0:
                        continue

                    total_real_tokens += 1

                    predicted_id = int(
                        predictions[
                            batch_index,
                            token_index
                        ].item()
                    )

                    predicted_counts[
                        predicted_id
                    ] += 1

                    probability9 = float(
                        probabilities[
                            batch_index,
                            token_index,
                            9
                        ].item()
                    )

                    if (
                        probability9
                        >
                        highest_output9_probability
                    ):

                        highest_output9_probability = (
                            probability9
                        )

                        highest_output9_info = {
                            "text":
                                text,

                            "token":
                                token,

                            "offset":
                                [start, end],

                            "probability9":
                                probability9,

                            "predicted_id":
                                predicted_id,
                        }

                    if predicted_id == 9:

                        total_output9 += 1

                        if (
                            len(output9_examples)
                            < 20
                        ):

                            output9_examples.append({
                                "text":
                                    text,

                                "token":
                                    token,

                                "offset":
                                    [start, end],

                                "probability":
                                    probability9,
                            })

            processed = min(
                batch_start + BATCH_SIZE,
                len(texts),
            )

            if (
                processed % 500 < BATCH_SIZE
                or
                processed == len(texts)
            ):

                print(
                    f"{processed}/"
                    f"{len(texts)}"
                )

    print(
        "\n" + "=" * 70
    )

    print("PREDICTION COUNTS")

    print("=" * 70)

    labels = {
        0: "O",
        1: "B-neu",
        2: "I-neu",
        3: "B-neg",
        4: "B-con",
        5: "I-pos",
        6: "B-pos",
        7: "I-con",
        8: "I-neg",
        9: "<UNKNOWN>",
    }

    for label_id in range(10):

        count = predicted_counts.get(
            label_id,
            0,
        )

        percentage = (
            count / total_real_tokens * 100
            if total_real_tokens
            else 0
        )

        print(
            f"{label_id:>2} "
            f"{labels[label_id]:<12}"
            f"{count:>10} "
            f"({percentage:.6f}%)"
        )

    print(
        "\nTotal real tokens:",
        total_real_tokens,
    )

    print(
        "Output-9 predictions:",
        total_output9,
    )

    print(
        "\nHighest probability ever "
        "assigned to output 9:"
    )

    print(
        f"{highest_output9_probability:.8f}"
    )

    if highest_output9_info:

        print(
            "\nHighest output-9 example:"
        )

        print(
            "Token:",
            highest_output9_info[
                "token"
            ],
        )

        print(
            "Predicted class:",
            highest_output9_info[
                "predicted_id"
            ],
        )

        print(
            "Text:"
        )

        print(
            highest_output9_info[
                "text"
            ]
        )

    print(
        "\n" + "=" * 70
    )

    print("OUTPUT 9 EXAMPLES")

    print("=" * 70)

    if not output9_examples:

        print(
            "\nNo real token was predicted "
            "as output 9."
        )

    else:

        for number, item in enumerate(
            output9_examples,
            start=1,
        ):

            print(
                f"\n#{number}"
            )

            print(
                "Token:",
                item["token"],
            )

            print(
                "Probability:",
                item[
                    "probability"
                ],
            )

            print(
                "Text:"
            )

            print(
                item["text"]
            )

    print(
        "\n" + "=" * 70
    )

    if total_output9 == 0:

        print(
            "RESULT: Output 9 was never "
            "the winning class."
        )

    else:

        print(
            "RESULT: Output 9 IS used by "
            "the original model."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()