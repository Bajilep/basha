import time
import gc

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
MAX_SAMPLES = 1000


MODELS = {

    "CardiffNLP": (
        "cardiffnlp/"
        "twitter-roberta-base-sentiment-latest"
    ),

    "Hartmann-RoBERTa-Large": (
        "j-hartmann/"
        "sentiment-roberta-large-english-3-classes"
    ),

    "BERTweet": (
        "finiteautomata/"
        "bertweet-base-sentiment-analysis"
    ),
}


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("LOADING TWEETEVAL")
print("=" * 80)


dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "sentiment",
)


full_test = dataset["test"]


test_data = (
    full_test
    .shuffle(seed=SEED)
    .select(
        range(
            min(
                MAX_SAMPLES,
                len(full_test),
            )
        )
    )
)


print(
    "Samples:",
    len(test_data)
)

print(
    "Seed:",
    SEED
)


# ============================================================
# TRUE LABELS
# ============================================================

y_true = [
    int(example["label"])
    for example in test_data
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).lower()

    # Negative
    if label in {
        "negative",
        "neg",
        "label_0",
    }:
        return 0

    # Neutral
    if label in {
        "neutral",
        "neu",
        "label_1",
    }:
        return 1

    # Positive
    if label in {
        "positive",
        "pos",
        "label_2",
    }:
        return 2

    raise ValueError(
        f"Unknown model label: {label}"
    )


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark_model(
    display_name,
    model_name,
):

    print("\n")
    print("=" * 80)

    print(
        f"MODEL: {display_name}"
    )

    print(
        f"CHECKPOINT: {model_name}"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name
        )
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model.to(device)

    model.eval()


    print(
        "Device:",
        device
    )


    print(
        "Labels:",
        model.config.id2label
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = []


    start_time = time.perf_counter()


    for index, example in enumerate(
        test_data
    ):

        text = example["text"]


        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )


        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }


        with torch.inference_mode():

            output = model(
                **encoded
            )


        predicted_id = int(
            torch.argmax(
                output.logits,
                dim=-1,
            ).item()
        )


        raw_label = (
            model.config.id2label[
                predicted_id
            ]
        )


        normalized = normalize_label(
            raw_label
        )


        predictions.append(
            normalized
        )


        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_data)}"
            )


    end_time = time.perf_counter()


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    total_seconds = (
        end_time
        - start_time
    )


    milliseconds_per_sample = (
        total_seconds
        / len(test_data)
        * 1000
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    macro_precision = float(
        precision.mean()
    )


    macro_recall = float(
        recall.mean()
    )


    macro_f1 = float(
        f1.mean()
    )


    print("\nRESULT")

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )


    print(
        "\nCLASS F1"
    )

    print(
        f"Negative: "
        f"{f1[0] * 100:.2f}%"
    )

    print(
        f"Neutral: "
        f"{f1[1] * 100:.2f}%"
    )

    print(
        f"Positive: "
        f"{f1[2] * 100:.2f}%"
    )


    print(
        "\nSPEED"
    )

    print(
        f"Total inference time: "
        f"{total_seconds:.2f} sec"
    )

    print(
        f"Average: "
        f"{milliseconds_per_sample:.2f} "
        f"ms/sample"
    )


    result = {

        "name": display_name,

        "accuracy": (
            accuracy * 100
        ),

        "macro_f1": (
            macro_f1 * 100
        ),

        "negative_f1": (
            f1[0] * 100
        ),

        "neutral_f1": (
            f1[1] * 100
        ),

        "positive_f1": (
            f1[2] * 100
        ),

        "seconds": (
            total_seconds
        ),

        "ms_per_sample": (
            milliseconds_per_sample
        ),
    }


    # --------------------------------------------------------
    # Free memory before next model
    # --------------------------------------------------------

    del model
    del tokenizer

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# RUN ALL MODELS
# ============================================================

results = []


for display_name, model_name in MODELS.items():

    try:

        result = benchmark_model(
            display_name,
            model_name,
        )

        results.append(
            result
        )


    except Exception as error:

        print(
            f"\nFAILED: "
            f"{display_name}"
        )

        print(
            "ERROR:",
            error
        )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")

print("=" * 110)

print(
    "FINAL MODEL COMPARISON"
)

print("=" * 110)


header = (
    f"{'MODEL':<28}"
    f"{'ACC':>10}"
    f"{'MACRO F1':>12}"
    f"{'NEG F1':>10}"
    f"{'NEU F1':>10}"
    f"{'POS F1':>10}"
    f"{'MS/SAMPLE':>14}"
)


print(
    header
)

print(
    "-" * 110
)


for result in results:

    print(

        f"{result['name']:<28}"

        f"{result['accuracy']:>9.2f}%"

        f"{result['macro_f1']:>11.2f}%"

        f"{result['negative_f1']:>9.2f}%"

        f"{result['neutral_f1']:>9.2f}%"

        f"{result['positive_f1']:>9.2f}%"

        f"{result['ms_per_sample']:>14.2f}"
    )


# ============================================================
# WINNER
# ============================================================

if results:

    winner = max(
        results,
        key=lambda item: item[
            "macro_f1"
        ],
    )


    print("\n" + "=" * 110)

    print(
        "BEST MACRO-F1 MODEL:"
    )

    print(
        winner["name"]
    )

    print(
        f"Macro F1: "
        f"{winner['macro_f1']:.2f}%"
    )

    print("=" * 110)