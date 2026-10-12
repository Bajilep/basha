from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gauneg/roberta-base-absa-ate-sentiment"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "gauneg_tokenized_train.json"
VALIDATION_FILE = DATA_DIR / "gauneg_tokenized_validation.json"

OUTPUT_DIR = MODEL_DIR / "gauneg_basha_v1"
FINAL_MODEL_DIR = MODEL_DIR / "gauneg_basha_v1" / "best_model"

TRAINING_REPORT = (
    RESULTS_DIR / "gauneg_finetuning_report.json"
)

SEED = 42

NUM_EPOCHS = 3

# CPU-safe starting point.
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8

# Effective training batch:
# 4 × 2 = 8
GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 2

IGNORE_INDEX = -100


# ============================================================
# GAUNEG LABELS
# ============================================================

LABEL2ID = {
    "O": 0,
    "B-neu": 1,
    "I-neu": 2,
    "B-neg": 3,
    "B-con": 4,
    "I-pos": 5,
    "B-pos": 6,
    "I-con": 7,
    "I-neg": 8,
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# DATASET
# ============================================================

class TokenizedABSADataset(Dataset):

    def __init__(self, examples):

        self.examples = examples

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        return {
            "input_ids":
                example["input_ids"],

            "attention_mask":
                example["attention_mask"],

            "labels":
                example["labels"],
        }


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(
    examples,
    split_name,
):

    print(
        f"\nValidating {split_name}..."
    )

    invalid_labels = 0
    length_mismatches = 0
    empty_examples = 0

    valid_training_labels = set(
        LABEL2ID.values()
    )

    for index, example in enumerate(
        examples
    ):

        input_ids = example[
            "input_ids"
        ]

        attention_mask = example[
            "attention_mask"
        ]

        labels = example[
            "labels"
        ]

        if not input_ids:
            empty_examples += 1

        if not (
            len(input_ids)
            ==
            len(attention_mask)
            ==
            len(labels)
        ):

            length_mismatches += 1

            print(
                f"Length mismatch "
                f"at example {index}"
            )

        for label in labels:

            if (
                label != IGNORE_INDEX
                and
                label not in valid_training_labels
            ):

                invalid_labels += 1

                print(
                    f"Invalid label "
                    f"{label} "
                    f"at example {index}"
                )

    print(
        "Examples:",
        len(examples),
    )

    print(
        "Length mismatches:",
        length_mismatches,
    )

    print(
        "Invalid labels:",
        invalid_labels,
    )

    print(
        "Empty examples:",
        empty_examples,
    )

    if (
        length_mismatches > 0
        or invalid_labels > 0
        or empty_examples > 0
    ):

        raise RuntimeError(
            f"{split_name} validation failed."
        )

    print(
        f"{split_name}: PASS"
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    true_labels = []
    predicted_labels = []

    for prediction_row, label_row in zip(
        predictions,
        labels,
    ):

        for predicted, gold in zip(
            prediction_row,
            label_row,
        ):

            if gold == IGNORE_INDEX:
                continue

            true_labels.append(
                int(gold)
            )

            predicted_labels.append(
                int(predicted)
            )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int64,
    )

    predicted_labels = np.asarray(
        predicted_labels,
        dtype=np.int64,
    )

    if len(true_labels) == 0:

        return {
            "token_accuracy": 0.0,
            "aspect_token_accuracy": 0.0,
            "output9_rate": 0.0,
        }

    token_accuracy = float(
        np.mean(
            predicted_labels
            ==
            true_labels
        )
    )

    # --------------------------------------------------------
    # Aspect-token accuracy
    #
    # Ignore O tokens here because otherwise the very large
    # number of O labels can make the model look better than
    # it actually is.
    # --------------------------------------------------------

    aspect_mask = (
        true_labels != LABEL2ID["O"]
    )

    if np.any(aspect_mask):

        aspect_token_accuracy = float(
            np.mean(
                predicted_labels[
                    aspect_mask
                ]
                ==
                true_labels[
                    aspect_mask
                ]
            )
        )

    else:

        aspect_token_accuracy = 0.0

    output9_rate = float(
        np.mean(
            predicted_labels == 9
        )
    )

    # --------------------------------------------------------
    # Per-label recall
    # --------------------------------------------------------

    metrics = {
        "token_accuracy":
            token_accuracy,

        "aspect_token_accuracy":
            aspect_token_accuracy,

        "output9_rate":
            output9_rate,
    }

    for label_id, label_name in (
        ID2LABEL.items()
    ):

        mask = (
            true_labels == label_id
        )

        if not np.any(mask):
            continue

        recall = float(
            np.mean(
                predicted_labels[mask]
                ==
                label_id
            )
        )

        safe_name = (
            label_name
            .replace("-", "_")
        )

        metrics[
            f"recall_{safe_name}"
        ] = recall

    return metrics


# ============================================================
# TRAINING REPORT
# ============================================================

def save_training_report(
    trainer,
    train_size,
    validation_size,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    best_metric = (
        trainer.state.best_metric
    )

    report = {
        "base_model":
            MODEL_NAME,

        "device":
            str(
                trainer.args.device
            ),

        "train_examples":
            train_size,

        "validation_examples":
            validation_size,

        "epochs_requested":
            NUM_EPOCHS,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "eval_batch_size":
            EVAL_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            (
                TRAIN_BATCH_SIZE
                *
                GRADIENT_ACCUMULATION_STEPS
            ),

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_ratio":
            WARMUP_RATIO,

        "best_checkpoint":
            best_checkpoint,

        "best_metric":
            best_metric,

        "global_step":
            trainer.state.global_step,

        "epoch":
            trainer.state.epoch,

        "log_history":
            trainer.state.log_history,
    }

    with TRAINING_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "BASHA — EXPERIMENTAL "
        "GAUNEG FINE-TUNING"
    )
    print("=" * 75)

    set_seed(SEED)

    print(
        "\nPyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "CPU threads:",
        torch.get_num_threads(),
    )

    print(
        "\nBase model:"
    )

    print(MODEL_NAME)

    print(
        "\nLoading prepared datasets..."
    )

    train_examples = load_json(
        TRAIN_FILE
    )

    validation_examples = load_json(
        VALIDATION_FILE
    )

    validate_dataset(
        train_examples,
        "TRAIN",
    )

    validate_dataset(
        validation_examples,
        "VALIDATION",
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    print(
        "Loading original Gauneg model..."
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # CRITICAL MODEL CHECKS
    # --------------------------------------------------------

    print(
        "\nClassifier outputs:",
        model.classifier.out_features,
    )

    if (
        model.classifier.out_features
        != 10
    ):

        raise RuntimeError(
            "Expected original Gauneg "
            "10-output classifier."
        )

    print(
        "Preserving original "
        "10-output classifier."
    )

    print(
        "Training targets use "
        "only labels 0-8."
    )

    print(
        "Label -100 remains "
        "the ignore index."
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        TokenizedABSADataset(
            train_examples
        )
    )

    validation_dataset = (
        TokenizedABSADataset(
            validation_examples
        )
    )

    # Dynamic padding saves CPU work compared with padding
    # every sentence to length 512.
    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=IGNORE_INDEX,
            return_tensors="pt",
        )
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR
        ),

        do_train=True,
        do_eval=True,

        use_cpu=True,

        num_train_epochs=(
            NUM_EPOCHS
        ),

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=(
            LEARNING_RATE
        ),

        weight_decay=(
            WEIGHT_DECAY
        ),

        # 10% of total training steps.
        warmup_steps=0,

        lr_scheduler_type="linear",

        optim="adamw_torch",

        max_grad_norm=1.0,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",

        logging_steps=100,
        logging_first_step=True,

        load_best_model_at_end=True,

        metric_for_best_model=(
            "aspect_token_accuracy"
        ),

        greater_is_better=True,

        save_total_limit=2,

        seed=SEED,
        data_seed=SEED,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        report_to="none",

        fp16=False,
        bf16=False,

        remove_unused_columns=True,
    )

    # --------------------------------------------------------
    # WARMUP
    #
    # Your Transformers build exposes warmup_steps rather
    # than warmup_ratio in the signature you printed.
    # Calculate the desired ~10% ourselves.
    # --------------------------------------------------------

    batches_per_epoch = math.ceil(
        len(train_dataset)
        /
        TRAIN_BATCH_SIZE
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        /
        GRADIENT_ACCUMULATION_STEPS
    )

    estimated_total_steps = (
        optimizer_steps_per_epoch
        *
        NUM_EPOCHS
    )

    calculated_warmup_steps = max(
        1,
        int(
            estimated_total_steps
            *
            WARMUP_RATIO
        ),
    )

    training_args.warmup_steps = (
        calculated_warmup_steps
    )

    print(
        "\nTraining configuration"
    )

    print("-" * 50)

    print(
        "Train examples:",
        len(train_dataset),
    )

    print(
        "Validation examples:",
        len(validation_dataset),
    )

    print(
        "Epochs:",
        NUM_EPOCHS,
    )

    print(
        "Train batch:",
        TRAIN_BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Effective batch:",
        TRAIN_BATCH_SIZE
        *
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Estimated optimizer steps:",
        estimated_total_steps,
    )

    print(
        "Warmup steps:",
        calculated_warmup_steps,
    )

    print(
        "Device: CPU"
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            validation_dataset
        ),

        data_collator=(
            data_collator
        ),

        processing_class=(
            tokenizer
        ),

        compute_metrics=(
            compute_metrics
        ),

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    # --------------------------------------------------------
    # BASELINE EVALUATION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "ORIGINAL GAUNEG "
        "VALIDATION BASELINE"
    )

    print("=" * 75)

    baseline_metrics = (
        trainer.evaluate()
    )

    for key, value in (
        baseline_metrics.items()
    ):

        print(
            f"{key:<40}"
            f"{value}"
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print("STARTING TRAINING")

    print("=" * 75)

    print(
        "\nThis is CPU training."
    )

    print(
        "Do not close PyCharm/"
        "PowerShell while it is running."
    )

    train_result = trainer.train()

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "BEST MODEL VALIDATION"
    )

    print("=" * 75)

    final_metrics = (
        trainer.evaluate()
    )

    for key, value in (
        final_metrics.items()
    ):

        print(
            f"{key:<40}"
            f"{value}"
        )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    print(
        "\nSaving best experimental model..."
    )

    trainer.save_model(
        str(FINAL_MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_MODEL_DIR)
    )

    save_training_report(
        trainer=trainer,
        train_size=len(
            train_dataset
        ),
        validation_size=len(
            validation_dataset
        ),
    )

    # Save baseline/final metrics separately.
    metrics_file = (
        RESULTS_DIR
        /
        "gauneg_finetuning_metrics.json"
    )

    with metrics_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "baseline_validation":
                    baseline_metrics,

                "best_model_validation":
                    final_metrics,

                "train_metrics":
                    train_result.metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n" + "=" * 75
    )

    print("TRAINING COMPLETE")

    print("=" * 75)

    print(
        "\nBest checkpoint:"
    )

    print(
        trainer.state.best_model_checkpoint
    )

    print(
        "\nBest validation metric:"
    )

    print(
        trainer.state.best_metric
    )

    print(
        "\nExperimental model saved to:"
    )

    print(
        FINAL_MODEL_DIR
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        metrics_file
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT replace Basha's current "
        "Gauneg model yet."
    )

    print(
        "The new model must pass the "
        "independent benchmarks first."
    )


if __name__ == "__main__":
    main()