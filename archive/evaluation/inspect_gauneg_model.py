from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_NAME = "gauneg/roberta-base-absa-ate-sentiment"


def main():

    print("=" * 70)
    print("BASHA — GAUNEG MODEL INSPECTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    print("\nLoading model configuration...")

    config = AutoConfig.from_pretrained(
        MODEL_NAME
    )

    print("\nMODEL")
    print("-" * 50)

    print("Model name:")
    print(MODEL_NAME)

    print("\nArchitecture:")
    print(config.architectures)

    print("\nNumber of labels:")
    print(config.num_labels)

    # ---------------------------------------------------------
    # 2. Label mapping
    # ---------------------------------------------------------

    print("\nID → LABEL")
    print("-" * 50)

    for label_id, label in sorted(
        config.id2label.items(),
        key=lambda item: int(item[0]),
    ):
        print(
            f"{label_id:>3} → {label}"
        )

    print("\nLABEL → ID")
    print("-" * 50)

    for label, label_id in sorted(
        config.label2id.items(),
        key=lambda item: item[1],
    ):
        print(
            f"{label:<25} → {label_id}"
        )

    # ---------------------------------------------------------
    # 3. Tokenizer
    # ---------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print("\nTOKENIZER")
    print("-" * 50)

    print(
        "Tokenizer class:",
        tokenizer.__class__.__name__,
    )

    print(
        "Fast tokenizer:",
        tokenizer.is_fast,
    )

    print(
        "Model max length:",
        tokenizer.model_max_length,
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    print(
        "PAD token:",
        tokenizer.pad_token,
    )

    print(
        "PAD token ID:",
        tokenizer.pad_token_id,
    )

    print(
        "CLS/BOS token:",
        tokenizer.bos_token,
    )

    print(
        "SEP/EOS token:",
        tokenizer.eos_token,
    )

    # ---------------------------------------------------------
    # 4. Inspect tokenization
    # ---------------------------------------------------------

    examples = [
        "Food is pretty good but the service is horrific.",
        "The battery life is excellent but the camera quality is terrible.",
        "The customer service was slow, although the food was amazing.",
    ]

    print("\nTOKENIZATION EXAMPLES")
    print("=" * 70)

    for text in examples:

        print("\nTEXT:")
        print(text)

        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        print("\nTOKEN".ljust(25), "OFFSET")
        print("-" * 45)

        for token, offset in zip(
            tokens,
            encoded["offset_mapping"],
        ):
            print(
                f"{token:<25} {offset}"
            )

    # ---------------------------------------------------------
    # 5. Load actual model
    # ---------------------------------------------------------

    print("\nLoading token classification model...")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME
    )

    print("\nMODEL HEAD")
    print("-" * 50)

    print(model.classifier)

    print(
        "\nClassifier output labels:",
        model.config.num_labels,
    )

    print("\n" + "=" * 70)
    print("GAUNEG INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()