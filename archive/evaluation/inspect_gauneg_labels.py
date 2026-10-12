from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)

MODEL_NAME = "gauneg/roberta-base-absa-ate-sentiment"


def main():

    print("=" * 70)
    print("BASHA — GAUNEG LABEL ARCHITECTURE CHECK")
    print("=" * 70)

    config = AutoConfig.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    print("\nCONFIG")
    print("-" * 50)

    print(
        "num_labels:",
        config.num_labels
    )

    print(
        "classifier.out_features:",
        model.classifier.out_features
    )

    print("\nRAW id2label:")
    print(config.id2label)

    print("\nRAW label2id:")
    print(config.label2id)

    print("\nALL CLASSIFIER OUTPUT INDICES")
    print("-" * 50)

    for index in range(
        model.classifier.out_features
    ):

        label = config.id2label.get(
            index,
            "<NO LABEL IN CONFIG>"
        )

        print(
            f"output index {index:>2}"
            f" → {label}"
        )

    print("\nCLASSIFIER PARAMETERS")
    print("-" * 50)

    print(
        "weight shape:",
        tuple(
            model.classifier.weight.shape
        )
    )

    print(
        "bias shape:",
        tuple(
            model.classifier.bias.shape
        )
    )

    print("\nOUTPUT HEAD WEIGHT NORMS")
    print("-" * 50)

    # Looking at the norm is useful for seeing whether
    # output 9 looks like a real learned classifier row.
    weight = (
        model.classifier.weight
        .detach()
        .cpu()
    )

    bias = (
        model.classifier.bias
        .detach()
        .cpu()
    )

    for index in range(
        weight.shape[0]
    ):

        norm = (
            weight[index]
            .norm()
            .item()
        )

        bias_value = (
            bias[index]
            .item()
        )

        label = config.id2label.get(
            index,
            "<NO LABEL>"
        )

        print(
            f"{index:>2} "
            f"{label:<15} "
            f"norm={norm:.6f} "
            f"bias={bias_value:.6f}"
        )

    print("\nEXPECTED BASHA TRAINING LABELS")
    print("-" * 50)

    expected = {
        0: "O",
        1: "B-neu",
        2: "I-neu",
        3: "B-neg",
        4: "B-con",
        5: "I-pos",
        6: "B-pos",
        7: "I-con",
        8: "I-neg",
    }

    for index, label in expected.items():

        config_label = (
            config.id2label.get(index)
        )

        status = (
            "✓"
            if config_label == label
            else "✗"
        )

        print(
            f"{status} "
            f"{index} "
            f"expected={label:<8} "
            f"config={config_label}"
        )

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)

    print(
        "\nDo NOT fine-tune yet."
    )


if __name__ == "__main__":
    main()