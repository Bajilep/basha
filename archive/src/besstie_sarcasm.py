from __future__ import annotations

from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class BesstieSarcasmModel:
    """
    BESSTIE DeBERTa-v3 sarcasm detector.

    Used as a comparison candidate against
    CardiffNLP Twitter-RoBERTa Irony.
    """

    MODEL_NAME = (
        "ppokhrel2109/"
        "besstie-sarcasm-deberta-v3"
    )

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Loading BESSTIE sarcasm model "
            f"on {self.device}..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.MODEL_NAME
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.MODEL_NAME
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Model labels:",
            self.model.config.id2label,
        )

    def predict(
        self,
        text: str,
    ) -> Dict:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():

            output = self.model(
                **encoded
            )

        probabilities = (
            torch.softmax(
                output.logits,
                dim=-1,
            )[0]
            .detach()
            .cpu()
            .tolist()
        )

        id2label = (
            self.model.config.id2label
        )

        sarcasm_index = None
        non_sarcasm_index = None

        # -----------------------------------------------
        # Discover labels from model configuration.
        # -----------------------------------------------

        for index, label in id2label.items():

            label_lower = str(
                label
            ).lower()

            index = int(
                index
            )

            if (
                "sarcasm" in label_lower
                or "sarcastic" in label_lower
            ):

                if (
                    "non" not in label_lower
                    and "not" not in label_lower
                ):

                    sarcasm_index = index

            if (
                "non" in label_lower
                or "not" in label_lower
            ):

                non_sarcasm_index = index

        # -----------------------------------------------
        # Fallback for standard binary LABEL_0/LABEL_1
        # configuration.
        #
        # We will VERIFY this from the printed id2label
        # before trusting benchmark results.
        # -----------------------------------------------

        if sarcasm_index is None:

            sarcasm_index = 1

        if non_sarcasm_index is None:

            non_sarcasm_index = (
                0
                if sarcasm_index == 1
                else 1
            )

        sarcasm_probability = float(
            probabilities[
                sarcasm_index
            ]
        )

        non_sarcasm_probability = float(
            probabilities[
                non_sarcasm_index
            ]
        )

        return {
            "sarcasm":
                round(
                    sarcasm_probability,
                    4,
                ),

            "non_sarcasm":
                round(
                    non_sarcasm_probability,
                    4,
                ),

            "raw_probabilities": [
                round(
                    probability,
                    4,
                )
                for probability
                in probabilities
            ],
        }