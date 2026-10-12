from __future__ import annotations

from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class EnglishSarcasmModel:
    """
    Production English sarcasm detector for Basha.

    Model:
        ppokhrel2109/besstie-sarcasm-deberta-v3

    Basha V1 decision threshold:
        0.80

    Sarcasm is treated as a separate linguistic signal.
    It does NOT automatically reverse sentiment.
    """

    MODEL_NAME = (
        "ppokhrel2109/"
        "besstie-sarcasm-deberta-v3"
    )

    DEFAULT_THRESHOLD = 0.80

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        self.threshold = threshold

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Loading English sarcasm model "
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
            key: value.to(self.device)
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():

            output = self.model(
                **encoded
            )

        probabilities = torch.softmax(
            output.logits,
            dim=-1,
        )[0]

        probabilities = (
            probabilities
            .detach()
            .cpu()
            .tolist()
        )

        # BESSTIE binary classification:
        #
        # LABEL_0 -> non-sarcasm
        # LABEL_1 -> sarcasm
        #
        # This mapping is frozen for Basha V1.

        non_sarcasm_probability = float(
            probabilities[0]
        )

        sarcasm_probability = float(
            probabilities[1]
        )

        detected = (
            sarcasm_probability
            >= self.threshold
        )

        return {
            "detected": detected,

            "label": (
                "sarcasm"
                if detected
                else "non_sarcasm"
            ),

            "confidence": round(
                (
                    sarcasm_probability
                    if detected
                    else non_sarcasm_probability
                ),
                4,
            ),

            "sarcasm": round(
                sarcasm_probability,
                4,
            ),

            "non_sarcasm": round(
                non_sarcasm_probability,
                4,
            ),

            "threshold": self.threshold,
        }