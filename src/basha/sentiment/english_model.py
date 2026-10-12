from __future__ import annotations

from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class EnglishSentimentModel:
    """
    English sentiment classifier for Basha.

    Labels
    ------
    0 -> Negative
    1 -> Neutral
    2 -> Positive
    """

    MODEL_NAME = (
        "cardiffnlp/"
        "twitter-roberta-base-sentiment-latest"
    )

    LABELS = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading English sentiment model "
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

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess(text: str) -> str:
        """
        Preprocessing recommended for the
        Twitter RoBERTa sentiment model.
        """

        tokens = []

        for token in text.split():

            if (
                token.startswith("@")
                and len(token) > 1
            ):
                token = "@user"

            elif token.startswith(
                ("http://", "https://")
            ):
                token = "http"

            tokens.append(token)

        return " ".join(tokens)

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

        clean_text = self.preprocess(text)

        encoded = self.tokenizer(
            clean_text,
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

        prediction_id = int(
            max(
                range(len(probabilities)),
                key=probabilities.__getitem__,
            )
        )

        return {
            "label": self.LABELS[
                prediction_id
            ],
            "confidence": round(
                probabilities[
                    prediction_id
                ],
                4,
            ),
            "negative": round(
                probabilities[0],
                4,
            ),
            "neutral": round(
                probabilities[1],
                4,
            ),
            "positive": round(
                probabilities[2],
                4,
            ),
        }