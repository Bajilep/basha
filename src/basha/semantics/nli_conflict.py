from __future__ import annotations

from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class NLIConflictDetector:
    """
    Semantic contradiction detector for Basha.

    This component compares two pieces of text and asks:

        Do they support each other?
        Are they unrelated?
        Do they contradict each other?

    It will later help sarcasm resolution.

    Example:

        premise:
            "My laptop crashed before I saved my work."

        hypothesis:
            "This was a perfect experience."

        High contradiction can support sarcastic interpretation.
    """

    MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading NLI model on {self.device}..."
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
            "NLI labels:",
            self.model.config.id2label,
        )

    def predict(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict:

        if not isinstance(premise, str):
            raise TypeError(
                "premise must be a string"
            )

        if not isinstance(hypothesis, str):
            raise TypeError(
                "hypothesis must be a string"
            )

        premise = premise.strip()
        hypothesis = hypothesis.strip()

        if not premise:
            raise ValueError(
                "premise cannot be empty"
            )

        if not hypothesis:
            raise ValueError(
                "hypothesis cannot be empty"
            )

        encoded = self.tokenizer(
            premise,
            hypothesis,
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

        # Read model labels instead of assuming
        # a fixed class order.

        scores = {}

        for index, probability in enumerate(
            probabilities
        ):

            label = (
                self.model.config.id2label[
                    index
                ]
                .lower()
            )

            scores[label] = float(
                probability
            )

        contradiction = 0.0
        entailment = 0.0
        neutral = 0.0

        for label, score in scores.items():

            if "contradiction" in label:
                contradiction = score

            elif "entailment" in label:
                entailment = score

            elif "neutral" in label:
                neutral = score

        predicted_label = max(
            scores,
            key=scores.get,
        )

        return {
            "label":
                predicted_label,

            "confidence":
                round(
                    scores[
                        predicted_label
                    ],
                    4,
                ),

            "contradiction":
                round(
                    contradiction,
                    4,
                ),

            "entailment":
                round(
                    entailment,
                    4,
                ),

            "neutral":
                round(
                    neutral,
                    4,
                ),

            "raw_scores": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in scores.items()
            },
        }