from __future__ import annotations

from typing import Dict, List

import torch
from transformers import pipeline


class DebertaAspectModel:
    """
    Basha ABSA Candidate #2.

    End-to-end aspect extraction + sentiment
    using DeBERTa-v3-base.
    """

    MODEL_NAME = (
        "yangheng/"
        "deberta-v3-base-end2end-absa"
    )

    def __init__(self) -> None:

        self.device = (
            0
            if torch.cuda.is_available()
            else -1
        )

        device_name = (
            "cuda"
            if self.device == 0
            else "cpu"
        )

        print(
            "Loading DeBERTa ABSA model "
            f"on {device_name}..."
        )

        self.pipeline = pipeline(
            task="token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy="simple",
            device=self.device,
        )

    @staticmethod
    def _extract_sentiment(
        label: str,
    ) -> str | None:

        label = str(label).strip()

        if not label:
            return None

        lower = label.lower()

        # Examples:
        # B-ASP-Positive
        # I-ASP-Negative
        # ASP-Positive
        # Positive

        if "positive" in lower:
            return "positive"

        if "negative" in lower:
            return "negative"

        if "neutral" in lower:
            return "neutral"

        # Outside/non-aspect label
        if lower == "o":
            return None

        return None

    def predict(
        self,
        text: str,
    ) -> List[Dict]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        raw_results = self.pipeline(
            text
        )

        aspects: List[Dict] = []

        for item in raw_results:

            label = item.get(
                "entity_group",
                "",
            )

            sentiment = (
                self._extract_sentiment(
                    label
                )
            )

            if sentiment is None:
                continue

            aspect = str(
                item.get(
                    "word",
                    "",
                )
            ).strip()

            if not aspect:
                continue

            aspects.append(
                {
                    "aspect": aspect,

                    "sentiment":
                        sentiment,

                    "confidence": round(
                        float(
                            item.get(
                                "score",
                                0.0,
                            )
                        ),
                        4,
                    ),

                    "start": int(
                        item.get(
                            "start",
                            0,
                        )
                    ),

                    "end": int(
                        item.get(
                            "end",
                            0,
                        )
                    ),
                }
            )

        return aspects