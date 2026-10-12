from __future__ import annotations

import re
from typing import Dict, List

import torch
from transformers import pipeline


class EnglishAspectModel:
    """
    End-to-end English Aspect-Based Sentiment Analysis.

    Extracts:
        - aspect term
        - aspect sentiment
        - confidence
        - character position

    Example:
        "The battery is excellent but the camera is terrible."

    Expected:
        battery -> positive
        camera  -> negative
    """

    MODEL_NAME = (
        "gauneg/"
        "roberta-base-absa-ate-sentiment"
    )

    LABEL_MAP = {
        "pos": "positive",
        "positive": "positive",

        "neg": "negative",
        "negative": "negative",

        "neu": "neutral",
        "neutral": "neutral",
    }

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
            "Loading English aspect model "
            f"on {device_name}..."
        )

        self.pipeline = pipeline(
            task="token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy="simple",
            device=self.device,
        )

    @classmethod
    def _normalize_label(
        cls,
        label: str,
    ) -> str:

        normalized = (
            str(label)
            .strip()
            .lower()
        )

        return cls.LABEL_MAP.get(
            normalized,
            normalized,
        )

    @staticmethod
    def _is_valid_aspect(
        aspect: str,
    ) -> bool:
        """
        Check whether an extracted aspect contains
        meaningful characters.

        Rejects punctuation-only predictions such as:
            "."
            ","
            "!"
            "?"
    """

        aspect = aspect.strip()

        if not aspect:
            return False

        # Aspect must contain at least one
        # English letter or number.
        if not re.search(
            r"[A-Za-z0-9]",
            aspect,
        ):
            return False

        return True

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

            aspect = (
                str(item["word"])
                .strip()
            )

            # Remove invalid predictions such as "."
            if not self._is_valid_aspect(
                aspect
            ):
                continue

            sentiment = (
                self._normalize_label(
                    item["entity_group"]
                )
            )

            aspects.append(
                {
                    "aspect": aspect,

                    "sentiment":
                        sentiment,

                    "confidence": round(
                        float(
                            item["score"]
                        ),
                        4,
                    ),

                    "start": int(
                        item["start"]
                    ),

                    "end": int(
                        item["end"]
                    ),
                }
            )

        return aspects