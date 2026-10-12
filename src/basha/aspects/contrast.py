from __future__ import annotations

import re
from typing import Dict, List


class ContrastAnalyzer:
    """
    Detects contrastive clauses inside English sentences.

    Examples:
        The battery is excellent but the camera is terrible.

        I love the design, but I hate the battery.

        The performance is great although the software is buggy.
    """

    CONTRAST_WORDS = (
        "but",
        "however",
        "although",
        "though",
        "yet",
        "whereas",
        "while",
    )

    def __init__(
        self,
        sentiment_model,
        strong_threshold: float = 0.60,
    ) -> None:

        self.sentiment_model = (
            sentiment_model
        )

        self.strong_threshold = (
            strong_threshold
        )

        pattern = "|".join(
            re.escape(word)
            for word in self.CONTRAST_WORDS
        )

        self.pattern = re.compile(
            rf"\b({pattern})\b",
            flags=re.IGNORECASE,
        )

    def analyze(
        self,
        sentence: str,
    ) -> Dict:

        if not isinstance(sentence, str):
            raise TypeError(
                "sentence must be a string"
            )

        sentence = sentence.strip()

        if not sentence:
            raise ValueError(
                "sentence cannot be empty"
            )

        # ----------------------------------------------------
        # Find contrast marker
        # ----------------------------------------------------

        match = self.pattern.search(
            sentence
        )

        if match is None:

            normal_result = (
                self.sentiment_model.predict(
                    sentence
                )
            )

            return {
                "contrast_detected": False,
                "marker": None,
                "label": normal_result[
                    "label"
                ],
                "confidence": normal_result[
                    "confidence"
                ],
                "clauses": [],
                "original_sentiment":
                    normal_result,
            }

        marker = match.group(1).lower()

        # ----------------------------------------------------
        # Split around marker
        # ----------------------------------------------------

        left = sentence[
            :match.start()
        ].strip(
            " ,;:-"
        )

        right = sentence[
            match.end():
        ].strip(
            " ,;:-"
        )

        # Safety fallback
        if not left or not right:

            normal_result = (
                self.sentiment_model.predict(
                    sentence
                )
            )

            return {
                "contrast_detected": False,
                "marker": None,
                "label": normal_result[
                    "label"
                ],
                "confidence": normal_result[
                    "confidence"
                ],
                "clauses": [],
                "original_sentiment":
                    normal_result,
            }

        # ----------------------------------------------------
        # Analyze complete sentence
        # ----------------------------------------------------

        original_result = (
            self.sentiment_model.predict(
                sentence
            )
        )

        # ----------------------------------------------------
        # Analyze each clause independently
        # ----------------------------------------------------

        left_result = (
            self.sentiment_model.predict(
                left
            )
        )

        right_result = (
            self.sentiment_model.predict(
                right
            )
        )

        clauses: List[Dict] = [
            {
                "text": left,
                **left_result,
            },
            {
                "text": right,
                **right_result,
            },
        ]

        # ----------------------------------------------------
        # Check for genuine polarity conflict
        # ----------------------------------------------------

        left_label = left_result[
            "label"
        ]

        right_label = right_result[
            "label"
        ]

        opposite_polarity = (
            {
                left_label,
                right_label,
            }
            == {
                "positive",
                "negative",
            }
        )

        left_strong = (
            left_result["confidence"]
            >= self.strong_threshold
        )

        right_strong = (
            right_result["confidence"]
            >= self.strong_threshold
        )

        # ----------------------------------------------------
        # Mixed sentiment
        # ----------------------------------------------------

        if (
            opposite_polarity
            and left_strong
            and right_strong
        ):

            label = "mixed"

            # Temporary mixed confidence.
            # We will improve this later.
            confidence = (
                left_result["confidence"]
                + right_result["confidence"]
            ) / 2

        else:

            label = original_result[
                "label"
            ]

            confidence = original_result[
                "confidence"
            ]

        return {
            "contrast_detected": True,

            "marker": marker,

            "label": label,

            "confidence": round(
                confidence,
                4,
            ),

            "clauses": clauses,

            "original_sentiment":
                original_result,
        }