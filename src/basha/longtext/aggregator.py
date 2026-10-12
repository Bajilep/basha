from __future__ import annotations

from typing import Dict, List


class SentimentAggregator:
    """
    Combines sentence-level sentiment probabilities
    into document-level sentiment.

    Mixed is derived from conflicting strong
    positive and negative sentence evidence.
    """

    def __init__(
        self,
        strong_threshold: float = 0.65,
        mixed_min_ratio: float = 0.25,
    ) -> None:

        self.strong_threshold = (
            strong_threshold
        )

        self.mixed_min_ratio = (
            mixed_min_ratio
        )

    def aggregate(
        self,
        results: List[Dict],
    ) -> Dict:

        if not results:
            raise ValueError(
                "results cannot be empty"
            )

        count = len(results)

        # --------------------------------------------
        # Average document probabilities
        # --------------------------------------------

        negative = sum(
            item["negative"]
            for item in results
        ) / count

        neutral = sum(
            item["neutral"]
            for item in results
        ) / count

        positive = sum(
            item["positive"]
            for item in results
        ) / count

        # --------------------------------------------
        # Strong sentence evidence
        # --------------------------------------------

        strong_positive = sum(
            1
            for item in results
            if (
                item["label"] == "positive"
                and item["positive"]
                >= self.strong_threshold
            )
        )

        strong_negative = sum(
            1
            for item in results
            if (
                item["label"] == "negative"
                and item["negative"]
                >= self.strong_threshold
            )
        )

        positive_ratio = (
            strong_positive / count
        )

        negative_ratio = (
            strong_negative / count
        )

        # --------------------------------------------
        # Mixed detection
        # --------------------------------------------

        is_mixed = (
            strong_positive > 0
            and strong_negative > 0
            and positive_ratio
            >= self.mixed_min_ratio
            and negative_ratio
            >= self.mixed_min_ratio
        )

        if is_mixed:

            label = "mixed"

            confidence = (
                positive
                + negative
            )

        else:

            probabilities = {
                "negative": negative,
                "neutral": neutral,
                "positive": positive,
            }

            label = max(
                probabilities,
                key=probabilities.get,
            )

            confidence = probabilities[
                label
            ]

        return {
            "label": label,

            "confidence": round(
                confidence,
                4,
            ),

            "negative": round(
                negative,
                4,
            ),

            "neutral": round(
                neutral,
                4,
            ),

            "positive": round(
                positive,
                4,
            ),

            "strong_positive_sentences":
                strong_positive,

            "strong_negative_sentences":
                strong_negative,

            "sentence_count":
                count,
        }