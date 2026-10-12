from __future__ import annotations

from typing import Dict, List


class MixedDetector:
    """
    Detect whether a text contains genuinely mixed sentiment.

    Mixed sentiment means that meaningful positive AND
    negative evidence exist in the same text.

    Evidence can come from:

        1. Aspect sentiment
        2. Contrast analysis
        3. Sentence-level sentiment

    The normal document sentiment model is NOT enough by
    itself to declare Mixed.
    """

    VALID_SENTIMENTS = {
        "positive",
        "negative",
        "neutral",
    }

    def __init__(
        self,
        aspect_confidence_threshold: float = 0.50,
        sentence_confidence_threshold: float = 0.65,
        contrast_confidence_threshold: float = 0.60,
    ) -> None:

        self.aspect_confidence_threshold = (
            aspect_confidence_threshold
        )

        self.sentence_confidence_threshold = (
            sentence_confidence_threshold
        )

        self.contrast_confidence_threshold = (
            contrast_confidence_threshold
        )

    # ========================================================
    # ASPECT EVIDENCE
    # ========================================================

    def _analyze_aspects(
        self,
        aspects: List[Dict],
    ) -> Dict:

        positive = []
        negative = []
        neutral = []

        for aspect in aspects:

            sentiment = (
                str(
                    aspect.get(
                        "sentiment",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            confidence = float(
                aspect.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                sentiment
                not in self.VALID_SENTIMENTS
            ):
                continue

            # We do not completely delete low-confidence
            # predictions from Basha.
            #
            # We only prevent weak predictions from being
            # strong evidence for Mixed.
            if (
                confidence
                < self.aspect_confidence_threshold
            ):
                continue

            evidence = {
                "aspect": aspect.get(
                    "aspect"
                ),
                "sentiment": sentiment,
                "confidence": confidence,
            }

            if sentiment == "positive":

                positive.append(
                    evidence
                )

            elif sentiment == "negative":

                negative.append(
                    evidence
                )

            else:

                neutral.append(
                    evidence
                )

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        }

    # ========================================================
    # SENTENCE EVIDENCE
    # ========================================================

    def _analyze_sentences(
        self,
        sentences: List[Dict],
    ) -> Dict:

        positive = []
        negative = []

        for sentence in sentences:

            label = (
                str(
                    sentence.get(
                        "label",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            confidence = float(
                sentence.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                confidence
                < self.sentence_confidence_threshold
            ):
                continue

            evidence = {
                "text": sentence.get(
                    "text"
                ),
                "label": label,
                "confidence": confidence,
            }

            if label == "positive":

                positive.append(
                    evidence
                )

            elif label == "negative":

                negative.append(
                    evidence
                )

        return {
            "positive": positive,
            "negative": negative,
        }

    # ========================================================
    # CONTRAST EVIDENCE
    # ========================================================

    def _analyze_contrast(
        self,
        contrast: Dict | None,
    ) -> Dict:

        result = {
            "supports_mixed": False,
            "confidence": 0.0,
        }

        if not contrast:
            return result

        label = (
            str(
                contrast.get(
                    "label",
                    "",
                )
            )
            .strip()
            .lower()
        )

        confidence = float(
            contrast.get(
                "confidence",
                0.0,
            )
        )

        if (
            label == "mixed"
            and confidence
            >= self.contrast_confidence_threshold
        ):

            result[
                "supports_mixed"
            ] = True

            result[
                "confidence"
            ] = confidence

        return result

    # ========================================================
    # FINAL DECISION
    # ========================================================

    def detect(
        self,
        aspects: List[Dict] | None = None,
        sentences: List[Dict] | None = None,
        contrast: Dict | None = None,
    ) -> Dict:

        aspects = (
            aspects
            if aspects is not None
            else []
        )

        sentences = (
            sentences
            if sentences is not None
            else []
        )

        # ----------------------------------------------------
        # Collect evidence
        # ----------------------------------------------------

        aspect_evidence = (
            self._analyze_aspects(
                aspects
            )
        )

        sentence_evidence = (
            self._analyze_sentences(
                sentences
            )
        )

        contrast_evidence = (
            self._analyze_contrast(
                contrast
            )
        )

        # ----------------------------------------------------
        # Positive / negative evidence
        # ----------------------------------------------------

        positive_aspects = (
            aspect_evidence[
                "positive"
            ]
        )

        negative_aspects = (
            aspect_evidence[
                "negative"
            ]
        )

        positive_sentences = (
            sentence_evidence[
                "positive"
            ]
        )

        negative_sentences = (
            sentence_evidence[
                "negative"
            ]
        )

        has_positive = bool(
            positive_aspects
            or positive_sentences
        )

        has_negative = bool(
            negative_aspects
            or negative_sentences
        )

        # ----------------------------------------------------
        # Strongest evidence
        # ----------------------------------------------------

        positive_scores = [
            item["confidence"]
            for item in (
                positive_aspects
                + positive_sentences
            )
        ]

        negative_scores = [
            item["confidence"]
            for item in (
                negative_aspects
                + negative_sentences
            )
        ]

        strongest_positive = (
            max(positive_scores)
            if positive_scores
            else 0.0
        )

        strongest_negative = (
            max(negative_scores)
            if negative_scores
            else 0.0
        )

        # ----------------------------------------------------
        # Mixed rules
        # ----------------------------------------------------

        mixed = False
        reasons = []

        # Rule 1:
        # Opposite aspect sentiments are our strongest
        # direct evidence for Mixed.
        if (
            positive_aspects
            and negative_aspects
        ):

            mixed = True

            reasons.append(
                "opposite_aspect_sentiments"
            )

        # Rule 2:
        # Different sentences express opposite sentiment.
        if (
            positive_sentences
            and negative_sentences
        ):

            mixed = True

            reasons.append(
                "opposite_sentence_sentiments"
            )

        # Rule 3:
        # Contrast supports Mixed, but we also require
        # positive AND negative evidence somewhere else.
        #
        # This prevents:
        #
        # "I expected it to be terrible,
        #  but it is actually excellent."
        #
        # from becoming Mixed only because of contrast.
        if (
            contrast_evidence[
                "supports_mixed"
            ]
            and has_positive
            and has_negative
        ):

            mixed = True

            reasons.append(
                "contrast_with_opposite_evidence"
            )

        # ----------------------------------------------------
        # Mixed confidence
        # ----------------------------------------------------

        if mixed:

            # Mixed should be strongest when BOTH sides
            # have strong evidence.
            balance_strength = (
                strongest_positive
                * strongest_negative
            ) ** 0.5

            evidence_bonus = min(
                0.10,
                0.02
                * len(reasons),
            )

            mixed_confidence = min(
                1.0,
                balance_strength
                + evidence_bonus,
            )

        else:

            mixed_confidence = 0.0

        # ----------------------------------------------------
        # Return explainable result
        # ----------------------------------------------------

        return {
            "mixed": mixed,

            "confidence": round(
                mixed_confidence,
                4,
            ),

            "reasons": reasons,

            "evidence": {
                "positive_aspects":
                    positive_aspects,

                "negative_aspects":
                    negative_aspects,

                "positive_sentences":
                    positive_sentences,

                "negative_sentences":
                    negative_sentences,

                "contrast":
                    contrast_evidence,
            },
        }