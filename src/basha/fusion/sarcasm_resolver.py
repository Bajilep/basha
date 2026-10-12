from __future__ import annotations

from typing import Dict, List, Optional


class SarcasmResolver:
    """
    Resolve the intended sentiment of sarcastic text.

    IMPORTANT:

    Sarcasm does NOT automatically mean negative.

    We require:

        strong sarcasm

                +

        supporting negative evidence

    Supporting evidence can come from:

        1. negative aspect sentiment
        2. negative sentence sentiment
        3. strong core negative probability
        4. semantic contradiction detected by NLI
    """

    # ======================================================
    # THRESHOLDS
    # ======================================================

    MIN_SARCASM = 0.80

    ASPECT_THRESHOLD = 0.50

    SENTENCE_THRESHOLD = 0.65

    CORE_NEGATIVE_THRESHOLD = 0.60

    SEMANTIC_CONFLICT_THRESHOLD = 0.85

    # ======================================================
    # EMPTY RESULT
    # ======================================================

    @staticmethod
    def _empty_result() -> Dict:

        return {
            "resolved": False,
            "label": None,
            "confidence": 0.0,
            "reason": None,

            "evidence": {

                "sarcasm_probability":
                    0.0,

                "negative_aspects":
                    [],

                "negative_sentences":
                    [],

                "core_negative":
                    0.0,

                "semantic_conflict":
                    0.0,
            },
        }

    # ======================================================
    # RESOLVE
    # ======================================================

    def resolve(
        self,
        sarcasm: Dict,
        overall: Dict,
        aspects: List[Dict],
        sentences: List[Dict],
        mixed: Dict,
        discourse: Dict,
        semantic_conflict: Optional[Dict] = None,
    ) -> Dict:

        result = (
            self._empty_result()
        )

        # ==================================================
        # 1. DISCOURSE HAS PRIORITY
        # ==================================================

        if discourse.get(
            "resolved",
            False,
        ):

            return result

        # ==================================================
        # 2. SARCASM MUST BE DETECTED
        # ==================================================

        if not sarcasm.get(
            "detected",
            False,
        ):

            return result

        sarcasm_probability = float(
            sarcasm.get(
                "sarcasm",
                0.0,
            )
        )

        result["evidence"][
            "sarcasm_probability"
        ] = sarcasm_probability

        # Extra safety.
        #
        # Even if a future sarcasm wrapper changes how
        # "detected" is produced, this resolver still
        # enforces Basha's V1 threshold.

        if (
            sarcasm_probability
            < self.MIN_SARCASM
        ):

            return result

        # ==================================================
        # 3. NEGATIVE ASPECT EVIDENCE
        # ==================================================

        negative_aspects = []

        for aspect in aspects:

            sentiment = (
                aspect.get(
                    "sentiment"
                )
            )

            confidence = float(
                aspect.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                sentiment == "negative"
                and
                confidence
                >= self.ASPECT_THRESHOLD
            ):

                negative_aspects.append(
                    {
                        "aspect":
                            aspect.get(
                                "aspect"
                            ),

                        "confidence":
                            confidence,
                    }
                )

        result["evidence"][
            "negative_aspects"
        ] = negative_aspects

        # ==================================================
        # 4. NEGATIVE SENTENCE EVIDENCE
        # ==================================================

        negative_sentences = []

        for sentence in sentences:

            label = (
                sentence.get(
                    "label"
                )
            )

            confidence = float(
                sentence.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                label == "negative"
                and
                confidence
                >= self.SENTENCE_THRESHOLD
            ):

                negative_sentences.append(
                    {
                        "text":
                            sentence.get(
                                "text"
                            ),

                        "confidence":
                            confidence,
                    }
                )

        result["evidence"][
            "negative_sentences"
        ] = negative_sentences

        # ==================================================
        # 5. CORE NEGATIVE EVIDENCE
        # ==================================================

        core_negative = float(
            overall.get(
                "negative",
                0.0,
            )
        )

        result["evidence"][
            "core_negative"
        ] = core_negative

        # ==================================================
        # 6. SEMANTIC CONFLICT
        # ==================================================

        semantic_conflict_score = 0.0

        if semantic_conflict:

            conflict_detected = bool(
                semantic_conflict.get(
                    "conflict",
                    False,
                )
            )

            conflict_confidence = float(
                semantic_conflict.get(
                    "confidence",
                    0.0,
                )
            )

            if conflict_detected:

                semantic_conflict_score = (
                    conflict_confidence
                )

        result["evidence"][
            "semantic_conflict"
        ] = semantic_conflict_score

        # ==================================================
        # 7. COLLECT SUPPORTING EVIDENCE
        # ==================================================

        evidence_scores = []

        evidence_reasons = []

        # --------------------------------------------------
        # Aspect evidence
        # --------------------------------------------------

        if negative_aspects:

            strongest_aspect = max(
                item["confidence"]
                for item
                in negative_aspects
            )

            evidence_scores.append(
                strongest_aspect
            )

            evidence_reasons.append(
                "negative_aspect"
            )

        # --------------------------------------------------
        # Sentence evidence
        # --------------------------------------------------

        if negative_sentences:

            strongest_sentence = max(
                item["confidence"]
                for item
                in negative_sentences
            )

            evidence_scores.append(
                strongest_sentence
            )

            evidence_reasons.append(
                "negative_sentence"
            )

        # --------------------------------------------------
        # Core sentiment evidence
        # --------------------------------------------------

        if (
            core_negative
            >= self.CORE_NEGATIVE_THRESHOLD
        ):

            evidence_scores.append(
                core_negative
            )

            evidence_reasons.append(
                "core_negative"
            )

        # --------------------------------------------------
        # Semantic contradiction evidence
        # --------------------------------------------------

        if (
            semantic_conflict_score
            >=
            self.SEMANTIC_CONFLICT_THRESHOLD
        ):

            evidence_scores.append(
                semantic_conflict_score
            )

            evidence_reasons.append(
                "semantic_conflict"
            )

        # ==================================================
        # 8. NO SUPPORTING NEGATIVE EVIDENCE
        # ==================================================

        if not evidence_scores:

            return result

        # ==================================================
        # 9. STRONGEST EVIDENCE
        # ==================================================

        strongest_evidence = max(
            evidence_scores
        )

        # ==================================================
        # 10. FUSED CONFIDENCE
        #
        # Geometric mean:
        #
        # sqrt(
        #     sarcasm_probability
        #     *
        #     strongest_negative_evidence
        # )
        #
        # Both signals must therefore be strong.
        # ==================================================

        confidence = (
            sarcasm_probability
            * strongest_evidence
        ) ** 0.5

        confidence = min(
            confidence,
            1.0,
        )

        # ==================================================
        # 11. EXPLAINABLE REASON
        # ==================================================

        reason = (
            "sarcasm_with_"
            + "_and_".join(
                evidence_reasons
            )
        )

        # ==================================================
        # 12. FINAL RESOLUTION
        # ==================================================

        result.update(
            {
                "resolved":
                    True,

                "label":
                    "negative",

                "confidence":
                    round(
                        confidence,
                        4,
                    ),

                "reason":
                    reason,
            }
        )

        return result