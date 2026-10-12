from __future__ import annotations

from typing import Dict, List

from ..sentiment.english_model import (
    EnglishSentimentModel,
)

from ..aspects.aspect_model import (
    EnglishAspectModel,
)

from ..aspects.contrast import (
    ContrastAnalyzer,
)

from ..longtext.splitter import (
    split_sentences,
)

from ..sarcasm.english_sarcasm import (
    EnglishSarcasmModel,
)

from ..semantics.semantic_conflict import (
    SemanticConflictDetector,
)

from .mixed_detector import (
    MixedDetector,
)

from .discourse_resolver import (
    DiscourseResolver,
)

from .sarcasm_resolver import (
    SarcasmResolver,
)


class EnglishFusion:
    """
    Main English intelligence/fusion pipeline for Basha.

    Current decision priority:

        1. Discourse resolution
        2. Sarcasm resolution
        3. Mixed sentiment
        4. Core sentiment

    Models/components:

        Core sentiment:
            CardiffNLP RoBERTa

        Aspect sentiment:
            Gauneg RoBERTa ABSA
            (or a fine-tuned checkpoint via aspect_model_name)

        Sarcasm:
            BESSTIE DeBERTa-v3

        Semantic conflict:
            NLI DeBERTa-v3

        Additional reasoning:
            sentence analysis
            contrast analysis
            mixed detection
            discourse resolution
            sarcasm resolution

    Parameters
    ----------
    aspect_model_name:
        Optional Hugging Face model name or local folder
        path for the aspect model. If omitted, the
        original Gauneg model is used.
    """

    def __init__(
        self,
        aspect_model_name: str | None = None,
    ) -> None:

        print(
            "Initializing Basha English Fusion..."
        )

        # ==================================================
        # MODELS
        # ==================================================

        self.sentiment_model = (
            EnglishSentimentModel()
        )

        self.aspect_model = (
            EnglishAspectModel(
                aspect_model_name
            )
        )

        self.sarcasm_model = (
            EnglishSarcasmModel()
        )

        # --------------------------------------------------
        # Semantic contradiction model
        #
        # NOTE:
        # The model is loaded here once.
        #
        # Inference is NOT performed for every input.
        # We only call it when sarcasm is detected.
        # --------------------------------------------------

        self.semantic_conflict_detector = (
            SemanticConflictDetector()
        )

        # ==================================================
        # REASONING / FUSION COMPONENTS
        # ==================================================

        self.contrast_analyzer = (
            ContrastAnalyzer(
                self.sentiment_model
            )
        )

        self.mixed_detector = (
            MixedDetector()
        )

        self.discourse_resolver = (
            DiscourseResolver()
        )

        self.sarcasm_resolver = (
            SarcasmResolver()
        )

    # ======================================================
    # SENTENCE ANALYSIS
    # ======================================================

    def _analyze_sentences(
        self,
        text: str,
    ) -> List[Dict]:

        """
        Split text into sentences and run the core
        sentiment model independently on each sentence.
        """

        sentence_texts = (
            split_sentences(text)
        )

        results: List[Dict] = []

        for sentence in sentence_texts:

            sentiment = (
                self.sentiment_model.predict(
                    sentence
                )
            )

            results.append(
                {
                    "text":
                        sentence,

                    "label":
                        sentiment["label"],

                    "confidence":
                        sentiment["confidence"],

                    "positive":
                        sentiment["positive"],

                    "neutral":
                        sentiment["neutral"],

                    "negative":
                        sentiment["negative"],
                }
            )

        return results

    # ======================================================
    # EMPTY SEMANTIC-CONFLICT RESULT
    # ======================================================

    @staticmethod
    def _empty_semantic_conflict(
        reason: str,
    ) -> Dict:

        """
        Return a consistent semantic-conflict result when
        NLI inference does not need to run.
        """

        return {
            "conflict": False,
            "confidence": 0.0,
            "evaluation": None,
            "premise": None,
            "hypothesis": None,
            "nli": None,
            "reason": reason,
        }

    # ======================================================
    # FINAL DECISION
    # ======================================================

    @staticmethod
    def _final_decision(
        overall: Dict,
        mixed: Dict,
        discourse: Dict,
        sarcasm_resolution: Dict,
    ) -> Dict:

        """
        Select the final sentiment decision.

        Priority is important.

        Discourse must be checked before sarcasm because:

            "I expected it to be terrible,
             but it is actually excellent."

        should follow the actual outcome.

        Sarcasm must be checked before Mixed because:

            "Great, another software crash.
             Exactly what I needed."

        can create fake positive + negative sentence
        disagreement even though the intended sentiment
        is negative.
        """

        # ==================================================
        # PRIORITY 1 — DISCOURSE
        # ==================================================

        if discourse.get(
            "resolved",
            False,
        ):

            return {
                "label":
                    discourse["label"],

                "confidence":
                    discourse["confidence"],

                "source":
                    "discourse_resolution",
            }

        # ==================================================
        # PRIORITY 2 — SARCASM
        # ==================================================

        if sarcasm_resolution.get(
            "resolved",
            False,
        ):

            return {
                "label":
                    sarcasm_resolution[
                        "label"
                    ],

                "confidence":
                    sarcasm_resolution[
                        "confidence"
                    ],

                "source":
                    "sarcasm_resolution",
            }

        # ==================================================
        # PRIORITY 3 — MIXED
        # ==================================================

        if mixed.get(
            "mixed",
            False,
        ):

            return {
                "label":
                    "mixed",

                "confidence":
                    mixed["confidence"],

                "source":
                    "mixed_fusion",
            }

        # ==================================================
        # PRIORITY 4 — CORE MODEL
        # ==================================================

        return {
            "label":
                overall["label"],

            "confidence":
                overall["confidence"],

            "source":
                "core_sentiment",
        }

    # ======================================================
    # MAIN ANALYSIS
    # ======================================================

    def analyze(
        self,
        text: str,
    ) -> Dict:

        # ==================================================
        # INPUT VALIDATION
        # ==================================================

        if not isinstance(
            text,
            str,
        ):

            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "text cannot be empty"
            )

        # ==================================================
        # 1. CORE WHOLE-TEXT SENTIMENT
        # ==================================================

        overall = (
            self.sentiment_model.predict(
                text
            )
        )

        # ==================================================
        # 2. SENTENCE-LEVEL SENTIMENT
        # ==================================================

        sentences = (
            self._analyze_sentences(
                text
            )
        )

        # ==================================================
        # 3. ASPECT-BASED SENTIMENT
        # ==================================================

        aspects = (
            self.aspect_model.predict(
                text
            )
        )

        # ==================================================
        # 4. CONTRAST ANALYSIS
        # ==================================================

        contrast = (
            self.contrast_analyzer.analyze(
                text
            )
        )

        # ==================================================
        # 5. MIXED-SENTIMENT EVIDENCE
        # ==================================================

        mixed = (
            self.mixed_detector.detect(
                aspects=aspects,
                sentences=sentences,
                contrast=contrast,
            )
        )

        # ==================================================
        # 6. DISCOURSE RESOLUTION
        # ==================================================

        discourse = (
            self.discourse_resolver.resolve(
                contrast
            )
        )

        # ==================================================
        # 7. SARCASM DETECTION
        # ==================================================

        sarcasm = (
            self.sarcasm_model.predict(
                text
            )
        )

        # ==================================================
        # 8. SEMANTIC CONFLICT
        #
        # NLI inference is relatively expensive.
        #
        # Therefore:
        #
        #     normal text
        #         ↓
        #     sarcasm = False
        #         ↓
        #     SKIP NLI
        #
        #     sarcastic text
        #         ↓
        #     sarcasm = True
        #         ↓
        #     RUN NLI
        #
        # This avoids unnecessary NLI inference for
        # ordinary positive/negative/neutral reviews.
        # ==================================================

        if sarcasm.get(
            "detected",
            False,
        ):

            semantic_conflict = (
                self
                .semantic_conflict_detector
                .analyze(
                    text
                )
            )

        else:

            semantic_conflict = (
                self._empty_semantic_conflict(
                    reason=(
                        "sarcasm_not_detected"
                    )
                )
            )

        # ==================================================
        # 9. SARCASM RESOLUTION
        # ==================================================

        sarcasm_resolution = (
            self.sarcasm_resolver.resolve(
                sarcasm=sarcasm,
                overall=overall,
                aspects=aspects,
                sentences=sentences,
                mixed=mixed,
                discourse=discourse,
                semantic_conflict=(
                    semantic_conflict
                ),
            )
        )

        # ==================================================
        # 10. FINAL FUSION DECISION
        # ==================================================

        final = (
            self._final_decision(
                overall=overall,
                mixed=mixed,
                discourse=discourse,
                sarcasm_resolution=(
                    sarcasm_resolution
                ),
            )
        )

        # ==================================================
        # 11. FINAL RESULT
        # ==================================================

        return {

            # ------------------------------------------------
            # LANGUAGE
            # ------------------------------------------------

            "language":
                "en",

            # ------------------------------------------------
            # FINAL FUSED SENTIMENT
            # ------------------------------------------------

            "sentiment": {

                "label":
                    final["label"],

                "confidence":
                    final["confidence"],
            },

            # ------------------------------------------------
            # WHERE DID THE FINAL DECISION COME FROM?
            # ------------------------------------------------

            "decision_source":
                final["source"],

            # ------------------------------------------------
            # RAW CARDIFF SENTIMENT
            #
            # IMPORTANT:
            #
            # These probabilities belong ONLY to Cardiff.
            #
            # They are NOT fused probabilities.
            #
            # Keeping them separate prevents output such as:
            #
            # label = mixed
            # negative = 0.77
            #
            # from appearing as though those probabilities
            # describe the final fused decision.
            # ------------------------------------------------

            "core_sentiment":
                overall,

            # ------------------------------------------------
            # SARCASM
            # ------------------------------------------------

            "sarcasm":
                sarcasm,

            # ------------------------------------------------
            # SEMANTIC CONFLICT / NLI
            # ------------------------------------------------

            "semantic_conflict":
                semantic_conflict,

            # ------------------------------------------------
            # SARCASM RESOLUTION
            # ------------------------------------------------

            "sarcasm_resolution":
                sarcasm_resolution,

            # ------------------------------------------------
            # ASPECT SENTIMENT
            # ------------------------------------------------

            "aspects":
                aspects,

            # ------------------------------------------------
            # SENTENCE SENTIMENT
            # ------------------------------------------------

            "sentences":
                sentences,

            # ------------------------------------------------
            # CONTRAST
            # ------------------------------------------------

            "contrast":
                contrast,

            # ------------------------------------------------
            # MIXED ANALYSIS
            # ------------------------------------------------

            "mixed_analysis":
                mixed,

            # ------------------------------------------------
            # DISCOURSE
            # ------------------------------------------------

            "discourse":
                discourse,
        }