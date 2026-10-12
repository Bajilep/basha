from __future__ import annotations

import re
from typing import Dict, List, Optional

from .nli_conflict import NLIConflictDetector


class SemanticConflictDetector:
    """
    Detect semantic incongruity between an event and
    an evaluative expression.

    Designed mainly as supporting evidence for sarcasm.

    Example:

        "Perfect, my laptop just crashed before I saved my work."

    becomes approximately:

        premise:
            "My laptop just crashed before I saved my work."

        hypothesis:
            "This was perfect."

    NLI can then test whether these meanings conflict.
    """

    CONTRADICTION_THRESHOLD = 0.85

    # Small structural vocabulary.
    #
    # This is NOT a negative-event dictionary.
    # These are evaluation words that can be separated
    # from the event being evaluated.
    EVALUATION_WORDS = {
        "amazing",
        "awesome",
        "beautiful",
        "brilliant",
        "excellent",
        "fantastic",
        "good",
        "great",
        "helpful",
        "impressive",
        "lovely",
        "nice",
        "perfect",
        "wonderful",
    }

    def __init__(
        self,
        nli_model: Optional[NLIConflictDetector] = None,
    ) -> None:

        self.nli = (
            nli_model
            if nli_model is not None
            else NLIConflictDetector()
        )

    @staticmethod
    def _clean(text: str) -> str:

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def _find_evaluation(
        self,
        text: str,
    ) -> Optional[str]:

        words = re.findall(
            r"[A-Za-z']+",
            text.lower(),
        )

        for word in words:

            if word in self.EVALUATION_WORDS:

                return word

        return None

    def _remove_evaluation(
        self,
        text: str,
        evaluation: str,
    ) -> str:

        pattern = (
            r"\b"
            + re.escape(evaluation)
            + r"\b"
        )

        event = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        # Clean punctuation left by removal.

        event = re.sub(
            r"^[\s,;:!\-]+",
            "",
            event,
        )

        event = re.sub(
            r"\s+",
            " ",
            event,
        )

        return event.strip()

    @staticmethod
    def _build_hypothesis(
        evaluation: str,
    ) -> str:

        return (
            f"This was {evaluation}."
        )

    def analyze(
        self,
        text: str,
    ) -> Dict:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = self._clean(
            text
        )

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        evaluation = (
            self._find_evaluation(
                text
            )
        )

        if evaluation is None:

            return {
                "conflict": False,
                "confidence": 0.0,
                "evaluation": None,
                "premise": None,
                "hypothesis": None,
                "nli": None,
                "reason":
                    "no_evaluation_expression",
            }

        event = (
            self._remove_evaluation(
                text,
                evaluation,
            )
        )

        if not event:

            return {
                "conflict": False,
                "confidence": 0.0,
                "evaluation": evaluation,
                "premise": None,
                "hypothesis": None,
                "nli": None,
                "reason":
                    "no_event_content",
            }

        hypothesis = (
            self._build_hypothesis(
                evaluation
            )
        )

        nli_result = (
            self.nli.predict(
                premise=event,
                hypothesis=hypothesis,
            )
        )

        contradiction = float(
            nli_result.get(
                "contradiction",
                0.0,
            )
        )

        conflict = (
            contradiction
            >= self.CONTRADICTION_THRESHOLD
        )

        return {
            "conflict":
                conflict,

            "confidence":
                round(
                    contradiction,
                    4,
                ),

            "evaluation":
                evaluation,

            "premise":
                event,

            "hypothesis":
                hypothesis,

            "nli":
                nli_result,

            "reason": (
                "nli_evaluative_contradiction"
                if conflict
                else
                "no_strong_contradiction"
            ),
        }