from __future__ import annotations

from typing import Dict


class DiscourseResolver:
    """
    Resolve cases where contrast clauses do not represent
    two simultaneous opinions.

    Example:

        I expected it to be terrible,
        but it is actually excellent.

    The first clause describes a prior expectation.
    The second clause describes the actual/current result.

    Therefore the final sentiment should normally follow
    the actual-result clause rather than become Mixed.
    """

    EXPECTATION_MARKERS = {
        "expected",
        "expecting",
        "thought",
        "assumed",
        "anticipated",
        "feared",
    }

    ACTUALITY_MARKERS = {
        "actually",
        "really",
        "in fact",
        "turned out",
        "turns out",
        "ended up",
        "in reality",
    }

    def _contains_marker(
        self,
        text: str,
        markers: set[str],
    ) -> bool:

        text = text.lower()

        return any(
            marker in text
            for marker in markers
        )

    def resolve(
        self,
        contrast: Dict | None,
    ) -> Dict:

        default = {
            "resolved": False,
            "label": None,
            "confidence": 0.0,
            "reason": None,
        }

        if not contrast:
            return default

        if not contrast.get(
            "contrast_detected",
            False,
        ):
            return default

        clauses = contrast.get(
            "clauses",
            [],
        )

        if len(clauses) != 2:
            return default

        first = clauses[0]
        second = clauses[1]

        first_text = str(
            first.get(
                "text",
                "",
            )
        )

        second_text = str(
            second.get(
                "text",
                "",
            )
        )

        first_is_expectation = (
            self._contains_marker(
                first_text,
                self.EXPECTATION_MARKERS,
            )
        )

        second_is_actual = (
            self._contains_marker(
                second_text,
                self.ACTUALITY_MARKERS,
            )
        )

        # ----------------------------------------------------
        # Expectation → actual result
        # ----------------------------------------------------

        if (
            first_is_expectation
            and second_is_actual
        ):

            label = (
                str(
                    second.get(
                        "label",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            confidence = float(
                second.get(
                    "confidence",
                    0.0,
                )
            )

            if label in {
                "positive",
                "negative",
                "neutral",
            }:

                return {
                    "resolved": True,
                    "label": label,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "reason":
                        "expectation_vs_actual",
                }

        return default