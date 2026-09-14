"""Rules for detecting possible sarcasm."""


# These expressions are strong sarcasm indicators.
STRONG_SARCASM_MARKERS = {
    "yeah right",
    "as if",
    "/s",
}


# These expressions may be sarcastic, but they can
# also be used normally. They require negative context.
CONTEXTUAL_SARCASM_MARKERS = {
    "what a surprise",
    "thanks a lot",
    "just great",
    "great job",
    "nice going",
    "how wonderful",
    "well done",
    "perfect timing",
    "exactly what i needed",
}


NEGATIVE_EVENTS = {
    "crash",
    "crashed",
    "crashing",
    "fail",
    "failed",
    "failing",
    "failure",
    "broken",
    "broke",
    "error",
    "errors",
    "late",
    "delay",
    "delayed",
    "lost",
    "lose",
    "problem",
    "problems",
    "worst",
    "terrible",
    "awful",
    "useless",
    "stopped",
    "stuck",
    "cancelled",
    "canceled",
    "damaged",
    "dead",
    "missing",
    "forgot",
    "forgotten",
    "ruined",
    "destroyed",
    "freezing",
    "frozen",
    "unavailable",
    "rejected",
}


SARCASM_EMOJIS = {
    "🙄",
    "😒",
    "🤦",
    "🤦‍♂️",
    "🤦‍♀️",
}


def _find_markers(text, markers):
    """Return sarcasm markers found in text."""

    return [
        marker
        for marker in markers
        if marker in text
    ]


def _find_sarcasm_emojis(text):
    """Return sarcasm emojis without duplicate matching."""

    found_emojis = []
    remaining_text = text

    sorted_emojis = sorted(
        SARCASM_EMOJIS,
        key=len,
        reverse=True,
    )

    for emoji in sorted_emojis:
        if emoji not in remaining_text:
            continue

        found_emojis.append(emoji)

        remaining_text = remaining_text.replace(
            emoji,
            "",
        )

    return found_emojis


def detect_sarcasm(text, words, lexicon):
    """
    Detect common sarcasm patterns.

    This is a rule-based detector. It reports
    possible sarcasm rather than guaranteed sarcasm.

    Returns:
        detected:
            True when sarcasm is likely.

        confidence:
            Score from 0.0 to 1.0.

        reasons:
            Rules that caused the detection.
    """

    lowercase_text = text.lower()
    word_set = set(words)

    reasons = []
    confidence_scores = []

    positive_words = [
        word
        for word in words
        if lexicon.get(word, 0) > 0
    ]

    negative_words = [
        word
        for word in words
        if lexicon.get(word, 0) < 0
    ]

    negative_events = (
        word_set & NEGATIVE_EVENTS
    )

    strong_markers = _find_markers(
        lowercase_text,
        STRONG_SARCASM_MARKERS,
    )

    contextual_markers = _find_markers(
        lowercase_text,
        CONTEXTUAL_SARCASM_MARKERS,
    )

    sarcasm_emojis = _find_sarcasm_emojis(
        text
    )

    # Strong expressions such as "yeah right".
    for marker in strong_markers:
        reasons.append(
            f"strong sarcasm marker: {marker}"
        )

        confidence_scores.append(0.85)

    # Emojis such as eye-roll and facepalm.
    for emoji in sarcasm_emojis:
        reasons.append(
            f"sarcasm emoji: {emoji}"
        )

        confidence_scores.append(0.75)

    has_positive_word = bool(positive_words)
    has_negative_word = bool(negative_words)
    has_negative_event = bool(negative_events)

    # Example:
    # "Great, the application crashed again."
    if (
        has_positive_word
        and has_negative_event
        and "again" in word_set
    ):
        reasons.append(
            "positive expression with "
            "repeated negative event"
        )

        confidence_scores.append(0.90)

    # Example:
    # "Wonderful, another error."
    if (
        has_positive_word
        and has_negative_event
        and (
            "another" in word_set
            or "exactly" in word_set
        )
    ):
        reasons.append(
            "positive expression with "
            "negative event"
        )

        confidence_scores.append(0.80)

    # Contextual phrases are considered sarcastic
    # only when negative evidence is present.
    if (
        contextual_markers
        and (
            has_negative_event
            or has_negative_word
            or sarcasm_emojis
        )
    ):
        for marker in contextual_markers:
            reasons.append(
                f"contextual sarcasm marker: "
                f"{marker}"
            )

        confidence_scores.append(0.75)


    if confidence_scores:
        confidence = max(confidence_scores)

        # Multiple different indicators increase
        # confidence slightly.
        if len(confidence_scores) > 1:
            confidence += min(
                0.05 * (len(confidence_scores) - 1),
                0.15,
            )

        confidence = min(confidence, 1.0)

    else:
        confidence = 0.0

    detected = confidence >= 0.60

    return {
        "detected": detected,
        "confidence": round(confidence, 2),
        "reasons": reasons,
    }