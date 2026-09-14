"""Sentiment scores for complete phrase variations."""


PHRASE_SCORES = {
    # Positive phrases
    "highly recommended": 4.0,
    "must buy": 3.5,
    "worth the money": 3.5,
    "value for money": 3.0,
    "works perfectly": 3.5,
    "very helpful": 3.0,
    "easy to use": 3.0,
    "well made": 2.5,
    "good quality": 3.0,
    "fast delivery": 2.5,
    "customer friendly": 2.5,
    "pleasant experience": 3.0,
    "not bad": 2.5,
    "pretty good": 2.5,
    "above expectations": 3.0,
    "exceeded expectations": 3.5,
    "works as expected": 2.5,
    "would recommend": 3.0,
    "love this product": 3.5,
    "best purchase": 3.5,
    "five stars": 3.5,
    "great customer service": 3.5,
    "what a product": 3.0,

    # Negative phrases
    "waste of money": -4.0,
    "not worth it": -4.0,
    "piece of junk": -4.0,
    "poor quality": -3.5,
    "never again": -4.0,
    "does not work": -4.0,
    "did not work": -4.0,
    "not working": -3.5,
    "stopped working": -3.5,
    "bad experience": -3.5,
    "customer service was terrible": -4.0,
    "very disappointed": -3.5,
    "too expensive": -2.5,
    "late delivery": -2.5,
    "would not recommend": -4.0,
    "do not recommend": -4.0,
    "not good": -3.0,
    "far from good": -3.5,
    "could be better": -1.5,
    "complete disappointment": -4.0,
    "worst purchase": -4.0,
    "one star": -3.5,
}


PHRASE_VARIATIONS = {
    "highly recommended": {
        "strongly recommended",
        "definitely recommended",
    },

    "must buy": {
        "must have",
        "definitely buy",
    },

    "worth the money": {
        "worth my money",
        "worth every penny",
        "well worth the money",
    },

    "waste of money": {
        "complete waste of money",
        "total waste of money",
        "absolute waste of money",
        "waste of my money",
        "waste of your money",
        "waste of our money",
    },

    "not worth it": {
        "really not worth it",
        "definitely not worth it",
        "not worth buying",
    },

    "piece of junk": {
        "complete piece of junk",
        "total piece of junk",
        "absolute piece of junk",
    },

    "does not work": {
        "doesn't work",
        "doesnt work",
        "does not even work",
    },

    "did not work": {
        "didn't work",
        "didnt work",
        "never worked",
    },

    "stopped working": {
        "suddenly stopped working",
        "completely stopped working",
    },

    "would recommend": {
        "definitely recommend",
        "strongly recommend",
    },

    "would not recommend": {
        "wouldn't recommend",
        "wouldnt recommend",
        "definitely would not recommend",
    },

    "what a product": {
        "wow what a product",
        "ooww what a product",
    },

    "poor quality": {
        "very poor quality",
        "extremely poor quality",
    },

    "great customer service": {
        "excellent customer service",
        "amazing customer service",
    },
}


def _build_phrase_patterns():
    """Build phrase patterns only once."""

    patterns = []

    for canonical_phrase, score in (
        PHRASE_SCORES.items()
    ):
        variations = {
            canonical_phrase
        }

        variations.update(
            PHRASE_VARIATIONS.get(
                canonical_phrase,
                set(),
            )
        )

        for variation in variations:
            variation_words = tuple(
                variation.split()
            )

            patterns.append({
                "canonical": canonical_phrase,
                "matched_text": variation,
                "words": variation_words,
                "score": score,
            })

    patterns.sort(
        key=lambda pattern: len(
            pattern["words"]
        ),
        reverse=True,
    )

    return tuple(patterns)


PHRASE_PATTERNS = _build_phrase_patterns()


def find_phrase_matches(words):
    """Find non-overlapping sentiment phrases."""

    matches = []
    occupied_positions = set()

    for pattern in PHRASE_PATTERNS:
        phrase_words = pattern["words"]
        phrase_length = len(phrase_words)

        if phrase_length > len(words):
            continue

        last_start = (
            len(words) - phrase_length + 1
        )

        for start in range(last_start):
            end = start + phrase_length

            sentence_part = tuple(
                words[start:end]
            )

            if sentence_part != phrase_words:
                continue

            positions = set(
                range(start, end)
            )

            if positions & occupied_positions:
                continue

            matches.append({
                "phrase": pattern["canonical"],
                "matched_text": (
                    pattern["matched_text"]
                ),
                "score": pattern["score"],
                "start": start,
                "end": end,
                "positions": positions,
            })

            occupied_positions.update(
                positions
            )

    matches.sort(
        key=lambda match: match["start"]
    )

    return matches