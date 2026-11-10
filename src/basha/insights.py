"""Additional, non-duplicating analysis metadata.

These helpers describe intensity, emoji combinations, vocabulary coverage,
and mixed polarity. They do not score sentiment a second time; the main
analyzer remains the single source of the final sentiment score.
"""

import re

from .symbols import SYMBOL_SCORES


EMOJI_COMBINATIONS = {
    "😂😭": {"meaning": "laughing and crying", "emotion": "mixed"},
    "😂💀": {"meaning": "extreme laughter", "emotion": "joy"},
    "🤣💀": {"meaning": "extreme laughter", "emotion": "joy"},
    "😍🔥": {"meaning": "strong admiration", "emotion": "joy"},
    "🥰❤️": {"meaning": "love and affection", "emotion": "joy"},
    "😡🔥": {"meaning": "intense anger", "emotion": "anger"},
    "😭💔": {"meaning": "heartbreak", "emotion": "sadness"},
    "❤️🔥": {"meaning": "passion", "emotion": "joy"},
    "😒🙄": {"meaning": "annoyance or sarcasm", "emotion": "disgust"},
    "😱😨": {"meaning": "intense fear", "emotion": "fear"},
    "🙏❤️": {"meaning": "gratitude", "emotion": "joy"},
    "👏🔥": {"meaning": "strong approval", "emotion": "joy"},
    "💔😭": {"meaning": "heartbreak", "emotion": "sadness"},
}


def analyze_intensity(text, words, intensifiers, diminishers):
    """Return bounded intensity metadata for a piece of text."""

    exclamations = text.count("!")
    questions = text.count("?")

    word_multiplier = 1.0
    for word in words:
        if word in intensifiers:
            word_multiplier *= intensifiers[word]
        elif word in diminishers:
            word_multiplier *= diminishers[word]

    word_multiplier = max(0.5, min(word_multiplier, 2.0))
    punctuation_multiplier = min(1.0 + 0.10 * exclamations, 1.5)
    question_multiplier = min(1.0 + 0.05 * questions, 1.3)

    letters = [character for character in text if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters else 0.0
    )
    caps_multiplier = 1.25 if uppercase_ratio >= 0.70 else 1.0

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    maximum_repetition = max(counts.values(), default=1)
    repetition_multiplier = min(
        1.0 + 0.10 * max(0, maximum_repetition - 1),
        1.4,
    )

    final_multiplier = min(
        word_multiplier
        * punctuation_multiplier
        * question_multiplier
        * caps_multiplier
        * repetition_multiplier,
        2.0,
    )

    return {
        "score": round(min(final_multiplier / 2.0, 1.0), 4),
        "multiplier": round(final_multiplier, 4),
        "factors": {
            "words": round(word_multiplier, 4),
            "exclamations": exclamations,
            "questions": questions,
            "capitalization": round(caps_multiplier, 4),
            "repetition": round(repetition_multiplier, 4),
        },
    }


def analyze_emoji_patterns(text):
    """Describe known emojis and meaningful adjacent combinations."""

    detected = []
    for symbol in sorted(SYMBOL_SCORES, key=len, reverse=True):
        count = text.count(symbol)
        if count:
            detected.append({
                "symbol": symbol,
                "count": count,
                "score": SYMBOL_SCORES[symbol],
            })

    combinations = []
    for sequence, information in EMOJI_COMBINATIONS.items():
        if sequence in text:
            combinations.append({"sequence": sequence, **information})

    return {
        "items": detected,
        "combinations": combinations,
        "count": sum(item["count"] for item in detected),
    }


def vocabulary_coverage(words, lexicon, ignored_words):
    """Report which meaningful alphabetic words are known or unknown."""

    candidates = [
        word for word in words
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word)
        and word not in ignored_words
    ]
    known = [word for word in candidates if word in lexicon]
    unknown = [word for word in candidates if word not in lexicon]
    total = len(candidates)

    return {
        "known_words": known,
        "unknown_words": unknown,
        "ratio": round(len(known) / total, 3) if total else 0.0,
    }


def has_mixed_polarity(explanation):
    """Return True when both positive and negative evidence are present."""

    scores = [item.get("final_score", 0.0) for item in explanation]
    return any(score > 0 for score in scores) and any(
        score < 0 for score in scores
    )
