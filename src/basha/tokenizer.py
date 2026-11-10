"""Fast functions for separating text into words."""

import re


WORD_PATTERN = re.compile(
    r"[A-Za-z]+"
    r"(?:'[A-Za-z]+)*"
    r"(?:-[A-Za-z]+)*"
)


# This pattern finds words and sentence boundaries.
WORD_OR_BOUNDARY_PATTERN = re.compile(
    r"[A-Za-z]+"
    r"(?:'[A-Za-z]+)*"
    r"(?:-[A-Za-z]+)*"
    r"|[.!?;:\n]+"
)


APOSTROPHE_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "´": "'",
    "`": "'",
})


SENTENCE_BOUNDARIES = {
    ".",
    "!",
    "?",
    ";",
    ":",
    "\n",
}


def normalize_text(text):
    """Normalise different apostrophe characters."""

    return text.translate(
        APOSTROPHE_TRANSLATION
    )


def tokenize_with_boundaries(text):
    """
    Return words and their sentence numbers.

    Example:
        "Not good. Excellent."

    Returns approximately:
        original_words = ["Not", "good", "Excellent"]
        lowercase_words = ["not", "good", "excellent"]
        sentence_ids = [0, 0, 1]
    """

    normalized_text = normalize_text(text)

    original_words = []
    lowercase_words = []
    sentence_ids = []

    current_sentence = 0

    for match in WORD_OR_BOUNDARY_PATTERN.finditer(
        normalized_text
    ):
        token = match.group()

        if WORD_PATTERN.fullmatch(token):
            original_words.append(token)
            lowercase_words.append(
                token.lower()
            )
            sentence_ids.append(
                current_sentence
            )

        else:
            current_sentence += 1

    return (
        original_words,
        lowercase_words,
        sentence_ids,
    )


def tokenize(text):
    """
    Return original and lowercase words.

    This function is kept for backward compatibility.
    """

    original_words, lowercase_words, _ = (
        tokenize_with_boundaries(text)
    )

    return original_words, lowercase_words