from __future__ import annotations

import re
from typing import List


# Common abbreviations that should not normally
# create a sentence boundary.
ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
}


def _normalize_whitespace(
    text: str,
) -> str:
    """
    Replace repeated whitespace with a single space.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _is_decimal_point(
    text: str,
    index: int,
) -> bool:
    """
    Detect decimal numbers such as:

        6.5
        3.14

    The decimal point must not split the sentence.
    """

    if text[index] != ".":
        return False

    if index == 0:
        return False

    if index + 1 >= len(text):
        return False

    return (
        text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _previous_word(
    text: str,
    index: int,
) -> str:
    """
    Return the word ending at the supplied position.
    """

    start = index

    while (
        start > 0
        and not text[start - 1].isspace()
    ):
        start -= 1

    return (
        text[start:index + 1]
        .strip()
        .lower()
    )


def _is_abbreviation(
    text: str,
    index: int,
) -> bool:
    """
    Check whether a period belongs to a common
    abbreviation.
    """

    if text[index] != ".":
        return False

    word = _previous_word(
        text,
        index,
    )

    return (
        word in ABBREVIATIONS
    )


def split_sentences(
    text: str,
) -> List[str]:
    """
    Split English text into sentences.

    Handles:
        - periods
        - question marks
        - exclamation marks
        - repeated punctuation
        - decimals
        - common abbreviations

    Example:

        The camera is great.
        The battery is terrible.

    becomes:

        [
            "The camera is great.",
            "The battery is terrible."
        ]
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string"
        )

    text = _normalize_whitespace(
        text
    )

    if not text:
        return []

    sentences: List[str] = []

    start = 0
    index = 0

    text_length = len(
        text
    )

    while index < text_length:

        character = text[index]

        # ----------------------------------------------------
        # Not sentence punctuation
        # ----------------------------------------------------

        if character not in ".!?":

            index += 1
            continue

        # ----------------------------------------------------
        # Decimal:
        #
        # 6.5
        #
        # Do not split here.
        # ----------------------------------------------------

        if (
            character == "."
            and _is_decimal_point(
                text,
                index,
            )
        ):

            index += 1
            continue

        # ----------------------------------------------------
        # Abbreviation:
        #
        # Dr.
        # Mr.
        # e.g.
        #
        # Do not split here.
        # ----------------------------------------------------

        if (
            character == "."
            and _is_abbreviation(
                text,
                index,
            )
        ):

            index += 1
            continue

        # ----------------------------------------------------
        # Repeated punctuation:
        #
        # !!!
        # ???
        # !?
        #
        # Move to the final punctuation character.
        # ----------------------------------------------------

        end = index

        while (
            end + 1 < text_length
            and text[end + 1]
            in ".!?"
        ):

            end += 1

        # ----------------------------------------------------
        # Sentence boundary only when punctuation is followed
        # by whitespace OR end of text.
        # ----------------------------------------------------

        next_index = end + 1

        is_boundary = (
            next_index
            >= text_length
            or text[
                next_index
            ].isspace()
        )

        if not is_boundary:

            index = end + 1
            continue

        sentence = (
            text[
                start:end + 1
            ]
            .strip()
        )

        if sentence:

            sentences.append(
                sentence
            )

        # Skip whitespace before next sentence.
        start = end + 1

        while (
            start < text_length
            and text[start].isspace()
        ):

            start += 1

        index = start

    # --------------------------------------------------------
    # Remaining text without final punctuation
    # --------------------------------------------------------

    if start < text_length:

        remaining = (
            text[start:]
            .strip()
        )

        if remaining:

            sentences.append(
                remaining
            )

    return sentences