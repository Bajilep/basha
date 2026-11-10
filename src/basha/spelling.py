"""Fast and safe spelling correction for sentiment words."""

import re


COMMON_TYPOS = {
    # Positive words
    "graet": "great",
"ooww": "wow",
    "gret": "great",
    "grate": "great",
    "goood": "good",
    "gud": "good",
    "excelent": "excellent",
    "exellent": "excellent",
    "amazng": "amazing",
    "amzing": "amazing",
    "awsome": "awesome",
    "fantstic": "fantastic",
    "wonderfull": "wonderful",
    "briliant": "brilliant",
    "perfct": "perfect",
    "beautifull": "beautiful",
    "recomend": "recommend",
    "recomended": "recommended",
    "hapy": "happy",
    "happpy": "happy",
    "lovly": "lovely",

    # Negative words
    "terible": "terrible",
    "terrble": "terrible",
    "horrble": "horrible",
    "horible": "horrible",
    "dissapointed": "disappointed",
    "disapointed": "disappointed",
    "useess": "useless",
    "useles": "useless",
    "wrost": "worst",
    "baad": "bad",
    "patetic": "pathetic",
    "garbadge": "garbage",
    "unaceptable": "unacceptable",
}


def build_spelling_index(lexicon):
    """
    Group lexicon words according to their length.

    This improves runtime because spelling correction
    does not need to search the complete lexicon.
    """

    length_index = {}

    for word in lexicon:
        word_length = len(word)

        length_index.setdefault(
            word_length,
            []
        ).append(word)

    return {
        length: tuple(words)
        for length, words in length_index.items()
    }


def normalize_repeated_letters(word):
    """
    Reduce letters repeated three or more times.

    Examples:
        goooood -> good
        baaad -> baad
        amazinggg -> amazingg
    """

    return re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        word
    )


def _one_edit_away(first, second):
    """
    Check whether two words differ by one small mistake.

    Supported mistakes:
    - One incorrect letter
    - One missing letter
    - One extra letter
    - Two neighbouring letters exchanged
    """

    if first == second:
        return True

    if abs(len(first) - len(second)) > 1:
        return False

    # One wrong letter or two neighbouring
    # letters exchanged.
    if len(first) == len(second):
        differences = [
            index
            for index in range(len(first))
            if first[index] != second[index]
        ]

        if len(differences) == 1:
            return True

        if len(differences) == 2:
            first_index = differences[0]
            second_index = differences[1]

            return (
                second_index == first_index + 1
                and first[first_index]
                == second[second_index]
                and first[second_index]
                == second[first_index]
            )

        return False

    # One missing or extra letter.
    if len(first) < len(second):
        shorter = first
        longer = second
    else:
        shorter = second
        longer = first

    short_index = 0
    long_index = 0
    mistakes = 0

    while (
        short_index < len(shorter)
        and long_index < len(longer)
    ):
        if (
            shorter[short_index]
            == longer[long_index]
        ):
            short_index += 1
            long_index += 1

        else:
            mistakes += 1
            long_index += 1

            if mistakes > 1:
                return False

    return True


def find_closest_sentiment_word(
    word,
    lexicon,
    length_index=None,
    correction_cache=None,
):
    """
    Find one clear sentiment word close to a typo.

    Returns:
        Corrected word when one safe match is found.
        None when no safe correction is available.
    """

    clean_word = word.lower().strip()

    # Return an earlier result immediately.
    if (
        correction_cache is not None
        and clean_word in correction_cache
    ):
        return correction_cache[clean_word]

    # Check manually known spelling mistakes first.
    corrected_word = COMMON_TYPOS.get(clean_word)

    if (
        corrected_word is not None
        and corrected_word in lexicon
    ):
        if correction_cache is not None:
            correction_cache[clean_word] = (
                corrected_word
            )

        return corrected_word

    # Correct long repeated letters.
    normalized_word = normalize_repeated_letters(
        clean_word
    )

    if (
        normalized_word != clean_word
        and normalized_word in lexicon
    ):
        if correction_cache is not None:
            correction_cache[clean_word] = (
                normalized_word
            )

        return normalized_word

    # Short unknown words are risky to correct.
    if len(clean_word) < 6:
        if correction_cache is not None:
            correction_cache[clean_word] = None

        return None

    if length_index is None:
        length_index = build_spelling_index(
            lexicon
        )

    possible_words = []
    word_length = len(clean_word)

    # Only check words with a similar length.
    candidate_lengths = (
        word_length - 1,
        word_length,
        word_length + 1,
    )

    for candidate_length in candidate_lengths:
        candidates = length_index.get(
            candidate_length,
            ()
        )

        for sentiment_word in candidates:
            if _one_edit_away(
                clean_word,
                sentiment_word
            ):
                possible_words.append(
                    sentiment_word
                )

                # More than one result is ambiguous.
                if len(possible_words) > 1:
                    if correction_cache is not None:
                        correction_cache[
                            clean_word
                        ] = None

                    return None

    if len(possible_words) == 1:
        result = possible_words[0]

        if correction_cache is not None:
            correction_cache[clean_word] = result

        return result

    if correction_cache is not None:
        correction_cache[clean_word] = None

    return None