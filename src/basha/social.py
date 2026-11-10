"""Social-media text normalization.

This module:
- Removes URLs
- Removes usernames such as @company
- Converts hashtags into normal words
- Converts slang such as luv and gr8
- Reduces repeated letters
- Keeps a record of every change
"""

import re


# Detect complete URLs.
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)

# Detect social-media usernames.
MENTION_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]+"
)

# Detect hashtags.
HASHTAG_PATTERN = re.compile(
    r"#([A-Za-z][A-Za-z0-9_]*)"
)

# Detect words, including apostrophes.
WORD_PATTERN = re.compile(
    r"\b[A-Za-z0-9]+(?:'[A-Za-z]+)?\b"
)

# Split CamelCase hashtags.
CAMEL_CASE_PATTERN = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z]|\d|\b)
    |
    [A-Z]?[a-z]+
    |
    \d+
    """,
    re.VERBOSE,
)


# Informal words and abbreviations.
SLANG_MAP = {
    # Positive slang
    "luv": "love",
    "lovedd": "loved",
    "gr8": "great",
    "g8": "great",
    "gud": "good",
    "goood": "good",
    "awsm": "awesome",
    "awsome": "awesome",
    "fab": "fabulous",
    "perf": "perfect",
    "thx": "thanks",
    "thanx": "thanks",
    "ty": "thanks",
    "congrats": "congratulations",
    "slay": "amazing",
    "lit": "excellent",
    "goated": "best",
    "goat": "best",
    "bussin": "excellent",
    "based": "admirable",
    "ate": "excellent",
    "w": "winning",

    # Negative slang
    "h8": "hate",
    "sux": "bad",
    "sucks": "bad",
    "worsttt": "worst",
    "terribl": "terrible",
    "disapointed": "disappointed",

    # Common abbreviations
    "omg": "wow",
    "lol": "laugh",
    "lmao": "laugh",
    "rofl": "laugh",
    "yay": "happy",
    "idk": "uncertain",
    "imo": "opinion",
    "imho": "opinion",
    "btw": "by the way",
    "tbh": "honestly",
    "fyi": "information",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "bc": "because",
    "cuz": "because",
    "coz": "because",
    "msg": "message",
    "ppl": "people",
    "wth": "what",
    "wtf": "terrible",
    "smh": "disappointed",
    "mid": "mediocre",
    "sus": "suspicious",
    "cringe": "embarrassed",
    "cooked": "ruined",
    "l": "failure",
    "yikes": "worried",
    "oof": "painful",
    "lmfao": "laugh",
    "fr": "really",
    "frfr": "really",
    "ngl": "honestly",
    "ikr": "agree",
    "idc": "unconcerned",
    "rn": "now",
    "irl": "in real life",
    "asap": "soon",
}


def _split_hashtag(hashtag):
    """
    Convert a hashtag into normal words.

    Examples:
        AmazingProduct -> amazing product
        poor_service   -> poor service
        AMAZINGPRODUCT -> amazingproduct
    """

    hashtag = hashtag.replace("_", " ")

    parts = []

    for section in hashtag.split():
        camel_parts = CAMEL_CASE_PATTERN.findall(
            section
        )

        if camel_parts:
            parts.extend(camel_parts)
        else:
            parts.append(section)

    return " ".join(
        part.lower()
        for part in parts
        if part
    )


def _reduce_repeated_letters(word):
    """
    Reduce three or more repeated letters to two.

    Examples:
        gooood -> good
        amaaazing -> amaazing
        loveeee -> lovee

    Two letters are preserved because words such as
    'good', 'happy' and 'cool' naturally contain them.
    """

    return re.sub(
        r"([A-Za-z])\1{2,}",
        r"\1\1",
        word,
        flags=re.IGNORECASE,
    )


def _replace_word(match, changes):
    """Normalize one word."""

    original_word = match.group(0)
    lowercase_word = original_word.lower()

    # First check the exact slang dictionary.
    if lowercase_word in SLANG_MAP:
        replacement = SLANG_MAP[lowercase_word]

        changes.append({
            "type": "slang",
            "original": original_word,
            "normalized": replacement,
        })

        return replacement

    reduced_word = _reduce_repeated_letters(
        lowercase_word
    )

    # Check slang again after repeated letters
    # have been reduced.
    if reduced_word in SLANG_MAP:
        replacement = SLANG_MAP[reduced_word]

        changes.append({
            "type": "repeated_letters_and_slang",
            "original": original_word,
            "normalized": replacement,
        })

        return replacement

    if reduced_word != lowercase_word:
        changes.append({
            "type": "repeated_letters",
            "original": original_word,
            "normalized": reduced_word,
        })

        return reduced_word

    return original_word


def normalize_social_text(text):
    """
    Normalize text commonly found on social media.

    Returns:
        tuple:
            normalized_text: cleaned text
            changes: list describing every normalization
    """

    if not isinstance(text, str):
        raise TypeError(
            "Social text must be a string."
        )

    changes = []
    normalized_text = text

    # Step 1: Remove URLs completely.
    urls = URL_PATTERN.findall(normalized_text)

    if urls:
        normalized_text = URL_PATTERN.sub(
            " ",
            normalized_text,
        )

        for url in urls:
            changes.append({
                "type": "url_removed",
                "original": url,
                "normalized": "",
            })

    # Step 2: Remove usernames.
    mentions = MENTION_PATTERN.findall(
        normalized_text
    )

    if mentions:
        normalized_text = MENTION_PATTERN.sub(
            " ",
            normalized_text,
        )

        for mention in mentions:
            changes.append({
                "type": "mention_removed",
                "original": mention,
                "normalized": "",
            })

    # Step 3: Convert hashtags to words.
    def replace_hashtag(match):
        original_hashtag = match.group(0)

        replacement = _split_hashtag(
            match.group(1)
        )

        changes.append({
            "type": "hashtag",
            "original": original_hashtag,
            "normalized": replacement,
        })

        return replacement

    normalized_text = HASHTAG_PATTERN.sub(
        replace_hashtag,
        normalized_text,
    )

    # Step 4: Convert slang and repeated letters.
    normalized_text = WORD_PATTERN.sub(
        lambda match: _replace_word(
            match,
            changes,
        ),
        normalized_text,
    )

    # Step 5: Remove unnecessary spaces while
    # preserving sentence punctuation.
    normalized_text = re.sub(
        r"[ \t]+",
        " ",
        normalized_text,
    )

    normalized_text = re.sub(
        r" *\n *",
        "\n",
        normalized_text,
    )

    normalized_text = normalized_text.strip()

    return normalized_text, changes
