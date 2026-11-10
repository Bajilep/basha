"""Rule-based emotion detection.

Supported emotions:
- joy
- sadness
- anger
- fear
- surprise
- disgust
- neutral
- mixed
"""


# Each word has an emotion strength from 1.0 to 4.0.
EMOTION_WORDS = {
    "joy": {
        "happy": 3.0,
        "happiness": 3.0,
        "joy": 3.5,
        "joyful": 3.5,
        "delighted": 3.5,
        "excited": 3.0,
        "cheerful": 3.0,
        "pleased": 2.5,
        "glad": 2.5,
        "grateful": 2.5,
        "love": 3.5,
        "loved": 3.5,
        "lovely": 2.5,
        "wonderful": 3.5,
        "amazing": 3.5,
        "awesome": 3.5,
        "excellent": 3.5,
        "fantastic": 3.5,
        "enjoy": 2.5,
        "enjoyed": 2.5,
        "smile": 2.0,
        "smiling": 2.0,
        "laugh": 2.5,
        "laughing": 2.5,
        "celebrate": 3.0,
        "celebrating": 3.0,
        "proud": 2.5,
        "thrilled": 3.5,
        "satisfied": 2.5,
    },

    "sadness": {
        "sad": 3.0,
        "sadness": 3.5,
        "unhappy": 3.0,
        "depressed": 4.0,
        "miserable": 3.5,
        "heartbroken": 4.0,
        "cry": 2.5,
        "crying": 3.0,
        "cried": 3.0,
        "tears": 2.5,
        "lonely": 3.0,
        "hurt": 2.5,
        "pain": 2.5,
        "painful": 3.0,
        "disappointed": 3.0,
        "disappointing": 2.5,
        "regret": 2.5,
        "regretted": 2.5,
        "hopeless": 3.5,
        "gloomy": 2.5,
        "upset": 2.5,
        "devastated": 4.0,
        "devastating": 4.0,
        "grief": 4.0,
        "sorrow": 3.5,
        "lost": 2.5,
    },

    "anger": {
        "angry": 3.5,
        "anger": 3.5,
        "furious": 4.0,
        "mad": 3.0,
        "annoyed": 2.0,
        "annoying": 2.0,
        "irritated": 2.5,
        "irritating": 2.5,
        "outraged": 4.0,
        "outrageous": 3.0,
        "rage": 4.0,
        "hate": 3.5,
        "hated": 3.5,
        "hostile": 3.0,
        "frustrated": 3.0,
        "frustrating": 2.5,
        "infuriating": 4.0,
        "offended": 2.5,
        "insulted": 3.0,
        "betrayed": 3.5,
        "cheated": 3.0,
        "rude": 2.0,
        "disrespectful": 2.5,
    },

    "fear": {
        "afraid": 3.5,
        "fear": 3.5,
        "fearful": 3.5,
        "scared": 3.5,
        "terrified": 4.0,
        "terrifying": 4.0,
        "worried": 2.5,
        "worry": 2.5,
        "nervous": 2.5,
        "anxious": 3.0,
        "anxiety": 3.0,
        "panic": 4.0,
        "panicked": 4.0,
        "danger": 3.0,
        "dangerous": 3.0,
        "threat": 3.0,
        "threatened": 3.5,
        "unsafe": 3.0,
        "horror": 3.5,
        "horrified": 4.0,
        "nightmare": 3.5,
        "frightened": 3.5,
        "shaking": 2.0,
    },

    "surprise": {
        "surprised": 3.0,
        "surprise": 3.0,
        "surprising": 3.0,
        "shocked": 3.5,
        "shocking": 3.5,
        "unexpected": 2.5,
        "unexpectedly": 2.5,
        "amazed": 3.0,
        "astonished": 3.5,
        "astonishing": 3.5,
        "unbelievable": 3.0,
        "unbelievably": 3.0,
        "speechless": 3.0,
        "suddenly": 2.0,
        "sudden": 2.0,
        "wow": 3.0,
    },

    "disgust": {
        "disgust": 3.5,
        "disgusted": 3.5,
        "disgusting": 4.0,
        "gross": 3.0,
        "revolting": 4.0,
        "nasty": 3.0,
        "filthy": 3.0,
        "dirty": 2.0,
        "horrible": 3.0,
        "awful": 3.0,
        "repulsive": 4.0,
        "sickening": 4.0,
        "rotten": 3.0,
        "smelly": 2.5,
        "vomit": 3.5,
        "toxic": 3.0,
        "pathetic": 2.5,
        "trash": 2.5,
        "garbage": 2.5,
    },
}


EMOTION_EMOJIS = {
    # Joy
    "😀": ("joy", 3.0),
    "😃": ("joy", 3.0),
    "😄": ("joy", 3.0),
    "😁": ("joy", 2.5),
    "😊": ("joy", 3.0),
    "😍": ("joy", 3.5),
    "🥰": ("joy", 3.5),
    "😂": ("joy", 3.0),
    "🤣": ("joy", 3.0),
    "❤️": ("joy", 3.5),
    "❤": ("joy", 3.5),
    "🥳": ("joy", 3.0),

    # Sadness
    "😔": ("sadness", 2.5),
    "😞": ("sadness", 3.0),
    "😢": ("sadness", 3.5),
    "😭": ("sadness", 4.0),
    "💔": ("sadness", 4.0),

    # Anger
    "😠": ("anger", 3.0),
    "😡": ("anger", 3.5),
    "🤬": ("anger", 4.0),

    # Fear
    "😨": ("fear", 3.0),
    "😰": ("fear", 3.0),
    "😱": ("fear", 4.0),

    # Surprise
    "😮": ("surprise", 3.0),
    "😲": ("surprise", 3.5),
    "🤯": ("surprise", 4.0),

    # Disgust
    "🤢": ("disgust", 3.5),
    "🤮": ("disgust", 4.0),
}


# Create a fast word-to-emotion lookup only once.
WORD_TO_EMOTIONS = {}

for emotion_name, emotion_words in EMOTION_WORDS.items():
    for emotion_word, emotion_score in emotion_words.items():
        WORD_TO_EMOTIONS.setdefault(
            emotion_word,
            []
        ).append(
            (emotion_name, emotion_score)
        )


SORTED_EMOTION_EMOJIS = tuple(
    sorted(
        EMOTION_EMOJIS,
        key=len,
        reverse=True,
    )
)


def _previous_words(words, index, window=3):
    """Return words before the current word."""

    return words[
        max(0, index - window):index
    ]


def _apply_modifier(
    score,
    previous_words,
    negations,
    intensifiers,
    diminishers,
):
    """Apply negation and strength modifiers."""

    rules = []

    negation_count = sum(
        word in negations
        for word in previous_words
    )

    # If the emotion is negated, do not guess
    # another emotion.
    if negation_count % 2 == 1:
        return 0.0, ["negated emotion"]

    if previous_words:
        previous_word = previous_words[-1]

        if previous_word in intensifiers:
            multiplier = intensifiers[
                previous_word
            ]

            score *= multiplier

            rules.append(
                f"intensifier ×{multiplier}"
            )

        elif previous_word in diminishers:
            multiplier = diminishers[
                previous_word
            ]

            score *= multiplier

            rules.append(
                f"diminisher ×{multiplier}"
            )

    return score, rules


def _normalise_scores(raw_scores):
    """Convert raw emotion scores into proportions."""

    total = sum(raw_scores.values())

    if total == 0:
        return {
            emotion: 0.0
            for emotion in raw_scores
        }

    return {
        emotion: round(
            score / total,
            3,
        )
        for emotion, score in raw_scores.items()
    }


def detect_emotion(
    text,
    words,
    negations,
    intensifiers,
    diminishers,
):
    """
    Detect the main emotion in a sentence.

    Returns:
        label:
            Main emotion, neutral or mixed.

        confidence:
            Value between 0.0 and 1.0.

        scores:
            Proportion for every emotion.

        evidence:
            Words and emojis responsible for the result.
    """

    raw_scores = {
        emotion: 0.0
        for emotion in EMOTION_WORDS
    }

    evidence = []

    # Analyse emotion words.
    for index, word in enumerate(words):
        emotion_matches = WORD_TO_EMOTIONS.get(
            word,
            ()
        )

        if not emotion_matches:
            continue

        previous = _previous_words(
            words,
            index,
        )

        for emotion, base_score in emotion_matches:
            final_score, rules = _apply_modifier(
                base_score,
                previous,
                negations,
                intensifiers,
                diminishers,
            )

            if final_score == 0:
                continue

            raw_scores[emotion] += final_score

            evidence.append({
                "item": word,
                "emotion": emotion,
                "base_score": base_score,
                "final_score": round(
                    final_score,
                    3,
                ),
                "rules": rules,
            })

    # Analyse emotion emojis.
    remaining_text = text

    for emoji in SORTED_EMOTION_EMOJIS:
        count = remaining_text.count(emoji)

        if count == 0:
            continue

        emotion, base_score = EMOTION_EMOJIS[
            emoji
        ]

        final_score = base_score * count

        raw_scores[emotion] += final_score

        rules = []

        if count > 1:
            rules.append(
                f"repeated ×{count}"
            )

        evidence.append({
            "item": emoji,
            "emotion": emotion,
            "base_score": base_score,
            "final_score": round(
                final_score,
                3,
            ),
            "rules": rules,
        })

        remaining_text = remaining_text.replace(
            emoji,
            "",
        )

    scores = _normalise_scores(
        raw_scores
    )

    ranked_emotions = sorted(
        raw_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    main_emotion, main_score = ranked_emotions[0]

    if main_score == 0:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "scores": scores,
            "evidence": [],
        }

    second_score = ranked_emotions[1][1]
    total_score = sum(raw_scores.values())

    confidence = main_score / total_score

    # Return mixed when the strongest two emotions
    # have almost the same score.
    if (
        second_score > 0
        and abs(main_score - second_score)
        / max(main_score, 1.0) < 0.15
    ):
        label = "mixed"
    else:
        label = main_emotion

    return {
        "label": label,
        "confidence": round(
            confidence,
            2,
        ),
        "scores": scores,
        "evidence": evidence,
    }