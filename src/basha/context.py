"""Context rules for words with multiple meanings."""


CONTEXT_RULES = {
    "sick": {
        "positive_context": {
            "movie", "song", "music", "beat",
            "performance", "game", "design",
            "style", "trick", "skills",
            "awesome", "amazing",
        },
        "negative_context": {
            "feel", "feeling", "felt", "became",
            "ill", "hospital", "doctor",
            "medicine", "health", "patient",
            "child", "vomit", "fever", "pain",
        },
        "neutral_context": set(),
        "positive_score": 2.5,
        "negative_score": -2.5,
        "neutral_score": 0.0,
    },

    "killer": {
        "positive_context": {
            "design", "feature", "performance",
            "look", "style", "song", "beat",
            "idea", "product", "application",
            "app",
        },
        "negative_context": {
            "murder", "crime", "attack", "police",
            "arrested", "weapon", "victim",
            "court", "criminal", "serial",
        },
        "neutral_context": {
            "person", "name",
        },
        "positive_score": 2.5,
        "negative_score": -3.5,
        "neutral_score": 0.0,
    },

    "fire": {
        "positive_context": {
            "song", "music", "beat",
            "performance", "design", "style",
            "track", "album",
        },
        "negative_context": {
            "burn", "burning", "smoke",
            "building", "house", "forest",
            "accident", "emergency", "alarm",
            "injured",
        },
        "neutral_context": {
            "camp", "wood", "cooking",
        },
        "positive_score": 2.5,
        "negative_score": -3.0,
        "neutral_score": 0.0,
    },

    "sharp": {
        "positive_context": {
            "image", "display", "picture",
            "camera", "design", "mind",
            "details", "quality",
        },
        "negative_context": {
            "pain", "injury", "hurt",
            "headache",
        },
        "neutral_context": {
            "knife", "edge", "corner",
            "object",
        },
        "positive_score": 2.0,
        "negative_score": -2.5,
        "neutral_score": 0.0,
    },

    "hot": {
        "positive_context": {
            "design", "style", "look",
            "trend", "product",
        },
        "negative_context": {
            "overheating", "burning",
            "painful", "dangerous",
        },
        "neutral_context": {
            "weather", "food", "drink",
            "water", "tea", "coffee",
            "temperature",
        },
        "positive_score": 2.0,
        "negative_score": -2.0,
        "neutral_score": 0.0,
    },

    "cold": {
        "positive_context": {
            "refreshing",
        },
        "negative_context": {
            "person", "response", "attitude",
            "behaviour", "behavior",
            "unfriendly",
        },
        "neutral_context": {
            "weather", "water", "drink",
            "food", "temperature", "winter",
        },
        "positive_score": 1.5,
        "negative_score": -2.0,
        "neutral_score": 0.0,
    },

    "sweet": {
        "positive_context": {
            "person", "gesture", "message",
            "voice", "smile", "kind",
        },
        "negative_context": {
            "excessive", "overly",
        },
        "neutral_context": {
            "food", "tea", "coffee",
            "cake", "sugar", "taste",
        },
        "positive_score": 2.0,
        "negative_score": -1.0,
        "neutral_score": 0.0,
    },

    "dark": {
        "positive_context": {
            "mode", "theme", "design",
            "style",
        },
        "negative_context": {
            "thoughts", "mood", "future",
            "secret", "dangerous",
        },
        "neutral_context": {
            "room", "night", "colour",
            "color", "sky",
        },
        "positive_score": 1.5,
        "negative_score": -2.0,
        "neutral_score": 0.0,
    },

    "wicked": {
        "positive_context": {
            "performance", "song", "beat",
            "skills", "design", "trick",
            "game", "style", "awesome",
        },
        "negative_context": {
            "person", "witch", "evil",
            "cruel", "crime", "act",
        },
        "neutral_context": set(),
        "positive_score": 2.5,
        "negative_score": -3.0,
        "neutral_score": 0.0,
    },

    "mad": {
        "positive_context": {
            "skills", "talent", "performance",
            "design", "style", "beat", "game",
        },
        "negative_context": {
            "angry", "furious", "upset",
            "annoyed", "temper", "rage",
        },
        "neutral_context": set(),
        "positive_score": 2.0,
        "negative_score": -2.5,
        "neutral_score": 0.0,
    },

    "crazy": {
        "positive_context": {
            "good", "great", "amazing",
            "awesome", "talented", "skills",
            "performance", "design", "idea",
        },
        "negative_context": {
            "dangerous", "angry", "violent",
            "unstable", "behaviour",
            "behavior", "driving",
        },
        "neutral_context": set(),
        "positive_score": 2.0,
        "negative_score": -2.5,
        "neutral_score": 0.0,
    },

    "dead": {
        "positive_context": {
            "funny", "laughing", "joke",
            "humour", "humor",
        },
        "negative_context": {
            "person", "animal", "battery",
            "phone", "computer", "plant",
            "body", "found",
        },
        "neutral_context": set(),
        "positive_score": 1.5,
        "negative_score": -3.5,
        "neutral_score": 0.0,
    },
}


def find_context_score(word, nearby_words):
    """Return the score determined by nearby words."""

    rule = CONTEXT_RULES.get(word)

    if rule is None:
        return None

    nearby_set = set(nearby_words)

    matches = {
        "positive": len(
            nearby_set
            & rule["positive_context"]
        ),
        "negative": len(
            nearby_set
            & rule["negative_context"]
        ),
        "neutral": len(
            nearby_set
            & rule["neutral_context"]
        ),
    }

    highest_count = max(
        matches.values()
    )

    if highest_count == 0:
        return None

    winning_contexts = [
        context_name
        for context_name, count in matches.items()
        if count == highest_count
    ]

    # Do not guess when two contexts are tied.
    if len(winning_contexts) != 1:
        return None

    winning_context = winning_contexts[0]

    return rule[
        f"{winning_context}_score"
    ]