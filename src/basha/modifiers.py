"""Words that modify sentiment direction or strength."""


# Number of previous words checked for negation.
NEGATION_WINDOW = 3


NEGATIONS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "rarely",
    "seldom",
    "without",

    # Cannot
    "cannot",
    "can't",
    "cant",

    # Is not
    "isn't",
    "isnt",

    # Are not
    "aren't",
    "arent",

    # Was not
    "wasn't",
    "wasnt",

    # Were not
    "weren't",
    "werent",

    # Do not
    "don't",
    "dont",

    # Does not
    "doesn't",
    "doesnt",

    # Did not
    "didn't",
    "didnt",

    # Has not
    "hasn't",
    "hasnt",

    # Have not
    "haven't",
    "havent",

    # Had not
    "hadn't",
    "hadnt",

    # Will not
    "won't",
    "wont",

    # Would not
    "wouldn't",
    "wouldnt",

    # Should not
    "shouldn't",
    "shouldnt",

    # Could not
    "couldn't",
    "couldnt",

    # Must not
    "mustn't",
    "mustnt",

    # Informal
    "ain't",
    "aint",
}


INTENSIFIERS = {
    "very": 1.50,
    "really": 1.40,
    "extremely": 2.00,
    "absolutely": 1.80,
    "incredibly": 1.70,
    "so": 1.30,
    "too": 1.30,
    "highly": 1.50,
    "totally": 1.50,
    "completely": 1.60,
    "super": 1.50,
    "deeply": 1.40,
    "especially": 1.30,
    "exceptionally": 1.70,
    "hugely": 1.50,
    "immensely": 1.60,
    "particularly": 1.30,
    "perfectly": 1.40,
    "remarkably": 1.50,
    "seriously": 1.40,
    "strongly": 1.40,
    "truly": 1.40,
    "utterly": 1.70,
    "awfully": 1.40,
    "terribly": 1.50,
    "amazingly": 1.60,
    "insanely": 1.70,
    "unbelievably": 1.70,
}


DIMINISHERS = {
    "slightly": 0.50,
    "somewhat": 0.70,
    "little": 0.70,
    "barely": 0.50,
    "fairly": 0.80,
    "partly": 0.70,
    "almost": 0.80,
    "rather": 0.80,
    "kind-of": 0.70,
    "kinda": 0.70,
    "sort-of": 0.70,
    "sorta": 0.70,
    "marginally": 0.60,
    "occasionally": 0.80,
    "relatively": 0.80,
    "reasonably": 0.80,
    "moderately": 0.75,
}


# These words stop a previous negation from affecting
# words appearing after them.
NEGATION_BREAKERS = {
    "but",
    "however",
    "although",
    "though",
    "yet",
    "except",
    "nevertheless",
}


# The part after these words normally has more importance.
CONTRAST_WORDS = {
    "but",
    "however",
    "although",
    "though",
    "yet",
}