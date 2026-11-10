"""Weighted sentiment vocabulary.

Scores range from:
-4.0 = extremely negative
+4.0 = extremely positive
"""


def _split_words(text):
    """Convert a block of text into unique words."""

    return tuple(
        dict.fromkeys(text.split())
    )


SCORE_GROUPS = {
    4.0: _split_words("""
        amazing awesome brilliant excellent exceptional
        fantastic flawless incredible magnificent
        outstanding perfect phenomenal remarkable
        sensational spectacular superb terrific thrilled
        triumphant wonderful wow best love loved
        delightful extraordinary gorgeous ideal
        masterpiece miraculous breathtaking legendary
        unbeatable ecstatic
    """),

    3.0: _split_words("""
        great admirable adorable appreciate appreciated
        beautiful cheerful commendable confident
        congratulations cool creative enjoyable excited
        exciting fabulous favorable favourite favorite
        friendly fun generous glad grateful happy helpful
        impressive inspiring joyful kind lovely marvelous
        marvellous optimistic pleased positive proud
        recommend recommended reliable satisfying
        satisfied smooth strong success successful
        superior talented useful valuable vibrant win
        winner winning worthwhile pleasurable refreshing
        dependable trustworthy excellent-looking
        impressive-looking outstandingly delighted
        adore admire joy
    """),

    2.0: _split_words("""
        acceptable agree agreeable benefit beneficial
        better bright capable comfortable convenient
        decent effective efficient encouraging enjoy
        enjoyed entertaining fair fast fortunate fresh
        good hopeful improved improving interesting like
        liked lively nice pleasant promising quality safe
        smart stable supportive sweet thanks thankful
        worthy accurate affordable clean easy elegant
        flexible genuine innovative neat productive quick
        responsive secure skillful solid welcoming works
        worked progress improvement advantage advantages
        helpfulness durable
        clever funny hope laugh relaxed respect trust
    """),

    1.0: _split_words("""
        calm clear correct enough okay ok prepared ready
        reasonable simple suitable timely welcome
        balanced practical satisfactory fine peaceful
        proper steady adequate manageable pass passed
        solved solution solutions usable valid warm
        functional
    """),

    -1.0: _split_words("""
        average concern concerned inconvenient limited
        low minor ordinary questionable sadly shortage
        small strange uncertain unclear unexpected unknown
        annoying awkward basic bothered delay doubt
        doubtful dull lacking nervous odd restricted
        scarce upset inconvenience disappointing-looking
        unimpressive
        bored
    """),

    -2.0: _split_words("""
        bad buggy careless complaint complaints confusing
        difficult dirty dislike disliked expensive fear
        flawed glitch hard incorrect issue issues late
        lazy mediocre missing noisy offensive regret
        regretted risky rough slow sorry suspicious tired
        trouble troubling uncomfortable unfair unhappy
        unpleasant unsafe unstable warning worry worried
        wrong inaccurate inadequate incomplete
        inconsistent ineffective inefficient overpriced
        unreliable weak limitation limitations malfunction
        malfunctions dissatisfied laggy freezing frozen
        annoyed anxious frustrated hurt pain scary
    """),

    -3.0: _split_words("""
        angry boring corrupt cruel damaged dead defective
        delayed disappointed disappointing disaster error
        errors fake frustrating furious harmful hopeless
        inferior misleading nasty negative painful poor
        problem problems rejected ridiculous rude sad sick
        stolen stressful stupid ugly waste abusive alarmed
        chaotic cheated depressed disturbing embarrassed
        faulty fearful greedy guilty hostile ignored
        impossible insulting irritating miserable ruined
        shameful shocking crash crashed crashing cancelled
        canceled stuck failure-prone
        disgusted idiotic terrifying
    """),

    -4.0: _split_words("""
        atrocious awful broken catastrophic disgusting
        dreadful fail failed failure garbage hate hated
        horrible horrific nightmare pathetic scam terrible
        toxic trash unacceptable unusable useless worst
        worthless abysmal disastrous disgraceful fraud
        fraudulent horrendous horrid infuriating outrageous
        revolting unbearable unforgivable appalling
        devastated devastating dangerous ripoff
        rip-off counterfeit
    """),
}


def _create_base_lexicon():
    """Create and validate the base sentiment lexicon."""

    lexicon = {}

    for score, words in SCORE_GROUPS.items():
        for word in words:
            if word in lexicon:
                raise ValueError(
                    f"Duplicate lexicon word: {word}"
                )

            lexicon[word] = float(score)

    return lexicon


# Build the main vocabulary only once.
_BASE_LEXICON = _create_base_lexicon()


def build_lexicon():
    """
    Return a separate copy of the lexicon.

    This allows each analyzer to add custom words
    without changing other analyzer objects.
    """

    return _BASE_LEXICON.copy()
