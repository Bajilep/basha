"""Main sentiment-analysis class."""

import math
from collections.abc import Iterable

from .context import find_context_score
from .emotions import detect_emotion
from .insights import (
    analyze_emoji_patterns,
    analyze_intensity,
    has_mixed_polarity,
    vocabulary_coverage,
)
from .lexicon import build_lexicon
from .modifiers import (
    CONTRAST_WORDS,
    DIMINISHERS,
    INTENSIFIERS,
    NEGATIONS,
    NEGATION_BREAKERS,
    NEGATION_WINDOW,
)
from .phrases import find_phrase_matches
from .sarcasm import detect_sarcasm
from .social import normalize_social_text
from .spelling import (
    build_spelling_index,
    find_closest_sentiment_word,
)
from .symbols import (
    SORTED_SYMBOLS,
    SYMBOL_SCORES,
)
from .tokenizer import tokenize_with_boundaries


class BashaAnalyser:
    """Analyse sentiment and emotion."""

    def __init__(self):
        """Create a sentiment analyzer."""

        self.lexicon = build_lexicon()

        self.negations = NEGATIONS
        self.intensifiers = INTENSIFIERS
        self.diminishers = DIMINISHERS

        self.negation_window = NEGATION_WINDOW
        self.negation_breakers = NEGATION_BREAKERS
        self.contrast_words = CONTRAST_WORDS

        self.symbol_scores = SYMBOL_SCORES
        self.sorted_symbols = SORTED_SYMBOLS

        self.spelling_index = build_spelling_index(
            self.lexicon
        )

        self.spelling_cache = {}

    def predict(self, text):
        """Analyse one sentence or paragraph."""

        self._validate_text(text)

        # Prepare informal/social-media text for analysis while
        # preserving the user's original text in the final result.
        analysis_text, social_changes = normalize_social_text(
            text
        )

        (
            original_words,
            words,
            sentence_ids,
        ) = tokenize_with_boundaries(analysis_text)

        contributions = []
        raw_score = 0.0

        positive_items = 0
        negative_items = 0

        # Find contrast words separately
        # for every sentence.
        contrast_indexes = (
            self._find_contrast_indexes(
                words,
                sentence_ids,
            )
        )

        # Detect complete phrases.
        phrase_matches = find_phrase_matches(
            words
        )

        # Remove phrases that incorrectly cross
        # a sentence boundary.
        phrase_matches = [
            match
            for match in phrase_matches
            if self._phrase_is_in_one_sentence(
                match,
                sentence_ids,
            )
        ]

        phrase_positions = set()

        # Analyse complete phrases first.
        for match in phrase_matches:
            phrase_positions.update(
                match["positions"]
            )

            base_score = match["score"]
            final_score = base_score
            applied_modifiers = ["phrase"]

            phrase_sentence = sentence_ids[
                match["start"]
            ]

            contrast_index = contrast_indexes.get(
                phrase_sentence
            )

            if contrast_index is not None:
                if match["start"] < contrast_index:
                    contrast_multiplier = 0.50
                else:
                    contrast_multiplier = 1.50

                final_score *= contrast_multiplier

                applied_modifiers.append(
                    f"contrast ×{contrast_multiplier}"
                )

            raw_score += final_score

            if final_score > 0:
                positive_items += 1

            elif final_score < 0:
                negative_items += 1

            contributions.append({
                "item": match["phrase"],
                "base_score": base_score,
                "modifiers": applied_modifiers,
                "final_score": round(
                    final_score,
                    3,
                ),
            })

        # Analyse individual words.
        for index, word in enumerate(words):
            # Phrase words must not be counted again.
            if index in phrase_positions:
                continue

            current_sentence = sentence_ids[index]

            nearby_start = max(
                0,
                index - self.negation_window,
            )

            nearby_end = min(
                len(words),
                index + self.negation_window + 1,
            )

            # Only use context words from the
            # current sentence.
            nearby_words = [
                words[nearby_index]
                for nearby_index in range(
                    nearby_start,
                    nearby_end,
                )
                if (
                    nearby_index != index
                    and sentence_ids[nearby_index]
                    == current_sentence
                )
            ]

            applied_modifiers = []
            lookup_word = word

            context_score = find_context_score(
                word,
                nearby_words,
            )

            if context_score is not None:
                base_score = context_score
                applied_modifiers.append("context")

            else:
                # Try spelling correction.
                if word not in self.lexicon:
                    corrected_word = (
                        find_closest_sentiment_word(
                            word,
                            self.lexicon,
                            self.spelling_index,
                            self.spelling_cache,
                        )
                    )

                    if corrected_word is None:
                        continue

                    lookup_word = corrected_word

                    applied_modifiers.append(
                        f"spelling: {word} → "
                        f"{corrected_word}"
                    )

                base_score = self.lexicon[
                    lookup_word
                ]

            final_score = base_score

            previous_start = max(
                0,
                index - self.negation_window,
            )

            # Only use previous words from the
            # current sentence.
            previous_words = [
                words[previous_index]
                for previous_index in range(
                    previous_start,
                    index,
                )
                if (
                    sentence_ids[previous_index]
                    == current_sentence
                )
            ]

            previous_words = (
                self._words_after_last_breaker(
                    previous_words
                )
            )

            # Apply negation.
            negation_count = sum(
                previous in self.negations
                for previous in previous_words
            )

            if negation_count % 2 == 1:
                final_score *= -1

                applied_modifiers.append(
                    "negation"
                )

            # Apply intensifiers and diminishers.
            if index > 0:
                previous_index = index - 1

                # The previous word must belong to
                # the same sentence.
                if (
                    sentence_ids[previous_index]
                    == current_sentence
                ):
                    previous_word = words[
                        previous_index
                    ]

                    if (
                        previous_word
                        in self.intensifiers
                    ):
                        multiplier = (
                            self.intensifiers[
                                previous_word
                            ]
                        )

                        final_score *= multiplier

                        applied_modifiers.append(
                            f"intensifier ×"
                            f"{multiplier}"
                        )

                    elif (
                        previous_word
                        in self.diminishers
                    ):
                        multiplier = (
                            self.diminishers[
                                previous_word
                            ]
                        )

                        final_score *= multiplier

                        applied_modifiers.append(
                            f"diminisher ×"
                            f"{multiplier}"
                        )

            # Apply capital-letter emphasis.
            original_word = original_words[index]

            if (
                original_word.isupper()
                and len(original_word) > 1
            ):
                final_score *= 1.25

                applied_modifiers.append(
                    "CAPS ×1.25"
                )

            # Apply contrast only within
            # the current sentence.
            contrast_index = contrast_indexes.get(
                current_sentence
            )

            if contrast_index is not None:
                if index < contrast_index:
                    contrast_multiplier = 0.50
                else:
                    contrast_multiplier = 1.50

                final_score *= contrast_multiplier

                applied_modifiers.append(
                    f"contrast ×{contrast_multiplier}"
                )

            raw_score += final_score

            if final_score > 0:
                positive_items += 1

            elif final_score < 0:
                negative_items += 1

            contributions.append({
                "item": original_word,
                "base_score": base_score,
                "modifiers": applied_modifiers,
                "final_score": round(
                    final_score,
                    3,
                ),
            })

        # Analyse emojis and emoticons.
        symbol_result = self._analyse_symbols(
            analysis_text
        )

        raw_score += symbol_result["score"]

        positive_items += symbol_result[
            "positive_items"
        ]

        negative_items += symbol_result[
            "negative_items"
        ]

        contributions.extend(
            symbol_result["contributions"]
        )

        # Apply exclamation-mark emphasis.
        exclamation_count = min(
            text.count("!"),
            4,
        )

        if (
            exclamation_count > 0
            and raw_score != 0
        ):
            punctuation_multiplier = (
                1 + exclamation_count * 0.10
            )

            old_score = raw_score
            raw_score *= punctuation_multiplier

            punctuation_change = (
                raw_score - old_score
            )

            contributions.append({
                "item": "!" * exclamation_count,
                "base_score": 0.0,
                "modifiers": [
                    f"punctuation ×"
                    f"{round(punctuation_multiplier, 2)}"
                ],
                "final_score": round(
                    punctuation_change,
                    3,
                ),
            })

        # Detect possible sarcasm.
        sarcasm_result = detect_sarcasm(
            analysis_text,
            words,
            self.lexicon,
        )

        if sarcasm_result["detected"]:
            old_score = raw_score

            if raw_score > 0:
                raw_score = -max(
                    raw_score * 0.75,
                    2.0,
                )

            elif raw_score < 0:
                raw_score *= 1.25

            else:
                raw_score = -2.0

            contributions.append({
                "item": "sarcasm",
                "base_score": round(
                    old_score,
                    3,
                ),
                "modifiers": sarcasm_result[
                    "reasons"
                ],
                "final_score": round(
                    raw_score,
                    3,
                ),
            })

        # Detect emotion separately.
        emotion_result = detect_emotion(
            analysis_text,
            words,
            self.negations,
            self.intensifiers,
            self.diminishers,
        )

        emotion_result[
            "sarcasm_adjusted"
        ] = False

        # Do not report surface joy as the
        # real emotion when sarcasm is detected.
        if (
            sarcasm_result["detected"]
            and emotion_result["label"] == "joy"
        ):
            emotion_result["label"] = "uncertain"
            emotion_result["confidence"] = 0.0

            emotion_result[
                "sarcasm_adjusted"
            ] = True

            emotion_result["evidence"].append({
                "item": "sarcasm",
                "emotion": "uncertain",
                "base_score": 0.0,
                "final_score": 0.0,
                "rules": [
                    "surface joy may be sarcastic"
                ],
            })

        compound = self._calculate_compound(
            raw_score
        )

        sentiment = self._get_label(
            compound
        )

        semantic_word_count = (
            len(words)
            - len(phrase_positions)
            + len(phrase_matches)
        )

        proportions = self._calculate_proportions(
            total_words=semantic_word_count,
            positive_items=positive_items,
            negative_items=negative_items,
        )

        explanation = self._format_explanation(
            contributions
        )

        intensity_result = analyze_intensity(
            text, words, self.intensifiers, self.diminishers
        )
        emoji_result = analyze_emoji_patterns(analysis_text)
        ignored_words = (
            self.negations
            | set(self.intensifiers)
            | set(self.diminishers)
            | self.negation_breakers
            | self.contrast_words
        )
        coverage_result = vocabulary_coverage(
            words, self.lexicon, ignored_words
        )

        return {
            "text": text,
            "label": sentiment,
            "emotion": emotion_result,
            "intensity": intensity_result,
            "emoji": emoji_result,
            "coverage": coverage_result,
            "scores": {
                "compound": compound,
                "positive": proportions[
                    "positive"
                ],
                "negative": proportions[
                    "negative"
                ],
                "neutral": proportions[
                    "neutral"
                ],
            },
            "flags": {
                "normalized_text": analysis_text,
                "social_normalizations": social_changes,
                "sarcasm": sarcasm_result[
                    "detected"
                ],
                "sarcasm_confidence": (
                    sarcasm_result["confidence"]
                ),
                "sarcasm_reasons": (
                    sarcasm_result["reasons"]
                ),
                "mixed_polarity": has_mixed_polarity(
                    explanation
                ),
            },
            "explanation": explanation,
        }

    def predict_batch(self, texts):
        """Analyse multiple texts."""

        if (
            isinstance(texts, (str, bytes))
            or not isinstance(texts, Iterable)
        ):
            raise TypeError(
                "Input must be a collection "
                "of sentences."
            )

        return [
            self.predict(text)
            for text in texts
        ]

    def analyze(self, text):
        """Alias for predict()."""
        return self.predict(text)

    def sentiment(self, text):
        """Return only the sentiment label."""
        return self.predict(text)["label"]

    def polarity(self, text):
        """Return only the compound score."""
        return self.predict(text)["scores"]["compound"]

    def emotions(self, text):
        """Return only the emotion result."""
        return self.predict(text)["emotion"]

    def intensity(self, text):
        """Return only the intensity result."""
        return self.predict(text)["intensity"]

    def emoji_analysis(self, text):
        """Return only the emoji result."""
        return self.predict(text)["emoji"]

    def explain(self, text):
        """Return only the token-level explanation."""
        return self.predict(text)["explanation"]

    def add_word(self, word, score):
        """Add a custom sentiment word."""

        if (
            not isinstance(word, str)
            or not word.strip()
        ):
            raise ValueError(
                "Word must be non-empty text."
            )

        if (
            isinstance(score, bool)
            or not isinstance(
                score,
                (int, float),
            )
        ):
            raise TypeError(
                "Score must be a number."
            )

        if (
            score == 0
            or score < -4
            or score > 4
        ):
            raise ValueError(
                "Score must be between -4 and +4 "
                "and cannot be zero."
            )

        clean_word = word.lower().strip()

        self.lexicon[clean_word] = float(
            score
        )

        self.spelling_index = (
            build_spelling_index(
                self.lexicon
            )
        )

        self.spelling_cache.clear()

    @property
    def vocabulary_size(self):
        """Return sentiment vocabulary size."""

        return len(self.lexicon)

    def _find_contrast_indexes(
        self,
        words,
        sentence_ids,
    ):
        """Find one contrast word per sentence."""

        contrast_indexes = {}

        for index, word in enumerate(words):
            sentence_id = sentence_ids[index]

            if (
                word in self.contrast_words
                and sentence_id
                not in contrast_indexes
            ):
                contrast_indexes[
                    sentence_id
                ] = index

        return contrast_indexes

    @staticmethod
    def _phrase_is_in_one_sentence(
        phrase_match,
        sentence_ids,
    ):
        """Check that a phrase does not cross punctuation."""

        phrase_sentence_ids = {
            sentence_ids[position]
            for position in phrase_match[
                "positions"
            ]
        }

        return len(phrase_sentence_ids) == 1

    def _words_after_last_breaker(
        self,
        previous_words,
    ):
        """Remove words before the latest breaker."""

        last_breaker = None

        for index, word in enumerate(
            previous_words
        ):
            if word in self.negation_breakers:
                last_breaker = index

        if last_breaker is None:
            return previous_words

        return previous_words[
            last_breaker + 1:
        ]

    @staticmethod
    def _format_explanation(contributions):
        """Create the public explanation format."""

        return [
            {
                "token": contribution["item"],
                "base_score": contribution[
                    "base_score"
                ],
                "final_score": contribution[
                    "final_score"
                ],
                "rules": contribution[
                    "modifiers"
                ],
            }
            for contribution in contributions
        ]

    @staticmethod
    def _validate_text(text):
        """Validate the supplied text."""

        if not isinstance(text, str):
            raise TypeError(
                "Input must be text."
            )

        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

    def _analyse_symbols(self, text):
        """Analyse emojis and emoticons."""

        score = 0.0
        positive_items = 0
        negative_items = 0
        contributions = []

        remaining_text = text

        for symbol in self.sorted_symbols:
            count = remaining_text.count(
                symbol
            )

            if count == 0:
                continue

            base_score = self.symbol_scores[
                symbol
            ]

            symbol_score = (
                base_score * count
            )

            score += symbol_score

            if symbol_score > 0:
                positive_items += count

            elif symbol_score < 0:
                negative_items += count

            modifiers = []

            if count > 1:
                modifiers.append(
                    f"repeated ×{count}"
                )

            contributions.append({
                "item": symbol,
                "base_score": base_score,
                "modifiers": modifiers,
                "final_score": round(
                    symbol_score,
                    3,
                ),
            })

            remaining_text = (
                remaining_text.replace(
                    symbol,
                    "",
                )
            )

        return {
            "score": score,
            "positive_items": positive_items,
            "negative_items": negative_items,
            "contributions": contributions,
        }

    @staticmethod
    def _calculate_compound(raw_score):
        """Convert a raw score to -1 through +1."""

        if raw_score == 0:
            return 0.0

        compound = raw_score / math.sqrt(
            raw_score ** 2 + 16
        )

        return round(
            compound,
            4,
        )

    @staticmethod
    def _get_label(compound):
        """Convert compound score into a label."""

        if compound >= 0.05:
            return "positive"

        if compound <= -0.05:
            return "negative"

        return "neutral"

    @staticmethod
    def _calculate_proportions(
        total_words,
        positive_items,
        negative_items,
    ):
        """Calculate sentiment proportions."""

        neutral_items = max(
            total_words
            - positive_items
            - negative_items,
            0,
        )

        total_items = (
            positive_items
            + negative_items
            + neutral_items
        )

        if total_items == 0:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

        return {
            "positive": round(
                positive_items / total_items,
                3,
            ),
            "negative": round(
                negative_items / total_items,
                3,
            ),
            "neutral": round(
                neutral_items / total_items,
                3,
            ),
        }
