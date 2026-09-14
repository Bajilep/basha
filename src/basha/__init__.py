"""Basha sentiment-analysis library."""

from .analyzer import BashaAnalyser

BashaAnalyzer = BashaAnalyser


# Lowercase public name requested by the library author.
bashaanalyser = BashaAnalyser

# Keep the old name temporarily for existing users.
SentimentAnalyzer = BashaAnalyser

_default_analyzer = BashaAnalyser()


def analyze(text):
    return _default_analyzer.predict(text)


def sentiment(text):
    return _default_analyzer.sentiment(text)


def polarity(text):
    return _default_analyzer.polarity(text)


def emotions(text):
    return _default_analyzer.emotions(text)


def intensity(text):
    return _default_analyzer.intensity(text)


def emoji_analysis(text):
    return _default_analyzer.emoji_analysis(text)


def explain(text):
    return _default_analyzer.explain(text)


__version__ = "0.3.0"

__all__ = [
    "BashaAnalyser",
    "BashaAnalyzer",
    "bashaanalyser",
    "SentimentAnalyzer",
    "analyze",
    "sentiment",
    "polarity",
    "emotions",
    "intensity",
    "emoji_analysis",
    "explain",
]
