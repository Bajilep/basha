
import io
import os

# Disable Hugging Face and tqdm download progress displays.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache
from pathlib import Path
import logging
from transformers.utils import logging as hf_logging
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)

from basha.sentiment.english_model import EnglishSentimentModel
from basha.fusion.english_fusion import EnglishFusion
from basha.longtext.document_analyzer import LongDocumentAnalyzer


class AnalysisResult(dict):
    """Structured result with professional, readable display."""

    @staticmethod
    def _percentage(value):
        if value is None:
            return "N/A"
        return f"{float(value) * 100:.2f}%"

    def display(self):
        print(self)

    def __str__(self):
        lines = [
            "BASHA | SENTIMENT ANALYSIS",
            "=" * 42,
        ]

        sentiment = self.get("sentiment", {})

        # Basic sentiment result
        if "label" in self:
            label = self.get("label", "unknown")
            confidence = self.get("confidence")

            lines.extend([
                f"Overall Sentiment : {str(label).upper()}",
                f"Confidence        : {self._percentage(confidence)}",
            ])

            scores = [
                ("Positive", self.get("positive")),
                ("Neutral", self.get("neutral")),
                ("Negative", self.get("negative")),
            ]

            if any(value is not None for _, value in scores):
                lines.extend([
                    "",
                    "SENTIMENT SCORES",
                    "-" * 42,
                ])
                for name, value in scores:
                    if value is not None:
                        lines.append(
                            f"{name:<18}: {self._percentage(value)}"
                        )

        # Full fusion or document result
        elif isinstance(sentiment, dict):
            lines.extend([
                f"Overall Sentiment : {str(sentiment.get('label', 'unknown')).upper()}",
                f"Confidence        : {self._percentage(sentiment.get('confidence'))}",
            ])

        aspects = self.get("aspects", [])

        if aspects:
            lines.extend([
                "",
                "ASPECT ANALYSIS",
                "-" * 42,
            ])

            for item in aspects:
                name = item.get("aspect", "Unknown")
                label = str(item.get("sentiment", "unknown")).upper()
                confidence = self._percentage(item.get("confidence"))

                lines.append(f"{name:<20}: {label} ({confidence})")

        sarcasm = self.get("sarcasm")

        if isinstance(sarcasm, dict):
            detected = sarcasm.get("detected", False)
            confidence = self._percentage(sarcasm.get("confidence"))

            lines.extend([
                "",
                "SARCASM ANALYSIS",
                "-" * 42,
                f"Detected          : {'Yes' if detected else 'No'}",
                f"Confidence        : {confidence}",
            ])

        if "mixed" in self:
            lines.extend([
                "",
                f"Mixed Sentiment   : {'Yes' if self['mixed'] else 'No'}",
            ])

        sentences = self.get("sentences", [])

        if sentences:
            lines.extend([
                "",
                "SENTENCE ANALYSIS",
                "-" * 42,
                f"Total Sentences   : {len(sentences)}",
            ])

            for index, item in enumerate(sentences, start=1):
                lines.extend([
                    "",
                    f"{index}. {item.get('text', '')}",
                    f"   Sentiment      : {str(item.get('label', 'unknown')).upper()}",
                    f"   Confidence     : {self._percentage(item.get('confidence'))}",
                ])

        lines.append("=" * 42)

        return "\n".join(lines)

    __repr__ = __str__


def _quietly_create(factory):
    """Suppress initialization messages and progress bars."""

    previous_verbosity = hf_logging.get_verbosity()
    transformer_progress_enabled = hf_logging.is_progress_bar_enabled()
    hub_progress_was_disabled = are_progress_bars_disabled()

    hub_loggers = [
        logging.getLogger("huggingface_hub"),
        logging.getLogger("huggingface_hub.utils._http"),
    ]

    previous_logger_levels = [logger.level for logger in hub_loggers]

    try:
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        disable_progress_bars()

        for logger in hub_loggers:
            logger.setLevel(logging.ERROR)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return factory()

    finally:
        hf_logging.set_verbosity(previous_verbosity)

        if transformer_progress_enabled:
            hf_logging.enable_progress_bar()
        else:
            hf_logging.disable_progress_bar()

        if not hub_progress_was_disabled:
            enable_progress_bars()

        for logger, level in zip(hub_loggers, previous_logger_levels):
            logger.setLevel(level)

@lru_cache(maxsize=1)
def _get_sentiment_model():
    return _quietly_create(EnglishSentimentModel)


@lru_cache(maxsize=1)
def _get_fusion():
    return _quietly_create(EnglishFusion)


@lru_cache(maxsize=1)
def _get_document_analyzer():
    return _quietly_create(
        lambda: LongDocumentAnalyzer(fusion=_get_fusion())
    )


def _clean_analysis(result):
    """Convert internal analysis into a clean public result."""

    sarcasm = result.get("sarcasm", {})
    mixed = result.get("mixed_analysis", {})

    aspects = [
        {
            "aspect": item["aspect"],
            "sentiment": item["sentiment"],
            "confidence": item["confidence"],
        }
        for item in result.get("aspects", [])
    ]

    return AnalysisResult({
        "language": result.get("language", "en"),
        "sentiment": result["sentiment"],
        "aspects": aspects,
        "sarcasm": {
            "detected": sarcasm.get("detected", False),
            "confidence": sarcasm.get("confidence", 0.0),
        },
        "mixed": mixed.get("mixed", False),
    })


def sentiment(text: str) -> AnalysisResult:
    """Return basic English sentiment."""

    result = _get_sentiment_model().predict(text)
    return AnalysisResult(result)


def analyze(text: str) -> AnalysisResult:
    """Return complete English sentiment analysis."""

    result = _get_fusion().analyze(text)
    return _clean_analysis(result)


def analyze_document(document: str | os.PathLike) -> AnalysisResult:
    """Analyze a document supplied as a file path or raw text."""

    if isinstance(document, os.PathLike):
        text = Path(document).read_text(encoding="utf-8")

    elif isinstance(document, str):
        path = Path(document)

        try:
            is_file = path.is_file()
        except OSError:
            is_file = False

        if is_file:
            text = path.read_text(encoding="utf-8")
        else:
            text = document

    else:
        raise TypeError("document must be a string or file path")

    result = _get_document_analyzer().analyze(text)
    clean_result = _clean_analysis(result)

    clean_result["sentences"] = [
        {
            "text": item["text"],
            "label": item["label"],
            "confidence": item["confidence"],
        }
        for item in result.get("sentences", [])
    ]

    return clean_result
