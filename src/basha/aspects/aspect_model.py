from __future__ import annotations

import re
from typing import Dict, List

import torch
from transformers import pipeline

from ..longtext.splitter import split_sentences


class EnglishAspectModel:
    """
    End-to-end English Aspect-Based Sentiment Analysis.

    Extracts:
        - aspect term
        - aspect sentiment
        - confidence
        - character position

    Example:
        "The battery is excellent but the camera is terrible."

    Expected:
        battery -> positive
        camera  -> negative

    Parameters
    ----------
    model_name:
        Hugging Face model name or local folder path.
        If omitted, the original Gauneg model is used.

    sentence_level:
        If True, every input is analyzed sentence by sentence
        regardless of length. Default False.

    Long text
    ---------
    The aspect model was trained on single sentences and
    finds far fewer aspects in long input. Text longer than
    SENTENCE_MODE_MIN_CHARS is therefore analyzed sentence by
    sentence automatically. Short text is analyzed in one
    pass, exactly as before.

    Whenever text is split, "start" and "end" refer to the
    whitespace-normalized text, not the original string.
    """

    MODEL_NAME = (
        "gauneg/"
        "roberta-base-absa-ate-sentiment"
    )

    # Above this length, analyze sentence by sentence.
    # The frozen benchmark's longest text is 357 characters,
    # so benchmark results are not affected.
    SENTENCE_MODE_MIN_CHARS = 500

    # A single sentence longer than this is split by words
    # (about 375 tokens, safely under the 512 limit).
    MAX_PIECE_CHARS = 1500

    LABEL_MAP = {
        "pos": "positive",
        "positive": "positive",

        "neg": "negative",
        "negative": "negative",

        "neu": "neutral",
        "neutral": "neutral",

        # The conflict label was never learned by the
        # fine-tuned model (recall 0.0), so it is treated
        # as neutral instead of leaking out as "con".
        "con": "neutral",
        "conflict": "neutral",
    }

    def __init__(
        self,
        model_name: str | None = None,
        sentence_level: bool = False,
    ) -> None:

        self.model_name = (
            model_name
            or self.MODEL_NAME
        )

        self.sentence_level = sentence_level

        self.device = (
            0
            if torch.cuda.is_available()
            else -1
        )

        device_name = (
            "cuda"
            if self.device == 0
            else "cpu"
        )

        print(
            "Loading English aspect model "
            f"({self.model_name}) "
            f"on {device_name}..."
        )

        self.pipeline = pipeline(
            task="token-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",
            device=self.device,
        )

    @classmethod
    def _normalize_label(
        cls,
        label: str,
    ) -> str:

        normalized = (
            str(label)
            .strip()
            .lower()
        )

        return cls.LABEL_MAP.get(
            normalized,
            normalized,
        )

    @staticmethod
    def _is_valid_aspect(
        aspect: str,
    ) -> bool:
        """
        Check whether an extracted aspect contains
        meaningful characters.

        Rejects punctuation-only predictions such as:
            "."
            ","
            "!"
            "?"
        """

        aspect = aspect.strip()

        if not aspect:
            return False

        # Aspect must contain at least one
        # English letter or number.
        if not re.search(
            r"[A-Za-z0-9]",
            aspect,
        ):
            return False

        return True

    def _split_oversized(
        self,
        sentence: str,
    ) -> List[str]:
        """
        Break a single very long sentence into pieces
        of at most MAX_PIECE_CHARS characters.
        """

        pieces: List[str] = []
        current: List[str] = []
        length = 0

        for word in sentence.split():

            if (
                current
                and length + len(word) + 1
                > self.MAX_PIECE_CHARS
            ):
                pieces.append(
                    " ".join(current)
                )
                current = []
                length = 0

            current.append(word)
            length += len(word) + 1

        if current:
            pieces.append(
                " ".join(current)
            )

        return pieces

    def _sentence_pieces(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into one piece per sentence. A sentence
        that is itself too long is split further by words.
        """

        pieces: List[str] = []

        for sentence in split_sentences(
            text
        ):
            if (
                len(sentence)
                > self.MAX_PIECE_CHARS
            ):
                pieces.extend(
                    self._split_oversized(
                        sentence
                    )
                )
            else:
                pieces.append(sentence)

        return pieces

    def _run_pipeline(
        self,
        piece: str,
        offset: int,
    ) -> List[Dict]:
        """
        Run the aspect model on one piece of text and
        shift character positions by offset.
        """

        raw_results = self.pipeline(
            piece
        )

        aspects: List[Dict] = []

        for item in raw_results:

            aspect = (
                str(item["word"])
                .strip()
            )

            # Remove invalid predictions such as "."
            if not self._is_valid_aspect(
                aspect
            ):
                continue

            sentiment = (
                self._normalize_label(
                    item["entity_group"]
                )
            )

            aspects.append(
                {
                    "aspect": aspect,

                    "sentiment":
                        sentiment,

                    "confidence": round(
                        float(
                            item["score"]
                        ),
                        4,
                    ),

                    "start": int(
                        item["start"]
                    ) + offset,

                    "end": int(
                        item["end"]
                    ) + offset,
                }
            )

        return aspects

    def predict(
        self,
        text: str,
    ) -> List[Dict]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        per_sentence = (
            self.sentence_level
            or len(text)
            > self.SENTENCE_MODE_MIN_CHARS
        )

        # Short text: one pass, unchanged behavior.
        if not per_sentence:
            return self._run_pipeline(
                text,
                0,
            )

        # Long text (or sentence_level=True):
        # one pass per sentence.
        aspects: List[Dict] = []
        offset = 0

        for piece in self._sentence_pieces(
            text
        ):
            aspects.extend(
                self._run_pipeline(
                    piece,
                    offset,
                )
            )

            # +1 for the space joining pieces in the
            # normalized text.
            offset += len(piece) + 1

        return aspects