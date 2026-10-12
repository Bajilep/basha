
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ..fusion.english_fusion import EnglishFusion
from .aggregator import SentimentAggregator
from .splitter import split_sentences


class LongDocumentAnalyzer:
    """
    Long-document adapter for Basha EnglishFusion.

    Preserves the existing short-text fusion pipeline.
    Analyzes longer documents in token-bounded chunks.
    """

    def __init__(
        self,
        fusion: EnglishFusion | None = None,
        max_tokens: int = 480,
    ) -> None:
        self.fusion = fusion or EnglishFusion()
        self.aggregator = SentimentAggregator()

        if max_tokens < 32 or max_tokens > 480:
            raise ValueError("max_tokens must be between 32 and 480")

        self.max_tokens = max_tokens
        self.tokenizer = self.fusion.sentiment_model.tokenizer

        # Use the smallest supported budget among models
        # that expose a tokenizer.
        tokenizers = [self.fusion.sentiment_model.tokenizer]

        sarcasm_tokenizer = getattr(
            self.fusion.sarcasm_model, "tokenizer", None
        )
        if sarcasm_tokenizer is not None:
            tokenizers.append(sarcasm_tokenizer)

        aspect_pipeline = getattr(
            self.fusion.aspect_model, "pipeline", None
        )
        aspect_tokenizer = getattr(
            aspect_pipeline, "tokenizer", None
        )
        if aspect_tokenizer is not None:
            tokenizers.append(aspect_tokenizer)

        self.tokenizers = tokenizers

    def _fits(self, text: str) -> bool:
        for tokenizer in self.tokenizers:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )
            if len(encoded["input_ids"]) > self.max_tokens:
                return False
        return True

    def _split_oversized_sentence(self, sentence: str) -> List[str]:
        """
        Split unusually long sentences on word boundaries.
        No text is intentionally discarded.
        """
        words = sentence.split()
        pieces = []
        current = []

        for word in words:
            candidate = " ".join(current + [word])

            if current and not self._fits(candidate):
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

            if len(current) == 1 and not self._fits(current[0]):
                raise ValueError(
                    "A single whitespace-delimited item exceeds "
                    "the token budget. Character-level fallback "
                    "is required for this input."
                )

        if current:
            pieces.append(" ".join(current))

        return pieces

    def split(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")

        sentences = split_sentences(text)
        units = []

        for sentence in sentences:
            if self._fits(sentence):
                units.append(sentence)
            else:
                units.extend(self._split_oversized_sentence(sentence))

        chunks = []
        current = []

        for unit in units:
            candidate = " ".join(current + [unit])

            if current and not self._fits(candidate):
                chunks.append(" ".join(current))
                current = [unit]
            else:
                current.append(unit)

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _combine_probabilities(
        results: List[Dict[str, Any]],
        weights: List[int],
    ) -> Dict[str, Any]:
        total_weight = sum(weights)
        probabilities = {}

        for label in ("negative", "neutral", "positive"):
            probabilities[label] = sum(
                float(result["core_sentiment"][label]) * weight
                for result, weight in zip(results, weights)
            ) / total_weight

        label = max(probabilities, key=probabilities.get)

        return {
            "label": label,
            "confidence": round(probabilities[label], 4),
            **{
                key: round(value, 4)
                for key, value in probabilities.items()
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        chunks = self.split(text)

        # Preserve original fusion behavior for short inputs.
        if len(chunks) == 1:
            result = self.fusion.analyze(chunks[0])
            result["longtext"] = {
                "enabled": False,
                "chunk_count": 1,
                "complete": True,
            }
            return result

        chunk_results = []
        weights = []

        for chunk in chunks:
            chunk_results.append(self.fusion.analyze(chunk))
            weights.append(
                len(
                    self.tokenizer(
                        chunk,
                        add_special_tokens=False,
                    )["input_ids"]
                )
            )

        # Preserve every sentence and its chunk origin.
        sentences = []
        aspects = []
        sarcasm_evidence = []

        for index, result in enumerate(chunk_results):
            for sentence in result.get("sentences") or []:
                sentences.append({
                    **sentence,
                    "chunk_index": index,
                })

            for aspect in result.get("aspects") or []:
                # Existing aspect offsets are chunk-local.
                aspects.append({
                    **aspect,
                    "chunk_index": index,
                    "offset_scope": "chunk",
                })

            sarcasm = result.get("sarcasm") or {}
            if sarcasm.get("detected"):
                sarcasm_evidence.append({
                    "chunk_index": index,
                    "score": sarcasm.get("sarcasm", 0.0),
                })

        core = self._combine_probabilities(chunk_results, weights)

        # Reuse Basha's existing sentence-level aggregator.
        sentence_summary = self.aggregator.aggregate(sentences)

        # Retain final fusion labels from each chunk.
        final_labels = [
            result.get("sentiment", {}).get("label", "unknown")
            for result in chunk_results
        ]

        counts = Counter(final_labels)

        # Document-level decision:
        # use the existing aggregator's mixed evidence,
        # plus explicit mixed decisions from chunk fusion.
        strong_positive = sentence_summary[
            "strong_positive_sentences"
        ]
        strong_negative = sentence_summary[
            "strong_negative_sentences"
        ]

        mixed = (
            sentence_summary["label"] == "mixed"
            or (
                strong_positive > 0
                and strong_negative > 0
                and (
                    "mixed" in counts
                    or (
                        "positive" in counts
                        and "negative" in counts
                    )
                )
            )
        )

        if mixed:
            final_label = "mixed"
            decision_source = "longtext_mixed_fusion"
        else:
            final_label = core["label"]
            decision_source = "longtext_aggregated_core"

        # This is an evidence score, not a calibrated
        # document-level probability.
        confidence = (
            sentence_summary["confidence"]
            if mixed and sentence_summary["label"] == "mixed"
            else core["confidence"]
        )

        max_sarcasm = max(
            (
                float(
                    result.get("sarcasm", {}).get("sarcasm", 0.0)
                )
                for result in chunk_results
            ),
            default=0.0,
        )

        return {
            "language": "en",
            "sentiment": {
                "label": final_label,
                "confidence": round(confidence, 4),
            },
            "decision_source": decision_source,
            "core_sentiment": core,
            "aspects": aspects,
            "sentences": sentences,
            "sarcasm": {
                "detected": bool(sarcasm_evidence),
                "sarcasm": round(max_sarcasm, 4),
                "evidence": sarcasm_evidence,
            },
            "mixed_analysis": {
                "mixed": mixed,
                "confidence": round(confidence, 4)
                if mixed else 0.0,
                "reasons": [
                    "long_document_opposing_evidence"
                ] if mixed else [],
            },
            # Whole-document contrast and discourse are not
            # inferred from one arbitrary chunk.
            "contrast": {
                "contrast_detected": any(
                    r.get("contrast", {}).get(
                        "contrast_detected", False
                    )
                    for r in chunk_results
                ),
                "marker": None,
                "label": None,
                "clauses": [],
            },
            "discourse": {
                "resolved": False,
                "label": None,
                "confidence": 0.0,
                "reason": "document_level_resolution_not_implemented",
            },
            "semantic_conflict": {
                "conflict": any(
                    r.get("semantic_conflict", {}).get(
                        "conflict", False
                    )
                    for r in chunk_results
                ),
                "confidence": max(
                    (
                        float(
                            r.get("semantic_conflict", {}).get(
                                "confidence", 0.0
                            )
                        )
                        for r in chunk_results
                    ),
                    default=0.0,
                ),
                "reason": "aggregated_chunk_evidence",
            },
            "sarcasm_resolution": {
                "resolved": False,
                "label": None,
                "confidence": 0.0,
                "reason": "chunk_resolutions_retained_in_longtext",
            },
            "longtext": {
                "enabled": True,
                "complete": True,
                "chunk_count": len(chunks),
                "max_tokens": self.max_tokens,
                "chunk_labels": final_labels,
                "chunk_label_counts": dict(counts),
                "sentence_summary": sentence_summary,
                "chunk_results": chunk_results,
                "confidence_calibrated": False,
            },
        }
