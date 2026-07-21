"""Sentiment-model adapters used by the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


VALID_LABELS = {"positive", "neutral", "negative"}


class SentimentAnalyzer(Protocol):
    """Interface implemented by all sentiment analyzers."""

    name: str

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Return one sentiment label for every input text."""


@dataclass(frozen=True)
class RuleBasedAnalyzer:
    """Deterministic financial-text baseline with no external model download."""

    name: str = "rule_based"

    positive_terms: tuple[str, ...] = (
        "beat",
        "approved",
        "climbed",
        "expanded",
        "growth",
        "higher profit",
        "improved",
        "large contract",
        "profit",
        "profitable",
        "raised",
        "secured",
        "strong",
        "buyback",
        "debt declined",
    )
    negative_terms: tuple[str, ...] = (
        "cut",
        "declined",
        "defaults",
        "fell",
        "investigation",
        "layoffs",
        "loss",
        "missed",
        "pressure",
        "recall",
        "suspended",
        "weakened",
        "warned",
        "credit losses",
    )

    def predict(self, texts: Sequence[str]) -> list[str]:
        predictions: list[str] = []
        for text in texts:
            lowered = text.lower()
            positive_score = sum(term in lowered for term in self.positive_terms)
            negative_score = sum(term in lowered for term in self.negative_terms)

            if positive_score > negative_score:
                predictions.append("positive")
            elif negative_score > positive_score:
                predictions.append("negative")
            else:
                predictions.append("neutral")
        return predictions


@dataclass(frozen=True)
class VaderAnalyzer:
    """VADER baseline using compound-score thresholds."""

    name: str = "vader"
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05

    def predict(self, texts: Sequence[str]) -> list[str]:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise RuntimeError(
                "VADER is not installed. Run: pip install -r requirements.txt"
            ) from exc

        analyzer = SentimentIntensityAnalyzer()
        predictions: list[str] = []
        for text in texts:
            compound = analyzer.polarity_scores(text)["compound"]
            if compound >= self.positive_threshold:
                predictions.append("positive")
            elif compound <= self.negative_threshold:
                predictions.append("negative")
            else:
                predictions.append("neutral")
        return predictions


@dataclass
class FinBERTAnalyzer:
    """FinBERT adapter backed by Hugging Face Transformers and PyTorch."""

    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    name: str = "finbert"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def predict(self, texts: Sequence[str]) -> list[str]:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT dependencies are not installed. "
                "Run: pip install -r requirements-finbert.txt"
            ) from exc

        classifier = pipeline(
            task="text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=-1,
        )
        outputs = classifier(
            list(texts),
            batch_size=self.batch_size,
            truncation=True,
        )

        predictions: list[str] = []
        for output in outputs:
            label = str(output["label"]).lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected FinBERT label: {label}")
            predictions.append(label)
        return predictions
