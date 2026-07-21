"""Tests for deterministic sentiment-model behaviour."""

from financial_sentiment.models import FinBERTAnalyzer, RuleBasedAnalyzer


def test_rule_based_analyzer_detects_each_class() -> None:
    analyzer = RuleBasedAnalyzer()
    predictions = analyzer.predict(
        [
            "Profit improved and guidance was raised.",
            "The company filed its annual report.",
            "Revenue declined and management cut its outlook.",
        ]
    )
    assert predictions == ["positive", "neutral", "negative"]


def test_finbert_rejects_invalid_batch_size() -> None:
    try:
        FinBERTAnalyzer(batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive batch size")
