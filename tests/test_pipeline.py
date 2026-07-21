"""Tests for the end-to-end prediction pipeline."""

from pathlib import Path

import pandas as pd

from financial_sentiment.models import RuleBasedAnalyzer
from financial_sentiment.pipeline import run_pipeline


def test_pipeline_writes_predictions_and_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": [
                "Profit improved.",
                "The company filed a report.",
                "Revenue declined.",
            ],
            "label": ["positive", "neutral", "negative"],
        }
    ).to_csv(input_path, index=False)

    frame, metrics = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "results",
        analyzer=RuleBasedAnalyzer(),
    )

    assert len(frame) == 3
    assert metrics["accuracy"] == 1.0
    assert (tmp_path / "results" / "rule_based_predictions.csv").exists()
    assert (tmp_path / "results" / "rule_based_metrics.json").exists()
