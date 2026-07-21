"""Data loading, prediction, evaluation, and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .models import SentimentAnalyzer


LABEL_ORDER = ["negative", "neutral", "positive"]


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, Any]:
    """Compute classification metrics using a stable label order."""
    report = classification_report(
        actual,
        predicted,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=LABEL_ORDER)

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "labels": LABEL_ORDER,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    analyzer: SentimentAnalyzer,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run sentiment inference and save predictions and metrics."""
    frame = pd.read_csv(input_path)
    required_columns = {"text", "label"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["prediction"] = analyzer.predict(frame["text"].astype(str).tolist())

    metrics = evaluate_predictions(frame["label"], frame["prediction"])
    metrics["model"] = analyzer.name
    metrics["n_observations"] = int(len(frame))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{analyzer.name}_predictions.csv", index=False)
    (output_dir / f"{analyzer.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return frame, metrics
