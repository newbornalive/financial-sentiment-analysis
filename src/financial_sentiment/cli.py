"""Command-line interface for the financial sentiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import FinBERTAnalyzer, RuleBasedAnalyzer, VaderAnalyzer
from .pipeline import run_pipeline
from .visualization import save_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run financial-text sentiment classification."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_financial_text.csv"),
        help="CSV file containing text and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for predictions, metrics, and plots.",
    )
    parser.add_argument(
        "--model",
        choices=("rule", "vader", "finbert"),
        default="rule",
        help="Sentiment model to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="FinBERT inference batch size.",
    )
    return parser


def main() -> None:
    """Execute the command-line workflow."""
    args = build_parser().parse_args()

    if args.model == "rule":
        analyzer = RuleBasedAnalyzer()
    elif args.model == "vader":
        analyzer = VaderAnalyzer()
    else:
        analyzer = FinBERTAnalyzer(batch_size=args.batch_size)

    _, metrics = run_pipeline(args.input, args.output_dir, analyzer)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["labels"],
        args.output_dir / f"{analyzer.name}_confusion_matrix.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
