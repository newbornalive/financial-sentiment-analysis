"""Visualisation helpers for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    output_path: Path,
) -> None:
    """Save a labelled confusion-matrix figure."""
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values)
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Sentiment Classification Confusion Matrix",
    )

    threshold = values.max() / 2 if values.size else 0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(values[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] > threshold else "black",
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
