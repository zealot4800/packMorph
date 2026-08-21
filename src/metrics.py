from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _pyplot():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-packmorph")
    import matplotlib.pyplot as plt

    return plt


def confusion_matrix(
    truth: np.ndarray, prediction: np.ndarray, class_count: int
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (truth.astype(int), prediction.astype(int)), 1)
    return matrix


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_names: list[str] | np.ndarray,
) -> dict:
    matrix = confusion_matrix(truth, prediction, len(class_names))
    per_class = {}
    f1_values = []
    for index, name in enumerate(class_names):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[str(name)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_csv(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def plot_confusion(
    matrix: np.ndarray, class_names: list[str], title: str, path: Path
) -> None:
    plt = _pyplot()

    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / row_totals
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted service")
    axis.set_ylabel("True service")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Row-normalized fraction")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
