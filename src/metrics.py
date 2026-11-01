"""Classification metrics for held-out flow captures."""
from __future__ import annotations

import numpy as np


def capture_macro_accuracy(labels, predicted, captures, class_count):
    """Give each class equal weight and each capture equal weight within class."""
    if captures is None:
        scores = []
        for label in range(class_count):
            selected = labels == label
            if not np.any(selected):
                raise ValueError('No validation examples for one or more classes')
            scores.append(np.mean(predicted[selected] == label))
        return float(np.mean(scores))
    # Group by class and capture together; capture names need not be globally unique.
    _, capture_ids = np.unique(captures, return_inverse=True)
    groups, inverse = np.unique(np.column_stack((labels, capture_ids)),
                                axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    correct = np.bincount(inverse, weights=predicted == labels)
    class_counts = np.bincount(groups[:, 0], minlength=class_count)
    if np.any(class_counts == 0):
        raise ValueError("No validation examples for one or more classes")
    class_scores = np.bincount(groups[:, 0], weights=correct / counts,
                               minlength=class_count) / class_counts
    return float(class_scores.mean())



def evaluate(model, features, labels, captures, class_names, batch_size):
    predicted = model.predict_probabilities(features, batch_size).argmax(axis=1)
    per_class = []
    for label, name in enumerate(class_names):
        tp = int(np.sum((predicted == label) & (labels == label)))
        fp = int(np.sum((predicted == label) & (labels != label)))
        fn = int(np.sum((predicted != label) & (labels == label)))
        per_class.append(dict(name=name, recall=tp / (tp + fn) if tp + fn else 0.0,
                              f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                              support=int(np.sum(labels == label))))
    return dict(accuracy=float(np.mean(predicted == labels)),
                balanced_accuracy=float(np.mean([c['recall'] for c in per_class])),
                macro_f1=float(np.mean([c['f1'] for c in per_class])),
                capture_macro_accuracy=(None if captures is None else
                    capture_macro_accuracy(labels, predicted, captures, len(class_names))),
                per_class=per_class, flows=len(labels),
                captures=None if captures is None else len(np.unique(captures)))
