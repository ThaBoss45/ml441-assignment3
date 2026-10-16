import numpy as np
from sklearn.metrics import confusion_matrix


def from_confusion(matrix, labels):
    matrix = np.asarray(matrix, dtype=np.int64)
    labels = np.asarray(labels)
    if matrix.shape != (len(labels), len(labels)) or np.any(matrix < 0):
        raise ValueError("Invalid fixed-label confusion matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(len(labels)), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=support != 0)
    f1 = np.divide(2 * true_positive, support + predicted, out=np.zeros(len(labels)), where=support + predicted != 0)
    normal = np.flatnonzero(labels == 0)
    false_alarm = None
    if len(normal) == 1 and support[normal[0]]:
        false_alarm = float(1 - recall[normal[0]])
    return {
        "labels": labels.tolist(),
        "zero_division": 0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(true_positive.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "normal_false_alarm_rate": false_alarm,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion": matrix.tolist(),
        "row_normalised_confusion": np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0).tolist(),
    }


def classification_metrics(true, predicted, labels):
    true, predicted, labels = np.asarray(true), np.asarray(predicted), np.asarray(labels)
    if true.shape != predicted.shape or not np.isin(true, labels).all() or not np.isin(predicted, labels).all():
        raise ValueError("Prediction alignment or label set is invalid")
    return from_confusion(confusion_matrix(true, predicted, labels=labels), labels)


def paired_fold_differences(ensemble, comparator, seeds):
    keys = [(int(row["fold"]), int(row["seed"])) for row in ensemble]
    folds = sorted(comparator)
    if sorted(keys) != sorted((fold, seed) for fold in folds for seed in seeds):
        raise ValueError("Incomplete or duplicate fold-seed comparison")
    return [
        {"fold": fold, "ensemble_seed_mean_macro_f1": float(np.mean([row["macro_f1"] for row in ensemble if row["fold"] == fold])),
         "comparator_macro_f1": float(comparator[fold]),
         "difference": float(np.mean([row["macro_f1"] for row in ensemble if row["fold"] == fold]) - comparator[fold])}
        for fold in folds
    ]
