import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .records import file_hash


def verify_prediction_archive(path, expected_rows, target, labels, record):
    with np.load(path, allow_pickle=False) as saved:
        rows, true, predicted, probabilities = (saved[key] for key in ["row_index", "true", "predicted", "probabilities"])
        if not np.array_equal(rows, expected_rows) or not np.array_equal(true, target[expected_rows]):
            raise ValueError("Saved prediction rows or targets differ")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Duplicate saved evaluation row")
        if probabilities.shape != (len(rows), len(labels)) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid saved probability shape or values")
        if not np.isin(predicted, labels).all() or np.any(probabilities < 0):
            raise ValueError("Invalid saved prediction label or probability")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(predicted, np.asarray(labels)[np.argmax(probabilities, axis=1)])
        precision, recall, f1, support = precision_recall_fscore_support(true, predicted, labels=labels, zero_division=0)
        for name, value in [("precision", precision), ("recall", recall), ("f1", f1), ("support", support)]:
            np.testing.assert_allclose(record[name], value, atol=1e-14, rtol=0)
        np.testing.assert_allclose(record["macro_f1"], f1_score(true, predicted, labels=labels, average="macro", zero_division=0), atol=1e-14, rtol=0)
        np.testing.assert_allclose(record["balanced_accuracy"], recall.mean(), atol=1e-14, rtol=0)
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        positions = {label: position for position, label in enumerate(labels)}
        for actual, guessed in zip(true, predicted):
            matrix[positions[actual], positions[guessed]] += 1
        np.testing.assert_array_equal(record["confusion"], matrix)
    return {"status": "passed", "rows": len(expected_rows), "archive_sha256": file_hash(path)}


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts.json").read_text())
    for relative, expected in manifest.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Saved artifact mismatch: {relative}")
    return {"status": "passed", "artifacts": len(manifest)}
