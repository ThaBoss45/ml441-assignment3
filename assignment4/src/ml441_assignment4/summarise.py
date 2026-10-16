import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import classification_metrics, paired_fold_differences
from .records import file_hash, write_json
from .verify_results import verify_prediction_archive


def summarise_primary(root, directory, protocol, target, mapping):
    root, directory = Path(root), Path(directory)
    labels, seeds = protocol["data"]["class_order"], protocol["ensemble"]["seeds"]
    path = root / "materials/assignment2_baseline_records/outer_predictions.csv.gz"
    if file_hash(path) != protocol["comparators"]["prediction_sha256"]:
        raise ValueError("Historical prediction provenance differs")
    baseline = pd.read_csv(path)
    comparator = {}
    baseline_summaries = {}
    for model, data in baseline.groupby("model", sort=True):
        data = data.sort_values("row_index")
        np.testing.assert_array_equal(data.row_index, np.arange(len(target)))
        np.testing.assert_array_equal(data.true, target)
        np.testing.assert_array_equal(data.outer_fold, mapping)
        folds = {int(fold): classification_metrics(rows.true.to_numpy(), rows.predicted.to_numpy(), labels) for fold, rows in data.groupby("outer_fold")}
        comparator[model] = {fold: record["macro_f1"] for fold, record in folds.items()}
        baseline_summaries[model] = {"folds": folds, "pooled": classification_metrics(target, data.predicted.to_numpy(), labels)}
    seed_rows, folded, pooled = [], [], {}
    for seed in seeds:
        predicted = np.empty(len(target), dtype=np.int8)
        covered = np.zeros(len(target), dtype=np.int8)
        for fold in range(1, protocol["splits"]["outer_folds"] + 1):
            run = directory / "evaluation" / f"fold_{fold}" / f"seed_{seed}"
            record = json.loads((run / "record.json").read_text())
            rows = np.flatnonzero(mapping == fold)
            verify_prediction_archive(run / "predictions.npz", rows, target, labels, record["metrics"])
            with np.load(run / "predictions.npz", allow_pickle=False) as saved:
                predicted[rows] = saved["predicted"]
                covered[rows] += 1
            seed_rows.append({"fold": fold, "seed": seed, **record["metrics"]})
        if not np.all(covered == 1):
            raise ValueError("Incomplete or duplicated pooled rows")
        pooled[seed] = classification_metrics(target, predicted, labels)
    for fold in range(1, protocol["splits"]["outer_folds"] + 1):
        scores = [record["macro_f1"] for record in seed_rows if record["fold"] == fold]
        folded.append({"fold": fold, "macro_f1_seed_mean": float(np.mean(scores)), "macro_f1_seed_sd": float(np.std(scores, ddof=1))})
    summary = {
        "status": "verified", "evidence_role": protocol["evidence_role"], "folds": folded, "seed_fold_records": seed_rows,
        "macro_f1_fold_mean": float(np.mean([row["macro_f1_seed_mean"] for row in folded])),
        "macro_f1_fold_sd": float(np.std([row["macro_f1_seed_mean"] for row in folded], ddof=1)),
        "pooled_by_seed": pooled, "baselines": baseline_summaries,
        "paired_comparisons": {name: paired_fold_differences(seed_rows, values, seeds) for name, values in comparator.items()},
        "inference": "Descriptive paired folds; no independence or formal significance claim",
    }
    write_json(directory / "comparison_summary.json", summary)
    return summary
