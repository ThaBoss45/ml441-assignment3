from itertools import product

import numpy as np


def structural_candidates(protocol):
    grid = protocol["ensemble"]["structural_grid"]
    candidates = []
    for leaf, features, weight in product(grid["min_samples_leaf"], grid["max_features"], grid["class_weight"]):
        feature_id = "sqrt" if features == "sqrt" else "all"
        weight_id = "none" if weight is None else weight
        candidates.append({
            "candidate_id": f"leaf{leaf:03d}_features_{feature_id}_weight_{weight_id}",
            "setting": {"min_samples_leaf": leaf, "max_features": features, "class_weight": weight},
        })
    return sorted(candidates, key=lambda row: row["candidate_id"])


def select_across_seeds(records, seeds, tolerance=1e-12, count_stage=False):
    if tolerance < 0:
        raise ValueError("Tie tolerance must be nonnegative")
    candidates = sorted(set(row["candidate_id"] for row in records))
    means = []
    for candidate in candidates:
        runs = [row for row in records if row["candidate_id"] == candidate]
        if sorted(row["seed"] for row in runs) != sorted(seeds):
            raise ValueError("Selection needs exactly one record for every fixed seed")
        summary = {"candidate_id": candidate}
        for key in ["macro_f1", "balanced_accuracy", "total_nodes", "trees"]:
            values = [row[key] for row in runs]
            if not np.isfinite(values).all():
                raise ValueError("Selection contains nonfinite evidence")
            summary[key] = float(np.mean(values))
        means.append(summary)
    if not means:
        raise ValueError("Selection has no candidates")
    for key in ["macro_f1", "balanced_accuracy"]:
        best = max(row[key] for row in means)
        means = [row for row in means if best - row[key] <= tolerance]
    cost = "trees" if count_stage else "total_nodes"
    smallest = min(row[cost] for row in means)
    return min((row for row in means if row[cost] == smallest), key=lambda row: row["candidate_id"])
