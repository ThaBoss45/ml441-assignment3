import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "verification"
OLD = ROOT.parent / "assignment2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def metrics(matrix):
    matrix = matrix.astype(float)
    tp = matrix.diagonal()
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros(10), where=actual != 0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(10), where=actual + predicted != 0)
    n = matrix.sum()
    correct = tp.sum()
    expected = actual @ predicted / n ** 2
    denominator = np.sqrt((n ** 2 - predicted @ predicted) * (n ** 2 - actual @ actual))
    scores = {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(correct / n),
        "weighted_f1": float(f1 @ actual / n),
        "kappa": float((correct / n - expected) / (1 - expected)),
        "mcc": float((correct * n - actual @ predicted) / denominator) if denominator else 0.0,
    }
    return scores, precision, recall, f1, actual


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "materials/preparation_manifest.json").read_text())
    hash_checks = []
    entries = manifest.get("files", manifest.get("resources", []))
    if not entries:
        entries = next(v for v in manifest.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "source" in v[0])
    for entry in entries:
        for role, name in [("copy", entry.get("local_path")), ("original", entry.get("source"))]:
            if not name or name.startswith("https://"):
                continue
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            exists = path.is_file()
            actual = digest(path) if exists else None
            matched = exists and actual == entry["sha256"]
            hash_checks.append({"role": role, "path": str(path), "exists": exists, "sha256": actual, "matches_manifest": matched})
            assert matched, str(path)
    raw_path = ROOT / "data/raw/networkTraffic.csv"
    assert digest(raw_path) == "acde2c646b55c9c6907ade03b1ec985bf58f53926c6045ae221a61ab17535c3b"
    raw = pd.read_csv(raw_path, na_values=["?"], low_memory=False)
    assert raw.shape == (257673, 44)
    y = pd.to_numeric(raw["attack_cat"], errors="raise").astype("int8").to_numpy()
    assert set(y) == set(range(10))
    categorical = ["proto", "service", "state"]
    X = raw.drop(columns=["attack_cat", "id", "is_ftp_login", "tcprtt"]).copy()
    for col in categorical:
        X[col] = X[col].astype("string").fillna("Missing")
    numeric = [col for col in X if col not in categorical]
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="raise")
    assert np.isfinite(X[numeric].to_numpy()).all()
    hashes = pd.util.hash_pandas_object(X, index=False, categorize=True).to_numpy()
    groups, _ = pd.factorize(hashes, sort=False)
    unique = X.assign(audit_group=groups).drop_duplicates()
    assert not unique["audit_group"].duplicated().any(), "Hash collision between unequal corrected vectors"
    assert len(unique) == len(X.drop_duplicates())
    X32 = X.copy()
    X32[numeric] = X32[numeric].astype("float32")
    hashes32 = pd.util.hash_pandas_object(X32, index=False, categorize=True).to_numpy()
    assert len(X32.drop_duplicates()) == len(unique)
    group_counts = pd.Series(groups).value_counts()
    target_groups = pd.DataFrame({"group": groups, "target": y})
    conflicting = target_groups.groupby("group")["target"].nunique().gt(1)
    timing_error = (raw["tcprtt"] - raw["synack"] - raw["ackdat"]).abs()
    predictions = pd.read_csv(ROOT / "materials/assignment2_baseline_records/outer_predictions.csv.gz")
    models = sorted(predictions.model.unique())
    assert models == ["Classification tree", "kNN"]
    sorted_models = {}
    for model in models:
        rows = predictions[predictions.model == model].sort_values("row_index")
        assert np.array_equal(rows.row_index.to_numpy(), np.arange(len(y)))
        assert np.array_equal(rows.true.to_numpy(), y)
        assert rows.predicted.isin(range(10)).all()
        assert rows.outer_fold.isin(range(1, 6)).all()
        sorted_models[model] = rows
    mapping = sorted_models[models[0]].outer_fold.to_numpy()
    assert np.array_equal(mapping, sorted_models[models[1]].outer_fold.to_numpy())
    assert pd.DataFrame({"group": groups, "fold": mapping}).groupby("group").fold.nunique().eq(1).all()
    assert pd.DataFrame({"group": hashes32, "fold": mapping}).groupby("group").fold.nunique().eq(1).all()
    selected = json.loads((ROOT / "materials/assignment2_baseline_records/selected_parameters.json").read_text())
    tuning_path = OLD / "output/results/tuning_results.csv"
    tuning = pd.read_csv(tuning_path)
    tuning.to_csv(OUT / "historical_tuning_records.csv", index=False)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260826)
    split_records = []
    arrays = {"outer_fold": mapping, "corrected_group": groups}
    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        assert np.array_equal(test, np.flatnonzero(mapping == fold)), f"Outer reconstruction {fold}"
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260826 + fold)
        fit_rel, validation_rel = next(inner.split(X.iloc[train], y[train], groups[train]))
        fit, validation = train[fit_rel], train[validation_rel]
        assert not np.intersect1d(groups[fit], groups[validation]).size
        params = selected[fold - 1]
        for name, values in [("outer_train", train), ("outer_test", test), ("inner_train", fit), ("inner_validation", validation)]:
            assert params[name + "_size"] == len(values)
            arrays[f"fold_{fold}_{name}"] = values
        knn = tuning[(tuning.outer_fold == fold) & (tuning.model == "kNN")]
        tree = tuning[(tuning.outer_fold == fold) & (tuning.model == "Classification tree")]
        assert len(knn) == 56 and len(tree) == 39
        best_knn = max(knn.to_dict("records"), key=lambda r: (r["macro_f1"], r["balanced_accuracy"], -r["k"], -r["vote_exponent"]))
        for key in ["k", "p", "weights", "vote_exponent"]:
            assert params["knn"][key] == best_knn[key]
        coarse = tree[tree.phase == "complexity_grid"]
        best_coarse = max(coarse.to_dict("records"), key=lambda r: (r["macro_f1"], r["balanced_accuracy"], -r["node_count"]))
        refine = tree[tree.phase == "pruning_refinement"]
        assert len(coarse) == 36 and len(refine) == 3
        for key in ["criterion", "max_depth", "min_samples_leaf", "class_weight"]:
            values = refine[key].fillna("None").tolist()
            base = best_coarse[key] if pd.notna(best_coarse[key]) else "None"
            assert all(value == base for value in values)
        best_tree = max(tree.to_dict("records"), key=lambda r: (r["macro_f1"], r["balanced_accuracy"], -r["node_count"]))
        for key in ["criterion", "max_depth", "min_samples_leaf", "class_weight", "ccp_alpha"]:
            actual = best_tree[key] if pd.notna(best_tree[key]) else None
            assert params["classification_tree"][key] == actual
        split_records.append({
            "outer_fold": fold,
            "hashes": {name: array_digest(arrays[f"fold_{fold}_{name}"]) for name in ["outer_train", "outer_test", "inner_train", "inner_validation"]},
            "sizes": {name: len(arrays[f"fold_{fold}_{name}"]) for name in ["outer_train", "outer_test", "inner_train", "inner_validation"]},
            "class_support": {name: np.bincount(y[arrays[f"fold_{fold}_{name}"]], minlength=10).tolist() for name in ["outer_train", "outer_test", "inner_train", "inner_validation"]},
            "historical_seeds": {"inner_split_and_tree": 20260826 + fold, "inner_neighbour_sampling": 20260926 + fold, "outer_neighbour_sampling": 20261026 + fold, "neighbour_training_diagnostic": 20261126 + fold},
            "historical_candidates": {"neighbour": 56, "tree_coarse": 36, "tree_refinement": 3},
            "selection_matches_saved_tuning": True,
        })
    np.savez_compressed(OUT / "historical_splits.npz", **arrays)
    pd.DataFrame({"row_index": np.arange(len(y)), "outer_fold": mapping, "corrected_group": groups}).to_csv(OUT / "historical_row_mapping.csv.gz", index=False)
    saved_metrics = pd.read_csv(ROOT / "materials/assignment2_baseline_records/fold_metrics.csv")
    saved_class = pd.read_csv(ROOT / "materials/assignment2_baseline_records/per_class_metrics.csv")
    metric_records = []
    pooled = {}
    max_error = 0.0
    for model in models:
        rows = sorted_models[model]
        for fold in range(1, 6):
            frame = rows[rows.outer_fold == fold]
            matrix = np.bincount(10 * frame.true.to_numpy() + frame.predicted.to_numpy(), minlength=100).reshape(10, 10)
            scores, *_ = metrics(matrix)
            saved = saved_metrics[(saved_metrics.model == model) & (saved_metrics.outer_fold == fold) & (saved_metrics.split == "generalisation")].iloc[0]
            for key, value in scores.items():
                error = abs(saved[key] - value)
                max_error = max(max_error, error)
                assert error < 1e-12, (model, fold, key, error)
            metric_records.append({"model": model, "outer_fold": fold, **scores})
        matrix = np.bincount(10 * rows.true.to_numpy() + rows.predicted.to_numpy(), minlength=100).reshape(10, 10)
        scores, precision, recall, f1, support = metrics(matrix)
        pooled[model] = scores
        for class_id in range(10):
            saved = saved_class[(saved_class.model == model) & (saved_class.class_id == class_id)].iloc[0]
            assert saved.support == support[class_id]
            for key, value in [("precision", precision[class_id]), ("recall", recall[class_id]), ("f1", f1[class_id])]:
                error = abs(saved[key] - value)
                max_error = max(max_error, error)
                assert error < 1e-12
    metrics_frame = pd.DataFrame(metric_records)
    metrics_frame.to_csv(OUT / "historical_metrics_recalculated.csv", index=False)
    saved_summary = pd.read_csv(ROOT / "materials/assignment2_baseline_records/summary_metrics.csv")
    fold_summaries = {}
    for model in models:
        scores = metrics_frame[metrics_frame.model == model]
        saved = saved_summary[(saved_summary.model == model) & (saved_summary.split == "generalisation")].iloc[0]
        fold_summaries[model] = {}
        for key in ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "kappa", "mcc"]:
            mean, sd = scores[key].mean(), scores[key].std(ddof=1)
            assert abs(saved[key + "_mean"] - mean) < 1e-12
            assert abs(saved[key + "_std"] - sd) < 1e-12
            fold_summaries[model][key] = {"mean": float(mean), "sample_sd": float(sd)}
    paired = metrics_frame.pivot(index="outer_fold", columns="model", values="macro_f1")
    differences = paired["Classification tree"] - paired["kNN"]
    saved_paired = json.loads((ROOT / "materials/assignment2_baseline_records/paired_comparison.json").read_text())
    for fold, value in differences.items():
        assert abs(saved_paired["paired_differences"][str(fold)] - value) < 1e-12
    assert abs(saved_paired["median_difference_model_1_minus_model_2"] - differences.median()) < 1e-12
    evidence = {
        "role": "Independent preparation provenance audit; no model fitting or selection",
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "manifest_checks": hash_checks,
        "new_evidence_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [OUT / "historical_splits.npz", OUT / "historical_row_mapping.csv.gz", OUT / "historical_metrics_recalculated.csv", OUT / "historical_tuning_records.csv", ROOT / "src/audit_preparation.py"]},
        "historical_tuning_source": {"path": str(tuning_path), "sha256": digest(tuning_path)},
        "data": {"raw_sha256": digest(raw_path), "shape": list(raw.shape), "corrected_predictors": len(X.columns), "numeric_predictors": len(numeric), "categorical_levels": {col: int(X[col].nunique()) for col in categorical}, "unique_vectors": len(unique), "repeated_groups": int(group_counts.gt(1).sum()), "rows_in_repeated_groups": int(group_counts[group_counts.gt(1)].sum()), "conflicting_label_groups": int(conflicting.sum()), "exact_equality_within_every_hash_group": True, "maximum_timing_identity_absolute_error": float(timing_error.max()), "timing_identity_allclose_atol_1e_6_rtol_0": bool(np.allclose(raw.tcprtt, raw.synack + raw.ackdat, atol=1e-6, rtol=0)), "ftp_login_command_equal_rows": int(raw.is_ftp_login.eq(raw.ct_ftp_cmd).sum()), "invalid_ftp_login": int((~raw.is_ftp_login.isin([0, 1])).sum()), "missing_service": int(raw.service.isna().sum()), "all_corrected_numeric_finite": True, "deterministic_accuracy_ceiling_for_corrected_vectors": float(target_groups.value_counts(["group", "target"]).groupby(level="group").max().sum() / len(y))},
        "splits": split_records,
        "baseline_audit": {"prediction_rows": len(predictions), "coverage_and_targets_and_fold_mapping_match": True, "fold_membership_reconstructed_exactly": True, "maximum_metric_absolute_difference": max_error, "fold_summary": fold_summaries, "pooled_metrics": pooled, "reuse_recommendation": "Reuse both historical prediction sets for the matched retrospective benchmark; refresh fitting only for separately registered same-environment cost evidence or a disclosed revised comparator.", "limitations": ["Source and record hashes support provenance but are not signed execution attestations.", "Historical checkpoint acceptance checked existence only, not configuration identity.", "Historical fit timing boundaries differ between classifiers.", "Assignment 2 outer scores were inspected historically."]},
        "status": "passed",
    }
    evidence["data"]["float32_conversion_introduces_new_equal_vectors"] = False
    evidence["data"]["float32_groups_spanning_outer_folds"] = 0
    evidence["data"]["class_counts"] = np.bincount(y, minlength=10).tolist()
    evidence["data"]["class_map"] = pd.read_csv(ROOT / "data/raw/attack_category_map.csv").to_dict("records")
    evidence["data"]["service_missing_by_class"] = {str(k): float(raw.loc[y == k, "service"].isna().mean()) for k in range(10)}
    evidence["baseline_audit"]["paired_differences"] = {str(k): float(v) for k, v in differences.items()}
    evidence["baseline_audit"]["median_paired_difference"] = float(differences.median())
    (OUT / "gate1_provenance_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "manifest_checks": len(hash_checks), "data": evidence["data"], "baseline_audit": evidence["baseline_audit"], "splits": split_records}, indent=2))


if __name__ == "__main__":
    main()
