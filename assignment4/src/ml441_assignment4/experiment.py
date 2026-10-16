import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

from .data import corrected_groups, load_data
from .metrics import classification_metrics
from .models import diversity_record, make_forest, probability_aggregate, probability_labels, tree_costs
from .preprocessing import forest_matrix, make_preprocessor, representation_record
from .records import MemoryMonitor, array_hash, canonical_hash, code_identity, file_hash, load_checkpoint, save_checkpoint, software_versions, write_json
from .splits import load_verified_splits
from .verify_results import verify_manifest, verify_prediction_archive


def unknown_category_check(preprocessor, fitting_features, nominal):
    changed = fitting_features.iloc[:1].copy()
    encoder = preprocessor.named_transformers_["categorical"]
    sentinel = "__pilot_unseen_category__"
    for column, known in zip(nominal, encoder.categories_):
        if sentinel in known:
            raise ValueError("Synthetic unknown category already occurs in fitting data")
        changed[column] = sentinel
    encoded = forest_matrix(preprocessor.transform(changed))
    block = preprocessor.output_indices_["categorical"]
    if encoded[:, block].nnz:
        raise ValueError("Unknown categories do not map to all-zero nominal coordinates")
    for column, known in zip(nominal, encoder.categories_):
        if set(known) != set(fitting_features[column].unique()):
            raise ValueError("Encoder learned categories outside the fitting partition")
    return {"status": "passed", "unknown_nominal_coordinates_nonzero": encoded[:, block].nnz}


def pilot_run(project_root, workers=4):
    started_utc = datetime.now(timezone.utc).isoformat()
    root = Path(project_root).resolve()
    protocol_path = root / "configs/proposed_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if workers != protocol["budget"]["tree_workers"]:
        raise ValueError("Pilot worker count must match the declared proposal")
    environment = software_versions()
    if environment["packages"]["scikit-learn"] != protocol["comparators"]["historical_environment"]["scikit_learn"]:
        raise ValueError("Estimator version differs from the audited proposal")
    code = code_identity()
    identity = {"phase": "development_pilot", "protocol_sha256": canonical_hash(protocol), "code": code, "environment": environment, "workers": workers}
    output = root / "output/pilot" / canonical_hash(identity)[:16]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity.json", identity)
    write_json(output / "protocol.json", protocol)
    write_json(output / "registration.json", {
        "question": "Can the proposed forest representation, validation, persistence and estimator conventions be implemented reproducibly within the local resource budget?",
        "partition": "Historical outer fold 1, first inner fitting and held-out partitions only",
        "fixed": {"seed": 11, "trees": 8, "min_samples_leaf": 1, "class_weight": None},
        "varied": {"max_features": ["sqrt", 1.0]},
        "role": "Development pilot; excluded from final outer performance evidence",
        "selection_effect": "Feasibility and protocol review only; no structural winner selected by this pilot",
        "limitation": "Eight trees and one seed cannot establish full-grid runtime, selected controls or generalisation",
    })
    run_records = []
    with MemoryMonitor(limit_gib=protocol["budget"]["proposed_process_memory_gib"]) as total_memory:
        features, target = load_data(root, protocol)
        groups = corrected_groups(features)
        partitions = load_verified_splits(root, protocol, groups)
        fitting = partitions["fold_1_inner_train"]
        validation = partitions["fold_1_inner_validation"]
        labels = protocol["data"]["class_order"]
        np.savez_compressed(output / "indices.npz", fitting=fitting, validation=validation)
        with MemoryMonitor() as preprocessing_cost:
            preprocessor = make_preprocessor(features.iloc[fitting], protocol["representation"]["nominal"])
            matrix_fit = forest_matrix(preprocessor.fit_transform(features.iloc[fitting]), fitting=True)
            matrix_validation = forest_matrix(preprocessor.transform(features.iloc[validation]))
        unknown_check = unknown_category_check(preprocessor, features.iloc[fitting], protocol["representation"]["nominal"])
        representation = representation_record(preprocessor)
        representation["fitting_rows_sha256"] = array_hash(fitting)
        representation["matrix_shape_fitting"] = list(matrix_fit.shape)
        representation["matrix_shape_validation"] = list(matrix_validation.shape)
        representation["preprocessing_cost"] = preprocessing_cost.record()
        write_json(output / "representation.json", representation)
        specifications = [
            ("sqrt", {"min_samples_leaf": 1, "class_weight": None, "max_features": "sqrt"}),
            ("all", {"min_samples_leaf": 1, "class_weight": None, "max_features": 1.0}),
            ("sqrt_repeat", {"min_samples_leaf": 1, "class_weight": None, "max_features": "sqrt"}),
        ]
        probabilities_by_name = {}
        bags_by_name = {}
        predictions_by_name = {}
        for name, setting in specifications:
            print(f"Pilot {name}: 8 trees, {len(fitting)} fitting rows", flush=True)
            run_identity = {
                **identity, "run_name": name, "seed": 11, "trees": 8, "setting": setting,
                "data_sha256": protocol["data"]["sha256"],
                "split_sha256": protocol["splits"]["artifact_sha256"],
                "fitting_index_sha256": array_hash(fitting), "validation_index_sha256": array_hash(validation),
                "class_order": labels, "representation_sha256": canonical_hash({key: value for key, value in representation.items() if key != "preprocessing_cost"}),
            }
            directory = output / name
            directory.mkdir(exist_ok=True)
            record_path = directory / "record.json"
            resumed = record_path.exists() and (directory / "complete.json").exists()
            if resumed:
                payload = load_checkpoint(directory, run_identity)
                forest = payload["forest"]
                record = json.loads(record_path.read_text())
                if record["identity"] != run_identity:
                    raise ValueError("Saved run record identity differs")
            else:
                run_started_utc = datetime.now(timezone.utc).isoformat()
                forest = make_forest(protocol, setting, seed=11, trees=8, workers=workers)
                with threadpool_limits(limits=1), MemoryMonitor() as fit_cost:
                    forest.fit(matrix_fit, target[fitting])
                forest.set_params(n_jobs=1)
                with threadpool_limits(limits=1), MemoryMonitor() as predict_cost:
                    probabilities = forest.predict_proba(matrix_validation)
                    predicted = probability_labels(probabilities, labels)
                model_path = save_checkpoint(directory, run_identity, {"forest": forest, "preprocessor": preprocessor})
                joblib.dump(forest, directory / "forest.joblib", compress=0)
                record = {
                    "started_utc": run_started_utc,
                    "identity": run_identity, "fit_cost": fit_cost.record(), "prediction_cost": predict_cost.record(),
                    "prediction_timing_scope": "predict_proba plus class argmax; preprocessing and diversity diagnostics separately measured",
                    "fit_tree_workers": workers, "prediction_tree_workers": 1,
                    "checkpoint_bytes": model_path.stat().st_size,
                    "forest_serialised_bytes": (directory / "forest.joblib").stat().st_size,
                    "forest_parameters": forest.get_params(deep=False), "classes": forest.classes_.tolist(),
                    "costs": tree_costs(forest),
                }
            with threadpool_limits(limits=1):
                probabilities, hard = probability_aggregate(forest, matrix_validation, labels)
                library_probability = forest.predict_proba(matrix_validation)
            np.testing.assert_allclose(probabilities, library_probability, atol=1e-14, rtol=0)
            predicted = probability_labels(probabilities, labels)
            np.testing.assert_array_equal(predicted, forest.classes_[np.argmax(library_probability, axis=1)])
            diversity, hard_vote = diversity_record(hard, labels)
            record["aggregation_check"] = {
                "max_absolute_error_vs_library": float(np.max(np.abs(probabilities - library_probability))),
                "hard_vote_disagreements": int(np.sum(predicted != hard_vote)),
                "saved_aggregation": "Member-order aligned mean probabilities; class labels identical to library prediction",
            }
            record["diversity"] = diversity
            record["metrics"] = classification_metrics(target[validation], predicted, labels)
            bags = forest.estimators_samples_
            if not all(len(bag) == len(fitting) and bag.min() >= 0 and bag.max() < len(fitting) for bag in bags):
                raise ValueError("Bootstrap draw does not use the declared fitting row positions and size")
            if not all(len(np.unique(bag)) < len(bag) for bag in bags):
                raise ValueError("Bootstrap draws unexpectedly lack replacement duplicates")
            record["bootstrap"] = {
                "unit": "Fitting-partition row position, not corrected-vector group",
                "draws_per_tree": [len(bag) for bag in bags],
                "distinct_rows_per_tree": [len(np.unique(bag)) for bag in bags],
                "draw_hashes": [array_hash(bag) for bag in bags],
            }
            restored = load_checkpoint(directory, run_identity)
            restored_probability, _ = probability_aggregate(restored["forest"], matrix_validation, labels)
            np.testing.assert_array_equal(restored_probability, probabilities)
            incompatible = copy.deepcopy(run_identity)
            incompatible["seed"] = 23
            try:
                load_checkpoint(directory, incompatible)
            except ValueError:
                record["incompatible_checkpoint_rejected"] = True
            else:
                raise ValueError("Incompatible checkpoint was accepted")
            np.savez_compressed(directory / "predictions.npz", row_index=validation, true=target[validation], predicted=predicted, probabilities=probabilities, labels=np.asarray(labels))
            record["saved_record_verification"] = verify_prediction_archive(directory / "predictions.npz", validation, target, labels, record["metrics"])
            if not resumed:
                record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(record_path, record)
            run_records.append({"name": name, **record})
            probabilities_by_name[name] = probabilities
            predictions_by_name[name] = predicted
            bags_by_name[name] = record["bootstrap"]["draw_hashes"]
            del forest, restored
        np.testing.assert_array_equal(probabilities_by_name["sqrt"], probabilities_by_name["sqrt_repeat"])
        np.testing.assert_array_equal(predictions_by_name["sqrt"], predictions_by_name["sqrt_repeat"])
        if bags_by_name["sqrt"] != bags_by_name["all"] or bags_by_name["sqrt"] != bags_by_name["sqrt_repeat"]:
            raise ValueError("Feature-sampling contrast or repetition used different bootstrap draws")
    summary = {
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "phase": "development_pilot", "identity": identity,
        "fitting_rows": len(fitting), "validation_rows": len(validation), "encoded_features": matrix_fit.shape[1],
        "outer_rows_evaluated": 0, "unknown_category_check": unknown_check,
        "exact_repeat_probabilities_and_predictions": True,
        "identical_bootstrap_draws_across_feature_settings": True,
        "total_cost": total_memory.record(), "runs": run_records,
        "limitations": ["One development partition and seed; eight trees per fit", "Sampled process RSS may miss peaks", "Full-size forests may have nonlinear runtime and memory costs", "Weighted full-data costs are not measured by this pilot"],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifacts.json", {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifacts.json"})
    summary_verification = verify_manifest(output)
    write_json(root / "output/pilot/latest.json", {"directory": str(output.relative_to(root)), "summary_sha256": file_hash(output / "summary.json"), "verification": summary_verification})
    print(json.dumps({"status": "passed", "output": str(output), "total_seconds": total_memory.elapsed}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "verify-pilot"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "pilot":
        pilot_run(args.project_root, args.workers)
    else:
        latest = json.loads((args.project_root / "output/pilot/latest.json").read_text())
        directory = args.project_root / latest["directory"]
        if file_hash(directory / "summary.json") != latest["summary_sha256"]:
            raise ValueError("Pilot summary differs from the verified latest record")
        print(json.dumps(verify_manifest(directory)))


if __name__ == "__main__":
    main()
