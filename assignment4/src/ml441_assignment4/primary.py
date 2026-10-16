import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import corrected_groups, load_data
from .metrics import classification_metrics
from .models import diversity_record, make_forest, probability_aggregate, probability_labels, tree_costs
from .preprocessing import forest_matrix, make_preprocessor, representation_record
from .records import MemoryMonitor, array_hash, canonical_hash, code_identity, file_hash, load_checkpoint, save_checkpoint, software_versions, write_json
from .search import select_across_seeds, structural_candidates
from .splits import load_verified_splits
from .verify_results import verify_prediction_archive


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_frozen_protocol(root, protocol):
    if protocol.get("status") != "frozen" or not protocol.get("full_primary_run_authorised_by_this_file"):
        raise ValueError("Primary execution requires a frozen protocol")
    expected = (Path(root) / "configs/primary_protocol.sha256").read_text().strip()
    if canonical_hash(protocol) != expected:
        raise ValueError("Frozen protocol seal differs")
    if protocol["implementation_code"] != code_identity():
        raise ValueError("Current implementation differs from the frozen code")
    if protocol["execution_environment"] != software_versions():
        raise ValueError("Current environment differs from the frozen environment")
    for item in protocol["gate_evidence"]:
        if file_hash(Path(root) / item["path"]) != item["sha256"]:
            raise ValueError("Protocol gate evidence has changed")


def manifest_write(directory, names):
    write_json(Path(directory) / "artifacts.json", {name: file_hash(Path(directory) / name) for name in names})


def manifest_check(directory):
    directory = Path(directory)
    values = json.loads((directory / "artifacts.json").read_text())
    for name, expected in values.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError("Completed run artifact differs")


class Study:
    def __init__(self, root, protocol, features, target, partitions, output, role="primary"):
        self.root = Path(root).resolve()
        self.protocol = protocol
        self.features = features
        self.target = target
        self.partitions = partitions
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.labels = protocol["data"]["class_order"]
        self.seeds = protocol["ensemble"]["seeds"]
        self.folds = list(range(1, protocol["splits"]["outer_folds"] + 1))
        self.base_identity = {"protocol_sha256": canonical_hash(protocol), "code": code_identity(), "environment": software_versions(), "role": role}
        self.completed = []
        self.started = time.monotonic()

    def progress(self, stage, **details):
        write_json(self.output / "progress.json", {"updated_utc": utc_now(), "stage": stage, "completed_fit_records_this_process": len(self.completed), **details})

    def prepare(self, fitting, evaluation, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        resume = (directory / "preprocessing_manifest.json").exists()
        if resume:
            manifest = json.loads((directory / "preprocessing_manifest.json").read_text())
            for name, expected in manifest.items():
                if file_hash(directory / name) != expected:
                    raise ValueError("Saved preprocessing artifact differs")
            representation = json.loads((directory / "representation.json").read_text())
            if representation["fitting_index_sha256"] != array_hash(fitting):
                raise ValueError("Saved encoder fitting scope differs")
            with np.load(directory / "indices.npz", allow_pickle=False) as indices:
                np.testing.assert_array_equal(indices["evaluation"], evaluation)
            preprocessor = joblib.load(directory / "preprocessor.joblib")
        with MemoryMonitor() as cost:
            if resume:
                matrix_fit = forest_matrix(preprocessor.transform(self.features.iloc[fitting]), fitting=True)
            else:
                preprocessor = make_preprocessor(self.features.iloc[fitting], self.protocol["representation"]["nominal"])
                matrix_fit = forest_matrix(preprocessor.fit_transform(self.features.iloc[fitting]), fitting=True)
            matrix_evaluation = forest_matrix(preprocessor.transform(self.features.iloc[evaluation]))
        if resume:
            write_json(directory / "preprocessing_attempts" / (str(time.time_ns()) + ".json"), {"completed_utc": utc_now(), "resumed_transform_only": True, **cost.record()})
            return matrix_fit, matrix_evaluation, representation
        representation = representation_record(preprocessor)
        representation["fitting_index_sha256"] = array_hash(fitting)
        write_json(directory / "representation.json", representation)
        np.savez_compressed(directory / "indices.npz", fitting=fitting, evaluation=evaluation)
        joblib.dump(preprocessor, directory / "preprocessor.joblib", compress=0)
        write_json(directory / "preprocessing_cost.json", {"completed_utc": utc_now(), **cost.record()})
        write_json(directory / "preprocessing_manifest.json", {name: file_hash(directory / name) for name in ["representation.json", "indices.npz", "preprocessor.joblib", "preprocessing_cost.json"]})
        return matrix_fit, matrix_evaluation, representation

    def run_identity(self, fold, candidate_id, setting, seed, trees, fitting, evaluation, representation, stage):
        return {**self.base_identity, "fold": fold, "candidate_id": candidate_id, "setting": setting,
                "seed": seed, "trees": trees, "stage": stage, "fitting_sha256": array_hash(fitting),
                "evaluation_sha256": array_hash(evaluation), "representation_sha256": canonical_hash(representation),
                "class_order": self.labels}

    def completed_record(self, directory, identity, evaluation):
        directory = Path(directory)
        if not (directory / "artifacts.json").exists():
            return None
        manifest_check(directory)
        record = json.loads((directory / "record.json").read_text())
        if record["identity"] != identity:
            raise ValueError("Completed run identity differs")
        verify_prediction_archive(directory / "predictions.npz", evaluation, self.target, self.labels, record["metrics"])
        return record

    def check_budget(self):
        records = [json.loads(path.read_text()) for section in ["selection", "evaluation"] for path in (self.output / section).rglob("record.json")]
        used = sum(record["fit_cost"]["elapsed_seconds"] for record in records if record["fit_cost"] is not None)
        if used >= self.protocol["budget"]["maximum_fit_hours"] * 3600:
            raise RuntimeError("Declared fitting budget reached at a compatible checkpoint")

    def fit(self, directory, identity, matrices, fitting, evaluation, previous=None):
        directory = Path(directory)
        completed = self.completed_record(directory, identity, evaluation)
        if completed is not None:
            self.completed.append(completed)
            return completed
        self.check_budget()
        self.progress(identity["stage"], fold=identity["fold"], candidate_id=identity["candidate_id"], seed=identity["seed"], trees=identity["trees"])
        print(json.dumps({"event": "fit_start", "utc": utc_now(), "fold": identity["fold"], "stage": identity["stage"], "candidate": identity["candidate_id"], "seed": identity["seed"], "trees": identity["trees"]}), flush=True)
        started = utc_now()
        forest = previous
        previous_count = len(forest.estimators_) if forest is not None else 0
        if forest is None:
            forest = make_forest(self.protocol, identity["setting"], identity["seed"], identity["trees"], workers=4)
        else:
            forest.set_params(n_estimators=identity["trees"], warm_start=True, n_jobs=4)
        with threadpool_limits(limits=1), MemoryMonitor() as fit_cost:
            forest.fit(matrices[0], self.target[fitting])
        forest.set_params(n_jobs=1)
        record = self.persist(directory, identity, forest, matrices[1], fitting, evaluation, started, fit_cost.record(), identity["trees"] - previous_count)
        self.completed.append(record)
        del forest
        gc.collect()
        return record

    def persist(self, directory, identity, forest, matrix_evaluation, fitting, evaluation, started, fit_cost, new_trees):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=1), MemoryMonitor() as prediction_cost:
            library = forest.predict_proba(matrix_evaluation)
            predicted = probability_labels(library, self.labels)
        with threadpool_limits(limits=1), MemoryMonitor() as diagnostic_cost:
            probabilities, hard = probability_aggregate(forest, matrix_evaluation, self.labels)
        np.testing.assert_array_equal(library, probabilities)
        metrics = classification_metrics(self.target[evaluation], predicted, self.labels)
        costs = tree_costs(forest)
        bags = forest.estimators_samples_
        bootstrap = {"draw_sha256": [array_hash(bag) for bag in bags], "draw_sizes": [len(bag) for bag in bags]}
        if any(len(bag) != len(fitting) for bag in bags):
            raise ValueError("Bootstrap size differs from fitting partition")
        record = {"identity": identity, "started_utc": started, "fit_cost": fit_cost,
                  "prediction_cost": prediction_cost.record(), "diagnostic_cost": diagnostic_cost.record(),
                  "new_trees": new_trees, "metrics": metrics, "costs": costs,
                  "bootstrap": bootstrap, "forest_parameters": forest.get_params(deep=False),
                  "fit_tree_workers": 4, "prediction_tree_workers": 1,
                  "prediction_timing_scope": "Encoded-matrix predict_proba plus class argmax; preprocessing and member diagnostics separate"}
        with MemoryMonitor() as serialisation_cost:
            path = save_checkpoint(directory, identity, {"forest": forest})
        record["serialisation_cost"] = serialisation_cost.record()
        record["checkpoint_bytes"] = path.stat().st_size
        record["checkpoint_size_scope"] = "Uncompressed estimator checkpoint including identity wrapper; encoder saved once per fitting partition"
        np.savez_compressed(directory / "predictions.npz", row_index=evaluation, true=self.target[evaluation], predicted=predicted, probabilities=probabilities, labels=np.asarray(self.labels))
        if identity["stage"] != "outer":
            diversity, vote = diversity_record(hard, self.labels)
            errors = (hard != self.target[evaluation, None]).sum(axis=1)
            pairs = hard.shape[1] * (hard.shape[1] - 1) / 2
            diversity["mean_member_error"] = float(np.mean(hard != self.target[evaluation, None]))
            diversity["mean_pairwise_simultaneous_error"] = float(np.mean(errors * (errors - 1) / (2 * pairs))) if pairs else 0.0
            diversity["hard_vote_disagreements"] = int(np.sum(vote != predicted))
            record["diversity"] = diversity
            np.savez_compressed(directory / "members.npz", row_index=evaluation, hard_predictions=hard.astype(np.int8))
            supplementary = self.root / "output/supplementary" / canonical_hash(self.base_identity)[:16] / directory.relative_to(self.output)
            if self.role != "primary":
                supplementary = self.output / "supplementary" / directory.relative_to(self.output)
            write_json(supplementary / "diversity.json", {"role": "Predeclared inner mechanism diagnostic, no outer selection effect", "identity": identity, **diversity})
        record["completed_utc"] = utc_now()
        verify_prediction_archive(directory / "predictions.npz", evaluation, self.target, self.labels, metrics)
        write_json(directory / "record.json", record)
        names = ["record.json", "predictions.npz", "complete.json", "model.joblib"]
        if identity["stage"] != "outer":
            names.append("members.npz")
        manifest_write(directory, names)
        return record

    def summary_row(self, record, candidate_id=None):
        return {"candidate_id": candidate_id or record["identity"]["candidate_id"], "seed": record["identity"]["seed"],
                "macro_f1": record["metrics"]["macro_f1"], "balanced_accuracy": record["metrics"]["balanced_accuracy"],
                "total_nodes": record["costs"]["total_nodes"], "trees": record["identity"]["trees"]}

    def select(self):
        candidates = structural_candidates(self.protocol)
        counts = self.protocol["selection"]["tree_count_stage"]["counts"]
        structural_count = self.protocol["ensemble"]["structural_n_estimators"]
        selected = []
        evidence = []
        for fold in self.folds:
            fitting = self.partitions[f"fold_{fold}_inner_train"]
            validation = self.partitions[f"fold_{fold}_inner_validation"]
            fold_root = self.output / "selection" / f"fold_{fold}"
            fit_matrix, validation_matrix, representation = self.prepare(fitting, validation, fold_root)
            all_records = []
            locations = {}
            for candidate in candidates:
                for seed in self.seeds:
                    directory = fold_root / candidate["candidate_id"] / f"seed_{seed}_n{structural_count}"
                    identity = self.run_identity(fold, candidate["candidate_id"], candidate["setting"], seed, structural_count, fitting, validation, representation, "structure")
                    record = self.fit(directory, identity, (fit_matrix, validation_matrix), fitting, validation)
                    all_records.append(record)
                    locations[(candidate["candidate_id"], seed)] = directory
                    evidence.append(str((directory / "artifacts.json").relative_to(self.output)))
            matched_draws = {}
            for record in all_records:
                controls = record["identity"]["setting"]
                key = (controls["min_samples_leaf"], controls["class_weight"], record["identity"]["seed"])
                draws = record["bootstrap"]["draw_sha256"]
                if key in matched_draws and matched_draws[key] != draws:
                    raise ValueError("Feature-sampling contrast used different bootstrap draws")
                matched_draws[key] = draws
            winner = select_across_seeds([self.summary_row(record) for record in all_records], self.seeds)
            setting = next(row["setting"] for row in candidates if row["candidate_id"] == winner["candidate_id"])
            count_records = []
            for seed in self.seeds:
                source_directory = locations[(winner["candidate_id"], seed)]
                source_record = next(record for record in all_records if record["identity"]["candidate_id"] == winner["candidate_id"] and record["identity"]["seed"] == seed)
                for count in counts:
                    if count == structural_count:
                        count_records.append(self.summary_row(source_record, f"trees_{count:04d}"))
                        continue
                    directory = fold_root / "tree_count" / f"seed_{seed}_n{count}"
                    identity = self.run_identity(fold, winner["candidate_id"], setting, seed, count, fitting, validation, representation, "tree_count")
                    record = self.completed_record(directory, identity, validation)
                    if record is None:
                        forest = load_checkpoint(source_directory, source_record["identity"])["forest"]
                        if count < structural_count:
                            forest.estimators_ = forest.estimators_[:count]
                            forest.set_params(n_estimators=count, n_jobs=1)
                            record = self.persist(directory, identity, forest, validation_matrix, fitting, validation, utc_now(), None, 0)
                        else:
                            record = self.fit(directory, identity, (fit_matrix, validation_matrix), fitting, validation, previous=forest)
                        del forest
                        gc.collect()
                    count_records.append(self.summary_row(record, f"trees_{count:04d}"))
                    evidence.append(str((directory / "artifacts.json").relative_to(self.output)))
            count_winner = select_across_seeds(count_records, self.seeds, count_stage=True)
            selected.append({"fold": fold, "candidate_id": winner["candidate_id"], "setting": setting, "trees": int(count_winner["trees"]), "structural_selection": winner, "count_selection": count_winner})
            write_json(fold_root / "selected.json", selected[-1])
            evidence.append(str((fold_root / "selected.json").relative_to(self.output)))
            self.progress("fold_selection_complete", fold=fold)
            del fit_matrix, validation_matrix
            gc.collect()
        lock = {"identity": self.base_identity, "locked_utc": utc_now(), "selections": selected,
                "evidence": {name: file_hash(self.output / name) for name in evidence}}
        target = self.output / "selection_lock.json"
        if target.exists():
            old = self.require_lock()
            if old["selections"] != selected or old["evidence"] != lock["evidence"]:
                raise ValueError("Existing selection lock differs")
            return old
        write_json(target, lock)
        self.progress("all_selections_locked", folds=len(selected), selection_lock_sha256=file_hash(target))
        return lock

    def require_lock(self):
        path = self.output / "selection_lock.json"
        if not path.exists():
            raise ValueError("All fold selections must be locked before outer evaluation")
        lock = json.loads(path.read_text())
        if lock["identity"] != self.base_identity or [row["fold"] for row in lock["selections"]] != self.folds:
            raise ValueError("Selection lock identity or fold coverage differs")
        expected = len(self.folds) * (len(structural_candidates(self.protocol)) * len(self.seeds) + (len(self.protocol["selection"]["tree_count_stage"]["counts"]) - 1) * len(self.seeds) + 1)
        if len(lock["evidence"]) != expected:
            raise ValueError("Selection lock does not cover the complete selection budget")
        for name, expected in lock["evidence"].items():
            if file_hash(self.output / name) != expected:
                raise ValueError("Selection evidence changed after locking")
            if name.endswith("artifacts.json"):
                manifest_check((self.output / name).parent)
        return lock

    def evaluate(self):
        lock = self.require_lock()
        results = []
        for selected in lock["selections"]:
            fold = selected["fold"]
            fitting, evaluation = self.partitions[f"fold_{fold}_outer_train"], self.partitions[f"fold_{fold}_outer_test"]
            directory = self.output / "evaluation" / f"fold_{fold}"
            fit_matrix, evaluation_matrix, representation = self.prepare(fitting, evaluation, directory)
            for seed in self.seeds:
                identity = self.run_identity(fold, selected["candidate_id"], selected["setting"], seed, selected["trees"], fitting, evaluation, representation, "outer")
                record = self.fit(directory / f"seed_{seed}", identity, (fit_matrix, evaluation_matrix), fitting, evaluation)
                results.append({"fold": fold, "seed": seed, **record["metrics"], "checkpoint_bytes": record["checkpoint_bytes"]})
            del fit_matrix, evaluation_matrix
            gc.collect()
        write_json(self.output / "outer_records.json", results)
        self.progress("outer_evaluation_complete", models=len(results))
        return results


def run_primary(project_root, mode="all"):
    root = Path(project_root).resolve()
    protocol = json.loads((root / "configs/primary_protocol.json").read_text())
    validate_frozen_protocol(root, protocol)
    output = root / "output/primary" / canonical_hash(protocol)[:16]
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "process.lock"
    if lock_path.exists():
        previous_pid = int(lock_path.read_text())
        try:
            os.kill(previous_pid, 0)
        except ProcessLookupError:
            lock_path.unlink()
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        features, target = load_data(root, protocol)
        partitions = load_verified_splits(root, protocol, corrected_groups(features))
        study = Study(root, protocol, features, target, partitions, output)
        write_json(output / "protocol.json", protocol)
        write_json(root / "output/primary/latest.json", {"directory": str(output.relative_to(root)), "protocol_sha256": canonical_hash(protocol)})
        if mode in ["all", "select"]:
            if not (output / "selection_lock.json").exists():
                study.select()
            else:
                study.require_lock()
        if mode in ["all", "evaluate"]:
            study.evaluate()
            from .summarise import summarise_primary

            summarise_primary(root, output, protocol, target, partitions["outer_fold"])
    except Exception as error:
        write_json(output / "failure.json", {"utc": utc_now(), "type": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "select", "evaluate"], default="all")
    args = parser.parse_args()
    run_primary(args.project_root, args.mode)
