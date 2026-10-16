import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .data import corrected_groups, load_data
from .models import make_forest, probability_aggregate, probability_labels, tree_costs
from .preprocessing import forest_matrix, make_preprocessor, representation_record
from .records import MemoryMonitor, array_hash, canonical_hash, code_identity, file_hash, load_checkpoint, save_checkpoint, software_versions, write_json
from .splits import load_verified_splits


def run_cost_pilot(project_root):
    root = Path(project_root).resolve()
    protocol = json.loads((root / "configs/proposed_protocol.json").read_text())
    registration = json.loads((root / "configs/cost_pilot_protocol.json").read_text())
    identity = {"phase": "cost_pilot", "registration": registration, "protocol_sha256": canonical_hash(protocol), "code": code_identity(), "environment": software_versions()}
    output = root / "output/pilot" / ("cost_" + canonical_hash(identity)[:16])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity.json", identity)
    write_json(output / "registration.json", registration)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in identity["code"]:
        shutil.copyfile(Path(__file__).parent / name, snapshot / name)
    with MemoryMonitor() as total_cost:
        features, target = load_data(root, protocol)
        partitions = load_verified_splits(root, protocol, corrected_groups(features))
        fitting, validation = partitions["fold_1_inner_train"], partitions["fold_1_inner_validation"]
        preprocessor = make_preprocessor(features.iloc[fitting], protocol["representation"]["nominal"])
        with MemoryMonitor() as preprocessing_cost:
            fit_matrix = forest_matrix(preprocessor.fit_transform(features.iloc[fitting]), fitting=True)
            validation_matrix = forest_matrix(preprocessor.transform(features.iloc[validation]))
        write_json(output / "representation.json", {**representation_record(preprocessor), "cost": preprocessing_cost.record()})
        np.savez_compressed(output / "indices.npz", fitting=fitting, validation=validation)
        records = []
        for specification in registration["runs"]:
            forest = None
            for count in specification["counts"]:
                name = specification["id"] + f"_{count}"
                print(f"Cost pilot {name}", flush=True)
                directory = output / name
                run_identity = {**identity, "run": name, "setting": specification["setting"], "seed": registration["seed"], "trees": count, "fitting_sha256": array_hash(fitting), "validation_sha256": array_hash(validation)}
                if (directory / "record.json").exists():
                    forest = load_checkpoint(directory, run_identity)["forest"]
                    record = json.loads((directory / "record.json").read_text())
                    records.append(record)
                    continue
                started = datetime.now(timezone.utc).isoformat()
                previous = len(forest.estimators_) if forest is not None else 0
                if forest is None:
                    forest = make_forest(protocol, specification["setting"], registration["seed"], count, workers=4, warm_start=True)
                else:
                    forest.set_params(n_estimators=count, n_jobs=4)
                with threadpool_limits(limits=1), MemoryMonitor() as fit_cost:
                    forest.fit(fit_matrix, target[fitting])
                forest.set_params(n_jobs=1)
                with threadpool_limits(limits=1), MemoryMonitor() as predict_cost:
                    library = forest.predict_proba(validation_matrix)
                    predicted = probability_labels(library, protocol["data"]["class_order"])
                with threadpool_limits(limits=1):
                    manual, _ = probability_aggregate(forest, validation_matrix, protocol["data"]["class_order"])
                np.testing.assert_allclose(library, manual, atol=1e-14, rtol=0)
                np.testing.assert_array_equal(predicted, probability_labels(manual, protocol["data"]["class_order"]))
                with MemoryMonitor() as serialisation_cost:
                    path = save_checkpoint(directory, run_identity, {"forest": forest, "preprocessor": preprocessor})
                record = {
                    "identity": run_identity, "id": name, "setting_id": specification["id"], "trees": count,
                    "new_trees": count - previous, "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "fit_tree_workers": 4, "prediction_tree_workers": 1,
                    "fit_cost": fit_cost.record(), "prediction_cost": predict_cost.record(), "serialisation_cost": serialisation_cost.record(),
                    "checkpoint_bytes": path.stat().st_size, "tree_costs": tree_costs(forest),
                    "aggregation_max_error": float(np.max(np.abs(library - manual))),
                    "selection_scores_calculated": False, "outer_rows_evaluated": 0,
                }
                np.savez_compressed(directory / "predictions.npz", row_index=validation, predicted=predicted, probabilities=manual, labels=np.asarray(protocol["data"]["class_order"]))
                write_json(directory / "record.json", record)
                records.append(record)
                write_json(output / "progress.json", {"completed": len(records), "runs": [row["id"] for row in records]})
            del forest
            gc.collect()
    summary = {"status": "passed", "identity": identity, "records": records, "total_cost": total_cost.record(), "outer_rows_evaluated": 0, "selection_scores_calculated": False}
    write_json(output / "summary.json", summary)
    write_json(root / "output/pilot/latest_cost.json", {"directory": str(output.relative_to(root)), "summary_sha256": file_hash(output / "summary.json")})
    print(json.dumps({"status": "passed", "output": str(output)}), flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    run_cost_pilot(parser.parse_args().project_root)
