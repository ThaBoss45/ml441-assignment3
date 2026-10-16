import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .primary import Study
from .records import canonical_hash, code_identity, software_versions, write_json


def synthetic_inputs(protocol):
    protocol = copy.deepcopy(protocol)
    protocol["budget"]["maximum_fit_hours"] = 1
    protocol["ensemble"]["structural_n_estimators"] = 2
    protocol["selection"]["tree_count_stage"]["counts"] = [1, 2, 4]
    protocol["splits"]["outer_folds"] = 2
    protocol["ensemble"]["structural_grid"]["min_samples_leaf"] = [1, 3, 5]
    rng = np.random.default_rng(273)
    features = pd.DataFrame(rng.normal(size=(600, 4)), columns=["a", "b", "c", "d"])
    features["proto"] = np.where(np.arange(600) % 2, "tcp", "udp")
    features["service"] = np.where(np.arange(600) % 3, "Missing", "http")
    features["state"] = "known"
    target = np.tile(np.arange(10, dtype=np.int8), 60)
    partitions = {}
    for fold, start, other in [(1, 0, 300), (2, 300, 0)]:
        partitions[f"fold_{fold}_outer_train"] = np.arange(start, start + 300)
        partitions[f"fold_{fold}_outer_test"] = np.arange(other, other + 300)
        partitions[f"fold_{fold}_inner_train"] = np.arange(start, start + 200)
        partitions[f"fold_{fold}_inner_validation"] = np.arange(start + 200, start + 300)
    return protocol, features, target, partitions


def run_smoke(root):
    root = Path(root).resolve()
    proposal = json.loads((root / "configs/proposed_protocol.json").read_text())
    protocol, features, target, partitions = synthetic_inputs(proposal)
    identity = {"code": code_identity(), "environment": software_versions(), "protocol": canonical_hash(protocol)}
    output = root / "output/pilot" / ("runner_smoke_" + canonical_hash(identity)[:16])
    study = Study(root, protocol, features, target, partitions, output, role="synthetic_smoke")
    try:
        study.evaluate()
    except ValueError:
        pass
    else:
        if not (output / "selection_lock.json").exists():
            raise AssertionError("Premature outer evaluation was accepted")
    study.select()
    results = study.evaluate()
    for seed in protocol["ensemble"]["seeds"]:
        rows = []
        for fold in [1, 2]:
            with np.load(output / "evaluation" / f"fold_{fold}" / f"seed_{seed}" / "predictions.npz", allow_pickle=False) as record:
                rows.extend(record["row_index"].tolist())
        np.testing.assert_array_equal(np.sort(rows), np.arange(600))
    summary = {"status": "passed", "identity": identity, "synthetic_rows": 600, "synthetic_outer_models": len(results), "real_outer_rows_evaluated": 0, "selection_lock_verified": True}
    write_json(output / "summary.json", summary)
    write_json(root / "output/pilot/latest_smoke.json", {"directory": str(output.relative_to(root)), "summary": summary})
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    print(json.dumps(run_smoke(parser.parse_args().project_root)))
