import argparse
import json
from pathlib import Path

from ml441_assignment4.records import canonical_hash


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("An input path leaves the local evidence project")
    return path


def inspect_inputs(project_root, phase="primary"):
    if phase not in {"pilot", "primary", "cost", "report"}:
        raise ValueError("Unknown workflow phase")
    root = Path(project_root).expanduser().resolve()
    required = {
        "data/raw/networkTraffic.csv": "Supplied traffic observations",
        "data/raw/features.csv": "Supplied feature descriptions",
        "data/raw/attack_category_map.csv": "Fixed target mapping",
    }
    protocol_name = "proposed_protocol" if phase == "pilot" else "primary_protocol"
    protocol_relative = f"configs/{protocol_name}.json"
    required[protocol_relative] = "Local declared development design" if phase == "pilot" else "Original frozen primary protocol"
    path = local_path(root, protocol_relative)
    protocol = json.loads(path.read_text()) if path.is_file() else None
    seal_matches = None
    if phase != "pilot":
        seal_relative = "configs/primary_protocol.sha256"
        required[seal_relative] = "Frozen protocol seal"
        seal_path = local_path(root, seal_relative)
        if protocol is not None and seal_path.is_file():
            seal_matches = canonical_hash(protocol) == seal_path.read_text().strip()
    if protocol is not None:
        required[protocol["splits"]["source"]] = "Verified historical grouped row partitions"
        for item in protocol.get("gate_evidence", []):
            required[item["path"]] = "Original sealed execution gate"
    if phase in {"cost", "report"}:
        required.update({
            "configs/comparable_cost_protocol.json": "Registered supplementary cost design",
            "configs/comparable_cost_protocol.sha256": "Registered supplementary cost seal",
            "materials/assignment2_baseline_records/selected_parameters.json": "Original selected comparator controls",
            "materials/assignment2_baseline_records/outer_predictions.csv.gz": "Original matched comparator predictions",
            "output/verification/completed_model_runs_review.json": "Pointers to completed primary and cost audits",
        })
        if protocol is not None:
            directory = "output/primary/" + canonical_hash(protocol)[:16]
            required[directory + "/selection_lock.json"] = "Locked inner selections"
            required[directory + "/comparison_summary.json"] = "Completed matched comparison"
    if phase == "report":
        required["report/generated/asset_manifest.json"] = "Existing report assets for independent report checking"
        required["output/verification/gate1_provenance_audit.json"] = "Historical source and control provenance"
        required["output/supplementary/R02_screening_review/screening_diagnostic.json"] = "Post-hoc screening projection and sealed prediction-source identities"
    inputs = [{"path": name, "purpose": purpose, "exists": local_path(root, name).is_file()}
              for name, purpose in sorted(required.items())]
    missing = [item["path"] for item in inputs if not item["exists"]]
    status = "invalid_protocol_seal" if seal_matches is False else "missing_private_inputs" if missing else "inputs_present_requires_execution_gate_checks"
    return {"status": status, "phase": phase, "project_root": str(root), "inputs": inputs,
            "missing": missing, "protocol_seal_matches": seal_matches,
            "scope": "Presence inventory of local inputs; private data, configuration and evidence are excluded from the public code repository",
            "runtime_guards_remain_required": True, "new_fits": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "primary", "cost", "report"], default="primary")
    args = parser.parse_args()
    print(json.dumps(inspect_inputs(args.project_root, args.phase), indent=2))


if __name__ == "__main__":
    main()
