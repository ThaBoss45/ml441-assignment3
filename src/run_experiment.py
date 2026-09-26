
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_series
from .download_data import RAW, sha256
from .metrics import scores, training_mase_scale
from .splits import boundaries
from .training import causal_fill, fit_final, fit_fold


PROTOCOL = Path("configs/primary_protocol.json")
OUTPUT = Path("output/primary")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Existing experimental record differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def verify_protocol(protocol: dict) -> None:
    for filename, expected in protocol["raw_sha256"].items():
        actual = sha256(RAW / filename)
        if actual != expected:
            raise ValueError(f"Raw file changed: {filename}: {actual}")


def select_for_dataset(key: str, protocol: dict) -> dict:
    data = load_series(key)
    values = data.values.to_numpy(dtype=float)
    levels = data.levels.to_numpy(dtype=float)
    base = data.base.to_numpy(dtype=float)
    development_end, block, folds = boundaries(len(values))
    folder = OUTPUT / "selection" / key
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for architecture in protocol["architectures"]:
        for hidden in protocol["hidden_grid"][architecture]:
            for rate in protocol["learning_rate_grid"]:
                for fold in folds:
                    for seed in protocol["seeds"]:
                        name = f"{architecture}_h{hidden}_lr{rate:g}_f{fold.number}_s{seed}.json"
                        path = folder / name
                        if path.exists():
                            record = json.loads(path.read_text())
                        else:
                            result = fit_fold(values, data.history, fold, architecture,
                                              hidden, rate, seed,
                                              max_epochs=protocol["max_epochs"],
                                              patience=protocol["patience"],
                                              batch_size=protocol["batch_size"],
                                              weight_decay=protocol["weight_decay"],
                                              levels=levels, base=base)
                            record = {"dataset": key, "architecture": architecture,
                                      "hidden": hidden, "learning_rate": rate,
                                      "fold": fold.number, "seed": seed, **result}
                            write_json(path, record)
                        records.append(record)
    selected = {}
    for architecture in protocol["architectures"]:
        candidates = []
        for hidden in protocol["hidden_grid"][architecture]:
            for rate in protocol["learning_rate_grid"]:
                subset = [row for row in records if row["architecture"] == architecture
                          and row["hidden"] == hidden and row["learning_rate"] == rate]
                assert len(subset) == len(folds) * len(protocol["seeds"])
                mean_mae = float(np.mean([row["dataset_metrics"]["mae"] for row in subset]))
                candidates.append((mean_mae, subset[0]["parameter_count"], rate,
                                   hidden, subset))
        mean_mae, parameters, rate, hidden, subset = min(candidates, key=lambda c: c[:3])
        selected[architecture] = {"hidden": hidden, "learning_rate": rate,
                                  "mean_validation_mae": mean_mae,
                                  "parameter_count": parameters,
                                  "epochs": max(1, round(float(np.median([r["best_epoch"] for r in subset])))),
                                  "cap_limited_fits": sum(r["best_epoch"] == protocol["max_epochs"] for r in subset)}
    selection = {"dataset": key, "n": len(values), "development_end": development_end,
                 "validation_block": block, "history": data.history,
                 "difference_lag": data.difference_lag,
                 "test_start_date": str(data.values.index[development_end].date()),
                 "folds": [fold.__dict__ for fold in folds], "selected": selected}
    write_json(folder / "selected.json", selection)
    return selection


def evaluate_dataset(key: str, selection: dict, protocol: dict) -> dict:
    data = load_series(key)
    values = data.values.to_numpy(dtype=float)
    levels = data.levels.to_numpy(dtype=float)
    base = data.base.to_numpy(dtype=float)
    development_end = selection["development_end"]
    folder = OUTPUT / "evaluation" / key
    folder.mkdir(parents=True, exist_ok=True)
    runs = []
    for architecture, chosen in selection["selected"].items():
        for seed in protocol["seeds"]:
            path = folder / f"{architecture}_s{seed}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                result = fit_final(values, data.history, development_end,
                                   architecture, chosen["hidden"],
                                   chosen["learning_rate"], seed,
                                   chosen["epochs"],
                                   batch_size=protocol["batch_size"],
                                   weight_decay=protocol["weight_decay"],
                                   levels=levels, base=base)
                record = {"dataset": key, "architecture": architecture,
                          "seed": seed, "hidden": chosen["hidden"],
                          "learning_rate": chosen["learning_rate"], **result}
                write_json(path, record)
            runs.append(record)
    test_indices = np.arange(development_end, len(levels))
    test_indices = test_indices[np.isfinite(levels[test_indices])]
    filled = causal_fill(levels)
    scale = training_mase_scale(levels[:development_end])
    baselines = {"persistence": scores(levels[test_indices], filled[test_indices-1], scale)}
    if data.seasonal_lag is not None:
        baselines["seasonal_naive"] = scores(levels[test_indices],
                                              filled[test_indices-data.seasonal_lag], scale)
    assert all(record["indices"] == test_indices.tolist() for record in runs)
    summary = {"dataset": key, "test_start_date": selection["test_start_date"],
               "test_target_count": len(test_indices), "baselines": baselines,
               "networks": {architecture: {
                   "mean_mae": float(np.mean([row["dataset_metrics"]["mae"] for row in runs
                                              if row["architecture"] == architecture])),
                   "sd_mae": float(np.std([row["dataset_metrics"]["mae"] for row in runs
                                            if row["architecture"] == architecture], ddof=1)),
                   "mean_rmse": float(np.mean([row["dataset_metrics"]["rmse"] for row in runs
                                               if row["architecture"] == architecture])),
                   "mean_mase": float(np.mean([row["dataset_metrics"]["mase"] for row in runs
                                               if row["architecture"] == architecture])),
                   "parameters": selection["selected"][architecture]["parameter_count"],
                   "epochs": selection["selected"][architecture]["epochs"]}
                   for architecture in protocol["architectures"]}}
    write_json(folder / "summary.json", summary)
    return summary


def main(phase: str = "all") -> None:
    if phase not in {"selection", "evaluation", "all"}:
        raise ValueError(f"Unknown phase: {phase}")
    protocol = json.loads(PROTOCOL.read_text())
    verify_protocol(protocol)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "protocol_snapshot.json", protocol)
    if phase in {"selection", "all"}:
        selections = {}
        for key in protocol["datasets"]:
            selection = select_for_dataset(key, protocol)
            selections[key] = selection
            print("SELECTED", key, selection["selected"], flush=True)
        
        write_json(OUTPUT / "all_selected_before_test.json", selections)
    if phase in {"evaluation", "all"}:
        freeze = OUTPUT / "all_selected_before_test.json"
        if not freeze.exists():
            raise RuntimeError("Selection must be frozen before final evaluation")
        selections = json.loads(freeze.read_text())
        if list(selections) != protocol["datasets"]:
            raise RuntimeError("Frozen dataset order differs from the protocol")
        for key in protocol["datasets"]:
            if selections[key] != json.loads((OUTPUT / "selection" / key / "selected.json").read_text()):
                raise RuntimeError(f"Selected configuration changed after freeze: {key}")
        summaries = []
        for key in protocol["datasets"]:
            summary = evaluate_dataset(key, selections[key], protocol)
            summaries.append(summary)
            print("TEST", key, summary, flush=True)
        write_json(OUTPUT / "summary.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["selection", "evaluation", "all"], default="all")
    main(parser.parse_args().phase)
