
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PRIMARY = Path("output/primary/selection")
DEST = Path("output/diagnostics/training_curves.json")


def main() -> None:
    rows = []
    for dataset in ("co2", "sunspots", "power", "retail", "rain"):
        selected = json.loads((PRIMARY / dataset / "selected.json").read_text())["selected"]
        for architecture, config in selected.items():
            files = sorted((PRIMARY / dataset).glob(
                f"{architecture}_h{config['hidden']}_lr{config['learning_rate']:g}_f*_s*.json"))
            assert len(files) == 9
            for path in files:
                record = json.loads(path.read_text())
                curve = record["curve"]
                chosen = curve[record["best_epoch"] - 1]
                last = curve[-1]
                rows.append({"dataset": dataset, "architecture": architecture,
                             "fold": record["fold"], "seed": record["seed"],
                             "best_epoch": record["best_epoch"],
                             "epochs_run": record["epochs_run"],
                             "training_mse_lower_at_last_epoch": bool(
                                 last["train_mse_scaled"] < chosen["train_mse_scaled"] - 1e-12),
                             "tail_mae_increase": float(last["stop_mae"] - chosen["stop_mae"])})
    assert len(rows) == 135
    summary = {
        "role": "supplementary post-hoc diagnostic of saved development curves",
        "purpose": "Check whether training loss continued to fall after the tail-selected checkpoint",
        "partition": "development fitting and chronological early-stopping tails only",
        "fixed": "frozen selected configurations and saved curve records",
        "varied": "epoch, fold, seed, architecture and dataset as already recorded",
        "model_selection_effect": "none",
        "limitation": "The best tail MAE is selected by construction; training scaled MSE and tail original-unit MAE are different metrics on different periods. This diagnostic does not prove overfitting.",
        "selected_development_fits": len(rows),
        "training_loss_lower_after_selected_checkpoint": sum(
            row["training_mse_lower_at_last_epoch"] for row in rows),
        "all_tail_mae_increases_by_construction": all(
            row["tail_mae_increase"] >= -1e-10 for row in rows),
        "per_dataset": {
            dataset: {
                "fits": sum(row["dataset"] == dataset for row in rows),
                "training_loss_lower_after_selected_checkpoint": sum(
                    row["training_mse_lower_at_last_epoch"]
                    for row in rows if row["dataset"] == dataset),
                "median_tail_mae_increase": float(np.median([
                    row["tail_mae_increase"] for row in rows
                    if row["dataset"] == dataset]))
            } for dataset in ("co2", "sunspots", "power", "retail", "rain")},
        "records": rows,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(summary, indent=2))
    print({key: summary[key] for key in (
        "selected_development_fits", "training_loss_lower_after_selected_checkpoint",
        "all_tail_mae_increases_by_construction")})


if __name__ == "__main__":
    main()
