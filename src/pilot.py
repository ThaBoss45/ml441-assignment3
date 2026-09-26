
import json
from pathlib import Path

import numpy as np

from .data import load_series
from .metrics import scores, training_mase_scale
from .splits import boundaries
from .training import fit_fold


def main() -> None:
    destination = Path("output/pilot")
    destination.mkdir(parents=True, exist_ok=True)
    data = load_series("co2")
    values = data.values.to_numpy(dtype=float)
    levels = data.levels.to_numpy(dtype=float)
    base = data.base.to_numpy(dtype=float)
    development_end, block, folds = boundaries(len(values))
    fold = folds[0]
    assert fold.valid_end < development_end
    config = {"dataset": "co2", "fold": fold.__dict__, "history": data.history,
              "hidden": {"elman": 8, "jordan": 24, "multi": 8},
              "learning_rate": 0.001, "seeds": [11, 23], "max_epochs": 400,
              "patience": 20, "batch_size": 256, "weight_decay": 1e-4,
              "output_directory": str(destination),
              "difference_lag": data.difference_lag}
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    results = []
    for architecture, hidden in config["hidden"].items():
        for seed in config["seeds"]:
            result = fit_fold(values, data.history, fold, architecture, hidden,
                              config["learning_rate"], seed, max_epochs=400,
                              patience=20, levels=levels, base=base)
            row = {"architecture": architecture, "seed": seed, "hidden": hidden,
                   **result}
            results.append(row)
    
    repeat = fit_fold(values, data.history, fold, "elman", 8, 0.001, 11,
                      max_epochs=400, patience=20, levels=levels, base=base)
    assert np.array_equal(results[0]["forecast"], repeat["forecast"])
    actual = levels[results[0]["indices"]]
    baseline = levels[np.array(results[0]["indices"])-1]
    naive = scores(actual, baseline, training_mase_scale(levels[:fold.fit_end]))
    summary = {"development_end": development_end, "validation_block": block,
               "naive": naive, "deterministic_repeat": True,
               "runs": [{"architecture": r["architecture"], "seed": r["seed"],
                         "mae": r["dataset_metrics"]["mae"],
                         "best_epoch": r["best_epoch"],
                         "runtime_seconds": r["runtime_seconds"]} for r in results]}
    (destination / "results.json").write_text(json.dumps(results, indent=2))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
