import json
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from .data import load_series
from .splits import boundaries
from .training import causal_fill


PRIMARY = Path("output/primary")
DEST = Path("output/diagnostics")
DATASETS = ("co2", "sunspots", "power", "retail", "rain")
ARCHITECTURES = ("elman", "jordan", "multi")
SEEDS = (11, 23, 37)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(name: str, record: dict) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(json.dumps(record, indent=2) + "\n")


def paired_errors(summaries: list[dict]) -> dict:
    results = []
    for summary in summaries:
        dataset = summary["dataset"]
        ordered = sorted(ARCHITECTURES, key=lambda a: summary["networks"][a]["mean_mae"])
        winner, runner = ordered[:2]
        seed_rows = []
        differences = []
        for seed in SEEDS:
            leading = read(PRIMARY / "evaluation" / dataset / f"{winner}_s{seed}.json")
            following = read(PRIMARY / "evaluation" / dataset / f"{runner}_s{seed}.json")
            if leading["indices"] != following["indices"] or leading["actual"] != following["actual"]:
                raise ValueError(f"Paired target mismatch for {dataset}, seed {seed}")
            actual = np.asarray(leading["actual"], dtype=float)
            lead_error = np.abs(actual - np.asarray(leading["forecast"], dtype=float))
            follow_error = np.abs(actual - np.asarray(following["forecast"], dtype=float))
            difference = follow_error - lead_error
            differences.append(difference)
            seed_rows.append({"seed": seed, "mean_error_advantage": float(difference.mean()),
                              "target_count": len(difference)})
        paired = np.stack(differences)
        runner_mae = float(summary["networks"][runner]["mean_mae"])
        advantage = float(paired.mean())
        if not np.isclose(advantage, runner_mae - summary["networks"][winner]["mean_mae"], atol=1e-9):
            raise ValueError(f"Summary mismatch for {dataset}")
        thirds = [float(paired[:, block].mean()) for block in
                  np.array_split(np.arange(paired.shape[1]), 3)]
        results.append({"dataset": dataset, "winner": winner, "runner": runner,
                        "target_count": paired.shape[1], "seed_count": paired.shape[0],
                        "mean_error_advantage": advantage,
                        "relative_advantage_percent": 100 * advantage / runner_mae,
                        "paired_seed_wins": sum(row["mean_error_advantage"] > 0 for row in seed_rows),
                        "third_advantages": thirds, "seeds": seed_rows})
    return {"role": "supplementary post-hoc description of frozen final predictions",
            "partition": "saved final periods only", "fixed": "frozen targets, predictions and metrics",
            "varied": "architecture, seed and calendar third", "model_selection_effect": "none",
            "limitation": "Origins share history and thirds are descriptive, not independent test replications; the winner and runner were identified on the same final period.",
            "sign_convention": "positive advantage means lower absolute error for the observed winner",
            "results": results}


def capacity_sensitivity(protocol: dict) -> dict:
    results = []
    for dataset in DATASETS:
        budgets = []
        for budget in (0, 1):
            architectures = []
            for architecture in ARCHITECTURES:
                hidden = protocol["hidden_grid"][architecture][budget]
                rates = []
                for rate in protocol["learning_rate_grid"]:
                    files = sorted((PRIMARY / "selection" / dataset).glob(
                        f"{architecture}_h{hidden}_lr{rate:g}_f*_s*.json"))
                    if len(files) != 9:
                        raise ValueError(f"Expected nine development records: {dataset}, {architecture}, {hidden}, {rate}")
                    records = [read(path) for path in files]
                    rates.append({"learning_rate": rate, "mean_validation_mae": float(np.mean([
                        row["dataset_metrics"]["mae"] for row in records])),
                        "folds": sorted({row["fold"] for row in records}),
                        "seeds": sorted({row["seed"] for row in records})})
                best_rate = min(rates, key=lambda row: row["mean_validation_mae"])
                architectures.append({"architecture": architecture, "hidden": hidden,
                                      "parameter_count": records[0]["parameter_count"],
                                      "best_rate": best_rate, "rates": rates})
            leading = min(architectures, key=lambda row: row["best_rate"]["mean_validation_mae"])
            budgets.append({"budget": "small" if budget == 0 else "large",
                            "leader": leading["architecture"],
                            "leader_mean_validation_mae": leading["best_rate"]["mean_validation_mae"],
                            "architectures": architectures})
        results.append({"dataset": dataset, "budgets": budgets})
    return {"role": "supplementary post-hoc reorganisation of frozen development search records",
            "partition": "three development validation blocks only",
            "fixed": "frozen histories, origins, seeds, preprocessing, optimiser and candidate grid",
            "varied": "architecture and capacity budget; each budget uses its lower-error pre-existing learning-rate candidate",
            "model_selection_effect": "none; original selected settings and final predictions remain frozen",
            "limitation": "The best rate at each budget was identified on these same validation blocks; width and topology remain coupled, and this is not an independent assessment.",
            "results": results}


def fit_sensitivity() -> dict:
    results = []
    for dataset in DATASETS:
        selected = read(PRIMARY / "selection" / dataset / "selected.json")["selected"]
        rows = []
        for architecture in ARCHITECTURES:
            config = selected[architecture]
            files = sorted((PRIMARY / "selection" / dataset).glob(
                f"{architecture}_h{config['hidden']}_lr{config['learning_rate']:g}_f*_s*.json"))
            if len(files) != 9:
                raise ValueError(f"Expected nine selected development records: {dataset}, {architecture}")
            for path in files:
                record = read(path)
                curve = record["curve"]
                first, best, last = curve[0], curve[record["best_epoch"] - 1], curve[-1]
                rows.append({"architecture": architecture, "fold": record["fold"],
                             "seed": record["seed"], "best_epoch": record["best_epoch"],
                             "epochs_run": record["epochs_run"],
                             "train_reduction_to_checkpoint_percent": 100 * (1 - best["train_mse_scaled"] / first["train_mse_scaled"]),
                             "tail_reduction_to_checkpoint_percent": 100 * (1 - best["stop_mae"] / first["stop_mae"]),
                             "train_reduction_after_checkpoint_percent": 100 * (1 - last["train_mse_scaled"] / best["train_mse_scaled"]),
                             "tail_increase_after_checkpoint_percent": 100 * (last["stop_mae"] / best["stop_mae"] - 1)})
        def median(key: str) -> float:
            return float(np.median([row[key] for row in rows]))
        results.append({"dataset": dataset, "fit_count": len(rows),
                        "median_best_epoch": median("best_epoch"),
                        "median_train_reduction_to_checkpoint_percent": median("train_reduction_to_checkpoint_percent"),
                        "median_tail_reduction_to_checkpoint_percent": median("tail_reduction_to_checkpoint_percent"),
                        "median_train_reduction_after_checkpoint_percent": median("train_reduction_after_checkpoint_percent"),
                        "median_tail_increase_after_checkpoint_percent": median("tail_increase_after_checkpoint_percent"),
                        "train_loss_lower_after_checkpoint_count": sum(
                            row["train_reduction_after_checkpoint_percent"] > 0 for row in rows),
                        "rows": rows})
    return {"role": "supplementary post-hoc description of saved development training histories",
            "partition": "fitting prefixes and chronological stopping tails only",
            "fixed": "selected primary configurations, folds and seeds",
            "varied": "epoch as previously recorded", "model_selection_effect": "none",
            "limitation": "Training scaled mean squared error and stopping-tail original-unit mean absolute error differ in metric and calendar period; the selected tail minimum is best by construction. These changes cannot establish overfitting causally.",
            "results": results}


def stationarity_sensitivity() -> dict:
    results = []
    for dataset in DATASETS:
        data = load_series(dataset)
        development_end, _, _ = boundaries(len(data.values))
        series = {}
        for name, values in (("level", data.levels.to_numpy(dtype=float)),
                             ("modelled_target", data.values.to_numpy(dtype=float))):
            development = causal_fill(values[:development_end])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tests = {
                    specification: {
                        "adf_unit_root_null_p": float(adfuller(development, regression=specification, autolag="AIC")[1]),
                        "kpss_stationarity_null_p": float(kpss(development, regression=specification, nlags="auto")[1])
                    } for specification in ("c", "ct")
                }
            halves = np.array_split(development, 2)
            deviations = [float(np.std(half, ddof=1)) for half in halves]
            series[name] = {"tests": tests, "half_standard_deviations": deviations,
                            "second_to_first_standard_deviation_ratio": deviations[1] / deviations[0]}
        results.append({"dataset": dataset, "development_end": development_end,
                        "development_end_date": str(data.values.index[development_end - 1].date()),
                        "series": series})
    return {"role": "supplementary post-hoc development-only stationarity sensitivity",
            "partition": "observations strictly before each frozen final-test boundary",
            "fixed": "original and frozen transformed targets; causal fill for missing power observations",
            "varied": "constant versus constant-and-trend regression terms; first versus second development half",
            "model_selection_effect": "none; no target, horizon, split or transform was changed",
            "limitation": "The tests assume their stated regression forms, have limited power under structural change and seasonal dependence, and do not prove stationarity. Half-period spread is descriptive.",
            "results": results}


def main() -> None:
    if not (PRIMARY / "all_selected_before_test.json").exists():
        raise RuntimeError("Missing primary selection freeze")
    summaries = read(PRIMARY / "summary.json")
    protocol = read(Path("configs/primary_protocol.json"))
    save("paired_errors.json", paired_errors(summaries))
    save("capacity_sensitivity.json", capacity_sensitivity(protocol))
    save("fit_sensitivity.json", fit_sensitivity())
    save("stationarity_sensitivity.json", stationarity_sensitivity())
    print("Supplementary diagnostics generated from frozen records and development data")


if __name__ == "__main__":
    main()
