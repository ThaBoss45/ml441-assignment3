
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_series
from .download_data import RAW, sha256
from .models.recurrent import MODELS, count_parameters
from .splits import boundaries
from .training import causal_fill


ROOT = Path("output/primary")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def independent_metrics(actual: np.ndarray, forecast: np.ndarray,
                        scale: float) -> dict[str, float]:
    errors = actual - forecast
    return {"mae": float(np.abs(errors).sum() / len(errors)),
            "rmse": float(np.sqrt(np.square(errors).sum() / len(errors))),
            "mase": float(np.abs(errors).sum() / len(errors) / scale)}


def agree(left: dict, right: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if not np.isclose(left[key], right[key], atol=1e-10, rtol=1e-10):
            raise AssertionError(f"{key}: {left[key]} != {right[key]}")


def main() -> None:
    protocol = read(Path("configs/primary_protocol.json"))
    assert read(ROOT / "protocol_snapshot.json") == protocol
    for filename, expected in protocol["raw_sha256"].items():
        assert sha256(RAW / filename) == expected, filename
    freeze = read(ROOT / "all_selected_before_test.json")
    summaries = {row["dataset"]: row for row in read(ROOT / "summary.json")}
    selection_count = final_count = 0
    for key in protocol["datasets"]:
        data = load_series(key)
        values = data.values.to_numpy(dtype=float)
        levels = data.levels.to_numpy(dtype=float)
        selection = read(ROOT / "selection" / key / "selected.json")
        assert freeze[key] == selection
        development_end, block, folds = boundaries(len(values))
        assert selection["development_end"] == development_end
        assert selection["validation_block"] == block
        assert selection["folds"] == [fold.__dict__ for fold in folds]
        assert folds[0].fit_end < folds[0].stop_end < folds[0].valid_end
        assert all(folds[i].valid_end == folds[i+1].stop_end for i in (0, 1))
        assert folds[-1].valid_end == development_end
        scale = np.abs(np.diff(levels[:development_end]))
        mask = np.isfinite(levels[:development_end-1]) & np.isfinite(levels[1:development_end])
        mase_scale = float(scale[mask].mean())
        expected_test = np.arange(development_end, len(values))
        expected_test = expected_test[np.isfinite(levels[expected_test])]
        summary = summaries[key]
        assert summary["test_target_count"] == len(expected_test)

        for architecture in protocol["architectures"]:
            candidate_scores = []
            for hidden in protocol["hidden_grid"][architecture]:
                parameter_count = count_parameters(MODELS[architecture](hidden))
                for rate in protocol["learning_rate_grid"]:
                    rows = []
                    for fold in folds:
                        for seed in protocol["seeds"]:
                            path = ROOT / "selection" / key / (
                                f"{architecture}_h{hidden}_lr{rate:g}_f{fold.number}_s{seed}.json")
                            record = read(path)
                            selection_count += 1
                            assert (record["architecture"], record["hidden"],
                                    record["learning_rate"], record["seed"], record["fold"]) == (
                                    architecture, hidden, rate, seed, fold.number)
                            assert record["parameter_count"] == parameter_count
                            assert record["scaler"]["fitted_end"] == fold.fit_end
                            assert np.isclose(record["scaler"]["mean"],
                                              np.nanmean(values[:fold.fit_end]))
                            indices = np.asarray(record["indices"], dtype=int)
                            assert np.all((indices >= fold.stop_end) & (indices < fold.valid_end))
                            actual = np.asarray(record["actual"])
                            forecast = np.asarray(record["forecast"])
                            np.testing.assert_array_equal(actual, levels[indices])
                            fit_levels = levels[:fold.fit_end]
                            fit_mask = np.isfinite(fit_levels[:-1]) & np.isfinite(fit_levels[1:])
                            fold_scale = float(np.abs(np.diff(fit_levels))[fit_mask].mean())
                            agree(independent_metrics(actual, forecast, fold_scale),
                                  record["dataset_metrics"], ("mae", "rmse", "mase"))
                            rows.append(record)
                    candidate_scores.append((float(np.mean([r["dataset_metrics"]["mae"]
                                                            for r in rows])),
                                             parameter_count, rate, hidden, rows))
            winner = min(candidate_scores, key=lambda item: item[:3])
            chosen = selection["selected"][architecture]
            assert chosen["hidden"] == winner[3]
            assert chosen["learning_rate"] == winner[2]
            assert chosen["parameter_count"] == winner[1]
            assert np.isclose(chosen["mean_validation_mae"], winner[0])
            assert chosen["epochs"] == max(1, round(float(np.median(
                [r["best_epoch"] for r in winner[4]]))))

            metrics = []
            for seed in protocol["seeds"]:
                record = read(ROOT / "evaluation" / key / f"{architecture}_s{seed}.json")
                final_count += 1
                indices = np.asarray(record["indices"], dtype=int)
                np.testing.assert_array_equal(indices, expected_test)
                actual = np.asarray(record["actual"])
                np.testing.assert_array_equal(actual, levels[indices])
                assert record["scaler"]["fitted_end"] == development_end
                assert np.isclose(record["scaler"]["mean"], np.nanmean(values[:development_end]))
                assert record["parameter_count"] == chosen["parameter_count"]
                assert record["epochs"] == chosen["epochs"]
                measured = independent_metrics(actual, np.asarray(record["forecast"]), mase_scale)
                agree(measured, record["dataset_metrics"], ("mae", "rmse", "mase"))
                metrics.append(measured)
            network_summary = summary["networks"][architecture]
            assert np.isclose(network_summary["mean_mae"], np.mean([m["mae"] for m in metrics]))
            assert np.isclose(network_summary["sd_mae"], np.std([m["mae"] for m in metrics], ddof=1))
            assert np.isclose(network_summary["mean_rmse"], np.mean([m["rmse"] for m in metrics]))
            assert np.isclose(network_summary["mean_mase"], np.mean([m["mase"] for m in metrics]))

        filled = causal_fill(levels)
        for baseline, lag in (("persistence", 1), ("seasonal_naive", data.seasonal_lag)):
            if lag is None:
                assert baseline not in summary["baselines"]
                continue
            measured = independent_metrics(levels[expected_test],
                                           filled[expected_test-lag], mase_scale)
            agree(measured, summary["baselines"][baseline], ("mae", "rmse", "mase"))
    assert selection_count == 540
    assert final_count == 45
    result = {"status": "passed", "datasets": len(protocol["datasets"]),
              "selection_records": selection_count, "final_network_records": final_count,
              "raw_hashes": len(protocol["raw_sha256"]),
              "checks": ["chronology", "selection minimum", "selected epoch", "scaler scope",
                         "parameter count", "target alignment", "MAE", "RMSE", "MASE", "baselines"]}
    output = Path("output/diagnostics/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
