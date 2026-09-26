import numpy as np
import pytest
import torch

from src.metrics import scores, training_mase_scale
from src.models.recurrent import MODELS, count_parameters
from src import run_experiment
from src.splits import Fold, boundaries
from src.training import Scaler, causal_fill, fit_fold, windows


def test_chronology_and_target_nonoverlap():
    development_end, _, folds = boundaries(456)
    all_validation = []
    for fold in folds:
        assert max(fold.fit) < min(fold.stop)
        assert max(fold.stop) < min(fold.valid)
        assert max(fold.valid) < development_end
        all_validation.extend(fold.valid)
    assert len(all_validation) == len(set(all_validation))
    assert set(all_validation).isdisjoint(range(development_end, 456))


def test_scaler_only_uses_fitting_prefix_and_causal_fill():
    values = np.array([1.0, 2.0, np.nan, 3.0, 1_000_000.0])
    scaler = Scaler.fit(values, 4)
    assert scaler.mean == 2.0
    np.testing.assert_array_equal(causal_fill(values), [1, 2, 2, 3, 1_000_000])
    x, y, indices = windows(values, scaler, 2, 3, 5)
    assert x.shape == (2, 2, 1)
    assert y.shape == (2, 1)
    np.testing.assert_array_equal(indices, [3, 4])


@pytest.mark.parametrize("architecture,expected", [
    ("elman", 1**2 + 3*1 + 1),
    ("jordan", 4*1 + 1),
    ("multi", 1**2 + 4*1 + 1),
])
def test_recurrence_and_parameter_count(architecture, expected):
    model = MODELS[architecture](1)
    assert count_parameters(model) == expected
    with torch.no_grad():
        model.input_to_hidden.weight.fill_(1)
        model.input_to_hidden.bias.zero_()
        model.hidden_to_output.weight.fill_(2)
        model.hidden_to_output.bias.zero_()
        if hasattr(model, "hidden_to_hidden"):
            model.hidden_to_hidden.weight.fill_(1)
        if hasattr(model, "output_to_hidden"):
            model.output_to_hidden.weight.fill_(1)
    final, trace = model(torch.tensor([[[1.0], [0.0]]]), return_trace=True)
    first_hidden = np.tanh(1.0)
    feedback = {"elman": first_hidden, "jordan": 2*first_hidden,
                "multi": 3*first_hidden}[architecture]
    assert final.shape == (1, 1)
    assert len(trace) == 2
    assert float(final.detach()) == pytest.approx(2*np.tanh(feedback), abs=1e-6)


def test_metrics_hand_calculation():
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([0.0, 2.0, 5.0])
    assert training_mase_scale(actual) == 1.0
    result = scores(actual, forecast, 1.0)
    assert result["mae"] == 1.0
    assert result["rmse"] == pytest.approx(np.sqrt(5/3))
    assert result["mase"] == 1.0


def test_seed_reproduces_fold_fit():
    values = np.sin(np.arange(90) / 4.0)
    fold = Fold(1, 60, 70, 80)
    kwargs = dict(values=values, history=5, fold=fold, architecture="jordan",
                  hidden=4, learning_rate=0.001, seed=11, max_epochs=3)
    first = fit_fold(**kwargs)
    second = fit_fold(**kwargs)
    np.testing.assert_array_equal(first["forecast"], second["forecast"])
    assert first["best_epoch"] == second["best_epoch"]
    assert first["best_stop_mae"] == min(point["stop_mae"] for point in first["curve"])


def test_differenced_forecasts_reconstruct_levels():
    level = np.array([10.0, 11.0, 13.0, 12.0, 16.0])
    lag = 1
    target = level[lag:] - level[:-lag]
    base = level[:-lag]
    np.testing.assert_array_equal(target + base, level[lag:])
    assert base[2] == level[2]  


def test_saved_experiment_record_cannot_be_changed(tmp_path):
    path = tmp_path / "record.json"
    run_experiment.write_json(path, {"score": 1.0})
    run_experiment.write_json(path, {"score": 1.0})
    with pytest.raises(RuntimeError, match="differs"):
        run_experiment.write_json(path, {"score": 2.0})


def test_final_evaluation_requires_selection_freeze(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"datasets": [], "raw_sha256": {}}')
    monkeypatch.setattr(run_experiment, "PROTOCOL", protocol)
    monkeypatch.setattr(run_experiment, "OUTPUT", tmp_path / "primary")
    with pytest.raises(RuntimeError, match="Selection must be frozen"):
        run_experiment.main("evaluation")
