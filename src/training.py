
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from .metrics import scores, training_mase_scale
from .models.recurrent import MODELS, count_parameters
from .splits import Fold


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class Scaler:
    mean: float
    std: float
    fitted_end: int

    @classmethod
    def fit(cls, values: np.ndarray, end: int) -> "Scaler":
        observed = values[:end][np.isfinite(values[:end])]
        if len(observed) < 2:
            raise ValueError("Insufficient observed fitting targets")
        std = float(observed.std())
        return cls(float(observed.mean()), std if std > 0 else 1.0, end)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def causal_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values[0]):
        raise ValueError("The first input must be observed")
    result = values.copy()
    last = result[0]
    for index in range(1, len(result)):
        if np.isfinite(result[index]):
            last = result[index]
        else:
            result[index] = last
    return result


def windows(values: np.ndarray, scaler: Scaler, history: int,
            start: int, end: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if start < history:
        start = history
    observed_indices = np.arange(start, end)[np.isfinite(values[start:end])]
    filled = scaler.transform(causal_fill(values))
    inputs = np.stack([filled[index-history:index] for index in observed_indices])
    targets = scaler.transform(values[observed_indices])
    return (torch.from_numpy(inputs.astype(np.float32)).unsqueeze(-1),
            torch.from_numpy(targets.astype(np.float32)).unsqueeze(-1),
            observed_indices)


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict(model: torch.nn.Module, x: torch.Tensor, scaler: Scaler,
             batch_size: int = 256) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(model(x[start:start+batch_size]).squeeze(-1).numpy())
    return scaler.inverse(np.concatenate(parts))


def fit_fold(values: np.ndarray, history: int, fold: Fold, architecture: str,
             hidden: int, learning_rate: float, seed: int, *,
             max_epochs: int = 100, patience: int = 10, batch_size: int = 256,
             weight_decay: float = 1e-4, levels: np.ndarray | None = None,
             base: np.ndarray | None = None) -> dict:
    started = time.perf_counter()
    levels = values if levels is None else levels
    base = np.zeros_like(values) if base is None else base
    scaler = Scaler.fit(values, fold.fit_end)
    train_x, train_y, _ = windows(values, scaler, history, history, fold.fit_end)
    stop_x, _, stop_indices = windows(values, scaler, history, fold.fit_end, fold.stop_end)
    valid_x, _, valid_indices = windows(values, scaler, history, fold.stop_end, fold.valid_end)
    _seed(seed)
    model = MODELS[architecture](hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    best_state = None
    best_epoch = 0
    best_stop = float("inf")
    wait = 0
    curves = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for start in range(0, len(train_x), batch_size):
            optimiser.zero_grad(set_to_none=True)
            pred = model(train_x[start:start+batch_size])
            loss = torch.mean((pred - train_y[start:start+batch_size]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))
        stop_pred = _predict(model, stop_x, scaler) + base[stop_indices]
        stop_mae = scores(levels[stop_indices], stop_pred, 1.0)["mae"]
        curves.append({"epoch": epoch, "train_mse_scaled": float(np.mean(losses)),
                       "stop_mae": stop_mae})
        if stop_mae < best_stop - 1e-10:
            best_stop = stop_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    forecast = _predict(model, valid_x, scaler) + base[valid_indices]
    scale = training_mase_scale(levels[:fold.fit_end])
    result = scores(levels[valid_indices], forecast, scale)
    return {"dataset_metrics": result, "best_epoch": best_epoch,
            "epochs_run": len(curves), "best_stop_mae": best_stop,
            "parameter_count": count_parameters(model),
            "scaler": scaler.__dict__, "curve": curves,
            "indices": valid_indices.tolist(), "actual": levels[valid_indices].tolist(),
            "forecast": forecast.tolist(), "runtime_seconds": time.perf_counter()-started}


def fit_final(values: np.ndarray, history: int, development_end: int,
              architecture: str, hidden: int, learning_rate: float,
              seed: int, epochs: int, *, batch_size: int = 256,
              weight_decay: float = 1e-4, levels: np.ndarray | None = None,
              base: np.ndarray | None = None) -> dict:
    started = time.perf_counter()
    levels = values if levels is None else levels
    base = np.zeros_like(values) if base is None else base
    scaler = Scaler.fit(values, development_end)
    train_x, train_y, _ = windows(values, scaler, history, history, development_end)
    test_x, _, test_indices = windows(values, scaler, history, development_end, len(values))
    _seed(seed)
    model = MODELS[architecture](hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(train_x), batch_size):
            optimiser.zero_grad(set_to_none=True)
            pred = model(train_x[start:start+batch_size])
            loss = torch.mean((pred - train_y[start:start+batch_size]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
    forecast = _predict(model, test_x, scaler) + base[test_indices]
    result = scores(levels[test_indices], forecast,
                    training_mase_scale(levels[:development_end]))
    return {"dataset_metrics": result, "parameter_count": count_parameters(model),
            "epochs": epochs, "scaler": scaler.__dict__, "indices": test_indices.tolist(),
            "actual": levels[test_indices].tolist(), "forecast": forecast.tolist(),
            "runtime_seconds": time.perf_counter()-started}
