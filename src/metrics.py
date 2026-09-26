
import numpy as np


def scores(actual: np.ndarray, forecast: np.ndarray, mase_scale: float) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("Actual and forecast must be nonempty arrays with equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Metrics require finite observed targets and predictions")
    error = actual - forecast
    mae = float(np.mean(np.abs(error)))
    return {"mae": mae, "rmse": float(np.sqrt(np.mean(error**2))),
            "mase": mae / mase_scale if mase_scale > 0 else float("nan")}


def training_mase_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    if not valid_pairs.any():
        return float("nan")
    return float(np.mean(np.abs(values[1:][valid_pairs] - values[:-1][valid_pairs])))
