
import json
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from .data import load_series
from .splits import boundaries
from .training import causal_fill


def diagnostic_pvalues(x: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"adf_unit_root_null_p": float(adfuller(x, autolag="AIC")[1]),
                "kpss_level_stationarity_null_p": float(kpss(x, regression="c", nlags="auto")[1])}


def main() -> None:
    rows = []
    for key in ["co2", "sunspots", "power", "retail", "rain"]:
        data = load_series(key)
        n = len(data.values)
        development_end, block, folds = boundaries(n)
        level = data.levels.to_numpy(dtype=float)
        transformed = data.values.to_numpy(dtype=float)
        original = causal_fill(level[:development_end])
        modelled = causal_fill(transformed[:development_end])
        first, second = np.array_split(original, 2)
        row = {"dataset": key, "modelled_n": n,
               "first_date": str(data.values.index[0].date()),
               "last_date": str(data.values.index[-1].date()),
               "test_start": str(data.values.index[development_end].date()),
               "development_n": development_end,
               "test_n": n-development_end,
               "development_missing": int(np.isnan(level[:development_end]).sum()),
               "test_missing_count_for_integrity_only": int(np.isnan(level[development_end:]).sum()),
               "level_min_development": float(np.nanmin(level[:development_end])),
               "level_max_development": float(np.nanmax(level[:development_end])),
               "first_half_mean": float(first.mean()),
               "second_half_mean": float(second.mean()),
               "development_level": diagnostic_pvalues(original),
               "development_modelled_target": diagnostic_pvalues(modelled),
               "folds": [fold.__dict__ for fold in folds],
               "unit": data.unit, "source_note": data.source_note}
        rows.append(row)
    destination = Path("output/diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data_audit.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row["dataset"], row["development_level"], row["development_modelled_target"])


if __name__ == "__main__":
    main()
