
from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW = Path("data/raw")


@dataclass(frozen=True)
class SeriesData:
    key: str
    values: pd.Series
    levels: pd.Series
    base: pd.Series
    difference_lag: int
    unit: str
    frequency: str
    seasonal_lag: int | None
    history: int
    source_note: str


def _co2() -> SeriesData:
    frame = pd.read_csv(RAW / "co2_mm_mlo.csv", comment="#", skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    frame = frame.loc[(frame.year >= 1958) & (frame.year <= 2025)]
    dates = pd.to_datetime(frame.year.astype(str) + "-" + frame.month.astype(str) + "-01")
    values = pd.Series(frame.average.to_numpy(dtype=float), index=dates, name="co2_ppm")
    return SeriesData("co2", values, values, pd.Series(0.0, index=values.index), 0,
                      "ppm", "month", 12, 24,
                      "NOAA average column; source includes historical interpolations and pre-1974 Scripps measurements")


def _sunspots() -> SeriesData:
    frame = pd.read_csv(RAW / "SN_m_tot_V2.0.csv", sep=";", header=None,
                        names=["year", "month", "decimal", "count", "sd", "n", "definitive"])
    frame = frame.loc[(frame.year >= 1900) & (frame.year <= 2025)]
    dates = pd.to_datetime(frame.year.astype(str) + "-" + frame.month.astype(str) + "-01")
    values = pd.Series(frame["count"].replace(-1, np.nan).to_numpy(dtype=float),
                       index=dates, name="sunspot_number")
    return SeriesData("sunspots", values, values, pd.Series(0.0, index=values.index), 0,
                      "sunspot number", "month", None, 60,
                      "SILSO version 2 monthly total; 2025 definitive flags audited separately")


def _power() -> SeriesData:
    with zipfile.ZipFile(RAW / "household_power_consumption.zip") as archive:
        with archive.open("household_power_consumption.txt") as stream:
            frame = pd.read_csv(stream, sep=";", usecols=["Date", "Time", "Global_active_power"],
                                na_values=["?"], low_memory=False)
    dates = pd.to_datetime(frame.Date + " " + frame.Time, format="%d/%m/%Y %H:%M:%S")
    minute = pd.Series(pd.to_numeric(frame.Global_active_power, errors="coerce").to_numpy(),
                       index=dates).sort_index()
    daily = minute.resample("D").agg(["mean", "count"])
    
    
    daily.loc[daily["count"] < 1368, "mean"] = np.nan  
    values = daily["mean"].rename("daily_mean_kw")
    values = values.loc[values.first_valid_index():values.last_valid_index()]
    return SeriesData("power", values, values, pd.Series(0.0, index=values.index), 0,
                      "kW", "day", 7, 28,
                      "UCI minute active power aggregated to daily mean; at least 95% coverage required")


def _retail() -> SeriesData:
    rows: list[tuple[pd.Timestamp, float]] = []
    with (RAW / "ons_j448.csv").open(newline="") as stream:
        for cells in csv.reader(stream):
            if len(cells) != 2 or not re.fullmatch(r"\d{4} [A-Z]{3}", cells[0]):
                continue
            date = pd.to_datetime(cells[0], format="%Y %b")
            if date.year <= 2025:
                rows.append((date, float(cells[1])))
    values = pd.Series({date: value for date, value in rows}, name="retail_index").sort_index()
    return SeriesData("retail", values, values, pd.Series(0.0, index=values.index), 0,
                      "volume index", "month", 12, 24,
                      "ONS J448 unadjusted all-retailers-ex-fuel volume index")


def _rain() -> SeriesData:
    rows: list[tuple[pd.Timestamp, float]] = []
    with (RAW / "HadEWP_monthly_totals.txt").open() as stream:
        for line in stream:
            cells = line.split()
            if len(cells) != 14 or not cells[0].isdigit():
                continue
            year = int(cells[0])
            if 1931 <= year <= 2025:
                rows.extend((pd.Timestamp(year, month, 1), float(cells[month]))
                            for month in range(1, 13))
    values = pd.Series({date: value for date, value in rows}, name="rain_mm").sort_index()
    values = values.replace(-99.9, np.nan)
    return SeriesData("rain", values, values, pd.Series(0.0, index=values.index), 0,
                      "mm", "month", 12, 24,
                      "Met Office HadEWP monthly England and Wales precipitation")


LOADERS = {"co2": _co2, "sunspots": _sunspots, "power": _power,
           "retail": _retail, "rain": _rain}
DIFFERENCE_LAGS = {"co2": 1, "retail": 12}


def load_series(key: str) -> SeriesData:
    data = LOADERS[key]()
    lag = DIFFERENCE_LAGS.get(key, 0)
    if lag:
        level = data.levels.iloc[lag:].copy()
        base = pd.Series(data.levels.iloc[:-lag].to_numpy(), index=level.index)
        transformed = level - base
        data = SeriesData(key, transformed, level, base, lag, data.unit,
                          data.frequency, data.seasonal_lag, data.history,
                          data.source_note)
    expected = "MS" if data.frequency == "month" else "D"
    assert data.values.index.is_monotonic_increasing
    assert data.values.index.is_unique
    assert len(data.values) == len(pd.date_range(data.values.index[0], data.values.index[-1], freq=expected))
    assert np.isfinite(data.values.dropna().to_numpy()).all()
    return data
