
from __future__ import annotations

import hashlib
from pathlib import Path

import requests


RAW = Path("data/raw")
SOURCES = {
    "co2_mm_mlo.csv": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
    "SN_m_tot_V2.0.csv": "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
    "household_power_consumption.zip": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "ons_j448.csv": "https://www.ons.gov.uk/generator?format=csv&uri=/businessindustryandtrade/retailindustry/timeseries/j448/drsi",
    "HadEWP_monthly_totals.txt": "https://www.metoffice.gov.uk/hadobs/hadukp/data/monthly/HadEWP_monthly_totals.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        destination = RAW / filename
        if not destination.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if len(response.content) < 100:
                raise ValueError(f"Suspiciously short response from {url}")
            destination.write_bytes(response.content)
        print(f"{sha256(destination)}  {destination}  {url}")


if __name__ == "__main__":
    main()
