"""
generate_virginia_geo.py - Build data/virginia_geo.csv from the Census Gazetteer.

Creates the county lat/lon lookup table required by weatherapi.py.
No API key needed.

Usage:
    python generate_virginia_geo.py

Output:
    data/virginia_geo.csv  (fips, county_name, latitude, longitude)
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)
OUTPUT = Path("data/virginia_geo.csv")


def main():
    """Download the Census Gazetteer, filter to Virginia, and write data/virginia_geo.csv."""
    OUTPUT.parent.mkdir(exist_ok=True)

    print("Fetching Census Gazetteer ...")
    req = urllib.request.Request(_GAZETTEER_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        txt = next(n for n in zf.namelist() if n.endswith(".txt"))
        content = zf.read(txt).decode("utf-8")

    df = pd.read_csv(io.StringIO(content), sep="\t", dtype={"GEOID": str})
    df.columns = df.columns.str.strip()

    va = df[df["USPS"] == "VA"][["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]].copy()
    va.columns = ["fips", "county_name", "latitude", "longitude"]
    va = va.reset_index(drop=True)

    va.to_csv(OUTPUT, index=False)
    print(f"Saved {len(va)} Virginia counties to {OUTPUT}")


if __name__ == "__main__":
    main()
