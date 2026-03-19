"""
weatherapi.py - Fetch real-time weather for every Virginia county via Open-Meteo.

Queries Open-Meteo for each county centroid, maps the response to the exact
feature names expected by the outage prediction models, and saves a CSV.

No API key required.

Usage:
    python weatherapi.py

Output:
    data/virginia_county_weather.csv  (one row per county-hour)
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

GEO_FILE = Path("data/virginia_geo.csv")
OUTPUT   = Path("data/virginia_county_weather.csv")
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather code -> model binary flag mapping
_CODE_FLAGS = {
    "wt_fog":           {45, 48},
    "wt_thunder":       {95, 96, 99},
    "wt_snow":          {71, 73, 75, 77, 85, 86},
    "wt_freezing_rain": {66, 67},
    "wt_ice":           {66, 67},
    "wt_blowing_snow":  {77},
    "wt_drizzle":       {51, 53, 55},
    "wt_rain":          {61, 63, 65, 80, 81, 82},
}


def _weather_flags(code: int) -> dict:
    return {flag: int(code in codes) for flag, codes in _CODE_FLAGS.items()}


def fetch_county(lat: float, lon: float) -> dict | None:
    """Fetch today's hourly + daily weather for one lat/lon.

    Returns a dict with hourly lists and daily scalars, or None on error.
    """
    params = (
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,snowfall,snow_depth,"
        "windspeed_10m,windgusts_10m,weathercode"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=America%2FNew_York"
    )
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, Exception) as e:
        return None


def parse_county(fips: str, data: dict) -> list[dict]:
    """Convert one Open-Meteo response into a list of hourly model-ready rows."""
    hourly = data["hourly"]
    daily  = data["daily"]

    tmax = daily["temperature_2m_max"][0]
    tmin = daily["temperature_2m_min"][0]

    rows = []
    for i, time_str in enumerate(hourly["time"]):
        dt = pd.to_datetime(time_str)

        code    = int(hourly["weathercode"][i]  or 0)
        wind_kh = hourly["windspeed_10m"][i]    or 0.0
        gust_kh = hourly["windgusts_10m"][i]    or 0.0
        snwd_m  = hourly["snow_depth"][i]       or 0.0

        row = {
            "fips_code":    fips,
            "year":         dt.year,
            "month":        dt.month,
            "day":          dt.day,
            "hour":         dt.hour,
            "dayofweek":    dt.dayofweek,
            # continuous weather (model units)
            "prcp_mm":      hourly["precipitation"][i] or 0.0,
            "snow_mm":      (hourly["snowfall"][i]     or 0.0) * 10,  # cm -> mm
            "snwd_mm":      snwd_m * 1000,                             # m  -> mm
            "tmax_c":       tmax,
            "tmin_c":       tmin,
            "awnd_ms":      round(wind_kh / 3.6, 3),                  # km/h -> m/s
            "wsfg_ms":      round(gust_kh / 3.6, 3),                  # km/h -> m/s
            # derived storm context
            "has_weather_event": int(code >= 95),
            "max_magnitude":     max(hourly["precipitation"][i] or 0.0,
                                     round(wind_kh / 3.6, 3)),
            "magnitude_missing": 0,
        }
        row.update(_weather_flags(code))
        rows.append(row)

    return rows


def main():
    geo = pd.read_csv(GEO_FILE, dtype={"fips": str})
    print(f"Fetching weather for {len(geo)} Virginia counties ...")

    all_rows = []
    for _, county in geo.iterrows():
        fips = county["fips"]
        lat  = county["latitude"]
        lon  = county["longitude"]

        data = fetch_county(lat, lon)
        if data is None:
            print(f"  [WARN] {county['county_name']} ({fips}): fetch failed, skipping")
            continue

        all_rows.extend(parse_county(fips, data))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows ({df['fips_code'].nunique()} counties) to {OUTPUT}")


if __name__ == "__main__":
    main()
