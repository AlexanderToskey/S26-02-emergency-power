"""
download_openmeteo_historical.py - Download historical daily weather for all Virginia counties.

Fetches hourly data from the Open-Meteo Archive API (2014-01-01 to 2022-12-31),
aggregates it to daily county-level records, and appends each county to
data/openmeteo_va_historical.csv. Supports resuming a partial download —
counties already in the output file are skipped automatically.

No API key required. Rate-limit handling (HTTP 429) is built in.

Usage:
    python download_openmeteo_historical.py

Output:
    data/openmeteo_va_historical.csv  (one row per county-day)
"""

import pandas as pd
import requests
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GEO_FILE = DATA_DIR / "virginia_geo.csv"
OUTPUT_FILE = DATA_DIR / "openmeteo_va_historical.csv"

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

def fetch_county_history(fips, lat, lon, start="2014-01-01", end="2022-12-31"):
    """Fetch and aggregate historical daily weather for one county from Open-Meteo Archive.

    Retrieves hourly data, aggregates to daily records (max/min temp, sum precip/snow,
    max snow depth, mean wind, max gusts, weather type flags), and converts units to
    match the training pipeline (km/h → m/s, cm → mm, m → mm).

    Args:
        fips: County FIPS code (used to label output rows).
        lat: Latitude of the county centroid.
        lon: Longitude of the county centroid.
        start: Start date string in YYYY-MM-DD format.
        end: End date string in YYYY-MM-DD format.

    Returns:
        DataFrame with one row per day and model-ready feature columns,
        or None if the fetch fails after all retries.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,precipitation,snowfall,snow_depth,wind_speed_10m,wind_gusts_10m,weather_code",
        "timezone": "America/New_York"
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:

                data = response.json()
                
                if "hourly" not in data:
                    return None

                # Process hourly to daily (to match ghcnd_va_daily.csv format)
                df_hourly = pd.DataFrame(data["hourly"])
                df_hourly['date'] = pd.to_datetime(df_hourly['time']).dt.date
                
                for flag, codes in _CODE_FLAGS.items():
                    df_hourly[flag] = df_hourly['weather_code'].isin(codes).astype(int)
                
                agg_map = {
                    'temperature_2m': ['max', 'min'],
                    'precipitation': 'sum',
                    'snowfall': 'sum',
                    'snow_depth': 'max',
                    'wind_speed_10m': 'mean',
                    'wind_gusts_10m': 'max'
                }
                for flag in _CODE_FLAGS.keys(): agg_map[flag] = 'max'

                # Aggregate logic matching preprocessor needs
                daily = df_hourly.groupby('date').agg(agg_map).reset_index()
                
                daily.columns = ['date', 'tmax_c', 'tmin_c', 'prcp_mm', 'snow_mm', 'snwd_mm', 'awnd_ms', 'wsfg_ms'] + list(_CODE_FLAGS.keys())

                # Convert km/h to m/s as done in realtime pipeline
                daily['awnd_ms'] = daily['awnd_ms'] / 3.6
                daily['wsfg_ms'] = daily['wsfg_ms'] / 3.6
                daily['fips_code'] = str(fips).zfill(5)
                
                return daily
            
            elif response.status_code == 429: # Too Many Requests
                wait_time = 65
                print(f"  [!] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            else:
                print(f"  [!] API Error {response.status_code} for {fips}")
                return None

        except Exception as e:
            print(f"Error fetching FIPS {fips}: {e}")
            return time.sleep(5)
    return None
    
def main():
    """Iterate over all Virginia counties and download historical weather, skipping completed ones."""
    geo = pd.read_csv(GEO_FILE, dtype={"fips": str})

    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE, dtype={"fips_code": str})
        finished_fips = set(existing_df['fips_code'].unique())
        print(f"Resuming: {len(finished_fips)} counties already downloaded.")
    else:
        finished_fips = set()
        print("Starting fresh download.")

    for _, row in geo.iterrows():
        fips = str(row['fips']).zfill(5)
        if fips in finished_fips:
            continue
        print(f"Fetching {row['county_name']} ({row['fips']})...")
        df = fetch_county_history(row['fips'], row['latitude'], row['longitude'])
        if df is not None:
            file_exists = OUTPUT_FILE.exists()
            df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
            print(f"  [+] Saved {row['county_name']}")
        time.sleep(1)  # Respect Open-Meteo rate limits

    # final_df = pd.concat(all_data)
    # final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()