import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Load geo data
geo_df = pd.read_csv("virginia_geo.csv")

# Drop missing coords
geo_df = geo_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

# Setup API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
client = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"

# Prepare batched coordinates
latitudes = geo_df["latitude"].tolist()
longitudes = geo_df["longitude"].tolist()

params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": [
        "temperature_2m",
        "precipitation",
        "snowfall",
        "windspeed_10m",
        "weathercode"
    ],
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min"
    ],
    "forecast_days": 1,
    "timezone": "America/New_York"
}

def map_weather_flags(code):
    return {
        "wt_fog": int(code in [45, 48]),
        "wt_thunder": int(code in [95, 96, 99]),
        "wt_snow": int(code in [71, 73, 75, 77, 85, 86]),
        "wt_freezing_rain": int(code in [66, 67]),
        "wt_ice": int(code in [66, 67]),
        "wt_blowing_snow": int(code in [77])
    }

rows = []

# Make ONE API call
responses = client.weather_api(url, params=params)

# Loop through each county response
for idx, r in enumerate(responses):
    fips = geo_df.loc[idx, "fips"]

    try:
        # --- HOURLY ---
        hourly = r.Hourly()

        temp = hourly.Variables(0).ValuesAsNumpy()
        prcp = hourly.Variables(1).ValuesAsNumpy()
        snow = hourly.Variables(2).ValuesAsNumpy()
        wind = hourly.Variables(3).ValuesAsNumpy()
        codes = hourly.Variables(4).ValuesAsNumpy()

        start = hourly.Time()
        interval = hourly.Interval()
        n = len(temp)

        times = [start + i * interval for i in range(n)]

        # --- DAILY ---
        daily = r.Daily()
        tmax = daily.Variables(0).ValuesAsNumpy()[0]
        tmin = daily.Variables(1).ValuesAsNumpy()[0]

        # --- BUILD ROWS ---
        for i in range(n):
            dt = pd.to_datetime(times[i], unit="s")
            flags = map_weather_flags(codes[i])

            rows.append({
                "fips_code": fips,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "dayofweek": dt.dayofweek,

                # Weather
                "prcp_mm": prcp[i],
                "snow_mm": snow[i],
                "tmax_c": tmax,
                "tmin_c": tmin,
                "awnd_ms": wind[i],

                # Flags
                "wt_fog": flags["wt_fog"],
                "wt_thunder": flags["wt_thunder"],
                "wt_snow": flags["wt_snow"],
                "wt_freezing_rain": flags["wt_freezing_rain"],
                "wt_ice": flags["wt_ice"],
                "wt_blowing_snow": flags["wt_blowing_snow"],

                # Derived
                "has_weather_event": int(codes[i] >= 95),
                "max_magnitude": max(prcp[i], wind[i]),
                "magnitude_missing": 0
            })

    except Exception as e:
        print(f"Error for FIPS {fips}: {e}")

# Save output
df = pd.DataFrame(rows)
df.to_csv("virginia_county_weather.csv", index=False)

print("Batch CSV 'virginia_county_weather.csv' created successfully.")