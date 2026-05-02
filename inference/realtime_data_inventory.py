"""
realtime_data_inventory.py - Documents every real-time data type available
and maps each to the model features it satisfies.

Run this to print the inventory:
    python realtime_data_inventory.py
"""

# ── What each API provides ────────────────────────────────────────────────────

OPEN_METEO_FIELDS = {
    # hourly
    "temperature_2m":    "air temp at 2m, °C (hourly)",
    "precipitation":     "total precip, mm (hourly)  -> prcp_mm",
    "snowfall":          "snowfall, cm (hourly)       -> snow_mm",
    "snow_depth":        "snow depth on ground, m     -> snwd_mm",
    "windspeed_10m":     "avg wind speed, km/h        -> awnd_ms",
    "windgusts_10m":     "peak wind gust, km/h        -> wsfg_ms",
    "weathercode":       "WMO weather code (hourly)   -> wt_* flags",
    # daily
    "temperature_2m_max": "daily max temp, °C          -> tmax_c",
    "temperature_2m_min": "daily min temp, °C          -> tmin_c",
}

CENSUS_FIELDS = {
    "NAME":         "county name",
    "state":        "state FIPS (51 = Virginia)",
    "county":       "county FIPS suffix",
    # derived
    "fips_code":    "full 5-digit FIPS (state+county)",
    "latitude":     "county centroid lat (from Google Places or Nominatim)",
    "longitude":    "county centroid lon",
}

KUBRA_DOMINION_FIELDS = {
    "title":               "county name",
    "desc.cust_a.val":     "customers currently without power  -> initial_customers_affected",
    "desc.cust_s":         "total customers served in county",
    "desc.n_out":          "number of distinct outage incidents",
    "desc.percent_cust_a": "% customers affected",
    "desc.etr":            "estimated restoration time (ISO 8601 or ETR-NULL)",
    "desc.hierarchy":      "sub-region grouping",
}

KUBRA_APPALACHIAN_FIELDS = {
    "title":               "county / city name",
    "desc.cust_a.val":     "customers currently without power  -> initial_customers_affected",
    "desc.cust_s":         "total customers served",
    "desc.n_out":          "number of outage incidents",
    "desc.percent_cust_a": "% customers affected",
    "desc.etr":            "estimated restoration time (ISO 8601)",
    "desc.etr_confidence": "confidence level: HIGH / LOW",
    "desc.start_time":     "when outage(s) began",
}

# ── WMO weather code -> model flag mapping ────────────────────────────────────

WEATHERCODE_MAP = {
    "wt_fog":           [45, 48],
    "wt_thunder":       [95, 96, 99],
    "wt_snow":          [71, 73, 75, 77, 85, 86],
    "wt_freezing_rain": [66, 67],
    "wt_ice":           [66, 67],
    "wt_blowing_snow":  [77],
    "wt_drizzle":       [51, 53, 55],
    "wt_rain":          [61, 63, 65, 80, 81, 82],
}

# ── Model feature coverage ────────────────────────────────────────────────────

# source: occurrence model preprocessor_occurrence.py + scope/duration preprocessor.py
MODEL_FEATURES = {
    # Temporal — derived from current datetime, no API needed
    "year":         ("derived", "current date"),
    "month":        ("derived", "current date"),
    "day":          ("derived", "current date"),
    "hour":         ("derived", "current date"),
    "dayofweek":    ("derived", "current date"),
    "dayofyear":    ("derived", "current date"),
    "is_weekend":   ("derived", "current date"),

    # Spatial
    "fips_code":    ("Census API", "5-digit FIPS per county"),

    # Continuous weather — Open-Meteo
    "prcp_mm":      ("Open-Meteo", "precipitation (hourly, aggregate to daily for occurrence)"),
    "snow_mm":      ("Open-Meteo", "snowfall"),
    "snwd_mm":      ("Open-Meteo", "snow_depth — ADD to params"),
    "tmax_c":       ("Open-Meteo", "temperature_2m_max (daily)"),
    "tmin_c":       ("Open-Meteo", "temperature_2m_min (daily)"),
    "awnd_ms":      ("Open-Meteo", "windspeed_10m — convert km/h -> m/s (/3.6)"),
    "wsfg_ms":      ("Open-Meteo", "windgusts_10m — ADD to params, convert km/h -> m/s"),

    # Binary weather flags — derived from Open-Meteo weathercode
    "wt_fog":           ("Open-Meteo", "weathercode in [45,48]"),
    "wt_thunder":       ("Open-Meteo", "weathercode in [95,96,99]"),
    "wt_snow":          ("Open-Meteo", "weathercode in [71,73,75,77,85,86]"),
    "wt_freezing_rain": ("Open-Meteo", "weathercode in [66,67]"),
    "wt_ice":           ("Open-Meteo", "weathercode in [66,67]"),
    "wt_blowing_snow":  ("Open-Meteo", "weathercode in [77]"),
    "wt_drizzle":       ("Open-Meteo", "weathercode in [51,53,55] — ADD mapping"),
    "wt_rain":          ("Open-Meteo", "weathercode in [61,63,65,80,81,82] — ADD mapping"),

    # NOAA storm context — approximated from weather data
    "has_weather_event": ("Open-Meteo", "weathercode >= 95 (severe weather)"),
    "max_magnitude":     ("Open-Meteo", "max(prcp_mm, awnd_ms) as proxy"),
    "magnitude_missing": ("Open-Meteo", "0 (always present real-time)"),

    # Event type one-hot (event_Thunderstorm, event_Winter Storm, etc.)
    # — not available real-time; NOAA storm events are post-event records
    "event_*":      ("N/A", "default to zeros — no real-time NOAA named events"),

    # Live outage features (scope/duration models only) — from Kubra
    "initial_customers_affected":    ("Kubra", "desc.cust_a.val at first fetch"),
    "delta_customers_affected_15m":  ("Kubra", "cust_a[t+15min] - cust_a[t]  (requires two polls)"),
    "pct_growth_15m":                ("Kubra", "delta / initial  (requires two polls)"),

    # County historical stats — precomputed from training data, stored as lookup
    "county_median_duration": ("precomputed lookup", "median outage duration per FIPS"),
    "county_long_rate":       ("precomputed lookup", "rate of long (≥4hr) outages per FIPS"),
}


def main():
    print("=" * 60)
    print("REAL-TIME DATA INVENTORY")
    print("=" * 60)

    sources = {}
    for feature, (source, note) in MODEL_FEATURES.items():
        sources.setdefault(source, []).append((feature, note))

    for source, features in sources.items():
        print(f"\n[{source}]")
        for feature, note in features:
            print(f"  {feature:<35} {note}")

    print("\n" + "=" * 60)
    print("GAPS / ACTION ITEMS")
    print("=" * 60)
    gaps = [
        "Open-Meteo params: add snow_depth and windgusts_10m",
        "Open-Meteo weathercode map: add wt_drizzle and wt_rain",
        "Unit conversions: windspeed/gusts km/h -> m/s (/3.6); snow_depth m -> mm (×1000)",
        "event_* columns: fill with 0 (no real-time NOAA named storm data)",
        "Kubra delta features: requires two polls 15 min apart to compute growth",
        "County stats lookup: export county_median_duration + county_long_rate from training data",
    ]
    for i, gap in enumerate(gaps, 1):
        print(f"  {i}. {gap}")


if __name__ == "__main__":
    main()
