"""
export_county_stats.py - Compute county-level historical outage stats from EAGLE-I data.

These are precomputed lookup features used by the scope and duration models:
  - county_median_duration     median outage duration (minutes) per county
  - county_long_rate           fraction of outages >= 4 hours per county
  - county_median_scope        median peak customers affected per county
  - county_large_outage_rate   fraction of outages with >= 500 customers affected
  - county_max_customers       historical max customers affected (used for density)

Usage:
    python export_county_stats.py

Output:
    data/county_stats.csv
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT   = DATA_DIR / "county_stats.csv"

sys.path.insert(0, str(BASE_DIR))

from outage_duration.src.data_loader   import load_eagle_outages
from outage_duration.src.preprocessor  import (
    extract_temporal_features,
    calculate_duration,
    filter_valid_durations,
)


def main():
    """Compute per-county historical stats from EAGLE-I and write to data/county_stats.csv."""
    eagle_files = sorted(DATA_DIR.glob("eaglei_outages_*.csv"))
    print(f"Loading {len(eagle_files)} EAGLE-I files ...")
    outages = load_eagle_outages(eagle_files)

    df     = extract_temporal_features(outages)
    events = calculate_duration(df)
    events = filter_valid_durations(events)

    stats = (
        events.groupby("fips_code")
        .agg(
            county_median_duration    =("duration_minutes",      "median"),
            county_long_rate          =("duration_minutes",      lambda x: (x >= 240).mean()),
            county_median_scope       =("peak_customers_affected","median"),
            county_large_outage_rate  =("peak_customers_affected",lambda x: (x >= 500).mean()),
            county_max_customers      =("peak_customers_affected","max"),
        )
        .reset_index()
    )
    stats["county_max_customers"] = stats["county_max_customers"].clip(lower=1)

    stats.to_csv(OUTPUT, index=False)
    print(f"Saved stats for {len(stats)} counties to {OUTPUT}")
    print(stats.describe().to_string())


if __name__ == "__main__":
    main()
