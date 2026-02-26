"""
test_pipeline.py - Smoke test for the data loading and preprocessing pipeline.

Loads sample EAGLE-I and NOAA data, runs the full preprocessing pipeline,
and prints summary output to verify everything works end-to-end.
"""

from pathlib import Path
from src.data_loader import (
    load_eagle_outages,
    load_noaa_weather,
    merge_weather_outages,
    load_ghcnd_weather,      # Added
    merge_ghcnd_weather,     # Added
    validate_data,
)
from src.preprocessor import run_full_pipeline


def main():
    data_dir = Path('data')

    # --- Step 1: Load EAGLE-I outage data ---
    print("\n>>> STEP 1: Loading EAGLE-I outage data")
    eagle_files = sorted(data_dir.glob('eaglei_outages_*.csv'))
    if not eagle_files:
        print("ERROR: No EAGLE-I files found. Run download_eaglei_data.py first.")
        return
    print(f"    Found {len(eagle_files)} EAGLE-I file(s): "
          f"{[f.name for f in eagle_files]}")
    outages = load_eagle_outages(eagle_files)
    print(f"    Shape: {outages.shape}")
    print(f"    Columns: {outages.columns.tolist()}")
    print(outages.head())

    # --- Step 2: Load NOAA weather data ---
    print("\n>>> STEP 2: Loading NOAA weather data")
    # Prefer the full 2014-2022 bulk download; fall back to the search export
    noaa_bulk = data_dir / 'noaa_storm_events_va_2014_2022.csv'
    noaa_fallback = data_dir / 'storm_data_search_results.csv'
    noaa_path = noaa_bulk if noaa_bulk.exists() else noaa_fallback
    weather = load_noaa_weather(noaa_path)
    print(f"    Shape: {weather.shape}")
    print(f"    Event types: {weather['event_type'].unique().tolist()}")
    print(weather.head())

    # --- Step 3: Validate raw data ---
    print("\n>>> STEP 3: Validating raw data")
    validate_data(outages)
    validate_data(weather)

    # --- Step 4: Merge weather with outages ---
    print("\n>>> STEP 4: Merging weather and outage data")
    merged = merge_weather_outages(outages, weather)
    
    print("    Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")
    merged = merge_ghcnd_weather(merged, ghcnd)
    
    print(f"    Merged shape: {merged.shape}")

    # --- Step 5: Run full preprocessing pipeline ---
    print("\n>>> STEP 5: Running full preprocessing pipeline")
    X, y = run_full_pipeline(merged)

    # --- Step 6: Show final output ---
    print("\n>>> FINAL OUTPUT")
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Features: {X.columns.tolist()}")
    print(f"\n    X (first 5 rows):")
    print(X.head().to_string(index=False))
    print(f"\n    y (first 5 values):")
    print(y.head().to_string(index=False))
    print(f"\n    y stats: mean={y.mean():.1f}, median={y.median():.1f}, "
          f"min={y.min():.1f}, max={y.max():.1f}")


if __name__ == '__main__':
    main()
