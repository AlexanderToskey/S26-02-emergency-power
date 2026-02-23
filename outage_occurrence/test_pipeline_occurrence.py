from pathlib import Path
from data_loader_occurrence import (
    load_eagle_outages,
    load_ghcnd_weather,
    build_occurrence_labels,
    merge_occurrence_with_weather,
    validate_data,
    summarize_class_balance,
)
from preprocessor_occurrence import run_full_pipeline


def main():
    data_dir = Path("data")

    # --- Step 1: Load EAGLE-I outage data ---
    print("\n>>> STEP 1: Loading EAGLE-I outage data")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    if not eagle_files:
        print("ERROR: No EAGLE-I files found. Run download_eaglei_data.py first.")
        return

    print(f"    Found {len(eagle_files)} file(s): {[f.name for f in eagle_files]}")
    outages = load_eagle_outages(eagle_files)

    print(f"    Raw outage shape: {outages.shape}")
    print(outages.head())

    # --- Step 2: Build county-day occurrence labels ---
    print("\n>>> STEP 2: Building county-day occurrence labels")
    occurrence = build_occurrence_labels(outages)

    print(f"    Occurrence shape: {occurrence.shape}")
    print(occurrence.head())

    # --- Step 3: Load GHCN-Daily weather data ---
    print("\n>>> STEP 3: Loading GHCN-Daily weather data")
    ghcnd_path = data_dir / "ghcnd_va_daily.csv"

    if not ghcnd_path.exists():
        print("ERROR: GHCN-Daily file not found. Run download_ghcnd_data.py first.")
        return

    weather = load_ghcnd_weather(ghcnd_path)

    print(f"    Weather shape: {weather.shape}")
    print(weather.head())

    # --- Step 4: Merge occurrence with weather ---
    print("\n>>> STEP 4: Merging occurrence labels with weather")
    merged = merge_occurrence_with_weather(
        occurrence,
        weather,
    )

    print(f"    Final merged shape: {merged.shape}")

    # --- Step 5: Validate merged dataset ---
    print("\n>>> STEP 5: Validating merged dataset")
    validate_data(merged)
    summarize_class_balance(merged)

    # --- Step 6: Run full preprocessing pipeline ---
    print("\n>>> STEP 6: Running full preprocessing pipeline")
    X, y = run_full_pipeline(merged)

    # --- Step 7: Show final output ---
    print("\n>>> FINAL OUTPUT")
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Features ({len(X.columns)}): {X.columns.tolist()}")

    print("\n    X (first 5 rows):")
    print(X.head().to_string(index=False))

    print("\n    y (first 10 values):")
    print(y.head(10).to_string(index=False))

    print(
        "\n    y stats:"
        f"\n        Positive rate: {y.mean():.4f}"
        f"\n        Positives: {int(y.sum())}"
        f"\n        Negatives: {len(y) - int(y.sum())}"
    )


if __name__ == "__main__":
    main()