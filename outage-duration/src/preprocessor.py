"""
preprocessor.py - Data cleaning and feature engineering for outage prediction.

Transforms raw EAGLE-I snapshot data and NOAA weather events into
model-ready features. Handles outage duration calculation from
consecutive snapshots, temporal feature extraction, and train-ready
feature/target splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def extract_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = 'run_start_time',
) -> pd.DataFrame:
    """Extract time-based features from a timestamp column.

    Creates year, month, day, hour, and dayofweek columns from
    the specified timestamp.

    Args:
        df: DataFrame with a datetime timestamp column.
        timestamp_col: Name of the column to extract features from.
            Defaults to 'run_start_time'.

    Returns:
        New DataFrame with added columns: year, month, day, hour, dayofweek.

    Raises:
        ValueError: If timestamp_col is not found in the DataFrame.

    Examples:
        >>> df = extract_temporal_features(outages_df)
        >>> df[['year', 'month', 'hour']].head()
           year  month  hour
        0  2022      1     0
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Column '{timestamp_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"[preprocessor] Extracting temporal features from '{timestamp_col}' ...")

    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df['year'] = ts.dt.year
    df['month'] = ts.dt.month
    df['day'] = ts.dt.day
    df['hour'] = ts.dt.hour
    df['dayofweek'] = ts.dt.dayofweek

    print(f"[preprocessor] Added 5 temporal features "
          f"(year range: {df['year'].min()}-{df['year'].max()})")

    return df


def calculate_duration(
    df: pd.DataFrame,
    fips_col: str = "fips_code",
    time_col: str = "run_start_time",
    affected_col: str = "customers_affected",
    snapshot_minutes: int = 15,
    gap_multiplier: int = 2,   # break outage if gap > 2 * snapshot interval
) -> pd.DataFrame:
    """Convert EAGLE-I snapshots -> outage events + early-outage features (no leakage)."""

    for col in [fips_col, time_col, affected_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    print("[preprocessor] Calculating outage durations from snapshots ...")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df = df.sort_values([fips_col, time_col]).reset_index(drop=True)

    # outage flag
    df["is_outage"] = df[affected_col] > 0

    # NEW: break blocks on large time gaps (prevents merging separate outages)
    df["time_diff_min"] = (
        df.groupby(fips_col)[time_col]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    df["gap_break"] = df["time_diff_min"] > (gap_multiplier * snapshot_minutes)

    # contiguous blocks per county, with gap break
    df["block_start"] = (
        (df["is_outage"] != df["is_outage"].shift(1))
        | (df[fips_col] != df[fips_col].shift(1))
        | (df["gap_break"].fillna(False))
    )
    df["block"] = df["block_start"].cumsum()

    outage_rows = df[df["is_outage"]].copy()
    if len(outage_rows) == 0:
        print("[preprocessor] WARNING: No outage periods found (all customers_affected == 0)")
        return pd.DataFrame()

    outage_rows = outage_rows.sort_values([fips_col, time_col])

    # ---- EARLY OUTAGE FEATURES (no leakage) ----
    first_rows = (
        outage_rows.groupby("block", as_index=False)
        .first()
        .rename(columns={time_col: "start_time", affected_col: "initial_customers_affected"})
    )

    second_rows = (
        outage_rows.groupby("block")
        .nth(1)
        .reset_index()
        .rename(columns={time_col: "second_time", affected_col: "second_customers_affected"})
    )

    early = first_rows.merge(
        second_rows[["block", "second_customers_affected"]],
        on="block",
        how="left",
    )

    early["second_customers_affected"] = early["second_customers_affected"].fillna(
        early["initial_customers_affected"]
    )
    early["delta_customers_affected_15m"] = (
        early["second_customers_affected"] - early["initial_customers_affected"]
    )
    denom = early["initial_customers_affected"].clip(lower=1)
    early["pct_growth_15m"] = (early["delta_customers_affected_15m"] / denom).astype(float)

    # ---- EVENT-LEVEL AGGREGATION ----
    agg_dict = {
        time_col: ["last"],
        affected_col: "max",
        "county": "first" if "county" in outage_rows.columns else "first",
        "state": "first" if "state" in outage_rows.columns else "first",
        fips_col: "first",
    }
    if "event_type" in outage_rows.columns:
        agg_dict["event_type"] = "first"
    if "magnitude" in outage_rows.columns:
        agg_dict["magnitude"] = "max"

    # Pass GHCN-Daily daily weather columns through to the event level.
    # Use "first" since GHCN data is daily -- all snapshots on the same day
    # have the same value, and we want the weather at outage START.
    _ghcnd_cols = [
        "prcp_mm", "snow_mm", "snwd_mm", "tmax_c", "tmin_c", "awnd_ms",
        "wt_fog", "wt_thunder", "wt_ice", "wt_blowing_snow", "wt_freezing_rain", "wt_snow",
    ]
    for col in _ghcnd_cols:
        if col in outage_rows.columns:
            agg_dict[col] = "first"

    events = outage_rows.groupby("block").agg(agg_dict)
    events.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in events.columns]
    events = events.reset_index()

    rename_map = {
        f"{time_col}_last": "end_time",
        f"{affected_col}_max": "peak_customers_affected",
        f"{fips_col}_first": "fips_code",
    }
    if "county_first" in events.columns:
        rename_map["county_first"] = "county"
    if "state_first" in events.columns:
        rename_map["state_first"] = "state"
    if "event_type_first" in events.columns:
        rename_map["event_type_first"] = "event_type"
    if "magnitude_max" in events.columns:
        rename_map["magnitude_max"] = "max_magnitude"

    # GHCN-Daily columns come through as colname_first after groupby flatten
    for col in _ghcnd_cols:
        flat = f"{col}_first"
        if flat in events.columns:
            rename_map[flat] = col

    events = events.rename(columns=rename_map)

    # add early features (start_time + early growth)
    events = events.merge(
        early[
            [
                "block",
                "start_time",
                "initial_customers_affected",
                "delta_customers_affected_15m",
                "pct_growth_15m",
            ]
        ],
        on="block",
        how="left",
    )

    # duration
    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce")
    events["duration_minutes"] = (
        (events["end_time"] - events["start_time"]).dt.total_seconds() / 60
    ) + snapshot_minutes

    # weather summary across outage snapshots
    if "event_type" in outage_rows.columns:
        weather_counts = (
            outage_rows.groupby("block")["event_type"]
            .apply(lambda s: int(s.notna().sum()))
            .reset_index(name="weather_event_count")
        )
        events = events.merge(weather_counts, on="block", how="left")
        events["weather_event_count"] = events["weather_event_count"].fillna(0).astype(int)
        events["has_weather_event"] = (events["weather_event_count"] > 0).astype(int)

    if "max_magnitude" in events.columns:
        events["max_magnitude"] = pd.to_numeric(events["max_magnitude"], errors="coerce")
        events["magnitude_missing"] = events["max_magnitude"].isna().astype(int)
        events["max_magnitude"] = events["max_magnitude"].fillna(0.0)

    events = events.drop(columns=["block"])

    print(
        f"[preprocessor] Found {len(events):,} outage events "
        f"(median duration: {events['duration_minutes'].median():.0f} min)"
    )
    return events




def filter_valid_durations(
    df: pd.DataFrame,
    min_minutes: float = 0,
    max_minutes: float = 10080,
) -> pd.DataFrame:
    """Remove outage events with invalid durations.

    Filters out events with negative, zero, or excessively long durations.
    The default max of 10080 minutes = 7 days.

    Args:
        df: DataFrame with a 'duration_minutes' column.
        min_minutes: Minimum valid duration (exclusive). Defaults to 0.
        max_minutes: Maximum valid duration (inclusive). Defaults to 10080 (7 days).

    Returns:
        Filtered DataFrame with only valid-duration events.

    Raises:
        ValueError: If 'duration_minutes' column is not present.

    Examples:
        >>> filtered = filter_valid_durations(events_df)
        >>> filtered['duration_minutes'].min() > 0
        True
        >>> filtered['duration_minutes'].max() <= 10080
        True
    """
    if 'duration_minutes' not in df.columns:
        raise ValueError("DataFrame must have a 'duration_minutes' column")

    print("[preprocessor] Filtering invalid durations ...")

    before = len(df)

    mask = (df['duration_minutes'] > min_minutes) & (df['duration_minutes'] <= max_minutes)
    df_filtered = df[mask].copy()

    removed = before - len(df_filtered)
    print(f"[preprocessor] Removed {removed:,} events "
          f"(kept {len(df_filtered):,} of {before:,})")

    if removed > 0:
        neg_count = (df['duration_minutes'] <= 0).sum()
        long_count = (df['duration_minutes'] > max_minutes).sum()
        null_count = df['duration_minutes'].isna().sum()
        print(f"[preprocessor]   <= 0 min: {neg_count:,}, "
              f"> {max_minutes} min: {long_count:,}, null: {null_count:,}")

    return df_filtered.reset_index(drop=True)


def create_duration_category(
    df: pd.DataFrame,
    threshold_minutes: float = 240,
) -> pd.DataFrame:
    """Add a binary flag for short vs long outages.

    Creates 'is_short_outage' column: True if duration < threshold (4 hours
    by default), False otherwise. This aligns with the PRD tolerance tiers.

    Args:
        df: DataFrame with a 'duration_minutes' column.
        threshold_minutes: Cutoff in minutes. Defaults to 240 (4 hours).

    Returns:
        DataFrame with added 'is_short_outage' boolean column.

    Raises:
        ValueError: If 'duration_minutes' column is not present.

    Examples:
        >>> df = create_duration_category(events_df)
        >>> df['is_short_outage'].value_counts()
        True     8500
        False    1500
        Name: is_short_outage, dtype: int64
    """
    if 'duration_minutes' not in df.columns:
        raise ValueError("DataFrame must have a 'duration_minutes' column")

    df = df.copy()
    df['is_short_outage'] = df['duration_minutes'] < threshold_minutes

    short = df['is_short_outage'].sum()
    total = len(df)
    print(f"[preprocessor] Duration categories: "
          f"{short:,} short (<{threshold_minutes} min), "
          f"{total - short:,} long (>={threshold_minutes} min)")

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "duration_minutes",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (target) for outage duration modeling.

    This version:
      - uses only high-signal features (location, early outage dynamics, weather severity, timing)
      - one-hot encodes event_type (and does NOT drop most rows)
      - returns numeric-only X for XGBoost
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    df = df.copy()

    # ----------------------------
    # 1) Choose "important" base features
    # ----------------------------
    if feature_cols is None:
        feature_cols = [
            # location
            "fips_code",
            # operational timing
            "month",
            "hour",
            "dayofweek",
            # early outage dynamics
            "initial_customers_affected",
            "delta_customers_affected_15m",
            "pct_growth_15m",
            # NOAA Storm Events context (named events only, ~7% coverage)
            "has_weather_event",
            "max_magnitude",
            "magnitude_missing",
            # GHCN-Daily measured weather (~91% coverage)
            "prcp_mm",
            "snow_mm",
            "snwd_mm",
            "tmax_c",
            "tmin_c",
            "awnd_ms",
            "wt_fog",
            "wt_thunder",
            "wt_ice",
            "wt_blowing_snow",
            "wt_freezing_rain",
            "wt_snow",
        ]

    # Keep only columns that actually exist (avoid crashing if a feature wasn't created)
    base_available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[preprocessor] WARNING: Missing engineered features (skipping): {missing}")

    # ----------------------------
    # 2) Handle event_type -> one-hot
    # ----------------------------
    # If your merge produced event_type, create event_* columns here.
    if "event_type" in df.columns:
        # Fill NaN as "None" so unmatched weather doesn't get dropped
        df["event_type"] = df["event_type"].fillna("None").astype(str)

        # One-hot encode, keep all categories
        dummies = pd.get_dummies(df["event_type"], prefix="event")

        # Attach to df
        df = pd.concat([df, dummies], axis=1)

    # If event_* columns already exist (from earlier steps), we still include them.
    event_cols = [c for c in df.columns if c.startswith("event_")]

    # ----------------------------
    # 3) Build final feature list
    # ----------------------------
    final_features = base_available + event_cols

    # De-duplicate while preserving order
    seen = set()
    final_features = [c for c in final_features if not (c in seen or seen.add(c))]

    print(f"[preprocessor] Preparing features ({len(final_features)}): {final_features}")

    # ----------------------------
    # 4) Coerce types to numeric
    # ----------------------------
    if "fips_code" in final_features:
        df["fips_code"] = pd.to_numeric(df["fips_code"], errors="coerce")

    # Make sure boolean dummy cols become 0/1 ints (sometimes pandas keeps True/False)
    for c in event_cols:
        df[c] = df[c].astype(int)

    # Coerce any remaining columns to numeric where reasonable
    for c in final_features:
        if c not in df.columns:
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------------------
    # 5) Drop rows with NaN ONLY in core always-present features + target
    # ----------------------------
    # GHCN-Daily and weather event features are supplemental -- a missing
    # value just means no nearby station reported that day, not bad data.
    # Those NaNs get filled with 0 below. Only drop rows that are missing
    # the features we know will always exist.
    _core = {
        "fips_code", "month", "hour", "dayofweek",
        "initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m",
    }
    subset = [c for c in base_available if c in _core] + [target_col]
    before = len(df)
    df = df.dropna(subset=subset)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[preprocessor] Dropped {dropped:,} rows with NaN in {subset}")

    X = df[final_features].copy()
    y = df[target_col].copy()

    # Final sanity: ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[preprocessor] Final dataset: {len(X):,} samples, {X.shape[1]} features")

    return X, y



def run_full_pipeline(
    outages_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    timestamp_col: str = 'run_start_time',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Run the complete preprocessing pipeline end-to-end.

    Chains all preprocessing steps: temporal features, duration calculation,
    filtering, categorization, and feature preparation.

    Args:
        outages_df: Raw EAGLE-I DataFrame (from load_eagle_outages).
        weather_df: Optional NOAA weather DataFrame. If provided, outage
            data should already be merged (via merge_weather_outages).
        timestamp_col: Timestamp column name. Defaults to 'run_start_time'.

    Returns:
        Tuple of (X, y) ready for model training.

    Examples:
        >>> from src.data_loader import load_eagle_outages, load_noaa_weather
        >>> from src.data_loader import merge_weather_outages
        >>> outages = load_eagle_outages('data/eaglei_outages_2022.csv')
        >>> weather = load_noaa_weather('data/storm_data_search_results.csv')
        >>> merged = merge_weather_outages(outages, weather)
        >>> X, y = run_full_pipeline(merged)
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING PIPELINE START")
    print("=" * 50)

    # Step 1: Extract temporal features
    df = extract_temporal_features(outages_df, timestamp_col)

    # Step 2: Calculate outage durations from snapshots
    df = calculate_duration(df)

    if len(df) == 0:
        raise ValueError("No outage events found after duration calculation")
    
    # One-hot encode event_type into event_* columns (keep NaNs as "None")
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].fillna("None")
        df = pd.get_dummies(df, columns=["event_type"], prefix="event")

    # Step 3: Filter invalid durations
    df = filter_valid_durations(df)

    # Step 4: Add duration category
    df = create_duration_category(df)

    # Step 5: Extract temporal features on the event start_time
    df = extract_temporal_features(df, timestamp_col='start_time')

    # Step 6: Prepare final features and target
    X, y = prepare_features(df)

    print("=" * 50)
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"Output: {X.shape[0]:,} samples x {X.shape[1]} features")
    print("=" * 50 + "\n")

    return X, y
