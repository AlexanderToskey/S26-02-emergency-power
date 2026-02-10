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
    fips_col: str = 'fips_code',
    time_col: str = 'run_start_time',
    affected_col: str = 'customers_affected',
) -> pd.DataFrame:
    """Calculate outage duration in minutes from EAGLE-I snapshots.

    EAGLE-I data consists of periodic snapshots (rows) showing how many
    customers are affected at each timestamp per county. An outage is a
    contiguous stretch of snapshots where customers_affected > 0.
    Duration is computed as (last_snapshot_time - first_snapshot_time)
    for each contiguous outage event.

    Each row in the output represents one outage event (not a snapshot).

    Args:
        df: EAGLE-I DataFrame with snapshot data. Must have fips_code,
            run_start_time, and customers_affected columns.
        fips_col: Column identifying the county. Defaults to 'fips_code'.
        time_col: Timestamp column. Defaults to 'run_start_time'.
        affected_col: Column with customer count. Defaults to 'customers_affected'.

    Returns:
        DataFrame of outage events with columns: fips_code, county, state,
        start_time, end_time, duration_minutes, peak_customers_affected,
        plus any weather columns if present.

    Raises:
        ValueError: If required columns are missing.

    Examples:
        >>> events = calculate_duration(eagle_df)
        >>> events[['fips_code', 'duration_minutes']].head()
    """
    for col in [fips_col, time_col, affected_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    print("[preprocessor] Calculating outage durations from snapshots ...")

    df = df.copy()
    df = df.sort_values([fips_col, time_col]).reset_index(drop=True)

    # Flag rows where an outage is happening (customers > 0)
    df['is_outage'] = df[affected_col] > 0

    # Detect contiguous outage blocks per county
    # A new block starts when is_outage changes or fips_code changes
    df['block'] = (
        (df['is_outage'] != df['is_outage'].shift(1)) |
        (df[fips_col] != df[fips_col].shift(1))
    ).cumsum()

    # Keep only outage rows
    outage_rows = df[df['is_outage']].copy()

    if len(outage_rows) == 0:
        print("[preprocessor] WARNING: No outage periods found (all customers_affected == 0)")
        return pd.DataFrame()

    # Aggregate each contiguous outage block
    agg_dict = {
        time_col: ['first', 'last'],
        affected_col: 'max',
        'county': 'first',
        'state': 'first',
        fips_col: 'first',
    }

    # Include event_type if present (from weather merge)
    if 'event_type' in outage_rows.columns:
        agg_dict['event_type'] = 'first'

    events = outage_rows.groupby('block').agg(agg_dict)

    # Flatten multi-level columns
    events.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in events.columns
    ]

    # Rename to clean names
    rename_map = {
        f'{time_col}_first': 'start_time',
        f'{time_col}_last': 'end_time',
        f'{affected_col}_max': 'peak_customers_affected',
        'county_first': 'county',
        'state_first': 'state',
        f'{fips_col}_first': 'fips_code',
    }
    if 'event_type_first' in events.columns:
        rename_map['event_type_first'] = 'event_type'

    events = events.rename(columns=rename_map)

    # Calculate duration
    events['duration_minutes'] = (
        (events['end_time'] - events['start_time']).dt.total_seconds() / 60
    )

    events = events.reset_index(drop=True)

    print(f"[preprocessor] Found {len(events):,} outage events "
          f"(median duration: {events['duration_minutes'].median():.0f} min)")

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
    target_col: str = 'duration_minutes',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target vector y for modeling.

    Selects the specified feature columns and target, drops rows with
    any NaN in the selected columns, and returns clean arrays.

    Args:
        df: Preprocessed DataFrame with features and target.
        feature_cols: List of column names to use as features. Defaults to
            ['fips_code', 'year', 'month', 'day', 'hour', 'dayofweek'].
            Additional columns like 'event_type' are included if present.
        target_col: Name of the target column. Defaults to 'duration_minutes'.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features and y is a
        Series of target values, both with matching indices.

    Raises:
        ValueError: If target column is missing.

    Examples:
        >>> X, y = prepare_features(processed_df)
        >>> X.shape
        (10000, 7)
        >>> y.name
        'duration_minutes'
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    if feature_cols is None:
        feature_cols = [
            'fips_code', 'year', 'month', 'day', 'hour', 'dayofweek',
        ]

    # Add optional columns if they exist in the data
    optional_cols = ['event_type', 'peak_customers_affected']
    for col in optional_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    # Filter to columns that actually exist
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[preprocessor] WARNING: Missing feature columns: {missing}")

    print(f"[preprocessor] Preparing features: {available}")

    df = df.copy()

    # Encode fips_code as integer for the model
    if 'fips_code' in available:
        df['fips_code'] = pd.to_numeric(df['fips_code'], errors='coerce')

    # Encode event_type as categorical integer
    if 'event_type' in available:
        df['event_type'] = df['event_type'].fillna('None')
        df['event_type'] = df['event_type'].astype('category').cat.codes

    # Drop rows with NaN in features or target
    subset = available + [target_col]
    before = len(df)
    df = df.dropna(subset=subset)
    if len(df) < before:
        print(f"[preprocessor] Dropped {before - len(df):,} rows with NaN values")

    X = df[available].copy()
    y = df[target_col].copy()

    print(f"[preprocessor] Final dataset: {X.shape[0]:,} samples, "
          f"{X.shape[1]} features")

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
