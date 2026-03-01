import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def extract_temporal_features(df):
    """
    Extract temporal features from county-day date column.
    """

    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found. Available columns: {df.columns.tolist()}")

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


def create_outage_occurrence_target(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure outage_occurred column exists and is binary.
    """

    if "outage_occurred" not in df.columns:
        raise ValueError(
            f"Missing required column: 'outage_occurred'. "
            f"Available columns: {df.columns.tolist()}"
        )

    # If there are no outage records or only outage records, raise an error
    if not set(df["outage_occurred"].unique()).issubset({0, 1}):
        raise ValueError("outage_occurred must be binary (0/1)")

    return df

'''
def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "outage_occurred",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (binary target) for outage occurrence modeling.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    df = df.copy()

    # Training features
    if feature_cols is None:
        feature_cols = [
            "fips_code",
            "year",
            "month",
            "hour",
            "dayofweek",
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

    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Encode event_type if present
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].fillna("None").astype(str)
        dummies = pd.get_dummies(df["event_type"], prefix="event")
        df = pd.concat([df, dummies], axis=1)
        feature_cols += list(dummies.columns)

    print(f"[preprocessor] Preparing features ({len(feature_cols)}): {feature_cols}")

    # Ensure numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing core temporal + target
    core_cols = ["fips_code", "month", "hour", "dayofweek", target_col]
    df = df.dropna(subset=[c for c in core_cols if c in df.columns])

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = X.fillna(0)

    print(f"[preprocessor] Final dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"[preprocessor] Class balance: {y.mean():.4f} positive rate")

    return X, y

'''

def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "outage_occurred",
    include_temporal: bool = True,
    include_storm: bool = True,
    include_event_dummies: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (binary target) for outage occurrence modeling.
    
    Includes temporal features, weather, storm flags, and optional event dummies.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    df = df.copy()

    # --- 1. Temporal Features ---
    if include_temporal:
        if "date" not in df.columns:
            raise ValueError("'date' column required for temporal features")
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- 2. Default Weather & Core Features ---
    if feature_cols is None:
        feature_cols = [
            "fips_code",
            "year",
            "month",
            "dayofweek",
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

    # --- 3. Include storm flags if present ---
    if include_storm:
        storm_cols = [c for c in ["storm_event", "has_weather_event"] if c in df.columns]
        feature_cols += storm_cols

    # --- 4. Encode categorical event_type ---
    if include_event_dummies and "event_type" in df.columns:
        df["event_type"] = df["event_type"].fillna("None").astype(str)
        dummies = pd.get_dummies(df["event_type"], prefix="event")
        df = pd.concat([df, dummies], axis=1)
        feature_cols += list(dummies.columns)

    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # --- 5. Ensure numeric and fill missing ---
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)

    # --- 6. Drop rows missing core temporal + target ---
    core_cols = ["fips_code", "year", "month", "dayofweek", target_col]
    df = df.dropna(subset=[c for c in core_cols if c in df.columns])

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"[preprocessor] Final dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"[preprocessor] Class balance: {y.mean():.4f} positive rate")

    return X, y

def run_full_pipeline(
    outages_df: pd.DataFrame,
    timestamp_col: str = 'run_start_time',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run full preprocessing pipeline for outage occurrence prediction.
    Snapshot-level classification (NOT event aggregation).
    """

    print("\n" + "=" * 50)
    print("PREPROCESSING PIPELINE START (OCCURRENCE TARGET)")
    print("=" * 50)

    df = outages_df.copy()

    # Temporal features
    df = extract_temporal_features(df)

    # Binary target
    df = create_outage_occurrence_target(df)

    # Prepare X and y
    X, y = prepare_features(df)

    print("=" * 50)
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"Output: {X.shape[0]:,} samples x {X.shape[1]} features")
    print("=" * 50 + "\n")

    return X, y