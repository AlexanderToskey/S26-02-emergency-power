import pandas as pd


def load_eagle_outages(file_paths):
    """Load and concatenate EAGLE-I outage CSV files."""
    dfs = [pd.read_csv(fp) for fp in file_paths]
    outages = pd.concat(dfs, ignore_index=True)

    # Ensure datetime
    outages["run_start_time"] = pd.to_datetime(outages["run_start_time"])

    return outages


def load_noaa_weather(file_path):
    """Load NOAA storm event data."""
    weather = pd.read_csv(file_path)

    weather["begin_date_time"] = pd.to_datetime(weather["begin_date_time"])
    weather["end_date_time"] = pd.to_datetime(weather["end_date_time"])

    weather = weather.rename(columns={
        "begin_date_time": "event_start",
        "end_date_time": "event_end"
    })

    return weather


def merge_weather_outages(outages, weather):
    """
    Merge weather events onto outage records.
    Adds binary flag for whether outage occurred during storm.
    """

    merged = outages.copy()

    # Default: no storm
    merged["storm_event"] = 0

    for _, event in weather.iterrows():
        mask = (
            (merged["run_start_time"] >= event["event_start"]) &
            (merged["run_start_time"] <= event["event_end"])
        )
        merged.loc[mask, "storm_event"] = 1

    # ---- Create outage occurrence target ----
    # If customers_out > 0 → outage occurred
    merged["outage_occurred"] = (merged["customers_out"] > 0).astype(int)

    return merged

def merge_occurrence_with_weather(
    occurrence_df: pd.DataFrame,
    ghcnd_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge county-day outage occurrence labels with GHCN-Daily weather.

    This preserves ALL weather days, even if no outage occurred.
    Missing outage labels are filled as 0.
    """

    for col in ["fips_code", "date"]:
        if col not in occurrence_df.columns:
            raise ValueError(f"occurrence_df missing required column: {col}")
        if col not in ghcnd_df.columns:
            raise ValueError(f"ghcnd_df missing required column: {col}")

    print("[data_loader] Merging occurrence labels with weather ...")

    occurrence_df['date'] = pd.to_datetime(occurrence_df['date'])

    merged = pd.merge(
        ghcnd_df,
        occurrence_df,
        on=["fips_code", "date"],
        how="left",
    )

    # Fill missing outage days as 0
    merged["outage_occurred"] = merged["outage_occurred"].fillna(0).astype(int)
    merged["max_customers_affected"] = (
        merged["max_customers_affected"].fillna(0).astype(int)
    )

    print(
        f"[data_loader] Final dataset: {len(merged):,} county-day rows "
        f"({merged['outage_occurred'].sum():,} positive outage days)"
    )

    return merged

def build_occurrence_labels(outages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw EAGLE-I outage snapshots into county-day binary occurrence labels.

    For each (fips_code, date):
        outage_occurred = 1 if customers_out > 0 at any snapshot that day
        outage_occurred = 0 otherwise

    Returns:
        DataFrame with:
            fips_code
            date
            outage_occurred (0/1)
            max_customers_affected
    """
    required_cols = ["fips_code", "run_start_time", "customers_out"]
    for col in required_cols:
        if col not in outages_df.columns:
            raise ValueError(f"outages_df missing required column: {col}")

    df = outages_df.copy()
    df["date"] = df["run_start_time"].dt.date

    grouped = (
        df.groupby(["fips_code", "date"])
        .agg(
            max_customers_affected=("customers_out", "max"),
        )
        .reset_index()
    )

    grouped["outage_occurred"] = (
        grouped["max_customers_affected"] > 0
    ).astype(int)

    print(
        f"[data_loader] Built occurrence labels: "
        f"{len(grouped):,} county-day rows "
        f"({grouped['outage_occurred'].sum():,} outage days)"
    )

    return grouped


def load_ghcnd_weather(file_path):
    """Load GHCN-Daily weather data."""
    ghcnd = pd.read_csv(file_path)
    ghcnd["date"] = pd.to_datetime(ghcnd["date"])
    return ghcnd


def merge_ghcnd_weather(merged, ghcnd):
    """Merge daily weather features onto outage dataset."""
    merged["date"] = merged["run_start_time"].dt.date
    ghcnd["date"] = ghcnd["date"].dt.date

    merged = merged.merge(ghcnd, on="date", how="left")

    return merged


def validate_data(df):
    """Basic data validation."""
    print("    Missing values per column:")
    print(df.isna().sum())

    if "outage_occurred" in df.columns:
        print("    Outage occurrence distribution:")
        print(df["outage_occurred"].value_counts())

def summarize_class_balance(df: pd.DataFrame) -> None:
    """
    Print class balance for outage occurrence target.
    """
    if "outage_occurred" not in df.columns:
        raise ValueError("DataFrame missing 'outage_occurred' column")

    total = len(df)
    positives = df["outage_occurred"].sum()
    negatives = total - positives

    print("\n[data_loader] === Class Balance ===")
    print(f"Total samples: {total:,}")
    print(f"Outage days (1): {positives:,} ({positives/total*100:.2f}%)")
    print(f"No outage days (0): {negatives:,} ({negatives/total*100:.2f}%)")
    print("[data_loader] ======================\n")