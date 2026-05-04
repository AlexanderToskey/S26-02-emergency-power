#Loads EAGLE-I outage snapshots and converts them into county-day occurrence labels for model training
import pandas as pd

def load_eagle_outages(file_paths):
    """Load and concatenate EAGLE-I outage CSV files."""
    dfs = [pd.read_csv(fp) for fp in file_paths]
    outages = pd.concat(dfs, ignore_index=True)

    #Ensure run_start_time is datetime for merging with weather events
    outages["run_start_time"] = pd.to_datetime(
        outages["run_start_time"],
        errors="coerce"
    )

    outages = outages.dropna(subset=["run_start_time"])

    return outages


def load_noaa_weather(file_path):
    #Loads NOAA storm events data
    weather = pd.read_csv(file_path)

    weather["begin_date_time"] = pd.to_datetime(weather["begin_date_time"])
    weather["end_date_time"] = pd.to_datetime(weather["end_date_time"])

    weather = weather.rename(columns={
        "begin_date_time": "event_start",
        "end_date_time": "event_end"
    })

    return weather


def merge_weather_outages(outages, weather):

    #Merge weather events onto outage records
    #Adds binary flag for whether outage occurred during storm
   
    merged = outages.copy()

    #Default to no storm
    merged["storm_event"] = 0

    for _, event in weather.iterrows():
        mask = (
            (merged["run_start_time"] >= event["event_start"]) &
            (merged["run_start_time"] <= event["event_end"])
        )
        merged.loc[mask, "storm_event"] = 1

    # If customers_out > 0 then an outage occurred
    merged["outage_occurred"] = (merged["customers_out"] > 0).astype(int)

    return merged

def merge_occurrence_with_weather(
    occurrence_df: pd.DataFrame,
    ghcnd_df: pd.DataFrame,
) -> pd.DataFrame:
    
    #This merges county-day outage occurrence labels with GHCN daily weather
    #Left join keeps all weather rows — non-outage days need to stay in as negatives
    for col in ["fips_code", "date"]:
        if col not in occurrence_df.columns:
            raise ValueError(f"occurrence_df missing required column: {col}")
        if col not in ghcnd_df.columns:
            raise ValueError(f"ghcnd_df missing required column: {col}")

    print("[data_loader] Merging occurrence labels with weather ...")

    occurrence_df["date"] = pd.to_datetime(occurrence_df["date"]).dt.normalize()
    ghcnd_df["date"] = pd.to_datetime(ghcnd_df["date"]).dt.normalize()

    merged = pd.merge(
        ghcnd_df,
        occurrence_df,
        on=["fips_code", "date"],
        how="left",
    )

    #Fill missing outage days as 0
    merged["outage_occurred"] = merged["outage_occurred"].fillna(0).astype(int)
    merged["max_customers_affected"] = (
        merged["max_customers_affected"].fillna(0).astype(int)
    )

    print(
        f"[data_loader] Final dataset: {len(merged):,} county-day rows "
        f"({merged['outage_occurred'].sum():,} positive outage days)"
    )

    return merged

def build_occurrence_labels(outages_df: pd.DataFrame, min_customers: int = 100) -> pd.DataFrame:
    #EAGLE-I records 15-min snapshots — we reduce to one label per county per day
    #Using max so a brief but large spike still counts as an outage day
    required_cols = ["fips_code", "run_start_time", "customers_affected"]
    for col in required_cols:
        if col not in outages_df.columns:
            raise ValueError(f"outages_df missing required column: {col}")

    df = outages_df.copy()
    #Normalize strips the time component so timestamps align on day-level joins
    df["date"] = pd.to_datetime(df["run_start_time"]).dt.normalize()

    grouped = (
        df.groupby(["fips_code", "date"])
        .agg(
            max_customers_affected=("customers_affected", "max"),
        )
        .reset_index()
    )

    grouped["outage_occurred"] = (
        grouped["max_customers_affected"] >= min_customers
    ).astype(int)

    print(
        f"[data_loader] Built occurrence labels: "
        f"{len(grouped):,} county-day rows "
        f"({grouped['outage_occurred'].sum():,} outage days)"
    )

    return grouped


def load_ghcnd_weather(file_path):
    #Load GHCN weather data
    ghcnd = pd.read_csv(file_path)
    ghcnd["date"] = pd.to_datetime(ghcnd["date"]).dt.normalize()
    return ghcnd


def merge_ghcnd_weather(merged, ghcnd):
    #Merge daily weather features onto outage dataset
    merged["date"] = merged["run_start_time"].dt.normalize()
    ghcnd["date"] = pd.to_datetime(ghcnd["date"]).dt.normalize()

    merged = merged.merge(ghcnd, on="date", how="left")

    return merged


def validate_data(df):
    #Data validation
    print("    Missing values per column:")
    print(df.isna().sum())

    if "outage_occurred" in df.columns:
        print("    Outage occurrence distribution:")
        print(df["outage_occurred"].value_counts())

def summarize_class_balance(df: pd.DataFrame) -> None:
    #Print class balance for outage occurrence target
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