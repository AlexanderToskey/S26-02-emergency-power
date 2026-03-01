"""
data_loader.py - Load and validate EAGLE-I outage and NOAA weather datasets.

Handles reading CSVs, filtering for Virginia, merging on FIPS codes,
and basic data validation. Designed for the Outage Duration Prediction
Model (FUN-1).
"""

import io
import urllib.request
import urllib.error

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# NWS Zone-County Correlation File URL
_NWS_ZONE_COUNTY_URL = (
    "https://www.weather.gov/source/gis/Shapefiles/County/bp18mr25.dbx"
)
_ZONE_COUNTY_CACHE_PATH = Path("data/nws_zone_county_va.csv")


def load_zone_county_mapping(state: str = "VA") -> pd.DataFrame:
    """Load the NWS zone-to-county FIPS mapping.

    Downloads the NWS Zone-County Correlation File from weather.gov
    and caches it locally. Maps NWS forecast zone codes to full 5-digit
    county FIPS codes. One zone can map to multiple counties.

    Args:
        state: Two-letter state abbreviation to filter. Defaults to 'VA'.

    Returns:
        DataFrame with columns: zone (3-digit str), fips_code (5-digit str),
        zone_name, county_name.

    Examples:
        >>> mapping = load_zone_county_mapping('VA')
        >>> mapping[mapping['zone'] == '081']['fips_code'].tolist()
        ['51149', '51670', '51730']
    """
    # Use cached file if available
    if _ZONE_COUNTY_CACHE_PATH.exists():
        print(f"[data_loader] Loading zone-county mapping from cache ...")
        return pd.read_csv(_ZONE_COUNTY_CACHE_PATH, dtype=str)

    print(f"[data_loader] Downloading NWS zone-county mapping ...")
    try:
        req = urllib.request.Request(
            _NWS_ZONE_COUNTY_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise RuntimeError(
            f"Failed to download NWS zone-county file: {e}. "
            f"Check your internet connection."
        )

    # Parse pipe-delimited file (no header)
    # Format: STATE|ZONE|CWA|ZONE_NAME|STATE_ZONE|COUNTY_NAME|FIPS|TZ|FE|LAT|LON
    col_names = [
        "state_abbr", "zone", "cwa", "zone_name", "state_zone",
        "county_name", "fips_code", "timezone", "fe_area", "lat", "lon",
    ]
    df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=col_names, dtype=str)

    # Filter to requested state
    df = df[df["state_abbr"] == state].copy()

    # Keep only the columns we need
    df = df[["zone", "fips_code", "zone_name", "county_name"]].reset_index(drop=True)

    # Cache locally
    _ZONE_COUNTY_CACHE_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(_ZONE_COUNTY_CACHE_PATH, index=False)
    print(f"[data_loader] Cached {len(df)} zone-county mappings "
          f"({df['zone'].nunique()} zones -> {df['fips_code'].nunique()} counties)")

    return df


def _build_zone_to_counties_map(zone_county_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build a dict mapping zone code -> list of county FIPS codes.

    Args:
        zone_county_df: DataFrame from load_zone_county_mapping().

    Returns:
        Dict like {'081': ['51149', '51670', '51730'], ...}
    """
    return zone_county_df.groupby("zone")["fips_code"].apply(list).to_dict()


def load_eagle_outages(
    filepath: Union[str, Path, List[Union[str, Path]]],
) -> pd.DataFrame:
    """Load EAGLE-I outage CSV(s) and filter for Virginia records.

    Reads one or more EAGLE-I CSV files, parses timestamps, and filters to
    only include Virginia records. Accepts a single file path or a list of
    paths (e.g. one per year).

    Args:
        filepath: Path or list of paths to EAGLE-I CSV files. Expected
            columns: fips_code, county, state, sum, run_start_time.

    Returns:
        DataFrame with Virginia outage records, datetime-typed timestamps,
        and fips_code as zero-padded string.

    Raises:
        FileNotFoundError: If any file does not exist.
        ValueError: If required columns are missing.

    Examples:
        >>> df = load_eagle_outages('data/eaglei_outages_2022.csv')
        >>> df = load_eagle_outages([
        ...     'data/eaglei_outages_2021.csv',
        ...     'data/eaglei_outages_2022.csv',
        ... ])
    """
    if isinstance(filepath, (str, Path)):
        filepath = [filepath]

    frames = []
    for fp in filepath:
        fp = Path(fp)
        if not fp.exists():
            raise FileNotFoundError(f"EAGLE-I file not found: {fp}")

        print(f"[data_loader] Loading EAGLE-I data from {fp} ...")
        frames.append(pd.read_csv(fp, dtype={'fips_code': str}))

    df = pd.concat(frames, ignore_index=True)

    required_cols = {'fips_code', 'county', 'state', 'run_start_time'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"EAGLE-I CSV missing required columns: {missing}")

    # Parse timestamps
    df['run_start_time'] = pd.to_datetime(df['run_start_time'], errors='coerce')

    # Normalize the customer count column name across dataset versions.
    # 2014 uses 'customers_out', 2015+ uses 'sum'
    if 'sum' in df.columns and 'customers_out' in df.columns:
        # Both exist (concat of old + new format): merge into one
        df['customers_affected'] = df['customers_out'].fillna(df['sum'])
        df = df.drop(columns=['customers_out', 'sum'])
    elif 'sum' in df.columns:
        df = df.rename(columns={'sum': 'customers_affected'})
    elif 'customers_out' in df.columns:
        df = df.rename(columns={'customers_out': 'customers_affected'})

    # Ensure fips_code is zero-padded to 5 digits
    df['fips_code'] = df['fips_code'].str.zfill(5)

    # Filter for Virginia
    va_mask = df['state'].str.lower() == 'virginia'
    df = df[va_mask].copy()

    # Convert customers_affected to numeric (handles empty strings / NaN)
    df['customers_affected'] = pd.to_numeric(
        df['customers_affected'], errors='coerce'
    ).fillna(0).astype(int)

    print(f"[data_loader] Loaded {len(df):,} Virginia records "
          f"({df['fips_code'].nunique()} counties)")

    return df.reset_index(drop=True)


def load_noaa_weather(filepath_or_list: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load one or more NOAA Storm Events CSV files and concatenate.

    Reads NOAA storm event data, parses dates, and normalizes column names
    to lowercase for consistency.

    Args:
        filepath_or_list: Single path or list of paths to NOAA CSV files.
            Expected columns include: BEGIN_DATE, END_DATE, EVENT_TYPE,
            CZ_FIPS, CZ_TYPE, STATE_ABBR, etc.

    Returns:
        DataFrame with NOAA weather event records, datetime-typed dates,
        and a full 5-digit fips_code column for county-level events.

    Raises:
        FileNotFoundError: If any file does not exist.
        ValueError: If required columns are missing.

    Examples:
        >>> weather = load_noaa_weather('data/storm_data_search_results.csv')
        >>> 'event_type' in weather.columns
        True
    """
    if isinstance(filepath_or_list, (str, Path)):
        filepath_or_list = [filepath_or_list]

    frames = []
    for fp in filepath_or_list:
        fp = Path(fp)
        if not fp.exists():
            raise FileNotFoundError(f"NOAA file not found: {fp}")

        print(f"[data_loader] Loading NOAA data from {fp} ...")
        chunk = pd.read_csv(fp)
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # Lowercase column names for consistency
    df.columns = df.columns.str.lower()

    # Normalize column names across NOAA formats.
    # The interactive search export uses: begin_date, state_abbr
    # The bulk FTP files use: begin_date_time, state, state_fips
    rename_map = {}
    if 'begin_date_time' in df.columns and 'begin_date' not in df.columns:
        rename_map['begin_date_time'] = 'begin_date'
    if 'end_date_time' in df.columns and 'end_date' not in df.columns:
        rename_map['end_date_time'] = 'end_date'
    if 'state_fips' in df.columns and 'state_abbr' not in df.columns:
        # Build state_abbr from state name for FIPS mapping
        rename_map['state_fips'] = 'state_fips_code'
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = {'begin_date', 'event_type', 'cz_fips'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"NOAA CSV missing required columns: {missing}")

    # Parse date columns
    for col in ['begin_date', 'end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')

    # Build full 5-digit FIPS code for county-level events (CZ_TYPE == 'C')
    # Virginia state FIPS = 51, county FIPS are 3-digit in NOAA data
    df['cz_fips'] = df['cz_fips'].astype(str).str.zfill(3)

    if 'state_fips_code' in df.columns:
        # Bulk format: state_fips is numeric (e.g. 51 for Virginia)
        df['state_fips_str'] = df['state_fips_code'].astype(str).str.zfill(2)
        df['fips_code'] = df['state_fips_str'] + df['cz_fips']
        df = df.drop(columns=['state_fips_str'])
    elif 'state_abbr' in df.columns:
        # Interactive search format: state_abbr is e.g. 'VA'
        state_fips_map = {'VA': '51'}
        df['fips_code'] = df['state_abbr'].map(state_fips_map) + df['cz_fips']
    else:
        raise ValueError(
            "Cannot build fips_code: need either 'state_fips' or "
            "'state_abbr' column in NOAA data"
        )

    # Expand zone-level events (CZ_TYPE='Z') to county FIPS codes
    # using the NWS zone-county correlation file
    if 'cz_type' in df.columns:
        zone_mask = df['cz_type'].str.upper() == 'Z'
        zone_count = zone_mask.sum()
        county_count = (~zone_mask).sum()

        if zone_count > 0:
            print(f"[data_loader] Expanding {zone_count:,} zone-level events "
                  f"to county FIPS ({county_count:,} already county-level) ...")

            try:
                zone_county = load_zone_county_mapping("VA")
                zone_map = _build_zone_to_counties_map(zone_county)
            except RuntimeError as e:
                print(f"[data_loader] WARNING: {e}")
                print("[data_loader] Skipping zone expansion, zone events "
                      "will have incorrect FIPS codes.")
                zone_map = {}

            if zone_map:
                # Split into zone and county rows
                zone_rows = df[zone_mask].copy()
                county_rows = df[~zone_mask].copy()

                # For each zone row, look up the county FIPS codes and explode
                zone_rows['fips_code'] = zone_rows['cz_fips'].map(
                    lambda z: zone_map.get(z, [])
                )
                zone_expanded = zone_rows.explode('fips_code')

                # Drop rows that didn't match any county
                before_expand = len(zone_expanded)
                zone_expanded = zone_expanded.dropna(subset=['fips_code'])
                unmatched = before_expand - len(zone_expanded)

                df = pd.concat([county_rows, zone_expanded], ignore_index=True)

                print(f"[data_loader] Zone expansion: {zone_count:,} zone events "
                      f"-> {len(zone_expanded):,} county-level records "
                      f"({unmatched:,} unmatched zones dropped)")

    print(f"[data_loader] Loaded {len(df):,} weather event records "
          f"({df['event_type'].nunique()} event types)")

    return df.reset_index(drop=True)


def merge_weather_outages(
    outages_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge EAGLE-I outage data with NOAA weather events.

    Joins on fips_code and date. Each outage timestamp is matched to weather
    events whose [begin_date, end_date] window overlaps that date.

    Args:
        outages_df: EAGLE-I outage DataFrame (from load_eagle_outages).
            Must have columns: fips_code, run_start_time.
        weather_df: NOAA weather DataFrame (from load_noaa_weather).
            Must have columns: fips_code, begin_date, event_type.

    Returns:
        Merged DataFrame with weather event info joined to outage records.
        Adds 'event_type' as the weather context for each outage record.

    Raises:
        ValueError: If required columns are missing from either DataFrame.

    Examples:
        >>> merged = merge_weather_outages(outages, weather)
        >>> 'event_type' in merged.columns
        True
    """
    for col in ['fips_code', 'run_start_time']:
        if col not in outages_df.columns:
            raise ValueError(f"outages_df missing required column: {col}")
    for col in ['fips_code', 'begin_date', 'event_type']:
        if col not in weather_df.columns:
            raise ValueError(f"weather_df missing required column: {col}")

    print("[data_loader] Merging outage and weather data ...")

    # Create date-only columns for join
    outages = outages_df.copy()
    weather = weather_df.copy()

    outages['date'] = outages['run_start_time'].dt.date

    # Build expanded weather table: one row per (fips_code, date) covering
    # the full event date range, not just begin_date.
    # Previously only begin_date was used, so outages on day 2+ of a multi-day
    # storm (winter storms, ice storms, floods) never matched.
    weather_cols = ['fips_code', 'event_type']
    if 'magnitude' in weather.columns:
        weather_cols.append('magnitude')

    wdf = weather[weather_cols + ['begin_date']].copy()
    wdf['begin_date'] = pd.to_datetime(wdf['begin_date'])

    if 'end_date' in weather.columns:
        end_dt = pd.to_datetime(weather['end_date']).fillna(wdf['begin_date'])
        # cap at 7 days so long-running events (droughts etc.) don't explode row count
        max_end = wdf['begin_date'] + pd.Timedelta(days=7)
        wdf['end_date'] = end_dt.where(end_dt <= max_end, max_end)
    else:
        wdf['end_date'] = wdf['begin_date']

    print("[data_loader] Expanding weather events across date ranges (cap: 7 days) ...")
    wdf['date'] = [
        pd.date_range(start=r.begin_date, end=r.end_date, freq='D').date.tolist()
        for r in wdf.itertuples()
    ]
    wdf = wdf.explode('date').drop(columns=['begin_date', 'end_date'])
    wdf = wdf.drop_duplicates()

    merged = pd.merge(outages, wdf, on=['fips_code', 'date'], how='left')

    matched = merged['event_type'].notna().sum()
    total = len(merged)
    print(f"[data_loader] Merge complete: {total:,} records, "
          f"{matched:,} ({matched/total*100:.1f}%) matched to weather events")

    # Drop the temporary date column
    merged = merged.drop(columns=['date'])

    return merged


def load_ghcnd_weather(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load the GHCN-Daily county-day weather table produced by download_ghcnd_data.py.

    Args:
        filepath: Path to ghcnd_va_daily.csv.

    Returns:
        DataFrame with one row per (fips_code, date) and columns for
        precipitation, temperature, wind, snow, and weather type flags.

    Raises:
        FileNotFoundError: If the file doesn't exist (run download_ghcnd_data.py first).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"GHCN-Daily file not found: {filepath}. "
            f"Run download_ghcnd_data.py first."
        )

    print(f"[data_loader] Loading GHCN-Daily weather from {filepath} ...")
    df = pd.read_csv(filepath, dtype={"fips_code": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print(
        f"[data_loader] Loaded {len(df):,} county-day observations "
        f"({df['fips_code'].nunique()} counties, "
        f"{df['date'].min()} to {df['date'].max()})"
    )
    return df


def merge_ghcnd_weather(
    outages_df: pd.DataFrame,
    ghcnd_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge EAGLE-I outage snapshots with GHCN-Daily county-day weather.

    Joins on fips_code + date so every outage snapshot gets the measured
    weather for that county on that day (~91% match rate vs ~7% for NOAA
    Storm Events).

    Args:
        outages_df: EAGLE-I outage DataFrame (must have fips_code, run_start_time).
        ghcnd_df: DataFrame from load_ghcnd_weather().

    Returns:
        Merged DataFrame with GHCN-Daily weather columns added.
    """
    for col in ["fips_code", "run_start_time"]:
        if col not in outages_df.columns:
            raise ValueError(f"outages_df missing required column: {col}")

    print("[data_loader] Merging outage data with GHCN-Daily weather ...")

    outages = outages_df.copy()
    outages["date"] = outages["run_start_time"].dt.normalize()

    merged = pd.merge(outages, ghcnd_df, on=["fips_code", "date"], how="left")

    # use prcp_mm as the coverage proxy (it's the most widely reported element)
    matched = merged["prcp_mm"].notna().sum()
    total = len(merged)
    print(
        f"[data_loader] GHCN-Daily merge: {total:,} records, "
        f"{matched:,} ({matched / total * 100:.1f}%) matched to daily weather"
    )

    merged = merged.drop(columns=["date"])
    return merged


def validate_data(df: pd.DataFrame) -> dict:
    """Run validation checks on a DataFrame and print a summary.

    Checks for null values, prints date ranges, record counts, and
    column-level missing value stats.

    Args:
        df: Any DataFrame to validate.

    Returns:
        Dict with keys: 'num_records', 'num_columns', 'missing_by_column',
        'date_range' (if a datetime column exists), 'duplicate_rows'.

    Examples:
        >>> stats = validate_data(merged_df)
        >>> stats['num_records']
        150000
    """
    print("\n[data_loader] === Data Validation Summary ===")

    stats = {
        'num_records': len(df),
        'num_columns': len(df.columns),
        'missing_by_column': df.isnull().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
    }

    print(f"  Records:    {stats['num_records']:,}")
    print(f"  Columns:    {stats['num_columns']}")
    print(f"  Duplicates: {stats['duplicate_rows']:,}")

    # Find datetime columns and report date range
    dt_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if dt_cols:
        col = dt_cols[0]
        stats['date_range'] = {
            'column': col,
            'min': str(df[col].min()),
            'max': str(df[col].max()),
        }
        print(f"  Date range: {df[col].min()} to {df[col].max()} ({col})")

    # Missing values
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if len(has_missing) > 0:
        print(f"  Missing values:")
        for col_name, count in has_missing.items():
            pct = count / len(df) * 100
            print(f"    {col_name}: {count:,} ({pct:.1f}%)")
    else:
        print("  Missing values: none")

    print("[data_loader] =============================\n")

    return stats
