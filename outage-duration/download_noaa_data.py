"""
download_noaa_data.py - Download NOAA Storm Events data for Virginia (2014-2022).

Downloads bulk CSV files from NOAA's public archive, filters for Virginia
events, and saves a single combined CSV to the data/ folder.

Usage:
    python download_noaa_data.py

Output:
    data/noaa_storm_events_va_2014_2022.csv
"""

import gzip
import io
import re
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

# NOAA Storm Events bulk CSV directory
BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEARS = range(2014, 2023)
STATE_FILTER = "VIRGINIA"
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "noaa_storm_events_va_2014_2022.csv"


def get_filename_for_year(directory_html: str, year: int) -> str:
    """Find the StormEvents_details filename for a given year.

    The filenames include a date suffix that changes with each NOAA update,
    so we parse the directory listing to find the current one.

    Args:
        directory_html: HTML content of the NOAA CSV directory listing.
        year: The year to find (e.g. 2022).

    Returns:
        The full filename string (e.g. 'StormEvents_details-ftp_v1.0_d2022_c20250721.csv.gz').

    Raises:
        FileNotFoundError: If no matching file is found for the year.
    """
    pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
    matches = re.findall(pattern, directory_html)
    if not matches:
        raise FileNotFoundError(f"No StormEvents_details file found for {year}")
    # Take the last match (most recent version if multiple exist)
    return sorted(matches)[-1]


def download_and_extract(url: str) -> pd.DataFrame:
    """Download a gzipped CSV from a URL and return as a DataFrame.

    Args:
        url: Full URL to the .csv.gz file.

    Returns:
        DataFrame with the CSV contents.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        compressed = response.read()

    decompressed = gzip.decompress(compressed)
    return pd.read_csv(io.BytesIO(decompressed), low_memory=False)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check if output already exists
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, nrows=5)
        print(f"Output file already exists: {OUTPUT_FILE}")
        print(f"  ({len(pd.read_csv(OUTPUT_FILE)):,} records)")
        response = input("Re-download and overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return

    # Fetch the directory listing to find exact filenames
    print(f"Fetching NOAA file listing from {BASE_URL} ...")
    req = urllib.request.Request(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        directory_html = response.read().decode("utf-8")

    all_frames = []

    for year in YEARS:
        try:
            filename = get_filename_for_year(directory_html, year)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}, skipping.")
            continue

        url = BASE_URL + filename
        print(f"  Downloading {year}: {filename} ...", end=" ", flush=True)

        try:
            df = download_and_extract(url)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"FAILED ({e})")
            continue

        # Filter for Virginia
        if "STATE" in df.columns:
            va_df = df[df["STATE"].str.upper() == STATE_FILTER].copy()
        else:
            print(f"WARNING: no STATE column, keeping all rows")
            va_df = df

        all_frames.append(va_df)
        print(f"OK ({len(va_df):,} VA records out of {len(df):,} total)")

    if not all_frames:
        print("ERROR: No data downloaded. Check your internet connection.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Save to CSV
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved {len(combined):,} Virginia storm event records to {OUTPUT_FILE}")
    print(f"  Years: {combined['YEAR'].min()}-{combined['YEAR'].max()}")
    print(f"  Event types: {combined['EVENT_TYPE'].nunique()}")
    print(f"  Columns: {len(combined.columns)}")

    # Quick summary
    print("\nEvent type breakdown:")
    counts = combined["EVENT_TYPE"].value_counts()
    for event_type, count in counts.head(15).items():
        print(f"  {event_type}: {count:,}")
    if len(counts) > 15:
        print(f"  ... and {len(counts) - 15} more types")


if __name__ == "__main__":
    main()
