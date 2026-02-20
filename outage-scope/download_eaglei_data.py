"""
download_eaglei_data.py - Download EAGLE-I outage data for Virginia (2014-2022).

Downloads county-level outage CSVs from the Figshare public dataset,
filters for Virginia records, and saves to the data/ folder.

Source: ORNL EAGLE-I dataset (DOI: 10.6084/m9.figshare.24237376)

The full dataset is ~7.5 GB nationwide. This script streams each file
and filters for Virginia on-the-fly, so only ~1 GB of Virginia data
is written to disk.

Usage:
    python download_eaglei_data.py

Output:
    data/eaglei_outages_2014.csv  (Virginia only)
    data/eaglei_outages_2015.csv  (Virginia only)
    ...
    data/eaglei_outages_2022.csv  (Virginia only)
"""

import csv
import io
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Figshare article containing EAGLE-I 2014-2022
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/24237376/versions/1"
STATE_FILTER = "Virginia"
OUTPUT_DIR = Path("data")
YEARS = range(2014, 2023)

# Chunk size for streaming downloads (1 MB)
CHUNK_SIZE = 1024 * 1024


def get_download_urls() -> dict:
    """Fetch download URLs for all EAGLE-I files from Figshare API.

    Returns:
        Dict mapping year (int) to download URL (str).
    """
    print("[download] Fetching file list from Figshare API ...")
    req = urllib.request.Request(FIGSHARE_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    urls = {}
    for f in data["files"]:
        name = f["name"]
        # Match files like eaglei_outages_2022.csv
        if name.startswith("eaglei_outages_") and name.endswith(".csv"):
            year_str = name.replace("eaglei_outages_", "").replace(".csv", "")
            try:
                year = int(year_str)
                urls[year] = {
                    "url": f["download_url"],
                    "size": f["size"],
                    "name": name,
                }
            except ValueError:
                continue

    return urls


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_and_filter_year(url: str, output_path: Path, total_size: int) -> int:
    """Download a single year's CSV, filter for Virginia, write to disk.

    Streams the file in chunks to avoid loading multi-GB files into memory.

    Args:
        url: Download URL for the CSV.
        output_path: Where to write the Virginia-filtered CSV.
        total_size: Expected file size in bytes (for progress reporting).

    Returns:
        Number of Virginia records written.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=300)

    va_count = 0
    total_read = 0
    start_time = time.time()

    # We'll accumulate raw bytes, decode, then filter line by line.
    # The CSV is simple enough (no quoted newlines in these files) that
    # line-by-line processing is safe and memory-efficient.
    buffer = b""
    header = None

    with open(output_path, "w", newline="", encoding="utf-8") as out_file:
        writer = None

        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break

            total_read += len(chunk)
            buffer += chunk

            # Split into lines, keep the last partial line in the buffer
            lines = buffer.split(b"\n")
            buffer = lines[-1]  # may be partial
            lines = lines[:-1]

            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if header is None:
                    # First line is the header
                    header = line
                    out_file.write(header + "\n")
                    # Find the state column index
                    cols = header.split(",")
                    try:
                        state_idx = cols.index("state")
                    except ValueError:
                        raise ValueError(
                            f"CSV header missing 'state' column. Got: {cols}"
                        )
                    continue

                # Check if this row is Virginia
                fields = line.split(",")
                if len(fields) > state_idx and fields[state_idx] == STATE_FILTER:
                    out_file.write(line + "\n")
                    va_count += 1

            # Progress
            pct = (total_read / total_size * 100) if total_size > 0 else 0
            elapsed = time.time() - start_time
            speed = total_read / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(
                f"\r    Downloaded {format_size(total_read)} / "
                f"{format_size(total_size)} ({pct:.0f}%) "
                f"@ {speed:.1f} MB/s - {va_count:,} VA records",
                end="",
                flush=True,
            )

        # Process any remaining data in buffer
        if buffer:
            line = buffer.decode("utf-8", errors="replace").strip()
            if line and header is not None:
                fields = line.split(",")
                if len(fields) > state_idx and fields[state_idx] == STATE_FILTER:
                    out_file.write(line + "\n")
                    va_count += 1

    print()  # newline after progress
    return va_count


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Get download URLs from Figshare
    all_urls = get_download_urls()

    # Check which years we need
    years_to_download = []
    for year in YEARS:
        if year not in all_urls:
            print(f"  WARNING: {year} not available on Figshare, skipping.")
            continue

        output_path = OUTPUT_DIR / f"eaglei_outages_{year}.csv"
        if output_path.exists():
            # Check if it's already Virginia-filtered (small) or the full national file
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  {year}: {output_path.name} already exists ({size_mb:.0f} MB)")
            years_to_download.append((year, True))  # True = exists
        else:
            years_to_download.append((year, False))

    existing = [y for y, exists in years_to_download if exists]
    needed = [y for y, exists in years_to_download if not exists]

    if existing:
        print(f"\nAlready downloaded: {existing}")
    if not needed:
        print("All files already downloaded. Delete files in data/ to re-download.")
        return

    print(f"Files to download: {needed}")
    total_download_size = sum(all_urls[y]["size"] for y in needed)
    print(f"Total download size (before filtering): {format_size(total_download_size)}")
    print(f"Virginia records will be much smaller (~6-7% of total)\n")

    response = input(f"Download {len(needed)} files? [Y/n]: ").strip().lower()
    if response == "n":
        print("Cancelled.")
        return

    # Download each year
    total_records = 0
    for year in needed:
        info = all_urls[year]
        output_path = OUTPUT_DIR / f"eaglei_outages_{year}.csv"

        print(f"\n  {year}: Downloading {info['name']} ({format_size(info['size'])}) ...")

        try:
            count = download_and_filter_year(
                info["url"], output_path, info["size"]
            )
            total_records += count
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"    Saved {count:,} Virginia records ({size_mb:.1f} MB)")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"    FAILED: {e}")
            if output_path.exists():
                output_path.unlink()
            continue
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.")
            if output_path.exists():
                output_path.unlink()
            sys.exit(1)

    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Total Virginia records: {total_records:,}")
    print(f"  Files saved to: {OUTPUT_DIR.resolve()}")
    print(f"{'='*50}")

    # List all eagle-i files in data dir
    print(f"\nAvailable EAGLE-I data files:")
    for f in sorted(OUTPUT_DIR.glob("eaglei_outages_*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
