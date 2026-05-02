"""
download_ghcnd_data.py - Download NOAA GHCN-Daily weather data for Virginia (2014-2022).

Downloads daily weather observations from NOAA's GHCN-Daily dataset,
maps each station to its nearest Virginia county by Haversine distance,
and saves a county-day weather table ready to join with EAGLE-I outage data.

This replaces the NOAA Storm Events merge (~7% match rate) with continuous
daily weather coverage for every county every day (~95%+ match rate).

Source: NOAA GHCN-Daily (https://www.ncei.noaa.gov/pub/data/ghcn/daily/)

Usage:
    python download_ghcnd_data.py

Output:
    data/ghcnd_va_daily.csv       -- one row per (fips_code, date) with weather vars
    data/ghcnd_station_county.csv -- station-to-county mapping (cached for reuse)
"""

import gzip
import io
import math
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

import pandas as pd

# ── URLs ───────────────────────────────────────────────────────────────────────
_GHCND_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
_STATIONS_URL = _GHCND_BASE + "/ghcnd-stations.txt"
_BY_YEAR_URL = _GHCND_BASE + "/by_year/{year}.csv.gz"
_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)

# ── configuration ──────────────────────────────────────────────────────────────
YEARS = range(2014, 2023)
STATE = "VA"
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "ghcnd_va_daily.csv"
STATION_MAP_FILE = OUTPUT_DIR / "ghcnd_station_county.csv"
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB download chunks

# GHCN-Daily elements to keep.
# Raw values use NOAA's encoding; scale factors convert to real units.
ELEMENTS = {
    "PRCP": 0.1,   # tenths of mm  → mm
    "SNOW": 1.0,   # mm            → mm
    "SNWD": 1.0,   # mm            → mm
    "TMAX": 0.1,   # tenths of °C  → °C
    "TMIN": 0.1,   # tenths of °C  → °C
    "AWND": 0.1,   # tenths of m/s → m/s  (daily avg wind speed)
    "WSFG": 0.1,   # tenths of m/s → m/s  (peak gust)
    # weather type flags: value=1 when the condition occurred that day
    "WT01": 1.0,   # fog / ice fog
    "WT03": 1.0,   # thunder
    "WT06": 1.0,   # glaze or ice pellets
    "WT09": 1.0,   # blowing or drifting snow
    "WT14": 1.0,   # drizzle
    "WT16": 1.0,   # rain
    "WT17": 1.0,   # freezing rain
    "WT18": 1.0,   # snow
}

# Human-readable output column names
_RENAME = {
    "PRCP": "prcp_mm",
    "SNOW": "snow_mm",
    "SNWD": "snwd_mm",
    "TMAX": "tmax_c",
    "TMIN": "tmin_c",
    "AWND": "awnd_ms",
    "WSFG": "wsfg_ms",
    "WT01": "wt_fog",
    "WT03": "wt_thunder",
    "WT06": "wt_ice",
    "WT09": "wt_blowing_snow",
    "WT14": "wt_drizzle",
    "WT16": "wt_rain",
    "WT17": "wt_freezing_rain",
    "WT18": "wt_snow",
}

_WT_ELEMENTS = {k for k in ELEMENTS if k.startswith("WT")}


# ── helpers ────────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_chunked(url: str, label: str, timeout: int = 600) -> bytes:
    """Download a potentially large file in chunks with progress reporting."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise RuntimeError(f"Download failed for {url}: {e}")

    chunks = []
    total = 0
    start = time.time()
    while True:
        chunk = resp.read(CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        elapsed = time.time() - start or 0.001
        speed = total / elapsed / 1024 / 1024
        print(
            f"\r    {label}: {total / 1024 / 1024:.0f} MB @ {speed:.1f} MB/s",
            end="",
            flush=True,
        )

    print()  # newline after progress
    return b"".join(chunks)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── step 1: station list ───────────────────────────────────────────────────────

def load_va_stations() -> pd.DataFrame:
    """Parse ghcnd-stations.txt and return Virginia stations with lat/lon."""
    print("[download] Fetching GHCN-Daily station list ...")
    raw = _fetch(_STATIONS_URL).decode("utf-8")

    # Fixed-width format: ID(1-11), LAT(13-20), LON(21-30), ELEV(31-37), STATE(39-40), NAME(42-71)
    rows = []
    for line in raw.splitlines():
        if len(line) < 40:
            continue
        if line[38:40].strip() != STATE:
            continue
        try:
            rows.append({
                "station_id": line[0:11].strip(),
                "lat": float(line[12:20].strip()),
                "lon": float(line[21:30].strip()),
                "name": line[41:71].strip(),
            })
        except ValueError:
            continue

    df = pd.DataFrame(rows)
    print(f"[download] Found {len(df)} Virginia GHCN-Daily stations")
    return df


# ── step 2: county centroids ───────────────────────────────────────────────────

def load_county_centroids() -> pd.DataFrame:
    """Download Census Gazetteer and return Virginia county interior points."""
    print("[download] Fetching Census county centroids ...")
    raw = _fetch(_GAZETTEER_URL, timeout=120)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        content = zf.read(txt_name).decode("utf-8")

    counties = pd.read_csv(io.StringIO(content), sep="\t", dtype={"GEOID": str})
    counties.columns = counties.columns.str.strip()  # last column has trailing whitespace
    va = counties[counties["USPS"] == STATE][
        ["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]
    ].copy()
    va = va.rename(columns={"GEOID": "fips_code", "INTPTLAT": "clat", "INTPTLONG": "clon"})
    va = va.reset_index(drop=True)

    print(f"[download] Found {len(va)} Virginia counties/jurisdictions")
    return va


# ── step 3: station → county mapping ──────────────────────────────────────────

def build_station_county_map(stations: pd.DataFrame, counties: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each GHCN-Daily station to its nearest Virginia county
    using Haversine distance to county interior points.
    """
    print("[download] Mapping stations to nearest county (Haversine) ...")
    county_list = list(zip(counties["fips_code"], counties["clat"], counties["clon"]))

    rows = []
    for _, st in stations.iterrows():
        best_fips, best_dist = None, float("inf")
        for fips, clat, clon in county_list:
            d = haversine_km(st["lat"], st["lon"], clat, clon)
            if d < best_dist:
                best_dist, best_fips = d, fips
        rows.append({
            "station_id": st["station_id"],
            "station_name": st["name"],
            "fips_code": best_fips,
            "dist_km": round(best_dist, 1),
        })

    df = pd.DataFrame(rows)
    print(
        f"[download] {len(df)} stations mapped to {df['fips_code'].nunique()} unique counties "
        f"(max dist: {df['dist_km'].max():.0f} km)"
    )
    return df


# ── step 4: download one year ─────────────────────────────────────────────────

def download_year(year: int, va_station_ids: set) -> pd.DataFrame:
    """
    Download one year's GHCN-Daily by_year CSV.gz, filter to Virginia
    stations and target elements, and return as a long-format DataFrame.
    """
    url = _BY_YEAR_URL.format(year=year)
    print(f"  {year}: downloading ...")

    try:
        compressed = _fetch_chunked(url, label=str(year))
    except RuntimeError as e:
        print(f"    {e}")
        return pd.DataFrame()

    print(f"    {year}: filtering observations ...", end=" ", flush=True)

    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
        for raw_line in gz:
            line = raw_line.decode("utf-8", errors="replace")
            parts = line.split(",")
            if len(parts) < 6:
                continue

            sid = parts[0]
            if sid not in va_station_ids:
                continue

            element = parts[2]
            if element not in ELEMENTS:
                continue

            # skip quality-flagged observations (Q_FLAG is column 5)
            if parts[5].strip():
                continue

            try:
                value = float(parts[3]) * ELEMENTS[element]
            except ValueError:
                continue

            rows.append((sid, parts[1], element, value))

    # free compressed bytes before building DataFrame
    del compressed

    df = pd.DataFrame(rows, columns=["station_id", "date", "element", "value"])
    print(f"{len(df):,} Virginia observations kept")
    return df


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
        print(f"Output already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        answer = input("Re-download and overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping.")
            return

    # ── 1. VA station list ────────────────────────────────────────────────────
    stations = load_va_stations()
    va_station_ids = set(stations["station_id"])

    # ── 2. County centroids ───────────────────────────────────────────────────
    counties = load_county_centroids()

    # ── 3. Station → county map (cached) ─────────────────────────────────────
    if STATION_MAP_FILE.exists():
        print(f"[download] Loading cached station-county map ...")
        station_map = pd.read_csv(STATION_MAP_FILE, dtype={"fips_code": str})
    else:
        station_map = build_station_county_map(stations, counties)
        station_map.to_csv(STATION_MAP_FILE, index=False)
        print(f"[download] Saved station-county map to {STATION_MAP_FILE}")

    # ── 4. Download each year ─────────────────────────────────────────────────
    print(f"\nDownloading GHCN-Daily {YEARS.start}–{YEARS.stop - 1} ...")
    all_frames = []
    for year in YEARS:
        df = download_year(year, va_station_ids)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("ERROR: No data downloaded. Check your internet connection.")
        sys.exit(1)

    # ── 5. Pivot to wide: one row per (station_id, date) ─────────────────────
    print("\n[download] Pivoting to wide format ...")
    raw = pd.concat(all_frames, ignore_index=True)
    del all_frames

    pivot = (
        raw.pivot_table(
            index=["station_id", "date"],
            columns="element",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.columns.name = None
    del raw

    # rename to human-readable column names
    pivot = pivot.rename(columns=_RENAME)

    # WT flags: NaN means the condition wasn't reported (i.e., it didn't occur)
    wt_cols = [_RENAME[e] for e in _WT_ELEMENTS if e in _RENAME and _RENAME[e] in pivot.columns]
    for col in wt_cols:
        pivot[col] = pivot[col].fillna(0).astype(int)

    # ── 6. Join station → county ──────────────────────────────────────────────
    pivot = pivot.merge(
        station_map[["station_id", "fips_code"]], on="station_id", how="left"
    )

    # ── 7. Aggregate to (fips_code, date) ─────────────────────────────────────
    # counties served by multiple stations: mean for continuous, max for WT flags
    print("[download] Aggregating to county-day level ...")
    all_wx_cols = [c for c in pivot.columns if c not in ("station_id", "date", "fips_code")]
    cont_cols = [c for c in all_wx_cols if not c.startswith("wt_")]
    flag_cols = [c for c in all_wx_cols if c.startswith("wt_")]

    agg_dict = {c: "mean" for c in cont_cols}
    agg_dict.update({c: "max" for c in flag_cols})

    county_daily = pivot.groupby(["fips_code", "date"]).agg(agg_dict).reset_index()
    del pivot

    # parse YYYYMMDD string → date
    county_daily["date"] = pd.to_datetime(
        county_daily["date"], format="%Y%m%d"
    ).dt.date

    # ── 8. Save ───────────────────────────────────────────────────────────────
    county_daily.to_csv(OUTPUT_FILE, index=False)

    wx_cols = [c for c in county_daily.columns if c not in ("fips_code", "date")]
    print(f"\n{'=' * 55}")
    print(f"GHCN-Daily download complete!")
    print(f"  Rows:        {len(county_daily):,} county-day observations")
    print(f"  Counties:    {county_daily['fips_code'].nunique()} unique FIPS codes")
    print(f"  Date range:  {county_daily['date'].min()} to {county_daily['date'].max()}")
    print(f"  Columns:     {wx_cols}")
    print(f"  Output:      {OUTPUT_FILE}")
    print(f"{'=' * 55}")

    # coverage check
    total_county_days = county_daily["fips_code"].nunique() * len(
        pd.date_range("2014-01-01", "2022-12-31", freq="D")
    )
    coverage = len(county_daily) / total_county_days * 100
    print(f"\n  County-day coverage: {coverage:.1f}% of all possible county-days")
    print("  (gaps = days where no nearby station reported data)")


if __name__ == "__main__":
    main()
