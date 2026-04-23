"""
realtime_inference.py - Real-time prediction pipeline for Virginia power outages.

Assembles live features from Open-Meteo (weather) and Kubra/Dominion (live outage
data), runs all three cascading model stages, and returns county-level predictions.

Public API:
    init()                    load models + static lookup tables (call once at startup)
    run_inference()           fetch live data, run cascade, return predictions dict
    get_cached_predictions()  return last successful prediction result (thread-safe)
"""


import sys
import math
import json
import time
import threading
import urllib.request
import urllib.error
import shap
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
GEO_FILE   = DATA_DIR / "virginia_geo.csv"
STATS_FILE = DATA_DIR / "county_stats.csv"

# Make sure subpackages are importable
sys.path.insert(0, str(BASE_DIR))

from outage_occurrence.occurrence_model import OutageOccurrenceModel
from outage_scope.src.two_stage_model import TwoStageScopeModel
from outage_duration.src.two_stage_model import TwoStageOutageModel

from anomaly_detection.autoencoder import Autoencoder

# ── WMO weather code → binary flag mapping ────────────────────────────────────
_CODE_FLAGS = {
    "wt_fog":           {45, 48},
    "wt_thunder":       {95, 96, 99},
    "wt_snow":          {71, 73, 75, 77, 85, 86},
    "wt_freezing_rain": {66, 67},
    "wt_ice":           {66, 67},
    "wt_blowing_snow":  {77},
    "wt_drizzle":       {51, 53, 55},
    "wt_rain":          {61, 63, 65, 80, 81, 82},
}

_OPEN_METEO_URL  = "https://api.open-meteo.com/v1/forecast"

# Kubra / Dominion StormCenter IDs (same as dominion_scraper.py)
_STORMCENTER_ID  = "9c691bb6-767e-4532-b00e-286ac9adc223"
_VIEW_ID         = "38b5394c-8bca-4dfd-ac59-b321615446bd"

# Maximum parallel threads for weather fetching — keep low to avoid rate-limiting
_WEATHER_WORKERS = 1
_WEATHER_RETRIES = 3
_WEATHER_RETRY_DELAY = 2.0  # seconds between retries

# Minimum predicted customers to surface an occurrence as a real outage.
# Retrained model uses threshold >= 100 customers, but scope filter adds a secondary check.
_MIN_SCOPE_CUSTOMERS = 50

# ── Module-level state ─────────────────────────────────────────────────────────
_occ_model:   Optional[OutageOccurrenceModel] = None
_scope_model: Optional[TwoStageScopeModel]    = None
_dur_model:   Optional[TwoStageOutageModel]   = None
_scope_forecast: Optional[TwoStageScopeModel] = None
_dur_forecast: Optional[TwoStageOutageModel] = None
_geo:         Optional[pd.DataFrame]          = None   # county centroids (133 rows)
_county_stats: Optional[pd.DataFrame]         = None   # precomputed county-level stats

_prev_kubra:      Dict[str, float]  = {}     # fips_code -> customers (last poll)
_prev_kubra_time: Optional[datetime] = None

_cached_predictions: Dict[str, Any] = {}
_cached_features:    Dict[str, Any] = {}   # fips -> {weather, model_row} for explain endpoint
_last_updated:       Optional[str]  = None
_cache_lock = threading.Lock()
_occ_explainer = None   # shap.TreeExplainer, initialized in init()
_scope_explainer_small = None
_scope_explainer_large=None
_scope_explainer_classifier=None
_duration_explainer_small=None
_duration_explainer_large=None
_duration_explainer_classifier=None
_scope_forecast_explainer_small=None
_scope_forecast_explainer_large=None
_scope_forecast_explainer_classifier=None
_duration_forecast_explainer_small=None
_duration_forecast_explainer_large=None
_duration_forecast_explainer_classifier=None

_ae_model          = None   # Autoencoder, loaded in init()
_ae_mean           = None
_ae_std            = None
_ae_threshold      = None
_ae_feature_columns = None  # features the autoencoder was trained on

_cached_forecast: Dict[str, Any] = {}  # date_str -> {fips -> prediction}
_forecast_lock = threading.Lock()

_EXPLAINER_NAMES = {}

# ── Initialisation ─────────────────────────────────────────────────────────────

def init() -> bool:
    """
    Load trained models and static lookup tables.
    Call once at server startup before the first inference.
    Returns True if everything loaded cleanly, False if any component failed
    (the pipeline will still run with degraded features).
    """
    global _occ_model, _scope_model, _dur_model, _scope_forecast, _dur_forecast, _geo, _county_stats
    global _EXPLAINER_NAMES
    ok = True

    # --- Models ---
    try:
        _occ_model   = OutageOccurrenceModel.load(MODELS_DIR / "occurrence_model.joblib")
        _scope_model = TwoStageScopeModel.load(MODELS_DIR / "scope_model.joblib")
        _dur_model   = TwoStageOutageModel.load(MODELS_DIR / "duration_model.joblib")
        _scope_forecast = TwoStageScopeModel.load(MODELS_DIR / "scope_forecast.joblib")
        _dur_forecast = TwoStageOutageModel.load(MODELS_DIR / "duration_forecast.joblib")
        print("[realtime] All three models loaded successfully.")

        # Build SHAP explainer for the occurrence model
        try:

            global _occ_explainer, _scope_explainer_small, _scope_explainer_large, _scope_explainer_classifier
            global _duration_explainer_small, _duration_explainer_large, _duration_explainer_classifier
            global _scope_forecast_explainer_small, _scope_forecast_explainer_large, _scope_forecast_explainer_classifier
            global _duration_forecast_explainer_small, _duration_forecast_explainer_large, _duration_forecast_explainer_classifier
            _occ_explainer = shap.TreeExplainer(_occ_model.model)
            _scope_explainer_small = shap.TreeExplainer(_scope_model.shortRegressor)
            _scope_explainer_large = shap.TreeExplainer(_scope_model.longRegressor)
            _scope_explainer_classifier = shap.TreeExplainer(_scope_model.classifier)
            _duration_explainer_small = shap.TreeExplainer(_dur_model.shortRegressor)
            _duration_explainer_large = shap.TreeExplainer(_dur_model.longRegressor)
            _duration_explainer_classifier = shap.TreeExplainer(_dur_model.classifier)
            _scope_forecast_explainer_small = shap.TreeExplainer(_scope_forecast.shortRegressor)
            _scope_forecast_explainer_large = shap.TreeExplainer(_scope_forecast.longRegressor)
            _scope_forecast_explainer_classifier = shap.TreeExplainer(_scope_forecast.classifier)
            _duration_forecast_explainer_small = shap.TreeExplainer(_dur_forecast.shortRegressor)
            _duration_forecast_explainer_large = shap.TreeExplainer(_dur_forecast.longRegressor)
            _duration_forecast_explainer_classifier = shap.TreeExplainer(_dur_forecast.classifier)
            
            _EXPLAINER_NAMES = {
                "_occ_explainer" : (_occ_explainer, _occ_model),
                "_scope_explainer_small" : (_scope_explainer_small, _scope_model),
                "_scope_explainer_large" : (_scope_explainer_large, _scope_model),
                "_scope_explainer_classifier" : (_scope_explainer_classifier, _scope_model),
                "_duration_explainer_small" : (_duration_explainer_small, _dur_model),
                "_duration_explainer_large" : (_duration_explainer_large, _dur_model),
                "_duration_explainer_classifier" : (_duration_explainer_classifier, _dur_model),
                "_scope_forecast_explainer_small" : (_scope_forecast_explainer_small, _scope_forecast),
                "_scope_forecast_explainer_large" : (_scope_forecast_explainer_large, _scope_forecast),
                "_scope_forecast_explainer_classifier" : (_scope_forecast_explainer_classifier, _scope_forecast),
                "_duration_forecast_explainer_small" : (_duration_forecast_explainer_small, _dur_forecast),
                "_duration_forecast_explainer_large" : (_duration_forecast_explainer_large, _dur_forecast),
                "_duration_forecast_explainer_classifier" : (_duration_forecast_explainer_classifier, _dur_forecast),
            }

            
            print("[realtime] SHAP explainer initialized.")
        except Exception as shap_err:
            print(f"[realtime] WARNING: SHAP explainer failed to initialize: {shap_err}")

        try:
            load_autoencoder()
        except Exception as e:
            print(f"[realtime] WARNING: failed to load autoencoder: {e}")

    except Exception as e:
        print(f"[realtime] ERROR loading models: {e}")
        ok = False

    # --- County centroids ---
    try:
        _geo = pd.read_csv(GEO_FILE, dtype={"fips": str})
        print(f"[realtime] Geo table loaded: {len(_geo)} counties.")
    except Exception as e:
        print(f"[realtime] ERROR loading geo file: {e}")
        ok = False

    # --- County historical stats ---
    try:
        _county_stats = pd.read_csv(STATS_FILE, dtype={"fips_code": str})
        print(f"[realtime] County stats loaded: {len(_county_stats)} counties.")
    except FileNotFoundError:
        print(
            "[realtime] WARNING: county_stats.csv not found. "
            "Run export_county_stats.py to generate it. "
            "County-level historical features will default to 0."
        )
        _county_stats = pd.DataFrame(columns=[
            "fips_code", "county_median_duration", "county_long_rate",
            "county_median_scope", "county_large_outage_rate", "county_max_customers",
        ])
    except Exception as e:
        print(f"[realtime] ERROR loading county stats: {e}")
        _county_stats = pd.DataFrame(columns=[
            "fips_code", "county_median_duration", "county_long_rate",
            "county_median_scope", "county_large_outage_rate", "county_max_customers",
        ])

    init_fips_cache(GEO_FILE)

    return ok


# ── Haversine distance ─────────────────────────────────────────────────────────

# def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     """Return great-circle distance in km between two lat/lon points."""
#     R = 6371.0
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = (
#         math.sin(dlat / 2) ** 2
#         + math.cos(math.radians(lat1))
#         * math.cos(math.radians(lat2))
#         * math.sin(dlon / 2) ** 2
#     )
#     return R * 2 * math.asin(math.sqrt(a))


# def _nearest_fips(lat: float, lon: float) -> Optional[str]:
#     """Return the FIPS code of the nearest Virginia county centroid."""
#     if _geo is None:
#         return None
#     best_fips, best_dist = None, float("inf")
#     for _, row in _geo.iterrows():
#         d = _haversine(lat, lon, row["latitude"], row["longitude"])
#         if d < best_dist:
#             best_dist = d
#             best_fips = row["fips"]
#     return best_fips


# ── Open-Meteo weather fetching ────────────────────────────────────────────────

def _fetch_one_county_weather(lat: float, lon: float) -> Optional[dict]:
    """Fetch today's hourly + daily weather for one lat/lon from Open-Meteo.
    Retries up to _WEATHER_RETRIES times with a short delay on failure."""
    params = (
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,snowfall,snow_depth,"
        "windspeed_10m,windgusts_10m,weathercode"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=America%2FNew_York"
    )
    url = f"{_OPEN_METEO_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(_WEATHER_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception:
            if attempt < _WEATHER_RETRIES - 1:
                time.sleep(_WEATHER_RETRY_DELAY * (attempt + 1))
    return None


def _aggregate_to_county_day(fips: str, data: dict, now: datetime) -> dict:
    """
    Collapse 24 Open-Meteo hourly rows into one county-day feature row.
    Unit conversions match the training data exactly:
        snowfall cm  → mm  (*10)
        snow_depth m → mm  (*1000)
        wind km/h    → m/s (/3.6)
    """
    h = data["hourly"]
    d = data["daily"]

    precip_vals  = [v or 0.0 for v in h["precipitation"]]
    snow_vals    = [(v or 0.0) * 10   for v in h["snowfall"]]      # cm → mm
    snwd_vals    = [(v or 0.0) * 1000 for v in h["snow_depth"]]    # m  → mm
    wind_ms_vals = [(v or 0.0) / 3.6  for v in h["windspeed_10m"]] # km/h → m/s
    gust_ms_vals = [(v or 0.0) / 3.6  for v in h["windgusts_10m"]] # km/h → m/s
    codes        = [int(v or 0)        for v in h["weathercode"]]

    prcp_mm  = sum(precip_vals)
    snow_mm  = sum(snow_vals)
    snwd_mm  = max(snwd_vals)
    awnd_ms  = sum(wind_ms_vals) / max(len(wind_ms_vals), 1)
    wsfg_ms  = max(gust_ms_vals)
    tmax_c   = d["temperature_2m_max"][0] or 0.0
    tmin_c   = d["temperature_2m_min"][0] or 0.0

    # Binary flags: 1 if ANY hour triggered the flag
    flags = {
        flag: int(any(c in code_set for c in codes))
        for flag, code_set in _CODE_FLAGS.items()
    }

    has_event = int(any(c >= 95 for c in codes))

    row = {
        "fips_code":         fips,
        "year":              now.year,
        "month":             now.month,
        "day":               now.day,
        "hour":              now.hour,
        "dayofweek":         now.weekday(),
        "dayofyear":         now.timetuple().tm_yday,
        "is_weekend":        int(now.weekday() >= 5),
        "prcp_mm":           round(prcp_mm, 3),
        "snow_mm":           round(snow_mm, 3),
        "snwd_mm":           round(snwd_mm, 3),
        "tmax_c":            tmax_c,
        "tmin_c":            tmin_c,
        "awnd_ms":           round(awnd_ms, 3),
        "wsfg_ms":           round(wsfg_ms, 3),
        "has_weather_event": has_event,
        "max_magnitude":     round(max(prcp_mm, awnd_ms), 3),
        "magnitude_missing": 0,
    }
    row.update(flags)
    return row


def _make_zero_weather_row(fips: str, now: datetime) -> dict:
    """Fallback row for a county whose API call failed."""
    row = {
        "fips_code":         fips,
        "year":              now.year,
        "month":             now.month,
        "day":               now.day,
        "hour":              now.hour,
        "dayofweek":         now.weekday(),
        "dayofyear":         now.timetuple().tm_yday,
        "is_weekend":        int(now.weekday() >= 5),
        "prcp_mm":           0.0,
        "snow_mm":           0.0,
        "snwd_mm":           0.0,
        "tmax_c":            0.0,
        "tmin_c":            0.0,
        "awnd_ms":           0.0,
        "wsfg_ms":           0.0,
        "has_weather_event": 0,
        "max_magnitude":     0.0,
        "magnitude_missing": 1,   # mark as missing so model knows
    }
    row.update({flag: 0 for flag in _CODE_FLAGS})
    return row


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _fetch_weather_all_counties(now: datetime) -> pd.DataFrame:
    """
    Fetch + aggregate weather for all 133 Virginia counties in parallel.
    On failure, retries are handled in _fetch_one_county_weather. After all
    fetches, any county that still failed gets its weather filled from the
    nearest successful county rather than zeros.
    Returns a county-day DataFrame (one row per county).
    """
    rows: List[Optional[dict]] = [None] * len(_geo)
    failed_indices: List[int] = []

    def _worker(idx: int, fips: str, lat: float, lon: float):
        data = _fetch_one_county_weather(lat, lon)
        if data is None:
            return idx, None
        return idx, _aggregate_to_county_day(fips, data, now)

    with ThreadPoolExecutor(max_workers=_WEATHER_WORKERS) as executor:
        futures = {
            executor.submit(_worker, i, row["fips"], row["latitude"], row["longitude"]): i
            for i, row in _geo.iterrows()
        }
        for future in as_completed(futures):
            idx, row_dict = future.result()
            rows[idx] = row_dict

    # Identify failures
    for i, r in enumerate(rows):
        if r is None:
            failed_indices.append(i)

    if failed_indices:
        print(f"[realtime] Weather failed for {len(failed_indices)} / {len(_geo)} counties after retries.")

        # Build list of successful (lat, lon, row_dict) for nearest-neighbor lookup
        successful = [
            (_geo.at[i, "latitude"], _geo.at[i, "longitude"], rows[i])
            for i in range(len(_geo)) if i not in set(failed_indices) and rows[i] is not None
        ]

        for i in failed_indices:
            fips = _geo.at[i, "fips"]
            lat  = _geo.at[i, "latitude"]
            lon  = _geo.at[i, "longitude"]

            if successful:
                # Copy weather from nearest county that succeeded, override fips
                nearest = min(successful, key=lambda x: _haversine_km(lat, lon, x[0], x[1]))
                filled = dict(nearest[2])
                filled["fips_code"] = fips
                filled["magnitude_missing"] = 1
                rows[i] = filled
                print(f"[realtime]   {fips}: weather filled from nearest county (not zeros).")
            else:
                # All counties failed — last resort zero row
                rows[i] = _make_zero_weather_row(fips, now)
                print(f"[realtime]   {fips}: all fetches failed, using zero row.")
    else:
        print(f"[realtime] Weather OK for all {len(_geo)} counties.")

    return pd.DataFrame(rows)

# ── Kubra instance IDs ────────────────────────────────────────────────────────

_KUBRA_BASE = "https://kubra.io"

_DOMINION = {
    "name": "Dominion Energy",
    "instance": "9c691bb6-767e-4532-b00e-286ac9adc223",
    "view":     "38b5394c-8bca-4dfd-ac59-b321615446bd",
    "thematic": "thematic-1",
}

_APPALACHIAN = {
    "name": "Appalachian Power",
    "instance": "6674f49e-0236-4ed8-a40a-b31747557ab7",
    "view":     "8cfe790f-59f3-4ce3-a73f-a9642227411f",
    "thematic": "thematic-2",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ok(label: str):
    print(f"  [OK] {label}")


def _fail(label: str, e: Exception):
    print(f"  [FAIL] {label}: {e}")

_FIPS_CACHE = {}

def init_fips_cache(file_path: Path):
    global _FIPS_CACHE
    try:
        df = pd.read_csv(file_path, dtype={"county_name": str, "fips": str})
        if df.empty:
            raise ValueError(f"File {file_path.name} is empty.")
        
        for _, row in df.iterrows():
            fips = str(row["fips"])
            raw_name = row["county_name"].lower().replace(", virginia","").strip()

            aliases = [raw_name]
            # Strip only the FINAL word if it is "county" or "city" so that
            # compound names like "James City County" → "James City" not "James"
            for suffix in (" county", " city"):
                if raw_name.endswith(suffix):
                    aliases.append(raw_name[: -len(suffix)].strip())
                    break

            for a in aliases:
                if a not in _FIPS_CACHE or "county" in raw_name:
                    _FIPS_CACHE[a] = fips

        _ok(f"Successfully initialized FIPS Cache with {len(_FIPS_CACHE)} keys.")

    except Exception as e:
        print(f"[ERROR] Failed to load {file_path.name}: {e}")
        raise

def _name_to_fips(name: str) -> str:

    return _FIPS_CACHE.get(name.lower(), "")

# ── Kubra / Dominion outage fetching ──────────────────────────────────────────

def _fetch_kubra_by_fips(cfg: dict) -> Dict[str, float]:
    """
    Fetch live Dominion outage data via Kubra StormCenter quadkey tiles,
    map each outage point to its nearest Virginia county FIPS code, and
    aggregate to county-level customer counts.

    Returns:
        dict: { fips_code (str) -> customers_affected (float) }
        Empty dict if Kubra is unreachable or scraping fails.
    """
    try:

        state_url = (
                f"{_KUBRA_BASE}/stormcenter/api/v1/stormcenters"
                f"/{cfg['instance']}/views/{cfg['view']}/currentState?preview=false"
            )
        try:
            state = _get(state_url)
            data_path = state["data"]["interval_generation_data"]
            _ok(f"currentState  data_path={data_path}")
        except (urllib.error.URLError, KeyError, Exception) as e:
            _fail("currentState", e)
            return {}

        # Step 2: fetch county-level outage thematic layer
        thematic_url = f"{_KUBRA_BASE}/{data_path}/public/{cfg['thematic']}/thematic_areas.json"

        counties = _get(thematic_url)
        records = counties.get("file_data", counties)

        fips_counts : Dict[str,float]= {}
        seen_names: Dict[str, str] = {}   # name -> fips, for duplicate detection
        for record in records:
            name = record.get("title", "")
            desc = record.get("desc", {})
            cust_a = desc.get("cust_a", {})
            if isinstance(cust_a, dict):
                cust_a = float(cust_a.get("val", 0))
            else:
                cust_a = 0.0

            if cust_a <= 0:
                continue

            fips = _name_to_fips(name)
            if fips.isdigit():
                if name in seen_names and seen_names[name] == fips:
                    print(f"[realtime] WARNING: duplicate Kubra record for '{name}' "
                          f"(fips={fips}, adding {cust_a} to existing "
                          f"{fips_counts.get(fips, 0.0)}) — possible double-count")
                seen_names[name] = fips
                fips_counts[fips] = cust_a + fips_counts.get(fips, 0.0)
            else:
                print(f"[realtime] Unknown region name: {name}")

        total = sum(fips_counts.values())
        print(f"[realtime] Kubra: {len(fips_counts)} counties affected, "
              f"{int(total):,} customers total.")        

        return fips_counts

    except ImportError:
        print("[realtime] WARNING: 'requests' or 'polyline' not installed. "
              "Run: pip install requests polyline")
        return {}
    except Exception as e:
        print(f"[realtime] Kubra fetch failed: {e}. Defaulting to no live outages.")
        return {}


# ── Event flag synthesis ───────────────────────────────────────────────────────

def _compute_event_flags(row) -> dict:
    """Map aggregated weather features to NOAA storm event type binary flags."""
    prcp    = row.get("prcp_mm",  0.0)
    snow    = row.get("snow_mm",  0.0)
    snwd    = row.get("snwd_mm",  0.0)
    wind    = row.get("awnd_ms",  0.0)
    gust    = row.get("wsfg_ms",  0.0)
    tmin    = row.get("tmin_c",   0.0)
    fog     = row.get("wt_fog",   0)
    thunder = row.get("wt_thunder", 0)
    ice     = row.get("wt_ice",   0)
    frzrn   = row.get("wt_freezing_rain", 0)
    bsnow   = row.get("wt_blowing_snow",  0)
    wsnow   = row.get("wt_snow",  0)

    flags = {
        "event_Thunderstorm Wind":      int(thunder and gust > 13),
        "event_Lightning":              int(thunder),
        "event_Hail":                   int(thunder and gust > 18),
        "event_High Wind":              int(gust > 25),
        "event_Strong Wind":            int(wind > 13 and gust <= 25),
        "event_Dense Fog":              int(fog),
        "event_Heavy Rain":             int(prcp > 25),
        "event_Flash Flood":            int(prcp > 50),
        "event_Flood":                  int(prcp > 30),
        "event_Heavy Snow":             int(wsnow and snow > 100),
        "event_Blizzard":               int(bsnow and wind > 15),
        "event_Winter Storm":           int((snow > 50 or snwd > 50) and tmin < 0),
        "event_Winter Weather":         int(wsnow and snow <= 100),
        "event_Ice Storm":              int(ice or frzrn),
        "event_Frost/Freeze":           int(tmin < 0 and prcp == 0),
        "event_Avalanche":              0,
        "event_Coastal Flood":          0,
        "event_Cold/Wind Chill":        int(tmin < -10),
        "event_Debris Flow":            0,
        "event_Drought":                0,
        "event_Excessive Heat":         int(row.get("tmax_c", 0) > 38),
        "event_Extreme Cold/Wind Chill":int(tmin < -20),
        "event_Funnel Cloud":           0,
        "event_Heat":                   int(row.get("tmax_c", 0) > 32),
        "event_Rip Current":            0,
        "event_Tornado":                0,
        "event_Tropical Storm":         int(gust > 33 and prcp > 25),
        "event_Wildfire":               0,
    }
    flags["event_None"] = int(not any(flags.values()))
    return flags


# ── Feature alignment ──────────────────────────────────────────────────────────

def _align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Given a DataFrame, ensure it contains exactly the columns in feature_names
    in that order. Any missing columns are added as zeros.
    This handles event_* dummy columns that exist in training but not at inference.
    """
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# ── Tier 1 Anomaly Detection ──────────────────────────────────────────────────────────

def _anomaly_detection_tier1(df: pd.DataFrame) -> pd.Series:
    """
    Given a dataframe, look for any anamolous values for weather data (outside
    of certain z-scores, etc.). 
    
    Return a series with each county that has an anomaly
    """

    anomalies = pd.Series(False, index=df.index)

    outlier_mask = pd.Series((df["prcp_mm"] > 500) | (df["tmax_c"] > 60) | (df["tmin_c"] < -40) 
                             | (df["awnd_ms"] > 25) | (df["wsfg_ms"] > 80) | (df["awnd_ms"] < 0) | (df["wsfg_ms"] < 0))
    
    anomalies = anomalies | outlier_mask

    for col in ["prcp_mm","tmax_c","tmin_c","awnd_ms","prcp_mm", "snow_mm"]:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            z = np.abs((df[col] - mean)/std)
            anomalies = anomalies | (z > 4.0)
    return anomalies

# ── Main inference function ────────────────────────────────────────────────────

def run_inference() -> Dict[str, Any]:
    """
    Run the full real-time 3-stage prediction cascade for all Virginia counties.

    Flow:
        1. Fetch live weather from Open-Meteo (parallel, one request per county)
        2. Fetch live outages from Kubra StormCenter
        3. Merge county historical stats
        4. Stage 1 — Occurrence: which counties will have an outage today?
        5. Stage 2 — Scope: for predicted-outage counties, how many customers?
        6. Stage 3 — Duration: for predicted-outage counties, how long?
        7. Cache and return predictions as { fips: {occurrence, scope, duration} }

    Returns:
        dict: { fips_code (str) -> {"occurrence": bool, "scope": float,
                                    "duration": float, "occ_prob": float} }
    """
    global _prev_kubra, _prev_kubra_time, _cached_predictions, _cached_features, _last_updated

    if _occ_model is None:
        print("[realtime] Models not loaded. Call init() first.")
        return {}

    now = datetime.now(timezone.utc).astimezone()
    print(f"\n[realtime] === Inference run: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ===")

    # ── Step 1: Weather ────────────────────────────────────────────────────────
    print("[realtime] Fetching weather for all counties ...")
    weather_df = _fetch_weather_all_counties(now)
    weather_df = weather_df.reset_index(drop=True)

    # ── Step 2: Kubra outages ──────────────────────────────────────────────────

    print("[realtime] Fetching Kubra outage data ...")

    current_kubra: Dict[str, float] = _fetch_kubra_by_fips(_DOMINION)

    app_kubra: Dict[str, float] = _fetch_kubra_by_fips(_APPALACHIAN)

    for fips, cust_a in app_kubra.items():
        current_kubra[fips] = current_kubra.get(fips, 0.0) + cust_a


    # Compute delta vs previous poll (for scope/duration models)
    delta_kubra:   Dict[str, float] = {}
    pct_growth:    Dict[str, float] = {}
    for fips, cur_cust in current_kubra.items():
        prev_cust = _prev_kubra.get(fips, cur_cust)  # no prior data → delta = 0
        delta     = cur_cust - prev_cust
        delta_kubra[fips] = delta
        pct_growth[fips]  = delta / max(prev_cust, 1.0)

    _prev_kubra      = current_kubra
    _prev_kubra_time = now

    # ── Step 3: Merge county stats ─────────────────────────────────────────────
    if len(_county_stats) > 0:
        stats_cols = [c for c in _county_stats.columns if c != "fips_code"]
        merge_stats = _county_stats[["fips_code"] + stats_cols].copy()
        weather_df = weather_df.merge(merge_stats, on="fips_code", how="left")

    # Fill any missing county-stat columns with 0 (graceful degradation)
    for col in [
        "county_median_duration", "county_long_rate",
        "county_median_scope", "county_large_outage_rate", "county_max_customers",
    ]:
        if col not in weather_df.columns:
            weather_df[col] = 0.0

    weather_df = weather_df.fillna(0.0).reset_index(drop=True)

    # ── Tier 1 Anomaly Detection: Statistics ───────────────────────────────────
    print("[realtime] Running Tier 1 Anomaly Detection (Statistical weather checks)...")
    anomalies = _anomaly_detection_tier1(weather_df)
    num_anomalies = anomalies.sum()
    if num_anomalies > 0:
        print(f"[realtime] [WARNING] Anomalous data detected in {anomalies.sum()} counties (tier1)")

    # ── Step 4: Stage 1 — Occurrence ──────────────────────────────────────────
    print("[realtime] Running Stage 1 (Occurrence) ...")
    occ_df = weather_df.copy()
    occ_df["fips_code"] = pd.to_numeric(occ_df["fips_code"], errors="coerce")

    X_occ = _align_features(occ_df, _occ_model.feature_columns)
    X_occ = X_occ.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ------------------- Autoencoder Integration -------------------
    if _ae_model is not None:
        # 1. Select only the features the autoencoder was trained on, then normalize
        if _ae_feature_columns is not None:
            X_ae = _align_features(X_occ, _ae_feature_columns)
        else:
            X_ae = X_occ

        if "year" in X_ae.columns:
                X_ae["year"] = 2021

        X_occ_np = X_ae.values.astype(float)
        X_occ_norm = (X_occ_np - _ae_mean) / _ae_std

        # --- DIAGNOSTIC PRINT BLOCK ---
        print("\n--- AE DIAGNOSTICS ---")
        print(f"Type of _ae_mean: {type(_ae_mean)}")
        print(f"First 5 values of _ae_mean: {_ae_mean[:5]}")
        print(f"First 5 values of _ae_std: {_ae_std[:5]}")
        
        # Check for invalid math (NaNs or Infinities)
        invalid_count = np.sum(~np.isfinite(X_occ_norm))
        print(f"Number of NaN/Inf values in X_occ_norm: {invalid_count}")
        
        if len(X_occ_norm) > 0:
            print(f"First row of X_occ_norm (first 5 cols): {X_occ_norm[0][:5]}")
        print("----------------------\n")
        # ------------------------------


        # 2. Detect anomalies
        ae_errors, anomaly_flags = _ae_model.detect(X_occ_norm, _ae_threshold)

        # 3. Add results to occ_df for later merging into results
        occ_df["ae_error"] = ae_errors
        occ_df["anomaly_flag"] = anomaly_flags

        n_anomalies = anomaly_flags.sum()
        print(f"[AE] Detected {n_anomalies} anomalous counties out of {len(X_occ)}")
    # --------------------------------------------------------------

    occ_preds, occ_probs = _occ_model.predict(X_occ)

    occ_mask = pd.Series(occ_preds == 1, index=weather_df.index)
    n_flagged = occ_mask.sum()
    print(f"[realtime] Stage 1 complete: {n_flagged} / {len(weather_df)} counties flagged.")

    # ── Steps 5 & 6: Stages 2 & 3 — Scope and Duration ───────────────────────
    scope_preds = np.zeros(len(weather_df))
    dur_preds   = np.zeros(len(weather_df))

    if n_flagged > 0:
        print(f"[realtime] Running Stages 2 & 3 (Scope + Duration) on {n_flagged} counties ...")
        event_df = weather_df.loc[occ_mask].copy().reset_index(drop=True)

        # Attach live Kubra features
        event_df["initial_customers_affected"] = (
            event_df["fips_code"].map(lambda f: current_kubra.get(str(int(f)), 0.0))
        )
        event_df["delta_customers_affected_15m"] = (
            event_df["fips_code"].map(lambda f: delta_kubra.get(str(int(f)), 0.0))
        )
        event_df["pct_growth_15m"] = (
            event_df["fips_code"].map(lambda f: pct_growth.get(str(int(f)), 0.0))
        )

        # initial_impact_density: fraction of max county customers currently out
        max_cust = event_df["county_max_customers"].clip(lower=1)
        event_df["initial_impact_density"] = (
            event_df["initial_customers_affected"] / max_cust
        ).clip(upper=1.0)

        event_flags_df = event_df.apply(_compute_event_flags, axis=1, result_type="expand")
        event_df = pd.concat([event_df, event_flags_df], axis=1)

        # Convert fips to numeric for XGBoost
        event_df["fips_code"] = pd.to_numeric(event_df["fips_code"], errors="coerce")
        event_df = event_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Stage 2: Scope
        X_scope = _align_features(event_df.copy(), _scope_model.featureNames)
        raw_scope = _scope_model.predict(X_scope)
        scope_preds[occ_mask.values] = raw_scope

        # Stage 3: Duration
        X_dur = _align_features(event_df.copy(), _dur_model.featureNames)
        raw_dur = _dur_model.predict(X_dur)
        dur_preds[occ_mask.values] = raw_dur

        print("[realtime] Stages 2 & 3 complete.")

        # Scope sanity check — log any county predicted above 2x its historical max
        for pos, (_, row) in enumerate(event_df.iterrows()):
            pred_scope = raw_scope[pos]
            hist_max   = float(row.get("county_max_customers", 0))
            if hist_max > 0 and pred_scope > hist_max * 2:
                print(f"[scope-check] fips={row.get('fips_code','')}  "
                      f"predicted={pred_scope:.0f}  hist_max={hist_max:.0f}  "
                      f"kubra_live={row.get('initial_customers_affected',0):.0f}  "
                      f"tmax={row.get('tmax_c',0):.1f}C  "
                      f"prcp={row.get('prcp_mm',0):.1f}mm  "
                      f"wind={row.get('wsfg_ms',0):.1f}m/s  "
                      f"magnitude_missing={int(row.get('magnitude_missing',0))}")


    # ── Step 7: Build and cache output ────────────────────────────────────────
    # Cache weather + aligned model features per county for the /api/explain endpoint
    event_dict = {}
    if n_flagged > 0:
        for _, row in event_df.iterrows():
            fips_key = str(row["fips_code"])
            event_dict[fips_key] = row.to_dict()
    
    _weather_display_cols = [
        "tmax_c", "tmin_c", "awnd_ms", "wsfg_ms", "prcp_mm", "snow_mm", "snwd_mm",
        "wt_thunder", "wt_fog", "wt_snow", "wt_freezing_rain",
        "wt_ice", "wt_blowing_snow", "wt_drizzle", "wt_rain", "has_weather_event",
    ]
    new_features = {}
    for i in weather_df.index:
        fips = str(weather_df.at[i, "fips_code"])
        model_row = {col: float(X_occ.at[i, col]) for col in X_occ.columns}
        for col in _scope_model.featureNames + _dur_model.featureNames:
            if col not in model_row:
                model_row[col] = 0.0
        if fips in event_dict:
            for col in _scope_model.featureNames + _dur_model.featureNames:
                if col in event_dict[fips]:
                    model_row[col] = float(event_dict[fips][col])
        new_features[fips] = {
            "weather":   {k: float(weather_df.at[i, k]) for k in _weather_display_cols if k in weather_df.columns},
            "model_row": model_row,
        }
    with _cache_lock:
        _cached_features.update(new_features)


    results: Dict[str, Any] = {}
    for i in weather_df.index:
        fips = str(weather_df.at[i, "fips_code"])
        occ  = bool(occ_preds[i])
        prob = round(float(occ_probs[i]), 4)
        anomaly    = bool(occ_df.at[i, "anomaly_flag"]) if "anomaly_flag" in occ_df.columns else False
        ae_err     = round(float(occ_df.at[i, "ae_error"]), 4) if "ae_error" in occ_df.columns else 0.0
        raw_scope  = round(float(scope_preds[i]), 1) if occ else 0.0
        raw_dur    = round(float(dur_preds[i]) / 60.0, 2) if occ else 0.0
        # Suppress low-scope predictions — occurrence model flags ~94% of counties because
        # it was trained on customers_out > 0 (any outage). Filter to meaningful events.
        significant = occ and raw_scope >= _MIN_SCOPE_CUSTOMERS
        results[fips] = {
            "occurrence":   significant,
            "scope":        raw_scope if significant else 0.0,
            "duration":     raw_dur   if significant else 0.0,
            "occ_prob":     prob,
            "anomaly_flag": anomaly,
            "ae_error":     ae_err,
        }

    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    with _cache_lock:
        _cached_predictions = results
        _last_updated = ts

    n_out = sum(1 for v in results.values() if v["occurrence"])
    print(f"[realtime] Done. {n_out} counties predicted to have outages. ({ts})\n")
    return results


def get_cached_predictions() -> Dict[str, Any]:
    """Return the most recently cached predictions (thread-safe)."""
    with _cache_lock:
        return dict(_cached_predictions)


def get_last_updated() -> Optional[str]:
    """Return the timestamp string of the last successful inference run."""
    with _cache_lock:
        return _last_updated


def get_features_for_fips(fips: str) -> Optional[dict]:
    """Return the cached {weather, model_row} dict for a county (thread-safe)."""
    with _cache_lock:
        return _cached_features.get(fips)


def compute_shap_for_fips(fips: str, explainer_name: str) -> Optional[dict]:
    """
    Compute SHAP values for the occurrence model prediction for one county.
    Returns the top-10 features by absolute SHAP value, plus the base value.
    Returns None if the explainer is unavailable or features are not cached.
    """

    explainer, model = _EXPLAINER_NAMES[explainer_name]
    if explainer_name == '_occ_explainer':
        feature_names = model.feature_columns
    else:
        feature_names = model.featureNames

    if explainer is None or model is None:
        return None

    features = get_features_for_fips(fips)
    if not features:
        return None

    X_row = pd.DataFrame([features["model_row"]])[feature_names]
    shap_vals = explainer.shap_values(X_row)

    # TreeExplainer for binary classifier may return a list [neg_class, pos_class]
    # or a single array depending on the SHAP version
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1][0])
    else:
        sv = np.array(shap_vals[0])

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    model_row     = features["model_row"]

    # Normalize shap values
    sv = np.array(sv).astype(float).flatten()

    # Return top 10 features sorted by absolute SHAP contribution
    ranked = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )[:10]

    return {
        "base_value": round(base_value, 4),
        "features": [
            {
                "name":  name,
                "value": round(float(model_row.get(name, 0.0)), 3),
                "shap":  round(float(sv_val), 4),
            }
            for name, sv_val in ranked
        ],
    }

# --- Autoencoder loading ---

def load_autoencoder(path=MODELS_DIR / "autoencoder.pt"):
    """
    Load the autoencoder to detect anomalous counties
    Called in init()
    """
    global _ae_model, _ae_mean, _ae_std, _ae_threshold, _ae_feature_columns

    # Allow the numpy reconstruct function to unpickle properly
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # checkpoint = torch.load(path, map_location="cpu")

    input_dim = checkpoint["input_dim"]
    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _ae_mean = np.array(checkpoint["mean"])
    _ae_std = np.array(checkpoint["std"])
    _ae_threshold = checkpoint["threshold"]
    _ae_feature_columns = checkpoint.get("feature_columns")

    _ae_model = model


    if _ae_feature_columns:
        print(f"[AE] Successfully loaded {len(_ae_feature_columns)} feature columns for alignment.")
        print(_ae_feature_columns)
    else:
        print("[AE] ERROR: No feature_columns found in .pt file!")

    print(f"[AE] Loaded autoencoder from {path}, threshold={_ae_threshold:.6f}")


# ── 7-day forecast ─────────────────────────────────────────────────────────────

def _fetch_one_county_forecast(lat: float, lon: float, days: int = 7) -> Optional[dict]:
    """Fetch a multi-day hourly + daily forecast for one lat/lon from Open-Meteo."""
    params = (
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,snowfall,snow_depth,"
        "windspeed_10m,windgusts_10m,weathercode"
        "&daily=temperature_2m_max,temperature_2m_min"
        f"&forecast_days={days}&timezone=America%2FNew_York"
    )
    url = f"{_OPEN_METEO_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(_WEATHER_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception:
            if attempt < _WEATHER_RETRIES - 1:
                time.sleep(_WEATHER_RETRY_DELAY)
    return None


def _aggregate_forecast_days(fips: str, data: dict) -> List[dict]:
    """
    Slice a multi-day Open-Meteo response into one feature row per day.
    Returns a list of dicts ordered by day (index 0 = today).
    """
    h      = data["hourly"]
    d      = data["daily"]
    n_days = len(d["time"])
    rows   = []

    for day_idx in range(n_days):
        start = day_idx * 24
        end   = start + 24

        precip_vals  = [v or 0.0 for v in h["precipitation"][start:end]]
        snow_vals    = [(v or 0.0) * 10   for v in h["snowfall"][start:end]]
        snwd_vals    = [(v or 0.0) * 1000 for v in h["snow_depth"][start:end]]
        wind_ms_vals = [(v or 0.0) / 3.6  for v in h["windspeed_10m"][start:end]]
        gust_ms_vals = [(v or 0.0) / 3.6  for v in h["windgusts_10m"][start:end]]
        codes        = [int(v or 0)        for v in h["weathercode"][start:end]]

        prcp_mm = sum(precip_vals)
        snow_mm = sum(snow_vals)
        snwd_mm = max(snwd_vals) if snwd_vals else 0.0
        awnd_ms = sum(wind_ms_vals) / max(len(wind_ms_vals), 1)
        wsfg_ms = max(gust_ms_vals) if gust_ms_vals else 0.0
        tmax_c  = d["temperature_2m_max"][day_idx] or 0.0
        tmin_c  = d["temperature_2m_min"][day_idx] or 0.0

        from datetime import datetime as _dt
        dt = _dt.strptime(d["time"][day_idx], "%Y-%m-%d")
        flags = {
            flag: int(any(c in code_set for c in codes))
            for flag, code_set in _CODE_FLAGS.items()
        }

        row = {
            "fips_code":         fips,
            "date":              d["time"][day_idx],
            "year":              dt.year,
            "month":             dt.month,
            "day":               dt.day,
            "hour":              12,
            "dayofweek":         dt.weekday(),
            "dayofyear":         dt.timetuple().tm_yday,
            "is_weekend":        int(dt.weekday() >= 5),
            "prcp_mm":           round(prcp_mm, 3),
            "snow_mm":           round(snow_mm, 3),
            "snwd_mm":           round(snwd_mm, 3),
            "tmax_c":            tmax_c,
            "tmin_c":            tmin_c,
            "awnd_ms":           round(awnd_ms, 3),
            "wsfg_ms":           round(wsfg_ms, 3),
            "has_weather_event": int(any(c >= 95 for c in codes)),
            "max_magnitude":     round(max(prcp_mm, awnd_ms), 3),
            "magnitude_missing": 0,
        }
        row.update(flags)
        rows.append(row)

    return rows


def run_forecast(days: int = 7) -> Dict[str, Any]:
    """
    Fetch a multi-day weather forecast for all 133 counties and run the
    occurrence + scope + duration cascade for each day.

    Returns:
        { date_str: { fips: {occurrence, scope, duration, occ_prob} } }
    No live Kubra data is used — future outage counts are unknown, so
    initial_customers_affected and related features default to 0.
    """
    global _cached_forecast

    if _occ_model is None:
        print("[forecast] Models not loaded.")
        return {}

    print(f"\n[forecast] === Forecast run ({days} days) ===")

    # Fetch multi-day weather in parallel (same worker limit as realtime)
    all_county_rows: List[Optional[List[dict]]] = [None] * len(_geo)

    def _worker(idx: int, fips: str, lat: float, lon: float):
        data = _fetch_one_county_forecast(lat, lon, days=days)
        if data is None:
            return idx, None
        return idx, _aggregate_forecast_days(fips, data)

    with ThreadPoolExecutor(max_workers=_WEATHER_WORKERS) as executor:
        futures = {
            executor.submit(_worker, i, row["fips"], row["latitude"], row["longitude"]): i
            for i, row in _geo.iterrows()
        }
        for future in as_completed(futures):
            idx, rows = future.result()
            all_county_rows[idx] = rows

    # Get date labels from any successful fetch
    date_keys: Optional[List[str]] = None
    for r in all_county_rows:
        if r is not None:
            date_keys = [row["date"] for row in r]
            break
    if date_keys is None:
        print("[forecast] All weather fetches failed.")
        return {}

    n_failed = sum(1 for r in all_county_rows if r is None)
    if n_failed:
        print(f"[forecast] {n_failed} counties failed — filling from nearest neighbor.")
        successful_idx = [i for i, r in enumerate(all_county_rows) if r is not None]
        for i, rows in enumerate(all_county_rows):
            if rows is not None:
                continue
            lat = _geo.at[i, "latitude"]
            lon = _geo.at[i, "longitude"]
            fips = _geo.at[i, "fips"]
            nearest_idx = min(
                successful_idx,
                key=lambda j: _haversine_km(lat, lon, _geo.at[j, "latitude"], _geo.at[j, "longitude"])
            )
            filled = []
            for day_row in all_county_rows[nearest_idx]:
                d = dict(day_row)
                d["fips_code"] = fips
                d["magnitude_missing"] = 1
                filled.append(d)
            all_county_rows[i] = filled

    results_by_date: Dict[str, Any] = {}

    for day_idx, date_str in enumerate(date_keys):
        day_rows = [all_county_rows[i][day_idx] for i in range(len(_geo))]
        day_df   = pd.DataFrame(day_rows).reset_index(drop=True)

        # Merge county historical stats
        if len(_county_stats) > 0:
            stats_cols = [c for c in _county_stats.columns if c != "fips_code"]
            day_df = day_df.merge(
                _county_stats[["fips_code"] + stats_cols], on="fips_code", how="left"
            )
        for col in ["county_median_duration", "county_long_rate",
                    "county_median_scope", "county_large_outage_rate", "county_max_customers"]:
            if col not in day_df.columns:
                day_df[col] = 0.0
        day_df = day_df.fillna(0.0).reset_index(drop=True)

        # Stage 1: Occurrence
        occ_df = day_df.copy()
        occ_df["fips_code"] = pd.to_numeric(occ_df["fips_code"], errors="coerce")
        X_occ = _align_features(occ_df, _occ_model.feature_columns)
        X_occ = X_occ.apply(pd.to_numeric, errors="coerce").fillna(0)
        occ_preds, occ_probs = _occ_model.predict(X_occ)
        occ_mask = pd.Series(occ_preds == 1, index=day_df.index)

        scope_preds = np.zeros(len(day_df))
        dur_preds   = np.zeros(len(day_df))

        if occ_mask.sum() > 0:
            event_df = day_df.loc[occ_mask].copy().reset_index(drop=True)

            # No live Kubra data for future days
            event_df["initial_customers_affected"]   = 0.0
            event_df["delta_customers_affected_15m"] = 0.0
            event_df["pct_growth_15m"]               = 0.0
            event_df["initial_impact_density"]       = 0.0
            cols_to_drop = ["initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m",
                            "initial_impact_density"]
            event_df = event_df.drop(columns=cols_to_drop)

            event_flags_df = event_df.apply(_compute_event_flags, axis=1, result_type="expand")
            event_df = pd.concat([event_df, event_flags_df], axis=1)
            event_df["fips_code"] = pd.to_numeric(event_df["fips_code"], errors="coerce")
            event_df = event_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            X_scope = _align_features(event_df.copy(), _scope_forecast.featureNames)
            scope_preds[occ_mask.values] = _scope_forecast.predict(X_scope)

            X_dur = _align_features(event_df.copy(), _dur_forecast.featureNames)
            dur_preds[occ_mask.values] = _dur_forecast.predict(X_dur)

        day_results: Dict[str, Any] = {}
        for i in day_df.index:
            fips      = str(day_df.at[i, "fips_code"])
            occ       = bool(occ_preds[i])
            prob      = round(float(occ_probs[i]), 4)
            raw_scope = round(float(scope_preds[i]), 1) if occ else 0.0
            raw_dur   = round(float(dur_preds[i]) / 60.0, 2) if occ else 0.0
            significant = occ and raw_scope >= _MIN_SCOPE_CUSTOMERS
            day_results[fips] = {
                "occurrence": significant,
                "scope":      raw_scope if significant else 0.0,
                "duration":   raw_dur   if significant else 0.0,
                "occ_prob":   prob,
            }

        n_out = sum(1 for v in day_results.values() if v["occurrence"])
        print(f"[forecast]   {date_str}: {n_out} / {len(day_df)} counties flagged")
        results_by_date[date_str] = day_results

    with _forecast_lock:
        _cached_forecast = results_by_date

    print("[forecast] Done.\n")
    return results_by_date


def get_cached_forecast() -> Dict[str, Any]:
    """Return the most recently cached forecast (thread-safe)."""
    with _forecast_lock:
        return dict(_cached_forecast)