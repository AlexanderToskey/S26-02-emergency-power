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
import threading
import urllib.request
import urllib.error
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

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

# Maximum parallel threads for weather fetching
_WEATHER_WORKERS = 20

# ── Module-level state ─────────────────────────────────────────────────────────
_occ_model:   Optional[OutageOccurrenceModel] = None
_scope_model: Optional[TwoStageScopeModel]    = None
_dur_model:   Optional[TwoStageOutageModel]   = None
_geo:         Optional[pd.DataFrame]          = None   # county centroids (133 rows)
_county_stats: Optional[pd.DataFrame]         = None   # precomputed county-level stats

_prev_kubra:      Dict[str, float]  = {}     # fips_code -> customers (last poll)
_prev_kubra_time: Optional[datetime] = None

_cached_predictions: Dict[str, Any] = {}
_cached_features:    Dict[str, Any] = {}   # fips -> {weather, model_row} for explain endpoint
_last_updated:       Optional[str]  = None
_cache_lock = threading.Lock()
_occ_explainer = None   # shap.TreeExplainer, initialized in init()


# ── Initialisation ─────────────────────────────────────────────────────────────

def init() -> bool:
    """
    Load trained models and static lookup tables.
    Call once at server startup before the first inference.
    Returns True if everything loaded cleanly, False if any component failed
    (the pipeline will still run with degraded features).
    """
    global _occ_model, _scope_model, _dur_model, _geo, _county_stats
    ok = True

    # --- Models ---
    try:
        _occ_model   = OutageOccurrenceModel.load(MODELS_DIR / "occurrence_model.joblib")
        _scope_model = TwoStageScopeModel.load(MODELS_DIR / "scope_model.joblib")
        _dur_model   = TwoStageOutageModel.load(MODELS_DIR / "duration_model.joblib")
        print("[realtime] All three models loaded successfully.")

        # Build SHAP explainer for the occurrence model
        try:
            import shap as _shap
            global _occ_explainer
            _occ_explainer = _shap.TreeExplainer(_occ_model.model)
            print("[realtime] SHAP explainer initialized.")
        except Exception as shap_err:
            print(f"[realtime] WARNING: SHAP explainer failed to initialize: {shap_err}")

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
    """Fetch today's hourly + daily weather for one lat/lon from Open-Meteo."""
    params = (
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,snowfall,snow_depth,"
        "windspeed_10m,windgusts_10m,weathercode"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=America%2FNew_York"
    )
    url = f"{_OPEN_METEO_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
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


def _fetch_weather_all_counties(now: datetime) -> pd.DataFrame:
    """
    Fetch + aggregate weather for all 133 Virginia counties in parallel.
    Returns a county-day DataFrame (one row per county).
    """
    rows: List[dict] = [None] * len(_geo)

    def _worker(idx: int, fips: str, lat: float, lon: float):
        data = _fetch_one_county_weather(lat, lon)
        if data is None:
            print(f"[realtime]   Weather failed for {fips}, using zeros.")
            return idx, _make_zero_weather_row(fips, now)
        return idx, _aggregate_to_county_day(fips, data, now)

    with ThreadPoolExecutor(max_workers=_WEATHER_WORKERS) as executor:
        futures = {
            executor.submit(_worker, i, row["fips"], row["latitude"], row["longitude"]): i
            for i, row in _geo.iterrows()
        }
        for future in as_completed(futures):
            idx, row_dict = future.result()
            rows[idx] = row_dict

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

            aliases = []

            aliases.append(raw_name)
            # aliases.append(raw_name.replace(", virginia","").strip())
            aliases.append(raw_name.replace(" county","").replace(" city","").strip())

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

    # ── Step 4: Stage 1 — Occurrence ──────────────────────────────────────────
    print("[realtime] Running Stage 1 (Occurrence) ...")
    occ_df = weather_df.copy()
    occ_df["fips_code"] = pd.to_numeric(occ_df["fips_code"], errors="coerce")

    X_occ = _align_features(occ_df, _occ_model.feature_columns)
    X_occ = X_occ.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Cache weather + aligned model features per county for the /api/explain endpoint
    _weather_display_cols = [
        "tmax_c", "tmin_c", "awnd_ms", "wsfg_ms", "prcp_mm", "snow_mm", "snwd_mm",
        "wt_thunder", "wt_fog", "wt_snow", "wt_freezing_rain",
        "wt_ice", "wt_blowing_snow", "wt_drizzle", "wt_rain", "has_weather_event",
    ]
    new_features = {}
    for i in weather_df.index:
        fips = str(weather_df.at[i, "fips_code"])
        new_features[fips] = {
            "weather":   {k: float(weather_df.at[i, k]) for k in _weather_display_cols if k in weather_df.columns},
            "model_row": {col: float(X_occ.at[i, col]) for col in X_occ.columns},
        }
    with _cache_lock:
        _cached_features.update(new_features)

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

        # Synthesize event_* columns from weather data.
        # These map to the NOAA storm event types the models were trained on.
        def _event_flags(row) -> dict:
            prcp  = row.get("prcp_mm",  0.0)
            snow  = row.get("snow_mm",  0.0)
            snwd  = row.get("snwd_mm",  0.0)
            wind  = row.get("awnd_ms",  0.0)
            gust  = row.get("wsfg_ms",  0.0)
            tmin  = row.get("tmin_c",   0.0)
            fog   = row.get("wt_fog",   0)
            thunder = row.get("wt_thunder", 0)
            ice   = row.get("wt_ice",   0)
            frzrn = row.get("wt_freezing_rain", 0)
            bsnow = row.get("wt_blowing_snow",  0)
            wsnow = row.get("wt_snow",  0)

            flags = {
                "event_Thunderstorm Wind": int(thunder and gust > 13),
                "event_Lightning":         int(thunder),
                "event_Hail":              int(thunder and gust > 18),
                "event_High Wind":         int(gust > 25),
                "event_Strong Wind":       int(wind > 13 and gust <= 25),
                "event_Dense Fog":         int(fog),
                "event_Heavy Rain":        int(prcp > 25),
                "event_Flash Flood":       int(prcp > 50),
                "event_Flood":             int(prcp > 30),
                "event_Heavy Snow":        int(wsnow and snow > 100),
                "event_Blizzard":          int(bsnow and wind > 15),
                "event_Winter Storm":      int((snow > 50 or snwd > 50) and tmin < 0),
                "event_Winter Weather":    int(wsnow and snow <= 100),
                "event_Ice Storm":         int(ice or frzrn),
                "event_Frost/Freeze":      int(tmin < 0 and prcp == 0),
                # remaining event types have no reliable real-time proxy → 0
                "event_Avalanche":         0,
                "event_Coastal Flood":     0,
                "event_Cold/Wind Chill":   int(tmin < -10),
                "event_Debris Flow":       0,
                "event_Drought":           0,
                "event_Excessive Heat":    int(row.get("tmax_c", 0) > 38),
                "event_Extreme Cold/Wind Chill": int(tmin < -20),
                "event_Funnel Cloud":      0,
                "event_Heat":              int(row.get("tmax_c", 0) > 32),
                "event_Rip Current":       0,
                "event_Tornado":           0,
                "event_Tropical Storm":    int(gust > 33 and prcp > 25),
                "event_Wildfire":          0,
            }
            any_event = any(flags.values())
            flags["event_None"] = int(not any_event)
            return flags

        event_flags_df = event_df.apply(_event_flags, axis=1, result_type="expand")
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

    # ── Step 7: Build and cache output ────────────────────────────────────────
    results: Dict[str, Any] = {}
    for i in weather_df.index:
        fips = str(weather_df.at[i, "fips_code"])
        occ  = bool(occ_preds[i])
        prob = round(float(occ_probs[i]), 4)
        results[fips] = {
            "occurrence": occ,
            "scope":      round(float(scope_preds[i]), 1) if occ else 0.0,
            "duration":   round(float(dur_preds[i]) / 60.0, 2) if occ else 0.0,  # min → hrs
            "occ_prob":   prob,
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


def compute_shap_for_fips(fips: str) -> Optional[dict]:
    """
    Compute SHAP values for the occurrence model prediction for one county.
    Returns the top-10 features by absolute SHAP value, plus the base value.
    Returns None if the explainer is unavailable or features are not cached.
    """
    if _occ_explainer is None or _occ_model is None:
        return None

    features = get_features_for_fips(fips)
    if not features:
        return None

    X_row = pd.DataFrame([features["model_row"]])[_occ_model.feature_columns]
    shap_vals = _occ_explainer.shap_values(X_row)

    # TreeExplainer for binary classifier may return a list [neg_class, pos_class]
    # or a single array depending on the SHAP version
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1][0])
    else:
        sv = np.array(shap_vals[0])

    base_value = _occ_explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    feature_names = _occ_model.feature_columns
    model_row     = features["model_row"]

    # Return top 10 features sorted by absolute SHAP contribution
    ranked = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
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
