"""
api_test.py - Test all real-time data APIs used in the outage prediction pipeline.

APIs tested:
  1. Open-Meteo     - Hourly/daily weather per lat/lon (no key required)
  2. US Census      - Virginia county FIPS + names (no key required)
  3. Dominion Energy  - Live outage data via Kubra StormCenter (no key required)
  4. Appalachian Power - Live outage data via Kubra StormCenter (no key required)

Usage:
    python api_test.py
"""

import json
import urllib.request
import urllib.error

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


# ── individual tests ──────────────────────────────────────────────────────────

def test_open_meteo():
    """Fetch 1-day hourly forecast for Richmond, VA (proxy lat/lon)."""
    print("\n--- Open-Meteo ---")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.54&longitude=-77.43"
        "&hourly=temperature_2m,precipitation,snowfall,windspeed_10m,weathercode"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=America/New_York"
    )
    try:
        data = _get(url)
        hourly_keys = list(data.get("hourly", {}).keys())
        daily_keys  = list(data.get("daily", {}).keys())
        _ok(f"hourly={hourly_keys}  daily={daily_keys}")
    except (urllib.error.URLError, Exception) as e:
        _fail("Open-Meteo forecast", e)


def test_census():
    """Fetch all Virginia county names + FIPS codes."""
    print("\n--- US Census ---")
    url = "https://api.census.gov/data/2020/dec/pl?get=NAME&for=county:*&in=state:51"
    try:
        data = _get(url)
        n = len(data) - 1  # first row is header
        _ok(f"{n} Virginia counties returned")
        print(f"     sample: {data[1]}")
    except (urllib.error.URLError, Exception) as e:
        _fail("Census county list", e)


def _test_kubra(cfg: dict):
    """Two-step Kubra fetch: currentState → thematic county data."""
    print(f"\n--- {cfg['name']} (Kubra) ---")

    # Step 1: get current data path (changes each refresh cycle)
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
        return

    # Step 2: fetch county-level outage thematic layer
    thematic_url = f"{_KUBRA_BASE}/{data_path}/public/{cfg['thematic']}/thematic_areas.json"
    try:
        counties = _get(thematic_url)
        records = counties.get("file_data", counties)
        if isinstance(records, list):
            n = len(records)
            sample = records[0] if records else {}
            name  = sample.get("title", "?")
            desc  = sample.get("desc", {})
            cust_a = desc.get("cust_a", {})
            if isinstance(cust_a, dict):
                cust_a = cust_a.get("val", 0)
            cust_s = desc.get("cust_s", "?")
            n_out  = desc.get("n_out", "?")
            _ok(
                f"{n} county records  "
                f"sample=({name}: {cust_a}/{cust_s} customers, {n_out} incidents)"
            )
        else:
            _ok(f"response keys: {list(records.keys())[:8]}")

        total = 0
        for i, record in enumerate(records):
            name = record.get("title", "Unknown")
            desc = record.get("desc", {})
            
            # Extract metrics
            cust_a = desc.get("cust_a", {}).get("val", 0) if isinstance(desc.get("cust_a"), dict) else 0
            cust_s = desc.get("cust_s", 0)
            n_out  = desc.get("n_out", 0)

            total += cust_a
            
            print(f"  {name:<25} | {n_out:<8} | {cust_a:<10} | {cust_s}")

        print('total_outage: ', total)
    except (urllib.error.URLError, Exception) as e:
        _fail("thematic county data", e)


def test_dominion():
    _test_kubra(_DOMINION)


def test_appalachian():
    _test_kubra(_APPALACHIAN)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("API TEST SUITE")
    print("=" * 50)

    test_open_meteo()
    test_census()
    test_dominion()
    test_appalachian()

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
