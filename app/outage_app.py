"""
outage_app.py - Flask web application for the Virginia Outage Prediction Dashboard.

Serves the interactive county map and exposes a JSON API consumed by the frontend.
On startup, loads all models and runs an initial inference pass so the first page
load is never empty. Background threads then refresh live predictions every 15
minutes and the 7-day forecast every hour.

API routes:
    GET /                       Render the map dashboard
    GET /api/counties           GeoJSON for all Virginia counties
    GET /api/predictions        Live county-level outage predictions
    GET /api/forecast           7-day county-level forecast
    GET /api/explain/<fips>     Weather inputs + SHAP values for one county
    GET /api/status             Model load status and last inference timestamp
    GET /api/logs               Last 500 server log lines

Usage:
    python outage_app.py
"""

from flask import Flask, render_template, jsonify, request
import collections
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
import realtime_inference

app = Flask(__name__)

# ── Log capture ────────────────────────────────────────────────────────────────

_log_buffer: collections.deque = collections.deque(maxlen=500)
_log_lock = threading.Lock()

class _TeeStream:
    """Writes to the original stdout and appends each line to _log_buffer."""
    def __init__(self, original):
        self._original = original

    def write(self, data):
        self._original.write(data)
        if data and data != "\n":
            with _log_lock:
                _log_buffer.append(data.rstrip("\n"))

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)

sys.stdout = _TeeStream(sys.stdout)

#  Startup: load models and run first inference

def _background_refresh(interval_seconds: int = 900):
    """Re-runs live inference every 15 minutes."""
    while True:
        try:
            realtime_inference.run_inference()
        except Exception as e:
            print(f"[app] Background inference error: {e}")
        time.sleep(interval_seconds)


def _background_forecast(interval_seconds: int = 3600):
    """Re-runs 7-day forecast every hour."""
    while True:
        try:
            realtime_inference.run_forecast(days=7)
        except Exception as e:
            print(f"[app] Background forecast error: {e}")
        time.sleep(interval_seconds)


def _startup():
    """Load models, run an initial inference, then launch the refresh thread."""
    ok = realtime_inference.init()
    if not ok:
        print("[app] WARNING: Some components failed to load. "
              "Predictions may be degraded or unavailable.")

    # First inference run (blocking, so the first API call is never empty)
    print("[app] Running initial inference ...")
    try:
        realtime_inference.run_inference()
    except Exception as e:
        print(f"[app] Initial inference failed: {e}")

    # Background live-inference thread (every 15 min)
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    print("[app] Background refresh thread started (interval: 15 min).")

    # Initial forecast run then hourly refresh thread
    print("[app] Running initial 7-day forecast ...")
    try:
        realtime_inference.run_forecast(days=7)
    except Exception as e:
        print(f"[app] Initial forecast failed: {e}")
    tf = threading.Thread(target=_background_forecast, daemon=True)
    tf.start()
    print("[app] Background forecast thread started (interval: 60 min).")


# Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/counties")
def counties():
    filepath = os.path.join(os.path.dirname(__file__), "static", "counties.json")
    with open(filepath) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/predictions")
def predictions():
    """
    Returns live county-level outage predictions.
    Each key is a 5-digit FIPS code; each value contains:
        occurrence  bool    whether an outage is predicted
        scope       float   predicted customers affected (0 if occurrence=False)
        duration    float   predicted duration in hours  (0 if occurrence=False)
        occ_prob    float   raw occurrence probability (0–1)
    """
    data = realtime_inference.get_cached_predictions()
    if not data:
        return jsonify({"error": "Predictions not yet available. Try again shortly."}), 503
    return jsonify(data)


_EXPLAINER_NAMES = [
    "_occ_explainer",
    "_scope_explainer_small",
    "_scope_explainer_large",
    "_scope_explainer_classifier",
    "_duration_explainer_small",
    "_duration_explainer_large",
    "_duration_explainer_classifier",
    "_scope_forecast_explainer_small",
    "_scope_forecast_explainer_large",
    "_scope_forecast_explainer_classifier",
    "_duration_forecast_explainer_small",
    "_duration_forecast_explainer_large",
    "_duration_forecast_explainer_classifier",
]

@app.route("/api/explain/<fips>")
def explain(fips):
    """
    Returns weather inputs and SHAP feature contributions for one county.
    Used by the explanation panel when a county is clicked.
    Response shape:
        { fips, occurrence, occ_prob, scope, duration,
          weather: { tmax_c, tmin_c, awnd_ms, wsfg_ms, prcp_mm, ... },
          shap:    { base_value, features: [{name, value, shap}, ...] } }
    """

    date_str = request.args.get('date', 'Live')

    
    if date_str == "Live":
        pred = realtime_inference.get_cached_predictions().get(fips)
    else:
        # Look inside the forecast cache for that specific date
        forecast = realtime_inference.get_cached_forecast()
        pred = forecast.get(date_str, {}).get(fips)
    if pred is None:
        return jsonify({"error": "No prediction cached for this county."}), 404

    features = realtime_inference.get_features_for_fips(fips, date_str=date_str)
    if not features:
        return jsonify({"error": "No features cached for this county."}), 404

    shap_data = []

    for model in _EXPLAINER_NAMES:
        # print(f"\n[DEBUG Outage_app.py] Attempting SHAP for model: {model} | Date: {date_str}")
        shap_data += [realtime_inference.compute_shap_for_fips(fips, explainer_name=model, date_str=date_str)]

    return jsonify({
        "fips":         fips,
        "date_str":     date_str,
        "occurrence":   pred["occurrence"],
        "occ_prob":     pred["occ_prob"],
        "scope":        pred["scope"],
        "duration":     pred["duration"],
        "anomaly_flag": pred.get("anomaly_flag", False),
        "ae_error":     pred.get("ae_error", 0.0),
        "weather":      features["weather"],
        "_occ_explainer":               shap_data[0],
        "_scope_explainer_small":       shap_data[1],
        "_scope_explainer_large":       shap_data[2],
        "_scope_explainer_classifier":  shap_data[3],
        "_duration_explainer_small":    shap_data[4],
        "_duration_explainer_large":    shap_data[5],
        "_duration_explainer_classifier": shap_data[6],
        "_scope_forecast_explainer_small": shap_data[7],
        "_scope_forecast_explainer_large": shap_data[8],
        "_scope_forecast_explainer_classifier": shap_data[9],
        "_duration_forecast_explainer_small": shap_data[10],
        "_duration_forecast_explainer_large": shap_data[11],
        "_duration_forecast_explainer_classifier": shap_data[12],
    })


@app.route("/api/forecast")
def forecast():
    """
    Returns 7-day county-level outage forecasts.
    Shape: { date_str: { fips: {occurrence, scope, duration, occ_prob} } }
    """
    data = realtime_inference.get_cached_forecast()
    if not data:
        return jsonify({"error": "Forecast not yet available. Try again shortly."}), 503
    return jsonify(data)


@app.route("/api/logs")
def logs():
    """Returns the last N server log lines."""
    with _log_lock:
        lines = list(_log_buffer)
    return jsonify({"lines": lines})


@app.route("/api/status")
def status():
    """Returns model load status and last inference timestamp."""
    return jsonify({
        "models_loaded": realtime_inference._occ_model is not None,
        "last_updated":  realtime_inference.get_last_updated(),
        "counties_in_cache": len(realtime_inference.get_cached_predictions()),
    })


# Entry point

if __name__ == "__main__":
    _startup()
    # use_reloader=False prevents the startup logic from running twice
    # in Flask's debug reloader subprocess
    app.run(debug=True, use_reloader=False)
