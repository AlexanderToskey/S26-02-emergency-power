from flask import Flask, render_template, jsonify
import json
import os
import threading
import time

import realtime_inference

app = Flask(__name__)

#  Startup: load models and run first inference

def _background_refresh(interval_seconds: int = 900):
    """
    Background thread: re-runs the full inference pipeline every `interval_seconds`
    (default 15 minutes) so predictions stay current without blocking API requests.
    """
    while True:
        try:
            realtime_inference.run_inference()
        except Exception as e:
            print(f"[app] Background inference error: {e}")
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

    # Background refresh thread (daemon=True so it dies with the server)
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    print("[app] Background refresh thread started (interval: 15 min).")


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
    pred = realtime_inference.get_cached_predictions().get(fips)
    if pred is None:
        return jsonify({"error": "No prediction cached for this county."}), 404

    features = realtime_inference.get_features_for_fips(fips)
    if not features:
        return jsonify({"error": "No features cached for this county."}), 404

    shap_data = realtime_inference.compute_shap_for_fips(fips)

    return jsonify({
        "fips":       fips,
        "occurrence": pred["occurrence"],
        "occ_prob":   pred["occ_prob"],
        "scope":      pred["scope"],
        "duration":   pred["duration"],
        "weather":    features["weather"],
        "shap":       shap_data,
    })


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
