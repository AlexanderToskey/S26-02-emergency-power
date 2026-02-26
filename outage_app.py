from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

# Serve the main map page
@app.route("/")
def index():
    return render_template("index.html")

# Serve the counties GeoJSON
@app.route("/api/counties")
def counties():
    filepath = os.path.join(os.path.dirname(__file__), "static", "counties.json")
    with open(filepath) as f:
        data = json.load(f)
    return jsonify(data)

# Serve example outage predictions
@app.route("/api/predictions")
def predictions():
    # Example: only two counties have predictions
    data = {
        "51107": { "occurrence": True, "scope": 5600, "duration": 4.5 },
        "51121": { "occurrence": True, "scope": 1200, "duration": 1.5 }
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)