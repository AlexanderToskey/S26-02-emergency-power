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
    # TODO: Replace with model outputs and real-time integration
    # Example: counties with True have a prediction and counties with False don't
    data = {
        "51107": { "occurrence": True, "scope": 5600, "duration": 4.5 },
        "51121": { "occurrence": True, "scope": 1200, "duration": 1.5 },

        "51013": { "occurrence": True,  "scope": 7342, "duration": 12.7 },
        "51041": { "occurrence": True,  "scope": 1985, "duration": 37.4 },
        "51047": { "occurrence": True,  "scope": 8421, "duration": 5.2 },
        "51059": { "occurrence": True,  "scope": 6523, "duration": 16.8 },
        "51061": { "occurrence": True,  "scope": 412,  "duration": 2.6 },
        "51069": { "occurrence": True,  "scope": 9055, "duration": 54.3 },
        "51087": { "occurrence": True,  "scope": 7210, "duration": 23.1 },
        "51099": { "occurrence": True,  "scope": 3567, "duration": 8.9 },
        "51153": { "occurrence": True,  "scope": 918,  "duration": 3.7 },
        "51157": { "occurrence": True,  "scope": 6475, "duration": 42.5 },
        "51177": { "occurrence": True,  "scope": 2740, "duration": 9.4 },
        "51179": { "occurrence": True,  "scope": 7833, "duration": 65.2 },
        "51187": { "occurrence": False,  "scope": 0, "duration": 0 },
        "51199": { "occurrence": False,  "scope": 0, "duration": 0 },
        "51710": { "occurrence": False,  "scope": 0, "duration": 0 },
        "51810": { "occurrence": False,  "scope": 0, "duration": 0 },
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)