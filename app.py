"""
Flask app: predicts Top 10 in Qualifying and Race for Singapore GP (or any
event) using the trained models + each driver's latest rolling form.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import sys
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Anchor paths to this file's own folder, so `python app.py` works no matter
# what directory you launch it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import lap_time_model  # noqa: E402  (needs sys.path set up first)

quali_bundle = joblib.load(os.path.join(BASE_DIR, "models", "quali_top10_model.pkl"))
race_bundle = joblib.load(os.path.join(BASE_DIR, "models", "race_top10_model.pkl"))
latest_form = pd.read_csv(os.path.join(BASE_DIR, "data", "latest_form.csv"))


@app.route("/")
def index():
    drivers = latest_form.to_dict(orient="records")
    races = lap_time_model.list_available_races()
    return render_template("index.html", drivers=drivers, races=races)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    grid_positions = payload.get("grid_positions", {})  # {driver: assumed grid pos}

    quali_model, quali_feats = quali_bundle["model"], quali_bundle["features"]
    race_model, race_feats = race_bundle["model"], race_bundle["features"]

    results = []
    for _, row in latest_form.iterrows():
        driver = row["Driver"]

        # Quali prediction
        qx = pd.DataFrame([{f: row[f] for f in quali_feats}])
        quali_prob = quali_model.predict_proba(qx)[0, 1]

        # Race prediction — uses the user-editable assumed grid position
        assumed_grid = grid_positions.get(driver, row["DriverAvgRace_form"])
        rx = pd.DataFrame([{
            "DriverAvgRace_form": row["DriverAvgRace_form"],
            "TeamAvgRace_form": row["TeamAvgRace_form"],
            "GridPosition": assumed_grid,
        }])
        race_prob = race_model.predict_proba(rx)[0, 1]

        results.append({
            "driver": driver,
            "team": row["Team"],
            "quali_top10_prob": round(float(quali_prob), 3),
            "race_top10_prob": round(float(race_prob), 3),
        })

    quali_ranked = sorted(results, key=lambda r: -r["quali_top10_prob"])
    race_ranked = sorted(results, key=lambda r: -r["race_top10_prob"])

    return jsonify({
        "quali": quali_ranked,
        "race": race_ranked,
    })


@app.route("/races")
def races():
    return jsonify(lap_time_model.list_available_races())


@app.route("/predict_lastlap", methods=["POST"])
def predict_lastlap():
    payload = request.get_json()
    year = int(payload.get("year"))
    gp = payload.get("gp")
    session = payload.get("session", "R")

    try:
        result = lap_time_model.predict_final_laps(year, gp, session)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
