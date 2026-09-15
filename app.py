"""Flask application for the 2026 F1 qualifying/race predictor."""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if os.path.isdir(SRC_DIR):
    sys.path.insert(0, SRC_DIR)
else:
    sys.path.insert(0, BASE_DIR)

import fastf1  # noqa: E402
import collect_data  # noqa: E402
import build_features  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(MODELS_DIR, "quali_top10_model.pkl"), "rb") as f:
    quali_bundle = joblib.load(f)
with open(os.path.join(MODELS_DIR, "race_top10_model.pkl"), "rb") as f:
    race_bundle = joblib.load(f)

# Official 2026 grid. This is deliberately explicit so 2025-only drivers such
# as Tsunoda and Doohan cannot leak into the prediction population.
DRIVERS_2026 = [
    {"driver": "RUS", "team": "Mercedes"},
    {"driver": "ANT", "team": "Mercedes"},
    {"driver": "LEC", "team": "Ferrari"},
    {"driver": "HAM", "team": "Ferrari"},
    {"driver": "NOR", "team": "McLaren"},
    {"driver": "PIA", "team": "McLaren"},
    {"driver": "VER", "team": "Red Bull Racing"},
    {"driver": "HAD", "team": "Red Bull Racing"},
    {"driver": "LAW", "team": "Racing Bulls"},
    {"driver": "LIN", "team": "Racing Bulls"},
    {"driver": "GAS", "team": "Alpine"},
    {"driver": "COL", "team": "Alpine"},
    {"driver": "OCO", "team": "Haas F1 Team"},
    {"driver": "BEA", "team": "Haas F1 Team"},
    {"driver": "HUL", "team": "Audi"},
    {"driver": "BOR", "team": "Audi"},
    {"driver": "SAI", "team": "Williams"},
    {"driver": "ALB", "team": "Williams"},
    {"driver": "ALO", "team": "Aston Martin"},
    {"driver": "STR", "team": "Aston Martin"},
    {"driver": "PER", "team": "Cadillac"},
    {"driver": "BOT", "team": "Cadillac"},
]


def get_2026_events():
    collect_data.enable_cache()
    schedule = fastf1.get_event_schedule(2026)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    return [
        {"year": 2026, "round": int(e["RoundNumber"]), "gp": str(e["EventName"])}
        for _, e in schedule.iterrows()
    ]


def _season_history_before_target(year, target_gp):
    """Return 2025 + completed 2026 history before target round."""
    schedule = fastf1.get_event_schedule(year)
    target = schedule[schedule["EventName"].astype(str) == str(target_gp)]
    if target.empty:
        raise ValueError(f"2026 race '{target_gp}' was not found in the FastF1 schedule.")
    target_round = int(target.iloc[0]["RoundNumber"])

    collect_data.collect_season_results(2025)
    current = collect_data.collect_season_results(2026)
    current = current[current["Round"] < target_round].copy()

    frames = [pd.read_csv(os.path.join(DATA_DIR, "2025_season_results.csv"))]
    if not current.empty:
        frames.append(current)
    history = pd.concat(frames, ignore_index=True)
    return history, target_round


def _build_features_from_history(history):
    """Same feature engineering as build_features.py, without writing files."""
    df = history.copy().sort_values(["Year", "Round"]).reset_index(drop=True)
    df["QualiTop10"] = (df["QualiPosition"] <= 10).astype(int)
    df["RaceTop10"] = (df["RacePosition"] <= 10).astype(int)
    df["DriverAvgQuali_form"] = df.groupby("Driver")["QualiPosition"].transform(
        lambda s: s.shift(1).rolling(build_features.ROLLING_WINDOW, min_periods=1).mean()
    )
    df["DriverAvgRace_form"] = df.groupby("Driver")["RacePosition"].transform(
        lambda s: s.shift(1).rolling(build_features.ROLLING_WINDOW, min_periods=1).mean()
    )
    team = df.groupby(["Year", "Round", "Team"])["RacePosition"].mean().reset_index()
    team = team.sort_values(["Team", "Year", "Round"])
    team["TeamAvgRace_form"] = team.groupby("Team")["RacePosition"].transform(
        lambda s: s.shift(1).rolling(build_features.ROLLING_WINDOW, min_periods=1).mean()
    )
    return df.merge(team[["Year", "Round", "Team", "TeamAvgRace_form"]],
                    on=["Year", "Round", "Team"], how="left")


def _latest_forms(history):
    features = _build_features_from_history(history)
    valid = features.dropna(subset=[
        "DriverAvgQuali_form", "DriverAvgRace_form", "TeamAvgRace_form"
    ])
    latest_driver = (
        valid.sort_values(["Driver", "Year", "Round"])
        .groupby("Driver", as_index=False).tail(1)
        [["Driver", "Team", "DriverAvgQuali_form", "DriverAvgRace_form", "TeamAvgRace_form"]]
    )
    latest_team = (
        valid.sort_values(["Team", "Year", "Round"])
        .groupby("Team", as_index=False).tail(1)
        [["Team", "TeamAvgRace_form"]]
    )
    return latest_driver, latest_team


def _make_2026_feature_rows(history):
    latest_driver, latest_team = _latest_forms(history)
    overall = {
        "DriverAvgQuali_form": float(latest_driver["DriverAvgQuali_form"].median()),
        "DriverAvgRace_form": float(latest_driver["DriverAvgRace_form"].median()),
        "TeamAvgRace_form": float(latest_driver["TeamAvgRace_form"].median()),
    }
    rows = []
    for item in DRIVERS_2026:
        d = latest_driver[latest_driver["Driver"] == item["driver"]]
        t = latest_team[latest_team["Team"] == item["team"]]
        # Prefer the driver's most recent 2026/2025 form. For a rookie/new
        # driver, use their current team's latest historical pace, then the
        # overall median as a final fallback.
        if not d.empty:
            r = d.iloc[-1]
            qf, rf, tf = float(r.DriverAvgQuali_form), float(r.DriverAvgRace_form), float(r.TeamAvgRace_form)
        else:
            team_form = float(t.iloc[-1].TeamAvgRace_form) if not t.empty else overall["TeamAvgRace_form"]
            qf = overall["DriverAvgQuali_form"]
            rf = team_form
            tf = team_form
        rows.append({"driver": item["driver"], "team": item["team"],
                     "DriverAvgQuali_form": qf, "DriverAvgRace_form": rf,
                     "TeamAvgRace_form": tf})
    return rows


def _final_lap_forecast(target_year, gp, race_ranked):
    """Estimate 2026 final-lap times using the previous year's same GP.

    The selected 2026 race's actual telemetry is not available before the race,
    so this is a forecast, not the retrospective held-out final-lap evaluation.
    A regressor is trained on all 2025 laps from the same GP. Each 2026 driver
    gets a 2025 final-lap feature template when available; rookies fall back to
    team/global templates. Predicted race position is used as the position input.
    """
    try:
        hist = collect_data.collect_previous_year_race(target_year, gp)
    except Exception as exc:
        return {"available": False, "message": f"Final-lap forecast unavailable: {exc}", "drivers": []}

    if hist.empty or "LapTimeSeconds" not in hist.columns:
        return {"available": False, "message": "No previous-year lap data available.", "drivers": []}

    df = hist.copy()
    num_cols = [c for c in ["TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                            "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Position"] if c in df.columns]
    cat_cols = [c for c in ["Compound", "Team", "Driver"] if c in df.columns]
    usable = df.dropna(subset=["LapTimeSeconds"] + num_cols + cat_cols).copy()
    if len(usable) < 30:
        return {"available": False, "message": "Not enough previous-year telemetry for a final-lap forecast.", "drivers": []}

    # Keep the same model family used by the original Final Lap feature.
    from sklearn.ensemble import RandomForestRegressor
    X_num = usable[num_cols].astype(float)
    X_cat = pd.get_dummies(usable[cat_cols].astype(str), drop_first=False)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = usable["LapTimeSeconds"].astype(float).reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X, y)

    final_rows = usable.sort_values("LapNumber").groupby("Driver", as_index=False).tail(1)
    global_template = usable[num_cols].median(numeric_only=True)
    team_templates = usable.groupby("Team")[num_cols].median(numeric_only=True)

    predictions = []
    for race_row in race_ranked:
        driver = race_row["driver"]
        team = race_row["team"]
        same = final_rows[final_rows["Driver"] == driver]
        if not same.empty:
            base = same.iloc[-1].copy()
        else:
            same_team = final_rows[final_rows["Team"] == team]
            base = same_team.iloc[0].copy() if not same_team.empty else None
            if base is None:
                base = pd.Series({c: global_template.get(c, np.nan) for c in num_cols})
                base["Compound"] = "MEDIUM"
                base["Team"] = team
                base["Driver"] = driver

        values = {}
        for c in num_cols:
            values[c] = float(base[c]) if pd.notna(base.get(c)) else float(global_template.get(c, 0))
        values["Position"] = float(race_row["position"])
        row_df = pd.DataFrame([values])
        for c in cat_cols:
            row_df[c] = str(base.get(c, team if c == "Team" else driver))
        row_cat = pd.get_dummies(row_df[cat_cols].astype(str), drop_first=False)
        row_X = pd.concat([row_df[num_cols].astype(float), row_cat], axis=1)
        row_X = row_X.reindex(columns=X.columns, fill_value=0)
        pred = float(model.predict(row_X)[0])
        predictions.append({"driver": driver, "predicted_final_lap": round(pred, 3)})

    return {
        "available": True,
        "historical_year": target_year - 1,
        "gp": gp,
        "drivers": predictions,
    }


def make_prediction(year: int, gp: str):
    if year != 2026:
        raise ValueError("This predictor is configured for the 2026 season.")

    history, target_round = _season_history_before_target(year, gp)
    feature_rows = _make_2026_feature_rows(history)
    quali_model, race_model = quali_bundle["model"], race_bundle["model"]
    quali_feats, race_feats = quali_bundle["features"], race_bundle["features"]

    results = []
    for row in feature_rows:
        qx = pd.DataFrame([{f: row[f] for f in quali_feats}])
        qprob = float(quali_model.predict_proba(qx)[0, 1])
        results.append({**row, "quali_top10_prob": qprob})

    # Qualifying ranking is generated only from the 2026 roster.
    results.sort(key=lambda r: (-r["quali_top10_prob"], r["DriverAvgQuali_form"], r["driver"]))
    for pos, r in enumerate(results, 1):
        r["predicted_grid"] = pos

    for r in results:
        rx = pd.DataFrame([{
            "DriverAvgRace_form": r["DriverAvgRace_form"],
            "TeamAvgRace_form": r["TeamAvgRace_form"],
            "GridPosition": r["predicted_grid"],
        }])
        r["race_top10_prob"] = float(race_model.predict_proba(rx)[0, 1])

    quali_ranked = sorted(results, key=lambda r: (-r["quali_top10_prob"], r["DriverAvgQuali_form"], r["driver"]))
    race_ranked = sorted(results, key=lambda r: (-r["race_top10_prob"], r["predicted_grid"], r["driver"]))

    race_output = []
    for pos, r in enumerate(race_ranked, 1):
        race_output.append({
            "position": pos, "driver": r["driver"], "team": r["team"],
            "race_top10_prob": round(r["race_top10_prob"], 4),
        })

    final_lap = _final_lap_forecast(year, gp, race_output)
    final_by_driver = {x["driver"]: x["predicted_final_lap"] for x in final_lap.get("drivers", [])}
    for row in race_output:
        row["predicted_final_lap"] = final_by_driver.get(row["driver"])

    quali_output = [{
        "position": i, "driver": r["driver"], "team": r["team"],
        "quali_top10_prob": round(r["quali_top10_prob"], 4),
    } for i, r in enumerate(quali_ranked, 1)]

    return {
        "year": year, "gp": gp, "historical_year": 2025,
        "target_round": target_round,
        "quali": quali_output,
        "race": race_output,
        "final_lap": final_lap,
    }


@app.route("/")
def index():
    try:
        events = get_2026_events()
        schedule_error = None
    except Exception as exc:
        events, schedule_error = [], str(exc)
    return render_template("index.html", events=events, schedule_error=schedule_error)


@app.route("/events/2026")
def events_2026():
    try:
        return jsonify(get_2026_events())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    year = int(payload.get("year", 2026))
    gp = payload.get("gp")
    if not gp:
        return jsonify({"error": "Please select a 2026 race."}), 400
    try:
        return jsonify(make_prediction(year, gp))
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
