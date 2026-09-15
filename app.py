"""Flask application for the 2026 F1 qualifying/race predictor."""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(BASE_DIR, "src")
if os.path.isdir(src_dir):
    sys.path.insert(0, src_dir)

import fastf1  # noqa: E402
import collect_data  # noqa: E402
import build_features  # noqa: E402

try:
    import lap_time_model  # noqa: E402
except ImportError:
    lap_time_model = None

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

quali_bundle = joblib.load(os.path.join(MODELS_DIR, "quali_top10_model.pkl"))
race_bundle = joblib.load(os.path.join(MODELS_DIR, "race_top10_model.pkl"))

# Official 2026 race-driver lineup.  This is deliberately separate from
# historical FastF1 results so a 2025-only driver cannot enter predictions.
CURRENT_2026_GRID = [
    ("NOR", "McLaren"), ("PIA", "McLaren"),
    ("RUS", "Mercedes"), ("ANT", "Mercedes"),
    ("LEC", "Ferrari"), ("HAM", "Ferrari"),
    ("VER", "Red Bull Racing"), ("HAD", "Red Bull Racing"),
    ("LAW", "Racing Bulls"), ("LIN", "Racing Bulls"),
    ("GAS", "Alpine"), ("COL", "Alpine"),
    ("OCO", "Haas F1 Team"), ("BEA", "Haas F1 Team"),
    ("HUL", "Audi"), ("BOR", "Audi"),
    ("SAI", "Williams"), ("ALB", "Williams"),
    ("ALO", "Aston Martin"), ("STR", "Aston Martin"),
    ("PER", "Cadillac"), ("BOT", "Cadillac"),
]
CURRENT_2026_DRIVERS = {driver for driver, _ in CURRENT_2026_GRID}


def get_2026_events():
    """Return all 2026 GP events exposed by FastF1."""
    collect_data.enable_cache()
    schedule = fastf1.get_event_schedule(2026)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()

    events = []
    for _, event in schedule.iterrows():
        events.append({
            "year": 2026,
            "round": int(event["RoundNumber"]),
            "gp": str(event["EventName"]),
        })
    return events


def get_event_round(year, gp):
    schedule = fastf1.get_event_schedule(year)
    matches = schedule[schedule["EventName"].astype(str).str.lower() == str(gp).lower()]
    if matches.empty:
        # Fallback for small naming differences such as GP suffixes.
        matches = schedule[schedule["EventName"].astype(str).str.contains(str(gp), case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Could not find {gp} in the {year} FastF1 schedule.")
    return int(matches.iloc[0]["RoundNumber"])


def get_form_before_race(year, gp):
    """Calculate form automatically for the selected race; no latest_form.csv needed."""
    if year != 2026:
        raise ValueError("This predictor is currently configured for 2026 races.")

    target_round = get_event_round(year, gp)
    previous_year = year - 1

    # 2025 provides the historical baseline.  2026 provides every completed
    # race before the selected round.  collect_season_results() caches both.
    collect_data.collect_season_results(previous_year)
    if target_round > 1:
        collect_data.collect_season_results(year, max_round=target_round - 1)

    years = [previous_year]
    current_path = os.path.join(DATA_DIR, f"{year}_season_results.csv")
    if os.path.exists(current_path):
        years.append(year)

    features_df = build_features.build_features(years)
    if features_df.empty:
        raise ValueError("Not enough historical results to build prediction features.")

    # Only information known before the selected race is allowed.
    prior = features_df[
        (features_df["Year"] < year) |
        ((features_df["Year"] == year) & (features_df["Round"] < target_round))
    ].copy()
    if prior.empty:
        raise ValueError("No historical results are available before this race.")

    # Latest driver form: prefer 2026-to-date, otherwise 2025.
    driver_latest = (
        prior.sort_values(["Year", "Round"])
        .groupby("Driver", as_index=False)
        .tail(1)
        [["Driver", "Team", "DriverAvgQuali_form", "DriverAvgRace_form"]]
        .copy()
    )

    # Latest team pace is also calculated from data available before the race.
    team_latest = (
        prior.sort_values(["Year", "Round"])
        .groupby("Team", as_index=False)
        .tail(1)
        [["Team", "TeamAvgRace_form"]]
        .copy()
    )

    # Fallbacks for rookies/new combinations.  They remain in the 2026 grid.
    global_q = float(prior["DriverAvgQuali_form"].median())
    global_r = float(prior["DriverAvgRace_form"].median())
    global_team = float(prior["TeamAvgRace_form"].median())

    driver_latest = driver_latest.drop_duplicates("Driver", keep="last").set_index("Driver")
    team_latest = team_latest.drop_duplicates("Team", keep="last").set_index("Team")

    snapshot = []
    for driver, team in CURRENT_2026_GRID:
        if driver in driver_latest.index:
            d = driver_latest.loc[driver]
            q_form = d["DriverAvgQuali_form"]
            r_form = d["DriverAvgRace_form"]
        else:
            q_form = np.nan
            r_form = np.nan

        team_form = team_latest.loc[team, "TeamAvgRace_form"] if team in team_latest.index else np.nan

        snapshot.append({
            "Driver": driver,
            "Team": team,
            "DriverAvgQuali_form": float(q_form) if pd.notna(q_form) else global_q,
            "DriverAvgRace_form": float(r_form) if pd.notna(r_form) else global_r,
            "TeamAvgRace_form": float(team_form) if pd.notna(team_form) else global_team,
        })

    return pd.DataFrame(snapshot), previous_year, target_round


def get_final_lap_estimates(previous_year, gp, race_ranked):
    """Return a final-lap forecast for every predicted 2026 race driver.

    The app automatically collects the previous year's equivalent race if that
    lap-level CSV is not already present. The existing final-lap ML routine then
    trains/evaluates on that historical race and supplies a baseline estimate.
    Current 2026 drivers are mapped by driver first, then by normalized team,
    then by the historical global median.
    """
    if lap_time_model is None:
        raise RuntimeError("lap_time_model.py is required for final-lap forecasting.")

    csv_path = os.path.join(
        DATA_DIR, f"{previous_year}_{gp.replace(' ', '_')}_R.csv"
    )

    # Automatically collect the previous year's equivalent race when needed.
    if not os.path.exists(csv_path):
        app.logger.info("Collecting %s %s race telemetry for final-lap forecast", previous_year, gp)
        df = collect_data.collect(previous_year, gp, "R")
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(csv_path, index=False)

    historical = lap_time_model.predict_final_laps(previous_year, gp, "R")
    rows = historical.get("drivers", [])
    if not rows:
        raise ValueError(f"No historical final-lap estimates available for {gp} {previous_year}.")

    def norm_team(team):
        t = str(team).strip().lower()
        aliases = {
            "rb": "racing bulls",
            "racing bulls": "racing bulls",
            "kick sauber": "audi",
            "sauber": "audi",
            "audi": "audi",
            "haas": "haas",
            "haas f1 team": "haas",
            "red bull racing": "red bull racing",
            "red bull": "red bull racing",
        }
        return aliases.get(t, t)

    by_driver = {str(r["driver"]): float(r["predicted"]) for r in rows}
    team_values = {}
    for r in rows:
        team_values.setdefault(norm_team(r["team"]), []).append(float(r["predicted"]))
    team_medians = {k: float(np.median(v)) for k, v in team_values.items() if v}
    global_value = float(np.median(list(by_driver.values())))

    estimates = {}
    for r in race_ranked:
        driver = str(r["driver"])
        team = norm_team(r["team"])
        if driver in by_driver:
            estimates[driver] = by_driver[driver]
        elif team in team_medians:
            estimates[driver] = team_medians[team]
        else:
            estimates[driver] = global_value

    return estimates


def make_prediction(year: int, gp: str):
    """Predict 2026 qualifying, race order, and final-lap time."""
    form, previous_year, target_round = get_form_before_race(year, gp)

    quali_model = quali_bundle["model"]
    race_model = race_bundle["model"]
    quali_feats = quali_bundle["features"]
    race_feats = race_bundle["features"]

    results = []
    for _, row in form.iterrows():
        qx = pd.DataFrame([{f: row[f] for f in quali_feats}])
        quali_prob = float(quali_model.predict_proba(qx)[0, 1])
        results.append({
            "driver": row["Driver"],
            "team": row["Team"],
            "quali_top10_prob": quali_prob,
            "DriverAvgQuali_form": float(row["DriverAvgQuali_form"]),
            "DriverAvgRace_form": float(row["DriverAvgRace_form"]),
            "TeamAvgRace_form": float(row["TeamAvgRace_form"]),
        })

    results.sort(key=lambda r: (-r["quali_top10_prob"], r["DriverAvgQuali_form"], r["driver"]))
    for pos, result in enumerate(results, start=1):
        result["predicted_grid"] = pos

    for result in results:
        rx = pd.DataFrame([{
            "DriverAvgRace_form": result["DriverAvgRace_form"],
            "TeamAvgRace_form": result["TeamAvgRace_form"],
            "GridPosition": result["predicted_grid"],
        }])
        race_prob = float(race_model.predict_proba(rx)[0, 1])
        result["race_top10_prob"] = race_prob

    quali_ranked = sorted(results, key=lambda r: (-r["quali_top10_prob"], r["DriverAvgQuali_form"], r["driver"]))
    race_ranked = sorted(results, key=lambda r: (-r["race_top10_prob"], r["predicted_grid"], r["driver"]))
    final_laps = get_final_lap_estimates(previous_year, gp, race_ranked)

    quali_output = []
    for pos, r in enumerate(quali_ranked, 1):
        quali_output.append({
            "position": pos,
            "driver": r["driver"],
            "team": r["team"],
            "quali_top10_prob": round(r["quali_top10_prob"], 4),
        })

    race_output = []
    for pos, r in enumerate(race_ranked, 1):
        final_lap = final_laps.get(r["driver"])
        race_output.append({
            "position": pos,
            "driver": r["driver"],
            "team": r["team"],
            "race_top10_prob": round(r["race_top10_prob"], 4),
            "final_lap_time": round(final_lap, 3) if final_lap is not None and pd.notna(final_lap) else None,
        })

    return {
        "year": year,
        "gp": gp,
        "historical_year": previous_year,
        "target_round": target_round,
        "drivers_used": len(CURRENT_2026_GRID),
        "quali": quali_output,
        "race": race_output,
    }


@app.route("/")
def index():
    try:
        events = get_2026_events()
        schedule_error = None
    except Exception as e:
        events = []
        schedule_error = str(e)
    return render_template("index.html", events=events, schedule_error=schedule_error)


@app.route("/events/2026")
def events_2026():
    try:
        return jsonify(get_2026_events())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    year = int(payload.get("year", 2026))
    gp = payload.get("gp")
    if not gp:
        return jsonify({"error": "Please select a 2026 race."}), 400
    try:
        return jsonify(make_prediction(year, gp))
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
