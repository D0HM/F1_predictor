"""
Predicts each driver's final-lap time for a given race weekend.

Same methodology as the CRISP-DM notebook this was built from: train on every
lap EXCEPT each driver's final lap, then predict specifically that final lap
-- so the held-out test set is exactly the lap we care about, not a random
sample of laps.

Used by the race prediction app to forecast the selected event using the
previous year's equivalent race. Can also be run standalone:
    python src/lap_time_model.py --year 2025 --gp Monaco --session R
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FEATURE_COLS_NUM = [
    "TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "AirTemp", "TrackTemp", "Humidity", "WindSpeed",
]
FEATURE_COLS_CAT = ["Compound", "Team"]

CANDIDATE_MODELS = {
    "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42),
    "SVR": Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10))]),
}


def list_available_races():
    """Scans data/ for CSVs matching {year}_{gp}_{session}.csv and returns
    them as selectable options for the UI."""
    races = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        fname = os.path.basename(path)
        m = re.match(r"^(\d{4})_(.+)_(R|Q|FP1|FP2|FP3)\.csv$", fname)
        if m:
            year, gp, session = m.groups()
            races.append({
                "year": int(year),
                "gp": gp.replace("_", " "),
                "session": session,
                "filename": fname,
            })
    races.sort(key=lambda r: (r["year"], r["gp"]))
    return races


def filter_outliers(group):
    q1, q3 = group["LapTimeSeconds"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return group[(group["LapTimeSeconds"] >= lower) & (group["LapTimeSeconds"] <= upper)]


def predict_final_laps(year: int, gp: str, session: str) -> dict:
    """Loads the given race's collected CSV, trains on non-final laps, and
    predicts each driver's actual final lap. Returns a results dict ready
    to serialize as JSON."""
    csv_path = os.path.join(DATA_DIR, f"{year}_{gp.replace(' ', '_')}_{session}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No collected data for {gp} {year} ({session}). "
            f"Run: python src/collect_data.py --year {year} --gp \"{gp}\" --session {session}"
        )

    df = pd.read_csv(csv_path)
    df_clean = df.groupby("Driver", group_keys=False)[df.columns].apply(filter_outliers)

    last_lap_per_driver = df.groupby("Driver")["LapNumber"].max().rename("FinalLapNumber")
    df_clean = df_clean.merge(last_lap_per_driver, on="Driver", how="left")
    df_clean["IsLastLap"] = df_clean["LapNumber"] == df_clean["FinalLapNumber"]

    feature_cols_num = [c for c in FEATURE_COLS_NUM if c in df_clean.columns]
    feature_cols_cat = [c for c in FEATURE_COLS_CAT if c in df_clean.columns]
    model_df = df_clean.dropna(subset=feature_cols_num + feature_cols_cat + ["LapTimeSeconds"])

    X_num = model_df[feature_cols_num]
    X_cat = pd.get_dummies(model_df[feature_cols_cat], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    y = model_df["LapTimeSeconds"]

    is_last = model_df["IsLastLap"].values
    X_train, y_train = X[~is_last], y[~is_last]
    X_test, y_test = X[is_last], y[is_last]

    if len(X_train) < 20 or len(X_test) < 2:
        raise ValueError("Not enough laps in this race's data to train/evaluate a model.")

    cv_results = []
    for name, model in CANDIDATE_MODELS.items():
        scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train) // 5 or 1),
                                  scoring="neg_mean_absolute_error")
        cv_results.append((name, -scores.mean()))
    best_name = min(cv_results, key=lambda r: r[1])[0]
    best_model = CANDIDATE_MODELS[best_name]

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    report = model_df.loc[is_last, ["Driver", "Team", "Compound", "TyreLife"]].copy()
    report["actual_lap_time"] = y_test.values
    report["predicted_lap_time"] = preds
    report["error_seconds"] = report["predicted_lap_time"] - report["actual_lap_time"]
    report = report.sort_values("actual_lap_time").reset_index(drop=True)

    return {
        "gp": gp,
        "year": year,
        "session": session,
        "best_model": best_name,
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "r2": round(float(r2_score(y_test, preds)), 3),
        "drivers": [
            {
                "driver": row["Driver"],
                "team": row["Team"],
                "compound": row["Compound"],
                "tyre_life": int(row["TyreLife"]),
                "actual": round(float(row["actual_lap_time"]), 3),
                "predicted": round(float(row["predicted_lap_time"]), 3),
                "error": round(float(row["error_seconds"]), 3),
            }
            for _, row in report.iterrows()
        ],
    }


def forecast_previous_year_final_laps(year: int, gp: str) -> dict:
    """Return model-based final-lap forecasts from the previous year's equivalent race.

    This is used for a future-race prediction. The historical model is trained and
    evaluated on the previous year's race, then its per-driver estimates are used as
    the historical baseline for the selected race. The function deliberately does not
    require a current-race CSV because the selected race may not have happened yet.
    """
    previous_year = int(year) - 1
    return predict_final_laps(previous_year, gp, "R")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", type=str, required=True)
    parser.add_argument("--session", type=str, default="R")
    args = parser.parse_args()

    result = predict_final_laps(args.year, args.gp, args.session)
    print(f"Best model: {result['best_model']}  MAE={result['mae']}s  R2={result['r2']}")
    for d in result["drivers"]:
        print(f"  {d['driver']:4s} actual={d['actual']:7.3f}  predicted={d['predicted']:7.3f}  err={d['error']:+.3f}")
