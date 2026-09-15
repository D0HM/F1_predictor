"""
Data collection: pulls lap-level data from FastF1 for a given race weekend
and saves a clean CSV for modeling.

Run locally (this needs internet access to F1's timing/Ergast servers):
    python src/collect_data.py --year 2024 --gp "Bahrain" --session R
"""
import argparse
import os
import fastf1
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect(year: int, gp: str, session_type: str, cache_dir: str = None) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = os.path.join(BASE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, session_type)
    session.load()  # loads laps, telemetry, weather, results

    laps = session.laps.copy()

    # Merge in weather (nearest-time join)
    weather = session.weather_data
    if weather is not None and not weather.empty:
        laps = pd.merge_asof(
            laps.sort_values("Time"),
            weather.sort_values("Time"),
            on="Time",
            direction="nearest",
        )

    # Basic lap-level features
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Sector1Seconds"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Seconds"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Seconds"] = laps["Sector3Time"].dt.total_seconds()

    keep_cols = [
        "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife",
        "LapTimeSeconds", "Sector1Seconds", "Sector2Seconds", "Sector3Seconds",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TrackStatus", "IsPersonalBest",
        "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed",
        "Position",
    ]
    keep_cols = [c for c in keep_cols if c in laps.columns]
    clean = laps[keep_cols].dropna(subset=["LapTimeSeconds"])

    # Tag which race this data came from, so files can be told apart if you
    # ever concatenate multiple races together.
    clean.insert(0, "GrandPrix", gp)
    clean.insert(0, "Year", year)

    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--gp", type=str, default="Monaco")
    parser.add_argument("--session", type=str, default="R")  # R=Race, Q=Quali, FP1/2/3
    args = parser.parse_args()

    df = collect(args.year, args.gp, args.session)
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{args.year}_{args.gp.replace(' ', '_')}_{args.session}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} laps to {out_path}")
    print(df.head())
