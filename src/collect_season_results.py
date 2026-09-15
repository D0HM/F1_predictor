"""
Collects driver-level quali + race results for every round in a season.
Produces one row per driver per race weekend, which build_features.py
turns into rolling-form features.

Run locally (needs internet access to F1 timing/Ergast servers):
    python src/collect_season_results.py --year 2024
    python src/collect_season_results.py --year 2025
"""
import argparse
import os
import fastf1
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_season(year: int, cache_dir: str = None) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = os.path.join(BASE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year)
    # Only completed conventional race weekends (skip testing rounds etc.)
    schedule = schedule[schedule["RoundNumber"] > 0]

    rows = []
    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        # --- Qualifying ---
        quali_pos = {}
        try:
            q = fastf1.get_session(year, round_num, "Q")
            q.load(laps=False, telemetry=False, weather=False, messages=False)
            for _, r in q.results.iterrows():
                quali_pos[r["Abbreviation"]] = r["Position"]
        except Exception as e:
            print(f"  [skip Q] {event_name}: {e}")

        # --- Race ---
        try:
            race = fastf1.get_session(year, round_num, "R")
            race.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"  [skip R] {event_name}: {e}")
            continue

        for _, r in race.results.iterrows():
            drv = r["Abbreviation"]
            rows.append({
                "Year": year,
                "Round": round_num,
                "EventName": event_name,
                "Driver": drv,
                "Team": r.get("TeamName"),
                "GridPosition": r.get("GridPosition"),
                "QualiPosition": quali_pos.get(drv),
                "RacePosition": r.get("Position"),
                "Points": r.get("Points"),
                "Status": r.get("Status"),
            })
        print(f"  loaded round {round_num}: {event_name}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    df = collect_season(args.year)
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{args.year}_season_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} driver-race rows to {out_path}")
