"""FastF1 data collection for the F1 predictor."""
import argparse
import os
from typing import Optional

import fastf1
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")


def enable_cache(cache_dir: Optional[str] = None):
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def collect(year: int, gp: str, session_type: str, cache_dir: str = None) -> pd.DataFrame:
    """Collect lap-level data for one race weekend."""
    enable_cache(cache_dir)
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps = session.laps.copy()

    weather = session.weather_data
    if weather is not None and not weather.empty and "Time" in laps and "Time" in weather:
        laps = pd.merge_asof(
            laps.sort_values("Time"), weather.sort_values("Time"),
            on="Time", direction="nearest",
        )

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col.replace("Time", "Seconds")] = laps[col].dt.total_seconds()

    keep_cols = [
        "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife",
        "LapTimeSeconds", "Sector1Seconds", "Sector2Seconds", "Sector3Seconds",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TrackStatus", "IsPersonalBest", "AirTemp", "TrackTemp",
        "Humidity", "Rainfall", "WindSpeed", "Position",
    ]
    keep_cols = [c for c in keep_cols if c in laps.columns]
    clean = laps[keep_cols].dropna(subset=["LapTimeSeconds"])
    clean.insert(0, "GrandPrix", gp)
    clean.insert(0, "Year", year)
    return clean


def _collect_round(year, round_num):
    """Collect one completed race round as driver-level results."""
    event = fastf1.get_event_schedule(year)
    event = event[event["RoundNumber"] == round_num]
    if event.empty:
        return []
    event_name = str(event.iloc[0]["EventName"])

    quali_pos = {}
    try:
        q = fastf1.get_session(year, round_num, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        for _, r in q.results.iterrows():
            driver = r.get("Abbreviation")
            pos = r.get("Position")
            if pd.notna(driver) and pd.notna(pos):
                quali_pos[str(driver)] = float(pos)
    except Exception as e:
        print(f"    [skip Q] {event_name}: {e}")

    try:
        race = fastf1.get_session(year, round_num, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"    [skip R] {event_name}: {e}")
        return []

    rows = []
    for _, r in race.results.iterrows():
        drv = r.get("Abbreviation")
        if pd.isna(drv):
            continue
        rows.append({
            "Year": year,
            "Round": round_num,
            "EventName": event_name,
            "Driver": str(drv),
            "Team": r.get("TeamName"),
            "GridPosition": r.get("GridPosition"),
            "QualiPosition": quali_pos.get(str(drv)),
            "RacePosition": r.get("Position"),
            "Points": r.get("Points"),
            "Status": r.get("Status"),
        })
    return rows


def collect_season_results(year: int, cache_dir: str = None, force: bool = False,
                           max_round: Optional[int] = None) -> pd.DataFrame:
    """Collect available season results, optionally only through max_round.

    Existing CSV data is reused. Missing rounds are appended automatically, so
    the web app can update 2026-to-date form without a manually maintained
    latest_form.csv.
    """
    enable_cache(cache_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{year}_season_results.csv")

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    available_rounds = [int(x) for x in schedule["RoundNumber"].tolist()]
    if max_round is not None:
        available_rounds = [r for r in available_rounds if r <= int(max_round)]

    if os.path.exists(out_path) and not force:
        existing = pd.read_csv(out_path)
        existing_rounds = set(existing["Round"].dropna().astype(int).unique()) if not existing.empty else set()
    else:
        existing = pd.DataFrame()
        existing_rounds = set()

    rows = []
    for round_num in available_rounds:
        if round_num in existing_rounds:
            continue
        print(f"  Loading {year} round {round_num}")
        rows.extend(_collect_round(year, round_num))

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing.copy()

    if combined.empty:
        raise ValueError(f"No completed race results are available for {year}.")

    combined = combined.drop_duplicates(subset=["Year", "Round", "Driver"], keep="last")
    combined = combined.sort_values(["Year", "Round", "Driver"]).reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} driver-race rows to {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--gp", type=str, default=None)
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--season-results", action="store_true")
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.season_results:
        collect_season_results(args.year, max_round=args.max_round, force=args.force)
    else:
        if not args.gp:
            parser.error("--gp is required unless --season-results is used")
        df = collect(args.year, args.gp, args.session)
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, f"{args.year}_{args.gp.replace(' ', '_')}_{args.session}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} laps to {out_path}")
