"""FastF1 data collection for the 2026 F1 predictor.

This module provides:
- full-season result collection for model features;
- lap-level collection for the previous-year equivalent race, used to
  estimate a selected 2026 race's final-lap time.

All FastF1 downloads are cached locally.
"""
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


def _safe_float(value):
    try:
        return float(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def collect(year: int, gp: str, session_type: str, cache_dir: str = None) -> pd.DataFrame:
    """Collect lap-level data for one race/session."""
    enable_cache(cache_dir)
    os.makedirs(DATA_DIR, exist_ok=True)

    out_path = os.path.join(DATA_DIR, f"{year}_{gp.replace(' ', '_')}_{session_type}.csv")
    if os.path.exists(out_path):
        return pd.read_csv(out_path)

    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps = session.laps.copy()

    weather = session.weather_data
    if weather is not None and not weather.empty and "Time" in laps.columns and "Time" in weather.columns:
        laps = pd.merge_asof(
            laps.sort_values("Time"),
            weather.sort_values("Time"),
            on="Time",
            direction="nearest",
        )

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col + "Seconds"] = laps[col].dt.total_seconds()

    keep_cols = [
        "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife",
        "LapTimeSeconds", "Sector1TimeSeconds", "Sector2TimeSeconds", "Sector3TimeSeconds",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TrackStatus", "IsPersonalBest", "AirTemp", "TrackTemp", "Humidity",
        "Rainfall", "WindSpeed", "Position",
    ]
    keep_cols = [c for c in keep_cols if c in laps.columns]
    clean = laps[keep_cols].dropna(subset=["LapTimeSeconds"]).copy()
    clean.insert(0, "GrandPrix", gp)
    clean.insert(0, "Year", year)
    clean.to_csv(out_path, index=False)
    return clean


def collect_season_results(year: int, cache_dir: str = None, force: bool = False) -> pd.DataFrame:
    """Collect qualifying and race results for all available rounds."""
    enable_cache(cache_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{year}_season_results.csv")

    if os.path.exists(out_path) and not force:
        return pd.read_csv(out_path)

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    rows = []

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        quali_pos = {}

        try:
            q = fastf1.get_session(year, round_num, "Q")
            q.load(laps=False, telemetry=False, weather=False, messages=False)
            for _, r in q.results.iterrows():
                drv = r.get("Abbreviation")
                pos = r.get("Position")
                if pd.notna(drv) and pd.notna(pos):
                    quali_pos[str(drv)] = _safe_float(pos)
        except Exception as exc:
            print(f"[Q unavailable] {year} R{round_num} {event_name}: {exc}")

        try:
            race = fastf1.get_session(year, round_num, "R")
            race.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"[R unavailable] {year} R{round_num} {event_name}: {exc}")
            continue

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
                "GridPosition": _safe_float(r.get("GridPosition")),
                "QualiPosition": quali_pos.get(str(drv)),
                "RacePosition": _safe_float(r.get("Position")),
                "Points": _safe_float(r.get("Points")),
                "Status": r.get("Status"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"FastF1 returned no completed race results for {year}.")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} driver-race rows to {out_path}")
    return df


def collect_previous_year_race(year: int, gp: str, cache_dir: str = None) -> pd.DataFrame:
    """Collect the previous year's equivalent race for final-lap forecasting.

    For example, selecting Singapore 2026 automatically downloads Singapore
    2025 race laps. FastF1 handles the event-name lookup.
    """
    if year <= 1:
        raise ValueError("Invalid target year")
    previous_year = year - 1
    return collect(previous_year, gp, "R", cache_dir=cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--gp", type=str)
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--season-results", action="store_true")
    parser.add_argument("--previous-year-race", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.season_results:
        collect_season_results(args.year, force=args.force)
    elif args.previous_year_race:
        if not args.gp:
            parser.error("--gp is required with --previous-year-race")
        df = collect_previous_year_race(args.year, args.gp)
        print(f"Collected {len(df)} laps from {args.year - 1} {args.gp} R")
    else:
        if not args.gp:
            parser.error("--gp is required")
        df = collect(args.year, args.gp, args.session)
        print(f"Saved {len(df)} laps")
