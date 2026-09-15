"""
Builds a driver-race level feature table from one or more season_results CSVs:
- Rolling form (avg quali/race position over last N races) computed BEFORE
  each race, so there's no leakage from the race being predicted.
- Team pace (rolling avg team finishing position).
- Binary labels: QualiTop10, RaceTop10.

Run:
    python src/build_features.py --years 2023 2024 2025
"""
import argparse
import glob
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ROLLING_WINDOW = 4


def build_features(years):
    frames = []
    for y in years:
        for path in glob.glob(os.path.join(DATA_DIR, f"{y}_season_results.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No season_results CSVs found for given years")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    # Labels
    df["QualiTop10"] = (df["QualiPosition"] <= 10).astype(int)
    df["RaceTop10"] = (df["RacePosition"] <= 10).astype(int)

    # Rolling driver form — shifted so the current race isn't included
    df["DriverAvgQuali_form"] = (
        df.groupby("Driver")["QualiPosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df["DriverAvgRace_form"] = (
        df.groupby("Driver")["RacePosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    # Rolling team pace (avg race finishing position across both cars)
    team_race = df.groupby(["Year", "Round", "Team"])["RacePosition"].mean().reset_index()
    team_race = team_race.sort_values(["Team", "Year", "Round"])
    team_race["TeamAvgRace_form"] = (
        team_race.groupby("Team")["RacePosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df = df.merge(
        team_race[["Year", "Round", "Team", "TeamAvgRace_form"]],
        on=["Year", "Round", "Team"], how="left",
    )

    # Drop early-season rows with no history yet
    features_df = df.dropna(
        subset=["DriverAvgQuali_form", "DriverAvgRace_form", "TeamAvgRace_form"]
    )

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()

    features_df = build_features(args.years)
    os.makedirs(DATA_DIR, exist_ok=True)
    features_df.to_csv(os.path.join(DATA_DIR, "features_dataset.csv"), index=False)
    print(f"Saved {len(features_df)} rows to data/features_dataset.csv")

    # Also save the most recent known form per driver — this is what the
    # live app uses to predict an upcoming race.
    latest = (
        features_df.sort_values(["Driver", "Year", "Round"])
        .groupby("Driver")
        .tail(1)[["Driver", "Team", "DriverAvgQuali_form", "DriverAvgRace_form", "TeamAvgRace_form"]]
    )
    latest.to_csv(os.path.join(DATA_DIR, "latest_form.csv"), index=False)
    print(f"Saved latest form snapshot for {len(latest)} drivers to data/latest_form.csv")
