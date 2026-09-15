"""Build race-level historical features without maintaining a live latest_form file.

The web app calculates the form snapshot dynamically for the selected race, using
only results available before that race. This module remains responsible for
creating the training dataset used by train_model.py.
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
        paths = glob.glob(os.path.join(DATA_DIR, f"{y}_season_results.csv"))
        for path in paths:
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No season_results CSVs found for given years")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    df["QualiTop10"] = (df["QualiPosition"] <= 10).astype(int)
    df["RaceTop10"] = (df["RacePosition"] <= 10).astype(int)

    # Shift first: the current race can never contribute to its own features.
    df["DriverAvgQuali_form"] = (
        df.groupby("Driver")["QualiPosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df["DriverAvgRace_form"] = (
        df.groupby("Driver")["RacePosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    team_race = (
        df.groupby(["Year", "Round", "Team"])["RacePosition"]
        .mean()
        .reset_index()
        .sort_values(["Team", "Year", "Round"])
    )
    team_race["TeamAvgRace_form"] = (
        team_race.groupby("Team")["RacePosition"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df = df.merge(
        team_race[["Year", "Round", "Team", "TeamAvgRace_form"]],
        on=["Year", "Round", "Team"],
        how="left",
    )

    return df.dropna(subset=[
        "DriverAvgQuali_form", "DriverAvgRace_form", "TeamAvgRace_form"
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()

    features_df = build_features(args.years)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "features_dataset.csv")
    features_df.to_csv(out_path, index=False)
    print(f"Saved {len(features_df)} rows to {out_path}")
    print("No latest_form.csv is generated. The web app calculates pre-race form dynamically.")
