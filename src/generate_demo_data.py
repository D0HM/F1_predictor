"""
Generates a synthetic season_results.csv with realistic structure so you can
try the full pipeline (build_features -> train_model -> app) immediately,
without waiting on real FastF1 downloads.

Run:
    python src/generate_demo_data.py
Then:
    python src/build_features.py --years 2099
    python src/train_model.py
    python app.py
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(42)

TEAMS = {
    "Red Bull Racing": ["VER", "TSU"],
    "Ferrari": ["LEC", "HAM"],
    "McLaren": ["NOR", "PIA"],
    "Mercedes": ["RUS", "ANT"],
    "Aston Martin": ["ALO", "STR"],
    "Williams": ["ALB", "SAI"],
    "Haas": ["OCO", "BEA"],
    "RB": ["HAD", "LAW"],
    "Kick Sauber": ["HUL", "BOR"],
    "Alpine": ["GAS", "COL"],
}

# rough relative team pace (lower = faster), used to bias simulated results
TEAM_PACE = {
    "Red Bull Racing": 1, "McLaren": 1, "Ferrari": 2, "Mercedes": 2,
    "Aston Martin": 4, "Williams": 5, "RB": 5, "Haas": 6,
    "Kick Sauber": 7, "Alpine": 7,
}

ROUNDS = 12
YEAR = 2099  # sentinel year so it never collides with a real collected season

rows = []
for rnd in range(1, ROUNDS + 1):
    driver_scores = []
    for team, drivers in TEAMS.items():
        for drv in drivers:
            base = TEAM_PACE[team] * 1.8
            noise = rng.normal(0, 2.2)
            driver_scores.append((drv, team, base + noise))

    # Rank into grid/quali/race positions with some independent noise so
    # race result isn't a perfect copy of grid position
    driver_scores.sort(key=lambda x: x[2])
    quali_order = [d for d, t, s in driver_scores]

    race_scores = [
        (drv, team, score + rng.normal(0, 1.6))
        for drv, team, score in driver_scores
    ]
    race_scores.sort(key=lambda x: x[2])
    race_order = [d for d, t, s in race_scores]

    team_lookup = {drv: team for team, drivers in TEAMS.items() for drv in drivers}

    for pos, drv in enumerate(quali_order, start=1):
        race_pos = race_order.index(drv) + 1
        rows.append({
            "Year": YEAR,
            "Round": rnd,
            "EventName": f"Demo Round {rnd}",
            "Driver": drv,
            "Team": team_lookup[drv],
            "GridPosition": pos,
            "QualiPosition": pos,
            "RacePosition": race_pos,
            "Points": max(0, 26 - race_pos * 2),
            "Status": "Finished",
        })

df = pd.DataFrame(rows)
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, f"{YEAR}_season_results.csv")
df.to_csv(out_path, index=False)
print(f"Saved synthetic demo data: {len(df)} rows -> {out_path}")
