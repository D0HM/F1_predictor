# F1 2026 Race Predictor

Flask web app for 2026 Formula 1 qualifying/race prediction using FastF1 historical data and scikit-learn models.

## What it does

1. Select a 2026 Grand Prix.
2. Automatically collects/caches 2025 season results and completed 2026 results before the selected round.
3. Calculates pre-race driver/team form dynamically — no `latest_form.csv` maintenance is required.
4. Restricts predictions to the current 2026 22-driver grid.
5. Predicts qualifying order and race order.
6. Automatically collects the previous year's equivalent race telemetry when needed and adds a final-lap forecast to every race prediction.

The separate Final Lap Time UI section has been removed.

## Important final-lap behavior

The selected 2026 race may not have happened yet, so the app cannot use that race's actual final-lap telemetry as input. Instead, it uses the previous year's equivalent race as the historical baseline. `src/lap_time_model.py` is required for this feature.

## Project structure

```text
F1_predictor/
├── app.py
├── README.md
├── src/
│   ├── build_features.py
│   ├── collect_data.py
│   ├── lap_time_model.py
│   └── train_model.py
├── static/
│   ├── app.js
│   └── style.css
├── templates/
│   └── index.html
├── data/
│   └── *.csv
└── models/
    ├── quali_top10_model.pkl
    └── race_top10_model.pkl
```

`cache/`, `__pycache__/`, and similar generated files should stay out of Git.

## Install

```bash
python -m pip install flask fastf1 pandas numpy scikit-learn joblib
```

## Run

From the project root:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Model notes

The qualifying and race models currently predict Top-10 probability and rank drivers by that probability. Displayed P1–P22 values are therefore rankings, not direct exact-position probabilities.

The final-lap model is a historical forecast based on the previous year's equivalent race; it is not an actual 2026 race result.
