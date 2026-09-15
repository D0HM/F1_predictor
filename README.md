# Marina Bay Predictor — Singapore GP Top 10 (Quali + Race)

A Flask web app that predicts each driver's probability of finishing top 10 in
Qualifying and in the Race, using rolling driver/team form built from FastF1
historical data. Two RandomForest classifiers, one HTML dashboard.

## How it fits together

```
src/collect_season_results.py  → pulls Q + R results for every round of a season
src/generate_demo_data.py      → OR: synthetic data to try the app immediately
src/build_features.py          → builds rolling-form features + top10 labels
src/train_model.py             → trains quali_top10_model.pkl, race_top10_model.pkl
app.py + templates/ + static/  → Flask app serving the prediction dashboard
```

## Quick start (synthetic demo data — works immediately, no internet needed)

```bash
pip install fastf1 pandas scikit-learn flask joblib

python src/generate_demo_data.py
python src/build_features.py --years 2099
python src/train_model.py
python app.py
```
Open **http://127.0.0.1:5000**. Adjust a driver's assumed grid position and click
"Run prediction" — the race-top10 odds update live.

## Using real F1 data (run this locally — FastF1 needs to reach F1's timing/Ergast servers)

```bash
python src/collect_season_results.py --year 2023
python src/collect_season_results.py --year 2024
python src/collect_season_results.py --year 2025
python src/build_features.py --years 2023 2024 2025
python src/train_model.py
python app.py
```

This pulls every completed round's qualifying + race results, computes each
driver's rolling form (avg finishing position over their last 4 races) and each
team's rolling pace, going into the season right up to the most recent race —
so `latest_form.csv` reflects form heading into Singapore.

## What the models actually use

- **Qualifying model**: driver's rolling avg quali position, team's rolling avg
  race pace.
- **Race model**: driver's rolling avg race position, team's rolling avg race
  pace, and grid position (the editable input in the UI — this is what lets you
  ask "what if they qualify P3 instead of P8?").

## Honest limitations

- Street circuits like Marina Bay are notoriously unpredictable (safety cars,
  barrier contact, heat) — the model has no signal for circuit-specific chaos.
- No live current-season data is baked in; you need to run the collection
  scripts yourself with an internet connection before Singapore weekend to get
  fresh form.
- This is a rolling-average model, not a physics or race-strategy simulation —
  treat probabilities as a form guide, not a forecast.

## Extending

- Add a "street circuit" feature and a driver's historical street-circuit
  finishing position for circuit-specific signal.
- Swap RandomForest for Gradient Boosting or logistic regression and compare
  (same pattern as the earlier lap-time notebook).
- Deploy properly with `gunicorn app:app` behind a real web server instead of
  Flask's dev server.
