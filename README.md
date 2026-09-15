# F1 2026 Race Predictor

A Flask-based Formula 1 prediction web application that uses **FastF1 historical data** and machine-learning models to predict the 2026 qualifying and race order.

The application is designed around the following workflow:

```text
Select 2026 Grand Prix
        ↓
FastF1 2026 schedule
        ↓
Collect previous-season history + completed 2026 results
        ↓
Build driver/team form features
        ↓
Predict qualifying order
        ↓
Use predicted qualifying order as race grid
        ↓
Predict race order
        ↓
Forecast final-lap time
```

## Main Features

- Select any race from the **2026 F1 schedule**.
- Uses an explicit **2026 22-driver grid**, preventing drivers who are no longer race drivers in 2026 from appearing in predictions.
- Automatically collects historical data through **FastF1**.
- Uses:
  - 2025 season history
  - Completed 2026 races before the selected round
- Predicts:
  - Qualifying P1–P22
  - Race P1–P22
  - Final-lap time for each predicted race driver
- Final-lap forecasting is integrated directly into the **Race** prediction section.
- The old standalone **Final Lap Time** section has been removed.

## Project Structure

The project should have a structure similar to:

```text
f1-predictor/
│
├── app.py
├── collect_data.py
├── index.html
├── app.js
│
├── build_features.py
├── train_model.py
├── lap_time_model.py
│
├── models/
│   ├── quali_top10_model.pkl
│   └── race_top10_model.pkl
│
├── data/
│   ├── 2025_season_results.csv
│   ├── 2026_season_results.csv
│   └── ...
│
└── static/
    └── style.css
```

If the project keeps Python source files inside a `src/` directory, that is also supported by `app.py`.

## Requirements

Python 3.10+ is recommended.

Install the required Python packages:

```bash
pip install flask fastf1 pandas numpy scikit-learn joblib
```

If `pip` is not recognized or installs into a different Python environment, use:

```bash
python -m pip install flask fastf1 pandas numpy scikit-learn joblib
```

## Running the Application

From the project root:

```bash
python app.py
```

Then open the local address shown by Flask, normally:

```text
http://127.0.0.1:5000
```

## How Prediction Works

### 1. 2026 Driver Grid

The application uses a fixed 2026 driver list rather than taking every driver found in the 2025 dataset.

This is important because the historical dataset can contain drivers who are not racing in 2026.

For example, a 2025 driver can exist in the historical data but must not automatically become a 2026 prediction candidate.

### 2. Historical Data

When a prediction is requested, the application obtains:

- 2025 season results
- Completed 2026 results before the selected race

FastF1 is responsible for retrieving the F1 session data.

Previously collected season CSV files are reused when available, which avoids downloading the same season repeatedly.

### 3. Driver Form

The feature-building process calculates rolling form using the previous four races:

```text
DriverAvgQuali_form
DriverAvgRace_form
TeamAvgRace_form
```

The current race is excluded from the rolling calculation to reduce data leakage.

### 4. Qualifying Prediction

The qualifying model uses:

```text
DriverAvgQuali_form
TeamAvgRace_form
```

It predicts the probability that a driver finishes in the qualifying Top 10.

The application then uses those probabilities to create a predicted P1–P22 qualifying order.

**Important:** the current model is a Top-10 classifier. Therefore, P1–P22 is a ranking generated from Top-10 probabilities, not a model that directly predicts the exact finishing position.

### 5. Race Prediction

The race model uses:

```text
DriverAvgRace_form
TeamAvgRace_form
GridPosition
```

The predicted qualifying position is passed into the race model as the driver's predicted grid position.

The resulting race Top-10 probabilities are then ranked to produce the predicted P1–P22 race order.

### 6. Final-Lap Forecast

Final-lap prediction is now part of the Race result.

For the selected 2026 Grand Prix, the application automatically requests the **previous year's equivalent race**.

For example:

```text
2026 Singapore GP
        ↓
2025 Singapore GP
        ↓
FastF1 race lap data
        ↓
Random Forest regression
        ↓
2026 final-lap forecast
```

The final-lap model uses available historical variables such as:

```text
TyreLife
SpeedI1
SpeedI2
SpeedFL
SpeedST
AirTemp
TrackTemp
Humidity
WindSpeed
Position
Compound
Team
Driver
```

The predicted race position is used as the position input for the forecast.

For a driver without usable data from the previous year's race, the application falls back to the driver's team data or a global historical template.

The final-lap value is a **forecast**, not an actual race result.

## Data Collection Commands

Collect an entire season:

```bash
python collect_data.py --season-results --year 2025
```

Force a fresh download:

```bash
python collect_data.py --season-results --year 2025 --force
```

Collect a specific race/session:

```bash
python collect_data.py --year 2025 --gp Singapore --session R
```

Collect the previous year's race for a target 2026 GP:

```bash
python collect_data.py --year 2026 --gp Singapore --previous-year-race
```

## Training the Existing Models

The existing training pipeline creates two Random Forest classifiers.

Build the feature dataset:

```bash
python build_features.py --years 2023 2024 2025
```

Then train the models:

```bash
python train_model.py
```

The models are saved as:

```text
models/quali_top10_model.pkl
models/race_top10_model.pkl
```

## Data and Model Limitations

### Exact finishing positions

The current qualifying and race models predict **Top-10 probability**, then rank the drivers by that probability.

They do not directly estimate:

```text
P(P1)
P(P2)
...
P(P22)
```

Therefore, the displayed P1–P22 order should be interpreted as a ranking derived from the current Top-10 models.

### New drivers

A driver without sufficient 2025 F1 history cannot have the same historical form features as an established driver.

The application therefore uses a fallback based on the driver's current team's historical form, followed by an overall historical median when necessary.

### Team changes

Historical team performance does not always represent a team's 2026 performance after regulation changes or major driver/team changes.

The predictions should therefore be treated as model-based estimates rather than guaranteed race outcomes.

### Final-lap forecast

The final-lap model uses the previous year's equivalent Grand Prix as its main historical reference.

Actual 2026 conditions can be substantially different because of:

- weather
- tyre strategy
- fuel load
- safety cars
- track evolution
- setup
- damage
- race incidents
- regulation/car-performance changes

## FastF1 Cache

FastF1 caching is enabled by the data-collection module.

The first data collection can take several minutes because FastF1 may need to download session data.

Later runs should be faster because previously downloaded data can be reused.

## Troubleshooting

### `ModuleNotFoundError`

Install the missing package, for example:

```bash
python -m pip install fastf1
```

### FastF1 cannot load a session

Check:

1. Internet connection
2. The selected Grand Prix name
3. Whether FastF1 has data available for that session
4. Whether the FastF1 cache directory is writable

### No model files found

Make sure these files exist:

```text
models/quali_top10_model.pkl
models/race_top10_model.pkl
```

If they do not exist, train the models with:

```bash
python build_features.py --years 2023 2024 2025
python train_model.py
```

### Old drivers still appear

The prediction code should use the explicit 2026 driver list in `app.py`.

If old results still appear, make sure the browser is loading the updated `app.py` and restart Flask after replacing the file.

## Notes

This project is intended for educational/data-science use. Predictions are statistical estimates based on historical data and the trained models. They should not be treated as official F1 predictions or guaranteed race outcomes.
