"""
Trains two RandomForest classifiers on data/features_dataset.csv:
- quali_top10_model: predicts P(finishing top 10 in qualifying)
- race_top10_model: predicts P(finishing top 10 in the race)

Run:
    python src/train_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Project root = one level up from this file (src/train_model.py -> project/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

QUALI_FEATURES = ["DriverAvgQuali_form", "TeamAvgRace_form"]
RACE_FEATURES = ["DriverAvgRace_form", "TeamAvgRace_form", "GridPosition"]


def train_one(df, feature_cols, target_col, model_name):
    data = df.dropna(subset=feature_cols + [target_col])
    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, preds))
    print(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump({"model": model, "features": feature_cols}, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "features_dataset.csv"))
    train_one(df, QUALI_FEATURES, "QualiTop10", "quali_top10_model")
    train_one(df, RACE_FEATURES, "RaceTop10", "race_top10_model")
