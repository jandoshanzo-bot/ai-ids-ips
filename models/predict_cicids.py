from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "cicids_xgb.joblib")
scaler = joblib.load(MODEL_DIR / "cicids_scaler.joblib")
label_encoder = joblib.load(MODEL_DIR / "cicids_label_encoder.joblib")

with open(MODEL_DIR / "cicids_features.json", "r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

CLASSES = list(label_encoder.classes_)
BENIGN_INDEX = CLASSES.index("BENIGN") if "BENIGN" in CLASSES else None


def predict_cicids_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = df.select_dtypes(include=[np.number]).copy()

    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0

    X = X[FEATURE_NAMES]

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)

    results = []

    for i in range(len(pred)):
        label = label_encoder.inverse_transform([int(pred[i])])[0]
        confidence = float(np.max(prob[i]))

        if BENIGN_INDEX is not None:
            attack_probability = 1.0 - float(prob[i][BENIGN_INDEX])
        else:
            attack_probability = confidence

        results.append({
            "prediction": label,
            "is_attack": label.upper() != "BENIGN",
            "confidence": confidence,
            "attack_probability": attack_probability,
            "class_probabilities": {
                str(CLASSES[j]): float(prob[i][j])
                for j in range(len(CLASSES))
            },
            "model_name": "CICIDS2017 XGBoost"
        })

    return results


def predict_cicids_csv(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    return predict_cicids_dataframe(df)


if __name__ == "__main__":
    results = predict_cicids_csv(
        "datasets/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )
    print(results[:5])