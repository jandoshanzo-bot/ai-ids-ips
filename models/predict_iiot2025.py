from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

IIOT2025_MODEL_NAME = "IIoT2025 XGBoost"

REQUIRED_FILES = {
    "model": MODEL_DIR / "iiot2025_xgb.joblib",
    "label_encoder": MODEL_DIR / "iiot2025_label_encoder.joblib",
    "features": MODEL_DIR / "iiot2025_features.joblib",
}

missing_files = [str(path) for path in REQUIRED_FILES.values() if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "IIoT2025 model files are missing: " + ", ".join(missing_files)
    )

model = joblib.load(REQUIRED_FILES["model"])
label_encoder = joblib.load(REQUIRED_FILES["label_encoder"])
FEATURE_NAMES = list(joblib.load(REQUIRED_FILES["features"]))
CLASSES = list(label_encoder.classes_)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Uploaded data is empty. Please provide a CSV with IIoT2025 feature columns.")

    prepared = df.copy()
    prepared.columns = prepared.columns.astype(str).str.strip()
    prepared = prepared.drop(
        columns=["label_full", "label1", "label2", "label3", "label4", "Label", "label"],
        errors="ignore",
    )
    prepared.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Reindex by the exact feature list saved during IIoT2025 training.
    prepared = prepared.reindex(columns=FEATURE_NAMES, fill_value=0)
    for column in FEATURE_NAMES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared.fillna(0, inplace=True)
    return prepared


def predict_iiot2025_dataframe(df: pd.DataFrame):
    X = _prepare_features(df)
    pred = model.predict(X)
    prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for idx, raw_pred in enumerate(pred):
        label = label_encoder.inverse_transform([int(raw_pred)])[0]

        if prob is not None:
            row_prob = np.asarray(prob[idx], dtype=float)
            confidence = float(np.max(row_prob)) if row_prob.size else 0.0
            class_probabilities = {
                str(CLASSES[j]): float(row_prob[j])
                for j in range(min(len(CLASSES), len(row_prob)))
            }
        else:
            confidence = 1.0
            class_probabilities = {str(label): 1.0}

        results.append({
            "prediction": str(label),
            "is_attack": True,
            "confidence": confidence,
            "attack_probability": confidence,
            "class_probabilities": class_probabilities,
            "model_name": IIOT2025_MODEL_NAME,
        })

    return results


def predict_iiot2025_csv(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    return predict_iiot2025_dataframe(df)
