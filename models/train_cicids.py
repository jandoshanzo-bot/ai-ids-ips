from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "CICIDS2017"
SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

csv_files = list(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATASET_DIR}")

frames = []

for file in csv_files:
    print(f"Loading: {file.name}")
    df_part = pd.read_csv(file, low_memory=False)
    df_part.columns = df_part.columns.str.strip()
    frames.append(df_part)

df = pd.concat(frames, ignore_index=True)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

if "Label" not in df.columns:
    raise ValueError("Label column not found")

y_raw = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])
feature_names = list(X.columns)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
y_raw = y_raw.str.replace("�", "-", regex=False)
y_raw = y_raw.str.replace("Web Attack - Brute Force", "Web Attack Brute Force", regex=False)
y_raw = y_raw.str.replace("Web Attack - Sql Injection", "Web Attack SQL Injection", regex=False)
y_raw = y_raw.str.replace("Web Attack - XSS", "Web Attack XSS", regex=False)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_classes = len(label_encoder.classes_)

if num_classes == 2:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
else:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        random_state=42,
        n_jobs=-1,
    )

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred))
print("F1 weighted:", f1_score(y_test, pred, average="weighted"))
print(classification_report(y_test, pred, target_names=label_encoder.classes_))

joblib.dump(model, SAVE_DIR / "cicids_xgb.joblib")
joblib.dump(scaler, SAVE_DIR / "cicids_scaler.joblib")
joblib.dump(label_encoder, SAVE_DIR / "cicids_label_encoder.joblib")

with open(SAVE_DIR / "cicids_features.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=2)

print("Saved CICIDS2017 model successfully")