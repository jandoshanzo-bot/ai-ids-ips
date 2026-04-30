import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

CSV_PATH = "datasets/CIC_IIoT_2025/cleaned_final.csv"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# DataSense үшін ең дұрыс жалпы label
label_col = "label2"

if label_col not in df.columns:
    raise ValueError(f"{label_col} бағаны табылмады. Бар бағандар: {df.columns.tolist()}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print("label_full counts:")
print(df["label_full"].value_counts().head(20))

print("label1 counts:")
print(df["label1"].value_counts().head(20))

print("label2 counts:")
print(df["label2"].value_counts().head(20))

print("label3 counts:")
print(df["label3"].value_counts().head(20))

print("label4 counts:")
print(df["label4"].value_counts().head(20))

# Тез оқыту үшін 10000 жол ғана аламыз
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

y = df[label_col]
X = df.drop(columns=["label_full", "label1", "label2", "label3", "label4"], errors="ignore")

# Тек сандық feature қалдырамыз
X = X.select_dtypes(include=["number"])

print("Selected label:", label_col)
print("Features count:", len(X.columns))
print("Classes:", sorted(y.unique()))

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Егер сирек класс болса, stratify қате бермеуі үшін тексереміз
class_counts = pd.Series(y_encoded).value_counts()
use_stratify = class_counts.min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded if use_stratify else None
)

model = XGBClassifier(
    n_estimators=80,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Weighted F1:", f1_score(y_test, pred, average="weighted"))
print(classification_report(y_test, pred, target_names=le.classes_))

joblib.dump(model, "models/saved_models/iiot2025_xgb.joblib")
joblib.dump(le, "models/saved_models/iiot2025_label_encoder.joblib")
joblib.dump(X.columns.tolist(), "models/saved_models/iiot2025_features.joblib")

print("Saved: saved_models/iiot2025_xgb.joblib")