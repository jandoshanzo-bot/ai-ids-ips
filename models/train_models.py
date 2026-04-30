"""
Machine learning training pipeline for IDS.

Pipeline:
Network traffic CSV -> preprocessing -> train/test split -> train models ->
evaluate -> save models and plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - dependency check
    XGBClassifier = None

from .preprocess import CATEGORICAL_FEATURES, DataPreprocessor, FEATURE_NAMES, NUMERIC_FEATURES
from .visualizations import (
    plot_attack_distribution_pie,
    plot_confusion_matrix,
    plot_model_accuracy_comparison,
    plot_roc_curve,
    plot_xgboost_feature_importance,
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model artifacts directory
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# IDS multi-class labels (stable order across training/inference).
CLASS_LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID_TO_CLASS = {idx: label for label, idx in CLASS_TO_ID.items()}

# NSL-KDD / KDD 99 specific attack -> family mapping.
ATTACK_TO_FAMILY = {
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}
GENERIC_ATTACK_LABELS = {"attack", "anomaly", "intrusion", "malicious"}


def _map_raw_label_to_family(raw_label: str) -> str:
    """Map dataset raw label into IDS family class."""
    normalized = str(raw_label).strip().lower().replace(".", "")
    if not normalized:
        return "Normal"

    if normalized == "normal":
        return "Normal"

    canonical = {
        "dos": "DoS",
        "probe": "Probe",
        "r2l": "R2L",
        "u2r": "U2R",
    }
    if normalized in canonical:
        return canonical[normalized]

    if normalized in ATTACK_TO_FAMILY:
        return ATTACK_TO_FAMILY[normalized]

    # Backward compatibility for generic binary labels.
    if normalized in GENERIC_ATTACK_LABELS:
        logger.warning("Generic label '%s' mapped to DoS family for multi-class training.", raw_label)
        return "DoS"

    logger.warning("Unknown label '%s' mapped to DoS family.", raw_label)
    return "DoS"


class IDSModelTrainer:
    """Train, evaluate and persist IDS models."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict] = {}
        self.preprocessor: Optional[DataPreprocessor] = None
        self.metrics: Dict[str, Dict] = {}
        self.plot_paths: Dict[str, str] = {}
        self.is_trained = False
        self.best_model_name: Optional[str] = None
        self.class_labels = list(CLASS_LABELS)
        self.class_to_id = dict(CLASS_TO_ID)
        self.id_to_class = dict(ID_TO_CLASS)

    @staticmethod
    def _prepare_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataset has IDS feature schema."""
        X = X.copy()

        for feature in FEATURE_NAMES:
            if feature not in X.columns:
                X[feature] = "unknown" if feature in CATEGORICAL_FEATURES else 0

        # Drop extras and keep the expected order.
        X = X[FEATURE_NAMES]

        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].fillna("unknown").astype(str)

        return X

    def load_dataset_from_csv(self, csv_path: str | Path, label_column: str = "label") -> Tuple[pd.DataFrame, np.ndarray]:
        """Load IDS dataset from CSV and map labels to IDS attack families."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' was not found in dataset.")

        y_raw = df[label_column].astype(str).str.strip()
        y_family = y_raw.map(_map_raw_label_to_family)
        y = y_family.map(self.class_to_id).astype(int).to_numpy()

        X = df.drop(columns=[label_column])
        X = self._prepare_feature_frame(X)

        logger.info("Loaded dataset from %s", csv_path)
        logger.info("Samples: %d", len(df))
        class_counts = y_family.value_counts().to_dict()
        logger.info(
            "Class distribution | Normal=%d DoS=%d Probe=%d R2L=%d U2R=%d",
            int(class_counts.get("Normal", 0)),
            int(class_counts.get("DoS", 0)),
            int(class_counts.get("Probe", 0)),
            int(class_counts.get("R2L", 0)),
            int(class_counts.get("U2R", 0)),
        )

        return X, y

    def generate_synthetic_data(self, n_samples: int = 10000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate realistic synthetic IDS data — raw KDD Cup диапазондарында.

        Маңызды: барлық мәндер RAW диапазонда жасалады (CSV файлдармен бірдей),
        содан кейін DataPreprocessor (RobustScaler) scale жасайды.
        Осылайша CSV файлдардағы мәндер train деректесімен дәл сәйкес келеді.

        Шабуыл түрлері:
          DoS   (30%) — count↑↑ (200-511), serror_rate↑ (0.7-1.0), flag=S0/REJ
          Probe (25%) — dst_host_diff_srv_rate↑ (0.6-1.0), srv_count аз (1-5),
                        diff_srv_rate↑ (0.7-1.0), protocol=icmp, duration аз (0-2)
          R2L   (25%) — num_failed_logins↑ (1-5), logged_in=0, duration↑ (100-500)
          U2R   (20%) — root_shell=1, num_root↑ (1-5), num_shells↑ (1-3)
        """
        rng = np.random.default_rng(random_state)

        n_normal = int(n_samples * 0.55)
        n_left   = n_samples - n_normal
        n_dos    = int(n_left * 0.30)
        n_probe  = int(n_left * 0.25)
        n_r2l    = int(n_left * 0.25)
        n_u2r    = n_left - n_dos - n_probe - n_r2l

        rows_normal, rows_dos, rows_probe, rows_r2l, rows_u2r = [], [], [], [], []

        def _row():
            """Барлық белгілер 0-мен толтырылған негізгі жол."""
            return {f: 0.0 for f in NUMERIC_FEATURES}

        # ── NORMAL ────────────────────────────────────────────────────────────
        for _ in range(n_normal):
            r = _row()
            r["duration"]       = float(rng.integers(0, 300))
            r["src_bytes"]      = float(rng.integers(100, 5000))
            r["dst_bytes"]      = float(rng.integers(500, 50000))
            r["count"]          = float(rng.integers(1, 50))
            r["srv_count"]      = float(rng.integers(1, 50))
            r["serror_rate"]    = float(rng.uniform(0, 0.1))
            r["srv_serror_rate"]= float(rng.uniform(0, 0.1))
            r["rerror_rate"]    = float(rng.uniform(0, 0.1))
            r["logged_in"]      = float(rng.choice([0, 1], p=[0.2, 0.8]))
            r["same_srv_rate"]  = float(rng.uniform(0.7, 1.0))
            r["diff_srv_rate"]  = float(rng.uniform(0.0, 0.3))
            r["dst_host_count"] = float(rng.integers(10, 255))
            r["dst_host_srv_count"] = float(rng.integers(10, 255))
            r["dst_host_same_srv_rate"]  = float(rng.uniform(0.7, 1.0))
            r["dst_host_diff_srv_rate"]  = float(rng.uniform(0.0, 0.2))
            r["dst_host_serror_rate"]    = float(rng.uniform(0, 0.1))
            r["dst_host_srv_serror_rate"]= float(rng.uniform(0, 0.1))
            rows_normal.append(r)

        # ── DoS ───────────────────────────────────────────────────────────────
        for _ in range(n_dos):
            r = _row()
            r["duration"]       = 0.0
            r["src_bytes"]      = float(rng.integers(20000, 80000))
            r["dst_bytes"]      = 0.0
            r["count"]          = float(rng.integers(200, 511))
            r["srv_count"]      = float(rng.integers(150, 511))
            r["serror_rate"]    = float(rng.uniform(0.7, 1.0))
            r["srv_serror_rate"]= float(rng.uniform(0.7, 1.0))
            r["rerror_rate"]    = float(rng.uniform(0.0, 0.2))
            r["logged_in"]      = 0.0
            r["same_srv_rate"]  = float(rng.uniform(0.8, 1.0))
            r["diff_srv_rate"]  = float(rng.uniform(0.0, 0.1))
            r["dst_host_count"] = float(rng.integers(200, 255))
            r["dst_host_srv_count"] = float(rng.integers(150, 255))
            r["dst_host_same_srv_rate"]   = float(rng.uniform(0.8, 1.0))
            r["dst_host_diff_srv_rate"]   = float(rng.uniform(0.0, 0.1))
            r["dst_host_serror_rate"]     = float(rng.uniform(0.7, 1.0))
            r["dst_host_srv_serror_rate"] = float(rng.uniform(0.7, 1.0))
            rows_dos.append(r)

        # ── Probe ─────────────────────────────────────────────────────────────
        for _ in range(n_probe):
            r = _row()
            r["duration"]       = float(rng.uniform(0, 2))          # қысқа
            r["src_bytes"]      = float(rng.choice([8, 520, 1032]))  # icmp packet
            r["dst_bytes"]      = 0.0
            r["count"]          = float(rng.integers(1, 10))         # аз
            r["srv_count"]      = float(rng.integers(1, 5))          # аз
            r["serror_rate"]    = 0.0
            r["srv_serror_rate"]= 0.0
            r["rerror_rate"]    = float(rng.uniform(0, 0.5))
            r["logged_in"]      = 0.0
            r["same_srv_rate"]  = float(rng.uniform(0.05, 0.20))     # аз
            r["diff_srv_rate"]  = float(rng.uniform(0.80, 1.00))     # жоғары ✓
            r["srv_diff_host_rate"] = float(rng.uniform(0.70, 1.00)) # жоғары ✓
            r["dst_host_count"] = float(rng.integers(2, 30))         # аз
            r["dst_host_srv_count"] = float(rng.integers(1, 6))      # аз
            r["dst_host_same_srv_rate"]      = float(rng.uniform(0.01, 0.10))
            r["dst_host_diff_srv_rate"]      = float(rng.uniform(0.70, 1.00))  # жоғары ✓
            r["dst_host_same_src_port_rate"] = float(rng.uniform(0.0, 0.1))
            r["dst_host_srv_diff_host_rate"] = float(rng.uniform(0.50, 1.00))  # жоғары ✓
            r["dst_host_serror_rate"]        = 0.0
            r["dst_host_srv_serror_rate"]    = 0.0
            r["dst_host_rerror_rate"]        = float(rng.uniform(0, 0.3))
            rows_probe.append(r)

        # ── R2L ───────────────────────────────────────────────────────────────
        for _ in range(n_r2l):
            r = _row()
            r["duration"]           = float(rng.integers(100, 500))   # ұзақ
            r["src_bytes"]          = float(rng.integers(100, 800))
            r["dst_bytes"]          = float(rng.integers(500, 3000))
            r["num_failed_logins"]  = float(rng.integers(1, 5))        # ✓
            r["logged_in"]          = 0.0                              # ✓
            r["is_guest_login"]     = float(rng.choice([0, 1], p=[0.3, 0.7]))
            r["hot"]                = float(rng.integers(0, 3))
            r["count"]              = float(rng.integers(1, 20))
            r["srv_count"]          = float(rng.integers(1, 20))
            r["serror_rate"]        = float(rng.uniform(0, 0.2))
            r["same_srv_rate"]      = float(rng.uniform(0.5, 1.0))
            r["diff_srv_rate"]      = float(rng.uniform(0.0, 0.4))
            r["dst_host_count"]     = float(rng.integers(5, 100))
            r["dst_host_srv_count"] = float(rng.integers(5, 100))
            r["dst_host_same_srv_rate"] = float(rng.uniform(0.5, 1.0))
            r["dst_host_diff_srv_rate"] = float(rng.uniform(0.0, 0.3))
            r["dst_host_serror_rate"]   = float(rng.uniform(0, 0.2))
            rows_r2l.append(r)

        # ── U2R ───────────────────────────────────────────────────────────────
        for _ in range(n_u2r):
            r = _row()
            r["duration"]            = float(rng.integers(0, 100))
            r["src_bytes"]           = float(rng.integers(500, 5000))
            r["dst_bytes"]           = float(rng.integers(500, 5000))
            r["root_shell"]          = 1.0                             # ✓
            r["num_root"]            = float(rng.integers(1, 5))       # ✓
            r["num_file_creations"]  = float(rng.integers(1, 4))       # ✓
            r["num_shells"]          = float(rng.integers(1, 3))       # ✓
            r["su_attempted"]        = float(rng.choice([0, 1], p=[0.3, 0.7]))
            r["num_compromised"]     = float(rng.integers(1, 5))
            r["hot"]                 = float(rng.integers(1, 5))
            r["logged_in"]           = float(rng.choice([0, 1], p=[0.3, 0.7]))
            r["count"]               = float(rng.integers(1, 30))
            r["srv_count"]           = float(rng.integers(1, 30))
            r["serror_rate"]         = float(rng.uniform(0, 0.1))
            r["same_srv_rate"]       = float(rng.uniform(0.5, 1.0))
            r["dst_host_count"]      = float(rng.integers(5, 100))
            r["dst_host_srv_count"]  = float(rng.integers(5, 100))
            r["dst_host_same_srv_rate"] = float(rng.uniform(0.5, 1.0))
            r["dst_host_diff_srv_rate"] = float(rng.uniform(0.0, 0.2))
            rows_u2r.append(r)

        # ── Hard negatives: 10% шабуыл Normal-ге ұқсас ───────────────────────
        def _add_noise(rows, n_hard):
            for i in range(n_hard):
                rows[i] = {k: float(rng.normal(v, abs(v)*0.3 + 0.1))
                           for k, v in rows[i].items()}
        for lst in [rows_dos, rows_probe, rows_r2l, rows_u2r]:
            _add_noise(lst, int(len(lst) * 0.10))

        # ── DataFrame жасау ───────────────────────────────────────────────────
        all_rows = rows_normal + rows_dos + rows_probe + rows_r2l + rows_u2r
        y = np.array(
            [self.class_to_id["Normal"]]*n_normal
            + [self.class_to_id["DoS"]]*n_dos
            + [self.class_to_id["Probe"]]*n_probe
            + [self.class_to_id["R2L"]]*n_r2l
            + [self.class_to_id["U2R"]]*n_u2r,
            dtype=int,
        )
        X = pd.DataFrame(all_rows, columns=NUMERIC_FEATURES)

        # Categorical features
        proto_n = rng.choice(["tcp","udp"],         n_normal, p=[0.70,0.30])
        proto_d = rng.choice(["tcp","udp"],         n_dos,    p=[0.65,0.35])
        proto_p = rng.choice(["icmp","tcp","udp"],  n_probe,  p=[0.60,0.25,0.15])
        proto_r = rng.choice(["tcp"],               n_r2l,    p=[1.00])
        proto_u = rng.choice(["tcp"],               n_u2r,    p=[1.00])
        X["protocol_type"] = np.concatenate([proto_n,proto_d,proto_p,proto_r,proto_u])

        svc_n = rng.choice(["http","smtp","ftp","other"],   n_normal, p=[0.50,0.20,0.15,0.15])
        svc_d = rng.choice(["http","ftp","private","other"],n_dos,    p=[0.30,0.20,0.35,0.15])
        svc_p = rng.choice(["ecr_i","eco_i","tim_i","urp_i"],n_probe, p=[0.35,0.25,0.25,0.15])
        svc_r = rng.choice(["ftp","telnet","http","other"], n_r2l,    p=[0.35,0.30,0.20,0.15])
        svc_u = rng.choice(["telnet","ftp","http","other"], n_u2r,    p=[0.40,0.25,0.20,0.15])
        X["service"] = np.concatenate([svc_n,svc_d,svc_p,svc_r,svc_u])

        flag_n = rng.choice(["SF","S1","S0"],         n_normal, p=[0.80,0.12,0.08])
        flag_d = rng.choice(["S0","REJ","RSTO","SF"], n_dos,    p=[0.45,0.30,0.15,0.10])
        flag_p = rng.choice(["SF","REJ","OTH"],       n_probe,  p=[0.60,0.25,0.15])
        flag_r = rng.choice(["SF","REJ","S1"],        n_r2l,    p=[0.55,0.30,0.15])
        flag_u = rng.choice(["SF","S1","REJ"],        n_u2r,    p=[0.60,0.25,0.15])
        X["flag"] = np.concatenate([flag_n,flag_d,flag_p,flag_r,flag_u])

        X = self._prepare_feature_frame(X)

        idx = rng.permutation(len(X))
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

        logger.info(
            "Synthetic data (raw ranges): %d total | Normal=%d DoS=%d Probe=%d R2L=%d U2R=%d",
            n_samples, n_normal, n_dos, n_probe, n_r2l, n_u2r,
        )
        return X, y
        rng = np.random.default_rng(random_state)

        # ── Үлес бөлу ─────────────────────────────────────────────────────────
        n_normal = int(n_samples * 0.55)   # 55% қалыпты
        n_left   = n_samples - n_normal
        n_dos    = int(n_left * 0.30)
        n_probe  = int(n_left * 0.25)
        n_r2l    = int(n_left * 0.25)
        n_u2r    = n_left - n_dos - n_probe - n_r2l  # қалғаны

        nf = len(NUMERIC_FEATURES)

        def _zeros(n):
            return np.zeros((n, nf))

        def _fi(name):
            return NUMERIC_FEATURES.index(name)

        # ── NORMAL трафик ──────────────────────────────────────────────────────
        X_normal = rng.normal(0.0, 0.6, (n_normal, nf))
        X_normal[:, _fi("src_bytes")]  = rng.normal(0.5, 0.5, n_normal).clip(0)
        X_normal[:, _fi("dst_bytes")]  = rng.normal(1.0, 0.5, n_normal).clip(0)
        X_normal[:, _fi("serror_rate")]        = rng.beta(1, 9, n_normal)   # ~0.1
        X_normal[:, _fi("dst_host_serror_rate")] = rng.beta(1, 9, n_normal)
        X_normal[:, _fi("srv_serror_rate")]    = rng.beta(1, 9, n_normal)
        X_normal[:, _fi("count")]      = rng.normal(10, 5, n_normal).clip(1, 50)
        X_normal[:, _fi("logged_in")]  = rng.choice([0, 1], n_normal, p=[0.2, 0.8])
        X_normal[:, _fi("dst_host_diff_srv_rate")] = rng.beta(2, 8, n_normal)  # ~0.2
        X_normal[:, _fi("num_failed_logins")] = np.zeros(n_normal)
        X_normal[:, _fi("root_shell")]        = np.zeros(n_normal)
        X_normal[:, _fi("num_root")]          = np.zeros(n_normal)
        X_normal[:, _fi("num_file_creations")] = np.zeros(n_normal)
        X_normal[:, _fi("num_shells")]        = np.zeros(n_normal)

        # ── DoS шабуылы ────────────────────────────────────────────────────────
        # Белгілері: жоғары count, serror_rate=1.0, src_bytes↑↑
        X_dos = rng.normal(0.0, 0.6, (n_dos, nf))
        X_dos[:, _fi("count")]           = rng.normal(400, 80, n_dos).clip(100, 511)
        X_dos[:, _fi("srv_count")]       = rng.normal(350, 60, n_dos).clip(50, 511)
        X_dos[:, _fi("serror_rate")]     = rng.normal(0.90, 0.08, n_dos).clip(0.6, 1.0)
        X_dos[:, _fi("srv_serror_rate")] = rng.normal(0.90, 0.08, n_dos).clip(0.6, 1.0)
        X_dos[:, _fi("dst_host_serror_rate")]     = rng.normal(0.88, 0.08, n_dos).clip(0.5, 1.0)
        X_dos[:, _fi("dst_host_srv_serror_rate")] = rng.normal(0.88, 0.08, n_dos).clip(0.5, 1.0)
        X_dos[:, _fi("src_bytes")]       = rng.normal(3.5, 0.8, n_dos).clip(1.5)
        X_dos[:, _fi("dst_bytes")]       = rng.normal(-0.5, 0.3, n_dos).clip(-1, 0.5)
        X_dos[:, _fi("logged_in")]       = np.zeros(n_dos)
        X_dos[:, _fi("dst_host_diff_srv_rate")] = rng.beta(1, 9, n_dos)   # аз

        # ── Probe сканерлеу ────────────────────────────────────────────────────
        # Белгілері: dst_host_diff_srv_rate↑↑, srv_count↓, icmp, duration қысқа
        X_probe = rng.normal(0.0, 0.5, (n_probe, nf))
        X_probe[:, _fi("dst_host_diff_srv_rate")]      = rng.normal(3.5, 0.5, n_probe).clip(2.0)
        X_probe[:, _fi("dst_host_srv_diff_host_rate")] = rng.normal(3.0, 0.5, n_probe).clip(1.5)
        X_probe[:, _fi("srv_diff_host_rate")]          = rng.normal(3.0, 0.5, n_probe).clip(1.5)
        X_probe[:, _fi("diff_srv_rate")]               = rng.normal(3.0, 0.4, n_probe).clip(1.5)
        X_probe[:, _fi("dst_host_same_srv_rate")]      = rng.normal(-2.0, 0.4, n_probe)  # аз
        X_probe[:, _fi("same_srv_rate")]               = rng.normal(-2.0, 0.4, n_probe)  # аз
        X_probe[:, _fi("srv_count")]     = rng.normal(-2.5, 0.4, n_probe).clip(-3.5, -1.0)
        X_probe[:, _fi("count")]         = rng.normal(-1.5, 0.4, n_probe)
        X_probe[:, _fi("duration")]      = rng.normal(-1.0, 0.3, n_probe)
        X_probe[:, _fi("serror_rate")]   = rng.beta(1, 9, n_probe)
        X_probe[:, _fi("src_bytes")]     = rng.normal(0.5, 0.4, n_probe)
        X_probe[:, _fi("dst_bytes")]     = rng.normal(-0.5, 0.4, n_probe)
        X_probe[:, _fi("logged_in")]     = np.zeros(n_probe)
        X_probe[:, _fi("num_failed_logins")] = np.zeros(n_probe)
        X_probe[:, _fi("root_shell")]    = np.zeros(n_probe)

        # ── R2L (Remote to Local) ──────────────────────────────────────────────
        # Белгілері: num_failed_logins↑↑, logged_in=0, duration ұзақ
        X_r2l = rng.normal(0.0, 0.5, (n_r2l, nf))
        X_r2l[:, _fi("num_failed_logins")] = rng.normal(2.5, 0.6, n_r2l).clip(1.0)
        X_r2l[:, _fi("logged_in")]         = np.zeros(n_r2l)
        X_r2l[:, _fi("is_guest_login")]    = rng.choice([0, 1], n_r2l, p=[0.4, 0.6])
        X_r2l[:, _fi("duration")]          = rng.normal(2.0, 0.5, n_r2l).clip(0.5)   # ұзақ
        X_r2l[:, _fi("src_bytes")]         = rng.normal(0.3, 0.5, n_r2l)
        X_r2l[:, _fi("hot")]               = rng.normal(0.5, 0.3, n_r2l).clip(0)
        X_r2l[:, _fi("serror_rate")]       = rng.beta(2, 8, n_r2l)
        X_r2l[:, _fi("count")]             = rng.normal(-0.5, 0.4, n_r2l)
        X_r2l[:, _fi("root_shell")]        = np.zeros(n_r2l)
        X_r2l[:, _fi("num_root")]          = np.zeros(n_r2l)
        X_r2l[:, _fi("dst_host_diff_srv_rate")] = rng.beta(1, 9, n_r2l)

        # ── U2R (User to Root) ─────────────────────────────────────────────────
        # Белгілері: root_shell=1, num_root↑↑, num_file_creations↑, num_shells↑
        X_u2r = rng.normal(0.0, 0.5, (n_u2r, nf))
        X_u2r[:, _fi("root_shell")]          = rng.normal(2.5, 0.4, n_u2r).clip(1.5)
        X_u2r[:, _fi("num_root")]            = rng.normal(2.5, 0.5, n_u2r).clip(1.0)
        X_u2r[:, _fi("num_file_creations")]  = rng.normal(1.8, 0.5, n_u2r).clip(0.5)
        X_u2r[:, _fi("num_shells")]          = rng.normal(2.0, 0.5, n_u2r).clip(0.5)
        X_u2r[:, _fi("su_attempted")]        = rng.normal(1.5, 0.5, n_u2r).clip(0)
        X_u2r[:, _fi("num_compromised")]     = rng.normal(1.5, 0.5, n_u2r).clip(0)
        X_u2r[:, _fi("hot")]                 = rng.normal(1.5, 0.4, n_u2r).clip(0)
        X_u2r[:, _fi("logged_in")]           = rng.choice([0, 1], n_u2r, p=[0.3, 0.7])
        X_u2r[:, _fi("serror_rate")]         = rng.beta(1, 9, n_u2r)
        X_u2r[:, _fi("count")]               = rng.normal(0.0, 0.5, n_u2r)
        X_u2r[:, _fi("dst_host_diff_srv_rate")] = rng.beta(1, 9, n_u2r)

        # ── Hard negatives: шабуылдардың 10% нормалға ұқсас ──────────────────
        for X_atk in [X_dos, X_probe, X_r2l, X_u2r]:
            n_hard = int(len(X_atk) * 0.10)
            X_atk[:n_hard] = rng.normal(0.0, 0.8, (n_hard, nf))

        # ── Барлығын біріктіру ─────────────────────────────────────────────────
        X_num = np.vstack([X_normal, X_dos, X_probe, X_r2l, X_u2r])
        y = np.concatenate([
            np.full(n_normal, self.class_to_id["Normal"], dtype=int),
            np.full(n_dos, self.class_to_id["DoS"], dtype=int),
            np.full(n_probe, self.class_to_id["Probe"], dtype=int),
            np.full(n_r2l, self.class_to_id["R2L"], dtype=int),
            np.full(n_u2r, self.class_to_id["U2R"], dtype=int),
        ])

        X = pd.DataFrame(X_num, columns=NUMERIC_FEATURES)

        # ── Категориялық белгілер ─────────────────────────────────────────────
        # protocol_type
        proto_normal = rng.choice(["tcp", "udp"],        n_normal, p=[0.70, 0.30])
        proto_dos    = rng.choice(["tcp", "udp"],        n_dos,    p=[0.60, 0.40])
        proto_probe  = rng.choice(["tcp", "udp", "icmp"],n_probe,  p=[0.30, 0.20, 0.50])  # icmp↑
        proto_r2l    = rng.choice(["tcp"],               n_r2l,    p=[1.00])
        proto_u2r    = rng.choice(["tcp"],               n_u2r,    p=[1.00])
        X["protocol_type"] = np.concatenate([proto_normal, proto_dos, proto_probe, proto_r2l, proto_u2r])

        # service
        svc_normal = rng.choice(["http","smtp","ftp","other"], n_normal, p=[0.50,0.20,0.15,0.15])
        svc_dos    = rng.choice(["http","ftp","private","other"], n_dos, p=[0.30,0.20,0.35,0.15])
        svc_probe  = rng.choice(["ecr_i","eco_i","http","other"], n_probe, p=[0.35,0.25,0.25,0.15])
        svc_r2l    = rng.choice(["ftp","telnet","http","other"], n_r2l,   p=[0.35,0.30,0.20,0.15])
        svc_u2r    = rng.choice(["telnet","ftp","http","other"], n_u2r,   p=[0.40,0.25,0.20,0.15])
        X["service"] = np.concatenate([svc_normal, svc_dos, svc_probe, svc_r2l, svc_u2r])

        # flag
        flag_normal = rng.choice(["SF","S1","S0"],        n_normal, p=[0.80,0.12,0.08])
        flag_dos    = rng.choice(["S0","REJ","RSTO","SF"], n_dos,    p=[0.45,0.30,0.15,0.10])
        flag_probe  = rng.choice(["SF","REJ","OTH"],       n_probe,  p=[0.60,0.25,0.15])
        flag_r2l    = rng.choice(["SF","REJ","S1"],        n_r2l,    p=[0.55,0.30,0.15])
        flag_u2r    = rng.choice(["SF","S1","REJ"],        n_u2r,    p=[0.60,0.25,0.15])
        X["flag"] = np.concatenate([flag_normal, flag_dos, flag_probe, flag_r2l, flag_u2r])

        X = self._prepare_feature_frame(X)

        # ── Shuffle ───────────────────────────────────────────────────────────
        idx = rng.permutation(len(X))
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

        logger.info(
            "Synthetic data: %d total | Normal=%d DoS=%d Probe=%d R2L=%d U2R=%d",
            n_samples, n_normal, n_dos, n_probe, n_r2l, n_u2r,
        )
        return X, y

    def train_logistic_regression(self, X_train, y_train, tune_hyperparams: bool = False):
        """Train Logistic Regression baseline model."""
        logger.info("Training Logistic Regression...")

        if tune_hyperparams:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"],
                "class_weight": ["balanced", None],
            }
            search = GridSearchCV(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring="f1_weighted",
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info("Best Logistic Regression params: %s", search.best_params_)
        else:
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)

        return model

    def train_random_forest(self, X_train, y_train, tune_hyperparams: bool = False):
        """Train Random Forest baseline model."""
        logger.info("Training Random Forest...")

        if tune_hyperparams:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced", None],
            }
            search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring="f1_weighted",
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info("Best Random Forest params: %s", search.best_params_)
        else:
            model = RandomForestClassifier(
                n_estimators=250,
                max_depth=16,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        return model

    def train_xgboost(self, X_train, y_train, tune_hyperparams: bool = False):
        """Train XGBoost model (main IDS model)."""
        logger.info("Training XGBoost...")
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run `pip install -r requirements.txt` first.")

        if tune_hyperparams:
            param_grid = {
                "n_estimators": [200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            }
            search = GridSearchCV(
                estimator=XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=len(self.class_labels),
                    random_state=42,
                    n_jobs=-1,
                ),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring="f1_weighted",
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info("Best XGBoost params: %s", search.best_params_)
        else:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(self.class_labels),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        return model

    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evaluate model and return metrics + predictions."""
        y_pred = model.predict(X_test)

        model_classes = [int(c) for c in np.asarray(getattr(model, "classes_", np.unique(y_test))).tolist()]
        class_index = {class_id: idx for idx, class_id in enumerate(model_classes)}
        normal_id = self.class_to_id["Normal"]
        normal_prob_col = class_index.get(normal_id)

        if hasattr(model, "predict_proba"):
            y_prob_all = np.asarray(model.predict_proba(X_test))
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            y_prob_all = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)
        else:
            y_prob_all = np.zeros((len(y_pred), len(model_classes)), dtype=float)
            for row_idx, class_id in enumerate(y_pred):
                col_idx = class_index.get(int(class_id))
                if col_idx is not None:
                    y_prob_all[row_idx, col_idx] = 1.0

        if normal_prob_col is not None and y_prob_all.ndim == 2 and y_prob_all.shape[1] > normal_prob_col:
            y_prob_attack = 1.0 - y_prob_all[:, normal_prob_col]
        else:
            y_prob_attack = (y_pred != normal_id).astype(float)

        y_test_binary = (y_test != normal_id).astype(int)

        try:
            if y_prob_all.ndim == 2 and y_prob_all.shape[1] > 2:
                roc_auc = float(
                    roc_auc_score(
                        y_test,
                        y_prob_all,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            else:
                roc_auc = float(roc_auc_score(y_test_binary, y_prob_attack))
        except ValueError:
            roc_auc = 0.5

        present_labels = sorted(int(lbl) for lbl in np.unique(np.concatenate([y_test, y_pred])))
        present_target_names = [self.id_to_class.get(label_id, str(label_id)) for label_id in present_labels]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": roc_auc,
            "class_labels": list(self.class_labels),
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=list(range(len(self.class_labels))),
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=present_labels,
                target_names=present_target_names,
                zero_division=0,
            ),
        }

        logger.info(
            "%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

        return metrics, y_pred, y_prob_attack

    def _generate_plots(
        self,
        y_all: np.ndarray,
        y_test: np.ndarray,
        y_prob_lr: np.ndarray,
        y_prob_rf: np.ndarray,
        y_pred_xgb: np.ndarray,
        y_prob_xgb: np.ndarray,
        xgb_model,
        feature_names: list[str],
        plots_dir: str | Path | None = None,
    ) -> Dict[str, str]:
        """Generate all requested graphs and return their paths."""
        plot_dir = Path(plots_dir) if plots_dir else MODEL_DIR / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        model_metrics = {
            "Logistic Regression": self.models["logistic_regression"]["metrics"],
            "Random Forest": self.models["random_forest"]["metrics"],
            "XGBoost": self.models["xgboost"]["metrics"],
        }
        y_test_binary = (np.asarray(y_test) != self.class_to_id["Normal"]).astype(int)

        roc_plot_path, roc_auc = plot_roc_curve(
            y_test_binary,
            y_prob_xgb,
            model_scores={
                "Logistic Regression": y_prob_lr,
                "Random Forest": y_prob_rf,
                "XGBoost": y_prob_xgb,
            },
            save_path=plot_dir / "xgboost_roc_curve.png",
        )

        plot_paths = {
            "confusion_matrix": plot_confusion_matrix(
                y_test,
                y_pred_xgb,
                labels=self.class_labels,
                label_ids=list(range(len(self.class_labels))),
                save_path=plot_dir / "xgboost_confusion_matrix.png",
            ),
            "roc_curve": roc_plot_path,
            "feature_importance": plot_xgboost_feature_importance(
                xgb_model,
                feature_names=feature_names,
                max_num_features=16,
                save_path=plot_dir / "xgboost_feature_importance.png",
            ),
            "accuracy_comparison": plot_model_accuracy_comparison(
                model_metrics,
                save_path=plot_dir / "model_accuracy_comparison.png",
            ),
            "attack_distribution": plot_attack_distribution_pie(
                y_all,
                class_labels=self.class_labels,
                save_path=plot_dir / "attack_distribution_pie.png",
            ),
        }

        self.models["xgboost"]["metrics"]["roc_auc"] = float(roc_auc)
        logger.info("Saved evaluation plots to: %s", plot_dir)
        return plot_paths

    def train(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        csv_path: str | Path | None = None,
        label_column: str = "label",
        test_size: float = 0.2,
        tune_hyperparams: bool = False,
        generate_plots: bool = True,
        plots_dir: str | Path | None = None,
    ) -> Dict[str, Dict]:
        """Train Logistic Regression, Random Forest and XGBoost models."""
        if csv_path:
            X, y = self.load_dataset_from_csv(csv_path, label_column=label_column)
        elif X is None or y is None:
            X, y = self.generate_synthetic_data(n_samples=10000)
        else:
            X = self._prepare_feature_frame(X)
            y = np.asarray(y)

        logger.info("Preprocessing dataset...")
        self.preprocessor = DataPreprocessor(use_robust=True)
        X_processed = self.preprocessor.fit_transform(X)
        processed_feature_names = [col for col in NUMERIC_FEATURES if col in X.columns] + [
            col for col in CATEGORICAL_FEATURES if col in X.columns
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

        logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

        # 1) Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train, tune_hyperparams=tune_hyperparams)
        lr_metrics, _, y_prob_lr = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        self.models["logistic_regression"] = {
            "model": lr_model,
            "metrics": lr_metrics,
            "name": "Logistic Regression",
        }

        # 2) Random Forest
        rf_model = self.train_random_forest(X_train, y_train, tune_hyperparams=tune_hyperparams)
        rf_metrics, _, y_prob_rf = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        self.models["random_forest"] = {
            "model": rf_model,
            "metrics": rf_metrics,
            "name": "Random Forest",
        }

        # 3) XGBoost (main model)
        xgb_model = self.train_xgboost(X_train, y_train, tune_hyperparams=tune_hyperparams)
        xgb_metrics, y_pred_xgb, y_prob_xgb = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        self.models["xgboost"] = {
            "model": xgb_model,
            "metrics": xgb_metrics,
            "name": "XGBoost",
        }

        self.best_model_name = max(
            self.models.keys(),
            key=lambda model_name: self.models[model_name]["metrics"]["f1_score"],
        )
        self.is_trained = True
        logger.info("Best model by F1-score: %s", self.best_model_name)

        if generate_plots:
            self.plot_paths = self._generate_plots(
                y_all=y,
                y_test=y_test,
                y_prob_lr=y_prob_lr,
                y_prob_rf=y_prob_rf,
                y_pred_xgb=y_pred_xgb,
                y_prob_xgb=y_prob_xgb,
                xgb_model=xgb_model,
                feature_names=processed_feature_names,
                plots_dir=plots_dir,
            )

        return self.models

    def save_models(self) -> None:
        """Persist models, preprocessor and metrics to disk."""
        if not self.is_trained:
            raise ValueError("Models are not trained. Run train() first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save preprocessor
        preprocessor_path = MODEL_DIR / "preprocessor.joblib"
        self.preprocessor.save(preprocessor_path)
        logger.info("Saved preprocessor: %s", preprocessor_path)

        for model_name, model_data in self.models.items():
            model_path = MODEL_DIR / f"{model_name}.joblib"
            metrics_path = MODEL_DIR / f"{model_name}_metrics.json"

            joblib.dump(model_data["model"], model_path)
            with open(metrics_path, "w", encoding="utf-8") as file:
                json.dump(model_data["metrics"], file, indent=2)

            logger.info("Saved model: %s", model_path)
            logger.info("Saved metrics: %s", metrics_path)

        info = {
            "timestamp": timestamp,
            "models": list(self.models.keys()),
            "best_model": self.best_model_name,
            "plots": self.plot_paths,
            "class_labels": self.class_labels,
            "class_to_id": self.class_to_id,
        }
        info_path = MODEL_DIR / "model_info.json"
        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)

        logger.info("All model artifacts saved.")

    def load_models(self) -> bool:
        """Load saved models and preprocessor."""
        try:
            preprocessor_path = MODEL_DIR / "preprocessor.joblib"
            if preprocessor_path.exists():
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load(preprocessor_path)

            model_paths = {
                "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
                "random_forest": MODEL_DIR / "random_forest.joblib",
                "xgboost": MODEL_DIR / "xgboost.joblib",
            }

            self.models = {}
            for model_name, model_path in model_paths.items():
                if not model_path.exists():
                    continue

                model_obj = joblib.load(model_path)
                metrics_path = MODEL_DIR / f"{model_name}_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as file:
                        metrics = json.load(file)
                else:
                    metrics = {}

                pretty_name = {
                    "logistic_regression": "Logistic Regression",
                    "random_forest": "Random Forest",
                    "xgboost": "XGBoost",
                }[model_name]

                self.models[model_name] = {
                    "model": model_obj,
                    "metrics": metrics,
                    "name": pretty_name,
                }

            info_path = MODEL_DIR / "model_info.json"
            if info_path.exists():
                with open(info_path, "r", encoding="utf-8") as file:
                    info = json.load(file)
                self.best_model_name = info.get("best_model")
                self.plot_paths = info.get("plots", {})
                loaded_class_labels = info.get("class_labels")
                loaded_class_to_id = info.get("class_to_id")
                if isinstance(loaded_class_labels, list) and loaded_class_labels:
                    self.class_labels = [str(label) for label in loaded_class_labels]
                if isinstance(loaded_class_to_id, dict) and loaded_class_to_id:
                    self.class_to_id = {str(label): int(idx) for label, idx in loaded_class_to_id.items()}
                else:
                    self.class_to_id = {label: idx for idx, label in enumerate(self.class_labels)}
                self.id_to_class = {idx: label for label, idx in self.class_to_id.items()}
            elif self.models:
                self.best_model_name = max(
                    self.models.keys(),
                    key=lambda model_name: self.models[model_name].get("metrics", {}).get("f1_score", 0),
                )
                self.class_labels = list(CLASS_LABELS)
                self.class_to_id = dict(CLASS_TO_ID)
                self.id_to_class = dict(ID_TO_CLASS)

            self.is_trained = len(self.models) > 0
            return self.is_trained

        except Exception as exc:
            logger.error("Failed to load models: %s", exc)
            return False

    def has_required_attack_classes(self) -> bool:
        """Return True when loaded models support Normal/DoS/Probe/R2L/U2R classes."""
        required = set(CLASS_LABELS)
        if not self.models:
            return False

        for model_data in self.models.values():
            model_obj = model_data.get("model")
            model_classes = getattr(model_obj, "classes_", None)
            if model_classes is None:
                return False

            mapped_labels = set()
            for raw_class in np.asarray(model_classes).tolist():
                if isinstance(raw_class, str):
                    mapped_labels.add(raw_class)
                    continue
                try:
                    class_id = int(raw_class)
                except (TypeError, ValueError):
                    continue
                mapped_labels.add(self.id_to_class.get(class_id, str(class_id)))

            if not required.issubset(mapped_labels):
                return False

        return True


def train_and_save_models(
    force_retrain: bool = False,
    csv_path: str | Path | None = None,
    label_column: str = "label",
    test_size: float = 0.2,
    tune_hyperparams: bool = False,
    generate_plots: bool = True,
) -> IDSModelTrainer:
    """High-level helper to train/load and persist IDS models."""
    trainer = IDSModelTrainer()

    expected_artifacts = [
        MODEL_DIR / "preprocessor.joblib",
        MODEL_DIR / "logistic_regression.joblib",
        MODEL_DIR / "random_forest.joblib",
        MODEL_DIR / "xgboost.joblib",
    ]
    model_exists = all(path.exists() for path in expected_artifacts)

    # If CSV is provided, retrain from that dataset by design.
    should_train = force_retrain or not model_exists or csv_path is not None

    if not should_train and trainer.load_models():
        if trainer.has_required_attack_classes():
            logger.info("Loaded existing model artifacts.")
            return trainer
        logger.warning("Loaded artifacts are binary/legacy. Retraining multi-class models...")

    logger.info("Training new models...")
    trainer.train(
        csv_path=csv_path,
        label_column=label_column,
        test_size=test_size,
        tune_hyperparams=tune_hyperparams,
        generate_plots=generate_plots,
    )
    trainer.save_models()
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IDS models with XGBoost and generate evaluation plots.")
    parser.add_argument("--csv", type=str, default=None, help="Path to dataset CSV file.")
    parser.add_argument("--label-column", type=str, default="label", help="Label column name in CSV (default: label).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2).")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots.")
    args = parser.parse_args()

    trainer = train_and_save_models(
        force_retrain=True,
        csv_path=args.csv,
        label_column=args.label_column,
        test_size=args.test_size,
        tune_hyperparams=args.tune,
        generate_plots=not args.no_plots,
    )

    print("Training completed.")
    print(f"Best model: {trainer.best_model_name}")
    for model_name, model_data in trainer.models.items():
        metrics = model_data["metrics"]
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1_score']:.4f}"
        )
