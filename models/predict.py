"""
Prediction module for IDS.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .preprocess import FEATURE_NAMES, validate_input_data
from .train_models import IDSModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSPredictor:
    """Run single or batch IDS predictions."""

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.trainer = IDSModelTrainer()
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.load_model()

    def load_model(self) -> bool:
        """Load trained artifacts; retrain if legacy/binary models are found."""
        try:
            success = self.trainer.load_models()
            if not success or not self.trainer.has_required_attack_classes():
                logger.warning("Legacy or missing model artifacts detected. Retraining multi-class models...")
                self.trainer.train()
                self.trainer.save_models()

            if self.model_name in self.trainer.models:
                selected = self.model_name
            else:
                preferred = self.trainer.best_model_name or "xgboost"
                if preferred in self.trainer.models:
                    selected = preferred
                elif self.trainer.models:
                    selected = next(iter(self.trainer.models.keys()))
                else:
                    raise ValueError("No IDS models are available.")

            self.model_name = selected
            self.model = self.trainer.models[selected]["model"]
            self.preprocessor = self.trainer.preprocessor
            self.is_loaded = True
            logger.info("Model loaded: %s", self.model_name)
            return True

        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self.is_loaded = False
            return False

    def _class_id_to_label(self, class_id) -> str:
        if isinstance(class_id, str):
            return class_id
        try:
            return self.trainer.id_to_class.get(int(class_id), str(class_id))
        except (TypeError, ValueError):
            return str(class_id)

    def predict(self, data, return_features: bool = False):
        """Predict IDS status + attack family."""
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model is not loaded",
                "prediction": None,
                "probability": None,
                "confidence": None,
            }

        validation = validate_input_data(data)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["message"],
                "prediction": None,
                "probability": None,
                "confidence": None,
            }

        try:
            X = validation["data"]
            X_processed = self.preprocessor.transform(X)

            predicted_raw = self.model.predict(X_processed)[0]
            model_classes = np.asarray(getattr(self.model, "classes_", []))

            if hasattr(self.model, "predict_proba"):
                prediction_prob = np.asarray(self.model.predict_proba(X_processed)[0], dtype=float)
            else:
                if model_classes.size == 0:
                    model_classes = np.asarray([predicted_raw])
                prediction_prob = np.zeros(len(model_classes), dtype=float)
                pred_idx = np.where(model_classes == predicted_raw)[0]
                if pred_idx.size == 0:
                    model_classes = np.append(model_classes, predicted_raw)
                    prediction_prob = np.append(prediction_prob, 0.0)
                    pred_idx = np.where(model_classes == predicted_raw)[0]
                prediction_prob[int(pred_idx[0])] = 1.0

            if model_classes.size == 0:
                model_classes = np.arange(len(prediction_prob))

            class_probabilities = {label: 0.0 for label in self.trainer.class_labels}
            for raw_class, prob in zip(model_classes.tolist(), prediction_prob.tolist()):
                class_label = self._class_id_to_label(raw_class)
                class_probabilities[class_label] = float(prob)

            predicted_family = self._class_id_to_label(predicted_raw)
            confidence = float(np.max(prediction_prob)) if prediction_prob.size else 0.0

            normal_probability = float(class_probabilities.get("Normal", 0.0))
            attack_probability = float(max(0.0, min(1.0, 1.0 - normal_probability)))

            status = "Normal" if predicted_family == "Normal" else "Attack"
            attack_type = predicted_family if status == "Attack" else "None"
            attack_type_confidence = (
                float(class_probabilities.get(predicted_family, confidence))
                if status == "Attack"
                else None
            )

            result = {
                "success": True,
                "prediction": status,
                "probability": attack_probability,
                "confidence": confidence,
                "attack_type": attack_type,
                "attack_type_confidence": attack_type_confidence,
                "prediction_class": predicted_family,
                "class_probabilities": class_probabilities,
                "model_name": self.model_name,
                "error": None,
            }

            if return_features:
                result["feature_importance"] = self.get_feature_importance(X)
                result["features"] = X.iloc[0].to_dict() if hasattr(X, "iloc") else None

            return result

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "prediction": None,
                "probability": None,
                "confidence": None,
            }

    def predict_batch(self, data):
        """Predict multiple rows."""
        if not self.is_loaded:
            return [{"success": False, "error": "Model is not loaded"}]

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return [{"success": False, "error": "Unsupported input data format"}]

        results = []
        for _, row in df.iterrows():
            results.append(self.predict(row.to_frame().T))
        return results

    def get_feature_importance(self, X):
        """Return normalized feature importance for current model."""
        if not hasattr(self.model, "feature_importances_") and not hasattr(self.model, "coef_"):
            return {}

        try:
            if hasattr(self.model, "feature_importances_"):
                importances = np.asarray(self.model.feature_importances_, dtype=float)
            else:
                coef = np.asarray(self.model.coef_, dtype=float)
                importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

            total = float(np.sum(importances))
            if total > 0:
                importances = importances / total

            if hasattr(X, "columns"):
                feature_names = X.columns.tolist()
            else:
                feature_names = FEATURE_NAMES[: len(importances)]

            importance_dict = {
                name: float(value) for name, value in zip(feature_names, importances.tolist())
            }
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            )
            return importance_dict

        except Exception as exc:
            logger.error("Failed to compute feature importance: %s", exc)
            return {}

    def get_model_info(self):
        if not self.is_loaded or self.model_name not in self.trainer.models:
            return {"name": self.model_name, "status": "not_loaded", "metrics": {}}

        model_data = self.trainer.models[self.model_name]
        return {
            "name": self.model_name,
            "display_name": model_data["name"],
            "status": "loaded",
            "metrics": model_data.get("metrics", {}),
            "class_labels": self.trainer.class_labels,
        }

    def switch_model(self, model_name):
        if model_name not in self.trainer.models:
            return {"success": False, "error": f"Model not found: {model_name}"}

        self.model_name = model_name
        self.model = self.trainer.models[model_name]["model"]
        return {"success": True, "message": f"Model switched: {model_name}"}

    def get_available_models(self):
        return [
            {
                "name": name,
                "display_name": data["name"],
                "metrics": data.get("metrics", {}),
                "class_labels": self.trainer.class_labels,
            }
            for name, data in self.trainer.models.items()
        ]


_predictor = None


def get_predictor(model_name: str = "xgboost"):
    """Return singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = IDSPredictor(model_name)
    elif model_name and model_name != _predictor.model_name:
        _predictor.switch_model(model_name)
    return _predictor


def predict_single(data, model_name: str = "xgboost", return_features: bool = True):
    predictor = get_predictor(model_name)
    return predictor.predict(data, return_features=return_features)


def predict_batch(data, model_name: str = "xgboost"):
    predictor = get_predictor(model_name)
    return predictor.predict_batch(data)


def get_model_metrics(model_name: str = "xgboost"):
    predictor = get_predictor(model_name)
    return predictor.get_model_info()


def get_all_models():
    predictor = get_predictor()
    return predictor.get_available_models()
