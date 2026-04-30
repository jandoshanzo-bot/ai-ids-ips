"""
Visualization helpers for IDS model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
try:
    from xgboost import plot_importance
except ImportError:  # pragma: no cover - dependency check
    plot_importance = None


def _prepare_path(save_path: str | Path | None, fallback_name: str) -> Path:
    if save_path is None:
        output_dir = Path("plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / fallback_name

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[str] = ("Normal", "Attack"),
    label_ids: Sequence[int] | None = None,
    save_path: str | Path | None = None,
) -> str:
    """Plot confusion matrix heatmap and save as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=list(label_ids) if label_ids is not None else None)
    labels = list(labels)
    path = _prepare_path(save_path, "confusion_matrix.png")

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Шатасу матрицасы")
    plt.xlabel("Болжанған белгі")
    plt.ylabel("Нақты белгі")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return str(path)


def plot_roc_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    model_scores: Mapping[str, Iterable[float]] | None = None,
    save_path: str | Path | None = None,
) -> Tuple[str, float]:
    """Plot ROC curve (single or multi-model) and return saved path with primary AUC."""
    path = _prepare_path(save_path, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#ededed")
    ax.set_facecolor("#ededed")
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9a9a9a",
        linewidth=2.0,
        label="Кездейсоқ классификатор",
    )

    primary_auc = 0.5
    color_map = {
        "logistic regression": "#3498db",
        "random forest": "#27ae60",
        "xgboost": "#e74c3c",
    }
    if model_scores:
        for model_name, scores in model_scores.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                auc_score = float(roc_auc_score(y_true, scores))
            except ValueError:
                fpr, tpr, auc_score = [0, 1], [0, 1], 0.5

            line_color = color_map.get(model_name.lower(), None)
            ax.plot(
                fpr,
                tpr,
                linewidth=3.0,
                alpha=0.95,
                color=line_color,
                label=f"{model_name} (AUC = {auc_score:.3f})",
            )
            if model_name.lower() == "xgboost":
                primary_auc = auc_score

        if "xgboost" not in [name.lower() for name in model_scores.keys()]:
            first_scores = next(iter(model_scores.values()))
            try:
                primary_auc = float(roc_auc_score(y_true, first_scores))
            except ValueError:
                primary_auc = 0.5
    else:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            primary_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            fpr, tpr, primary_auc = [0, 1], [0, 1], 0.5

        ax.plot(
            fpr,
            tpr,
            linewidth=3.0,
            color="#e74c3c",
            label=f"XGBoost (AUC = {primary_auc:.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "ROC қисығы - модельдерді салыстыру",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Жалған оң көрсеткіш (FPR)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Шынайы оң көрсеткіш (TPR)", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return str(path), float(primary_auc)


def plot_xgboost_feature_importance(
    xgb_model,
    feature_names: Iterable[str] | None = None,
    max_num_features: int = 20,
    save_path: str | Path | None = None,
) -> str:
    """Plot XGBoost feature importance in F-score style."""
    path = _prepare_path(save_path, "xgboost_feature_importance.png")

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#ededed")
    ax.set_facecolor("#ededed")

    if not hasattr(xgb_model, "feature_importances_"):
        if plot_importance is None:
            raise ImportError("xgboost is not installed and model has no feature_importances_.")
        plot_importance(
            xgb_model,
            ax=ax,
            max_num_features=max_num_features,
            importance_type="gain",
            show_values=True,
        )
        ax.set_xlabel("Маңыздылық балы", fontsize=16, fontweight="bold")
        ax.set_ylabel("Белгілер", fontsize=14, fontweight="bold")
        ax.set_title(
            "Белгілер маңыздылығы (XGBoost)\nШабуылды анықтаудағы үздік белгілер",
            fontsize=20,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    importances = np.asarray(xgb_model.feature_importances_, dtype=float)
    if importances.size == 0:
        raise ValueError("XGBoost feature_importances_ is empty.")

    if feature_names is not None:
        feature_names = list(feature_names)
    if feature_names and len(feature_names) == len(importances):
        labels = feature_names
    else:
        labels = [f"f{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({"feature": labels, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(max_num_features)

    palette = sns.color_palette("RdYlGn", n_colors=len(importance_df))[::-1]
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=palette,
        edgecolor="#606060",
        linewidth=0.9,
    )
    ax.invert_yaxis()

    x_max = float(importance_df["importance"].max()) if len(importance_df) > 0 else 1.0
    for bar, value in zip(bars, importance_df["importance"]):
        ax.text(
            value + x_max * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )

    ax.set_xlabel("Маңыздылық балы", fontsize=16, fontweight="bold")
    ax.set_ylabel("", fontsize=12)
    ax.set_title(
        "Белгілер маңыздылығы (XGBoost)\nШабуылды анықтаудағы үздік белгілер",
        fontsize=20,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="both", alpha=0.2)
    ax.set_xlim(0, x_max * 1.2 if x_max > 0 else 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return str(path)


def plot_model_accuracy_comparison(
    model_metrics: Mapping[str, Mapping[str, float]],
    save_path: str | Path | None = None,
) -> str:
    """Plot grouped metrics chart (accuracy/precision/recall/f1) across models."""
    path = _prepare_path(save_path, "model_accuracy_comparison.png")

    models = list(model_metrics.keys())
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#f5f5f5")
    ax.set_facecolor("#f5f5f5")

    x = np.arange(len(metrics_to_plot))
    bar_width = 0.24 if len(models) >= 3 else 0.30
    offsets = np.linspace(
        -bar_width * (len(models) - 1) / 2,
        bar_width * (len(models) - 1) / 2,
        len(models),
    )
    colors = ["#98d8aa", "#f7b267", "#7fb3d5", "#c39bd3", "#76d7c4"]

    for idx, model_name in enumerate(models):
        values = [float(model_metrics[model_name].get(metric_key, 0.0)) for metric_key, _ in metrics_to_plot]
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=bar_width,
            label=model_name,
            color=colors[idx % len(colors)],
            edgecolor="gray",
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics_to_plot])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value")
    ax.set_xlabel("Көрсеткіштер")
    ax.set_title("Модель метрикаларын салыстыру")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return str(path)


def plot_attack_distribution_pie(
    labels: Iterable[int],
    normal_value: int = 0,
    attack_value: int = 1,
    class_labels: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> str:
    """Plot pie chart showing class distribution (binary or multi-class)."""
    path = _prepare_path(save_path, "attack_distribution.png")
    label_series = pd.Series(list(labels))

    if class_labels:
        class_labels = list(class_labels)
        counts = [int((label_series == idx).sum()) for idx in range(len(class_labels))]
        if sum(counts) == 0:
            counts = [1] + [0] * (len(class_labels) - 1)
        color_map = {
            "Normal": "#10b981",
            "DoS": "#ef4444",
            "Probe": "#f59e0b",
            "R2L": "#8b5cf6",
            "U2R": "#ec4899",
        }
        colors = [color_map.get(label, "#94a3b8") for label in class_labels]

        fig, ax = plt.subplots(figsize=(7.2, 6.5), facecolor="#f2f2f2")
        ax.set_facecolor("#f2f2f2")
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f"{label} ({count})" for label, count in zip(class_labels, counts)],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.02 if label == "Normal" else 0.05 for label in class_labels],
            shadow=True,
            wedgeprops={"edgecolor": "white"},
        )
        for text in texts + autotexts:
            text.set_fontsize(9)
        ax.set_title("Шабуылдардың үлесі")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    normal_count = int((label_series == normal_value).sum())
    attack_count = int((label_series == attack_value).sum())

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#f2f2f2")
    ax.set_facecolor("#f2f2f2")
    wedges, texts, autotexts = ax.pie(
        [normal_count, attack_count],
        labels=["Қалыпты", "Шабуыл"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4caf50", "#f44336"],
        explode=[0.02, 0.05],
        shadow=True,
        wedgeprops={"edgecolor": "white"},
    )
    for text in texts + autotexts:
        text.set_fontsize(10)
    ax.set_title("Шабуылдардың үлесі")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return str(path)
