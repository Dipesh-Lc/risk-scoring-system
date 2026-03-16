"""
Model evaluation: metrics, calibration curves, confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_METRICS, ensure_dirs

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    name: str = "model",
) -> dict[str, Any]:
    """Compute a comprehensive set of classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "name": name,
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
        "default_rate": float(y_true.mean()),
        "predicted_default_rate": float(y_pred.mean()),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    logger.info(
        "[%s] ROC-AUC=%.4f | PR-AUC=%.4f | Brier=%.4f | F1=%.4f",
        name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["brier_score"],
        metrics["f1"],
    )
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = "f1"
) -> float:
    """Find the probability threshold that maximises a given metric."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    metric_fns = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    fn = metric_fns.get(metric, f1_score)

    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [fn(y_true, (y_pred_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = float(thresholds[np.argmax(scores)])
    logger.info("Optimal threshold for %s: %.2f (score=%.4f)", metric, best_t, max(scores))
    return best_t


def save_metrics(metrics: dict, name: str = "metrics") -> Path:
    """Persist metrics as JSON in artifacts/metrics/."""
    ensure_dirs()
    path = ARTIFACTS_METRICS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved -> %s", path)
    return path


def load_metrics(name: str = "metrics") -> dict:
    path = ARTIFACTS_METRICS / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Return a summary DataFrame comparing multiple model metrics."""
    rows = []
    for m in metrics_list:
        rows.append(
            {
                "Model": m.get("name", "?"),
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "Brier": round(m["brier_score"], 4),
                "F1": round(m["f1"], 4),
                "Recall": round(m["recall"], 4),
                "Precision": round(m["precision"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def plot_roc_curve(
    y_true: np.ndarray, models: dict[str, np.ndarray], save_path: Path | None = None
) -> None:
    """Plot ROC curves for one or more models."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, proba in models.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("ROC curve saved -> %s", save_path)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    models: dict[str, np.ndarray],
    save_path: Path | None = None,
) -> None:
    """Plot calibration curves (reliability diagrams)."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    for label, proba in models.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Calibration curve saved -> %s", save_path)
    plt.close()
