"""
SHAP-based explainability: global feature importance and per-prediction drivers.
Falls back gracefully when the shap library is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_SHAP, ensure_dirs

logger = get_logger(__name__)

try:
    import shap 

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability features will return empty results.")


def _get_explainer(model) -> Any:
    """
    Build the appropriate SHAP explainer for the given model type.
    TreeExplainer for XGBoost/GBDT, LinearExplainer for LR, KernelExplainer as fallback.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed.")

    import shap

    # Unwrap CalibratedClassifierCV to reach the base estimator
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator

    if hasattr(base, "get_booster"):  # XGBoost
        return shap.TreeExplainer(base)
    elif hasattr(base, "estimators_") and hasattr(base, "learning_rate"):  # GradientBoosting
        return shap.TreeExplainer(base)
    elif hasattr(base, "coef_"):  # Linear model
        return shap.LinearExplainer(
            base, masker=shap.maskers.Independent(np.zeros((1, base.coef_.shape[1])))
        )
    else:
        logger.warning("Using slow KernelExplainer -- TreeExplainer not applicable.")
        return shap.KernelExplainer(model.predict_proba, shap.sample(np.zeros((100, 1)), 50))


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 500,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a sample of X.

    Returns:
        (shap_values array, X_sample DataFrame)
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Install it with: pip install shap")

    import shap

    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    explainer = _get_explainer(model)

    try:
        shap_values = explainer.shap_values(X)
        # For binary classifiers, shap_values may be list of [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        raise

    logger.info("SHAP values computed for %d samples × %d features.", *shap_values.shape)
    return shap_values, X


def global_feature_importance(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute and optionally save global mean |SHAP| feature importances.

    Returns:
        DataFrame with columns [feature, mean_abs_shap], sorted descending.
    """
    shap_values, X_sample = compute_shap_values(model, X)

    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top 5 SHAP features:\n%s", importance.head(5).to_string(index=False))

    if save:
        ensure_dirs()
        path = ARTIFACTS_SHAP / "global_importance.csv"
        importance.to_csv(path, index=False)
        logger.info("Global SHAP importance saved -> %s", path)

    return importance


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Generate and optionally save a SHAP beeswarm summary plot."""
    import matplotlib.pyplot as plt
    import shap

    shap_values, X_sample = compute_shap_values(model, X)
    X_sample.columns = feature_names[: X_sample.shape[1]]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()

    if save_path is None:
        save_path = ARTIFACTS_SHAP / "shap_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("SHAP summary plot saved -> %s", save_path)
    plt.close()


def explain_single(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Compute SHAP values for a single row and return the top N drivers.
    Returns empty list if shap is not installed.
    """
    if not _SHAP_AVAILABLE:
        logger.debug("shap not available - returning empty drivers list.")
        return []

    import shap

    explainer = _get_explainer(model)
    try:
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        shap_row = sv[0] if sv.ndim == 2 else sv
    except Exception as exc:
        logger.warning("Single-row SHAP failed: %s", exc)
        return []

    row_values = X_row.iloc[0].values
    drivers = sorted(
        zip(feature_names, shap_row, row_values),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": feat,
            "shap_value": round(float(sv), 4),
            "feature_value": round(float(val), 4),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
        }
        for feat, sv, val in drivers
    ]
