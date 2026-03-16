"""
Prediction helpers: load model + produce scored output for a record or batch.
"""

from __future__ import annotations

import pandas as pd

from src.models.calibrate import build_score_output
from src.models.registry import load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_single(
    features: dict,
    model=None,
    scaler=None,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Score a single customer record.

    Args:
        features: Dict of raw feature values (matches original UCI columns).
        model: Fitted calibrated model. If None, loads from registry.
        scaler: Fitted StandardScaler. If None, loads from registry.
        feature_names: Ordered feature list expected by the model.

    Returns:
        Dict with default_probability, risk_score, risk_band, top_drivers.
    """
    if model is None:
        model = load_artifact("xgb_calibrated")
    if scaler is None:
        scaler = load_artifact("scaler")
    if feature_names is None:
        feature_names = load_artifact("feature_names")

    # Build feature row
    from src.features.feature_store import add_derived_features
    from src.processing.preprocess import apply_scaler, clean_raw

    df = pd.DataFrame([features])
    df = clean_raw(df)
    df = add_derived_features(df)

    # Align to expected feature columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    df_scaled = apply_scaler(df, scaler)

    proba = float(model.predict_proba(df_scaled)[0, 1])
    result = build_score_output(proba)

    # Attach SHAP drivers
    try:
        from src.explainability.shap_analysis import explain_single

        drivers = explain_single(model, df_scaled, feature_names)
        result["top_drivers"] = drivers
    except Exception as exc:
        logger.debug("SHAP explanation skipped: %s", exc)
        result["top_drivers"] = []

    return result


def predict_batch(
    df_raw: pd.DataFrame,
    model=None,
    scaler=None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score a batch of records.

    Returns the input DataFrame with added columns:
        default_probability, risk_score, risk_band
    """
    if model is None:
        model = load_artifact("xgb_calibrated")
    if scaler is None:
        scaler = load_artifact("scaler")
    if feature_names is None:
        feature_names = load_artifact("feature_names")

    from src.features.feature_store import add_derived_features
    from src.processing.preprocess import apply_scaler, clean_raw

    df = clean_raw(df_raw.copy())
    df = add_derived_features(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df_feat = df[feature_names]
    df_scaled = apply_scaler(df_feat, scaler)

    probas = model.predict_proba(df_scaled)[:, 1]
    results = build_score_output(probas)

    out = df_raw.copy()
    out["default_probability"] = [r["default_probability"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["risk_band"] = [r["risk_band"] for r in results]

    logger.info("Batch scored: %d records.", len(out))
    return out
