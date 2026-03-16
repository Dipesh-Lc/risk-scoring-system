"""
Preprocessing pipeline: cleans and transforms the raw credit dataset.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_MODELS, ensure_dirs

logger = get_logger(__name__)

TARGET = "default_payment_next_month"

# Categorical columns -- kept as-is (already numeric codes in this dataset)
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Payment status: ordinal numeric
PAYMENT_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERIC_COLS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop ID if present
    - Remap undocumented EDUCATION values (0, 5, 6) -> 4 (Other)
    - Remap undocumented MARRIAGE value 0 -> 3 (Other)
    - Clip extreme bill amounts
    """
    df = df.copy()
    df = df.drop(columns=["ID"], errors="ignore")

    # EDUCATION: values 0, 5, 6 are undocumented -> map to 4 (Other)
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: value 0 is undocumented -> map to 3 (Other)
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    logger.debug("Raw data cleaned: %d rows", len(df))
    return df


def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on numeric + payment-status columns."""
    scale_cols = NUMERIC_COLS + PAYMENT_STATUS_COLS
    present = [c for c in scale_cols if c in X_train.columns]
    scaler = StandardScaler()
    scaler.fit(X_train[present])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to the numeric columns."""
    X = X.copy()
    scale_cols = NUMERIC_COLS + PAYMENT_STATUS_COLS
    present = [c for c in scale_cols if c in X.columns]
    X[present] = scaler.transform(X[present])
    return X


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (everything except target)."""
    return [c for c in df.columns if c != TARGET]


def save_scaler(scaler: StandardScaler, path=None) -> None:
    ensure_dirs()
    path = path or ARTIFACTS_MODELS / "scaler.joblib"
    joblib.dump(scaler, path)
    logger.info("Scaler saved -> %s", path)


def load_scaler(path=None) -> StandardScaler:
    path = path or ARTIFACTS_MODELS / "scaler.joblib"
    return joblib.load(path)


def preprocess_dataframe(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False):
    """
    Full preprocessing for a raw DataFrame.

    Args:
        df: Raw input DataFrame (may include target column).
        scaler: Pre-fitted StandardScaler. If None and fit=True, one is created.
        fit: If True, fit a new scaler on this data.

    Returns:
        (X_processed, y, scaler) -- y is None if target not present.
    """
    df = clean_raw(df)

    y = df[TARGET].values if TARGET in df.columns else None
    X = df.drop(columns=[TARGET], errors="ignore")

    if fit:
        scaler = build_scaler(X)
    if scaler is not None:
        X = apply_scaler(X, scaler)

    return X, y, scaler
