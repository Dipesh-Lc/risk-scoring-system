"""
Feature engineering: derives business-oriented signals from the raw columns.

Derived features
----------------
util_ratio_1      BILL_AMT1 / LIMIT_BAL             -- current utilisation
util_ratio_avg    mean(BILL_AMT1..6) / LIMIT_BAL    -- average utilisation
pay_ratio_1       PAY_AMT1 / (BILL_AMT1 + 1)        -- latest payment ratio
pay_ratio_avg     mean(PAY_AMT) / (mean(BILL_AMT)+1) -- average payment ratio
delinquency_count number of months with PAY_i > 0   -- total late payments
max_delinquency   max of PAY_0..6                   -- worst delinquency
bill_trend        linear slope of BILL_AMT1..6      -- increasing/falling balance
pay_trend         linear slope of PAY_AMT1..6       -- increasing/falling payments
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Months index for trend calculation (most recent = 1, oldest = 6)
_MONTHS = np.array([1, 2, 3, 4, 5, 6], dtype=float)
_MONTHS_CENTERED = _MONTHS - _MONTHS.mean()


def _linear_slope(matrix: np.ndarray) -> np.ndarray:
    """Compute row-wise linear slope via least-squares projection."""
    # slope = cov(x, y) / var(x) — vectorised
    denom = (_MONTHS_CENTERED**2).sum()
    return (matrix * _MONTHS_CENTERED).sum(axis=1) / denom


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the DataFrame. Returns new DataFrame."""
    df = df.copy()

    limit = df["LIMIT_BAL"].values.astype(float) + 1e-6  # avoid /0

    bill_matrix = df[BILL_COLS].values.astype(float)
    pay_matrix = df[PAY_AMT_COLS].values.astype(float)
    pay_status = df[PAY_STATUS_COLS].values.astype(float)

    # Utilisation
    df["util_ratio_1"] = bill_matrix[:, 0] / limit
    df["util_ratio_avg"] = bill_matrix.mean(axis=1) / limit

    # Payment coverage ratios -- guard against zero/negative bill amounts
    df["pay_ratio_1"] = pay_matrix[:, 0] / (np.abs(bill_matrix[:, 0]) + 1)
    df["pay_ratio_avg"] = pay_matrix.mean(axis=1) / (np.abs(bill_matrix.mean(axis=1)) + 1)

    # Delinquency
    df["delinquency_count"] = (pay_status > 0).sum(axis=1)
    df["max_delinquency"] = pay_status.max(axis=1)

    # Trends (positive slope = growing balance / payments)
    df["bill_trend"] = _linear_slope(bill_matrix)
    df["pay_trend"] = _linear_slope(pay_matrix)

    # Clip extreme ratios and replace any remaining NaN/inf with 0
    for col in ["util_ratio_1", "util_ratio_avg", "pay_ratio_1", "pay_ratio_avg"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 10)

    logger.debug("Derived features added. Total columns: %d", len(df.columns))
    return df


def get_all_feature_names(
    df: pd.DataFrame, target_col: str = "default_payment_next_month"
) -> list[str]:
    """Return all feature column names (excluding target)."""
    return [c for c in df.columns if c != target_col]
