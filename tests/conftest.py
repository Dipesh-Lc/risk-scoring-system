"""
Shared pytest fixtures for the risk-scoring-system test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Minimal synthetic dataset

N_ROWS = 200
RANDOM_STATE = 42


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Small synthetic DataFrame matching the UCI schema."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = N_ROWS

    pay_vals = rng.choice([-1, 0, 1, 2], size=(n, 6), p=[0.4, 0.4, 0.1, 0.1])
    limit = rng.integers(10_000, 200_001, size=n).astype(float)

    df = pd.DataFrame(
        {
            "LIMIT_BAL": limit,
            "SEX": rng.choice([1, 2], size=n),
            "EDUCATION": rng.choice([1, 2, 3, 4], size=n),
            "MARRIAGE": rng.choice([1, 2, 3], size=n),
            "AGE": rng.integers(21, 65, size=n),
            "PAY_0": pay_vals[:, 0],
            "PAY_2": pay_vals[:, 1],
            "PAY_3": pay_vals[:, 2],
            "PAY_4": pay_vals[:, 3],
            "PAY_5": pay_vals[:, 4],
            "PAY_6": pay_vals[:, 5],
        }
    )

    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = (rng.uniform(0.05, 0.85, size=n) * limit).astype(int).astype(float)
        df[f"PAY_AMT{i}"] = rng.integers(0, 3001, size=n).astype(float)

    # Target: ~22% default
    max_pay = pay_vals.max(axis=1)
    prob = (0.05 + 0.2 * (max_pay > 0)).clip(0, 0.9)
    df["default_payment_next_month"] = rng.binomial(1, prob).astype(int)

    return df


@pytest.fixture(scope="session")
def feature_df(raw_df) -> pd.DataFrame:
    """DataFrame after clean + feature engineering."""
    from src.features.feature_store import add_derived_features
    from src.processing.preprocess import clean_raw

    df = clean_raw(raw_df)
    return add_derived_features(df)


@pytest.fixture(scope="session")
def split_data(feature_df):
    """Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)."""
    from src.processing.split import get_X_y, split_dataset

    df_train, df_val, df_test = split_dataset(feature_df, test_size=0.2, val_size=0.2)
    X_train, y_train = get_X_y(df_train)
    X_val, y_val = get_X_y(df_val)
    X_test, y_test = get_X_y(df_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, list(X_train.columns)


@pytest.fixture(scope="session")
def scaler_and_scaled(split_data):
    """Returns (scaler, X_train_scaled, X_val_scaled, X_test_scaled)."""
    from src.processing.preprocess import apply_scaler, build_scaler

    X_train, X_val, X_test, *_ = split_data
    scaler = build_scaler(X_train)
    return (
        scaler,
        apply_scaler(X_train, scaler),
        apply_scaler(X_val, scaler),
        apply_scaler(X_test, scaler),
    )


@pytest.fixture(scope="session")
def trained_lr(scaler_and_scaled, split_data):
    """A fitted Logistic Regression model."""
    from src.models.train import train_baseline

    (
        scaler,
        X_train_s,
        _,
        _,
    ) = scaler_and_scaled
    _, _, _, y_train, *_ = split_data
    return train_baseline(X_train_s, y_train)


@pytest.fixture(scope="session")
def trained_xgb(scaler_and_scaled, split_data):
    """A fitted gradient boosting model (fast: few estimators)."""
    from src.models.train import train_xgboost

    _, X_train_s, X_val_s, _ = scaler_and_scaled
    _, _, _, y_train, y_val, *_ = split_data
    return train_xgboost(
        X_train_s,
        y_train,
        X_val_s,
        y_val,
        params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1},
    )


@pytest.fixture(scope="session")
def calibrated_xgb(trained_xgb, scaler_and_scaled, split_data):
    """A calibrated XGBoost model."""
    from src.models.calibrate import calibrate_model

    _, _, X_val_s, _ = scaler_and_scaled
    _, _, _, _, y_val, *_ = split_data
    return calibrate_model(trained_xgb, X_val_s, y_val, method="isotonic")


@pytest.fixture
def sample_record() -> dict:
    """A single customer record dict."""
    return {
        "LIMIT_BAL": 80000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 20000,
        "BILL_AMT2": 18000,
        "BILL_AMT3": 16000,
        "BILL_AMT4": 14000,
        "BILL_AMT5": 12000,
        "BILL_AMT6": 10000,
        "PAY_AMT1": 3000,
        "PAY_AMT2": 2500,
        "PAY_AMT3": 2000,
        "PAY_AMT4": 1800,
        "PAY_AMT5": 1600,
        "PAY_AMT6": 1400,
    }
