"""
Train / validation / test splitting with stratification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "default_payment_next_month"


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / validation / test sets.

    Returns:
        (df_train, df_val, df_test)
    """
    y = df[TARGET]

    # First split off test set
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split val from trainval
    val_relative = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=df_trainval[TARGET],
    )

    logger.info(
        "Split -> train: %d | val: %d | test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        rate = split[TARGET].mean()
        logger.info("  %s default rate: %.2f%%", name, rate * 100)

    # Reset index so downstream code gets clean 0-based integer indices
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a DataFrame into features and target array."""
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    return X, y
