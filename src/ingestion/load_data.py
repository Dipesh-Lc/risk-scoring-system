"""
Data ingestion: downloads the UCI Default of Credit Card Clients dataset.

Run as a module:
    python -m src.ingestion.load_data
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import DATA_RAW, DATA_SAMPLES, ensure_dirs

logger = get_logger(__name__)

RAW_CSV = DATA_RAW / "credit_default.csv"
SAMPLE_CSV = DATA_SAMPLES / "sample_customers.csv"

# Column rename map: normalise the target column name
RENAME = {"default.payment.next.month": "default_payment_next_month"}


def download_dataset() -> pd.DataFrame:
    """
    Download the UCI Default of Credit Card Clients dataset via ucimlrepo.
    Falls back to a synthetic minimal dataset if the download fails (CI/offline).
    """
    ensure_dirs()

    if RAW_CSV.exists():
        logger.info("Raw data already exists at %s -- skipping download.", RAW_CSV)
        df = pd.read_csv(RAW_CSV)
        if "X1" in df.columns or "Y" in df.columns:
            logger.info("Remapping generic column names in existing CSV.")
            df = _remap_generic_columns(df)
            df = df.rename(columns=RENAME)
            df.to_csv(RAW_CSV, index=False)
        return df

    logger.info("Attempting to download UCI dataset via ucimlrepo …")
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore

        dataset = fetch_ucirepo(id=350)  # id=350 -> Default of Credit Card Clients
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        # ucimlrepo sometimes returns generic names (X1..X23, Y) instead of
        # the real UCI column names. Detect and remap them.
        if "X1" in df.columns or "Y" in df.columns:
            logger.info("Generic column names detected (X1..X23, Y) - remapping to UCI names.")
            df = _remap_generic_columns(df)

        # Normalise the target column name regardless of source format
        df = df.rename(columns=RENAME)
        df.to_csv(RAW_CSV, index=False)
        logger.info("Dataset downloaded: %d rows x %d cols -> %s", *df.shape, RAW_CSV)
    except Exception as exc:
        logger.warning("ucimlrepo download failed (%s). Generating synthetic data.", exc)
        df = _make_synthetic_data(n=1000)
        df.to_csv(RAW_CSV, index=False)
        logger.info("Synthetic fallback data saved -> %s", RAW_CSV)

    return df


def _remap_generic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ucimlrepo sometimes returns columns as X1..X23 + Y instead of the real
    UCI field names. This maps them back to the correct names in order.

    Official UCI column order:
      X1=LIMIT_BAL, X2=SEX, X3=EDUCATION, X4=MARRIAGE, X5=AGE,
      X6=PAY_0, X7=PAY_2, X8=PAY_3, X9=PAY_4, X10=PAY_5, X11=PAY_6,
      X12=BILL_AMT1..X17=BILL_AMT6,
      X18=PAY_AMT1..X23=PAY_AMT6,
      Y=default.payment.next.month
    """
    col_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "default_payment_next_month",
    }
    # Only rename columns that are actually present
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=rename)


def _make_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the UCI schema.
    Used for offline testing / CI environments.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    pay_vals = rng.choice([-1, 0, 1, 2, 3], size=(n, 6), p=[0.35, 0.40, 0.10, 0.10, 0.05])

    df = pd.DataFrame(
        {
            "ID": range(1, n + 1),
            "LIMIT_BAL": rng.integers(10_000, 500_001, size=n),
            "SEX": rng.choice([1, 2], size=n),
            "EDUCATION": rng.choice([1, 2, 3, 4], size=n, p=[0.15, 0.47, 0.35, 0.03]),
            "MARRIAGE": rng.choice([1, 2, 3], size=n, p=[0.45, 0.53, 0.02]),
            "AGE": rng.integers(21, 75, size=n),
            "PAY_0": pay_vals[:, 0],
            "PAY_2": pay_vals[:, 1],
            "PAY_3": pay_vals[:, 2],
            "PAY_4": pay_vals[:, 3],
            "PAY_5": pay_vals[:, 4],
            "PAY_6": pay_vals[:, 5],
        }
    )

    limit = df["LIMIT_BAL"].values
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = (rng.uniform(0.1, 0.9, size=n) * limit).astype(int)
        df[f"PAY_AMT{i}"] = rng.integers(0, 5001, size=n)

    # Target: roughly 22% default rate, correlated with delinquency
    max_pay = pay_vals.max(axis=1)
    prob = 0.05 + 0.15 * (max_pay > 0) + 0.25 * (max_pay > 1)
    prob = prob.clip(0, 0.95)
    df["default_payment_next_month"] = rng.binomial(1, prob).astype(int)

    return df


def load_raw() -> pd.DataFrame:
    """Load raw CSV into a DataFrame (downloads if missing)."""
    if not RAW_CSV.exists():
        download_dataset()
    df = pd.read_csv(RAW_CSV)

    # Handle case where CSV was previously saved with generic X1..X23, Y names
    if "X1" in df.columns or "Y" in df.columns:
        logger.info("Remapping generic column names in existing CSV.")
        df = _remap_generic_columns(df)
        # Overwrite the CSV with correct names so this only runs once
        df.to_csv(RAW_CSV, index=False)

    df = df.rename(columns=RENAME)
    # Drop the ID column if present - it's not a feature
    df = df.drop(columns=["ID"], errors="ignore")
    logger.info("Loaded raw data: %d rows x %d cols", *df.shape)
    return df


def save_sample(df: pd.DataFrame, n: int = 50) -> None:
    """Save a small sample for demo / batch scoring."""
    ensure_dirs()
    sample = df.drop(columns=["default_payment_next_month"], errors="ignore").head(n)
    sample.to_csv(SAMPLE_CSV, index=False)
    logger.info("Sample customers saved -> %s (%d rows)", SAMPLE_CSV, n)


if __name__ == "__main__":
    df = download_dataset()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Default rate: {df['default_payment_next_month'].mean():.2%}")
    save_sample(df)
