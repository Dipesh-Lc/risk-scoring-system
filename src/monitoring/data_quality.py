"""
Input data quality checks for scoring requests (production inference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Valid ranges for scoring-time input validation
SCORING_RANGES: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (0, 1_500_000),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    "AGE": (18, 100),
    "PAY_0": (-2, 9),
    "PAY_2": (-2, 9),
    "PAY_3": (-2, 9),
    "PAY_4": (-2, 9),
    "PAY_5": (-2, 9),
    "PAY_6": (-2, 9),
}

REQUIRED_FIELDS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


@dataclass
class QualityResult:
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def flag_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def flag_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def check_single_record(record: dict[str, Any]) -> QualityResult:
    """
    Validate a single scoring request record.
    Returns a QualityResult with .valid, .warnings, .errors.
    """
    result = QualityResult()

    # Missing fields
    missing = [f for f in REQUIRED_FIELDS if f not in record or record[f] is None]
    if missing:
        result.flag_error(f"Missing required fields: {missing}")

    # Nulls / NaN
    null_fields = [
        f for f, v in record.items() if v is not None and isinstance(v, float) and np.isnan(v)
    ]
    if null_fields:
        result.flag_error(f"NaN values in fields: {null_fields}")

    # Range checks
    for field_name, (lo, hi) in SCORING_RANGES.items():
        if field_name in record and record[field_name] is not None:
            val = record[field_name]
            if not (lo <= val <= hi):
                result.flag_warning(
                    f"Field '{field_name}' value {val} outside expected range [{lo}, {hi}]"
                )

    # Payment amounts should be non-negative
    for i in range(1, 7):
        key = f"PAY_AMT{i}"
        if key in record and record[key] is not None and record[key] < 0:
            result.flag_warning(f"'{key}' is negative ({record[key]}); expected >= 0")

    return result


def check_batch(df: pd.DataFrame, max_missing_pct: float = 0.05) -> QualityResult:
    """
    Validate a batch DataFrame before scoring.
    """
    result = QualityResult()

    # Missing columns
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing_cols:
        result.flag_error(f"Missing columns: {missing_cols}")
        return result  # Can't proceed

    # Missing values
    missing_pct = df[REQUIRED_FIELDS].isnull().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct]
    for col, pct in high_missing.items():
        result.flag_error(f"Column '{col}' has {pct:.1%} missing values in batch")

    # Inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        result.flag_error(f"Inf values in columns: {inf_cols}")

    return result
