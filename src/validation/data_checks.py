"""
Data validation: checks raw data quality before preprocessing.

Run as a module:
    python -m src.validation.data_checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected schema
EXPECTED_COLUMNS = [
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
    "default_payment_next_month",
]

VALID_RANGES: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (0, 1_000_000),
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
    "default_payment_next_month": (0, 1),
}


@dataclass
class ValidationReport:
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False
        logger.error("Validation issue: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("Validation warning: %s", msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"=== Data Validation: {status} ==="]
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            lines.extend(f"  [FAIL] {i}" for i in self.issues)
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f" [WARN]  {w}" for w in self.warnings)
        if not self.issues and not self.warnings:
            lines.append("  All checks passed.")
        return "\n".join(lines)


def check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check that all expected columns are present."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS and c != "ID"]
    if missing:
        report.add_issue(f"Missing columns: {missing}")
    if extra:
        report.add_warning(f"Unexpected extra columns: {extra}")
    report.stats["n_rows"] = len(df)
    report.stats["n_cols"] = len(df.columns)


def check_missing_values(
    df: pd.DataFrame, report: ValidationReport, threshold: float = 0.05
) -> None:
    """Flag columns with more than `threshold` missing values."""
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            report.add_issue(
                f"Column '{col}' has {pct:.1%} missing values (threshold={threshold:.0%})"
            )
    any_missing = missing_pct[missing_pct > 0]
    if not any_missing.empty:
        report.add_warning(f"Columns with any missing values: {list(any_missing.index)}")
    report.stats["total_missing"] = int(df.isnull().sum().sum())
    report.stats["missing_pct"] = float(df.isnull().mean().mean())


def check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """Flag duplicate rows."""
    n_dupes = df.duplicated().sum()
    report.stats["n_duplicates"] = int(n_dupes)
    if n_dupes > 0:
        report.add_warning(f"{n_dupes} duplicate rows found.")


def check_value_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check that numeric columns fall within expected ranges."""
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            pct = out_of_range / len(df)
            report.add_warning(
                f"Column '{col}': {out_of_range} ({pct:.1%}) values outside [{lo}, {hi}]"
            )


def check_class_balance(df: pd.DataFrame, report: ValidationReport) -> None:
    """Report target class distribution."""
    target = "default_payment_next_month"
    if target not in df.columns:
        return
    counts = df[target].value_counts()
    default_rate = df[target].mean()
    report.stats["default_rate"] = float(default_rate)
    report.stats["class_counts"] = counts.to_dict()
    if default_rate < 0.05 or default_rate > 0.50:
        report.add_warning(
            f"Unusual class balance: default rate = {default_rate:.1%}. "
            "Consider class weighting or resampling."
        )
    else:
        logger.info("Default rate: %.1f%%", default_rate * 100)


def check_bill_payment_consistency(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check that bill and payment amounts are non-negative."""
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    for col in pay_cols:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                report.add_warning(f"Column '{col}': {n_neg} negative values (expected >= 0)")
    # BILL_AMT can be negative (credit balance) -- just log
    for col in bill_cols:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                logger.info(
                    "Column '%s': %d negative bill amounts (credit balance - ok)", col, n_neg
                )


def run_all_checks(df: pd.DataFrame) -> ValidationReport:
    """Run all validation checks and return a report."""
    report = ValidationReport()
    logger.info("Running data validation on %d rows …", len(df))

    check_schema(df, report)
    check_missing_values(df, report)
    check_duplicates(df, report)
    check_value_ranges(df, report)
    check_class_balance(df, report)
    check_bill_payment_consistency(df, report)

    logger.info(report.summary())
    return report


if __name__ == "__main__":
    from src.ingestion.load_data import load_raw

    df = load_raw()
    report = run_all_checks(df)
    print(report.summary())
    print("\nStats:", report.stats)
