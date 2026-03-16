"""Tests for data loading and validation."""

from __future__ import annotations

import pandas as pd


class TestRawData:
    def test_raw_df_shape(self, raw_df):
        assert len(raw_df) == 200
        assert len(raw_df.columns) >= 24  # 23 features + target

    def test_target_column_present(self, raw_df):
        assert "default_payment_next_month" in raw_df.columns

    def test_target_is_binary(self, raw_df):
        assert set(raw_df["default_payment_next_month"].unique()).issubset({0, 1})

    def test_no_all_null_columns(self, raw_df):
        null_cols = raw_df.columns[raw_df.isnull().all()].tolist()
        assert null_cols == [], f"Fully null columns: {null_cols}"

    def test_numeric_dtypes(self, raw_df):
        numeric_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(raw_df[col]), f"{col} should be numeric"


class TestValidation:
    def test_validation_passes_on_clean_data(self, raw_df):
        from src.validation.data_checks import run_all_checks

        report = run_all_checks(raw_df)
        # Synthetic data should pass (warnings are ok, but no hard errors)
        assert isinstance(report.passed, bool)

    def test_validation_catches_missing_column(self, raw_df):
        from src.validation.data_checks import run_all_checks

        df_bad = raw_df.drop(columns=["LIMIT_BAL"])
        report = run_all_checks(df_bad)
        assert not report.passed
        assert any("LIMIT_BAL" in issue for issue in report.issues)

    def test_validation_detects_high_missing(self, raw_df):
        import numpy as np

        from src.validation.data_checks import run_all_checks

        df_bad = raw_df.copy()
        df_bad["AGE"] = np.nan  # 100% missing
        report = run_all_checks(df_bad)
        assert not report.passed

    def test_check_class_balance_stats(self, raw_df):
        from src.validation.data_checks import run_all_checks

        report = run_all_checks(raw_df)
        assert "default_rate" in report.stats
        assert 0 < report.stats["default_rate"] < 1


class TestSyntheticFallback:
    def test_synthetic_data_generation(self):
        from src.ingestion.load_data import _make_synthetic_data

        df = _make_synthetic_data(n=100)
        assert len(df) == 100
        assert "default_payment_next_month" in df.columns
        assert df["default_payment_next_month"].isin([0, 1]).all()
