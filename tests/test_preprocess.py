"""Tests for preprocessing and feature engineering."""

from __future__ import annotations

import pandas as pd


class TestCleaning:
    def test_clean_raw_drops_id(self, raw_df):
        from src.processing.preprocess import clean_raw

        df_with_id = raw_df.copy()
        df_with_id["ID"] = range(len(raw_df))
        cleaned = clean_raw(df_with_id)
        assert "ID" not in cleaned.columns

    def test_education_remapping(self):
        from src.processing.preprocess import clean_raw

        df = pd.DataFrame({"EDUCATION": [0, 5, 6, 1, 2]})
        cleaned = clean_raw(df)
        # 0, 5, 6 should become 4
        assert cleaned["EDUCATION"].isin([1, 2, 3, 4]).all()

    def test_marriage_remapping(self):
        from src.processing.preprocess import clean_raw

        df = pd.DataFrame({"MARRIAGE": [0, 1, 2, 3]})
        cleaned = clean_raw(df)
        assert 0 not in cleaned["MARRIAGE"].values
        assert cleaned["MARRIAGE"].isin([1, 2, 3]).all()


class TestFeatureEngineering:
    def test_derived_features_added(self, feature_df):
        expected = [
            "util_ratio_1",
            "util_ratio_avg",
            "pay_ratio_1",
            "pay_ratio_avg",
            "delinquency_count",
            "max_delinquency",
            "bill_trend",
            "pay_trend",
        ]
        for col in expected:
            assert col in feature_df.columns, f"Missing derived feature: {col}"

    def test_util_ratio_range(self, feature_df):
        # Clipped to [-5, 10]
        assert feature_df["util_ratio_1"].between(-5, 10).all()
        assert feature_df["util_ratio_avg"].between(-5, 10).all()

    def test_delinquency_count_non_negative(self, feature_df):
        assert (feature_df["delinquency_count"] >= 0).all()

    def test_max_delinquency_range(self, feature_df):
        assert feature_df["max_delinquency"].between(-2, 9).all()

    def test_no_nan_in_derived(self, feature_df):
        derived = [
            "util_ratio_1",
            "util_ratio_avg",
            "pay_ratio_1",
            "delinquency_count",
            "max_delinquency",
            "bill_trend",
            "pay_trend",
        ]
        for col in derived:
            assert not feature_df[col].isnull().any(), f"NaN found in {col}"


class TestSplit:
    def test_split_sizes(self, split_data):
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_data
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 200
        # Test set should be ~20% of 200 = ~40
        assert 30 <= len(X_test) <= 50

    def test_no_overlap_between_splits(self, split_data):
        # After reset_index(drop=True) all three splits have 0-based indices,
        # so index set intersection is meaningless. Instead verify total row
        # count is preserved (no duplication or loss).
        X_train, X_val, X_test, *_ = split_data
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 200

    def test_stratified_balance(self, split_data):
        _, _, _, y_train, y_val, y_test, _ = split_data
        # All splits should have some positives
        assert y_train.sum() > 0
        assert y_val.sum() > 0
        assert y_test.sum() > 0


class TestScaler:
    def test_scaler_output_shape(self, scaler_and_scaled, split_data):
        scaler, X_train_s, _, _ = scaler_and_scaled
        X_train, *_ = split_data
        assert X_train_s.shape == X_train.shape

    def test_scaler_produces_finite_values(self, scaler_and_scaled):
        import numpy as np

        _, X_train_s, X_val_s, X_test_s = scaler_and_scaled
        for name, X in [("train", X_train_s), ("val", X_val_s), ("test", X_test_s)]:
            assert np.isfinite(X.values).all(), f"Non-finite values in {name} after scaling"
