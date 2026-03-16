"""Tests for monitoring: data quality and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestDataQuality:
    def test_valid_record_passes(self, sample_record):
        from src.monitoring.data_quality import check_single_record

        result = check_single_record(sample_record)
        assert result.valid

    def test_missing_field_fails(self, sample_record):
        from src.monitoring.data_quality import check_single_record

        bad = dict(sample_record)
        del bad["LIMIT_BAL"]
        result = check_single_record(bad)
        assert not result.valid
        assert any("LIMIT_BAL" in e for e in result.errors)

    def test_out_of_range_sex_warns(self, sample_record):
        from src.monitoring.data_quality import check_single_record

        bad = dict(sample_record, SEX=5)
        result = check_single_record(bad)
        assert any("SEX" in w for w in result.warnings)

    def test_negative_pay_amount_warns(self, sample_record):
        from src.monitoring.data_quality import check_single_record

        bad = dict(sample_record, PAY_AMT1=-100)
        result = check_single_record(bad)
        assert any("PAY_AMT1" in w for w in result.warnings)

    def test_batch_check_valid(self, raw_df):
        from src.monitoring.data_quality import check_batch

        df = raw_df.drop(columns=["default_payment_next_month"])
        result = check_batch(df)
        assert result.valid

    def test_batch_check_missing_column(self, raw_df):
        from src.monitoring.data_quality import check_batch

        df = raw_df.drop(columns=["default_payment_next_month", "LIMIT_BAL"])
        result = check_batch(df)
        assert not result.valid


class TestDriftDetection:
    def test_ks_test_no_drift(self):
        from src.monitoring.drift import ks_drift_test

        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 500))
        cur = pd.Series(rng.normal(0, 1, 500))  # same distribution
        result = ks_drift_test(ref, cur, threshold=0.1)

        assert "statistic" in result
        assert "drifted" in result
        # Same distribution should not drift at standard threshold
        assert not result["drifted"]

    def test_ks_test_detects_drift(self):
        from src.monitoring.drift import ks_drift_test

        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 500))
        cur = pd.Series(rng.normal(5, 1, 500))  # very different distribution
        result = ks_drift_test(ref, cur, threshold=0.1)
        assert result["drifted"]

    def test_psi_no_drift(self):
        from src.monitoring.drift import compute_psi

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi < 0.1  # No significant shift

    def test_psi_detects_drift(self):
        from src.monitoring.drift import compute_psi

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(10, 1, 1000)  # Large shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_drift_report_structure(self, raw_df):
        from src.monitoring.drift import run_drift_report

        df1 = raw_df.drop(columns=["default_payment_next_month"]).head(100)
        df2 = raw_df.drop(columns=["default_payment_next_month"]).tail(100)
        report = run_drift_report(df1, df2, save=False)

        assert "n_features_checked" in report
        assert "n_drifted" in report
        assert "drifted_features" in report
        assert "feature_results" in report
        assert report["n_features_checked"] > 0

    def test_score_distribution_check(self):
        from src.monitoring.drift import check_score_distribution

        rng = np.random.default_rng(0)
        ref_scores = rng.integers(0, 101, 500).astype(float)
        cur_scores = rng.integers(0, 101, 500).astype(float)  # Same distribution
        result = check_score_distribution(ref_scores, cur_scores)

        assert "score_psi" in result
        assert "score_shifted" in result
        assert not result["score_shifted"]
