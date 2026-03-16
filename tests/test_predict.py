"""Tests for prediction pipeline (single + batch)."""

from __future__ import annotations

class TestPredictSingle:
    def test_predict_single_returns_required_keys(
        self, calibrated_xgb, scaler_and_scaled, split_data, sample_record
    ):
        from src.models.predict import predict_single

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        result = predict_single(
            features=sample_record,
            model=calibrated_xgb,
            scaler=scaler,
            feature_names=feature_names,
        )
        assert "default_probability" in result
        assert "risk_score" in result
        assert "risk_band" in result
        assert "top_drivers" in result

    def test_predict_single_probability_in_range(
        self, calibrated_xgb, scaler_and_scaled, split_data, sample_record
    ):
        from src.models.predict import predict_single

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        result = predict_single(
            features=sample_record,
            model=calibrated_xgb,
            scaler=scaler,
            feature_names=feature_names,
        )
        assert 0.0 <= result["default_probability"] <= 1.0

    def test_predict_single_score_in_range(
        self, calibrated_xgb, scaler_and_scaled, split_data, sample_record
    ):
        from src.models.predict import predict_single

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        result = predict_single(
            features=sample_record,
            model=calibrated_xgb,
            scaler=scaler,
            feature_names=feature_names,
        )
        assert 0 <= result["risk_score"] <= 100

    def test_predict_single_band_valid(
        self, calibrated_xgb, scaler_and_scaled, split_data, sample_record
    ):
        from src.models.predict import predict_single

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        result = predict_single(
            features=sample_record,
            model=calibrated_xgb,
            scaler=scaler,
            feature_names=feature_names,
        )
        assert result["risk_band"] in ("Low", "Medium", "High")

    def test_high_delinquency_has_higher_score(
        self, calibrated_xgb, scaler_and_scaled, split_data, sample_record
    ):
        """Customer with delinquency should get a higher risk score."""
        from src.models.predict import predict_single

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        risky_record = dict(sample_record)
        risky_record["PAY_0"] = 3
        risky_record["PAY_2"] = 3
        risky_record["PAY_3"] = 3

        normal = predict_single(sample_record, calibrated_xgb, scaler, feature_names)
        risky = predict_single(risky_record, calibrated_xgb, scaler, feature_names)

        assert (
            risky["risk_score"] >= normal["risk_score"]
        ), "Delinquent customer should have equal or higher score than on-time customer"


class TestPredictBatch:
    def test_batch_predict_output_columns(
        self, calibrated_xgb, scaler_and_scaled, split_data, raw_df
    ):
        from src.models.predict import predict_batch

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        batch = raw_df.drop(columns=["default_payment_next_month"]).head(10)
        result = predict_batch(
            batch, model=calibrated_xgb, scaler=scaler, feature_names=feature_names
        )

        assert "default_probability" in result.columns
        assert "risk_score" in result.columns
        assert "risk_band" in result.columns

    def test_batch_predict_row_count(self, calibrated_xgb, scaler_and_scaled, split_data, raw_df):
        from src.models.predict import predict_batch

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        n = 15
        batch = raw_df.drop(columns=["default_payment_next_month"]).head(n)
        result = predict_batch(
            batch, model=calibrated_xgb, scaler=scaler, feature_names=feature_names
        )
        assert len(result) == n

    def test_batch_scores_in_range(self, calibrated_xgb, scaler_and_scaled, split_data, raw_df):
        from src.models.predict import predict_batch

        scaler, *_ = scaler_and_scaled
        _, _, _, _, _, _, feature_names = split_data

        batch = raw_df.drop(columns=["default_payment_next_month"]).head(20)
        result = predict_batch(
            batch, model=calibrated_xgb, scaler=scaler, feature_names=feature_names
        )

        assert result["default_probability"].between(0, 1).all()
        assert result["risk_score"].between(0, 100).all()
        assert result["risk_band"].isin(["Low", "Medium", "High"]).all()
