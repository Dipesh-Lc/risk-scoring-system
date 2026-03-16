"""
End-to-end training pipeline.

Steps:
1. Load raw data
2. Validate
3. Clean + feature engineering
4. Split (train / val / test)
5. Fit scaler
6. Train baseline (Logistic Regression)
7. Train XGBoost
8. Calibrate XGBoost
9. Evaluate all three on test set
10. Generate global SHAP
11. Save all artifacts + metrics
12. Generate model report

Run:
    python -m src.pipelines.train_pipeline
"""

from __future__ import annotations

import time


from src.utils.logger import get_logger
from src.utils.paths import DATA_PROCESSED, ensure_dirs

logger = get_logger(__name__)


def run() -> None:
    start = time.time()
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # 1. Load
    # ---------------------------------------------------------------------------------
    from src.ingestion.load_data import load_raw, save_sample

    df_raw = load_raw()
    save_sample(df_raw)

    # 2. Validate
    # ---------------------------------------------------------------------------------
    from src.validation.data_checks import run_all_checks

    report = run_all_checks(df_raw)
    if not report.passed:
        logger.error("Data validation FAILED. Aborting pipeline.")
        raise SystemExit(1)

    # 3. Clean + feature engineering
    # ---------------------------------------------------------------------------------
    from src.features.feature_store import add_derived_features
    from src.processing.preprocess import clean_raw

    df_clean = clean_raw(df_raw)
    df_feat = add_derived_features(df_clean)
    logger.info("Features after engineering: %d cols", df_feat.shape[1])

    # 4. Split
    # ---------------------------------------------------------------------------------
    from src.processing.split import get_X_y, split_dataset

    df_train, df_val, df_test = split_dataset(df_feat)
    X_train_raw, y_train = get_X_y(df_train)
    X_val_raw, y_val = get_X_y(df_val)
    X_test_raw, y_test = get_X_y(df_test)

    feature_names = list(X_train_raw.columns)
    logger.info("Feature count: %d", len(feature_names))

    # 5. Fit scaler
    # ---------------------------------------------------------------------------------
    from src.processing.preprocess import apply_scaler, build_scaler, save_scaler

    scaler = build_scaler(X_train_raw)
    X_train = apply_scaler(X_train_raw, scaler)
    X_val = apply_scaler(X_val_raw, scaler)
    X_test = apply_scaler(X_test_raw, scaler)
    save_scaler(scaler)

    # Save processed data for monitoring reference
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    try:
        X_train_raw.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    except ImportError:
        X_train_raw.to_csv(DATA_PROCESSED / "X_train.csv", index=False)

    # 6. Baseline model
    # ---------------------------------------------------------------------------------
    logger.info("Training baseline (Logistic Regression) …")
    from src.models.evaluate import compute_metrics, save_metrics
    from src.models.registry import save_artifact
    from src.models.train import train_baseline

    baseline = train_baseline(X_train, y_train)
    bl_proba = baseline.predict_proba(X_test)[:, 1]
    bl_metrics = compute_metrics(y_test, bl_proba, name="Logistic Regression")
    save_metrics(bl_metrics, "baseline_metrics")
    save_artifact(baseline, "baseline_lr")

    # 7. XGBoost
    # ---------------------------------------------------------------------------------
    logger.info("Training XGBoost …")
    from src.models.train import train_xgboost

    xgb = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba, name="XGBoost (raw)")
    save_metrics(xgb_metrics, "xgb_metrics")
    save_artifact(xgb, "xgb_raw")

    # 8. Calibration
    # ---------------------------------------------------------------------------------
    logger.info("Calibrating XGBoost …")
    from src.models.calibrate import calibrate_model

    xgb_cal = calibrate_model(xgb, X_val, y_val, method="isotonic")
    cal_proba = xgb_cal.predict_proba(X_test)[:, 1]
    cal_metrics = compute_metrics(y_test, cal_proba, name="XGBoost (calibrated)")
    save_metrics(cal_metrics, "xgb_calibrated_metrics")
    save_artifact(xgb_cal, "xgb_calibrated")
    save_artifact(feature_names, "feature_names")

    # 9. Comparison table
    # ---------------------------------------------------------------------------------
    from src.models.evaluate import compare_models

    comparison = compare_models([bl_metrics, xgb_metrics, cal_metrics])
    logger.info("\n%s", comparison.to_string())
    comparison.to_csv(DATA_PROCESSED.parent.parent / "artifacts" / "metrics" / "comparison.csv")

    # 10. Plots
    # ---------------------------------------------------------------------------------
    try:
        from src.models.evaluate import plot_calibration_curve, plot_roc_curve
        from src.utils.paths import REPORTS

        plot_roc_curve(
            y_test,
            {
                "Logistic Regression": bl_proba,
                "XGBoost (raw)": xgb_proba,
                "XGBoost (calibrated)": cal_proba,
            },
            save_path=REPORTS / "figures" / "roc_curves.png",
        )
        plot_calibration_curve(
            y_test,
            {
                "XGBoost (raw)": xgb_proba,
                "XGBoost (calibrated)": cal_proba,
            },
            save_path=REPORTS / "figures" / "calibration_curves.png",
        )
    except Exception as e:
        logger.warning("Could not generate plots: %s", e)

    # 11. SHAP
    # ---------------------------------------------------------------------------------
    logger.info("Computing SHAP global importance …")
    try:
        from src.explainability.shap_analysis import global_feature_importance, plot_shap_summary

        importance = global_feature_importance(xgb_cal, X_test, feature_names)
        plot_shap_summary(xgb_cal, X_test, feature_names)
    except Exception as e:
        logger.warning("SHAP failed (non-fatal): %s", e)

    # 12. Report
    # ---------------------------------------------------------------------------------
    from src.monitoring.reporting import generate_model_report

    generate_model_report()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds.", elapsed)
    logger.info(
        "Calibrated XGBoost -> ROC-AUC: %.4f | Brier: %.4f",
        cal_metrics["roc_auc"],
        cal_metrics["brier_score"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
