"""
End-to-end verification script.
Runs the full pipeline and asserts correctness without requiring pytest.

Usage:
    python verify_pipeline.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure repo root is on the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

passed = []
failed = []


def ok(name: str) -> None:
    passed.append(name)
    print(f"  OK {name}")


def fail(name: str, exc: Exception) -> None:
    failed.append(name)
    print(f"  X {name}: {exc}")
    traceback.print_exc()


# 1. Synthetic data 
#---------------------------------------------------------------------------------
print("\n--- 1. Data ingestion ---")
try:
    from src.ingestion.load_data import _make_synthetic_data
    df_raw = _make_synthetic_data(n=500, seed=0)
    assert len(df_raw) == 500
    assert "default_payment_next_month" in df_raw.columns
    assert df_raw["default_payment_next_month"].isin([0, 1]).all()
    ok("synthetic data generation")
except Exception as e:
    fail("synthetic data generation", e)

# 2. Validation
#---------------------------------------------------------------------------------
print("\n--- 2. Data validation ---")
try:
    from src.validation.data_checks import run_all_checks
    report = run_all_checks(df_raw)
    assert isinstance(report.passed, bool)
    ok("validation report")
except Exception as e:
    fail("validation report", e)

try:
    from src.validation.data_checks import run_all_checks
    import pandas as pd
    bad_df = df_raw.drop(columns=["LIMIT_BAL"])
    report = run_all_checks(bad_df)
    assert not report.passed
    ok("validation catches missing column")
except Exception as e:
    fail("validation catches missing column", e)

# 3. Preprocessing 
#---------------------------------------------------------------------------------
print("\n--- 3. Preprocessing ---")
try:
    from src.processing.preprocess import clean_raw
    df_clean = clean_raw(df_raw)
    assert "ID" not in df_clean.columns
    # EDUCATION=0 → 4
    import pandas as pd
    test_ed = pd.DataFrame({"EDUCATION": [0, 5, 6, 1]})
    result = clean_raw(test_ed)
    assert result["EDUCATION"].isin([1, 2, 3, 4]).all()
    ok("clean_raw remapping")
except Exception as e:
    fail("clean_raw remapping", e)

# 4. Feature engineering 
#---------------------------------------------------------------------------------
print("\n--- 4. Feature engineering ---")
try:
    from src.features.feature_store import add_derived_features
    df_feat = add_derived_features(df_clean)
    expected = ["util_ratio_1", "util_ratio_avg", "pay_ratio_1", "pay_ratio_avg",
                "delinquency_count", "max_delinquency", "bill_trend", "pay_trend"]
    for col in expected:
        assert col in df_feat.columns, f"Missing: {col}"
    import numpy as np
    assert np.isfinite(df_feat[expected].values).all()
    ok("derived features: all present and finite")
except Exception as e:
    fail("derived features", e)

# 5. Split
#---------------------------------------------------------------------------------
print("\n--- 5. Split ---")
try:
    from src.processing.split import split_dataset, get_X_y

    df_train, df_val, df_test = split_dataset(df_feat, test_size=0.2, val_size=0.2)

    # Total rows preserved
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == len(df_feat), f"Row count mismatch: {total} != {len(df_feat)}"

    # All three splits non-empty
    assert len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0

    # Target present in each split
    assert "default_payment_next_month" in df_train.columns

    ok(f"split: train={len(df_train)} val={len(df_val)} test={len(df_test)}")
except Exception as e:
    fail("split", e)

X_train_raw, y_train = get_X_y(df_train)
X_val_raw,   y_val   = get_X_y(df_val)
X_test_raw,  y_test  = get_X_y(df_test)
feature_names = list(X_train_raw.columns)

# 6. Scaler 
#---------------------------------------------------------------------------------
print("\n--- 6. Scaler ---")
try:
    import numpy as np
    from src.processing.preprocess import build_scaler, apply_scaler
    scaler = build_scaler(X_train_raw)
    X_train = apply_scaler(X_train_raw, scaler)
    X_val   = apply_scaler(X_val_raw,   scaler)
    X_test  = apply_scaler(X_test_raw,  scaler)
    assert X_train.shape == X_train_raw.shape
    assert np.isfinite(X_train.values).all()
    ok("scaler: finite output, shape preserved")
except Exception as e:
    fail("scaler", e)

# 7. Baseline model 
#---------------------------------------------------------------------------------
print("\n--- 7. Baseline (Logistic Regression) ---")
try:
    from src.models.train import train_baseline
    lr = train_baseline(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    assert lr_proba.shape == (len(X_test),)
    assert (lr_proba >= 0).all() and (lr_proba <= 1).all()
    ok("LR: probabilities in [0, 1]")
except Exception as e:
    fail("LR training", e)

# 8. Evaluation 
#---------------------------------------------------------------------------------
print("\n--- 8. Evaluation ---")
try:
    from src.models.evaluate import compute_metrics
    metrics = compute_metrics(y_test, lr_proba, name="LR-verify")
    for key in ["roc_auc", "pr_auc", "brier_score", "f1"]:
        assert key in metrics
        assert 0 <= metrics[key] <= 1
    assert metrics["roc_auc"] > 0.5, "Model should beat random"
    ok(f"metrics: ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}")
except Exception as e:
    fail("evaluation", e)

# 9. Gradient boosting (XGBoost fallback)
#---------------------------------------------------------------------------------
print("\n--- 9. Gradient boosting model ---")
try:
    from src.models.train import train_xgboost
    gbm = train_xgboost(
        X_train, y_train, X_val, y_val,
        params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
    )
    gbm_proba = gbm.predict_proba(X_test)[:, 1]
    assert (gbm_proba >= 0).all() and (gbm_proba <= 1).all()
    ok(f"GBM: probabilities in [0, 1], type={type(gbm).__name__}")
except Exception as e:
    fail("GBM training", e)

# 10. Calibration
#---------------------------------------------------------------------------------
print("\n--- 10. Calibration ---")
try:
    from src.models.calibrate import calibrate_model
    from sklearn.metrics import brier_score_loss
    cal = calibrate_model(gbm, X_val, y_val, method="isotonic")
    cal_proba = cal.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test, gbm_proba)
    cal_brier = brier_score_loss(y_test, cal_proba)
    ok(f"calibration: raw_brier={raw_brier:.4f} → cal_brier={cal_brier:.4f}")
except Exception as e:
    fail("calibration", e)

# 11. Scoring logic 
#---------------------------------------------------------------------------------
print("\n--- 11. Scoring logic ---")
try:
    import numpy as np
    from src.models.calibrate import probability_to_score, score_to_band, build_score_output

    # Edge cases
    assert probability_to_score(0.0) == 0
    assert probability_to_score(1.0) == 100
    assert probability_to_score(0.5) == 50

    # Bands
    assert score_to_band(0)   == "Low"
    assert score_to_band(30)  == "Low"
    assert score_to_band(31)  == "Medium"
    assert score_to_band(60)  == "Medium"
    assert score_to_band(61)  == "High"
    assert score_to_band(100) == "High"

    # Array form
    scores = probability_to_score(np.linspace(0, 1, 101))
    assert (scores >= 0).all() and (scores <= 100).all()

    # Full output dict
    out = build_score_output(0.42)
    assert out["risk_band"] == "Medium"
    assert out["risk_score"] == 42

    ok("probability_to_score / score_to_band / build_score_output")
except Exception as e:
    fail("scoring logic", e)

# 12. Registry save/load 
#---------------------------------------------------------------------------------
print("\n--- 12. Model registry ---")
try:
    from src.models.registry import save_artifact, load_artifact, list_artifacts
    from src.utils.paths import ensure_dirs
    ensure_dirs()

    save_artifact(cal, "verify_model")
    save_artifact(scaler, "verify_scaler")
    save_artifact(feature_names, "verify_features")

    loaded_model    = load_artifact("verify_model")
    loaded_scaler   = load_artifact("verify_scaler")
    loaded_features = load_artifact("verify_features")

    assert loaded_features == feature_names
    proba_check = loaded_model.predict_proba(
        apply_scaler(X_test_raw, loaded_scaler)
    )[:, 1]
    assert (proba_check >= 0).all() and (proba_check <= 1).all()
    ok("save/load model artifacts round-trip")
except Exception as e:
    fail("model registry", e)

# 13. Predict single 
#---------------------------------------------------------------------------------
print("\n--- 13. Single-record prediction ---")
try:
    from src.models.predict import predict_single

    sample = {
        "LIMIT_BAL": 80000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 20000, "BILL_AMT2": 18000, "BILL_AMT3": 16000,
        "BILL_AMT4": 14000, "BILL_AMT5": 12000, "BILL_AMT6": 10000,
        "PAY_AMT1": 3000,  "PAY_AMT2": 2500,  "PAY_AMT3": 2000,
        "PAY_AMT4": 1800,  "PAY_AMT5": 1600,  "PAY_AMT6": 1400,
    }
    result = predict_single(sample, model=cal, scaler=scaler, feature_names=feature_names)
    assert "default_probability" in result
    assert "risk_score" in result
    assert "risk_band" in result
    assert 0 <= result["default_probability"] <= 1
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_band"] in ("Low", "Medium", "High")
    ok(f"predict_single: score={result['risk_score']}, band={result['risk_band']}")
except Exception as e:
    fail("predict_single", e)

# 14. Predict batch 
#---------------------------------------------------------------------------------
print("\n--- 14. Batch prediction ---")
try:
    import pandas as pd
    from src.models.predict import predict_batch

    batch = df_raw.drop(columns=["default_payment_next_month"]).head(20)
    scored = predict_batch(batch, model=cal, scaler=scaler, feature_names=feature_names)
    assert len(scored) == 20
    assert "default_probability" in scored.columns
    assert scored["default_probability"].between(0, 1).all()
    assert scored["risk_score"].between(0, 100).all()
    assert scored["risk_band"].isin(["Low", "Medium", "High"]).all()
    ok(f"predict_batch: {len(scored)} records, bands: {scored['risk_band'].value_counts().to_dict()}")
except Exception as e:
    fail("predict_batch", e)

# 15. Data quality monitoring 
#---------------------------------------------------------------------------------
print("\n--- 15. Data quality monitoring ---")
try:
    from src.monitoring.data_quality import check_single_record, check_batch

    valid_rec = dict(sample)
    assert check_single_record(valid_rec).valid

    bad_rec = {k: v for k, v in sample.items() if k != "LIMIT_BAL"}
    assert not check_single_record(bad_rec).valid

    batch_ok = check_batch(df_raw.drop(columns=["default_payment_next_month"]).head(50))
    assert batch_ok.valid

    ok("data quality: valid passes, missing field fails")
except Exception as e:
    fail("data quality monitoring", e)

# 16. Drift monitoring 
#---------------------------------------------------------------------------------
print("\n--- 16. Drift monitoring ---")
try:
    import numpy as np
    from src.monitoring.drift import ks_drift_test, compute_psi, run_drift_report

    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 300))
    same = pd.Series(rng.normal(0, 1, 300))
    shifted = pd.Series(rng.normal(5, 1, 300))

    # No drift
    r = ks_drift_test(ref, same)
    assert not r["drifted"]

    # Clear drift
    r2 = ks_drift_test(ref, shifted)
    assert r2["drifted"]

    # PSI
    assert compute_psi(rng.normal(0, 1, 500), rng.normal(0, 1, 500)) < 0.1
    assert compute_psi(rng.normal(0, 1, 500), rng.normal(10, 1, 500)) > 0.2

    # Full report
    df1 = df_raw.drop(columns=["default_payment_next_month"]).head(150)
    df2 = df_raw.drop(columns=["default_payment_next_month"]).tail(150)
    report = run_drift_report(df1, df2, save=False)
    assert "n_features_checked" in report
    assert report["n_features_checked"] > 0

    ok("drift: KS detects shift, PSI thresholds correct, drift_report runs")
except Exception as e:
    fail("drift monitoring", e)

# 17. Paths and dirs 
#---------------------------------------------------------------------------------
print("\n--- 17. Paths ---")
try:
    from src.utils.paths import ensure_dirs, ROOT, ARTIFACTS_MODELS
    ensure_dirs()
    assert ARTIFACTS_MODELS.exists()
    ok("ensure_dirs creates all artifact directories")
except Exception as e:
    fail("paths", e)

# Summary 
# #---------------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  PASSED: {len(passed)}/{len(passed)+len(failed)}")
if failed:
    print(f"  FAILED: {len(failed)}")
    for f in failed:
        print(f"    ✗ {f}")
    print(f"{'='*55}")
    sys.exit(1)
else:
    print(f"  All checks passed ")
    print(f"{'='*55}")
