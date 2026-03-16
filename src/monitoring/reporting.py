"""
Model report generation: produces reports/model_report.md.

Run as module:
    python -m src.monitoring.reporting
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_METRICS, ARTIFACTS_SHAP, REPORTS, ensure_dirs

logger = get_logger(__name__)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_model_report(output_path: Path | None = None) -> Path:
    """
    Collate metrics and SHAP results into a Markdown model report.
    """
    ensure_dirs()
    output_path = output_path or REPORTS / "model_report.md"

    baseline_metrics = _load_json(ARTIFACTS_METRICS / "baseline_metrics.json")
    final_metrics = _load_json(ARTIFACTS_METRICS / "xgb_metrics.json")
    calibrated_metrics = _load_json(ARTIFACTS_METRICS / "xgb_calibrated_metrics.json")
    drift_report = _load_json(ARTIFACTS_METRICS / "drift_report.json")

    shap_importance_path = ARTIFACTS_SHAP / "global_importance.csv"
    shap_rows = ""
    if shap_importance_path.exists():
        import pandas as pd

        top = pd.read_csv(shap_importance_path).head(10)
        for _, row in top.iterrows():
            shap_rows += f"| {row['feature']} | {row['mean_abs_shap']:.4f} |\n"

    def _metric_row(m: dict, name: str) -> str:
        if not m:
            return f"| {name} | N/A | N/A | N/A | N/A |\n"
        return (
            f"| {name} "
            f"| {m.get('roc_auc', 'N/A'):.4f} "
            f"| {m.get('pr_auc', 'N/A'):.4f} "
            f"| {m.get('brier_score', 'N/A'):.4f} "
            f"| {m.get('f1', 'N/A'):.4f} |\n"
        )

    drift_summary = "No drift data available."
    if drift_report:
        n_drifted = drift_report.get("n_drifted", 0)
        n_checked = drift_report.get("n_features_checked", 0)
        drifted = drift_report.get("drifted_features", [])
        drift_summary = (
            f"Checked {n_checked} features. "
            f"{n_drifted} showed drift: `{', '.join(drifted) or 'none'}`."
        )

    report = f"""# Risk Scoring System - Model Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Business Context

A calibrated risk scoring system that estimates the probability a credit card holder will default 
in the next month. Outputs a risk score (0-100) and SHAP-based explanation for each prediction.

**Target**: `default_payment_next_month`  
**Dataset**: UCI Default of Credit Card Clients (30,000 records, 23 raw features)

---

## 2. Model Performance (Test Set)

| Model | ROC-AUC | PR-AUC | Brier | F1 |
|---|---|---|---|---|
{_metric_row(baseline_metrics, "Logistic Regression")}
{_metric_row(final_metrics, "XGBoost (raw)")}
{_metric_row(calibrated_metrics, "XGBoost (calibrated)")}

> **Primary metric**: ROC-AUC (ranking quality).  
> **Calibration metric**: Brier Score (probability quality).

---

## 3. Score Band Distribution

| Band | Score Range | Interpretation |
|---|---|---|
| Low | 0-30 | Low default risk |
| Medium | 31-60 | Moderate risk - review recommended |
| High | 61-100 | High default risk - further scrutiny required |

---

## 4. Top SHAP Feature Drivers (Global)

| Feature | Mean |SHAP| |
|---|---|
{shap_rows if shap_rows else "| SHAP analysis not yet run | N/A |"}

---

## 5. Monitoring

{drift_summary}

**Monitoring plan:**
- Run KS test and PSI on scoring input distributions weekly
- Alert if PSI > 0.2 on any top-5 feature
- Re-train if ROC-AUC on a labelled holdout drops more than 0.03

---

## 6. Limitations

- Trained on Taiwanese consumer credit data (2005); may not generalise to other markets/eras
- No fairness audit - demographic parity across SEX, MARRIAGE, AGE not assessed
- Drift monitoring is statistical-only; no ground-truth label stream post-deployment
- SHAP values are additive approximations; interaction effects are not explicitly captured

---

## 7. Future Work

- Fairness / bias audit (demographic parity, equalized odds)
- Cost-sensitive threshold using German Credit cost matrix
- MLflow experiment tracking for all runs
- Real-time monitoring with label feedback loop
- Streamlit risk analyst dashboard
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Model report saved -> %s", output_path)
    return output_path


if __name__ == "__main__":
    path = generate_model_report()
    print(f"Report written to: {path}")
