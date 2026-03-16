# Risk Scoring System - Model Report

Generated: 2026-03-16 21:16

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
| Logistic Regression | 0.7484 | 0.5097 | 0.1913 | 0.5090 |

| XGBoost (raw) | 0.7819 | 0.5633 | 0.1705 | 0.5380 |

| XGBoost (calibrated) | 0.7804 | 0.5375 | 0.1347 | 0.4711 |


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
| PAY_0 | 0.3259 |
| delinquency_count | 0.3008 |
| max_delinquency | 0.1775 |
| PAY_AMT2 | 0.1547 |
| util_ratio_1 | 0.1464 |
| BILL_AMT1 | 0.1306 |
| PAY_AMT3 | 0.1028 |
| PAY_AMT1 | 0.0997 |
| bill_trend | 0.0956 |
| LIMIT_BAL | 0.0892 |


---

## 5. Monitoring

No drift data available.

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
