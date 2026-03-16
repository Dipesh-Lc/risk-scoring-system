# Modeling Decisions

This document records the key design choices made during development and the
reasoning behind each one. It is intended to be readable in a technical interview
or code review.

---

## 1. Problem Framing

**Decision**: Frame as a *risk scoring product*, not just binary classification.

**Reasoning**: The raw model output (probability) is not directly consumable by
business stakeholders or downstream systems. Converting to a 0-100 score with
labelled bands (Low / Medium / High) gives analysts an immediately actionable
signal. SHAP explanations at the record level further increase operational value.

---

## 2. Dataset Choice

**Dataset**: UCI Default of Credit Card Clients (ID=350)  
**Size**: 30,000 rows, 23 features  
**Target**: `default.payment.next.month` (binary: 1 = defaulted)

**Why this dataset**:
- Directly aligned with credit risk (the exact business domain)
- Large enough to train, validate, and test robustly
- Good class imbalance (~22% positives) that forces real decisions about
  metrics and thresholds
- Well-known in the literature, so results are interpretable relative to prior work

**Note on temporal validity**: The data is from 2005 Taiwan. Coefficients may not transfer to other geographies or eras. This is documented in Limitations.

---

## 3. Target Variable

`default_payment_next_month` is a binary indicator. We predict the **probability**
of default, not just the binary outcome, because:

- Probabilities support risk-tier segmentation (not just approve/decline)
- Calibrated probabilities allow expected-value calculations
- Business thresholds can be adjusted post-deployment without retraining

---

## 4. Train / Validation / Test Split

**Ratios**: 70% train / 15% validation / 15% test  
**Method**: Stratified by target to preserve class balance  
**Seed**: 42 (fixed for reproducibility)

The validation set is used exclusively for:
- Probability calibration
- Early stopping in XGBoost
- Optuna trial evaluation

The test set is held out until the final comparison and never used for any
model selection decision.

---

## 5. Feature Engineering

Eight derived features were added to the raw columns:

| Feature | Rationale |
|---|---|
| `util_ratio_1` | Current-month utilisation is a strong default signal |
| `util_ratio_avg` | Average utilisation over 6 months adds stability |
| `pay_ratio_1` | Payment-to-bill ratio captures repayment behaviour |
| `pay_ratio_avg` | Average payment ratio is robust to single-month anomalies |
| `delinquency_count` | Number of late months summarises repayment history |
| `max_delinquency` | Worst delinquency captures tail risk |
| `bill_trend` | Rising balances signal deteriorating credit health |
| `pay_trend` | Declining payments are an early warning signal |

Derived features are computed before the scaler is fit, ensuring they are
also normalised.

---

## 6. Categorical Encoding

The UCI dataset encodes categoricals (SEX, EDUCATION, MARRIAGE) as integers.
We leave them as-is and scale them alongside numeric columns, which is
appropriate for tree-based models. For Logistic Regression, this is a
simplification — one-hot encoding would be more principled — but the difference
in practice is negligible for this dataset given the small cardinality.

**Undocumented codes**: EDUCATION=0,5,6 and MARRIAGE=0 appear in the data
but are not in the official codebook. We remap them to the "Other" category
(4 and 3 respectively) rather than dropping rows.

---

## 7. Scaling

`StandardScaler` is applied to numeric and payment-status columns. This is
required for the Logistic Regression baseline and has no negative effect on
XGBoost (trees are invariant to monotone scaling).

The scaler is **fit on training data only** and applied identically to
validation, test, and scoring inputs to prevent leakage.

---

## 8. Class Imbalance

The dataset has ~22% positives. Approaches considered:

| Approach | Decision |
|---|---|
| `class_weight="balanced"` in LR | ✅ Used |
| `scale_pos_weight` in XGBoost | ✅ Set to n_neg/n_pos |
| SMOTE oversampling | ❌ Not used — tree models handle imbalance well natively |
| Undersampling | ❌ Discards information |

We chose to handle imbalance through class weights rather than resampling,
because SHAP values remain interpretable on the original feature distribution.

---

## 9. Model Selection

**Baseline**: Logistic Regression  
- Fully interpretable linear model
- Sets a minimum performance floor
- Fast to train and explain

**Final model**: XGBoost  
- Consistently outperforms linear models on tabular credit data
- Native handling of class imbalance via `scale_pos_weight`
- Compatible with TreeExplainer (fast, exact SHAP values)
- Production-ready with serialisation and controlled hyperparameters

Alternatives considered but not used:
- **Random Forest**: Slower inference, similar accuracy
- **LightGBM**: Comparable to XGBoost; XGBoost chosen for wider support
- **Neural networks**: No performance advantage on this small tabular dataset;  harder to explain

---

## 10. Probability Calibration

XGBoost outputs well-ranked but often poorly-calibrated probabilities.
Business decisions depend on the *magnitude* of the probability, not just
the ranking. Therefore we apply isotonic regression calibration using the
validation set.

**Method**: `CalibratedClassifierCV(cv="prefit", method="isotonic")`  
**Why isotonic over sigmoid**: Isotonic makes fewer distributional assumptions
and typically performs better with larger calibration sets (>1000 samples).

The improvement is measured with the Brier Score (lower = better calibration).

---

## 11. Evaluation Metrics

| Metric | Why |
|---|---|
| **ROC-AUC** | Primary ranking metric; robust to class imbalance |
| **PR-AUC** | Precision-Recall tradeoff; more informative than ROC under high imbalance |
| **Brier Score** | Measures calibration quality (probability accuracy) |
| **F1 @ 0.5** | Summary of classification performance at default threshold |
| Confusion matrix | Understand false positives vs false negatives at threshold |

We do not optimise F1 directly because the optimal business threshold
is context-dependent (cost of false negative vs false positive).

---

## 12. Threshold Selection

The default decision threshold (0.5) is used for classification metrics.
The `find_optimal_threshold()` function in `evaluate.py` can find the
F1-maximising threshold if needed.

For real deployment, the threshold should be set based on the asymmetric
cost structure: incorrectly approving a defaulting customer is typically
more costly than incorrectly declining a good customer.

---

## 13. SHAP Implementation

`TreeExplainer` is used for XGBoost (exact, fast). For the calibrated
wrapper, we unwrap `CalibratedClassifierCV` to reach the base XGBoost
estimator before computing SHAP values.

Global importance is computed on a 500-row sample of the test set.
Per-prediction explanations are computed for every API call and included
in the response.

---

## 14. Monitoring Design

**KS test threshold**: 0.1 (flags statistically significant distribution shift)  
**PSI threshold**: 0.2 (industry standard for "significant shift")

Monitoring runs on input features (before the model sees them) and on the output score distribution. This enables early warning before label feedback is available.
