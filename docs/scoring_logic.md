# Scoring Logic

## Overview

The model outputs a calibrated probability of default. This probability is
converted into a business-friendly **risk score (0-100)** and a **risk band**
(Low / Medium / High) for operational decision support.

---

## Probability -> Score Conversion

**Formula**:

```
score = round(probability x 100)
score = clamp(score, 0, 100)
```

**Examples**:

| Calibrated Probability | Risk Score | Risk Band |
|---|---|---|
| 0.02 | 2 | 🟢 Low |
| 0.15 | 15 | 🟢 Low |
| 0.30 | 30 | 🟢 Low |
| 0.31 | 31 | 🟡 Medium |
| 0.45 | 45 | 🟡 Medium |
| 0.60 | 60 | 🟡 Medium |
| 0.61 | 61 | 🔴 High |
| 0.80 | 80 | 🔴 High |
| 0.95 | 95 | 🔴 High |

---

## Risk Band Definitions

| Band | Score Range | Interpretation | Recommended Action |
|---|---|---|---|
| 🟢 **Low** | 0 - 30 | Low default risk. Customer profile consistent with reliable repayment. | Auto-approve; standard credit limit |
| 🟡 **Medium** | 31 - 60 | Moderate risk. Mixed signals in payment history or utilisation. | Manual analyst review; consider reduced limit |
| 🔴 **High** | 61 - 100 | High default risk. Significant delinquency signals or extreme utilisation. | Decline or require additional verification |

Band boundaries are set at the 20th and 50th percentiles of predicted
probabilities on the training set, rounded to the nearest 10-point score
boundary. This places roughly:

- ~55% of customers in the Low band
- ~25% in Medium
- ~20% in High

(These proportions reflect the ~22% default rate in the training population.)

---

## Why Linear Scaling?

Alternative scoring transformations considered:

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| `score = prob × 100` (linear) | Simple, transparent, directly interpretable | None significant | ✅ **Chosen** |
| Log-odds scaling | Compresses extremes | Less intuitive for stakeholders | ❌ |
| Percentile rank | Always uses full 0–100 range | Non-comparable across time | ❌ |
| Scorecard points | Industry-standard format | Requires additional calibration step | Future work |

Linear scaling was chosen for its simplicity and transparency. Because
probabilities are well-calibrated, a score of 45 means "approximately 45%
estimated default probability" — directly meaningful.

---

## Decision Threshold vs Score

The risk score is **separate** from any hard decision threshold.

- The score is a continuous ordinal measure of risk
- A human analyst or downstream system applies their own threshold
- The same model can serve different products with different risk appetites
  without retraining

To set a threshold: compute the expected cost of each error type and
find the score cutoff that minimises total cost:

```
total_cost = FN_cost × false_negatives + FP_cost × false_positives
```

For credit card issuers, `FN_cost >> FP_cost` (approving a defaulter costs
more than declining a good customer), so the optimal threshold will typically
be below 50.

---

## Implementation

The scoring logic lives in `src/models/calibrate.py`:

```python
def probability_to_score(probability: float | np.ndarray) -> int | np.ndarray:
    score = np.round(np.asarray(probability, dtype=float) * 100).astype(int)
    return np.clip(score, 0, 100)

def score_to_band(score: int) -> str:
    if score <= 30:   return "Low"
    elif score <= 60: return "Medium"
    else:             return "High"
```

---

## API Response Example

```json
{
  "default_probability": 0.42,
  "risk_score": 42,
  "risk_band": "Medium",
  "top_drivers": [
    {
      "feature": "PAY_0",
      "shap_value": 0.38,
      "feature_value": 2.0,
      "direction": "increases_risk"
    },
    {
      "feature": "util_ratio_avg",
      "shap_value": 0.21,
      "feature_value": 0.74,
      "direction": "increases_risk"
    },
    {
      "feature": "max_delinquency",
      "shap_value": 0.18,
      "feature_value": 2.0,
      "direction": "increases_risk"
    },
    {
      "feature": "pay_ratio_avg",
      "shap_value": -0.12,
      "feature_value": 0.41,
      "direction": "decreases_risk"
    },
    {
      "feature": "LIMIT_BAL",
      "shap_value": -0.08,
      "feature_value": 80000.0,
      "direction": "decreases_risk"
    }
  ]
}
```

The natural-language interpretation of this response would be:

> "This customer has a **Medium** risk score of **42**, indicating an estimated
> 42% probability of default. The primary risk drivers are a recent delinquency
> (`PAY_0 = 2`), high average utilisation (74%), and a 2-month past delinquency.
> These are partially offset by a reasonable payment ratio and a moderate credit limit."
