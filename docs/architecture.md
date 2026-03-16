# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      risk-scoring-system                        │
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐    │
│  │  Data    │    │ Validation │    │  Feature Engineering │    │
│  │ Ingestion│───▶│  Checks   │───▶│  (feature_store.py)  │    │
│  │          │    │           │    │  8 derived features  │    │
│  └──────────┘    └────────────┘    └──────────┬───────────┘    │
│                                               │                 │
│                                    ┌──────────▼───────────┐    │
│                                    │   Preprocessing       │    │
│                                    │  clean + StandardScal │    │
│                                    └──────────┬────────────┘    │
│                                               │                 │
│                              ┌────────────────▼──────────────┐ │
│                              │        Model Training         │ │
│                              │  Baseline (LR) ──▶ XGBoost   │ │
│                              │                    │           │ │
│                              │              Calibration       │ │
│                              │           (isotonic CV)        │ │
│                              └────────────────┬──────────────┘ │
│                                               │                 │
│                  ┌─────────────────┬──────────▼───────────┐    │
│                  │   Evaluation    │   SHAP Explainability│    │
│                  │ ROC/PR/Brier/F1 │   Global + Local     │    │
│                  └─────────────────┴──────────────────────┘    │
│                                               │                 │
│                              ┌────────────────▼──────────────┐ │
│                              │       Model Registry          │ │
│                              │  artifacts/models/*.joblib    │ │
│                              └────────────────┬──────────────┘ │
│                                               │                 │
│            ┌──────────────────────────────────▼──────────────┐ │
│            │               FastAPI Service                    │ │
│            │  GET  /health                                    │ │
│            │  POST /predict        → score + explanation      │ │
│            │  POST /predict_batch  → batch scoring            │ │
│            └──────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     Monitoring                           │  │
│  │  Input quality checks │ KS drift │ PSI │ Score shift     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Data Layer (`src/ingestion/`, `src/validation/`)
- **`load_data.py`**: Downloads UCI dataset via `ucimlrepo`. Includes a
  synthetic data fallback for CI/offline environments.
- **`data_checks.py`**: Schema validation, missing value thresholds, range
  checks, class balance reporting. Returns a `ValidationReport` with
  structured `issues` and `warnings`.

### Processing Layer (`src/processing/`, `src/features/`)
- **`preprocess.py`**: Applies `clean_raw()` (remapping undocumented categorical
  codes), then `StandardScaler` on numeric + ordinal payment-status columns.
  Scaler is fit on train only to prevent data leakage.
- **`split.py`**: Stratified 70/15/15 train/val/test split, preserving the
  ~22% default rate in all three partitions.
- **`feature_store.py`**: Adds 8 domain-driven derived features capturing
  credit utilisation, payment coverage, delinquency severity, and billing
  trends. These features are computed deterministically from raw columns.

### Modelling Layer (`src/models/`)
- **`train.py`**: Logistic Regression baseline + XGBoost final model. Includes
  an `Optuna`-based hyperparameter search function (optional).
- **`calibrate.py`**: `CalibratedClassifierCV(cv="prefit", method="isotonic")`
  fitted on the validation set. Converts raw probabilities into a 0–100 score
  and categorical band.
- **`evaluate.py`**: Computes ROC-AUC, PR-AUC, Brier Score, F1, precision,
  recall, confusion matrix. Generates ROC and calibration curve plots.
- **`registry.py`**: Thin wrapper around `joblib` for serialisation. All
  artifacts (model, scaler, feature list) are saved to `artifacts/models/`.
- **`predict.py`**: Inference helpers for single-record and batch scoring,
  including SHAP driver retrieval.

### Explainability (`src/explainability/`)
- **`shap_analysis.py`**: `TreeExplainer` for XGBoost. Provides global mean
  |SHAP| importance and per-prediction top-N driver dicts consumed by the API.

### API Layer (`src/api/`)
- **`app.py`**: FastAPI application. Models are loaded once at startup via the
  `lifespan` hook and stored in module-level state. A timing middleware records
  `X-Process-Time-Ms` on every response.
- **`schemas.py`**: Pydantic v2 models with field-level validation (range
  constraints, required fields). Provides OpenAPI documentation automatically.

### Monitoring (`src/monitoring/`)
- **`data_quality.py`**: Validates scoring-time inputs before they reach the
  model. Separates hard errors (missing fields) from warnings (range outliers).
- **`drift.py`**: KS test and PSI for feature drift; PSI for score distribution
  shift. Designed to run as a scheduled job against a rolling window of
  production requests.
- **`reporting.py`**: Generates `reports/model_report.md` from saved metrics
  and SHAP artefacts. Useful for stakeholder communication.

### Pipelines (`src/pipelines/`)
- **`train_pipeline.py`**: Orchestrates the full 12-step training flow. One
  command reproduces the complete experiment.
- **`batch_scoring_pipeline.py`**: Accepts a CSV, validates it, scores all
  records, and saves results. Simulates operational batch scoring.

---

## Data Flow

```
data/raw/credit_default.csv
    │
    ▼  load_raw()
raw DataFrame (30k × 23)
    │
    ▼  clean_raw() + add_derived_features()
feature DataFrame (30k × 31)
    │
    ▼  split_dataset()
train / val / test DataFrames
    │
    ▼  build_scaler(X_train) + apply_scaler()
scaled X_train / X_val / X_test
    │
    ├──▶ train_baseline()  → artifacts/models/baseline_lr.joblib
    │
    ├──▶ train_xgboost()   → artifacts/models/xgb_raw.joblib
    │
    ├──▶ calibrate_model() → artifacts/models/xgb_calibrated.joblib
    │
    ├──▶ evaluate()        → artifacts/metrics/*.json
    │
    ├──▶ global_feature_importance() → artifacts/shap/global_importance.csv
    │
    └──▶ generate_model_report() → reports/model_report.md
```

---

## Dependency Graph (simplified)

```
train_pipeline
    ├── ingestion.load_data
    ├── validation.data_checks
    ├── processing.preprocess
    ├── processing.split
    ├── features.feature_store
    ├── models.train
    ├── models.calibrate
    ├── models.evaluate
    ├── models.registry
    ├── explainability.shap_analysis
    └── monitoring.reporting

api.app
    ├── models.registry  (loads artifacts at startup)
    ├── models.predict
    │       ├── features.feature_store
    │       ├── processing.preprocess
    │       ├── models.calibrate
    │       └── explainability.shap_analysis
    └── api.schemas
```
