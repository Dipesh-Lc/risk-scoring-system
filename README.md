# risk-scoring-system

**Explainable credit risk scoring system with model calibration, monitoring, and API serving**

---

## Business Problem

A lender or fintech company needs to assess the **probability that a customer will default on their credit payment** in the next month. Rather than returning raw probabilities, this system converts model output into an **operational risk score (0-100)** with a risk band (Low / Medium / High) and a natural-language explanation of the top drivers.

**Decision supported:** Approve/decline credit applications, set credit limits, flag accounts for review.

**Model predicts:** Probability of default payment next month.

**Output per customer:**
- `default_probability` -- calibrated float (0-1)
- `risk_score` -- integer 0-100
- `risk_band` -- Low | Medium | High
- `top_drivers` -- list of SHAP-based feature explanations

**Success metrics:**
- ROC-AUC ≥ 0.78
- PR-AUC ≥ 0.45
- Well-calibrated probabilities (Brier score < 0.15)
- API p99 latency < 200ms

---

## Data Source

**UCI Default of Credit Card Clients**  
30,000 customers × 23 features. Binary target: `default.payment.next.month`.

| Feature | Description |
|---|---|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX` | 1=male, 2=female |
| `EDUCATION` | 1=grad school, 2=university, 3=high school, 4=other |
| `MARRIAGE` | 1=married, 2=single, 3=other |
| `AGE` | Age in years |
| `PAY_0`-`PAY_6` | Repayment status months 0–6 (-1=on time, 0= payment made but less than bill amount, 1-9=months delayed) |
| `BILL_AMT1`-`BILL_AMT6` | Bill statement amount months 1-6 |
| `PAY_AMT1`-`PAY_AMT6` | Payment amount months 1-6 |
| `default.payment.next.month` | **Target**: 1=default, 0=no default |

---

## Architecture

```
Raw CSV → Validation → Preprocessing → Feature Engineering
                                              ↓
                                    Train/Val/Test Split
                                              ↓
                             Baseline (LR) → XGBoost → Calibration
                                              ↓
                                    Evaluation + SHAP
                                              ↓
                              FastAPI  ←  Model Registry
                                              ↓
                                  Monitoring + Drift
```

---

## Score Logic

| Probability Range | Risk Score | Risk Band |
|---|---|---|
| 0.00 - 0.20 | 0 - 30 | 🟢 Low |
| 0.20 - 0.50 | 31 - 60 | 🟡 Medium |
| 0.50 - 1.00 | 61 - 100 | 🔴 High |

Score formula: `score = round(probability * 100)`  
Capped to [0, 100] and mapped to bands per above.

Full logic documented in [`docs/scoring_logic.md`](docs/scoring_logic.md).

---

## Model Performance 

| Metric | Logistic Regression | XGBoost (calibrated) |
|---|---|---|
| ROC-AUC | ~0.72 | ~0.78 |
| PR-AUC | ~0.38 | ~0.46 |
| Brier Score | ~0.18 | ~0.13 |

---

## Explainability

Global importance and per-prediction SHAP explanations are generated for every scoring request.

Top global drivers:
1. Recent repayment status (`PAY_0`)
2. Payment amount ratios
3. Credit limit utilization

---

## API Usage

```bash
# Start the server
uvicorn src.api.app:app --reload --port 8000

# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 15000,
    "BILL_AMT2": 14000,
    "BILL_AMT3": 13000,
    "BILL_AMT4": 12000,
    "BILL_AMT5": 11000,
    "BILL_AMT6": 10000,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 1800,
    "PAY_AMT3": 1600,
    "PAY_AMT4": 1500,
    "PAY_AMT5": 1400,
    "PAY_AMT6": 1300
  }'
```

Response:
```json
{
  "default_probability": 0.14,
  "risk_score": 14,
  "risk_band": "Low",
  "top_drivers": [
    {"feature": "PAY_0", "impact": -0.32, "direction": "decreases_risk"},
    {"feature": "LIMIT_BAL", "impact": -0.18, "direction": "decreases_risk"},
    {"feature": "PAY_AMT1", "impact": -0.09, "direction": "decreases_risk"}
  ]
}
```

---

## How to Run

> All commands work on **Windows (PowerShell), macOS, and Linux**.  
> Run everything from the project root folder: `cd risk-scoring-system`

---

### 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate risk-scoring-system
```

If you prefer pip over conda:
```bash
pip install -r requirements.txt
```

---

### 2. Download the dataset

```bash
python -m src.ingestion.load_data
```

Downloads the UCI Default of Credit Card Clients dataset into `data/raw/`.  
If there is no internet connection it automatically generates a synthetic fallback dataset instead.

---

### 3. Run the full training pipeline

```bash
python -m src.pipelines.train_pipeline
```

This single command runs all 12 steps end-to-end:
- Loads and validates data
- Cleans and engineers features
- Splits into train / val / test
- Trains the Logistic Regression baseline
- Trains XGBoost
- Calibrates probabilities
- Evaluates all models and saves metrics to `artifacts/metrics/`
- Computes SHAP global importance and saves to `artifacts/shap/`
- Saves all model artifacts to `artifacts/models/`
- Generates `reports/model_report.md`

---

### 4. Start the prediction API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser at:
- **Interactive docs (Swagger UI):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

To send a test prediction from PowerShell:
```powershell
$body = @{
    LIMIT_BAL=50000; SEX=2; EDUCATION=2; MARRIAGE=1; AGE=35
    PAY_0=0; PAY_2=0; PAY_3=0; PAY_4=0; PAY_5=0; PAY_6=0
    BILL_AMT1=15000; BILL_AMT2=14000; BILL_AMT3=13000
    BILL_AMT4=12000; BILL_AMT5=11000; BILL_AMT6=10000
    PAY_AMT1=2000; PAY_AMT2=1800; PAY_AMT3=1600
    PAY_AMT4=1500; PAY_AMT5=1400; PAY_AMT6=1300
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST `
    -ContentType "application/json" -Body $body
```

Or with curl (macOS / Linux / WSL):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 2000, "PAY_AMT2": 1800, "PAY_AMT3": 1600,
    "PAY_AMT4": 1500, "PAY_AMT5": 1400, "PAY_AMT6": 1300
  }'
```

---

### 5. Run batch scoring

```bash
python -m src.pipelines.batch_scoring_pipeline --input data/samples/sample_customers.csv
```

Scores all records in the CSV and writes results to `artifacts/scores/batch_results.csv`.

To specify a custom output path:
```bash
python -m src.pipelines.batch_scoring_pipeline --input data/samples/sample_customers.csv --output artifacts/scores/my_results.csv
```

---

### 6. Run the test suite

```bash
pytest tests/ -v
```

Run with coverage report:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run a single test file:
```bash
pytest tests/test_train.py -v
```

---

### 7. Run the linters

```bash
black src/ tests/ --check
isort src/ tests/ --check-only
```

To auto-fix formatting:
```bash
black src/ tests/
isort src/ tests/
```

---

### 8. Run with Docker

```bash
docker-compose up --build
```

Stop it:
```bash
docker-compose down
```

---

### 9. Generate the model report

```bash
python -m src.monitoring.reporting
```

Writes `reports/model_report.md` summarising metrics, SHAP drivers, and the monitoring plan.

---

### 10. Validate data quality manually

```bash
python -m src.validation.data_checks
```

---

### Quick reference

| What you want to do | Command |
|---|---|
| Create environment | `conda env create -f environment.yml` |
| Activate environment | `conda activate risk-scoring-system` |
| Download data | `python -m src.ingestion.load_data` |
| Train everything | `python -m src.pipelines.train_pipeline` |
| Start API | `uvicorn src.api.app:app --reload --port 8000` |
| Batch score | `python -m src.pipelines.batch_scoring_pipeline --input data/samples/sample_customers.csv` |
| Run tests | `pytest tests/ -v` |
| Run tests + coverage | `pytest tests/ -v --cov=src --cov-report=term-missing` |
| Auto-format code | `black src/ tests/` |
| Generate report | `python -m src.monitoring.reporting` |
| Validate data | `python -m src.validation.data_checks` |
| Start with Docker | `docker-compose up --build` |

---

## Project Structure

```
risk-scoring-system/
├── src/
│   ├── ingestion/        # Data loading
│   ├── validation/       # Data quality checks
│   ├── processing/       # Preprocessing + splits
│   ├── features/         # Feature engineering
│   ├── models/           # Train, evaluate, calibrate, registry
│   ├── explainability/   # SHAP analysis
│   ├── api/              # FastAPI service
│   ├── monitoring/       # Drift + quality monitoring
│   ├── pipelines/        # End-to-end orchestration
│   └── utils/            # Logger, paths
├── tests/                # pytest suite
├── notebooks/            # EDA + experiments
├── artifacts/            # Saved models, metrics, SHAP
├── reports/              # Model report
└── docs/                 # Architecture + decisions
```

---

## Limitations

- Model trained on Taiwanese credit data (2005); may not generalize globally
- No fairness analysis across protected attributes (SEX, MARRIAGE, AGE)
- Drift monitoring is statistical-only; no ground-truth label stream
- SHAP explanations are approximate for tree ensembles with interactions

## Future Work

- Fairness/bias audit (demographic parity, equalized odds)
- Real-time label feedback loop for monitoring
- Cost-sensitive threshold optimization (German Credit extension)
- MLflow experiment tracking integration
- Streamlit dashboard for risk analysts
