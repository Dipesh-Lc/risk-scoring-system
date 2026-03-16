"""
FastAPI prediction service for the risk scoring system.

Endpoints:
    GET  /health          — liveness check
    POST /predict         — score a single customer
    POST /predict_batch   — score up to 1000 customers

Run:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CustomerFeatures,
    HealthResponse,
    ScoreResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model state
# -------------------------------------------------------------------------------------
_state: dict[str, Any] = {
    "model": None,
    "scaler": None,
    "feature_names": None,
    "loaded": False,
}


def _load_models() -> None:
    """Load model artifacts into memory (called at startup)."""
    from src.models.registry import load_artifact

    try:
        _state["model"] = load_artifact("xgb_calibrated")
        _state["scaler"] = load_artifact("scaler")
        _state["feature_names"] = load_artifact("feature_names")
        _state["loaded"] = True
        logger.info("Models loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Model artifacts not found: %s — /predict will return 503.", exc)
        _state["loaded"] = False


# Lifespan
# -------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    logger.info("API shutting down.")


# App
# -------------------------------------------------------------------------------------
app = FastAPI(
    title="Risk Scoring API",
    description="Explainable credit risk scoring with calibrated probabilities and SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: request timing
# -------------------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# Routes
# -------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 if the service is running."""
    return HealthResponse(
        status="ok",
        model_loaded=_state["loaded"],
        version=app.version,
    )


@app.post("/predict", response_model=ScoreResponse, tags=["Scoring"])
def predict(customer: CustomerFeatures) -> ScoreResponse:
    """
    Score a single customer.

    Returns the calibrated default probability, risk score (0–100),
    risk band (Low / Medium / High), and top SHAP-based risk drivers.
    """
    if not _state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    from src.models.predict import predict_single

    try:
        result = predict_single(
            features=customer.model_dump(),
            model=_state["model"],
            scaler=_state["scaler"],
            feature_names=_state["feature_names"],
        )
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return ScoreResponse(**result)


@app.post("/predict_batch", response_model=BatchScoreResponse, tags=["Scoring"])
def predict_batch(request: BatchScoreRequest) -> BatchScoreResponse:
    """
    Score a batch of up to 1000 customers.

    Returns a list of score results in the same order as the input records.
    Note: SHAP explanations are omitted in batch mode for performance.
    """
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import pandas as pd

    from src.models.predict import predict_batch as _batch

    try:
        df = pd.DataFrame([r.model_dump() for r in request.records])
        scored = _batch(
            df,
            model=_state["model"],
            scaler=_state["scaler"],
            feature_names=_state["feature_names"],
        )
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")

    results = [
        ScoreResponse(
            default_probability=row["default_probability"],
            risk_score=row["risk_score"],
            risk_band=row["risk_band"],
            top_drivers=[],
        )
        for _, row in scored.iterrows()
    ]
    return BatchScoreResponse(results=results, count=len(results))


# Error handlers
# -------------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
