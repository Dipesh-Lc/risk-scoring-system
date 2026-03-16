"""Tests for the FastAPI prediction service."""

from __future__ import annotations

import pytest

# Skip entire module if FastAPI or httpx are not installed
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed")

from fastapi.testclient import TestClient

from src.api.app import _state, app


@pytest.fixture(scope="module")
def client_with_model(calibrated_xgb, scaler_and_scaled, split_data):
    """TestClient with a real model injected into the app state."""
    scaler, *_ = scaler_and_scaled
    _, _, _, _, _, _, feature_names = split_data

    # Inject test artifacts into app state
    _state["model"] = calibrated_xgb
    _state["scaler"] = scaler
    _state["feature_names"] = feature_names
    _state["loaded"] = True

    with TestClient(app) as c:
        yield c

    # Reset after tests
    _state["loaded"] = False


@pytest.fixture(scope="module")
def client_no_model():
    """TestClient with no model loaded -- only meaningful when artifacts don't exist."""
    original_state = dict(_state)
    _state["loaded"] = False
    _state["model"] = None
    _state["scaler"] = None
    _state["feature_names"] = None
    with TestClient(app) as c:
        yield c
    _state.update(original_state)


def _artifacts_exist() -> bool:
    """Return True if trained model artifacts are present on disk."""
    from pathlib import Path

    return Path("artifacts/models/xgb_calibrated.joblib").exists()


# --- Health ---


class TestHealth:
    def test_health_returns_200(self, client_with_model):
        resp = client_with_model.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded_true(self, client_with_model):
        resp = client_with_model.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    @pytest.mark.skipif(
        _artifacts_exist(),
        reason="Artifacts exist on disk so lifespan always loads the model -- "
        "this test is only meaningful in a clean CI environment.",
    )
    def test_health_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


# --- Single prediction ---

VALID_PAYLOAD = {
    "LIMIT_BAL": 80000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 20000,
    "BILL_AMT2": 18000,
    "BILL_AMT3": 16000,
    "BILL_AMT4": 14000,
    "BILL_AMT5": 12000,
    "BILL_AMT6": 10000,
    "PAY_AMT1": 3000,
    "PAY_AMT2": 2500,
    "PAY_AMT3": 2000,
    "PAY_AMT4": 1800,
    "PAY_AMT5": 1600,
    "PAY_AMT6": 1400,
}


class TestPredictEndpoint:
    def test_predict_status_200(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_keys(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "default_probability" in data
        assert "risk_score" in data
        assert "risk_band" in data
        assert "top_drivers" in data

    def test_predict_probability_range(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_score_range(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        score = resp.json()["risk_score"]
        assert 0 <= score <= 100

    def test_predict_band_valid(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["risk_band"] in ("Low", "Medium", "High")

    def test_predict_missing_field_422(self, client_with_model):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "LIMIT_BAL"}
        resp = client_with_model.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_invalid_sex_422(self, client_with_model):
        bad_payload = dict(VALID_PAYLOAD, SEX=5)
        resp = client_with_model.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    @pytest.mark.skipif(
        _artifacts_exist(),
        reason="Artifacts exist on disk so lifespan always loads the model -- "
        "this test is only meaningful in a clean CI environment.",
    )
    def test_predict_no_model_503(self, client_no_model):
        resp = client_no_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_predict_process_time_header(self, client_with_model):
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert "X-Process-Time-Ms" in resp.headers


# --- Batch prediction ---


class TestBatchPredictEndpoint:
    def test_batch_predict_status_200(self, client_with_model):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        resp = client_with_model.post("/predict_batch", json=payload)
        assert resp.status_code == 200

    def test_batch_predict_count(self, client_with_model):
        n = 5
        payload = {"records": [VALID_PAYLOAD] * n}
        resp = client_with_model.post("/predict_batch", json=payload)
        data = resp.json()
        assert data["count"] == n
        assert len(data["results"]) == n

    def test_batch_predict_empty_422(self, client_with_model):
        resp = client_with_model.post("/predict_batch", json={"records": []})
        assert resp.status_code == 422

    def test_batch_result_structure(self, client_with_model):
        payload = {"records": [VALID_PAYLOAD]}
        resp = client_with_model.post("/predict_batch", json=payload)
        result = resp.json()["results"][0]
        assert "default_probability" in result
        assert "risk_score" in result
        assert "risk_band" in result
