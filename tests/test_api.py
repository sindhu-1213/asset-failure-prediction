"""
tests/test_api.py
Integration tests for the FastAPI serving layer.
Uses TestClient so no server needs to be running.

Author: Amreen
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# models won't be loaded in CI without artifacts — test structure and error handling
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

SAMPLE_CYCLES = [
    {"cycle": i, "op1": 0.0, "op2": 0.0, "op3": 100.0,
     **{f"s{j}": 500.0 + (i * 0.1) for j in range(1, 22)}}
    for i in range(1, 31)
]


# ── health check ──────────────────────────────────────────────────────────────

def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_status_key():
    resp = client.get("/health")
    assert "status" in resp.json()


def test_health_has_models_key():
    resp = client.get("/health")
    assert "models" in resp.json()


# ── /predict/rul ──────────────────────────────────────────────────────────────

def test_rul_endpoint_returns_503_without_model():
    """Without saved artifacts, endpoint should 503 gracefully — not crash."""
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/rul", json=payload)
    # 503 if model not loaded, 200 if it is — both are acceptable
    assert resp.status_code in (200, 503)


def test_rul_endpoint_valid_response_structure():
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/rul", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        assert "unit_id"       in data
        assert "predicted_rul" in data
        assert data["predicted_rul"] >= 0


def test_rul_endpoint_rejects_empty_cycles():
    payload = {"unit_id": 1, "cycles": []}
    resp = client.post("/predict/rul", json=payload)
    assert resp.status_code == 422   # validation error


# ── /predict/risk ─────────────────────────────────────────────────────────────

def test_risk_endpoint_returns_503_without_model():
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/risk", json=payload)
    assert resp.status_code in (200, 503)


def test_risk_response_has_required_fields():
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/risk", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        assert "failure_prob"   in data
        assert "alert"          in data
        assert "risk_level"     in data
        assert "threshold_used" in data


def test_risk_probability_in_range():
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/risk", json=payload)
    if resp.status_code == 200:
        prob = resp.json()["failure_prob"]
        assert 0.0 <= prob <= 1.0


def test_risk_level_valid_value():
    payload = {"unit_id": 1, "cycles": SAMPLE_CYCLES}
    resp = client.post("/predict/risk", json=payload)
    if resp.status_code == 200:
        level = resp.json()["risk_level"]
        assert level in ("LOW", "MEDIUM", "HIGH")


# ── /drift/status ─────────────────────────────────────────────────────────────

def test_drift_endpoint_returns_200():
    resp = client.get("/drift/status")
    assert resp.status_code == 200


def test_drift_endpoint_returns_json():
    resp = client.get("/drift/status")
    assert resp.headers["content-type"].startswith("application/json")
