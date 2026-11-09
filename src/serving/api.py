"""
api.py
FastAPI app — serves RUL predictions and failure risk scores
for a given engine unit's recent sensor window.

Endpoints:
  GET  /health          → liveness check
  POST /predict/rul     → predicted remaining useful life (cycles)
  POST /predict/risk    → failure probability + binary alert
  GET  /drift/status    → latest drift check summary

Author: Amreen
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from src.models.xgboost_model import XGBRULModel, XGBClassifier
from src.features.build_features import (
    add_rolling_features, add_degradation_slope,
    add_cumulative_degradation, get_feature_cols, FLAT_SENSORS
)

# ── load artifacts at startup ─────────────────────────────────────────────────
ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "models/artifacts")

def _load_model(cls, filename):
    path = os.path.join(ARTIFACTS, filename)
    if not os.path.exists(path):
        return None
    try:
        return cls.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {filename}: {e}")
        return None

xgb_rul_model = _load_model(XGBRULModel,   "xgb_rul.pkl")
xgb_clf_model = _load_model(XGBClassifier, "xgb_clf.pkl")

# threshold from cost analysis in notebook 05
_threshold_path = os.path.join(ARTIFACTS, "best_threshold.json")
BEST_THRESHOLD  = 0.5
if os.path.exists(_threshold_path):
    with open(_threshold_path) as f:
        BEST_THRESHOLD = json.load(f).get("threshold", 0.5)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Asset Health Predictor",
    description=(
        "Predictive maintenance API for rotating industrial equipment. "
        "Predicts Remaining Useful Life (RUL) and near-term failure risk "
        "from sensor window data."
    ),
    version="1.0.0",
)


# ── schemas ───────────────────────────────────────────────────────────────────

class SensorWindow(BaseModel):
    """
    Latest N cycles of sensor data for one equipment unit.
    Pass as a list of dicts, each dict being one cycle's readings.
    """
    unit_id: int = Field(..., example=1)
    cycles: List[Dict] = Field(
        ...,
        description="List of sensor readings per cycle, sorted oldest→newest",
        min_items=1
    )


class RULResponse(BaseModel):
    unit_id:       int
    predicted_rul: float
    confidence:    Optional[str] = None


class RiskResponse(BaseModel):
    unit_id:          int
    failure_prob:     float
    alert:            bool
    threshold_used:   float
    risk_level:       str


# ── helpers ───────────────────────────────────────────────────────────────────

def _window_to_features(payload: SensorWindow) -> pd.DataFrame:
    """Convert the raw request into a single-row feature DataFrame."""
    df = pd.DataFrame(payload.cycles)
    df.insert(0, "unit_id", payload.unit_id)

    # must match exactly what build_features() does during training
    df = add_rolling_features(df)
    df = add_degradation_slope(df)
    df = add_cumulative_degradation(df)

    # drop flat sensors — same as training pipeline
    df.drop(columns=[s for s in FLAT_SENSORS if s in df.columns], inplace=True)

    last_row  = df.tail(1)
    feat_cols = get_feature_cols(last_row)
    return last_row[feat_cols]


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    models_loaded = {
        "xgb_rul":  xgb_rul_model is not None,
        "xgb_clf":  xgb_clf_model is not None,
    }
    return {"status": "ok", "models": models_loaded}


@app.post("/predict/rul", response_model=RULResponse, tags=["prediction"])
def predict_rul(payload: SensorWindow):
    if xgb_rul_model is None:
        raise HTTPException(status_code=503, detail="RUL model not loaded")

    try:
        X = _window_to_features(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    rul = float(xgb_rul_model.predict(X)[0])
    confidence = "high" if rul > 60 else ("medium" if rul > 20 else "low")

    return RULResponse(
        unit_id=payload.unit_id,
        predicted_rul=round(rul, 1),
        confidence=confidence
    )


@app.post("/predict/risk", response_model=RiskResponse, tags=["prediction"])
def predict_risk(payload: SensorWindow):
    if xgb_clf_model is None:
        raise HTTPException(status_code=503, detail="Classifier model not loaded")

    try:
        X = _window_to_features(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    prob  = float(xgb_clf_model.predict_proba(X)[0])
    alert = prob >= BEST_THRESHOLD

    return RiskResponse(
        unit_id=payload.unit_id,
        failure_prob=round(prob, 4),
        alert=alert,
        threshold_used=BEST_THRESHOLD,
        risk_level=_risk_level(prob)
    )


@app.get("/drift/status", tags=["monitoring"])
def drift_status():
    """Returns the last saved drift check summary if available."""
    path = os.path.join(ARTIFACTS, "last_drift_report.json")
    if not os.path.exists(path):
        return {"status": "no drift report available yet"}
    with open(path) as f:
        return json.load(f)


# ── entrypoint (local dev) ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)