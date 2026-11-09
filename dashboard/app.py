"""
dashboard/app.py
Streamlit fleet health dashboard.
Shows per-unit risk scores, RUL trends, alert list,
and drift monitor status — all from the saved model artifacts.

Run: streamlit run dashboard/app.py

Author: Amreen
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.load_data import load_train, load_test
from src.features.build_features import build_features, get_feature_cols
from src.monitoring.drift_check import DriftMonitor

ARTIFACTS = "models/artifacts"


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    from src.models.xgboost_model import XGBRULModel, XGBClassifier

    rul_path = f"{ARTIFACTS}/xgb_rul.pkl"
    clf_path = f"{ARTIFACTS}/xgb_clf.pkl"

    xgb_rul = XGBRULModel.load(rul_path)   if os.path.exists(rul_path) else None
    xgb_clf = XGBClassifier.load(clf_path) if os.path.exists(clf_path) else None

    threshold_path = f"{ARTIFACTS}/best_threshold.json"
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f).get("threshold", 0.5)
    return xgb_rul, xgb_clf, threshold


@st.cache_data
def load_and_feature_test_data():
    test_df, rul_df = load_test()
    feat_df = build_features(test_df, is_train=False)
    return feat_df, rul_df


# ── helpers ───────────────────────────────────────────────────────────────────

def get_latest_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cycle").groupby("unit_id").last().reset_index()


def risk_color(prob: float) -> str:
    if prob < 0.3:
        return "🟢"
    elif prob < 0.6:
        return "🟡"
    return "🔴"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asset Health Monitor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Asset Health Monitor")
st.caption("Industrial rotating equipment — failure risk & RUL tracker")
st.markdown("---")

# ── load ──────────────────────────────────────────────────────────────────────

xgb_rul, xgb_clf, threshold = load_models()
if xgb_rul is None or xgb_clf is None:
    st.error(
        "Model artifacts not found. "
        "Run notebooks/04_advanced_modeling.ipynb first to train and save models."
    )
    st.stop()

feat_df, rul_df = load_and_feature_test_data()
feat_cols = get_feature_cols(feat_df)
latest_df = get_latest_per_unit(feat_df)

X_latest    = latest_df[feat_cols]
rul_preds   = xgb_rul.predict(X_latest)
risk_probas = xgb_clf.predict_proba(X_latest)

latest_df = latest_df.copy()
latest_df["pred_rul"]     = np.clip(rul_preds, 0, None).round(1)
latest_df["failure_prob"] = risk_probas.round(4)
latest_df["alert"]        = risk_probas >= threshold
latest_df["risk_icon"]    = latest_df["failure_prob"].apply(risk_color)


# ── KPI row ───────────────────────────────────────────────────────────────────

n_total    = len(latest_df)
n_alert    = latest_df["alert"].sum()
n_healthy  = n_total - n_alert
avg_rul    = latest_df["pred_rul"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Units Monitored", n_total)
col2.metric("🔴 High Risk Alerts",   int(n_alert))
col3.metric("🟢 Healthy Units",      int(n_healthy))
col4.metric("Avg Predicted RUL",     f"{avg_rul:.0f} cycles")

st.markdown("---")


# ── fleet table ───────────────────────────────────────────────────────────────

st.subheader("Fleet Status")

display_cols = ["risk_icon", "unit_id", "cycle", "pred_rul", "failure_prob", "alert"]
display_df   = latest_df[display_cols].rename(columns={
    "risk_icon":    "Risk",
    "unit_id":      "Unit ID",
    "cycle":        "Last Cycle",
    "pred_rul":     "Predicted RUL",
    "failure_prob": "Failure Prob",
    "alert":        "Alert",
})

# sort: alerts first, then by failure probability
display_df = display_df.sort_values(["Alert", "Failure Prob"], ascending=[False, False])
st.dataframe(display_df, use_container_width=True, height=300)


# ── unit drilldown ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Unit Drilldown — RUL Trend")

unit_ids = sorted(feat_df["unit_id"].unique())
selected = st.selectbox("Select Unit ID", unit_ids)

unit_data = feat_df[feat_df["unit_id"] == selected].sort_values("cycle")

# predict rul over all cycles for trend line
X_unit = unit_data[feat_cols]
unit_data = unit_data.copy()
unit_data["pred_rul"] = np.clip(xgb_rul.predict(X_unit), 0, None)

col_a, col_b = st.columns(2)

with col_a:
    st.line_chart(
        unit_data.set_index("cycle")[["pred_rul"]],
        use_container_width=True,
        color="#2196F3"
    )
    st.caption("Predicted RUL over operational cycles")

with col_b:
    unit_data["failure_prob"] = xgb_clf.predict_proba(X_unit)
    st.area_chart(
        unit_data.set_index("cycle")[["failure_prob"]],
        use_container_width=True,
        color="#F44336"
    )
    st.caption("Failure probability over time")


# ── alerts panel ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔴 Active Alerts")

alerts = latest_df[latest_df["alert"]].sort_values("failure_prob", ascending=False)
if alerts.empty:
    st.success("No active alerts — all units within safe operating range.")
else:
    for _, row in alerts.iterrows():
        st.warning(
            f"Unit **{int(row['unit_id'])}** — "
            f"Failure prob: **{row['failure_prob']:.1%}** | "
            f"Predicted RUL: **{row['pred_rul']} cycles**"
        )


# ── drift status ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Data Drift Monitor")

drift_path = f"{ARTIFACTS}/last_drift_report.json"
if os.path.exists(drift_path):
    with open(drift_path) as f:
        drift = json.load(f)
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Features Checked",  drift.get("n_features_checked", "-"))
    col_d2.metric("Drifted Features",  drift.get("n_drifted", "-"))
    col_d3.metric("Worst KS Stat",     drift.get("worst_ks", "-"))
    if drift.get("n_drifted", 0) > 0:
        st.warning(
            f"Drift detected in {drift['n_drifted']} features. "
            f"Worst: **{drift.get('worst_feature', 'N/A')}**. "
            "Consider retraining the model."
        )
    else:
        st.success("No significant distribution drift detected.")
else:
    st.info("Run the drift check notebook (06) to populate this panel.")

st.markdown("---")
st.caption("Amreen · Predictive Asset Health MLOps · Data Science Portfolio")
