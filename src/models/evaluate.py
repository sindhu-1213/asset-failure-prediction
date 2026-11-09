"""
evaluate.py
Evaluation utilities for both the RUL regressor and the failure classifier.

Key design decision: metrics are presented in two layers —
  1. Standard ML metrics (RMSE, AUC, F1 etc.)
  2. Business translation — what does this actually mean in downtime/cost terms

The asymmetric scoring function (nasa_score) is standard for CMAPSS evaluation.
Late predictions (predicting failure too late) are penalised more than early ones.
This reflects real operational risk: missing a failure is worse than a false alarm.

Author: Amreen
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
from typing import Optional


# ── RUL regression ────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official CMAPSS scoring function.
    d = predicted_rul - true_rul  (positive = predicting too optimistically)
    score = sum(exp(-d/13) - 1) for d < 0  (early prediction, less penalty)
            sum(exp(d/10)  - 1) for d >= 0 (late prediction, more penalty)
    Lower is better.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())


def evaluate_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model"
) -> dict:
    metrics = {
        "rmse":       rmse(y_true, y_pred),
        "mae":        mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
    print(f"\n── {label} — RUL Regression ──")
    print(f"  RMSE        : {metrics['rmse']:.2f} cycles")
    print(f"  MAE         : {metrics['mae']:.2f} cycles")
    print(f"  NASA Score  : {metrics['nasa_score']:.0f}  (lower is better)")
    return metrics


# ── Classification ────────────────────────────────────────────────────────────

def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model"
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "auc":       float(roc_auc_score(y_true, y_proba)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n── {label} — Failure Classifier (threshold={threshold}) ──")
    print(f"  AUC         : {metrics['auc']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return metrics


# ── Cost-based threshold selection ───────────────────────────────────────────

def cost_based_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 50_000,   # cost of missed failure (unplanned downtime)
    cost_fp: float = 5_000,    # cost of false alarm (unnecessary maintenance visit)
    n_thresholds: int = 100
) -> tuple[float, pd.DataFrame]:
    """
    Sweep thresholds from 0→1, compute expected cost at each point.
    Returns (best_threshold, results_df).

    Default cost assumptions (document these in model card):
      - Unplanned failure event:  $50,000 (equipment damage + downtime)
      - Unnecessary maintenance:  $5,000  (labour + parts + lost production)
    These are illustrative — should be replaced with real ops data.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        total_cost = fn * cost_fn + fp * cost_fp
        rows.append({
            "threshold": round(t, 3),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "total_cost": total_cost,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0,
        })
    df = pd.DataFrame(rows)
    best_idx = df["total_cost"].idxmin()
    best_t   = float(df.loc[best_idx, "threshold"])

    print(f"\n── Cost-Based Threshold Selection ──")
    print(f"  FN cost (missed failure) : ${cost_fn:,.0f}")
    print(f"  FP cost (false alarm)    : ${cost_fp:,.0f}")
    print(f"  Best threshold           : {best_t}")
    print(f"  Min expected cost / batch: ${df.loc[best_idx, 'total_cost']:,.0f}")
    return best_t, df


# ── Business impact summary ──────────────────────────────────────────────────

def business_impact_summary(
    n_units_fleet: int,
    recall: float,
    precision: float,
    failure_rate_per_year: float = 0.15,
    cost_per_failure: float = 50_000,
    cost_per_false_alarm: float = 5_000
) -> dict:
    """
    Rough annualised business impact estimate.
    State all assumptions explicitly — this is what the business cares about.
    """
    expected_failures = n_units_fleet * failure_rate_per_year
    failures_caught   = expected_failures * recall
    false_alarms      = (n_units_fleet - expected_failures) * (1 - precision)

    cost_saved   = failures_caught   * cost_per_failure
    cost_wasted  = false_alarms      * cost_per_false_alarm
    net_benefit  = cost_saved - cost_wasted

    summary = {
        "fleet_size":          n_units_fleet,
        "expected_failures_yr":round(expected_failures, 1),
        "failures_caught_yr":  round(failures_caught, 1),
        "false_alarms_yr":     round(false_alarms, 1),
        "cost_saved_usd":      round(cost_saved),
        "cost_wasted_usd":     round(cost_wasted),
        "net_benefit_usd":     round(net_benefit),
    }

    print("\n── Estimated Annual Business Impact ──")
    print(f"  Fleet size              : {n_units_fleet} units")
    print(f"  Expected failures / yr  : {expected_failures:.1f}")
    print(f"  Failures caught         : {failures_caught:.1f}  (recall={recall:.2f})")
    print(f"  False alarms / yr       : {false_alarms:.1f}")
    print(f"  Cost avoided            : ${cost_saved:,.0f}")
    print(f"  Wasted maintenance cost : ${cost_wasted:,.0f}")
    print(f"  NET benefit             : ${net_benefit:,.0f} / year")
    print("  (Assumptions: all figures are illustrative estimates)")
    return summary
