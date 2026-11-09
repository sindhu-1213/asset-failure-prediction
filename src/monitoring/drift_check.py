"""
drift_check.py
Monitors incoming sensor feature distributions against a reference
(training data) using the Kolmogorov-Smirnov test.

In production, this would run on a sliding window of recent predictions.
Here it's implemented as a callable that can be wired to any scheduler.

Author: Amreen
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Optional
import json
import os


class DriftMonitor:
    """
    Fits on training data, then compares any new batch against
    the stored reference distributions.

    Usage:
        monitor = DriftMonitor(p_threshold=0.05)
        monitor.fit(train_feature_df)
        report = monitor.check(new_batch_df)
        monitor.save("models/artifacts/drift_monitor.json")
    """

    def __init__(self, p_threshold: float = 0.05):
        self.p_threshold  = p_threshold   # below this → flag as drifted
        self.reference_   = {}            # {feature: np.array of reference values}
        self.feature_cols = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[list] = None) -> "DriftMonitor":
        """Store reference distributions from training data."""
        self.feature_cols = feature_cols or [
            c for c in df.columns
            if c not in {"unit_id", "cycle", "rul", "will_fail", "op1", "op2", "op3"}
        ]
        for col in self.feature_cols:
            self.reference_[col] = df[col].dropna().values
        print(f"DriftMonitor fitted on {len(self.feature_cols)} features, "
              f"{len(df)} reference samples.")
        return self

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run KS test for each feature.
        Returns a DataFrame with one row per feature:
          feature | ks_stat | p_value | drifted
        """
        rows = []
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            new_vals = df[col].dropna().values
            if len(new_vals) < 10:
                # too few samples — skip rather than flag spuriously
                continue
            stat, p = ks_2samp(self.reference_[col], new_vals)
            rows.append({
                "feature": col,
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "drifted": p < self.p_threshold,
            })
        report = pd.DataFrame(rows)
        n_drifted = report["drifted"].sum()
        pct = 100 * n_drifted / max(len(report), 1)
        print(f"Drift check: {n_drifted}/{len(report)} features drifted ({pct:.1f}%)")
        if n_drifted > 0:
            print("  Drifted features:")
            drifted = report[report["drifted"]].sort_values("ks_stat", ascending=False)
            for _, row in drifted.iterrows():
                print(f"    {row['feature']:35s}  KS={row['ks_stat']:.4f}  p={row['p_value']:.6f}")
        return report

    def summary(self, report: pd.DataFrame) -> dict:
        return {
            "n_features_checked": len(report),
            "n_drifted":          int(report["drifted"].sum()),
            "pct_drifted":        round(100 * report["drifted"].mean(), 1),
            "worst_feature":      report.loc[report["ks_stat"].idxmax(), "feature"] if len(report) else None,
            "worst_ks":           float(report["ks_stat"].max()) if len(report) else 0.0,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "p_threshold":  self.p_threshold,
            "feature_cols": self.feature_cols,
            # store as list for JSON serialisability
            "reference_":   {k: v.tolist() for k, v in self.reference_.items()}
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"  DriftMonitor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DriftMonitor":
        with open(path) as f:
            payload = json.load(f)
        obj = cls(p_threshold=payload["p_threshold"])
        obj.feature_cols = payload["feature_cols"]
        obj.reference_   = {k: np.array(v) for k, v in payload["reference_"].items()}
        return obj
