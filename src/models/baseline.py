"""
baseline.py
Baseline models — the floor every advanced model must beat.
  - LinearRegressionRUL  : predicts remaining useful life
  - LogisticRiskClassifier : predicts binary near-term failure risk

Keeping these intentionally simple. The point is an honest comparison,
not a strawman.

Author: Amreen
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


class BaselineRULModel:
    """Linear regression baseline for RUL prediction."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ])
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineRULModel":
        self.feature_cols = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X)
        # RUL can't be negative physically
        return np.clip(preds, 0, None)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_cols": self.feature_cols}, path)
        print(f"  saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineRULModel":
        obj = cls()
        data = joblib.load(path)
        obj.pipeline     = data["pipeline"]
        obj.feature_cols = data["feature_cols"]
        return obj


class BaselineClassifier:
    """Logistic regression baseline for binary failure risk."""

    def __init__(self, max_iter: int = 1000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(max_iter=max_iter, class_weight="balanced"))
        ])
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineClassifier":
        self.feature_cols = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_cols": self.feature_cols}, path)
        print(f"  saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineClassifier":
        obj = cls()
        data = joblib.load(path)
        obj.pipeline     = data["pipeline"]
        obj.feature_cols = data["feature_cols"]
        return obj
