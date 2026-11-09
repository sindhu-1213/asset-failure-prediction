"""
xgboost_model.py
XGBoost models for both RUL regression and failure classification.
Hyperparameters were tuned via cross-validation in notebook 04.

Author: Amreen
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os


# Tuned params — arrived at after CV in notebook 04
XGB_RUL_PARAMS = {
    "n_estimators":   300,
    "max_depth":      5,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "random_state":   42,
    "n_jobs":         -1,
}

XGB_CLF_PARAMS = {
    "n_estimators":   300,
    "max_depth":      4,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,   # adjusts for class imbalance
    "eval_metric":    "logloss",
    "random_state":   42,
    "n_jobs":         -1,
}


class XGBRULModel:
    """XGBoost regressor for remaining useful life."""

    def __init__(self, params: dict = None):
        self.params = params or XGB_RUL_PARAMS
        self.model  = xgb.XGBRegressor(**self.params)
        self.feature_cols = None

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame = None, y_val: pd.Series = None
    ) -> "XGBRULModel":
        self.feature_cols = list(X_train.columns)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)
        print(f"  saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBRULModel":
        obj = cls()
        data = joblib.load(path)
        obj.model        = data["model"]
        obj.feature_cols = data["feature_cols"]
        return obj


class XGBClassifier:
    """XGBoost classifier for binary failure risk within N cycles."""

    def __init__(self, params: dict = None):
        self.params = params or XGB_CLF_PARAMS
        self.model  = xgb.XGBClassifier(**self.params)
        self.feature_cols = None

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame = None, y_val: pd.Series = None
    ) -> "XGBClassifier":
        self.feature_cols = list(X_train.columns)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)
        print(f"  saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBClassifier":
        obj = cls()
        data = joblib.load(path)
        obj.model        = data["model"]
        obj.feature_cols = data["feature_cols"]
        return obj
