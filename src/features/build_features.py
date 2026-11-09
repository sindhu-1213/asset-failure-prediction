"""
build_features.py
Transforms raw CMAPSS sensor data into model-ready features.

Three layers:
  1. RUL label generation (regression target)
  2. Rolling statistics per sensor per unit
  3. Degradation slope (linear trend in a window)
  4. Binary failure label (classification target)

These were prototyped in notebooks/02_feature_engineering.ipynb
and then moved here once they were stable.

Author: Amreen
"""

import pandas as pd
import numpy as np
from typing import List, Optional

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

# sensors that are flat in CMAPSS — drop them, they only add noise
FLAT_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}
INFORMATIVE_SENSORS = [s for s in SENSOR_COLS if s not in FLAT_SENSORS]

WINDOW_SIZES = [5, 15, 30]   # rolling window sizes (cycles)
FAILURE_HORIZON = 30          # binary label: 1 if RUL <= this


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for each row in the training set.
    RUL = max_cycle_for_unit - current_cycle
    """
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_failure_label(df: pd.DataFrame, horizon: int = FAILURE_HORIZON) -> pd.DataFrame:
    """Binary label: 1 if engine will fail within `horizon` cycles."""
    if "rul" not in df.columns:
        raise ValueError("Run add_rul_labels() before add_failure_label()")
    df = df.copy()
    df["will_fail"] = (df["rul"] <= horizon).astype(int)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Per unit, add rolling mean and std for each informative sensor.
    Groups must be sorted by cycle before calling this.
    """
    df = df.copy()
    sensors = sensors or INFORMATIVE_SENSORS
    windows = windows or WINDOW_SIZES

    for sensor in sensors:
        for w in windows:
            col_mean = f"{sensor}_roll{w}_mean"
            col_std  = f"{sensor}_roll{w}_std"
            df[col_mean] = (
                df.groupby("unit_id")[sensor]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[col_std] = (
                df.groupby("unit_id")[sensor]
                .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )
    return df


def add_degradation_slope(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    window: int = 15
) -> pd.DataFrame:
    """
    Linear slope of each sensor over the last `window` cycles.
    Positive slope on a sensor that increases toward failure = useful feature.
    """
    df = df.copy()
    sensors = sensors or INFORMATIVE_SENSORS

    def slope(series: pd.Series) -> pd.Series:
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            chunk = vals[start: i + 1]
            if len(chunk) < 2:
                result[i] = 0.0
                continue
            x = np.arange(len(chunk), dtype=float)
            # quick linear regression: slope = cov(x,y)/var(x)
            x -= x.mean()
            y = chunk - chunk.mean()
            denom = (x * x).sum()
            result[i] = float((x * y).sum() / denom) if denom != 0 else 0.0
        return pd.Series(result, index=series.index)

    for sensor in sensors:
        df[f"{sensor}_slope{window}"] = (
            df.groupby("unit_id")[sensor].transform(slope)
        )
    return df


def add_cumulative_degradation(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Cumulative absolute change from first observed value per unit.
    Captures total wear accumulated so far.
    """
    df = df.copy()
    sensors = sensors or INFORMATIVE_SENSORS
    for sensor in sensors:
        first_val = df.groupby("unit_id")[sensor].transform("first")
        df[f"{sensor}_cumdev"] = (df[sensor] - first_val).abs().groupby(df["unit_id"]).cumsum()
        # normalise by cycle so different-length units are comparable
        df[f"{sensor}_cumdev"] = df[f"{sensor}_cumdev"] / df["cycle"]
    return df


def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
    failure_horizon: int = FAILURE_HORIZON
) -> pd.DataFrame:
    """
    Master feature builder. Call this for both train and test sets.
    For test set (is_train=False), skip label generation.
    Returns the fully-featured DataFrame.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    if is_train:
        df = add_rul_labels(df)
        df = add_failure_label(df, horizon=failure_horizon)

    df = add_rolling_features(df)
    df = add_degradation_slope(df)
    df = add_cumulative_degradation(df)

    # drop flat sensors — they're confirmed uninformative in EDA
    df.drop(columns=[s for s in FLAT_SENSORS if s in df.columns], inplace=True)

    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the model input feature columns (excludes labels and metadata)."""
    exclude = {"unit_id", "cycle", "op1", "op2", "op3", "rul", "will_fail"}
    return [c for c in df.columns if c not in exclude]
