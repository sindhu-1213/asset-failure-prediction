"""
load_data.py
Reads the CMAPSS-style CSV files from data/raw/ and returns clean DataFrames.
Handles both real NASA CMAPSS (space-delimited .txt) and the generated .csv version.

Author: Amreen
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS     = ["op1", "op2", "op3"]
ALL_COLS    = ["unit_id", "cycle"] + OP_COLS + SENSOR_COLS


def _read_file(path: str) -> pd.DataFrame:
    """Reads either a space-delimited NASA .txt or a comma-delimited .csv."""
    if path.endswith(".txt"):
        df = pd.read_csv(path, sep=r"\s+", header=None)
        # NASA txt files have 2 trailing empty cols sometimes
        df = df.iloc[:, :len(ALL_COLS)]
        df.columns = ALL_COLS
    else:
        df = pd.read_csv(path)
    return df


def load_train(data_dir: str = "data/raw", fd: str = "FD001") -> pd.DataFrame:
    """
    Load training data for a given CMAPSS subset (FD001..FD004).
    Returns a DataFrame sorted by unit_id and cycle.
    """
    for ext in [".csv", ".txt"]:
        path = os.path.join(data_dir, f"train_{fd}{ext}")
        if os.path.exists(path):
            logger.info(f"Loading train data from {path}")
            df = _read_file(path)
            df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            return df
    raise FileNotFoundError(
        f"No train file found for {fd} in {data_dir}. "
        "Run scripts/generate_cmapss_data.py first."
    )


def load_test(data_dir: str = "data/raw", fd: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test data + ground truth RUL.
    Returns (test_df, rul_df).
    """
    test_df, rul_df = None, None
    for ext in [".csv", ".txt"]:
        tp = os.path.join(data_dir, f"test_{fd}{ext}")
        rp = os.path.join(data_dir, f"rul_{fd}{ext}")
        if os.path.exists(tp):
            test_df = _read_file(tp)
            test_df = test_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        if os.path.exists(rp):
            rul_df = pd.read_csv(rp) if rp.endswith(".csv") else pd.read_csv(rp, sep=r"\s+", header=None, names=["true_rul"])

    if test_df is None:
        raise FileNotFoundError(f"No test file found for {fd} in {data_dir}")

    return test_df, rul_df
