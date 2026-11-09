"""
conftest.py
Shared fixtures for the test suite.
Author: Amreen
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def sample_raw_df():
    """Minimal valid CMAPSS-style DataFrame reused across test modules."""
    rows = []
    for uid in range(1, 4):
        for cyc in range(1, 51):
            row = {"unit_id": uid, "cycle": cyc,
                   "op1": 0.0, "op2": 0.0, "op3": 100.0}
            sensor_safe = {
                "s1": 518, "s2": 642, "s3": 1582, "s4": 1398, "s5": 14,
                "s6": 21,  "s7": 554, "s8": 2388, "s9": 9046, "s10": 1.3,
                "s11": 47, "s12": 521,"s13": 2388,"s14": 8138,"s15": 8.3,
                "s16": 0.03,"s17": 392,"s18": 2388,"s19": 100,"s20": 38,"s21": 23,
            }
            row.update(sensor_safe)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sample_feature_df(sample_raw_df):
    """Fully featured DataFrame (with RUL labels) for model tests."""
    from src.features.build_features import build_features
    return build_features(sample_raw_df.copy(), is_train=True)
