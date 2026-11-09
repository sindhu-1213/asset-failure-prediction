"""
tests/test_features.py
Tests for src/features/build_features.py

Author: Amreen
"""

import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import (
    add_rul_labels, add_failure_label, add_rolling_features,
    add_degradation_slope, build_features, get_feature_cols,
    FLAT_SENSORS, INFORMATIVE_SENSORS, FAILURE_HORIZON
)


def _make_sensor_df(n_units=3, cycles_each=50) -> pd.DataFrame:
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_each + 1):
            row = {"unit_id": uid, "cycle": cyc, "op1": 0.0, "op2": 0.0, "op3": 100.0}
            for s in [f"s{i}" for i in range(1, 22)]:
                row[s] = np.random.normal(500, 5)
            rows.append(row)
    return pd.DataFrame(rows)


# ── RUL labels ────────────────────────────────────────────────────────────────

def test_rul_added():
    df = _make_sensor_df(n_units=2, cycles_each=30)
    df = add_rul_labels(df)
    assert "rul" in df.columns


def test_rul_last_cycle_is_zero():
    df = _make_sensor_df(n_units=2, cycles_each=30)
    df = add_rul_labels(df)
    for uid, grp in df.groupby("unit_id"):
        assert grp["rul"].min() == 0, f"unit {uid} last RUL should be 0"


def test_rul_first_cycle_equals_maxcycle_minus_1():
    df = _make_sensor_df(n_units=1, cycles_each=20)
    df = add_rul_labels(df)
    # first cycle should have rul = 19 (max_cycle=20, cycle=1)
    first_rul = df[df["cycle"] == 1]["rul"].values[0]
    assert first_rul == 19


def test_rul_non_negative():
    df = _make_sensor_df()
    df = add_rul_labels(df)
    assert (df["rul"] >= 0).all()


# ── failure label ─────────────────────────────────────────────────────────────

def test_failure_label_added():
    df = _make_sensor_df(n_units=2, cycles_each=50)
    df = add_rul_labels(df)
    df = add_failure_label(df, horizon=FAILURE_HORIZON)
    assert "will_fail" in df.columns


def test_failure_label_binary():
    df = _make_sensor_df(n_units=2, cycles_each=50)
    df = add_rul_labels(df)
    df = add_failure_label(df)
    assert set(df["will_fail"].unique()).issubset({0, 1})


def test_failure_label_consistent_with_rul():
    df = _make_sensor_df(n_units=1, cycles_each=50)
    df = add_rul_labels(df)
    df = add_failure_label(df, horizon=20)
    # rows where rul <= 20 should all be labelled 1
    mask = df["rul"] <= 20
    assert df.loc[mask, "will_fail"].all()
    assert not df.loc[~mask, "will_fail"].any()


def test_failure_label_requires_rul():
    df = _make_sensor_df()
    with pytest.raises(ValueError, match="rul"):
        add_failure_label(df)


# ── rolling features ──────────────────────────────────────────────────────────

def test_rolling_features_created():
    df = _make_sensor_df(n_units=2, cycles_each=40)
    df = add_rolling_features(df, sensors=["s2", "s3"], windows=[5, 15])
    for s in ["s2", "s3"]:
        for w in [5, 15]:
            assert f"{s}_roll{w}_mean" in df.columns
            assert f"{s}_roll{w}_std"  in df.columns


def test_rolling_no_cross_unit_leakage():
    """Rolling stats must not bleed across unit boundaries."""
    df = _make_sensor_df(n_units=2, cycles_each=20)
    df = add_rolling_features(df, sensors=["s2"], windows=[5])
    # first row of unit 2 should not be influenced by unit 1's values
    unit2_first = df[(df["unit_id"] == 2) & (df["cycle"] == 1)]["s2_roll5_mean"].values[0]
    unit2_s2    = df[(df["unit_id"] == 2) & (df["cycle"] == 1)]["s2"].values[0]
    assert unit2_first == pytest.approx(unit2_s2, rel=1e-3)


def test_rolling_std_non_negative():
    df = _make_sensor_df(n_units=2, cycles_each=30)
    df = add_rolling_features(df, sensors=["s2"], windows=[5])
    assert (df["s2_roll5_std"] >= 0).all()


# ── degradation slope ─────────────────────────────────────────────────────────

def test_slope_col_created():
    df = _make_sensor_df(n_units=1, cycles_each=20)
    df = add_degradation_slope(df, sensors=["s2"], window=10)
    assert "s2_slope10" in df.columns


def test_slope_flat_signal_near_zero():
    df = _make_sensor_df(n_units=1, cycles_each=20)
    df["s2"] = 500.0   # perfectly flat
    df = add_degradation_slope(df, sensors=["s2"], window=10)
    assert df["s2_slope10"].abs().max() < 1e-6


# ── build_features master function ───────────────────────────────────────────

def test_build_features_drops_flat_sensors():
    df = _make_sensor_df(n_units=2, cycles_each=40)
    df = build_features(df, is_train=True)
    for s in FLAT_SENSORS:
        assert s not in df.columns, f"Flat sensor {s} should have been dropped"


def test_build_features_has_labels_when_train():
    df = _make_sensor_df(n_units=2, cycles_each=40)
    df = build_features(df, is_train=True)
    assert "rul"       in df.columns
    assert "will_fail" in df.columns


def test_build_features_no_labels_when_test():
    df = _make_sensor_df(n_units=2, cycles_each=40)
    df = build_features(df, is_train=False)
    assert "rul"       not in df.columns
    assert "will_fail" not in df.columns


def test_get_feature_cols_excludes_metadata():
    df = _make_sensor_df(n_units=2, cycles_each=40)
    df = build_features(df, is_train=True)
    feat_cols = get_feature_cols(df)
    for exclude in ["unit_id", "cycle", "rul", "will_fail", "op1", "op2", "op3"]:
        assert exclude not in feat_cols
