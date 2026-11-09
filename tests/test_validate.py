"""
tests/test_validate.py
Unit tests for src/ingestion/validate.py

Author: Amreen
"""

import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.validate import (
    check_schema, check_missing, check_ranges,
    check_cycle_continuity, validate, assert_valid,
    REQUIRED_COLS
)


def _make_df(n_units=2, cycles_each=10) -> pd.DataFrame:
    """Minimal valid CMAPSS-style DataFrame."""
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_each + 1):
            row = {"unit_id": uid, "cycle": cyc, "op1": 0.0, "op2": 0.0, "op3": 100.0}
            for i in range(1, 22):
                row[f"s{i}"] = 500.0  # well within any sensor bound
            rows.append(row)
    return pd.DataFrame(rows)


# ── schema checks ─────────────────────────────────────────────────────────────

def test_schema_clean():
    df = _make_df()
    assert check_schema(df) == []


def test_schema_missing_col():
    df = _make_df().drop(columns=["s3"])
    issues = check_schema(df)
    assert any("s3" in i for i in issues)


def test_schema_multiple_missing():
    df = _make_df().drop(columns=["unit_id", "s1", "op2"])
    issues = check_schema(df)
    assert len(issues) == 3


# ── missing value checks ──────────────────────────────────────────────────────

def test_missing_clean():
    df = _make_df()
    assert check_missing(df) == []


def test_missing_detects_nulls():
    df = _make_df()
    df.loc[df.index[:5], "s4"] = np.nan
    issues = check_missing(df)
    assert any("s4" in i for i in issues)


def test_missing_percentage_in_message():
    df = _make_df(n_units=1, cycles_each=10)   # 10 rows
    df.loc[df.index[:5], "s7"] = np.nan        # 50% missing
    issues = check_missing(df)
    assert any("50.0%" in i for i in issues)


# ── range checks ──────────────────────────────────────────────────────────────

def test_ranges_clean():
    df = _make_df()
    # values well within widened physical bounds
    sensor_safe = {
        "s1": 518, "s2": 642, "s3": 1582, "s4": 1398, "s5": 14,
        "s6": 21,  "s7": 554, "s8": 2388, "s9": 9046, "s10": 1.3,
        "s11": 47, "s12": 521,"s13": 2388,"s14": 8138,"s15": 8.3,
        "s16": 0.03,"s17": 392,"s18": 2388,"s19": 100,"s20": 38,"s21": 23,
    }
    for s, v in sensor_safe.items():
        df[s] = v
    assert check_ranges(df) == []


def test_ranges_flags_out_of_bound():
    df = _make_df()
    df.loc[0, "s1"] = 9999   # way out of range
    issues = check_ranges(df)
    assert any("s1" in i for i in issues)


# ── cycle continuity ──────────────────────────────────────────────────────────

def test_cycle_continuity_clean():
    df = _make_df()
    assert check_cycle_continuity(df) == []


def test_cycle_continuity_gap():
    df = _make_df(n_units=1, cycles_each=10)
    # remove cycle 5 — creates a gap
    df = df[df["cycle"] != 5].reset_index(drop=True)
    issues = check_cycle_continuity(df)
    assert len(issues) > 0


def test_cycle_continuity_wrong_start():
    df = _make_df(n_units=1, cycles_each=5)
    df["cycle"] += 1   # starts at 2 instead of 1
    issues = check_cycle_continuity(df)
    assert any("start" in i.lower() or "1" in i for i in issues)


# ── assert_valid ──────────────────────────────────────────────────────────────

def test_assert_valid_passes():
    df = _make_df()
    assert_valid(df)   # should not raise


def test_assert_valid_raises_on_missing_col():
    df = _make_df().drop(columns=["unit_id"])
    with pytest.raises(ValueError, match="unit_id"):
        assert_valid(df)
