"""
validate.py
Schema checks, range checks, and missing value detection for CMAPSS data.
Returns a dict of issues rather than raising immediately — caller decides what to do.

Author: Amreen
"""

import pandas as pd
import numpy as np
from typing import Dict, List

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS     = ["op1", "op2", "op3"]
REQUIRED_COLS = ["unit_id", "cycle"] + OP_COLS + SENSOR_COLS

# Physical safety bounds — values outside these are almost certainly corrupt/sensor failure.
# These are intentionally wide: degradation moves sensors but shouldn't exceed these limits.
SENSOR_BOUNDS = {
    "s1":  (300,  800),   "s2":  (400,  900),   "s3":  (800,  4500),
    "s4":  (800,  3500),  "s5":  (0,    50),     "s6":  (0,    50),
    "s7":  (300,  800),   "s8":  (1500, 3500),   "s9":  (5000, 13000),
    "s10": (0,    5),     "s11": (20,   100),    "s12": (100,  800),
    "s13": (1500, 3500),  "s14": (4000, 13000),  "s15": (2,    20),
    "s16": (0,    1),     "s17": (0,    600),    "s18": (1500, 3500),
    "s19": (20,   200),   "s20": (15,   80),     "s21": (5,    60),
}


def check_schema(df: pd.DataFrame) -> List[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return [f"Missing column: {c}" for c in missing]


def check_missing(df: pd.DataFrame) -> List[str]:
    issues = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            pct = 100 * n / len(df)
            issues.append(f"{col}: {n} nulls ({pct:.1f}%)")
    return issues


def check_ranges(df: pd.DataFrame) -> List[str]:
    issues = []
    for s, (lo, hi) in SENSOR_BOUNDS.items():
        if s not in df.columns:
            continue
        out = ((df[s] < lo) | (df[s] > hi)).sum()
        if out > 0:
            issues.append(f"{s}: {out} values outside [{lo}, {hi}]")
    return issues


def check_cycle_continuity(df: pd.DataFrame) -> List[str]:
    """Each engine unit should have consecutive cycles starting from 1."""
    issues = []
    for uid, grp in df.groupby("unit_id"):
        cycles = grp["cycle"].sort_values().values
        if cycles[0] != 1:
            issues.append(f"unit {uid}: cycles don't start at 1 (start={cycles[0]})")
        gaps = np.diff(cycles)
        if (gaps != 1).any():
            issues.append(f"unit {uid}: non-consecutive cycles detected")
    return issues


def validate(df: pd.DataFrame, check_cycles: bool = True) -> Dict[str, List[str]]:
    """
    Run all checks. Returns dict with keys: schema, missing, ranges, cycles.
    Empty list means no issues in that category.
    """
    result = {
        "schema":  check_schema(df),
        "missing": check_missing(df),
        "ranges":  check_ranges(df),
    }
    if check_cycles:
        result["cycles"] = check_cycle_continuity(df)
    return result


def assert_valid(df: pd.DataFrame, label: str = "data") -> None:
    """Raise ValueError if any critical issues found."""
    issues = validate(df, check_cycles="unit_id" in df.columns)
    critical = issues["schema"] + issues["missing"]
    if critical:
        msg = "\n".join(critical)
        raise ValueError(f"Validation failed for {label}:\n{msg}")
    # warnings only for range / cycle issues
    for key in ("ranges", "cycles"):
        for w in issues.get(key, []):
            print(f"[WARN] {label} - {w}")
