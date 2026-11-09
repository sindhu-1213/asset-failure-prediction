"""
Synthetic CMAPSS-style dataset generator.
Mimics the NASA C-MAPSS Turbofan Engine Degradation dataset structure
so the rest of the pipeline works identically whether you use real CMAPSS
or this generated version.

Real dataset: https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip
Place the real files (train_FD001.txt etc.) in data/raw/ to use them instead.

Author: Amreen
"""

import numpy as np
import pandas as pd
import os
import argparse

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
COLS = ["unit_id", "cycle"] + OP_COLS + SENSOR_COLS

# rough sensor baselines loosely based on FD001 — engines run under one op condition
SENSOR_BASELINE = {
    "s1": 518.67, "s2": 642.68, "s3": 1582.09, "s4": 1398.21, "s5": 14.62,
    "s6": 21.61,  "s7": 554.36, "s8": 2388.01, "s9": 9046.19, "s10": 1.30,
    "s11": 47.20, "s12": 521.72,"s13": 2388.04,"s14": 8138.62,"s15": 8.3197,
    "s16": 0.03,  "s17": 392.0, "s18": 2388.0, "s19": 100.0,  "s20": 38.86,
    "s21": 23.35
}

# sensors known to be flat/uninformative in real CMAPSS (good EDA catch)
FLAT_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}

# sensors that degrade — direction of drift toward failure
DEGRADING = {
    "s2": +0.002, "s3": +0.008, "s4": +0.006, "s11": +0.003,
    "s12": -0.003, "s13": +0.002, "s15": +0.001, "s17": -0.005,
    "s20": +0.002, "s21": +0.001
}

NOISE_STD = {s: v * 0.002 for s, v in SENSOR_BASELINE.items()}


def generate_engine(unit_id: int, max_life: int, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cyc in range(1, max_life + 1):
        frac = cyc / max_life  # degradation progress 0→1
        row = {"unit_id": unit_id, "cycle": cyc,
               "op1": round(rng.normal(0, 0.0002), 4),
               "op2": round(rng.normal(0, 0.0003), 4),
               "op3": 100.0}
        for s, base in SENSOR_BASELINE.items():
            noise = rng.normal(0, NOISE_STD[s])
            if s in FLAT_SENSORS:
                row[s] = round(base + noise * 0.01, 4)
            elif s in DEGRADING:
                drift = DEGRADING[s] * base * frac * max_life * 0.5
                row[s] = round(base + drift + noise, 4)
            else:
                row[s] = round(base + noise, 4)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLS)


def generate_dataset(
    n_train: int = 80,
    n_test: int = 20,
    min_life: int = 120,
    max_life: int = 350,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_lives = rng.integers(min_life, max_life, size=n_train)
    test_lives  = rng.integers(min_life, max_life, size=n_test)
    # test engines are truncated — we don't observe the failure
    test_truncate = rng.uniform(0.4, 0.9, size=n_test)

    train_dfs = []
    for i, life in enumerate(train_lives, start=1):
        df = generate_engine(i, int(life), seed=seed + i)
        train_dfs.append(df)
    train = pd.concat(train_dfs, ignore_index=True)

    test_dfs = []
    rul_rows = []
    for i, (life, trunc) in enumerate(zip(test_lives, test_truncate), start=1):
        observed = max(int(life * trunc), 30)
        true_rul = life - observed
        df = generate_engine(i, observed, seed=seed + 1000 + i)
        test_dfs.append(df)
        rul_rows.append({"unit_id": i, "true_rul": true_rul})
    test = pd.concat(test_dfs, ignore_index=True)
    rul_df = pd.DataFrame(rul_rows)

    return train, test, rul_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test",  type=int, default=20)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Generating synthetic CMAPSS dataset  (seed={args.seed}) ...")
    train, test, rul = generate_dataset(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    train.to_csv(f"{args.out}/train_FD001.csv", index=False)
    test.to_csv(f"{args.out}/test_FD001.csv",  index=False)
    rul.to_csv(f"{args.out}/rul_FD001.csv",    index=False)
    print(f"  train : {train.shape}  ->  {args.out}/train_FD001.csv")
    print(f"  test  : {test.shape}   ->  {args.out}/test_FD001.csv")
    print(f"  rul   : {rul.shape}    ->  {args.out}/rul_FD001.csv")
    print("Done.")
