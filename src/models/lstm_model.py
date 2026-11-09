"""
lstm_model.py
LSTM-based sequence model for RUL regression.
Takes a sliding window of sensor readings per engine unit
and learns temporal degradation patterns that the tabular
XGBoost model can't capture.

Architecture: LSTM → Dropout → FC → ReLU → FC → scalar RUL

Author: Amreen
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 30   # look-back window in cycles


# ── Dataset ──────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Builds fixed-length windows from per-unit sensor time series.
    Each sample: (seq_len, n_features) → scalar RUL
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        label_col: str = "rul",
        seq_len: int = SEQ_LEN
    ):
        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.sequences    = []
        self.labels       = []

        for uid, grp in df.groupby("unit_id"):
            grp = grp.sort_values("cycle")
            vals   = grp[feature_cols].values.astype(np.float32)
            labels = grp[label_col].values.astype(np.float32)
            for i in range(len(grp)):
                start = max(0, i - seq_len + 1)
                seq   = vals[start: i + 1]
                # left-pad short sequences with the first row
                if len(seq) < seq_len:
                    pad = np.repeat(seq[[0]], seq_len - len(seq), axis=0)
                    seq = np.vstack([pad, seq])
                self.sequences.append(seq)
                self.labels.append(labels[i])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.labels[idx])
        )


# ── Model architecture ────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]   # take final timestep
        return self.head(last).squeeze(-1)


# ── Wrapper ───────────────────────────────────────────────────────────────────

class LSTMRULModel:
    """
    Wraps LSTMNet with sklearn-style fit/predict interface.
    Handles normalisation internally so callers don't need to worry about it.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        epochs:      int = 30,
        batch_size:  int = 256,
        lr:          float = 1e-3,
        seq_len:     int = SEQ_LEN,
        device:      str = DEVICE
    ):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.seq_len     = seq_len
        self.device      = device

        self.net          = None
        self.feature_cols = None
        self.mean_        = None
        self.std_         = None
        self.history      = []   # train loss per epoch

    # normalise using training stats
    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.feature_cols] = (df[self.feature_cols] - self.mean_) / (self.std_ + 1e-8)
        return df

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list,
        val_df: Optional[pd.DataFrame] = None
    ) -> "LSTMRULModel":
        self.feature_cols = feature_cols
        self.mean_ = train_df[feature_cols].mean()
        self.std_  = train_df[feature_cols].std()

        train_norm = self._normalise(train_df)
        train_ds   = SequenceDataset(train_norm, feature_cols, seq_len=self.seq_len)
        train_dl   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.net = LSTMNet(
            input_size  = len(feature_cols),
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout
        ).to(self.device)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimiser.zero_grad()
                preds = self.net(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item() * len(y_batch)

            avg_loss = total_loss / len(train_ds)
            self.history.append(avg_loss)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs}  train_loss={avg_loss:.2f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        df must contain unit_id, cycle, and feature_cols.
        Returns per-row RUL predictions (same row order as df).
        """
        norm_df  = self._normalise(df)
        # use label_col="rul" as a dummy if not present
        has_rul  = "rul" in df.columns
        if not has_rul:
            norm_df = norm_df.copy()
            norm_df["rul"] = 0.0

        ds = SequenceDataset(norm_df, self.feature_cols, label_col="rul", seq_len=self.seq_len)
        dl = DataLoader(ds, batch_size=512, shuffle=False)

        self.net.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self.device)
                out = self.net(X_batch).cpu().numpy()
                preds.extend(out.tolist())

        return np.clip(np.array(preds), 0, None)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "state_dict":   self.net.state_dict(),
            "feature_cols": self.feature_cols,
            "mean_":        self.mean_,
            "std_":         self.std_,
            "config": {
                "hidden_size": self.hidden_size,
                "num_layers":  self.num_layers,
                "dropout":     self.dropout,
                "seq_len":     self.seq_len,
            }
        }
        torch.save(payload, path)
        print(f"  saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = DEVICE) -> "LSTMRULModel":
        payload = torch.load(path, map_location=device)
        cfg     = payload["config"]
        obj     = cls(device=device, **cfg)
        obj.feature_cols = payload["feature_cols"]
        obj.mean_        = payload["mean_"]
        obj.std_         = payload["std_"]
        obj.net = LSTMNet(
            input_size  = len(obj.feature_cols),
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"]
        ).to(device)
        obj.net.load_state_dict(payload["state_dict"])
        obj.net.eval()
        return obj
