# Model Card — Predictive Asset Health System
**Author:** Amreen  
**Last updated:** 2024-12  
**Version:** 1.0.0

---

## Models in this system

| Model | Task | Algorithm | Primary Metric |
|---|---|---|---|
| `xgb_rul` | Remaining Useful Life regression | XGBoost | RMSE (cycles) |
| `xgb_clf` | Near-term failure risk | XGBoost Classifier | AUC + Cost-weighted F1 |
| `lstm_rul` | RUL from full degradation trajectory | LSTM (PyTorch) | RMSE + NASA Score |
| `baseline_rul` | RUL baseline | Linear Regression | RMSE |
| `baseline_clf` | Risk baseline | Logistic Regression | AUC |

---

## Dataset

**Source:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001 subset)  
**Rows:** ~19,000 training samples across 80 engines  
**Features:** 15 informative sensors (6 flat sensors dropped), 3 rolling window sizes, degradation slopes, cumulative deviation  
**Labels:**  
- `rul` — continuous, cycles until failure  
- `will_fail` — binary, 1 if failure within 30 cycles  

---

## Training setup

- Train/validation split: grouped by `unit_id` (80% units train, 20% val) to prevent leakage
- No temporal leakage: test engines have their final RUL withheld
- XGBoost hyperparameters: tuned manually in notebook 04; `scale_pos_weight=3` for class imbalance

---

## Evaluation

| Model | RMSE (cycles) | NASA Score ↓ | AUC | F1 @ op. threshold |
|---|---|---|---|---|
| Linear Regression | ~55 | ~6000 | — | — |
| Logistic Regression | — | — | ~0.82 | ~0.71 |
| XGBoost RUL | ~28 | ~1800 | — | — |
| XGBoost Classifier | — | — | ~0.94 | ~0.83 |
| LSTM RUL | ~24 | ~1500 | — | — |

*Exact numbers depend on random seed and dataset version. Run notebooks 03–04 to reproduce.*

---

## Threshold & cost assumptions

The classification threshold is set by cost minimisation, not default 0.5.

| Cost type | Assumed value | Rationale |
|---|---|---|
| False negative (missed failure) | $50,000 | Unplanned downtime + equipment damage |
| False positive (false alarm) | $5,000 | Unnecessary maintenance visit |

**These are illustrative.** Replace with actual ops cost data before production deployment.  
Threshold is saved to `models/artifacts/best_threshold.json` and used by the API automatically.

---

## Limitations

- Trained on a single operating condition (FD001). Models may degrade on engines with different op profiles.
- Synthetic dataset is representative but not a substitute for real sensor data.
- LSTM inference is slower — not suitable if latency < 50ms is a hard requirement.
- Drift monitor uses KS test which assumes feature independence. Correlated drift may be missed.
- Business impact figures (notebook 05) are estimates under stated assumptions — not guarantees.

---

## Intended use

- Scheduled batch scoring (daily / per shift) for a fleet of industrial rotating equipment
- Alert generation for maintenance scheduling
- Not intended for safety-critical real-time control loops

---

## Retraining trigger

Run notebook 06 drift check. If `pct_drifted > 20%`, schedule retraining on updated data.
