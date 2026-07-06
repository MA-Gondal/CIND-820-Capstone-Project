# src — Modular Pipeline Scripts

Reusable logic imported by the notebooks (the notebooks narrate; these modules do the work).

| Module | Pipeline stage | What it contains |
|---|---|---|
| `preprocessing.py` | Stages 2–3 | Date parsing, 3-class target engineering, cross-reference recovery for weight/freight, leakage field removal, stratified 80/20 split with saved indices |
| `features.py` | Stage 4 | `FeatureEngineer` class (fit on train only / transform any split): mode-stratified imputation with flags, log1p transforms, chi-squared selection, one-hot encoding, vendor risk tier |
| `models.py` | Stage 5 | Model factory (LR / RF / XGBoost), stratified 5-fold CV loop with fold-level SMOTE, XGBoost early stopping, and per-fold metrics |

All statistics are learned from the training split only — see the fit/transform contract in `features.py`.