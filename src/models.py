"""
models.py — CIND820 Capstone, Milestone 3
Pipeline Stage 5 (Model Training & Cross-Validation), per Milestone 2 report §3.2

Evaluates three classifiers under two class-imbalance strategies using
stratified 5-fold cross-validation on the training split only:

  Models:
    - Logistic Regression  (baseline;  C=1, class_weight support)
    - Random Forest        (alternate; 100 trees)
    - XGBoost              (champion candidate; early stopping)

  Imbalance strategies:
    - 'class_weight': balanced class weighting (sample weights for XGBoost)
    - 'smote':        SMOTE oversampling applied INSIDE each training fold
                      only (never to the validation fold), k_neighbors=5

The cross-validation loop is written explicitly (rather than via
cross_validate) so that fold-level SMOTE application, XGBoost sample
weights, and early-stopping eval sets are transparent and auditable.

Metrics per fold: macro-averaged F1, macro-averaged AUPRC (one-vs-rest
average precision), training time, inference time.

Author: Mubarak Ahmed (501345730)
"""

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]


def make_model(name: str, strategy: str):
    """
    Instantiate a fresh model configured for the given imbalance strategy.
    Under 'smote' the class_weight parameters are disabled, because the
    resampling itself rebalances the training data.
    """
    balanced = (strategy == "class_weight")
    if name == "LogisticRegression":
        # LR is scale-sensitive (lbfgs convergence), so it is wrapped in a
        # Pipeline with StandardScaler fitted WITHIN each training fold —
        # consistent, leakage-free scaling per the Milestone 2 fair-comparison
        # rule. Tree models are scale-invariant and receive raw features.
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced" if balanced else None,
                max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)),
        ])
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=500, learning_rate=0.1, max_depth=6,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", early_stopping_rounds=50,
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(name)


def _fit_one(model_name, strategy, X_fit, y_fit, X_val_es=None, y_val_es=None):
    """Fit a single model, handling XGBoost early stopping and sample weights."""
    model = make_model(model_name, strategy)
    t0 = time.perf_counter()
    if model_name == "XGBoost":
        # Early stopping monitors a held-out eval set carved from the
        # training fold (never the validation fold, never SMOTE samples).
        sw = (compute_sample_weight("balanced", y_fit)
              if strategy == "class_weight" else None)
        model.fit(X_fit, y_fit, sample_weight=sw,
                  eval_set=[(X_val_es, y_val_es)], verbose=False)
    else:
        model.fit(X_fit, y_fit)
    fit_time = time.perf_counter() - t0
    return model, fit_time


def run_cross_validation(X: pd.DataFrame, y: pd.Series,
                         n_splits: int = N_SPLITS) -> pd.DataFrame:
    """
    Stratified k-fold CV over all (model, strategy) combinations.
    Returns a long-format DataFrame with one row per model-strategy-fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    records = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Carve a 10% early-stopping eval set from the fold's training
        # portion (stratified). Used by XGBoost only; kept un-resampled.
        X_fit0, X_es, y_fit0, y_es = train_test_split(
            X_tr, y_tr, test_size=0.10, stratify=y_tr,
            random_state=RANDOM_STATE)

        for strategy in ["class_weight", "smote"]:
            if strategy == "smote":
                sm = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
                X_fit, y_fit = sm.fit_resample(X_fit0, y_fit0)
            else:
                X_fit, y_fit = X_fit0, y_fit0

            for model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
                model, fit_time = _fit_one(
                    model_name, strategy, X_fit, y_fit, X_es, y_es)

                t0 = time.perf_counter()
                proba = model.predict_proba(X_val)
                pred = np.argmax(proba, axis=1)
                pred_time = time.perf_counter() - t0

                y_bin = label_binarize(y_val, classes=CLASSES)
                records.append({
                    "model": model_name,
                    "strategy": strategy,
                    "fold": fold,
                    "macro_f1": f1_score(y_val, pred, average="macro"),
                    "auprc_macro": average_precision_score(
                        y_bin, proba, average="macro"),
                    "fit_time_s": round(fit_time, 3),
                    "pred_time_s": round(pred_time, 4),
                    "n_train_after_resample": len(X_fit),
                })
    return pd.DataFrame(records)


def summarize_cv(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per model-strategy, sorted by mean macro-F1."""
    g = results.groupby(["model", "strategy"])
    summary = pd.DataFrame({
        "macro_f1_mean": g["macro_f1"].mean().round(4),
        "macro_f1_std": g["macro_f1"].std().round(4),
        "auprc_mean": g["auprc_macro"].mean().round(4),
        "auprc_std": g["auprc_macro"].std().round(4),
        "fit_time_mean_s": g["fit_time_s"].mean().round(2),
        "pred_time_mean_s": g["pred_time_s"].mean().round(4),
    }).sort_values("macro_f1_mean", ascending=False)
    return summary.reset_index()
