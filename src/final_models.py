"""
final_models.py — CIND820 Capstone, Milestone 4
Final modelling stage: hyperparameter tuning, repeated cross-validation,
and deterministic final model fitting.

Extends the Milestone 3 modelling stage (src/models.py) in three ways,
each motivated by a weakness identified in the Milestone 3 report §5.4:

  1. Hyperparameter tuning of the XGBoost champion — motivated by the
     exhausted early-stopping budget observed in Milestone 3 (the model
     consumed all 500 boosting rounds, indicating headroom).
     Implemented as an explicit, auditable random-search loop (20
     configurations sampled with a fixed seed from a declared grid),
     each scored by stratified 5-fold CV with balanced sample weights —
     the same fold protocol and imbalance strategy as Milestone 3.

  2. Repeated stratified cross-validation (3 repeats × 5 folds = 15
     paired scores) for the champion-vs-baseline comparison — motivated
     by the five-pair minimum sample of the Milestone 3 Wilcoxon test,
     whose p-value sat exactly at the one-sided floor (0.03125).

  3. Deterministic final fitting (n_jobs=1, fixed seed, no early
     stopping in the final refit) — motivated by small run-to-run metric
     variation observed with multithreaded histogram training, so that
     the committed model artifact and every reported test number are
     exactly reproducible.

Leakage controls carried over from Milestone 3: all tuning and repeated
CV operate on the training split only; the held-out test set is touched
once, in Notebook 07, by the final fitted models.

Author: Mubarak Ahmed (501345730)
"""

import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42
CLASSES = [0, 1, 2]

# ---------------------------------------------------------------------------
# 1. Hyperparameter tuning
# ---------------------------------------------------------------------------

# Declared search grid. Ranges bracket the Milestone 3 defaults
# (500 trees, lr=0.1, depth 6) on both sides, so the search can confirm
# or reject the defaults rather than assume them.
PARAM_GRID = {
    "n_estimators":     [300, 600, 900],
    "learning_rate":    [0.05, 0.10],
    "max_depth":        [4, 6, 8],
    "min_child_weight": [1, 5],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda":       [1.0, 5.0],
}
N_CONFIGS = 20  # random-search sample size


def sample_configs(n_configs: int = N_CONFIGS, seed: int = RANDOM_STATE):
    """Sample n unique configurations from PARAM_GRID with a fixed seed."""
    keys = list(PARAM_GRID)
    all_combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_combos), size=n_configs, replace=False)
    return [dict(zip(keys, all_combos[i])) for i in sorted(idx)]


def _make_xgb(params: dict, early_stopping: bool = True) -> XGBClassifier:
    kwargs = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=1, **params)
    if early_stopping:
        kwargs["early_stopping_rounds"] = 50
    return XGBClassifier(**kwargs)


def tune_xgboost(X: pd.DataFrame, y: pd.Series,
                 n_splits: int = 5) -> pd.DataFrame:
    """
    Score each sampled configuration by stratified 5-fold CV
    (class-weight strategy, macro-F1 primary metric).
    Returns one row per configuration, ranked by mean macro-F1.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))
    rows = []
    for cfg_id, params in enumerate(sample_configs(), start=1):
        f1s, fit_times, best_iters = [], [], []
        for tr_idx, val_idx in folds:
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            # Same early-stopping protocol as Milestone 3: monitor a 10%
            # stratified carve-out of the fold's training portion.
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_tr, y_tr, test_size=0.10, stratify=y_tr,
                random_state=RANDOM_STATE)
            sw = compute_sample_weight("balanced", y_fit)
            model = _make_xgb(params, early_stopping=True)
            t0 = time.perf_counter()
            model.fit(X_fit, y_fit, sample_weight=sw,
                      eval_set=[(X_es, y_es)], verbose=False)
            fit_times.append(time.perf_counter() - t0)
            best_iters.append(getattr(model, "best_iteration",
                                      params["n_estimators"]))
            pred = model.predict(X_val)
            f1s.append(f1_score(y_val, pred, average="macro"))
        rows.append({
            "config_id": cfg_id, **params,
            "cv_macro_f1_mean": np.mean(f1s),
            "cv_macro_f1_std": np.std(f1s, ddof=1),
            "mean_fit_time_s": np.mean(fit_times),
            "mean_best_iteration": np.mean(best_iters),
        })
    results = pd.DataFrame(rows).sort_values(
        "cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return results


# ---------------------------------------------------------------------------
# 2. Repeated stratified cross-validation (champion vs baseline)
# ---------------------------------------------------------------------------

def make_baseline() -> SkPipeline:
    """Milestone 3 baseline unchanged: scaled LR with balanced weights."""
    return SkPipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=RANDOM_STATE)),
    ])


# Milestone 3 champion configuration, retained as a reference arm so
# the effect of tuning on stability can be measured on identical folds.
M3_DEFAULT_PARAMS = {"n_estimators": 500, "learning_rate": 0.1,
                     "max_depth": 6}


def repeated_cv_compare(X: pd.DataFrame, y: pd.Series, tuned_params: dict,
                        n_splits: int = 5, n_repeats: int = 3) -> pd.DataFrame:
    """
    Repeated stratified CV (n_repeats × n_splits) of the tuned champion
    against the Milestone 3 default configuration and the LR baseline on
    identical folds. Returns one row per model-repeat-fold with macro-F1,
    macro-AUPRC, and timing. The final refit protocol is mirrored here
    (no early stopping), so the CV estimate corresponds to the model
    actually deployed on the test set.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    records = []
    for i, (tr_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat, fold = divmod(i, n_splits)
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        y_bin = label_binarize(y_val, classes=CLASSES)
        for name, model in [
            ("XGBoost (tuned candidate)",
             _make_xgb(tuned_params, early_stopping=False)),
            ("XGBoost (M3 default)",
             _make_xgb(M3_DEFAULT_PARAMS, early_stopping=False)),
            ("LogisticRegression (baseline)", make_baseline()),
        ]:
            t0 = time.perf_counter()
            if name.startswith("XGBoost"):
                sw = compute_sample_weight("balanced", y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
            else:
                model.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            proba = model.predict_proba(X_val)
            pred_time = time.perf_counter() - t0
            pred = np.argmax(proba, axis=1)
            records.append({
                "model": name, "repeat": repeat + 1, "fold": fold + 1,
                "macro_f1": f1_score(y_val, pred, average="macro"),
                "auprc_macro": average_precision_score(
                    y_bin, proba, average="macro"),
                "fit_time_s": round(fit_time, 3),
                "pred_time_s": round(pred_time, 4),
            })
    return pd.DataFrame(records)


def select_champion(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Criterion-based champion selection between the two XGBoost arms:
    primary criterion = repeated-CV mean macro-F1; if the arms are within
    one pooled standard deviation of each other, the cheaper configuration
    (lower mean fit time) is preferred on parsimony/efficiency grounds.
    Returns a one-row frame with the selected arm and the decision basis.
    """
    g = (cv_df[cv_df["model"].str.startswith("XGBoost")]
         .groupby("model")
         .agg(macro_f1_mean=("macro_f1", "mean"),
              macro_f1_std=("macro_f1", "std"),
              fit_time_mean_s=("fit_time_s", "mean"))
         .sort_values("macro_f1_mean", ascending=False))
    top, second = g.iloc[0], g.iloc[1]
    within_noise = (top["macro_f1_mean"] - second["macro_f1_mean"]
                    ) < max(top["macro_f1_std"], second["macro_f1_std"])
    if within_noise and second["fit_time_mean_s"] < top["fit_time_mean_s"]:
        chosen, basis = second, ("arms within one SD; cheaper "
                                 "configuration preferred")
    else:
        chosen, basis = top, "higher repeated-CV mean macro-F1"
    out = g.loc[[chosen.name]].reset_index()
    out["selection_basis"] = basis
    return out


def wilcoxon_from_repeated_cv(cv_df: pd.DataFrame,
                              champion_name: str) -> dict:
    """
    One-sided Wilcoxon signed-rank test on the paired per-fold macro-F1
    differences (champion − baseline) across all repeats × folds.
    Direction stated in advance per RQ2 (champion exceeds baseline).
    """
    champ = cv_df[cv_df["model"] == champion_name] \
        .sort_values(["repeat", "fold"])["macro_f1"].to_numpy()
    base = cv_df[cv_df["model"].str.startswith("Logistic")] \
        .sort_values(["repeat", "fold"])["macro_f1"].to_numpy()
    diff = champ - base
    stat, p = wilcoxon(champ, base, alternative="greater")
    return {"test": "Wilcoxon signed-rank (one-sided)",
            "W": stat, "p_value": p, "n_pairs": len(diff),
            "mean_diff_macro_f1": round(float(diff.mean()), 4),
            "min_diff_macro_f1": round(float(diff.min()), 4)}


# ---------------------------------------------------------------------------
# 3. Deterministic final fit
# ---------------------------------------------------------------------------

def fit_final_models(X_train: pd.DataFrame, y_train: pd.Series,
                     tuned_params: dict):
    """
    Fit the tuned champion and the baseline on the full training split
    under deterministic settings (single thread, fixed seed, fixed tree
    count — no early stopping, so no carve-out and no thread-order
    variation). Returns (champion, baseline, timing dict).
    """
    champion = _make_xgb(tuned_params, early_stopping=False)
    sw = compute_sample_weight("balanced", y_train)
    t0 = time.perf_counter()
    champion.fit(X_train, y_train, sample_weight=sw, verbose=False)
    champ_time = time.perf_counter() - t0

    baseline = make_baseline()
    t0 = time.perf_counter()
    baseline.fit(X_train, y_train)
    base_time = time.perf_counter() - t0
    return champion, baseline, {
        "champion_fit_time_s": round(champ_time, 2),
        "baseline_fit_time_s": round(base_time, 2),
    }
