"""
impact.py — CIND820 Capstone, Milestone 4
Interpretation and operationalization layer.

Three components, each closing one research question:

  1. SHAP attribute-level importance (RQ1).
     TreeExplainer SHAP values are computed per one-hot column and then
     aggregated back to the original shipment attribute (all vendor_*
     columns sum to "Vendor", etc.), so importance is reported at the
     level a procurement planner actually controls.

  2. Cost-sensitive decision thresholding (RQ2 refinement / At-risk
     weakness). The default argmax decision rule treats all errors as
     equally costly. Operationally they are not: predicting On-time for
     a shipment that arrives late (a missed warning) forfeits the
     intervention window, while a false alarm merely triggers a low-cost
     review. Class-probability weight multipliers are selected on
     out-of-fold TRAINING predictions only (leakage-free), then applied
     once to the held-out test set.

  3. Dynamic safety-stock simulation (RQ3). The Milestone 1 formula
     SS = z × sigma(predicted lead-time deviation) × average daily demand
     is implemented with a class-mixture variance: each shipment's
     predicted class probabilities are combined with class-conditional
     lateness statistics learned from the TRAINING split. Because the
     dataset contains no demand quantities, buffers are expressed in
     days of cover (average daily demand normalized to 1); the
     comparison between policies is unaffected by this scaling.

Author: Mubarak Ahmed (501345730)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
CLASSES = [0, 1, 2]

# ---------------------------------------------------------------------------
# 1. SHAP attribute grouping (RQ1)
# ---------------------------------------------------------------------------

GROUP_PREFIXES = [
    ("shipment mode_", "Shipment Mode"),
    ("vendor_", "Vendor"),
    ("country_", "Country"),
    ("fulfill via_", "Fulfill Via"),
    ("dosage form_", "Dosage Form"),
    ("sub classification_", "Sub Classification"),
    ("mode_x_tier_", "Mode × Vendor Tier"),
]
SINGLE_FEATURES = {
    "log1p_freight cost (usd)": "Freight Cost (log)",
    "log1p_weight (kilograms)": "Weight (log)",
    "log1p_line item value": "Line Item Value (log)",
    "line item quantity": "Line Item Quantity",
    "freight cost_imputed": "Freight Cost Imputed Flag",
    "weight_imputed": "Weight Imputed Flag",
    "vendor_high_risk_tier": "Vendor Risk Tier",
}


def group_feature(col: str) -> str:
    """Map a one-hot / engineered column back to its shipment attribute."""
    if col in SINGLE_FEATURES:
        return SINGLE_FEATURES[col]
    for prefix, group in GROUP_PREFIXES:
        if col.startswith(prefix):
            return group
    return col


def grouped_shap_importance(shap_values: np.ndarray,
                            feature_names: list) -> pd.DataFrame:
    """
    Aggregate mean |SHAP| to attribute level.
    shap_values: array (n_samples, n_features, n_classes) or
                 (n_samples, n_features) for a single class.
    Returns attribute importance per class and overall, sorted overall.
    """
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, None]
    n_classes = shap_values.shape[2]
    mean_abs = np.abs(shap_values).mean(axis=0)         # (features, classes)
    df = pd.DataFrame(mean_abs,
                      columns=[f"class_{c}" for c in range(n_classes)])
    df["attribute"] = [group_feature(f) for f in feature_names]
    grouped = df.groupby("attribute").sum()
    grouped["overall"] = grouped.mean(axis=1)
    return grouped.sort_values("overall", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# 2. Cost-sensitive decision thresholding
# ---------------------------------------------------------------------------

# Illustrative asymmetric cost matrix, rows = true class, cols = predicted.
# Anchored to the operational asymmetry stated in Milestone 1: a missed
# late shipment (predict 0 when true 1/2) forfeits the intervention
# window, so it is costed at 5× a false alarm; missing a severe delay is
# costed higher than missing a marginal one. Sensitivity to this choice
# is reported alongside the main result.
DEFAULT_COST_MATRIX = np.array([
    #  pred 0  pred 1  pred 2
    [   0.0,    1.0,    1.0],   # true 0 (On-time): false alarms
    [   5.0,    0.0,    1.0],   # true 1 (At-risk): missed warning
    [  10.0,    2.0,    0.0],   # true 2 (Delayed): missed severe delay
])


def expected_cost(y_true, y_pred, cost_matrix=DEFAULT_COST_MATRIX) -> float:
    """Mean per-shipment cost of a prediction set under the cost matrix."""
    return float(np.mean(cost_matrix[np.asarray(y_true),
                                     np.asarray(y_pred)]))


def oof_probabilities(X: pd.DataFrame, y: pd.Series, make_champion,
                      n_splits: int = 5) -> np.ndarray:
    """
    Out-of-fold predicted probabilities on the training split, produced
    with the same fold seed as all other CV in this project. These OOF
    predictions are the only data used to select the decision rule, so
    the held-out test set plays no part in threshold selection.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    proba = np.zeros((len(y), len(CLASSES)))
    for tr_idx, val_idx in skf.split(X, y):
        model = make_champion()
        sw = compute_sample_weight("balanced", y.iloc[tr_idx])
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  sample_weight=sw, verbose=False)
        proba[val_idx] = model.predict_proba(X.iloc[val_idx])
    return proba


def weighted_argmax(proba: np.ndarray, weights) -> np.ndarray:
    """Decision rule: argmax over class probabilities × class weights."""
    return np.argmax(proba * np.asarray(weights)[None, :], axis=1)


def tune_decision_weights(proba_oof: np.ndarray, y: np.ndarray,
                          cost_matrix=DEFAULT_COST_MATRIX) -> pd.DataFrame:
    """
    Grid-search minority-class weight multipliers (w0 fixed at 1) that
    minimize expected cost on out-of-fold training predictions.
    Returns the full grid so the choice is auditable.
    """
    grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = []
    for w1 in grid:
        for w2 in grid:
            pred = weighted_argmax(proba_oof, [1.0, w1, w2])
            rows.append({
                "w_on_time": 1.0, "w_at_risk": w1, "w_delayed": w2,
                "oof_expected_cost": expected_cost(y, pred, cost_matrix),
                "oof_macro_f1": f1_score(y, pred, average="macro"),
                "oof_recall_at_risk": recall_score(
                    y, pred, labels=[1], average=None, zero_division=0)[0],
                "oof_recall_delayed": recall_score(
                    y, pred, labels=[2], average=None, zero_division=0)[0],
            })
    return pd.DataFrame(rows).sort_values(
        "oof_expected_cost").reset_index(drop=True)


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision / recall / F1 per class plus macro row."""
    rows = []
    labels = ["On-time (0)", "At-risk (1)", "Delayed (2)"]
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts().sort_index()
    for c in CLASSES:
        rows.append({"class": labels[c], "precision": round(p[c], 3),
                     "recall": round(r[c], 3), "f1": round(f[c], 3),
                     "support": int(support[c])})
    rows.append({"class": "Macro average",
                 "precision": round(p.mean(), 3),
                 "recall": round(r.mean(), 3),
                 "f1": round(f.mean(), 3),
                 "support": int(support.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Dynamic safety-stock simulation (RQ3)
# ---------------------------------------------------------------------------

Z_95 = 1.645  # standard normal z for a 95% cycle-service level


def class_lateness_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Class-conditional lateness statistics from the TRAINING split.
    Lateness = max(delay_days, 0): early arrival is not a stockout risk,
    consistent with the Milestone 3 target definition.
    """
    d = train_df.copy()
    d["lateness"] = d["delay_days"].clip(lower=0)
    stats = d.groupby("delay_class")["lateness"].agg(["mean", "var"])
    stats["var"] = stats["var"].fillna(0.0)
    return stats


def mixture_lateness_sigma(proba: np.ndarray,
                           stats: pd.DataFrame) -> np.ndarray:
    """
    Per-shipment predicted lateness standard deviation from the mixture
    over classes: Var = sum_k p_k (var_k + mean_k^2) − (sum_k p_k mean_k)^2.
    """
    mu = stats["mean"].to_numpy()
    var = stats["var"].to_numpy()
    e_l = proba @ mu
    e_l2 = proba @ (var + mu ** 2)
    return np.sqrt(np.maximum(e_l2 - e_l ** 2, 0.0))


def safety_stock_simulation(proba_test: np.ndarray, test_df: pd.DataFrame,
                            train_df: pd.DataFrame, z: float = Z_95,
                            daily_demand: float = 1.0) -> dict:
    """
    Compare the fixed-buffer policy against the model-driven dynamic
    policy on the held-out test set.

      Fixed:   SS_fixed = z × sigma(lateness_train) × d   (uniform)
      Dynamic: SS_i     = z × sigma_i(predicted lateness) × d

    Stockout exposure: a shipment whose ACTUAL lateness exceeds its
    buffer (in days of cover). Holding proxy: total buffer days held.
    Returns summary dict plus the per-shipment frame for plotting.
    """
    stats = class_lateness_stats(train_df)
    actual_lateness = test_df["delay_days"].clip(lower=0).to_numpy()

    sigma_fixed = float(train_df["delay_days"].clip(lower=0).std())
    buf_fixed = np.full(len(test_df), z * sigma_fixed * daily_demand)

    sigma_dyn = mixture_lateness_sigma(proba_test, stats)
    buf_dyn = z * sigma_dyn * daily_demand

    frame = pd.DataFrame({
        "actual_lateness_days": actual_lateness,
        "true_class": test_df["delay_class"].to_numpy(),
        "pred_class": np.argmax(proba_test, axis=1),
        "buffer_fixed_days": buf_fixed,
        "buffer_dynamic_days": buf_dyn,
        "stockout_fixed": actual_lateness > buf_fixed,
        "stockout_dynamic": actual_lateness > buf_dyn,
    })
    summary = {
        "z": z,
        "sigma_fixed_days": round(sigma_fixed, 2),
        "fixed_buffer_days_per_shipment": round(float(buf_fixed[0]), 2),
        "total_buffer_days_fixed": round(float(buf_fixed.sum()), 0),
        "total_buffer_days_dynamic": round(float(buf_dyn.sum()), 0),
        "buffer_days_reduction_pct": round(
            100 * (1 - buf_dyn.sum() / buf_fixed.sum()), 1),
        "stockouts_fixed": int(frame["stockout_fixed"].sum()),
        "stockouts_dynamic": int(frame["stockout_dynamic"].sum()),
        "stockout_rate_fixed_pct": round(
            100 * frame["stockout_fixed"].mean(), 2),
        "stockout_rate_dynamic_pct": round(
            100 * frame["stockout_dynamic"].mean(), 2),
    }
    return summary, frame
