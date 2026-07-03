"""
features.py — CIND820 Capstone, Milestone 3
Pipeline Stage 4 (Feature Engineering), per Milestone 2 report §3.2

Implements a FeatureEngineer class with a strict fit/transform contract:
  - fit() learns all statistics (imputation medians, chi-squared selection,
    encoder categories, vendor risk tiers) from the TRAINING split only
  - transform() applies the learned transformations to any split

This design makes leakage prevention structural: the test set can never
influence imputation values, feature selection, or encoding vocabulary.

Transformations (in order):
  1. Categorical NaN -> explicit 'Unknown' category
  2. Mode-stratified median imputation for freight cost and weight,
     with *_imputed indicator flags
  3. log1p transform of freight cost, weight, line item value
  4. Chi-squared pre-selection of categorical features (alpha = 0.05, on train)
  5. One-hot encoding of retained categoricals (handle_unknown='ignore')
  6. Vendor risk tier (train-fold historical on-time rate) and
     Mode x VendorRiskTier interaction features

Author: Mubarak Ahmed (501345730)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder

ALPHA = 0.05
CATEGORICAL_CANDIDATES = [
    "shipment mode", "vendor", "country", "fulfill via",
    "dosage form", "sub classification",
]
IMPUTE_COLS = ["freight cost (usd)", "weight (kilograms)"]
LOG_COLS = ["freight cost (usd)", "weight (kilograms)", "line item value"]
NUMERIC_COLS = ["freight cost (usd)", "weight (kilograms)",
                "line item value", "line item quantity"]


class FeatureEngineer:
    """Stage 4 feature engineering with a train-only fit contract."""

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.chi2_results_ = None       # chi-squared audit table (train)
        self.selected_categoricals_ = None
        self.impute_medians_ = None     # {col: {mode: median}}
        self.global_medians_ = None     # fallback medians
        self.vendor_tiers_ = None       # {vendor: 0/1}
        self.tier_threshold_ = None
        self.encoder_ = None
        self.feature_names_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        df = self._fill_unknown(train_df)

        # (2) mode-stratified imputation medians, learned on train
        self.impute_medians_ = {
            col: df.groupby("shipment mode")[col].median().to_dict()
            for col in IMPUTE_COLS
        }
        self.global_medians_ = {col: df[col].median() for col in IMPUTE_COLS}

        # (4) chi-squared pre-selection on the training split
        rows = []
        for col in CATEGORICAL_CANDIDATES:
            ct = pd.crosstab(df[col], df["delay_class"])
            chi2, p, dof, _ = chi2_contingency(ct)
            rows.append({"feature": col, "chi2": round(chi2, 2),
                         "p_value": p, "dof": dof,
                         "selected": p < self.alpha})
        self.chi2_results_ = pd.DataFrame(rows)
        self.selected_categoricals_ = (
            self.chi2_results_.loc[self.chi2_results_["selected"], "feature"]
            .tolist())

        # (6) vendor risk tier from train-fold historical on-time rate.
        # High-risk (1) = on-time rate below the overall train on-time rate.
        # Threshold = overall training on-time rate: a vendor is high-risk
        # if its historical on-time rate is below the network average.
        # (The median vendor rate is 100% -- most vendors are small and never
        # late -- so a median threshold would degenerate to 'any late shipment'.)
        vendor_rate = (df.assign(on_time=(df["delay_class"] == 0).astype(int))
                       .groupby("vendor")["on_time"].mean())
        self.tier_threshold_ = float((df["delay_class"] == 0).mean())
        self.vendor_tiers_ = (vendor_rate < self.tier_threshold_).astype(int).to_dict()

        # (5) one-hot encoder fitted on train categories only
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(df[self.selected_categoricals_])

        # freeze final feature name order
        self.feature_names_ = self._build_feature_names()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply learned transformations. Returns (X, y)."""
        df = self._fill_unknown(df)
        y = df["delay_class"].copy()

        # (2) imputation + flags, using TRAIN medians
        num = pd.DataFrame(index=df.index)
        for col in IMPUTE_COLS:
            flag = df[col].isna().astype(int)
            imputed = df.apply(
                lambda r: self.impute_medians_[col].get(
                    r["shipment mode"], self.global_medians_[col])
                if pd.isna(r[col]) else r[col], axis=1)
            num[col] = imputed
            num[col.split(" (")[0] + "_imputed"] = flag

        # (3) log1p on skewed monetary/weight fields
        for col in LOG_COLS:
            num["log1p_" + col] = np.log1p(num[col] if col in num else df[col])
        for col in LOG_COLS:
            if col in num:
                num = num.drop(columns=[col])
        num["line item quantity"] = df["line item quantity"]

        # (5) one-hot encoding of chi-squared-selected categoricals
        ohe = pd.DataFrame(
            self.encoder_.transform(df[self.selected_categoricals_]),
            index=df.index,
            columns=self.encoder_.get_feature_names_out(self.selected_categoricals_))

        # (6) vendor risk tier + Mode x Tier interaction.
        # Vendors unseen in training have no history: default low-risk (0),
        # documented as a design assumption.
        tier = df["vendor"].map(self.vendor_tiers_).fillna(0).astype(int)
        inter = pd.DataFrame(index=df.index)
        inter["vendor_risk_tier"] = tier
        for mode in self.encoder_.categories_[
                self.selected_categoricals_.index("shipment mode")]:
            col = f"inter_{mode}_x_highrisk"
            inter[col] = ((df["shipment mode"] == mode) & (tier == 1)).astype(int)

        X = pd.concat([num, ohe, inter], axis=1)[self.feature_names_]
        return X, y

    # -------------------------------------------------------------- helpers
    def _fill_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_CANDIDATES:
            df[col] = df[col].fillna("Unknown").astype(str)
        return df

    def _build_feature_names(self) -> list:
        names = []
        for col in IMPUTE_COLS:
            names.append(col)
            names.append(col.split(" (")[0] + "_imputed")
        names = [n for n in names if n not in LOG_COLS]
        names += ["log1p_" + c for c in LOG_COLS]
        names += ["line item quantity"]
        names += list(self.encoder_.get_feature_names_out(self.selected_categoricals_))
        names += ["vendor_risk_tier"]
        names += [f"inter_{m}_x_highrisk" for m in self.encoder_.categories_[
            self.selected_categoricals_.index("shipment mode")]]
        return names
