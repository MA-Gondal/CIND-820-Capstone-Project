"""
preprocessing.py — CIND820 Capstone, Milestone 3
Pipeline Stage 2 (Target Engineering & Cleaning) and Stage 3 (Train-Test Split)

Implements the preprocessing design specified in the Milestone 2 report,
Section 3.2 (Pipeline Stage Descriptions):
  - Parse date fields and derive the 3-class delay target
  - Resolve cross-referenced weight/freight values (deterministic record linkage)
  - Remove leakage fields and non-informative identifiers
  - Stratified 80/20 train-test split with saved indices for reproducibility

Author: Mubarak Ahmed (501345730)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration constants (documented in Milestone 2 report, Sections 2 & 3)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Fields unavailable at order placement (post-delivery information).
# Using them as predictors would leak the outcome into the features.
LEAKAGE_FIELDS = [
    "delivered to client date",   # defines the target
    "delivery recorded date",     # recorded after delivery
    "line item insurance (usd)",  # not reliably known at order placement (M2 §2.2.2)
]

# Identifier / bookkeeping fields with no operational meaning as predictors.
IDENTIFIER_FIELDS = [
    "project code", "pq #", "po / so #", "asn/dn #",
]

# Pre-order date fields not used in the Milestone 2 feature design.
UNUSED_FIELDS = [
    "pq first sent to client date", "po sent to vendor date",
    "managed by",                  # failed chi-squared test, p = 0.456 (M2 §2.3.2)
    "item description", "molecule/test type", "brand", "dosage",
    "unit of measure (per pack)", "manufacturing site",
    "pack price", "unit price",    # collapsed into line item value (value = qty x price)
    "first line designation", "product group", "vendor inco term",
]

# Candidate predictors retained for Stage 4 (feature engineering).
CATEGORICAL_CANDIDATES = [
    "shipment mode", "vendor", "country", "fulfill via",
    "dosage form", "sub classification",
]
NUMERIC_CANDIDATES = [
    "freight cost (usd)", "weight (kilograms)",
    "line item value", "line item quantity",
]

CROSS_REF_PATTERN = re.compile(r"ID#:(\d+)")


# ---------------------------------------------------------------------------
# Stage 2 — Target engineering & cleaning
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw USAID SCMS CSV (latin1 encoding, per Milestone 2 audit)."""
    df = pd.read_csv(path, encoding="latin1")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse scheduled and actual delivery dates to datetime."""
    df = df.copy()
    for col in ["scheduled delivery date", "delivered to client date"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive delay_days (actual - scheduled, calendar days) and the
    3-class target delay_class:
        0 = on-time or early (<= 0 days deviation)
        1 = at-risk        (1-14 days late)
        2 = delayed        (> 14 days late)
    Early deliveries are classed as on-time: early arrival is not a risk
    event from an inventory-planning perspective (M2 §2.2.1).
    """
    df = df.copy()
    df["delay_days"] = (df["delivered to client date"]
                        - df["scheduled delivery date"]).dt.days
    conditions = [df["delay_days"] <= 0,
                  df["delay_days"].between(1, 14),
                  df["delay_days"] > 14]
    df["delay_class"] = np.select(conditions, [0, 1, 2], default=-1)
    df.loc[df["delay_days"].isna(), "delay_class"] = -1
    return df


def resolve_cross_references(df: pd.DataFrame,
                             cols=("weight (kilograms)", "freight cost (usd)")
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve entries of the form 'See DN-xxx (ID#:yyyy)' by looking up the
    referenced row's value (one hop). This is deterministic record linkage
    using only row identifiers — no target information is used, so it is
    leakage-safe and performed before the train-test split.

    Returns the updated DataFrame and a recovery log.
    """
    df = df.copy()
    id_lookup = df.set_index("id")
    log_rows = []
    for col in cols:
        as_str = df[col].astype(str)
        direct = pd.to_numeric(df[col], errors="coerce")
        ref_ids = as_str.str.extract(CROSS_REF_PATTERN)[0]
        resolved = pd.to_numeric(
            ref_ids.dropna().astype(int).map(id_lookup[col]), errors="coerce")
        recovered = direct.copy()
        recovered.loc[resolved.index] = resolved
        n_direct = int(direct.notna().sum())
        n_recovered = int(recovered.notna().sum())
        log_rows.append({
            "column": col,
            "directly_numeric": n_direct,
            "cross_references_found": int(ref_ids.notna().sum()),
            "recovered_via_lookup": n_recovered - n_direct,
            "usable_after_recovery": n_recovered,
            "usable_pct_after_recovery": round(100 * n_recovered / len(df), 1),
        })
        df[col] = recovered  # numeric column; unresolved entries become NaN
    return df, pd.DataFrame(log_rows)


def drop_leakage_and_identifiers(df: pd.DataFrame
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove leakage fields, identifiers, and fields excluded by the
    Milestone 2 design. 'id' is retained for traceability only and is
    never used as a model feature. Returns the reduced DataFrame and an
    exclusion log documenting the rationale for every dropped column.
    """
    exclusion_log = (
        [{"column": c, "reason": "leakage — unavailable at order placement"}
         for c in LEAKAGE_FIELDS]
        + [{"column": c, "reason": "identifier — no operational meaning"}
           for c in IDENTIFIER_FIELDS]
        + [{"column": c, "reason": "excluded by Milestone 2 feature design"}
           for c in UNUSED_FIELDS]
    )
    keep = (["id", "scheduled delivery date"]
            + CATEGORICAL_CANDIDATES + NUMERIC_CANDIDATES
            + ["delay_days", "delay_class"])
    df = df[[c for c in keep if c in df.columns]].copy()
    return df, pd.DataFrame(exclusion_log)


# ---------------------------------------------------------------------------
# Stage 3 — Stratified train-test split
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE):
    """
    Stratified 80/20 split preserving class proportions (M2 Stage 3).
    Stratification on delay_class is appropriate because rows are
    independent shipment line items, not a forecasting time series;
    the 18.5:1 imbalance makes plain random splitting risky for the
    minority classes. Returns train_df, test_df, and a summary table.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["delay_class"])
    summary = pd.DataFrame({
        "train_count": train_df["delay_class"].value_counts().sort_index(),
        "train_pct": (train_df["delay_class"].value_counts(normalize=True)
                      .sort_index() * 100).round(2),
        "test_count": test_df["delay_class"].value_counts().sort_index(),
        "test_pct": (test_df["delay_class"].value_counts(normalize=True)
                     .sort_index() * 100).round(2),
    })
    summary.index = ["0 (On-time)", "1 (At-risk)", "2 (Delayed)"]
    return train_df, test_df, summary


def save_split_artifacts(train_df, test_df, summary,
                         processed_dir="../data/processed",
                         tables_dir="../outputs/tables"):
    """Persist split indices (JSON), class distribution (CSV), and parquet files."""
    processed = Path(processed_dir)
    tables = Path(tables_dir)
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    with open(processed / "split_indices.json", "w") as f:
        json.dump({"random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                   "train_ids": train_df["id"].tolist(),
                   "test_ids": test_df["id"].tolist()}, f)
    summary.to_csv(tables / "class_distribution_train_test.csv")
    train_df.to_parquet(processed / "train.parquet", index=False)
    test_df.to_parquet(processed / "test.parquet", index=False)
    return processed / "split_indices.json"
