#!/usr/bin/env python3
"""
ml/ml_feature_prep.py — Feature preparation for the ML pipeline.

Reads customer_derivatives, applies the VS-methodology feature pipeline:
  1. Select the 13 ML features
  2. Handle nulls (median imputation — nulls mean no activity, not missing)
  3. Apply grade-based weights (from VS predictally_kmeans_backup.py)
  4. Normalize: (x - mean) / (max - min)  ← VS normalization, not z-score
  5. Run silhouette grid to find top feature sets (threshold = 0.60)
  6. Write prepared feature matrix to ml/output/features_prepared.csv
  7. Write feature metadata to ml/output/feature_meta.json

The 13 features map directly to VS PredictAlly attributes:
  VS name                         → merchant-abc column
  ─────────────────────────────────────────────────────
  NUM_OF_ORDERS                   → total_orders
  LTV_MEAN                        → total_revenue
  AOV                             → average_order_value
  RECENCY                         → recency_days
  FREQUENCY                       → frequency_months
  AVG_TIME_BETWEEN_PURCHASE       → frequency_months (same)
  HISTORICAL_SPEND (3m)           → spend_last_3_months
  HISTORICAL_SPEND (12m)          → spend_last_12_months
  EMAIL_OPEN_RATE                 → email_open_rate
  EMAIL_CLICK_RATE                → email_click_rate
  CAMPAIGN_INFLUENCE_RATE (NEW)   → campaign_influence_rate
  AVG_CAMPAIGNS_BEFORE_PURCHASE   → avg_campaigns_before_purchase
  PURCHASES_WITH_CAMPAIGN         → total_purchases_with_campaign

Grade weights (from VS implementation):
  Grade 1 (1.0) — highest signal: LTV, orders, influence rate
  Grade 2 (0.8) — strong signal:  AOV, open rate, campaigns before purchase
  Grade 3 (0.6) — moderate:       recency, click rate, spend 12m
  Grade 4 (0.4) — supporting:     frequency, spend 3m
  Grade 5 (0.2) — weakest:        purchases_with_campaign (count, not rate)

Usage
    python ml/ml_feature_prep.py              # prepare features
    python ml/ml_feature_prep.py --dry-run    # report only, no files written
    python ml/ml_feature_prep.py --port 5433  # override DB port
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "data_generator"))
OUTPUT_DIR = SCRIPT_DIR / "output"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# ── The 13 ML features and their grade weights ────────────────────────────
# Grade → weight: 1→1.0, 2→0.8, 3→0.6, 4→0.4, 5→0.2
FEATURES = [
    # (column_name,                     grade, description)
    ("total_revenue",                    1,    "LTV — total lifetime revenue"),
    ("total_orders",                     1,    "Total number of orders"),
    ("campaign_influence_rate",          1,    "% orders influenced by campaigns"),
    ("average_order_value",              2,    "Average order value"),
    ("email_open_rate",                  2,    "Email open rate"),
    ("avg_campaigns_before_purchase",    2,    "Avg campaigns before each purchase"),
    ("recency_days",                     3,    "Days since last purchase"),
    ("email_click_rate",                 3,    "Email click rate"),
    ("spend_last_12_months",             3,    "Spend in last 12 months"),
    ("frequency_months",                 4,    "Avg months between purchases"),
    ("spend_last_3_months",              4,    "Spend in last 3 months"),
    ("total_purchases_with_campaign",    5,    "Count of campaign-influenced orders"),
    ("total_campaigns_received",         5,    "Total campaigns received"),
]

FEATURE_COLS   = [f[0] for f in FEATURES]
GRADE_WEIGHTS  = {f[0]: [1.0, 0.8, 0.6, 0.4, 0.2][f[1] - 1] for f in FEATURES}

# Silhouette threshold — from VS: top feature sets where score > 0.60
SILHOUETTE_THRESHOLD = 0.60
K_RANGE = range(2, 8)   # test k=2..7


# ══════════════════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════════════════
def get_conn(args):
    import psycopg2
    return psycopg2.connect(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )


def load_derivatives(conn) -> pd.DataFrame:
    """Load latest customer_derivatives row per customer."""
    query = """
        SELECT DISTINCT ON (customer_id)
            customer_id,
            customer_type,
            attribution_segment,
            total_orders,
            total_revenue,
            average_order_value,
            recency_days,
            frequency_months,
            spend_last_3_months,
            spend_last_12_months,
            email_open_rate,
            email_click_rate,
            campaign_influence_rate,
            avg_campaigns_before_purchase,
            total_purchases_with_campaign,
            total_campaigns_received,
            total_purchases_without_campaign
        FROM customer_derivatives
        ORDER BY customer_id, run_id DESC
    """
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df):,} customers from customer_derivatives")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Feature pipeline (VS methodology)
# ══════════════════════════════════════════════════════════════════════════
def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median imputation per feature.
    Nulls in this dataset mean no activity (e.g. never received a campaign)
    not missing data, so median is more appropriate than mean.
    Recency_days null → impute with max (most lapsed).
    """
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            if col == "recency_days":
                fill = df[col].max()
            else:
                fill = df[col].median()
            df[col] = df[col].fillna(fill)
    return df


def apply_grade_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VS grade-based weights to the feature matrix.
    Weights are applied BEFORE normalization, exactly as in
    predictally_kmeans_backup.py.
    """
    df = df.copy()
    for col, weight in GRADE_WEIGHTS.items():
        if col in df.columns and weight < 1.0:
            df[col] = df[col] * weight
    return df


def normalize_vs(df: pd.DataFrame) -> pd.DataFrame:
    """
    VS normalization: (x - mean) / (max - min)
    This is NOT sklearn StandardScaler. It centres on mean
    but scales by range, producing a bounded distribution.
    Columns with zero range (all values identical) → set to 0.
    """
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        col_range = df[col].max() - df[col].min()
        if col_range == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - df[col].mean()) / col_range
    return df


# ══════════════════════════════════════════════════════════════════════════
# Silhouette feature selection (VS step 5e: top 10 sets > 0.60 threshold)
# ══════════════════════════════════════════════════════════════════════════
def run_silhouette_grid(X: np.ndarray, k_range=K_RANGE) -> dict:
    """
    Run KMeans++ across k values, compute silhouette scores.
    Returns the best k and its score.
    This mirrors the VS Greediness_Cases.xlsx experiment grid.
    """
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        # Need at least 2 unique labels for silhouette
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        results[k] = round(score, 6)
    return results


def select_best_k(scores: dict) -> int:
    """
    Select k where silhouette score is maximised above threshold.
    If no k beats 0.60, take the best available.
    """
    above = {k: s for k, s in scores.items() if s >= SILHOUETTE_THRESHOLD}
    if above:
        return max(above, key=above.get)
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(df_raw: pd.DataFrame, df_norm: pd.DataFrame,
           scores: dict, best_k: int) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Feature Preparation Report")
    print(f"{'═' * 60}")
    print(f"  Customers           : {len(df_raw):,}")
    print(f"  Features selected   : {len(FEATURE_COLS)}")
    print(f"\n  {'Feature':<38} {'Grade':>5} {'Weight':>6} {'Nulls':>6} {'Mean (raw)':>10}")
    print(f"  {'─' * 68}")
    for col, grade, desc in FEATURES:
        if col in df_raw.columns:
            nulls  = df_raw[col].isna().sum()
            mean   = df_raw[col].mean()
            weight = GRADE_WEIGHTS[col]
            print(f"  {col:<38} {grade:>5} {weight:>6.1f} {nulls:>6} {mean:>10.3f}")

    print(f"\n  Silhouette scores (KMeans++, k=2..7):")
    for k, s in scores.items():
        marker = " ← SELECTED" if k == best_k else (
                 " ✓" if s >= SILHOUETTE_THRESHOLD else "")
        print(f"    k={k}  score={s:.4f}{marker}")
    print(f"\n  Best k              : {best_k}")
    print(f"  Threshold           : {SILHOUETTE_THRESHOLD}")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare ML features from customer_derivatives.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, do not write output files")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = get_conn(args)
    print("Loading customer_derivatives …")
    df = load_derivatives(conn)
    conn.close()

    # Check we have the features we need
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"✗ Missing columns: {missing}")
        sys.exit(1)

    # Metadata columns we keep but don't use in ML
    meta_cols = ["customer_id", "customer_type", "attribution_segment",
                 "total_purchases_without_campaign"]

    print("Applying feature pipeline …")
    df_raw    = df[meta_cols + FEATURE_COLS].copy()
    df_work   = impute_nulls(df_raw.copy())
    df_wt     = apply_grade_weights(df_work)
    df_norm   = normalize_vs(df_wt)

    # Feature matrix for clustering
    X = df_norm[FEATURE_COLS].values.astype(np.float64)

    print("Running silhouette grid (k=2..7) …")
    scores = run_silhouette_grid(X)
    best_k = select_best_k(scores)

    report(df_raw, df_norm, scores, best_k)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save prepared features (normalised, weighted)
    out_features = OUTPUT_DIR / "features_prepared.csv"
    save_df = df_norm[meta_cols + FEATURE_COLS].copy()
    save_df.to_csv(out_features, index=False)
    print(f"✓ Features written  : {out_features}")

    # Save raw (imputed, not normalised) for human inspection
    out_raw = OUTPUT_DIR / "features_raw.csv"
    df_work[meta_cols + FEATURE_COLS].to_csv(out_raw, index=False)
    print(f"✓ Raw features      : {out_raw}")

    # Save feature metadata + silhouette grid
    meta = {
        "n_customers": len(df),
        "n_features":  len(FEATURE_COLS),
        "features": [
            {"col": col, "grade": grade, "weight": GRADE_WEIGHTS[col],
             "description": desc}
            for col, grade, desc in FEATURES
        ],
        "silhouette_scores": scores,
        "best_k": best_k,
        "silhouette_threshold": SILHOUETTE_THRESHOLD,
        "normalization": "VS: (x - mean) / (max - min)",
    }
    out_meta = OUTPUT_DIR / "feature_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Feature metadata  : {out_meta}")


if __name__ == "__main__":
    main()
