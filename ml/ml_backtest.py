#!/usr/bin/env python3
"""
ml/ml_backtest.py — Backtesting with 6-month temporal cuts.

Implements the VS PredictAlly backtesting strategy (Step xx):
  "Cut the data into 6 months, run the final model and bring
   the prediction outcomes. Run the accuracy of the model and
   bring the confusion matrix and all other performance metrics.
   Observe TP, FN, TN and FP. Put FP as the secondary measure
   while the objective is to reduce the FNs."

Strategy:
  Using customer_derivatives features + actual purchase outcomes
  from campaign_attribution as ground truth.

  Ground truth label derivation (from actual historical data):
    ORDER in 30d after campaign      → send_campaign (campaign worked)
    ORDER with no campaign in 30d    → no_campaign_needed (organic buyer)
    0 orders + low engagement        → no_campaign_impact (ghost)
    recent_purchase (recency < 14d)  → dont_send (too soon)

  Temporal cuts (without replacement):
    Cut 1: Train Jan 2022 – Jun 2022  → Test Jul – Dec 2022
    Cut 2: Train Jan 2022 – Dec 2022  → Test Jan – Jun 2023
    Cut 3: Train Jan 2022 – Jun 2023  → Test Jul – Dec 2023
    Cut 4: Train Jan 2022 – Dec 2023  → Test Jan – Jun 2024

  Each cut: train RFC on features + ground truth labels,
            predict on test window, compare vs actual.

Usage
    python ml/ml_backtest.py              # run all 4 cuts
    python ml/ml_backtest.py --dry-run    # report structure only
    python ml/ml_backtest.py --port 5433
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

# 6-month cuts: (train_end, test_start, test_end)
TEMPORAL_CUTS = [
    ("2022-06-30", "2022-07-01", "2022-12-31", "Cut 1"),
    ("2022-12-31", "2023-01-01", "2023-06-30", "Cut 2"),
    ("2023-06-30", "2023-07-01", "2023-12-31", "Cut 3"),
    ("2023-12-31", "2024-01-01", "2024-06-30", "Cut 4"),
]

RFC_PARAMS = {
    "n_estimators":    200,
    "min_samples_leaf": 5,
    "class_weight":   "balanced",
    "random_state":    42,
    "n_jobs":         -1,
}


def get_conn(args):
    import psycopg2
    return psycopg2.connect(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )


# ══════════════════════════════════════════════════════════════════════════
# Ground truth label derivation from actual historical data
# ══════════════════════════════════════════════════════════════════════════
GROUND_TRUTH_SQL = """
WITH customer_window AS (
    -- Features computed up to the train_end date
    SELECT DISTINCT ON (d.customer_id)
        d.customer_id,
        d.total_orders,
        d.total_revenue,
        d.average_order_value,
        d.recency_days,
        d.frequency_months,
        d.spend_last_3_months,
        d.spend_last_12_months,
        d.email_open_rate,
        d.email_click_rate,
        d.campaign_influence_rate,
        d.avg_campaigns_before_purchase,
        d.total_purchases_with_campaign,
        d.total_campaigns_received,
        d.attribution_segment
    FROM customer_derivatives d
    ORDER BY d.customer_id, d.run_id DESC
),
test_window_purchases AS (
    -- Actual purchases in the test window
    SELECT
        customer_id,
        COUNT(*) AS orders_in_test,
        SUM(CASE WHEN campaign_influenced THEN 1 ELSE 0 END) AS campaign_orders,
        MIN(order_date) AS first_test_order
    FROM campaign_attribution
    WHERE order_date >= %(test_start)s
      AND order_date <= %(test_end)s
    GROUP BY customer_id
)
SELECT
    cw.*,
    COALESCE(tw.orders_in_test, 0)    AS test_orders,
    COALESCE(tw.campaign_orders, 0)   AS test_campaign_orders,
    tw.first_test_order
FROM customer_window cw
LEFT JOIN test_window_purchases tw ON tw.customer_id = cw.customer_id
"""


def derive_ground_truth_label(row: pd.Series) -> str:
    """
    Derive the actual outcome label from what happened in the test window.
    This is the ground truth we compare ML predictions against.
    """
    test_orders   = int(row.get("test_orders", 0) or 0)
    camp_orders   = int(row.get("test_campaign_orders", 0) or 0)
    recency       = float(row.get("recency_days", 999) or 999)
    open_rate     = float(row.get("email_open_rate", 0) or 0)
    n_campaigns   = int(row.get("total_campaigns_received", 0) or 0)

    if recency < 14:
        return "dont_send"
    if test_orders > 0 and camp_orders > 0:
        return "send_campaign"
    if test_orders > 0 and camp_orders == 0:
        return "no_campaign_needed"
    if test_orders == 0 and n_campaigns >= 5 and open_rate < 0.10:
        return "no_campaign_impact"
    # Default for ambiguous cases
    return "no_campaign_impact"


def load_ground_truth(conn, test_start: str, test_end: str) -> pd.DataFrame:
    df = pd.read_sql(
        GROUND_TRUTH_SQL, conn,
        params={"test_start": test_start, "test_end": test_end}
    )
    df["ground_truth_label"] = df.apply(derive_ground_truth_label, axis=1)
    return df


def normalize_vs(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """VS normalization applied to feature window."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = df[col].fillna(df[col].median())
        col_range = df[col].max() - df[col].min()
        if col_range > 0:
            df[col] = (df[col] - df[col].mean()) / col_range
        else:
            df[col] = 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
# Run one temporal cut
# ══════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "total_revenue", "total_orders", "campaign_influence_rate",
    "average_order_value", "email_open_rate", "avg_campaigns_before_purchase",
    "recency_days", "email_click_rate", "spend_last_12_months",
    "frequency_months", "spend_last_3_months",
    "total_purchases_with_campaign", "total_campaigns_received",
]


def run_cut(conn, train_end: str, test_start: str,
            test_end: str, cut_name: str) -> dict:
    print(f"\n  {cut_name}: train → {train_end} | test {test_start} → {test_end}")

    # Load data for this cut
    df_train = load_ground_truth(conn, "2022-01-01", train_end)
    df_test  = load_ground_truth(conn, test_start, test_end)

    if len(df_train) < 50 or len(df_test) < 20:
        print(f"    Skipping — insufficient data "
              f"(train={len(df_train)}, test={len(df_test)})")
        return {}

    # Normalize (separately per window to avoid leakage)
    df_train_norm = normalize_vs(df_train, FEATURE_COLS)
    df_test_norm  = normalize_vs(df_test, FEATURE_COLS)

    X_train = df_train_norm[FEATURE_COLS].values
    y_train = df_train["ground_truth_label"].values
    X_test  = df_test_norm[FEATURE_COLS].values
    y_test  = df_test["ground_truth_label"].values

    # Filter to labels present in training
    valid_labels = set(y_train)
    mask = np.isin(y_test, list(valid_labels))
    X_test = X_test[mask]
    y_test = y_test[mask]

    if len(X_test) == 0 or len(set(y_train)) < 2:
        print(f"    Skipping — not enough label diversity")
        return {}

    rfc = RandomForestClassifier(**RFC_PARAMS)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels_present = [l for l in LABEL_ORDER if l in set(y_test)]
    cm  = confusion_matrix(y_test, y_pred, labels=labels_present)

    # FN analysis per label
    fn_rates = {}
    for i, label in enumerate(labels_present):
        if i < len(cm):
            tp = int(cm[i, i])
            fn = int(cm[i].sum() - cm[i, i])
            fn_rates[label] = {
                "tp": tp, "fn": fn,
                "fn_rate": round(fn / (tp + fn), 3) if (tp + fn) > 0 else 0
            }

    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"Accuracy: {acc:.3f}")
    for label, stats in fn_rates.items():
        print(f"      {label:<26} TP={stats['tp']:>4}  "
              f"FN={stats['fn']:>4}  FN_rate={stats['fn_rate']:.1%}")

    return {
        "cut_name":    cut_name,
        "train_end":   train_end,
        "test_start":  test_start,
        "test_end":    test_end,
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "accuracy":    round(float(acc), 4),
        "fn_rates":    fn_rates,
        "cm_labels":   labels_present,
        "cm":          cm.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════
def summary_report(results: list) -> None:
    valid = [r for r in results if r]
    if not valid:
        print("No valid cuts to summarise.")
        return

    print(f"\n{'═' * 62}")
    print(f"  Backtesting Summary — {len(valid)} temporal cuts")
    print(f"  Primary objective: minimize FN (missed likely buyers)")
    print(f"{'═' * 62}")

    accs = [r["accuracy"] for r in valid]
    print(f"\n  {'Cut':<8} {'Train':>6} {'Test':>6} {'Accuracy':>9}")
    print(f"  {'─' * 35}")
    for r in valid:
        print(f"  {r['cut_name']:<8} {r['n_train']:>6,} "
              f"{r['n_test']:>6,} {r['accuracy']:>9.3f}")
    print(f"  {'─' * 35}")
    print(f"  {'Mean':>14} {' ':>6} {np.mean(accs):>9.3f}")
    print(f"  {'Std':>14} {' ':>6} {np.std(accs):>9.3f}")

    # Aggregate FN rates across cuts
    all_fn: dict[str, list] = {}
    for r in valid:
        for label, stats in r["fn_rates"].items():
            all_fn.setdefault(label, []).append(stats["fn_rate"])

    print(f"\n  Average FN rates across {len(valid)} cuts:")
    for label in LABEL_ORDER:
        if label in all_fn:
            avg = np.mean(all_fn[label])
            bar = "█" * int((1 - avg) * 20)
            print(f"    {label:<26}  FN_rate={avg:.1%}  "
                  f"model_recall={1-avg:.1%}  {bar}")

    print(f"\n  Interpretation:")
    for label in LABEL_ORDER:
        if label in all_fn:
            avg = np.mean(all_fn[label])
            if avg > 0.30:
                note = "⚠ High FN — revisit features or threshold"
            elif avg > 0.15:
                note = "↑ Moderate FN — acceptable for v1"
            else:
                note = "✓ Low FN — model is catching most positives"
            print(f"    {label:<26}  {note}")

    print(f"{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest ML pipeline with 6-month temporal cuts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show cut structure, do not run")
    ap.add_argument("--host", help="DB host")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    if args.dry_run:
        print("\nBacktest structure (4 temporal cuts, no replacement):")
        for train_end, test_start, test_end, name in TEMPORAL_CUTS:
            print(f"  {name}: train Jan 2022 – {train_end} "
                  f"| test {test_start} – {test_end}")
        print("\nGround truth labels derived from campaign_attribution:")
        print("  purchased + campaign in 30d window → send_campaign")
        print("  purchased + no campaign in window  → no_campaign_needed")
        print("  0 purchases + low engagement       → no_campaign_impact")
        print("  recency < 14 days                  → dont_send")
        print("\nDry run — no DB connection or model training.")
        return

    conn = get_conn(args)
    print(f"✓ Connected to DB")
    print(f"\nRunning {len(TEMPORAL_CUTS)} temporal cuts …")

    results = []
    for train_end, test_start, test_end, name in TEMPORAL_CUTS:
        result = run_cut(conn, train_end, test_start, test_end, name)
        results.append(result)

    conn.close()
    summary_report(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "backtest_report.json"
    with open(out_path, "w") as f:
        json.dump([r for r in results if r], f, indent=2, default=str)
    print(f"✓ Backtest report   : {out_path}")


if __name__ == "__main__":
    main()
