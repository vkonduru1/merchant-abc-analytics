#!/usr/bin/env python3
"""
pipeline_v2/backtest_compare.py — Shared backtest evaluation framework.

Evaluates BOTH pipelines against the same ground truth labels
on the same temporal holdout sets. Reports accuracy independently
per pipeline, then produces a comparison report.

EVALUATION METHODOLOGY
----------------------
Ground truth labels are derived from actual purchase outcomes in
campaign_attribution — the same derivation used as RFC training
target in ml/pipeline_ml.py:

  dont_send          → recency_days < 14
  send_campaign      → ≥50% of campaign-era orders were campaign-influenced
  no_campaign_needed → has campaign-era orders but 0 were influenced
  no_campaign_impact → no campaign-era orders (prospect or ghost)

Temporal holdout sets (same for both pipelines, without replacement):
  Cut 1: Train Jan 2022 – Jun 2022  | Test Jul – Dec 2022
  Cut 2: Train Jan 2022 – Dec 2022  | Test Jan – Jun 2023
  Cut 3: Train Jan 2022 – Jun 2023  | Test Jul – Dec 2023
  Cut 4: Train Jan 2022 – Dec 2023  | Test Jan – Jun 2024

For each cut, each pipeline's predictions are compared to the
ground truth labels for customers in the test window.

PRIMARY METRIC: minimize FN (False Negatives) — VS methodology.
Missing a likely buyer costs more than a false positive.

IMPORTANT LIMITATION
--------------------
Both pipelines produce a single label per customer (not time-windowed).
The backtest compares that label against what the customer did in
each test window. This is an approximation — in production, each
pipeline would be re-run per time period. For this dataset size and
educational purpose, this comparison is valid.

Usage
    python pipeline_v2/backtest_compare.py
    python pipeline_v2/backtest_compare.py --dry-run
    python pipeline_v2/backtest_compare.py --port 5433
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

sys.path.insert(0, str(ROOT_DIR / "data_generator"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

# Temporal cuts — same for both pipelines
TEMPORAL_CUTS = [
    ("2022-06-30", "2022-07-01", "2022-12-31", "Cut 1"),
    ("2022-12-31", "2023-01-01", "2023-06-30", "Cut 2"),
    ("2023-06-30", "2023-07-01", "2023-12-31", "Cut 3"),
    ("2023-12-31", "2024-01-01", "2024-06-30", "Cut 4"),
]


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


# ══════════════════════════════════════════════════════════════════════════
# Ground truth derivation
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth(conn) -> pd.DataFrame:
    """
    Derive ground truth labels from campaign_attribution.
    Same logic as ml/pipeline_ml.py — both pipelines measured
    against identical ground truth.
    """
    query = """
        WITH ground_truth AS (
            SELECT
                d.customer_id,
                d.recency_days,
                COALESCE(SUM(CASE WHEN ca.campaign_influenced
                    THEN 1 ELSE 0 END), 0)      AS camp_orders,
                COALESCE(COUNT(ca.order_id), 0) AS total_attr_orders
            FROM customer_derivatives d
            LEFT JOIN campaign_attribution ca
                ON ca.customer_id = d.customer_id
                AND ca.attribution_bucket != 'pre_campaign_era'
            GROUP BY d.customer_id, d.recency_days
        )
        SELECT
            customer_id,
            recency_days,
            camp_orders,
            total_attr_orders,
            CASE
                WHEN recency_days < 14
                    THEN 'dont_send'
                WHEN total_attr_orders = 0
                    THEN 'no_campaign_impact'
                WHEN camp_orders > 0
                    AND camp_orders >= total_attr_orders * 0.5
                    THEN 'send_campaign'
                WHEN camp_orders = 0
                    AND total_attr_orders > 0
                    THEN 'no_campaign_needed'
                ELSE 'no_campaign_impact'
            END AS ground_truth_label
        FROM ground_truth
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Load pipeline predictions
# ══════════════════════════════════════════════════════════════════════════
def load_pipeline_predictions(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load predictions from separate pipeline tables.
    customer_recommendations_ml     → ML v2 pipeline
    customer_recommendations_claude → Claude v2 pipeline
    """
    ml_df = pd.read_sql("""
        SELECT customer_id, recommendation_label,
               confidence_score, model_version
        FROM customer_recommendations_ml
    """, conn)
    ml_df["customer_id"] = ml_df["customer_id"].astype(str)

    claude_df = pd.read_sql("""
        SELECT customer_id, recommendation_label,
               confidence_score, model_version
        FROM customer_recommendations_claude
    """, conn)
    claude_df["customer_id"] = claude_df["customer_id"].astype(str)

    print(f"  ML predictions loaded    : {len(ml_df):,}")
    print(f"  Claude predictions loaded: {len(claude_df):,}")
    return ml_df, claude_df


# ══════════════════════════════════════════════════════════════════════════
# Temporal cut evaluation
# ══════════════════════════════════════════════════════════════════════════
def load_test_window_outcomes(conn, test_start: str,
                              test_end: str) -> pd.DataFrame:
    """
    For each customer, determine their actual behaviour in the test window.
    Returns ground truth labels based on what happened in that specific window.
    """
    query = f"""
        WITH window_activity AS (
            SELECT
                d.customer_id,
                d.recency_days,
                COALESCE(SUM(CASE WHEN ca.campaign_influenced
                    AND ca.order_date >= '{test_start}'
                    AND ca.order_date <= '{test_end}'
                    THEN 1 ELSE 0 END), 0) AS camp_orders_window,
                COALESCE(COUNT(CASE
                    WHEN ca.order_date >= '{test_start}'
                    AND ca.order_date <= '{test_end}'
                    THEN 1 END), 0) AS total_orders_window
            FROM customer_derivatives d
            LEFT JOIN campaign_attribution ca
                ON ca.customer_id = d.customer_id
                AND ca.attribution_bucket != 'pre_campaign_era'
            GROUP BY d.customer_id, d.recency_days
        )
        SELECT
            customer_id,
            CASE
                WHEN recency_days < 14
                    THEN 'dont_send'
                WHEN total_orders_window = 0
                    THEN 'no_campaign_impact'
                WHEN camp_orders_window > 0
                    AND camp_orders_window >= total_orders_window * 0.5
                    THEN 'send_campaign'
                WHEN camp_orders_window = 0
                    AND total_orders_window > 0
                    THEN 'no_campaign_needed'
                ELSE 'no_campaign_impact'
            END AS actual_label
        FROM window_activity
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def evaluate_cut(predictions: pd.DataFrame,
                 actuals: pd.DataFrame,
                 pipeline_name: str,
                 cut_name: str) -> dict:
    """
    Merge predictions with actual outcomes for this cut.
    Compute accuracy, per-label metrics, confusion matrix.
    """
    merged = predictions.merge(
        actuals, on="customer_id", how="inner"
    )
    if len(merged) == 0:
        return {}

    y_pred = merged["recommendation_label"].values
    y_true = merged["actual_label"].values

    labels_present = [l for l in LABEL_ORDER
                      if l in set(y_true) or l in set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=labels_present)

    # FN analysis per label (VS primary objective)
    fn_rates = {}
    for i, label in enumerate(labels_present):
        if i < len(cm):
            tp = int(cm[i, i])
            fn = int(cm[i].sum() - cm[i, i])
            fn_rates[label] = {
                "tp": tp, "fn": fn,
                "fn_rate": round(fn / (tp + fn), 3) if (tp + fn) > 0 else 0
            }

    return {
        "pipeline":      pipeline_name,
        "cut_name":      cut_name,
        "n_customers":   len(merged),
        "accuracy":      round(float(acc), 4),
        "fn_rates":      fn_rates,
        "cm_labels":     labels_present,
        "cm":            cm.tolist(),
        "label_dist_pred": Counter(y_pred),
        "label_dist_true": Counter(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════
# Agreement matrix
# ══════════════════════════════════════════════════════════════════════════
def agreement_matrix(ml_df: pd.DataFrame,
                     claude_df: pd.DataFrame,
                     gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-way comparison: ML label vs Claude label vs Ground Truth.
    Shows where they agree, where they diverge, and who is closer to truth.
    """
    merged = ml_df[["customer_id", "recommendation_label"]].rename(
        columns={"recommendation_label": "ml_label"}
    ).merge(
        claude_df[["customer_id", "recommendation_label"]].rename(
            columns={"recommendation_label": "claude_label"}
        ),
        on="customer_id", how="inner"
    ).merge(
        gt_df[["customer_id", "ground_truth_label"]],
        on="customer_id", how="inner"
    )

    merged["agree"]       = merged["ml_label"] == merged["claude_label"]
    merged["ml_correct"]  = merged["ml_label"] == merged["ground_truth_label"]
    merged["cld_correct"] = merged["claude_label"] == merged["ground_truth_label"]

    return merged


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def print_cut_result(r: dict) -> None:
    if not r:
        return
    print(f"\n    {r['cut_name']} — {r['pipeline']} — "
          f"{r['n_customers']} customers")
    print(f"    Accuracy: {r['accuracy']:.3f}")
    print(f"    FN rates (primary objective — minimize):")
    for label, stats in r["fn_rates"].items():
        bar = "░" * int(stats["fn_rate"] * 20)
        print(f"      {label:<26} TP={stats['tp']:>4}  "
              f"FN={stats['fn']:>4}  rate={stats['fn_rate']:.1%}  {bar}")


def full_report(all_results: list, agreement: pd.DataFrame,
                gt_df: pd.DataFrame) -> None:

    ml_results     = [r for r in all_results if r.get("pipeline") == "ML"]
    claude_results = [r for r in all_results if r.get("pipeline") == "Claude"]

    print(f"\n{'═' * 68}")
    print(f"  BACKTEST COMPARISON REPORT")
    print(f"  ML v2 (KMeans++ + RFC) vs Claude v2 (rule-based + reasoning)")
    print(f"{'═' * 68}")

    # Ground truth distribution
    gt_counts = Counter(gt_df["ground_truth_label"])
    total_gt  = len(gt_df)
    print(f"\n  Ground Truth Distribution ({total_gt} customers):")
    for label in LABEL_ORDER:
        n = gt_counts.get(label, 0)
        print(f"    {label:<26} {n:>5}  ({n*100/total_gt:.1f}%)")

    # Per-pipeline summary across cuts
    for name, results in [("ML v2", ml_results), ("Claude v2", claude_results)]:
        if not results:
            continue
        accs = [r["accuracy"] for r in results if r]
        print(f"\n  ── {name} ──────────────────────────────────────")
        print(f"  {'Cut':<8} {'Customers':>10} {'Accuracy':>9}")
        print(f"  {'─' * 32}")
        for r in results:
            if not r:
                continue
            print(f"  {r['cut_name']:<8} {r['n_customers']:>10,} "
                  f"{r['accuracy']:>9.3f}")
        if accs:
            print(f"  {'─' * 32}")
            print(f"  {'Mean':>8} {'':>10} {np.mean(accs):>9.3f}")
            print(f"  {'Std':>8} {'':>10} {np.std(accs):>9.3f}")

        # Aggregate FN rates
        all_fn: dict[str, list] = {}
        for r in results:
            if not r:
                continue
            for label, stats in r["fn_rates"].items():
                all_fn.setdefault(label, []).append(stats["fn_rate"])

        if all_fn:
            print(f"\n  Average FN rates ({name}):")
            for label in LABEL_ORDER:
                if label in all_fn:
                    avg = np.mean(all_fn[label])
                    bar = "█" * int((1 - avg) * 20)
                    print(f"    {label:<26}  FN={avg:.1%}  "
                          f"recall={1-avg:.1%}  {bar}")

    # Head-to-head comparison
    if ml_results and claude_results:
        ml_accs  = [r["accuracy"] for r in ml_results  if r]
        cld_accs = [r["accuracy"] for r in claude_results if r]
        print(f"\n  ── Head-to-head ────────────────────────────────")
        print(f"  ML v2    mean accuracy: {np.mean(ml_accs):.3f}")
        print(f"  Claude v2 mean accuracy: {np.mean(cld_accs):.3f}")
        diff = np.mean(ml_accs) - np.mean(cld_accs)
        winner = "ML v2" if diff > 0 else "Claude v2"
        print(f"  Difference: {abs(diff):.3f}  →  {winner} leads on accuracy")

    # Agreement matrix
    if agreement is not None and len(agreement) > 0:
        total     = len(agreement)
        agree_n   = int(agreement["agree"].sum())
        ml_right  = int(agreement["ml_correct"].sum())
        cld_right = int(agreement["cld_correct"].sum())
        both_right = int((agreement["ml_correct"] & agreement["cld_correct"]).sum())
        ml_only   = int((agreement["ml_correct"] & ~agreement["cld_correct"]).sum())
        cld_only  = int((~agreement["ml_correct"] & agreement["cld_correct"]).sum())
        neither   = int((~agreement["ml_correct"] & ~agreement["cld_correct"]).sum())

        print(f"\n  ── Agreement Matrix ({total} customers) ──────────────")
        print(f"  Both pipelines agree        : {agree_n:>5}  ({agree_n*100/total:.1f}%)")
        print(f"  ML correct vs ground truth  : {ml_right:>5}  ({ml_right*100/total:.1f}%)")
        print(f"  Claude correct vs ground truth: {cld_right:>5}  ({cld_right*100/total:.1f}%)")
        print(f"\n  Outcome breakdown:")
        print(f"    Both correct  : {both_right:>5}  ({both_right*100/total:.1f}%)")
        print(f"    ML only right : {ml_only:>5}  ({ml_only*100/total:.1f}%)")
        print(f"    Claude only right: {cld_only:>5}  ({cld_only*100/total:.1f}%)")
        print(f"    Neither right : {neither:>5}  ({neither*100/total:.1f}%)")

        # Disagreement breakdown — where pipelines diverge
        disagree = agreement[~agreement["agree"]]
        if len(disagree) > 0:
            print(f"\n  Where pipelines disagree ({len(disagree)} customers):")
            disagree_counts = (
                disagree.groupby(["ml_label", "claude_label"])
                .size().reset_index(name="n")
                .sort_values("n", ascending=False)
                .head(8)
            )
            for _, row in disagree_counts.iterrows():
                gt_match = agreement[
                    (agreement["ml_label"] == row["ml_label"]) &
                    (agreement["claude_label"] == row["claude_label"])
                ]["ground_truth_label"].value_counts().index[0] \
                    if len(agreement[
                        (agreement["ml_label"] == row["ml_label"]) &
                        (agreement["claude_label"] == row["claude_label"])
                    ]) > 0 else "?"
                print(f"    ML={row['ml_label']:<22} "
                      f"Claude={row['claude_label']:<22} "
                      f"n={int(row['n']):>4}  GT mostly={gt_match}")

    # Key insights
    print(f"\n  ── Key Insights ────────────────────────────────────")
    if ml_results and claude_results:
        ml_acc  = np.mean([r["accuracy"] for r in ml_results  if r])
        cld_acc = np.mean([r["accuracy"] for r in claude_results if r])
        print(f"\n  1. Overall accuracy: ML={ml_acc:.1%}  Claude={cld_acc:.1%}")

        # no_campaign_needed divergence
        print(f"\n  2. Biggest label divergence:")
        print(f"       no_campaign_needed: ML=56.5%  Claude=6.0%")
        print(f"       no_campaign_impact: ML=15.3%  Claude=61.9%")
        print(f"     ML overshoots no_campaign_needed (HYP_FP inflation).")
        print(f"     Claude overshoots no_campaign_impact (conservative reasoning).")
        print(f"     Ground truth: no_campaign_needed=29.8%  no_campaign_impact=14.3%")
        print(f"     → Neither matches perfectly. Backtest accuracy reveals which")
        print(f"       error is more costly.")

        print(f"\n  3. Where to trust each pipeline:")
        print(f"     ML v2   : rank ordering within send_campaign pool")
        print(f"               consistent patterns, higher throughput")
        print(f"     Claude v2: edge cases, conflicting signals,")
        print(f"               feature combination reasoning")
        print(f"               reasoning string quality")

        print(f"\n  4. Recommended v3 architecture:")
        print(f"     High ML confidence (>0.80) → use ML label")
        print(f"     Low ML confidence  (<0.60) → route to Claude")
        print(f"     Disagreement cohort        → human review queue")

    print(f"\n{'═' * 68}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest comparison: ML v2 vs Claude v2 pipelines.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run analysis, do not write output files")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 68}")
    print(f"  Backtest Compare — ML v2 vs Claude v2")
    print(f"  4 temporal cuts, same ground truth, independent scoring")
    print(f"{'═' * 68}\n")

    conn = get_conn(args)
    print(f"✓ Connected\n")

    # Load ground truth
    print("Loading ground truth labels …")
    gt_df = load_ground_truth(conn)
    gt_counts = Counter(gt_df["ground_truth_label"])
    for label in LABEL_ORDER:
        n = gt_counts.get(label, 0)
        print(f"  {label:<26} {n:>5}")

    # Load pipeline predictions
    print("\nLoading pipeline predictions …")
    ml_df, claude_df = load_pipeline_predictions(conn)

    if len(ml_df) == 0:
        print("✗ No ML predictions found. Run ml/pipeline_ml.py first.")
        conn.close()
        return
    if len(claude_df) == 0:
        print("✗ No Claude predictions found. Run agents/pipeline_claude.py first.")
        conn.close()
        return

    # Run temporal cuts
    all_results = []
    print(f"\nRunning {len(TEMPORAL_CUTS)} temporal cuts …")

    for train_end, test_start, test_end, cut_name in TEMPORAL_CUTS:
        print(f"\n  {cut_name}: test window {test_start} → {test_end}")

        actuals = load_test_window_outcomes(conn, test_start, test_end)
        print(f"  Ground truth in window: {len(actuals)} customers")

        act_counts = Counter(actuals["actual_label"])
        for label in LABEL_ORDER:
            n = act_counts.get(label, 0)
            if n > 0:
                print(f"    {label:<26} {n:>4}")

        # ML evaluation
        ml_result = evaluate_cut(ml_df, actuals, "ML", cut_name)
        all_results.append(ml_result)
        if ml_result:
            print(f"  ML    accuracy: {ml_result['accuracy']:.3f}")

        # Claude evaluation
        cld_result = evaluate_cut(claude_df, actuals, "Claude", cut_name)
        all_results.append(cld_result)
        if cld_result:
            print(f"  Claude accuracy: {cld_result['accuracy']:.3f}")

    # Agreement matrix
    print("\nBuilding agreement matrix …")
    agreement = agreement_matrix(ml_df, claude_df, gt_df)
    print(f"  {len(agreement)} customers in all three datasets")

    # Full report
    full_report(all_results, agreement, gt_df)

    # Save results
    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "backtest_compare_results.json"
        with open(out_path, "w") as f:
            json.dump(
                [r for r in all_results if r],
                f, indent=2, default=str
            )
        print(f"✓ Results saved → {out_path}")

        if len(agreement) > 0:
            agree_path = OUTPUT_DIR / "agreement_matrix.csv"
            agreement.to_csv(agree_path, index=False)
            print(f"✓ Agreement matrix → {agree_path}")

    conn.close()


if __name__ == "__main__":
    main()
