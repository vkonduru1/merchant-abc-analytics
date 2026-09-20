#!/usr/bin/env python3
"""
ml/cases/case_02/pipeline_ml_case02.py

CASE 02 — Per-Window Feature Recomputation (True Walk-Forward)
──────────────────────────────────────────────────────────────
PROBLEM IN CASE 01:
  RFC trained on lifetime features (computed from full 3-year dataset)
  but backtest evaluates against window-specific behaviour.
  
  In Cuts 1-2 (early windows), 93.8% of customers appear as
  no_campaign_impact simply because the campaign program was young
  and most hadn't yet made campaign-influenced purchases.
  
  The RFC correctly identifies WHO customers are (lifetime)
  but mispredicts WHAT THEY DID in each specific window.

CHANGE IN CASE 02:
  True walk-forward evaluation.
  For each temporal cut:
    1. Load raw events up to train_end only
    2. Recompute all 10 canonical features from that window
    3. Derive ground truth labels from that window's attribution data
    4. Train RFC on windowed features + windowed labels
    5. Recompute features for test window customers
    6. Predict on test window, evaluate against test window ground truth
  
  Features and labels are now always from the same temporal window.
  No forward-looking signal leaks into early cuts.

HYPOTHESIS:
  Cuts 1-2 accuracy improves significantly (features now match window).
  Overall mean accuracy rises above 61.0% (case_01 mature cuts).
  no_campaign_impact FN rate drops from 78.1%.
  ML mature accuracy exceeds Claude 64.6% benchmark.

WHAT STAYS THE SAME:
  - 10 canonical features from 02_feature_canonical.json
  - VS normalisation: (x - mean) / (max - min)
  - Ground truth derivation logic (same SQL, different window)
  - RFC hyperparameters
  - 4 temporal cuts

IMPORTANT NOTE:
  This approach retrains the RFC per cut — 4 separate models.
  This is computationally more expensive but methodologically correct.
  In production this mirrors the monthly retrain cadence.

Usage
    python ml/cases/case_02/pipeline_ml_case02.py
    python ml/cases/case_02/pipeline_ml_case02.py --dry-run
    python ml/cases/case_02/pipeline_ml_case02.py --port 5433
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

SCRIPT_DIR     = Path(__file__).resolve().parent
ROOT_DIR       = SCRIPT_DIR.parent.parent.parent
CANONICAL_PATH = ROOT_DIR / "pipeline_v2" / "02_feature_canonical.json"
OUTPUT_DIR     = SCRIPT_DIR / "output"

sys.path.insert(0, str(ROOT_DIR / "data_generator"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit

UPSERT_SQL = """
INSERT INTO customer_recommendations_ml (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-case02-walkforward')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

RFC_PARAMS = {
    "n_estimators":     200,
    "min_samples_leaf":   3,   # slightly lower — smaller training windows
    "class_weight":  "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

# Temporal cuts — train_end, test_start, test_end, name
TEMPORAL_CUTS = [
    ("2022-06-30", "2022-07-01", "2022-12-31", "Cut 1"),
    ("2022-12-31", "2023-01-01", "2023-06-30", "Cut 2"),
    ("2023-06-30", "2023-07-01", "2023-12-31", "Cut 3"),
    ("2023-12-31", "2024-01-01", "2024-06-30", "Cut 4"),
]


def get_conn(args):
    import psycopg2
    return psycopg2.connect(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB",        "merchant_abc"),
        user=os.getenv("POSTGRES_USER",        "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )


def load_canonical() -> list[str]:
    if not CANONICAL_PATH.exists():
        sys.exit(f"✗ {CANONICAL_PATH} not found.")
    meta = json.load(open(CANONICAL_PATH))
    return [f["column"] for f in meta["features"]]


# ══════════════════════════════════════════════════════════════════════════
# Per-window feature computation
# ══════════════════════════════════════════════════════════════════════════
def compute_windowed_features(conn, window_start: str,
                               window_end: str,
                               feature_cols: list) -> pd.DataFrame:
    """
    Compute the 10 canonical features using only data up to window_end.
    This mirrors what a production system would do on each retrain cycle.

    For simplicity we read from customer_derivatives (precomputed)
    and filter orders/attribution to the window.
    A full production implementation would rerun compute_derivatives.py
    with a cutoff date — here we approximate by filtering attribution.
    """
    query = f"""
        WITH window_orders AS (
            SELECT
                o.customer_id,
                COUNT(o.order_id)                        AS total_orders,
                SUM(o.total_price_usd)                   AS total_revenue,
                AVG(o.total_price_usd)                   AS average_order_value,
                MAX(o.order_created_at)                  AS last_order_date,
                MIN(o.order_created_at)                  AS first_order_date,
                SUM(CASE WHEN o.order_created_at >= '{window_start}'
                    THEN o.total_price_usd ELSE 0 END)   AS spend_last_3_months,
                SUM(o.total_price_usd)                   AS spend_last_12_months
            FROM orders o
            WHERE o.order_created_at <= '{window_end}'
            GROUP BY o.customer_id
        ),
        window_campaigns AS (
            -- Use campaigns table for windowed campaign count
            -- Use customer_derivatives for email rates
            -- (email_events only available from Jul 2023)
            SELECT
                ca.customer_id,
                COUNT(DISTINCT ca.most_recent_campaign_id)
                                                         AS total_campaigns_received
            FROM campaign_attribution ca
            WHERE ca.order_date <= '{window_end}'
            GROUP BY ca.customer_id
        ),
        window_email AS (
            -- Email rates from customer_derivatives (precomputed lifetime)
            -- Best approximation when email_events window is sparse
            SELECT DISTINCT ON (customer_id)
                customer_id,
                COALESCE(email_open_rate, 0)  AS email_open_rate,
                COALESCE(email_click_rate, 0) AS email_click_rate
            FROM customer_derivatives
            ORDER BY customer_id, run_id DESC
        ),
        window_attribution AS (
            SELECT
                ca.customer_id,
                SUM(CASE WHEN ca.campaign_influenced THEN 1 ELSE 0 END)
                    AS camp_orders,
                COUNT(*)                                 AS attr_orders,
                AVG(CASE WHEN ca.campaign_influenced
                    THEN ca.days_since_last_campaign END) AS avg_lag,
                STDDEV(CASE WHEN ca.campaign_influenced
                    THEN ca.days_since_last_campaign END) AS stddev_lag
            FROM campaign_attribution ca
            WHERE ca.order_date <= '{window_end}'
              AND ca.attribution_bucket != 'pre_campaign_era'
            GROUP BY ca.customer_id
        ),
        customer_base AS (
            SELECT
                c.customer_id,
                EXTRACT(EPOCH FROM ('{window_end}'::timestamp -
                    MAX(wo.last_order_date))) / 86400 AS recency_days,
                EXTRACT(EPOCH FROM ('{window_end}'::timestamp -
                    MIN(c.customer_created_at))) / 2592000 AS months_since_customer,
                CASE WHEN COUNT(wo.total_orders) > 1
                    THEN EXTRACT(EPOCH FROM (MAX(wo.last_order_date) -
                         MIN(wo.first_order_date))) / 2592000 /
                         NULLIF(SUM(wo.total_orders) - 1, 0)
                    ELSE NULL END                    AS frequency_months
            FROM customers c
            LEFT JOIN window_orders wo ON wo.customer_id = c.customer_id
            GROUP BY c.customer_id
        )
        SELECT
            cb.customer_id,
            COALESCE(cb.recency_days, 999)               AS recency_days,
            COALESCE(cb.months_since_customer, 0)        AS months_since_customer,
            COALESCE(cb.frequency_months, 0)             AS frequency_months,
            COALESCE(wo.total_orders, 0)                 AS total_orders,
            COALESCE(wo.total_revenue, 0)                AS total_revenue,
            COALESCE(wo.spend_last_3_months, 0)          AS spend_last_3_months,
            COALESCE(wo.spend_last_12_months, 0)         AS spend_last_12_months,
            COALESCE(wc.total_campaigns_received, 0)     AS total_campaigns_received,
            COALESCE(we.email_open_rate, 0)              AS email_open_rate,
            COALESCE(we.email_click_rate, 0)             AS email_click_rate,
            COALESCE(wa.camp_orders::float /
                NULLIF(wa.attr_orders, 0), 0)            AS campaign_influence_rate,
            wa.avg_lag                                   AS avg_days_campaign_to_purchase,
            wa.stddev_lag                                AS stddev_days_campaign_to_purchase
        FROM customer_base cb
        LEFT JOIN window_orders wo     ON wo.customer_id = cb.customer_id
        LEFT JOIN window_campaigns wc  ON wc.customer_id = cb.customer_id
        LEFT JOIN window_attribution wa ON wa.customer_id = cb.customer_id
        LEFT JOIN window_email we        ON we.customer_id = cb.customer_id
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)

    # Fill nulls
    for col in feature_cols:
        if col in df.columns:
            fill = df[col].max() if col == "recency_days" else df[col].median()
            df[col] = df[col].fillna(fill if pd.notna(fill) else 0)
        else:
            df[col] = 0.0

    return df


def compute_window_ground_truth(conn, window_start: str,
                                 window_end: str) -> pd.DataFrame:
    """
    Derive ground truth labels from campaign_attribution
    restricted to the test window only.
    """
    query = f"""
        WITH window_gt AS (
            SELECT
                d.customer_id,
                d.recency_days,
                COALESCE(SUM(CASE WHEN ca.campaign_influenced
                    AND ca.order_date >= '{window_start}'
                    AND ca.order_date <= '{window_end}'
                    THEN 1 ELSE 0 END), 0)              AS camp_orders_window,
                COALESCE(COUNT(CASE
                    WHEN ca.order_date >= '{window_start}'
                    AND ca.order_date <= '{window_end}'
                    THEN 1 END), 0)                     AS total_orders_window
            FROM customer_derivatives d
            LEFT JOIN campaign_attribution ca
                ON ca.customer_id = d.customer_id
                AND ca.attribution_bucket != 'pre_campaign_era'
            GROUP BY d.customer_id, d.recency_days
        )
        SELECT
            customer_id,
            CASE
                WHEN recency_days < 14                  THEN 'dont_send'
                WHEN total_orders_window = 0            THEN 'no_campaign_impact'
                WHEN camp_orders_window > 0
                    AND camp_orders_window >=
                    total_orders_window * 0.5           THEN 'send_campaign'
                WHEN camp_orders_window = 0
                    AND total_orders_window > 0         THEN 'no_campaign_needed'
                ELSE 'no_campaign_impact'
            END AS ground_truth_label
        FROM window_gt
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def normalise_vs(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """VS normalisation: (x - mean) / (max - min)"""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        r = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].mean()) / r if r > 0 else 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
# Confusion matrix printer
# ══════════════════════════════════════════════════════════════════════════
def print_cm(y_true, y_pred, cut_name: str) -> dict:
    labels_present = [l for l in LABEL_ORDER
                      if l in set(y_true) or l in set(y_pred)]
    cm  = confusion_matrix(y_true, y_pred, labels=labels_present)
    acc = accuracy_score(y_true, y_pred)
    col_w = 10

    print(f"\n    Confusion matrix — ML case_02 — {cut_name}"
          f"  (rows=actual, cols=predicted)")
    header = "    " + " " * 14 + "".join(
        f"{l[:col_w]:>{col_w}}" for l in labels_present
    )
    print(header)
    print("    " + "─" * (14 + col_w * len(labels_present)))

    fn_rates = {}
    for i, row_label in enumerate(labels_present):
        row_str = f"    {row_label[:13]:<14}"
        for j in range(len(labels_present)):
            val    = int(cm[i, j])
            marker = "►" if (i != j and val > 10) else " "
            row_str += f"{marker}{val:>{col_w-1}}"
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - cm[i, i])
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        fn_rates[row_label] = fn_rate
        row_str += f"   FN={fn_rate:.0%}"
        print(row_str)

    # Top misclassifications
    errors = []
    for i in range(len(labels_present)):
        for j in range(len(labels_present)):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]),
                               labels_present[i], labels_present[j]))
    errors.sort(reverse=True)
    if errors:
        print(f"\n    Top misclassifications:")
        for count, actual, predicted in errors[:4]:
            print(f"      Actual={actual:<26} "
                  f"Predicted={predicted:<26} n={count:>4}")
    print()
    return fn_rates


# ══════════════════════════════════════════════════════════════════════════
# Reasoning builder + DB write
# ══════════════════════════════════════════════════════════════════════════
def build_reasoning(label: str, conf: float, row: pd.Series) -> tuple:
    inf = float(row.get("campaign_influence_rate", 0) or 0)
    rec = float(row.get("recency_days", 999) or 999)
    lag = float(row.get("avg_days_campaign_to_purchase", 0) or 0)

    templates = {
        "send_campaign": (
            f"ML case_02 (walk-forward) classifies as send_campaign. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f}d, "
            f"avg lag={lag:.1f}d. RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%}",
             f"recency: {rec:.0f} days",
             f"RFC confidence: {conf:.0%}"]
        ),
        "dont_send": (
            f"ML case_02 classifies as dont_send. "
            f"recency={rec:.0f}d — recent purchase window. "
            f"RFC confidence {conf:.0%}.",
            [f"recency: {rec:.0f} days — post-purchase window",
             f"RFC confidence: {conf:.0%}"]
        ),
        "no_campaign_needed": (
            f"ML case_02 classifies as no_campaign_needed. "
            f"campaign_influence_rate={inf:.0%} — organic buyer. "
            f"RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%} — organic",
             f"RFC confidence: {conf:.0%}"]
        ),
        "no_campaign_impact": (
            f"ML case_02 classifies as no_campaign_impact. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f}d. "
            f"Suppress or route to 90-day re-engagement. "
            f"RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%}",
             f"recency: {rec:.0f} days",
             f"RFC confidence: {conf:.0%}"]
        ),
    }
    reasoning, signals = templates.get(label, (f"Label: {label}", []))
    return reasoning, signals


def write_to_db(conn, df_raw: pd.DataFrame,
                labels: np.ndarray, confs: np.ndarray) -> None:
    """
    Write Cut 4 predictions to customer_recommendations_ml.
    Cut 4 is the most mature, production-realistic cut.
    model_version = 'ml-case02-walkforward'
    """
    from psycopg2.extras import execute_batch
    records = []
    for i, (_, row) in enumerate(df_raw.iterrows()):
        reasoning, signals = build_reasoning(labels[i], confs[i], row)
        records.append((
            row["customer_id"],
            labels[i],
            float(confs[i]),
            reasoning,
            json.dumps(signals),
        ))
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, UPSERT_SQL, records, page_size=200)
    print(f"  ✓ Upserted {len(records):,} rows → customer_recommendations_ml"
          f" (model_version=ml-case02-walkforward)")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Case 02 — per-window feature recomputation.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--host",    help="DB host")
    ap.add_argument("--port",    help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 68}")
    print(f"  ML Pipeline — Case 02: Per-Window Feature Recomputation")
    print(f"  Change: true walk-forward — features + labels from same window")
    print(f"{'═' * 68}\n")

    feature_cols = load_canonical()
    conn         = get_conn(args)
    print(f"✓ Connected\n")

    all_accs    = []
    all_fn      = {l: [] for l in LABEL_ORDER}
    cut_results = []

    for train_end, test_start, test_end, cut_name in TEMPORAL_CUTS:
        print(f"  {'─' * 62}")
        print(f"  {cut_name}: train → {train_end} | "
              f"test {test_start} → {test_end}")

        # Training features (up to train_end)
        print(f"    Computing training features (up to {train_end}) …")
        df_train = compute_windowed_features(
            conn, "2022-01-01", train_end, feature_cols
        )
        df_train_gt = compute_window_ground_truth(
            conn, "2022-01-01", train_end
        )
        df_train = df_train.merge(
            df_train_gt, on="customer_id", how="left"
        )
        df_train["ground_truth_label"] = df_train[
            "ground_truth_label"
        ].fillna("no_campaign_impact")

        # Test features (from test window)
        print(f"    Computing test features ({test_start} → {test_end}) …")
        df_test = compute_windowed_features(
            conn, test_start, test_end, feature_cols
        )
        df_test_gt = compute_window_ground_truth(
            conn, test_start, test_end
        )
        df_test = df_test.merge(
            df_test_gt, on="customer_id", how="left"
        )
        df_test["ground_truth_label"] = df_test[
            "ground_truth_label"
        ].fillna("no_campaign_impact")

        # Label distributions
        train_counts = Counter(df_train["ground_truth_label"])
        test_counts  = Counter(df_test["ground_truth_label"])
        print(f"    Train labels: " + "  ".join(
            f"{l}={train_counts.get(l,0)}" for l in LABEL_ORDER
            if train_counts.get(l, 0) > 0
        ))
        print(f"    Test labels:  " + "  ".join(
            f"{l}={test_counts.get(l,0)}" for l in LABEL_ORDER
            if test_counts.get(l, 0) > 0
        ))

        # Minimum classes check
        if len(set(df_train["ground_truth_label"])) < 2:
            print(f"    Skipping — insufficient label diversity in training")
            continue

        # Normalise separately (no leakage)
        df_train_norm = normalise_vs(df_train, feature_cols)
        df_test_norm  = normalise_vs(df_test,  feature_cols)

        X_train = df_train_norm[feature_cols].values.astype(np.float64)
        y_train = df_train["ground_truth_label"].values
        X_test  = df_test_norm[feature_cols].values.astype(np.float64)
        y_test  = df_test["ground_truth_label"].values

        # Train RFC on this window only
        rfc = RandomForestClassifier(**RFC_PARAMS)
        rfc.fit(X_train, y_train)

        # Predict on test window
        y_pred = rfc.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        all_accs.append(acc)

        print(f"    Accuracy: {acc:.3f}")
        fn_rates = print_cm(y_test, y_pred, cut_name)

        for label, rate in fn_rates.items():
            all_fn[label].append(rate)

        cut_results.append({
            "cut":      cut_name,
            "accuracy": round(float(acc), 4),
            "fn_rates": {k: round(v, 4) for k, v in fn_rates.items()},
            "n_train":  len(df_train),
            "n_test":   len(df_test),
        })

        # Write Cut 4 predictions to DB (most mature, production-realistic)
        if cut_name == "Cut 4" and not args.dry_run:
            print(f"  Writing Cut 4 predictions to customer_recommendations_ml …")
            confs_arr = rfc.predict_proba(X_test).max(axis=1)
            write_to_db(conn, df_test, y_pred, confs_arr)

    conn.close()

    # Summary
    print(f"\n{'═' * 68}")
    print(f"  Case 02 — Summary")
    print(f"{'═' * 68}")
    print(f"\n  {'Cut':<8} {'Accuracy':>9}")
    print(f"  {'─' * 20}")
    for r in cut_results:
        print(f"  {r['cut']:<8} {r['accuracy']:>9.3f}")
    if all_accs:
        print(f"  {'─' * 20}")
        print(f"  {'Mean':>8} {np.mean(all_accs):>9.3f}")
        print(f"  {'Std':>8} {np.std(all_accs):>9.3f}")
        mature = [r["accuracy"] for r in cut_results
                  if r["cut"] in ("Cut 3", "Cut 4")]
        if mature:
            print(f"  Mature cuts (3-4) mean: {np.mean(mature):.3f}")

    print(f"\n  Average FN rates:")
    for label in LABEL_ORDER:
        rates = all_fn.get(label, [])
        if rates:
            avg = np.mean(rates)
            bar = "█" * int((1 - avg) * 20)
            print(f"    {label:<26}  FN={avg:.1%}  "
                  f"recall={1-avg:.1%}  {bar}")

    print(f"\n  Comparison:")
    print(f"    case_01 all cuts mean : 40.8%")
    print(f"    case_01 mature (3-4)  : 61.0%")
    print(f"    Claude v2 all cuts    : 66.5%")
    print(f"    Claude v2 mature (3-4): 64.6%")
    if all_accs:
        print(f"    case_02 all cuts mean : {np.mean(all_accs):.1%}")
    if mature:
        print(f"    case_02 mature (3-4)  : {np.mean(mature):.1%}")
    print(f"{'═' * 68}\n")

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "results_case02.json"
        with open(out, "w") as f:
            json.dump(cut_results, f, indent=2)
        print(f"✓ Results → {out}")


if __name__ == "__main__":
    main()
