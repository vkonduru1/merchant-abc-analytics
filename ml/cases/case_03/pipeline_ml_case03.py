#!/usr/bin/env python3
"""
ml/cases/case_03/pipeline_ml_case03.py

CASE 03 — Post-Prediction Recency Override
───────────────────────────────────────────
PROBLEM IN CASE 02:
  dont_send FN = 77.4% across all cuts.
  In EVERY cut, the same 18 customers are misclassified:
    Actual=dont_send → Predicted=send_campaign

  Why: These customers have high campaign_influence_rate
  (RFC importance 22.6%) so the RFC classifies them as
  send_campaign. But they purchased very recently
  (recency_days < 14) — making them dont_send by definition.

  The RFC weights the lifetime campaign signal more heavily
  than the recency signal for these edge cases.
  recency_days IS the #1 feature (38.5% importance) but
  the threshold at which it overrides campaign signal
  for these 18 customers is not learned cleanly.

CHANGE IN CASE 03:
  Add a deterministic post-prediction override AFTER RFC:
    IF predicted_label = 'send_campaign'
    AND recency_days < RECENCY_DONT_SEND_THRESHOLD (14 days)
    → override to 'dont_send'

  This is a business rule, not a learned threshold.
  It is always correct by definition:
    A customer who purchased in the last 14 days
    should NEVER receive a campaign regardless of
    their lifetime campaign response pattern.

HYPOTHESIS:
  dont_send FN drops from 77.4% to near 0%.
  send_campaign recall stays 100% for all other customers.
  Overall accuracy improves by ~3-4%.
  Mature cuts (3-4) accuracy improves from 70.8%.

WHAT STAYS THE SAME:
  Everything from case_02 — walk-forward features,
  per-window training, RFC argmax.
  Only the post-prediction step changes.

Usage
    python ml/cases/case_03/pipeline_ml_case03.py
    python ml/cases/case_03/pipeline_ml_case03.py --dry-run
    python ml/cases/case_03/pipeline_ml_case03.py --port 5433
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

# ── Post-prediction override threshold ───────────────────────────────────
RECENCY_DONT_SEND_DAYS = 14   # business rule — never send within 14 days

RFC_PARAMS = {
    "n_estimators":     200,
    "min_samples_leaf":   3,
    "class_weight":  "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

TEMPORAL_CUTS = [
    ("2022-06-30", "2022-07-01", "2022-12-31", "Cut 1"),
    ("2022-12-31", "2023-01-01", "2023-06-30", "Cut 2"),
    ("2023-06-30", "2023-07-01", "2023-12-31", "Cut 3"),
    ("2023-12-31", "2024-01-01", "2024-06-30", "Cut 4"),
]

UPSERT_SQL = """
INSERT INTO customer_recommendations_ml (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-case03-recency-override')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""


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


# ── Reuse case_02 data loading (identical) ────────────────────────────────
def compute_windowed_features(conn, window_start: str,
                               window_end: str,
                               feature_cols: list) -> pd.DataFrame:
    query = f"""
        WITH window_orders AS (
            SELECT
                o.customer_id,
                COUNT(o.order_id)                        AS total_orders,
                SUM(o.total_price_usd)                   AS total_revenue,
                AVG(o.total_price_usd)                   AS average_order_value,
                MAX(o.order_created_at)                  AS last_order_date,
                MIN(o.order_created_at)                  AS first_order_date,
                SUM(o.total_price_usd)                   AS spend_last_3_months,
                SUM(o.total_price_usd)                   AS spend_last_12_months
            FROM orders o
            WHERE o.order_created_at <= '{window_end}'
            GROUP BY o.customer_id
        ),
        window_campaigns AS (
            SELECT
                ca.customer_id,
                COUNT(DISTINCT ca.most_recent_campaign_id)
                                                         AS total_campaigns_received
            FROM campaign_attribution ca
            WHERE ca.order_date <= '{window_end}'
            GROUP BY ca.customer_id
        ),
        window_email AS (
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
        LEFT JOIN window_email we      ON we.customer_id = cb.customer_id
        LEFT JOIN window_attribution wa ON wa.customer_id = cb.customer_id
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)
    for col in feature_cols:
        if col in df.columns:
            fill = df[col].max() if col == "recency_days" \
                   else df[col].median()
            df[col] = df[col].fillna(
                fill if pd.notna(fill) else 0
            )
        else:
            df[col] = 0.0
    return df


def compute_window_ground_truth(conn, window_start: str,
                                 window_end: str) -> pd.DataFrame:
    query = f"""
        WITH window_gt AS (
            SELECT
                d.customer_id,
                d.recency_days,
                COALESCE(SUM(CASE WHEN ca.campaign_influenced
                    AND ca.order_date >= '{window_start}'
                    AND ca.order_date <= '{window_end}'
                    THEN 1 ELSE 0 END), 0)  AS camp_orders_window,
                COALESCE(COUNT(CASE
                    WHEN ca.order_date >= '{window_start}'
                    AND ca.order_date <= '{window_end}'
                    THEN 1 END), 0)          AS total_orders_window
            FROM customer_derivatives d
            LEFT JOIN campaign_attribution ca
                ON ca.customer_id = d.customer_id
                AND ca.attribution_bucket != 'pre_campaign_era'
            GROUP BY d.customer_id, d.recency_days
        )
        SELECT customer_id,
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
            END AS ground_truth_label
        FROM window_gt
    """
    df = pd.read_sql(query, conn)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def normalise_vs(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        r = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].mean()) / r if r > 0 else 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
# CASE 03 CORE — Post-prediction recency override
# ══════════════════════════════════════════════════════════════════════════
def apply_recency_override(labels: np.ndarray,
                           recency_days: np.ndarray,
                           threshold: int = RECENCY_DONT_SEND_DAYS
                           ) -> tuple[np.ndarray, int]:
    """
    Override send_campaign → dont_send when recency < threshold.

    Business rule (always correct by definition):
      A customer who purchased within the last N days should
      never receive a campaign, regardless of their lifetime
      campaign response pattern.

    Returns modified labels and count of overrides applied.
    """
    labels    = labels.copy()
    overrides = 0
    for i, (label, recency) in enumerate(zip(labels, recency_days)):
        if label == "send_campaign" and recency < threshold:
            labels[i] = "dont_send"
            overrides += 1
    return labels, overrides


# ── Confusion matrix printer ──────────────────────────────────────────────
def print_cm(y_true, y_pred, cut_name: str,
             label: str = "ML case_03") -> dict:
    labels_present = [l for l in LABEL_ORDER
                      if l in set(y_true) or l in set(y_pred)]
    cm  = confusion_matrix(y_true, y_pred, labels=labels_present)
    col_w = 10
    print(f"\n    Confusion matrix — {label} — {cut_name}")
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
    errors = []
    for i in range(len(labels_present)):
        for j in range(len(labels_present)):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]),
                               labels_present[i], labels_present[j]))
    errors.sort(reverse=True)
    if errors:
        print(f"\n    Top misclassifications:")
        for count, actual, predicted in errors[:3]:
            print(f"      Actual={actual:<26} "
                  f"Predicted={predicted:<26} n={count:>4}")
    print()
    return fn_rates


# ── Reasoning builder ─────────────────────────────────────────────────────
def build_reasoning(label: str, conf: float,
                    row: pd.Series,
                    overridden: bool = False) -> tuple:
    inf = float(row.get("campaign_influence_rate", 0) or 0)
    rec = float(row.get("recency_days", 999) or 999)
    lag = float(row.get("avg_days_campaign_to_purchase", 0) or 0)

    if overridden:
        reasoning = (
            f"ML case_03: RFC predicted send_campaign "
            f"(campaign_influence_rate={inf:.0%}) but recency={rec:.0f}d "
            f"< {RECENCY_DONT_SEND_DAYS}d threshold. "
            f"Post-prediction override applied → dont_send. "
            f"Re-evaluate in 2 weeks."
        )
        signals = [
            f"RFC original: send_campaign (conf={conf:.0%})",
            f"Override: recency={rec:.0f}d < {RECENCY_DONT_SEND_DAYS}d threshold",
            f"campaign_influence_rate: {inf:.0%} — will be campaign candidate soon",
        ]
        return reasoning, signals

    templates = {
        "send_campaign": (
            f"ML case_03 classifies as send_campaign. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f}d, "
            f"avg lag={lag:.1f}d. RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%}",
             f"recency: {rec:.0f}d",
             f"RFC confidence: {conf:.0%}"]
        ),
        "dont_send": (
            f"ML case_03 classifies as dont_send. "
            f"recency={rec:.0f}d — post-purchase window. "
            f"RFC confidence {conf:.0%}.",
            [f"recency: {rec:.0f}d — post-purchase window",
             f"RFC confidence: {conf:.0%}"]
        ),
        "no_campaign_needed": (
            f"ML case_03 classifies as no_campaign_needed. "
            f"campaign_influence_rate={inf:.0%} — organic buyer. "
            f"RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%} — organic",
             f"RFC confidence: {conf:.0%}"]
        ),
        "no_campaign_impact": (
            f"ML case_03 classifies as no_campaign_impact. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f}d. "
            f"Suppress — 90-day re-engagement. RFC confidence {conf:.0%}.",
            [f"campaign_influence_rate: {inf:.0%}",
             f"recency: {rec:.0f}d",
             f"RFC confidence: {conf:.0%}"]
        ),
    }
    reasoning, signals = templates.get(label, (f"Label: {label}", []))
    return reasoning, signals


def write_to_db(conn, df_raw: pd.DataFrame,
                labels: np.ndarray, confs: np.ndarray,
                overridden_mask: np.ndarray) -> None:
    from psycopg2.extras import execute_batch
    records = []
    for i, (_, row) in enumerate(df_raw.iterrows()):
        reasoning, signals = build_reasoning(
            labels[i], confs[i], row,
            overridden=bool(overridden_mask[i])
        )
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
          f" (model_version=ml-case03-recency-override)")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Case 03 — post-prediction recency override.")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--host",     help="DB host")
    ap.add_argument("--port",     help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 68}")
    print(f"  ML Pipeline — Case 03: Post-Prediction Recency Override")
    print(f"  Change: IF send_campaign AND recency < {RECENCY_DONT_SEND_DAYS}d"
          f" → dont_send")
    print(f"{'═' * 68}\n")

    feature_cols = load_canonical()
    conn         = get_conn(args)
    print(f"✓ Connected\n")

    all_accs       = []
    all_fn         = {l: [] for l in LABEL_ORDER}
    total_overrides = 0
    cut_results    = []

    for train_end, test_start, test_end, cut_name in TEMPORAL_CUTS:
        print(f"  {'─' * 62}")
        print(f"  {cut_name}: train → {train_end} | "
              f"test {test_start} → {test_end}")

        df_train = compute_windowed_features(
            conn, "2022-01-01", train_end, feature_cols)
        df_train_gt = compute_window_ground_truth(
            conn, "2022-01-01", train_end)
        df_train = df_train.merge(df_train_gt, on="customer_id", how="left")
        df_train["ground_truth_label"] = df_train[
            "ground_truth_label"].fillna("no_campaign_impact")

        df_test = compute_windowed_features(
            conn, test_start, test_end, feature_cols)
        df_test_gt = compute_window_ground_truth(
            conn, test_start, test_end)
        df_test = df_test.merge(df_test_gt, on="customer_id", how="left")
        df_test["ground_truth_label"] = df_test[
            "ground_truth_label"].fillna("no_campaign_impact")

        if len(set(df_train["ground_truth_label"])) < 2:
            print(f"    Skipping — insufficient label diversity")
            continue

        df_train_norm = normalise_vs(df_train, feature_cols)
        df_test_norm  = normalise_vs(df_test,  feature_cols)

        X_train = df_train_norm[feature_cols].values.astype(np.float64)
        y_train = df_train["ground_truth_label"].values
        X_test  = df_test_norm[feature_cols].values.astype(np.float64)
        y_test  = df_test["ground_truth_label"].values

        rfc = RandomForestClassifier(**RFC_PARAMS)
        rfc.fit(X_train, y_train)

        # RFC argmax prediction (same as case_02)
        y_pred_rfc = rfc.predict(X_test)
        confs      = rfc.predict_proba(X_test).max(axis=1)

        # Get raw recency values for override check
        recency_raw = df_test["recency_days"].values

        # ── CASE 03 OVERRIDE ──────────────────────────────────────────────
        y_pred, n_overrides = apply_recency_override(
            y_pred_rfc, recency_raw, RECENCY_DONT_SEND_DAYS
        )
        overridden_mask = (y_pred != y_pred_rfc)
        total_overrides += n_overrides
        if n_overrides > 0:
            print(f"    Recency override applied: {n_overrides} customers "
                  f"send_campaign → dont_send (recency < {RECENCY_DONT_SEND_DAYS}d)")
        # ─────────────────────────────────────────────────────────────────

        acc = accuracy_score(y_test, y_pred)
        all_accs.append(acc)
        print(f"    Accuracy: {acc:.3f}  "
              f"(case_02 was "
              f"{[0.458,0.458,0.663,0.752][TEMPORAL_CUTS.index((train_end,test_start,test_end,cut_name))]:.3f})")

        fn_rates = print_cm(y_test, y_pred, cut_name)
        for label, rate in fn_rates.items():
            all_fn[label].append(rate)

        cut_results.append({
            "cut":       cut_name,
            "accuracy":  round(float(acc), 4),
            "overrides": n_overrides,
            "fn_rates":  {k: round(v, 4) for k, v in fn_rates.items()},
        })

        # Write Cut 4 to DB
        if cut_name == "Cut 4" and not args.dry_run:
            print(f"  Writing Cut 4 predictions to customer_recommendations_ml …")
            write_to_db(conn, df_test, y_pred, confs, overridden_mask)

    conn.close()

    # Summary
    print(f"\n{'═' * 68}")
    print(f"  Case 03 — Summary")
    print(f"{'═' * 68}")
    print(f"\n  Total recency overrides applied: {total_overrides}")
    print(f"\n  {'Cut':<8} {'Accuracy':>9} {'Overrides':>10}")
    print(f"  {'─' * 32}")
    for r in cut_results:
        print(f"  {r['cut']:<8} {r['accuracy']:>9.3f} "
              f"{r['overrides']:>10}")
    if all_accs:
        print(f"  {'─' * 32}")
        print(f"  {'Mean':>8} {np.mean(all_accs):>9.3f}")
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
    print(f"    case_01 all cuts   : 40.8%  mature: 61.0%")
    print(f"    case_02 all cuts   : 58.3%  mature: 70.8%")
    print(f"    Claude v2 all cuts : 66.5%  mature: 64.6%")
    if all_accs:
        m = np.mean(all_accs)
        mature_m = np.mean([r["accuracy"] for r in cut_results
                            if r["cut"] in ("Cut 3","Cut 4")]) \
                   if len(cut_results) >= 3 else 0
        print(f"    case_03 all cuts   : {m:.1%}  mature: {mature_m:.1%}")
    print(f"{'═' * 68}\n")

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "results_case03.json"
        with open(out, "w") as f:
            json.dump(cut_results, f, indent=2)
        print(f"✓ Results → {out}")


if __name__ == "__main__":
    main()
