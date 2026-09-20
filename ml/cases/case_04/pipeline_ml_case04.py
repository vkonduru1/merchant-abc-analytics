#!/usr/bin/env python3
"""
ml/cases/case_04/pipeline_ml_case04.py

CASE 04 — Confidence-Based ML + Claude Ensemble (HITL Routing)
──────────────────────────────────────────────────────────────
WHAT THIS IS:
  Not a new ML model. A routing layer that decides — per customer —
  whether the ML prediction or the Claude prediction is more reliable,
  using RFC confidence score as the routing signal.

  This is the agentic ensemble: statistical prediction (ML) +
  contextual reasoning (Claude) routed by confidence.
  Each covers the other's blind spots.

WHY THIS WORKS:
  ML failure modes:   distributional edge cases, sparse training classes,
                      static feature / dynamic label mismatch in early windows
  Claude failure modes: statistical patterns it cannot infer from rules,
                        no LTV-based rank ordering, conservative on send_campaign

  These failure modes are largely ORTHOGONAL.
  When ML is uncertain (low confidence) Claude's reasoning fills the gap.
  When ML is certain (high confidence) Claude's rules are bypassed.

ROUTING LOGIC:
  RFC confidence >= 0.80  → ML label (high confidence, trust the model)
  RFC confidence <  0.60  → Claude label (low confidence, trust the reasoning)
  0.60 <= confidence < 0.80 → ML label + flag for review (middle band)

  This mirrors the CCAR-F D1D HITL escalation pattern:
  high confidence → automated decision
  low confidence  → human (or LLM) review
  disagreement    → escalation queue

INPUTS:
  customer_recommendations_ml     → case_02 predictions + confidence
  customer_recommendations_claude → Claude v2 predictions

HYPOTHESIS:
  Ensemble accuracy > both individual pipelines on mature cuts.
  dont_send FN improves (Claude FN=9.7% vs ML FN=74% on this label).
  send_campaign recall stays high (ML=100% on mature cuts).
  Expected mature accuracy: 72-76%

Usage
    python ml/cases/case_04/pipeline_ml_case04.py
    python ml/cases/case_04/pipeline_ml_case04.py --dry-run
    python ml/cases/case_04/pipeline_ml_case04.py --port 5433
    python ml/cases/case_04/pipeline_ml_case04.py --high 0.85 --low 0.55
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

sys.path.insert(0, str(ROOT_DIR / "data_generator"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

# Default routing thresholds
HIGH_CONF_THRESHOLD = 0.80   # above → trust ML
LOW_CONF_THRESHOLD  = 0.60   # below → trust Claude

TEMPORAL_CUTS = [
    ("2022-07-01", "2022-12-31", "Cut 1"),
    ("2023-01-01", "2023-06-30", "Cut 2"),
    ("2023-07-01", "2023-12-31", "Cut 3"),
    ("2024-01-01", "2024-06-30", "Cut 4"),
]

UPSERT_SQL = """
INSERT INTO customer_recommendations_ml (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-case04-ensemble')
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


# ══════════════════════════════════════════════════════════════════════════
# Load predictions from both pipelines
# ══════════════════════════════════════════════════════════════════════════
def load_predictions(conn) -> pd.DataFrame:
    """
    Load ML (case_02) and Claude v2 predictions.
    Merge on customer_id.
    """
    ml_df = pd.read_sql("""
        SELECT DISTINCT ON (customer_id)
            customer_id,
            recommendation_label AS ml_label,
            confidence_score     AS ml_confidence,
            model_version        AS ml_version
        FROM customer_recommendations_ml
        ORDER BY customer_id, recommended_at DESC
    """, conn)
    ml_df["customer_id"] = ml_df["customer_id"].astype(str)

    claude_df = pd.read_sql("""
        SELECT DISTINCT ON (customer_id)
            customer_id,
            recommendation_label AS claude_label,
            confidence_score     AS claude_confidence
        FROM customer_recommendations_claude
        ORDER BY customer_id, recommended_at DESC
    """, conn)
    claude_df["customer_id"] = claude_df["customer_id"].astype(str)

    df = ml_df.merge(claude_df, on="customer_id", how="inner")
    print(f"  {len(df):,} customers with both ML and Claude predictions")
    print(f"  ML version : {df['ml_version'].iloc[0] if len(df) else 'none'}")
    return df


def load_window_ground_truth(conn, test_start: str,
                              test_end: str) -> pd.DataFrame:
    query = f"""
        WITH window_gt AS (
            SELECT
                d.customer_id,
                d.recency_days,
                COALESCE(SUM(CASE WHEN ca.campaign_influenced
                    AND ca.order_date >= '{test_start}'
                    AND ca.order_date <= '{test_end}'
                    THEN 1 ELSE 0 END), 0)  AS camp_orders_window,
                COALESCE(COUNT(CASE
                    WHEN ca.order_date >= '{test_start}'
                    AND ca.order_date <= '{test_end}'
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


# ══════════════════════════════════════════════════════════════════════════
# CASE 04 CORE — Confidence-based routing
# ══════════════════════════════════════════════════════════════════════════
def route_ensemble(df: pd.DataFrame,
                   high_threshold: float,
                   low_threshold: float) -> pd.DataFrame:
    """
    Route each customer's final label based on ML confidence:
      ML conf >= high  → ML label (trust the model)
      ML conf <  low   → Claude label (trust the reasoning)
      middle band      → ML label + flag for review

    Also records routing decision and source for transparency.
    """
    df = df.copy()
    final_labels  = []
    final_confs   = []
    routing_source = []
    review_flags  = []

    for _, row in df.iterrows():
        ml_conf  = float(row["ml_confidence"] or 0)
        ml_label = row["ml_label"]
        cld_label = row["claude_label"]

        if ml_conf >= high_threshold:
            final_labels.append(ml_label)
            final_confs.append(ml_conf)
            routing_source.append("ml_high_conf")
            review_flags.append(False)
        elif ml_conf < low_threshold:
            final_labels.append(cld_label)
            final_confs.append(float(row["claude_confidence"] or 0.7))
            routing_source.append("claude_low_ml_conf")
            review_flags.append(False)
        else:
            # Middle band — ML label but flag for review
            final_labels.append(ml_label)
            final_confs.append(ml_conf)
            routing_source.append("ml_middle_band_flagged")
            review_flags.append(True)

    df["final_label"]    = final_labels
    df["final_conf"]     = final_confs
    df["routing_source"] = routing_source
    df["review_flag"]    = review_flags
    return df


# ── Confusion matrix printer ──────────────────────────────────────────────
def print_cm(y_true, y_pred, cut_name: str) -> dict:
    labels_present = [l for l in LABEL_ORDER
                      if l in set(y_true) or l in set(y_pred)]
    cm    = confusion_matrix(y_true, y_pred, labels=labels_present)
    col_w = 10
    print(f"\n    Confusion matrix — Ensemble — {cut_name}")
    header = "    " + " " * 14 + "".join(
        f"{l[:col_w]:>{col_w}}" for l in labels_present)
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


# ── Write to DB ───────────────────────────────────────────────────────────
def write_to_db(conn, result_df: pd.DataFrame) -> None:
    from psycopg2.extras import execute_batch
    records = []
    for _, row in result_df.iterrows():
        source = row["routing_source"]
        flag   = " [REVIEW]" if row["review_flag"] else ""
        reasoning = (
            f"Ensemble case_04: routed via {source}{flag}. "
            f"ML label={row['ml_label']} (conf={row['ml_confidence']:.0%}), "
            f"Claude label={row['claude_label']}. "
            f"Final={row['final_label']}."
        )
        signals = [
            f"routing: {source}",
            f"ml_label: {row['ml_label']} (conf={row['ml_confidence']:.0%})",
            f"claude_label: {row['claude_label']}",
            f"review_flag: {row['review_flag']}",
        ]
        records.append((
            row["customer_id"],
            row["final_label"],
            float(row["final_conf"]),
            reasoning,
            json.dumps(signals),
        ))
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, UPSERT_SQL, records, page_size=200)
    print(f"  ✓ Upserted {len(records):,} rows → customer_recommendations_ml"
          f" (model_version=ml-case04-ensemble)")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Case 04 — confidence-based ML + Claude ensemble.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--high",    type=float, default=HIGH_CONF_THRESHOLD,
                    help=f"ML conf threshold for ML label (default: {HIGH_CONF_THRESHOLD})")
    ap.add_argument("--low",     type=float, default=LOW_CONF_THRESHOLD,
                    help=f"ML conf threshold for Claude label (default: {LOW_CONF_THRESHOLD})")
    ap.add_argument("--host",    help="DB host")
    ap.add_argument("--port",    help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 68}")
    print(f"  ML Pipeline — Case 04: Confidence-Based Ensemble")
    print(f"  ML conf >= {args.high:.2f} → ML label")
    print(f"  ML conf <  {args.low:.2f} → Claude label")
    print(f"  Middle band → ML label + review flag")
    print(f"{'═' * 68}\n")

    conn = get_conn(args)
    print(f"✓ Connected\n")

    print("Loading predictions from both pipelines …")
    df_preds = load_predictions(conn)

    # Show routing distribution
    df_routed = route_ensemble(df_preds, args.high, args.low)
    routing_counts = Counter(df_routed["routing_source"])
    print(f"\n  Routing distribution (all 496 customers):")
    for source, n in sorted(routing_counts.items()):
        pct = n * 100 / len(df_routed)
        print(f"    {source:<35} {n:>4}  ({pct:.1f}%)")
    review_n = int(df_routed["review_flag"].sum())
    print(f"    Flagged for review                  {review_n:>4}")

    # Label agreement analysis
    agree_n = int((df_routed["ml_label"] == df_routed["claude_label"]).sum())
    print(f"\n  ML ↔ Claude agreement: "
          f"{agree_n}/{len(df_routed)} ({agree_n*100/len(df_routed):.1f}%)")

    # Temporal cut evaluation
    all_accs    = []
    all_fn      = {l: [] for l in LABEL_ORDER}
    cut_results = []

    print(f"\nRunning {len(TEMPORAL_CUTS)} temporal cuts …")
    for test_start, test_end, cut_name in TEMPORAL_CUTS:
        print(f"\n  {'─' * 62}")
        print(f"  {cut_name}: {test_start} → {test_end}")

        gt_df = load_window_ground_truth(conn, test_start, test_end)
        gt_counts = Counter(gt_df["ground_truth_label"])
        print(f"  GT labels: " + "  ".join(
            f"{l}={gt_counts.get(l,0)}"
            for l in LABEL_ORDER if gt_counts.get(l, 0) > 0
        ))

        merged = df_routed.merge(
            gt_df, on="customer_id", how="inner"
        )
        if len(merged) == 0:
            continue

        y_pred = merged["final_label"].values
        y_true = merged["ground_truth_label"].values

        acc    = accuracy_score(y_true, y_pred)
        all_accs.append(acc)
        print(f"  Ensemble accuracy: {acc:.3f}")

        # Compare ML-only and Claude-only for this cut
        ml_acc  = accuracy_score(y_true, merged["ml_label"].values)
        cld_acc = accuracy_score(y_true, merged["claude_label"].values)
        print(f"  ML-only accuracy : {ml_acc:.3f}")
        print(f"  Claude accuracy  : {cld_acc:.3f}")
        winner = "Ensemble" if acc >= max(ml_acc, cld_acc) else \
                 ("ML" if ml_acc >= cld_acc else "Claude")
        print(f"  Winner this cut  : {winner}")

        fn_rates = print_cm(y_true, y_pred, cut_name)
        for label, rate in fn_rates.items():
            all_fn[label].append(rate)

        cut_results.append({
            "cut":        cut_name,
            "accuracy":   round(float(acc), 4),
            "ml_only":    round(float(ml_acc), 4),
            "claude_only": round(float(cld_acc), 4),
            "fn_rates":   {k: round(v, 4) for k, v in fn_rates.items()},
        })

    # Summary
    print(f"\n{'═' * 68}")
    print(f"  Case 04 — Ensemble Summary")
    print(f"  Routing: ML conf≥{args.high} → ML | "
          f"conf<{args.low} → Claude | middle → ML+flag")
    print(f"{'═' * 68}")
    print(f"\n  {'Cut':<8} {'Ensemble':>9} {'ML-only':>9} {'Claude':>9}")
    print(f"  {'─' * 40}")
    for r in cut_results:
        winner_marker = (
            "← E" if r["accuracy"] >= max(r["ml_only"], r["claude_only"])
            else ("← M" if r["ml_only"] >= r["claude_only"] else "← C")
        )
        print(f"  {r['cut']:<8} {r['accuracy']:>9.3f} "
              f"{r['ml_only']:>9.3f} {r['claude_only']:>9.3f}  {winner_marker}")
    if all_accs:
        print(f"  {'─' * 40}")
        print(f"  {'Mean':>8} {np.mean(all_accs):>9.3f}")
        mature = [r["accuracy"] for r in cut_results
                  if r["cut"] in ("Cut 3", "Cut 4")]
        if mature:
            print(f"  Mature (3-4)     {np.mean(mature):>9.3f}")

    print(f"\n  Average FN rates (Ensemble):")
    for label in LABEL_ORDER:
        rates = all_fn.get(label, [])
        if rates:
            avg = np.mean(rates)
            bar = "█" * int((1 - avg) * 20)
            print(f"    {label:<26}  FN={avg:.1%}  "
                  f"recall={1-avg:.1%}  {bar}")

    print(f"\n  Final comparison:")
    print(f"    case_01 mature: 61.0%  |  all: 40.8%")
    print(f"    case_02 mature: 70.8%  |  all: 58.3%")
    print(f"    case_03 mature: 64.5%  |  all: 80.0%  [rejected]")
    print(f"    Claude  mature: 64.6%  |  all: 66.5%")
    if all_accs:
        mature_m = np.mean([r["accuracy"] for r in cut_results
                            if r["cut"] in ("Cut 3","Cut 4")])
        print(f"    case_04 mature: {mature_m:.1%}  |  "
              f"all: {np.mean(all_accs):.1%}")
    print(f"{'═' * 68}\n")

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "results_case04.json"
        with open(out, "w") as f:
            json.dump(cut_results, f, indent=2)
        print(f"✓ Results → {out}")
        write_to_db(conn, df_routed)

    conn.close()


if __name__ == "__main__":
    main()
