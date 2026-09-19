#!/usr/bin/env python3
"""
ml/ml_merge.py — Merge cluster + RFC likelihoods, rank, write to DB.

Implements the VS PredictAlly logical OR merge (Step 11):
  final_likely = cluster_likely OR rfc_likely
  0 | 0 = 0  → no_campaign_impact
  0 | 1 = 1  → use RFC label
  1 | 0 = 1  → use cluster label
  1 | 1 = 1  → RFC label wins (higher statistical confidence)

Rank formula (VS Step 13):
  Likely   → rank = ltv_mean × rfc_confidence × 1,000,000
  Unlikely → rank = ltv_mean × rfc_confidence × active_weight × 1,000,000
             active_weight = 0.5 if recency_days > 180 else 1.0

HYP segmentation (VS Steps 14-15):
  From LIKELY:   top 20% → HYP_TP, rest → HYP_FP
  From UNLIKELY: top 20% → HYP_FN, rest → HYP_TN

Final label mapping:
  HYP_TP → send_campaign        (top likely → send)
  HYP_FP → no_campaign_needed   (likely but lower rank → protect)
  HYP_FN → dont_send            (top unlikely → watch closely)
  HYP_TN → no_campaign_impact   (bottom unlikely → suppress)

Writes to customer_recommendations (upsert).

Usage
    python ml/ml_merge.py              # merge + write to DB
    python ml/ml_merge.py --dry-run    # report only
    python ml/ml_merge.py --port 5433
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

INACTIVITY_DAYS    = 180    # beyond this → penalise rank
ACTIVE_WEIGHT      = 1.0
INACTIVE_WEIGHT    = 0.5
HYP_TP_PERCENTILE  = 70     # top 30% of likely  → HYP_TP
HYP_FN_PERCENTILE  = 80     # top 20% of unlikely → HYP_FN
RANK_SCALE         = 1_000_000

UPSERT_SQL = """
INSERT INTO customer_recommendations (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-v1-kmeans-rfc')
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
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )


def load_inputs() -> pd.DataFrame:
    """Load and merge cluster + RFC results + raw features."""
    cluster_path  = OUTPUT_DIR / "cluster_results.csv"
    rfc_path      = OUTPUT_DIR / "rfc_results.csv"
    raw_path      = OUTPUT_DIR / "features_raw.csv"

    for p in [cluster_path, rfc_path, raw_path]:
        if not p.exists():
            sys.exit(f"✗ {p.name} not found. Run prior ML steps first.")

    df_cluster = pd.read_csv(cluster_path)
    df_rfc     = pd.read_csv(rfc_path)
    df_raw     = pd.read_csv(raw_path)

    df = df_cluster.merge(df_rfc, on="customer_id", how="inner")
    df = df.merge(
        df_raw[["customer_id", "total_revenue", "recency_days",
                "campaign_influence_rate", "attribution_segment"]],
        on="customer_id", how="left"
    )
    print(f"  Merged {len(df):,} customers")
    return df


def compute_rfc_likely(df: pd.DataFrame) -> pd.Series:
    """RFC likely = top predicted label has probability >= 0.5."""
    send_prob    = df.get("send_campaign_prob", 0)
    needed_prob  = df.get("no_campaign_needed_prob", 0)
    likely_prob  = send_prob.fillna(0) + needed_prob.fillna(0)
    return (likely_prob >= 0.5).astype(int)


def or_merge(cluster_likely: pd.Series,
             rfc_likely: pd.Series) -> pd.Series:
    """Logical OR: 1 if either model says Likely."""
    return ((cluster_likely == 1) | (rfc_likely == 1)).astype(int)


def compute_rank(df: pd.DataFrame,
                 final_likely: pd.Series) -> pd.Series:
    """
    VS Rank formula (Step 13):
      Likely   → ltv × rfc_confidence × 1M
      Unlikely → ltv × rfc_confidence × active_weight × 1M
    """
    ltv    = df["total_revenue"].fillna(0)
    conf   = df["rfc_confidence"].fillna(0.5)
    active = (df["recency_days"].fillna(999) <= INACTIVITY_DAYS
              ).map({True: ACTIVE_WEIGHT, False: INACTIVE_WEIGHT})

    rank = ltv * conf * RANK_SCALE
    # Penalise inactive customers in the unlikely pool
    rank = np.where(final_likely == 1, rank, rank * active)
    return pd.Series(rank, name="rank")


def assign_hyp_labels(df: pd.DataFrame,
                      final_likely: pd.Series,
                      rank: pd.Series) -> pd.Series:
    """
    VS HYP segmentation (Steps 14-15):
      Likely pool   → rank DESC → top 20% = HYP_TP, rest = HYP_FP
      Unlikely pool → rank DESC → top 20% = HYP_FN, rest = HYP_TN

    Map to 4 recommendation labels:
      HYP_TP → send_campaign
      HYP_FP → no_campaign_needed
      HYP_FN → dont_send
      HYP_TN → no_campaign_impact
    """
    likely_mask   = final_likely == 1
    unlikely_mask = ~likely_mask

    labels = pd.Series(["no_campaign_impact"] * len(df), name="final_label")

    # Likely pool
    if likely_mask.sum() > 0:
        likely_ranks    = rank[likely_mask]
        tp_threshold    = likely_ranks.quantile(HYP_TP_PERCENTILE / 100)
        hyp_tp_mask     = likely_mask & (rank >= tp_threshold)
        hyp_fp_mask     = likely_mask & (rank <  tp_threshold)
        labels[hyp_tp_mask] = "send_campaign"
        labels[hyp_fp_mask] = "no_campaign_needed"

    # Unlikely pool
    if unlikely_mask.sum() > 0:
        unlikely_ranks  = rank[unlikely_mask]
        fn_threshold    = unlikely_ranks.quantile(HYP_FN_PERCENTILE / 100)
        hyp_fn_mask     = unlikely_mask & (rank >= fn_threshold)
        hyp_tn_mask     = unlikely_mask & (rank <  fn_threshold)
        labels[hyp_fn_mask] = "dont_send"
        labels[hyp_tn_mask] = "no_campaign_impact"

    return labels


def build_reasoning(row: pd.Series) -> str:
    """
    Build a concise ML-grounded reasoning string.
    Claude could extend this in a future pass.
    """
    label = row["final_label"]
    ltv   = row.get("total_revenue", 0)
    conf  = row.get("rfc_confidence", 0)
    inf   = row.get("campaign_influence_rate", 0) or 0
    rec   = row.get("recency_days", 0) or 0
    cl    = int(row.get("cluster_likely", 0))
    rl    = int(row.get("rfc_likely", 0))

    model_agree = "KMeans++ and RFC both" if cl == rl else (
        "KMeans++" if cl else "RFC"
    )

    if label == "send_campaign":
        return (f"ML model classifies as HYP_TP (top-ranked Likely). "
                f"{model_agree} signal Likely. "
                f"Campaign influence rate {inf:.0%}, recency {rec:.0f} days, "
                f"LTV ${ltv:.2f}. RFC confidence {conf:.0%}.")
    elif label == "no_campaign_needed":
        return (f"ML model classifies as HYP_FP (Likely, lower rank). "
                f"Purchases occur without campaign influence ({inf:.0%} rate). "
                f"LTV ${ltv:.2f}. Send NPIs and loyalty offers only.")
    elif label == "dont_send":
        return (f"ML model classifies as HYP_FN (top-ranked Unlikely). "
                f"Monitor closely — high-rank unlikely customers are "
                f"potential converters. Recency {rec:.0f} days, "
                f"RFC confidence {conf:.0%}.")
    else:
        return (f"ML model classifies as HYP_TN (Unlikely, lower rank). "
                f"Campaign influence rate {inf:.0%}. "
                f"Suppress or route to 90-day re-engagement flow.")


def build_signals(row: pd.Series) -> list:
    signals = []
    inf  = float(row.get("campaign_influence_rate", 0) or 0)
    conf = float(row.get("rfc_confidence", 0) or 0)
    ltv  = float(row.get("total_revenue", 0) or 0)
    rec  = float(row.get("recency_days", 999) or 999)
    cl   = int(row.get("cluster_likely", 0))
    rl   = int(row.get("rfc_likely", 0))

    signals.append(f"KMeans++ likelihood: {'Likely' if cl else 'Unlikely'}")
    signals.append(f"RFC likelihood: {'Likely' if rl else 'Unlikely'} "
                   f"(conf {conf:.0%})")
    signals.append(f"Campaign influence rate: {inf:.0%}")
    if ltv > 0:
        signals.append(f"LTV: ${ltv:.2f}  Recency: {rec:.0f} days")
    return signals


def report(df: pd.DataFrame, final_labels: pd.Series,
           final_likely: pd.Series, rank: pd.Series) -> None:
    from collections import Counter
    counts = Counter(final_labels)
    total  = len(df)
    likely_n   = int(final_likely.sum())
    unlikely_n = total - likely_n

    print(f"\n{'═' * 62}")
    print(f"  ML Merge Report (VS OR logic + HYP segmentation)")
    print(f"{'═' * 62}")
    print(f"  Total customers     : {total:,}")
    print(f"  Likely pool (OR=1)  : {likely_n:,}  ({likely_n*100/total:.1f}%)")
    print(f"  Unlikely pool (OR=0): {unlikely_n:,}  ({unlikely_n*100/total:.1f}%)")
    print(f"\n  Final label distribution:")
    label_order = ["send_campaign", "dont_send",
                   "no_campaign_needed", "no_campaign_impact"]
    for label in label_order:
        n   = counts.get(label, 0)
        pct = n * 100 / total if total else 0
        print(f"    {label:<26} {n:>5}  ({pct:.1f}%)")
    print(f"\n  Rank stats:")
    print(f"    Min   : {rank.min():>12,.0f}")
    print(f"    Median: {rank.median():>12,.0f}")
    print(f"    Max   : {rank.max():>12,.0f}")
    print(f"{'═' * 62}\n")


def write_to_db(conn, records: list) -> None:
    from psycopg2.extras import execute_batch
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, UPSERT_SQL, records, page_size=200)
    print(f"✓ Upserted {len(records):,} rows into customer_recommendations")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge cluster + RFC likelihoods and write recommendations.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--host", help="DB host")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    print("Loading ML outputs …")
    df = load_inputs()

    rfc_likely    = compute_rfc_likely(df)
    final_likely  = or_merge(df["cluster_likely"], rfc_likely)
    rank          = compute_rank(df, final_likely)
    final_labels  = assign_hyp_labels(df, final_likely, rank)

    df["rfc_likely"]   = rfc_likely.values
    df["final_likely"] = final_likely.values
    df["rank"]         = rank.values
    df["final_label"]  = final_labels.values

    report(df, final_labels, final_likely, rank)

    if args.dry_run:
        print("Dry run — no DB writes.")
        out = OUTPUT_DIR / "merge_results.csv"
        df[["customer_id", "cluster_likely", "rfc_likely",
            "final_likely", "rank", "final_label"]].to_csv(out, index=False)
        print(f"✓ Merge preview saved: {out}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "merge_results.csv"
    df[["customer_id", "cluster_likely", "rfc_likely",
        "final_likely", "rank", "final_label"]].to_csv(out, index=False)
    print(f"✓ Merge results     : {out}")

    # Build DB records
    import json as json_mod
    records = []
    for _, row in df.iterrows():
        records.append((
            row["customer_id"],
            row["final_label"],
            float(row.get("rfc_confidence", 0.5) or 0.5),
            build_reasoning(row),
            json_mod.dumps(build_signals(row)),
        ))

    conn = get_conn(args)
    write_to_db(conn, records)
    conn.close()


if __name__ == "__main__":
    main()
