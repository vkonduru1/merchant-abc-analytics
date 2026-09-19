#!/usr/bin/env python3
"""
ml/pipeline_ml.py — Campaign Send Likelihood: ML Pipeline (v2)

PURPOSE
-------
Predict campaign send likelihood for each customer using a statistical
ML approach. Reads the canonical feature set from
pipeline_v2/02_feature_canonical.json — the single source of truth
shared with agents/pipeline_claude.py.

This pipeline is one half of a parallel comparison:
  ml/pipeline_ml.py        ← this file (statistical predictions)
  agents/pipeline_claude.py ← rule-based predictions (same features)

Both pipelines are evaluated against the same ground truth labels
derived from actual purchase outcomes in campaign_attribution,
using the same temporal holdout sets defined in
pipeline_v2/backtest_compare.py.

METHODOLOGY
-----------
Step 1 — Load & preprocess
  Read customer_derivatives for canonical features only.
  Median imputation for nulls.
  VS normalisation: (x - mean) / (max - min) per feature.

Step 2 — KMeans++ clustering
  k=2 (binary: Likely / Unlikely) — confirmed by silhouette analysis.
  Cluster interpretation: VS 3-of-5 anchor feature majority vote.
  Anchor features: recency_days, campaign_influence_rate,
                   total_campaigns_received, frequency_months,
                   spend_last_3_months.

Step 3 — RFC supervised model
  Target: KMeans++ cluster labels (proxy for campaign response).
  TimeSeriesSplit CV (5 folds) — respects temporal causality.
  Outputs: 4-class probability distribution per customer.

Step 4 — OR merge + HYP segmentation (VS methodology)
  final_likely = cluster_likely OR rfc_likely
  HYP_TP (top 30% of likely)  → send_campaign
  HYP_FP (rest of likely)     → no_campaign_needed
  HYP_FN (top 20% of unlikely)→ dont_send
  HYP_TN (rest of unlikely)   → no_campaign_impact

Step 5 — Write to customer_recommendations
  model_version = 'ml-v2-pipeline'
  Upserts — safe to re-run.

GROUND TRUTH (for backtest_compare.py)
---------------------------------------
NOT computed here. Ground truth labels are derived and evaluated
centrally in pipeline_v2/backtest_compare.py to ensure both pipelines
are measured against identical holdout sets.

Usage
    python ml/pipeline_ml.py                # full run
    python ml/pipeline_ml.py --dry-run      # report only, no DB writes
    python ml/pipeline_ml.py --port 5433    # override DB port
    python ml/pipeline_ml.py --skip-cv      # skip CV (faster, for iteration)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent
CANONICAL_PATH = ROOT_DIR / "pipeline_v2" / "02_feature_canonical.json"
OUTPUT_DIR   = SCRIPT_DIR / "output"

sys.path.insert(0, str(ROOT_DIR / "data_generator"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

# HYP segmentation thresholds (VS methodology)
HYP_TP_PERCENTILE = 70   # top 30% of likely → send_campaign
HYP_FN_PERCENTILE = 80   # top 20% of unlikely → dont_send
INACTIVITY_DAYS   = 180
RANK_SCALE        = 1_000_000

# Anchor features for cluster interpretation (VS 3-of-5 majority vote)
# (feature, higher_is_better)
ANCHOR_FEATURES = [
    ("recency_days",              False),  # lower = more recent = better
    ("campaign_influence_rate",   True),
    ("total_campaigns_received",  True),
    ("frequency_months",          False),  # lower = more frequent = better
    ("spend_last_3_months",       True),
]

UPSERT_SQL = """
INSERT INTO customer_recommendations (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-v2-pipeline')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""


# ══════════════════════════════════════════════════════════════════════════
# Helpers
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


def load_canonical() -> list[str]:
    if not CANONICAL_PATH.exists():
        sys.exit(f"✗ {CANONICAL_PATH} not found. "
                 "Run pipeline_v2/01_feature_analysis.py first.")
    meta = json.load(open(CANONICAL_PATH))
    cols = [f["column"] for f in meta["features"]]
    print(f"  Canonical features ({len(cols)}): {', '.join(cols)}")
    return cols


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Load & preprocess
# ══════════════════════════════════════════════════════════════════════════
def load_data(conn, feature_cols: list) -> pd.DataFrame:
    cols_sql = ", ".join(
        feature_cols +
        ["customer_id", "attribution_segment", "customer_type",
         "total_orders", "total_revenue", "recency_days"]
    )
    # Deduplicate — some columns may overlap with feature_cols
    seen = set()
    unique_cols = []
    for c in cols_sql.split(", "):
        c = c.strip()
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    query = f"""
        SELECT DISTINCT ON (customer_id)
            {', '.join(unique_cols)}
        FROM customer_derivatives
        ORDER BY customer_id, run_id DESC
    """
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df):,} customers")
    return df


def preprocess(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Median imputation + VS normalisation: (x - mean) / (max - min)"""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        # Imputation
        fill = df[col].max() if col == "recency_days" else df[col].median()
        df[col] = df[col].fillna(fill)
        # VS normalisation
        col_range = df[col].max() - df[col].min()
        if col_range > 0:
            df[col] = (df[col] - df[col].mean()) / col_range
        else:
            df[col] = 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — KMeans++ clustering
# ══════════════════════════════════════════════════════════════════════════
def cluster(X: np.ndarray, k: int = 2) -> tuple:
    km = KMeans(n_clusters=k, init="k-means++",
                n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    return labels, km.cluster_centers_, score


def interpret_clusters(df_norm: pd.DataFrame,
                       labels: np.ndarray,
                       feature_cols: list) -> dict:
    """VS 3-of-5 anchor majority vote → Likely (1) or Unlikely (0)."""
    unique = sorted(set(labels))
    means  = {
        c: {col: float(df_norm.loc[labels == c, col].mean())
            for col, _ in ANCHOR_FEATURES
            if col in df_norm.columns}
        for c in unique
    }

    result = {}
    for c in unique:
        wins = 0
        for col, higher_is_better in ANCHOR_FEATURES:
            if col not in df_norm.columns:
                continue
            my_mean    = means[c].get(col, 0)
            other_mean = np.mean([means[o].get(col, 0)
                                  for o in unique if o != c])
            if higher_is_better:
                wins += int(my_mean > other_mean)
            else:
                wins += int(my_mean < other_mean)
        result[c] = {
            "likely":      1 if wins >= 3 else 0,
            "anchor_wins": wins,
            "n":           int((labels == c).sum()),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — RFC supervised model
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth_labels(conn, customer_ids: pd.Series) -> pd.Series:
    """
    Derive 4-class ground truth labels from campaign_attribution.
    These labels reflect actual purchase outcomes — not cluster geometry.
    Using 4-class labels trains RFC to predict the business outcome directly
    rather than learning to replicate binary cluster membership.

    Label derivation:
      dont_send          → recency < 14 days (just purchased)
      send_campaign      → ≥50% of campaign-era orders were influenced
      no_campaign_needed → has orders but 0 were campaign-influenced
      no_campaign_impact → no campaign-era orders (prospect or ghost)
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
        SELECT customer_id,
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
    df_gt = pd.read_sql(query, conn)
    df_gt["customer_id"] = df_gt["customer_id"].astype(str)
    merged = pd.Series(customer_ids.astype(str).values,
                       name="customer_id").to_frame()
    merged = merged.merge(df_gt, on="customer_id", how="left")
    merged["ground_truth_label"] = merged["ground_truth_label"].fillna(
        "no_campaign_impact"
    )
    return merged["ground_truth_label"]


def train_rfc(X: np.ndarray, y: np.ndarray,
              skip_cv: bool = False) -> tuple:
    rfc = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    fold_scores = []
    if not skip_cv:
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            if len(set(y[tr])) < 2:
                continue
            rfc.fit(X[tr], y[tr])
            acc = accuracy_score(y[te], rfc.predict(X[te]))
            fold_scores.append(acc)
            print(f"    Fold {fold}: train={len(tr):,}  "
                  f"test={len(te):,}  acc={acc:.3f}")

    # Final model on full dataset
    rfc.fit(X, y)
    probs  = rfc.predict_proba(X)
    preds  = rfc.predict(X)
    return rfc, probs, preds, fold_scores


# ══════════════════════════════════════════════════════════════════════════
# Step 4 — OR merge + HYP segmentation
# ══════════════════════════════════════════════════════════════════════════
def merge_and_segment(df_raw: pd.DataFrame,
                      cluster_info: dict,
                      km_labels: np.ndarray,
                      rfc: RandomForestClassifier,
                      rfc_probs: np.ndarray) -> pd.DataFrame:
    classes = list(rfc.classes_)

    # Align RFC probs to LABEL_ORDER
    prob_df = pd.DataFrame(index=df_raw.index)
    for label in LABEL_ORDER:
        if label in classes:
            prob_df[f"{label}_prob"] = rfc_probs[:, classes.index(label)]
        else:
            prob_df[f"{label}_prob"] = 0.0

    # Likelihoods
    cluster_likely = pd.Series(
        [cluster_info[c]["likely"] for c in km_labels], name="cluster_likely"
    )
    rfc_likely = (
        (prob_df["send_campaign_prob"] +
         prob_df["no_campaign_needed_prob"]) >= 0.5
    ).astype(int)
    final_likely = ((cluster_likely == 1) | (rfc_likely == 1)).astype(int)

    # Rank formula (VS Step 13)
    # total_revenue fetched as rank-only field — not in canonical feature set
    # but essential for differentiating customers within Likely/Unlikely pools
    ltv    = df_raw["total_revenue"].fillna(0).reset_index(drop=True)
    conf   = prob_df[
        [f"{l}_prob" for l in LABEL_ORDER]
    ].max(axis=1).reset_index(drop=True)
    recency_reset = df_raw["recency_days"].fillna(999).reset_index(drop=True)
    active = (recency_reset <= INACTIVITY_DAYS).map({True: 1.0, False: 0.5})
    rank   = ltv * conf * RANK_SCALE
    rank   = np.where(
        final_likely.reset_index(drop=True) == 1,
        rank, rank * active
    )
    rank   = pd.Series(rank, name="rank")

    # HYP segmentation
    labels = pd.Series(["no_campaign_impact"] * len(df_raw),
                       name="final_label", index=df_raw.index)

    likely_mask   = final_likely == 1
    unlikely_mask = ~likely_mask

    if likely_mask.sum() > 0:
        tp_thresh = rank[likely_mask].quantile(HYP_TP_PERCENTILE / 100)
        labels[likely_mask & (rank >= tp_thresh)] = "send_campaign"
        labels[likely_mask & (rank <  tp_thresh)] = "no_campaign_needed"

    if unlikely_mask.sum() > 0:
        fn_thresh = rank[unlikely_mask].quantile(HYP_FN_PERCENTILE / 100)
        labels[unlikely_mask & (rank >= fn_thresh)] = "dont_send"
        labels[unlikely_mask & (rank <  fn_thresh)] = "no_campaign_impact"

    result = df_raw[["customer_id"]].copy()
    result["cluster_likely"]  = cluster_likely.values
    result["rfc_likely"]      = rfc_likely.values
    result["final_likely"]    = final_likely.values
    result["rank"]            = rank.values
    result["final_label"]     = labels.values
    result["rfc_confidence"]  = conf.values.round(4)
    for label in LABEL_ORDER:
        result[f"{label}_prob"] = prob_df[f"{label}_prob"].values.round(4)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Reasoning builder
# ══════════════════════════════════════════════════════════════════════════
def build_reasoning(row: pd.Series, df_raw: pd.DataFrame) -> tuple[str, list]:
    label  = row["final_label"]
    conf   = float(row.get("rfc_confidence", 0.5) or 0.5)
    cl     = int(row.get("cluster_likely", 0))
    rl     = int(row.get("rfc_likely", 0))
    cust   = df_raw.loc[row.name] if row.name in df_raw.index else {}

    inf_rate = float(cust.get("campaign_influence_rate", 0) or 0) \
               if hasattr(cust, "get") else 0.0
    recency  = float(cust.get("recency_days", 999) or 999) \
               if hasattr(cust, "get") else 999.0
    ltv      = float(cust.get("total_revenue", 0) or 0) \
               if hasattr(cust, "get") else 0.0
    avg_lag  = float(cust.get("avg_days_campaign_to_purchase", 0) or 0) \
               if hasattr(cust, "get") else 0.0

    agree = "KMeans++ and RFC both agree:" if cl == rl else (
        "KMeans++ signals Likely;" if cl else "RFC signals Likely;")

    if label == "send_campaign":
        reasoning = (
            f"ML pipeline (v2) classifies as HYP_TP — top-ranked Likely. "
            f"{agree} campaign_influence_rate={inf_rate:.0%}, "
            f"recency={recency:.0f} days, LTV=${ltv:.2f}. "
            f"Avg response lag {avg_lag:.1f} days after campaign. "
            f"RFC confidence {conf:.0%}."
        )
        signals = [
            f"campaign_influence_rate: {inf_rate:.0%}",
            f"recency: {recency:.0f} days",
            f"avg campaign-to-purchase lag: {avg_lag:.1f} days",
            f"RFC confidence: {conf:.0%}",
        ]
    elif label == "no_campaign_needed":
        reasoning = (
            f"ML pipeline (v2) classifies as HYP_FP — Likely but lower rank. "
            f"Campaign influence rate {inf_rate:.0%} suggests organic tendencies. "
            f"LTV=${ltv:.2f}. Send NPIs and loyalty offers only."
        )
        signals = [
            f"campaign_influence_rate: {inf_rate:.0%}",
            f"LTV: ${ltv:.2f}",
            f"Likely pool but below top-30% rank threshold",
        ]
    elif label == "dont_send":
        reasoning = (
            f"ML pipeline (v2) classifies as HYP_FN — top-ranked Unlikely. "
            f"Monitor closely — recency={recency:.0f} days. "
            f"RFC confidence {conf:.0%}. Re-evaluate in 2 weeks."
        )
        signals = [
            f"recency: {recency:.0f} days",
            f"Unlikely pool but high-rank — potential converter",
            f"RFC confidence: {conf:.0%}",
        ]
    else:
        reasoning = (
            f"ML pipeline (v2) classifies as HYP_TN — Unlikely, lower rank. "
            f"campaign_influence_rate={inf_rate:.0%}. "
            f"Suppress or route to 90-day re-engagement flow."
        )
        signals = [
            f"campaign_influence_rate: {inf_rate:.0%}",
            f"recency: {recency:.0f} days",
            f"Bottom of Unlikely pool — low conversion probability",
        ]

    return reasoning, signals


# ══════════════════════════════════════════════════════════════════════════
# Step 5 — Write to DB
# ══════════════════════════════════════════════════════════════════════════
def write_to_db(conn, result_df: pd.DataFrame,
                df_raw: pd.DataFrame) -> None:
    from psycopg2.extras import execute_batch
    records = []
    for _, row in result_df.iterrows():
        reasoning, signals = build_reasoning(row, df_raw)
        records.append((
            row["customer_id"],
            row["final_label"],
            float(row["rfc_confidence"]),
            reasoning,
            json.dumps(signals),
        ))
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, UPSERT_SQL, records, page_size=200)
    print(f"✓ Upserted {len(records):,} rows → customer_recommendations")


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(result_df: pd.DataFrame, fold_scores: list,
           cluster_info: dict, sil_score: float,
           rfc: RandomForestClassifier,
           feature_cols: list) -> None:
    from collections import Counter
    counts = Counter(result_df["final_label"])
    total  = len(result_df)
    likely = int(result_df["final_likely"].sum())

    print(f"\n{'═' * 64}")
    print(f"  ML Pipeline v2 — Results")
    print(f"{'═' * 64}")
    print(f"\n  Clustering (KMeans++, k=2)")
    print(f"    Silhouette score : {sil_score:.4f}")
    for c, info in sorted(cluster_info.items()):
        print(f"    Cluster {c}: {info['n']:>4} customers  "
              f"{'✓ Likely' if info['likely'] else '✗ Unlikely'}  "
              f"({info['anchor_wins']}/5 anchors)")

    if fold_scores:
        print(f"\n  RFC TimeSeriesSplit CV ({len(fold_scores)} folds)")
        for i, s in enumerate(fold_scores, 1):
            print(f"    Fold {i}: {s:.3f}")
        print(f"    Mean: {np.mean(fold_scores):.3f}  "
              f"Std: {np.std(fold_scores):.3f}")

    print(f"\n  RFC Feature Importance (canonical set):")
    imps = sorted(zip(feature_cols, rfc.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    for col, imp in imps:
        bar = "█" * int(imp * 150)
        print(f"    {col:<44} {imp:.4f}  {bar}")

    print(f"\n  OR Merge")
    print(f"    Likely pool  : {likely:,}  ({likely*100/total:.1f}%)")
    print(f"    Unlikely pool: {total-likely:,}  "
          f"({(total-likely)*100/total:.1f}%)")

    print(f"\n  Final label distribution:")
    for label in LABEL_ORDER:
        n   = counts.get(label, 0)
        pct = n * 100 / total
        bar = "█" * int(pct / 2)
        print(f"    {label:<26} {n:>5}  ({pct:.1f}%)  {bar}")
    print(f"{'═' * 64}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ML pipeline v2 — campaign send likelihood prediction.")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Report only, no DB writes")
    ap.add_argument("--skip-cv",  action="store_true",
                    help="Skip TimeSeriesSplit CV (faster iteration)")
    ap.add_argument("--host",     help="DB host (default: localhost)")
    ap.add_argument("--port",     help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 64}")
    print(f"  ML Pipeline v2 — Campaign Send Likelihood")
    print(f"{'═' * 64}")

    # Load canonical feature set
    print("\nLoading canonical feature set …")
    feature_cols = load_canonical()

    conn = get_conn(args)
    print(f"✓ Connected\n")

    # Step 1 — Load & preprocess
    print("Step 1 — Load & preprocess …")
    df_raw  = load_data(conn, feature_cols)
    df_norm = preprocess(df_raw.copy(), feature_cols)
    X       = df_norm[feature_cols].values.astype(np.float64)

    # Step 2 — KMeans++
    print("\nStep 2 — KMeans++ clustering (k=2) …")
    km_labels, centers, sil_score = cluster(X, k=2)
    cluster_info = interpret_clusters(df_norm, km_labels, feature_cols)
    for c, info in sorted(cluster_info.items()):
        print(f"  Cluster {c}: {info['n']:>4} customers  "
              f"{'Likely' if info['likely'] else 'Unlikely'}  "
              f"({info['anchor_wins']}/5 anchors)  "
              f"silhouette={sil_score:.4f}")

    # Step 3 — RFC
    print(f"\nStep 3 — RFC training "
          f"({'with' if not args.skip_cv else 'without'} CV) …")
    # Use 4-class ground truth labels as RFC target
    # These reflect actual purchase outcomes from campaign_attribution
    # rather than binary cluster membership — produces meaningful
    # 4-class probability distribution per customer
    print("  Loading 4-class ground truth labels from campaign_attribution …")
    ground_truth_labels = load_ground_truth_labels(conn, df_raw["customer_id"])
    from collections import Counter
    gt_counts = Counter(ground_truth_labels)
    for label, n in gt_counts.most_common():
        print(f"    {label:<26} {n:>4}")

    rfc, rfc_probs, rfc_preds, fold_scores = train_rfc(
        X, ground_truth_labels.values, skip_cv=args.skip_cv
    )

    # Step 4 — OR merge + HYP segmentation
    print("\nStep 4 — OR merge + HYP segmentation …")
    result_df = merge_and_segment(
        df_raw, cluster_info, km_labels, rfc, rfc_probs
    )

    # Report
    report(result_df, fold_scores, cluster_info,
           sil_score, rfc, feature_cols)

    # Save output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "ml_v2_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"✓ Results saved → {out_path}")

    if args.dry_run:
        print("Dry run — no DB writes.")
        conn.close()
        return

    # Step 5 — Write to DB
    print("\nStep 5 — Writing to customer_recommendations …")
    write_to_db(conn, result_df, df_raw)
    conn.close()
    print("\nDone. Run pipeline_v2/backtest_compare.py to evaluate accuracy.")


if __name__ == "__main__":
    main()
