#!/usr/bin/env python3
"""
ml/ml_cluster.py — KMeans++ clustering with VS interpretation logic.

Reads the prepared feature matrix from ml_feature_prep.py output,
runs KMeans++ with the best_k from the silhouette grid, then
applies the VS cluster interpretation logic to produce:
  - cluster_likely: 1 (Likely) or 0 (Unlikely) per customer
  - cluster_label:  one of the 4 recommendation labels

VS Cluster Interpretation Logic (from PredictAlly V1.0 doc, Step 6):
  For each cluster, compute the mean of 5 anchor features.
  If a cluster is SUPERIOR in >= 3 of the 5 anchors → tag as Likely (1)
  Else → Unlikely (0)

  Anchor features (mapped to merchant-abc):
    LTV_MEAN              → total_revenue
    NUM_OF_ORDERS         → total_orders
    FREQUENCY             → frequency_months  (lower = better, inverted)
    AVG_TIME_BETWEEN      → frequency_months  (same column)
    CAMPAIGN_INFLUENCE    → campaign_influence_rate  (NEW, replaces SPEND)

4-Label assignment from cluster:
  Likely   + high campaign_influence_rate  → send_campaign
  Likely   + low  campaign_influence_rate  → no_campaign_needed
  Unlikely + very recent (recency < 7d)   → dont_send
  Unlikely + otherwise                    → no_campaign_impact

Usage
    python ml/ml_cluster.py              # run clustering
    python ml/ml_cluster.py --dry-run    # report, no DB writes
    python ml/ml_cluster.py --k 5        # override k
    python ml/ml_cluster.py --port 5433
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Anchor features for cluster interpretation (VS Step 6)
# (column, higher_is_better)
ANCHOR_FEATURES = [
    ("total_revenue",          True),
    ("total_orders",           True),
    ("campaign_influence_rate", True),
    ("frequency_months",       False),  # lower frequency gap = better
    ("recency_days",           False),  # lower recency = more recent = better
]

# Thresholds for 4-label assignment
INFLUENCE_RATE_THRESHOLD = 0.30   # above = campaign driven
RECENCY_DONT_SEND_DAYS   = 14     # purchased very recently


def load_features() -> tuple[pd.DataFrame, dict]:
    features_path = OUTPUT_DIR / "features_prepared.csv"
    meta_path     = OUTPUT_DIR / "feature_meta.json"
    if not features_path.exists():
        sys.exit("✗ features_prepared.csv not found. Run ml_feature_prep.py first.")
    df   = pd.read_csv(features_path)
    meta = json.load(open(meta_path))
    print(f"  Loaded {len(df):,} customers, {meta['n_features']} features")
    return df, meta


def load_raw_features() -> pd.DataFrame:
    """Raw (unnormalised) features needed for label threshold decisions."""
    raw_path = OUTPUT_DIR / "features_raw.csv"
    if not raw_path.exists():
        sys.exit("✗ features_raw.csv not found. Run ml_feature_prep.py first.")
    return pd.read_csv(raw_path)


# ══════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════
def run_kmeans(X: np.ndarray, k: int) -> tuple:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    inertia = km.inertia_
    score   = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    return labels, centers, inertia, score


# ══════════════════════════════════════════════════════════════════════════
# VS Cluster Interpretation (Step 6)
# ══════════════════════════════════════════════════════════════════════════
def interpret_clusters(df_norm: pd.DataFrame, labels: np.ndarray,
                       feature_cols: list) -> dict:
    """
    For each cluster, compute mean of 5 anchor features.
    Compare clusters pairwise (all vs all).
    A cluster is 'superior' in a feature if its mean is better
    (higher for positive features, lower for inverse features).
    Tag cluster as Likely (1) if superior in >= 3 of 5 anchors.

    Returns: {cluster_id: {"likely": 0|1, "n_customers": int, "means": dict}}
    """
    unique_clusters = sorted(set(labels))
    cluster_means: dict[int, dict] = {}

    for c in unique_clusters:
        mask = labels == c
        means = {}
        for col, _ in ANCHOR_FEATURES:
            if col in df_norm.columns:
                means[col] = float(df_norm.loc[mask, col].mean())
        cluster_means[c] = {"n": int(mask.sum()), "means": means}

    # For each cluster, count how many anchors it wins against ALL others
    result = {}
    for c in unique_clusters:
        wins = 0
        for col, higher_is_better in ANCHOR_FEATURES:
            if col not in df_norm.columns:
                continue
            my_mean = cluster_means[c]["means"].get(col, 0)
            others  = [cluster_means[o]["means"].get(col, 0)
                       for o in unique_clusters if o != c]
            if not others:
                continue
            avg_other = np.mean(others)
            if higher_is_better:
                if my_mean > avg_other:
                    wins += 1
            else:
                if my_mean < avg_other:   # lower recency/frequency = better
                    wins += 1
        # VS rule: >= 3 of 5 anchor features superior → Likely
        likely = 1 if wins >= 3 else 0
        result[c] = {
            "likely":      likely,
            "anchor_wins": wins,
            "n_customers": cluster_means[c]["n"],
            "means":       cluster_means[c]["means"],
        }

    return result


# ══════════════════════════════════════════════════════════════════════════
# 4-Label assignment
# ══════════════════════════════════════════════════════════════════════════
def assign_labels(df_raw: pd.DataFrame, labels: np.ndarray,
                  cluster_info: dict) -> pd.Series:
    """
    Map KMeans cluster + anchor feature values → 4 recommendation labels.

    Logic:
      Likely   + campaign_influence_rate >= threshold → send_campaign
      Likely   + campaign_influence_rate <  threshold → no_campaign_needed
      Unlikely + recency_days < RECENCY_DONT_SEND    → dont_send
      Unlikely + otherwise                           → no_campaign_impact
    """
    result = []
    inf_col     = "campaign_influence_rate"
    recency_col = "recency_days"

    for i, cluster in enumerate(labels):
        likely    = cluster_info[cluster]["likely"]
        inf_rate  = float(df_raw.iloc[i].get(inf_col, 0) or 0)
        recency   = float(df_raw.iloc[i].get(recency_col, 999) or 999)

        if likely == 1:
            if inf_rate >= INFLUENCE_RATE_THRESHOLD:
                label = "send_campaign"
            else:
                label = "no_campaign_needed"
        else:
            if recency < RECENCY_DONT_SEND_DAYS:
                label = "dont_send"
            else:
                label = "no_campaign_impact"
        result.append(label)

    return pd.Series(result, name="cluster_label")


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(k: int, score: float, inertia: float,
           cluster_info: dict, label_counts: dict) -> None:
    print(f"\n{'═' * 60}")
    print(f"  KMeans++ Clustering Report  (k={k})")
    print(f"{'═' * 60}")
    print(f"  Silhouette score : {score:.4f}")
    print(f"  Inertia          : {inertia:,.1f}")
    print(f"\n  {'Cluster':>8} {'Customers':>10} {'Likely':>7} "
          f"{'Wins':>6} {'Means: LTV | orders | influence'}") 
    print(f"  {'─' * 58}")
    for c, info in sorted(cluster_info.items()):
        ltv = info["means"].get("total_revenue", 0)
        orders = info["means"].get("total_orders", 0)
        inf = info["means"].get("campaign_influence_rate", 0)
        print(f"  {c:>8} {info['n_customers']:>10,} "
              f"{'✓ Likely' if info['likely'] else '✗ Unlikely':>9} "
              f"{info['anchor_wins']:>4}/5  "
              f"  {ltv:.3f} | {orders:.3f} | {inf:.3f}")

    print(f"\n  4-Label distribution:")
    total = sum(label_counts.values())
    label_order = ["send_campaign", "dont_send",
                   "no_campaign_needed", "no_campaign_impact"]
    for label in label_order:
        n   = label_counts.get(label, 0)
        pct = n * 100 / total if total else 0
        print(f"    {label:<26} {n:>5}  ({pct:.1f}%)")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="KMeans++ clustering with VS interpretation logic.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--k",    type=int, help="Override cluster count")
    ap.add_argument("--host", help="DB host")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    print("Loading prepared features …")
    df_norm, meta = load_features()
    df_raw        = load_raw_features()

    feature_cols = [f["col"] for f in meta["features"]]
    X = df_norm[feature_cols].values.astype(np.float64)

    # Use k from silhouette grid unless overridden
    k = args.k or meta["best_k"]
    print(f"Running KMeans++ with k={k} …")
    labels, centers, inertia, score = run_kmeans(X, k)

    print("Interpreting clusters (VS 3-of-5 anchor vote) …")
    cluster_info = interpret_clusters(df_norm, labels, feature_cols)

    print("Assigning 4-labels …")
    cluster_labels = assign_labels(df_raw, labels, cluster_info)
    cluster_likely = pd.Series(
        [cluster_info[c]["likely"] for c in labels],
        name="cluster_likely"
    )

    label_counts = cluster_labels.value_counts().to_dict()
    report(k, score, inertia, cluster_info, label_counts)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cluster results
    result_df = df_norm[["customer_id", "customer_type",
                          "attribution_segment"]].copy()
    result_df["cluster"]        = labels
    result_df["cluster_likely"] = cluster_likely.values
    result_df["cluster_label"]  = cluster_labels.values

    out_path = OUTPUT_DIR / "cluster_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"✓ Cluster results   : {out_path}")

    # Save cluster metadata
    cluster_meta = {
        "k": k,
        "silhouette_score": round(score, 6),
        "inertia": round(inertia, 2),
        "label_counts": label_counts,
        "cluster_info": {
            str(c): {
                "likely": info["likely"],
                "anchor_wins": info["anchor_wins"],
                "n_customers": info["n_customers"],
            }
            for c, info in cluster_info.items()
        },
        "thresholds": {
            "influence_rate": INFLUENCE_RATE_THRESHOLD,
            "recency_dont_send_days": RECENCY_DONT_SEND_DAYS,
        }
    }
    out_meta = OUTPUT_DIR / "cluster_meta.json"
    with open(out_meta, "w") as f:
        json.dump(cluster_meta, f, indent=2)
    print(f"✓ Cluster metadata  : {out_meta}")


if __name__ == "__main__":
    main()
