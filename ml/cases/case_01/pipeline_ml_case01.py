#!/usr/bin/env python3
"""
ml/cases/case_01/pipeline_ml_case01.py

CASE 01 — RFC Argmax Direct (replacing HYP segmentation)
─────────────────────────────────────────────────────────
PROBLEM IN v2.0 BASELINE:
  HYP segmentation inflated no_campaign_needed to 56.5%.
  The Likely pool (80.8% of customers via OR merge) was too large.
  The 70th-percentile rank split then pushed most Likely customers
  into HYP_FP → no_campaign_needed, regardless of their actual
  RFC probability distribution.

  Result: ML accuracy 31.3% vs Claude 66.5%

CHANGE IN CASE 01:
  Remove the OR merge + HYP segmentation entirely.
  Replace with direct RFC argmax:
    final_label = argmax(RFC 4-class probability distribution)

  KMeans++ clustering is retained as a diagnostic signal
  (reported in output) but does NOT influence label assignment.
  The RFC trained on 4-class ground truth labels makes the call.

HYPOTHESIS:
  RFC trained directly on ground truth labels already encodes
  the optimal decision boundary. The HYP post-processing was
  overriding that with a rank formula that introduced noise.
  Removing it should push label distribution closer to ground truth:
    send_campaign      GT=49.6%  v2.0=24.4%  case01=?
    no_campaign_needed GT=29.8%  v2.0=56.5%  case01=?
    no_campaign_impact GT=14.3%  v2.0=15.3%  case01=?
    dont_send          GT= 6.3%  v2.0= 3.8%  case01=?

WHAT STAYS THE SAME:
  - 10 canonical features from 02_feature_canonical.json
  - VS normalisation: (x - mean) / (max - min)
  - 4-class ground truth labels from campaign_attribution
  - TimeSeriesSplit CV (5 folds)
  - Writes to customer_recommendations_ml (upsert)

Usage
    python ml/cases/case_01/pipeline_ml_case01.py
    python ml/cases/case_01/pipeline_ml_case01.py --dry-run
    python ml/cases/case_01/pipeline_ml_case01.py --port 5433
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
ROOT_DIR       = SCRIPT_DIR.parent.parent.parent   # merchant-abc-analytics/
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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, silhouette_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

RFC_PARAMS = {
    "n_estimators":     200,
    "min_samples_leaf":   5,
    "class_weight":  "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

UPSERT_SQL = """
INSERT INTO customer_recommendations_ml (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'ml-case01-rfc-argmax')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""


# ── DB ────────────────────────────────────────────────────────────────────
def get_conn(args):
    import psycopg2
    return psycopg2.connect(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB",       "merchant_abc"),
        user=os.getenv("POSTGRES_USER",       "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD","abc_dev_password"),
    )


def load_canonical() -> list[str]:
    if not CANONICAL_PATH.exists():
        sys.exit(f"✗ {CANONICAL_PATH} not found. "
                 "Run pipeline_v2/01_feature_analysis.py first.")
    meta = json.load(open(CANONICAL_PATH))
    cols = [f["column"] for f in meta["features"]]
    print(f"  {len(cols)} canonical features loaded")
    return cols


# ── Data ──────────────────────────────────────────────────────────────────
def load_data(conn, feature_cols: list) -> pd.DataFrame:
    extra    = ["customer_id", "customer_type", "attribution_segment",
                "total_orders", "total_revenue", "recency_days"]
    all_cols = list(dict.fromkeys(feature_cols + extra))
    query    = f"""
        SELECT DISTINCT ON (customer_id)
            {', '.join(all_cols)}
        FROM customer_derivatives
        ORDER BY customer_id, run_id DESC
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} customers loaded")
    return df


def load_ground_truth(conn, customer_ids: pd.Series) -> pd.Series:
    """4-class ground truth from campaign_attribution — same as pipeline_ml.py."""
    query = """
        WITH gt AS (
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
                WHEN recency_days < 14                              THEN 'dont_send'
                WHEN total_attr_orders = 0                          THEN 'no_campaign_impact'
                WHEN camp_orders > 0
                    AND camp_orders >= total_attr_orders * 0.5      THEN 'send_campaign'
                WHEN camp_orders = 0 AND total_attr_orders > 0      THEN 'no_campaign_needed'
                ELSE 'no_campaign_impact'
            END AS ground_truth_label
        FROM gt
    """
    df_gt = pd.read_sql(query, conn)
    df_gt["customer_id"] = df_gt["customer_id"].astype(str)
    merged = pd.Series(
        customer_ids.astype(str).values, name="customer_id"
    ).to_frame()
    merged = merged.merge(df_gt, on="customer_id", how="left")
    merged["ground_truth_label"] = merged["ground_truth_label"].fillna(
        "no_campaign_impact"
    )
    return merged["ground_truth_label"]


def preprocess(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """VS normalisation: (x - mean) / (max - min)"""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        fill = df[col].max() if col == "recency_days" else df[col].median()
        df[col] = df[col].fillna(fill)
        r = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].mean()) / r if r > 0 else 0.0
    return df


# ── KMeans++ (diagnostic only) ────────────────────────────────────────────
def run_kmeans_diagnostic(X: np.ndarray) -> tuple:
    km     = KMeans(n_clusters=2, init="k-means++",
                    n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    sizes  = Counter(labels)
    return labels, score, sizes


# ── RFC — TimeSeriesSplit CV + final model ────────────────────────────────
def train_rfc(X: np.ndarray, y: np.ndarray,
              skip_cv: bool = False) -> tuple:
    rfc         = RandomForestClassifier(**RFC_PARAMS)
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

    rfc.fit(X, y)
    return rfc, fold_scores


# ── CASE 01 CORE CHANGE: RFC argmax → label directly ─────────────────────
def assign_labels_argmax(rfc: RandomForestClassifier,
                         X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign final label as argmax of RFC 4-class probability distribution.
    No HYP segmentation. No OR merge. RFC decides directly.

    This replaces the entire merge_and_segment() function from v2.0.
    """
    probs  = rfc.predict_proba(X)
    labels = rfc.predict(X)
    confs  = probs.max(axis=1)
    return labels, confs, probs


# ── Reasoning builder ─────────────────────────────────────────────────────
def build_reasoning(label: str, conf: float,
                    row: pd.Series) -> tuple[str, list]:
    inf  = float(row.get("campaign_influence_rate", 0) or 0)
    rec  = float(row.get("recency_days", 999) or 999)
    ltv  = float(row.get("total_revenue", 0) or 0)
    lag  = float(row.get("avg_days_campaign_to_purchase", 0) or 0)
    std  = float(row.get("stddev_days_campaign_to_purchase", 0) or 0)

    if label == "send_campaign":
        reasoning = (
            f"RFC (case01-argmax) assigns highest probability to send_campaign. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f} days, "
            f"LTV=${ltv:.2f}. "
            f"Avg response lag {lag:.1f} days (stddev={std:.1f}). "
            f"RFC confidence {conf:.0%}."
        )
        signals = [
            f"RFC label: send_campaign (argmax, conf={conf:.0%})",
            f"campaign_influence_rate: {inf:.0%}",
            f"recency: {rec:.0f} days",
            f"avg lag: {lag:.1f} days  stddev: {std:.1f}",
        ]
    elif label == "no_campaign_needed":
        reasoning = (
            f"RFC (case01-argmax) assigns highest probability to "
            f"no_campaign_needed. "
            f"campaign_influence_rate={inf:.0%} — organic buyer pattern. "
            f"LTV=${ltv:.2f}. Send NPIs and loyalty offers only. "
            f"RFC confidence {conf:.0%}."
        )
        signals = [
            f"RFC label: no_campaign_needed (argmax, conf={conf:.0%})",
            f"campaign_influence_rate: {inf:.0%} — organic",
            f"LTV: ${ltv:.2f}",
        ]
    elif label == "dont_send":
        reasoning = (
            f"RFC (case01-argmax) assigns highest probability to dont_send. "
            f"recency={rec:.0f} days — recent purchase. "
            f"RFC confidence {conf:.0%}. Re-evaluate in 2 weeks."
        )
        signals = [
            f"RFC label: dont_send (argmax, conf={conf:.0%})",
            f"recency: {rec:.0f} days — post-purchase window",
        ]
    else:
        reasoning = (
            f"RFC (case01-argmax) assigns highest probability to "
            f"no_campaign_impact. "
            f"campaign_influence_rate={inf:.0%}, recency={rec:.0f} days. "
            f"Suppress or route to 90-day re-engagement flow. "
            f"RFC confidence {conf:.0%}."
        )
        signals = [
            f"RFC label: no_campaign_impact (argmax, conf={conf:.0%})",
            f"campaign_influence_rate: {inf:.0%}",
            f"recency: {rec:.0f} days",
        ]
    return reasoning, signals


# ── Report ────────────────────────────────────────────────────────────────
def report(y_gt: np.ndarray, y_pred: np.ndarray,
           confs: np.ndarray, fold_scores: list,
           feature_cols: list, rfc: RandomForestClassifier,
           km_score: float, km_sizes: dict) -> dict:

    acc    = accuracy_score(y_gt, y_pred)
    counts = Counter(y_pred)
    total  = len(y_pred)
    gt_counts = Counter(y_gt)

    print(f"\n{'═' * 66}")
    print(f"  ML Pipeline — Case 01: RFC Argmax Direct")
    print(f"{'═' * 66}")

    print(f"\n  KMeans++ (diagnostic only, k=2):")
    print(f"    Silhouette: {km_score:.4f}")
    for c, n in sorted(km_sizes.items()):
        print(f"    Cluster {c}: {n:,} customers")

    if fold_scores:
        print(f"\n  RFC TimeSeriesSplit CV ({len(fold_scores)} folds):")
        for i, s in enumerate(fold_scores, 1):
            print(f"    Fold {i}: {s:.3f}")
        print(f"    Mean: {np.mean(fold_scores):.3f}  "
              f"Std: {np.std(fold_scores):.3f}")

    print(f"\n  RFC Feature Importance:")
    imps = sorted(zip(feature_cols, rfc.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    for col, imp in imps:
        bar = "█" * int(imp * 150)
        print(f"    {col:<44} {imp:.4f}  {bar}")

    print(f"\n  Label distribution (pred vs ground truth):")
    print(f"  {'Label':<26} {'Pred':>6} {'Pred%':>6}  "
          f"{'GT':>6} {'GT%':>6}")
    print(f"  {'─' * 56}")
    for label in LABEL_ORDER:
        n_pred = counts.get(label, 0)
        n_gt   = gt_counts.get(label, 0)
        print(f"  {label:<26} {n_pred:>6}  "
              f"{n_pred*100/total:>5.1f}%  "
              f"{n_gt:>6}  {n_gt*100/total:>5.1f}%")

    print(f"\n  Overall accuracy vs ground truth: {acc:.3f}  ({acc:.1%})")
    print(f"  Avg RFC confidence: {confs.mean():.3f}")

    print(f"\n  Classification report:")
    labels_present = [l for l in LABEL_ORDER if l in set(y_gt)]
    print(classification_report(y_gt, y_pred,
          labels=labels_present, zero_division=0))

    print(f"\n  Confusion matrix (rows=GT, cols=pred):")
    cm = confusion_matrix(y_gt, y_pred, labels=labels_present)
    header = "  " + "".join(f"{l[:8]:>10}" for l in labels_present)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels_present[i][:10]:<12}"
              + "".join(f"{v:>10}" for v in row))

    print(f"{'═' * 66}\n")

    return {
        "case":          "case_01",
        "change":        "RFC argmax direct — HYP segmentation removed",
        "accuracy":      round(float(acc), 4),
        "mean_cv_acc":   round(float(np.mean(fold_scores)), 4)
                         if fold_scores else None,
        "avg_confidence": round(float(confs.mean()), 4),
        "label_dist":    {k: int(v) for k, v in counts.items()},
        "gt_dist":       {k: int(v) for k, v in gt_counts.items()},
        "feature_importance": {
            col: round(float(imp), 4)
            for col, imp in imps
        },
    }


# ── Write to DB ───────────────────────────────────────────────────────────
def write_to_db(conn, df_raw: pd.DataFrame,
                labels: np.ndarray, confs: np.ndarray) -> None:
    from psycopg2.extras import execute_batch
    records = []
    for i, (_, row) in enumerate(df_raw.iterrows()):
        reasoning, signals = build_reasoning(
            labels[i], confs[i], row
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
    print(f"✓ Upserted {len(records):,} rows → customer_recommendations_ml")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Case 01 — RFC argmax direct (no HYP segmentation).")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--skip-cv",  action="store_true")
    ap.add_argument("--host",     help="DB host")
    ap.add_argument("--port",     help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 66}")
    print(f"  ML Pipeline — Case 01: RFC Argmax Direct")
    print(f"  Change: remove HYP segmentation, RFC decides label directly")
    print(f"{'═' * 66}\n")

    feature_cols = load_canonical()
    conn         = get_conn(args)
    print(f"✓ Connected\n")

    print("Loading data …")
    df_raw  = load_data(conn, feature_cols)
    df_norm = preprocess(df_raw.copy(), feature_cols)
    X       = df_norm[feature_cols].values.astype(np.float64)

    print("Loading 4-class ground truth labels …")
    y_gt = load_ground_truth(conn, df_raw["customer_id"])
    gt_counts = Counter(y_gt)
    for label in LABEL_ORDER:
        print(f"  {label:<26} {gt_counts.get(label, 0):>4}")

    print(f"\nKMeans++ diagnostic (k=2) …")
    km_labels, km_score, km_sizes = run_kmeans_diagnostic(X)

    print(f"\nRFC training "
          f"({'with' if not args.skip_cv else 'without'} CV) …")
    rfc, fold_scores = train_rfc(X, y_gt.values,
                                 skip_cv=args.skip_cv)

    print(f"\nAssigning labels via RFC argmax …")
    labels, confs, probs = assign_labels_argmax(rfc, X)

    metrics = report(y_gt.values, labels, confs, fold_scores,
                     feature_cols, rfc, km_score, km_sizes)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results_case01.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Results → {results_path}")

    # Save per-customer predictions
    pred_df = df_raw[["customer_id"]].copy()
    pred_df["label"]      = labels
    pred_df["confidence"] = confs.round(4)
    pred_df["gt_label"]   = y_gt.values
    pred_df["correct"]    = pred_df["label"] == pred_df["gt_label"]
    pred_df.to_csv(OUTPUT_DIR / "predictions_case01.csv", index=False)
    print(f"✓ Predictions → {OUTPUT_DIR}/predictions_case01.csv")

    if args.dry_run:
        print("\nDry run — no DB writes.")
        conn.close()
        return

    write_to_db(conn, df_raw, labels, confs)
    conn.close()
    print("\nDone. Compare with pipeline_v2/backtest_compare.py")


if __name__ == "__main__":
    main()
