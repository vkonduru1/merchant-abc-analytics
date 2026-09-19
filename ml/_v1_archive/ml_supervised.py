#!/usr/bin/env python3
"""
ml/ml_supervised.py — Supervised RFC with TimeSeriesSplit CV.

Reads cluster_results.csv (labels from KMeans++) as training targets,
trains a RandomForestClassifier on the 13 ML features, evaluates with
TimeSeriesSplit cross-validation, and outputs per-customer probability
distributions across the 4 recommendation labels.

Why TimeSeriesSplit (not KFold):
  Customer purchase behaviour has temporal autocorrelation. A future
  purchase appearing in the training set would leak the answer and
  inflate accuracy scores. TimeSeriesSplit always trains on the past
  and tests on the future — the only honest CV for this data.

Objective (from VS PredictAlly doc):
  Minimize FN (False Negatives) — missing a likely buyer costs more
  than a false positive. We track FN rate explicitly across all labels.

Output per customer:
  send_campaign_prob, dont_send_prob,
  no_campaign_needed_prob, no_campaign_impact_prob,
  rfc_label, rfc_confidence

Usage
    python ml/ml_supervised.py              # train + evaluate
    python ml/ml_supervised.py --dry-run    # CV report only
    python ml/ml_supervised.py --port 5433
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

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LABEL_ORDER = [
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
]

RFC_PARAMS = {
    "n_estimators":      200,
    "max_depth":         None,
    "min_samples_leaf":  5,
    "class_weight":      "balanced",   # handles class imbalance
    "random_state":      42,
    "n_jobs":            -1,
}

N_CV_FOLDS = 5


# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    features_path = OUTPUT_DIR / "features_prepared.csv"
    cluster_path  = OUTPUT_DIR / "cluster_results.csv"
    meta_path     = OUTPUT_DIR / "feature_meta.json"

    for p in [features_path, cluster_path, meta_path]:
        if not p.exists():
            sys.exit(f"✗ {p.name} not found. Run ml_feature_prep.py and "
                     f"ml_cluster.py first.")

    df_feat    = pd.read_csv(features_path)
    df_cluster = pd.read_csv(cluster_path)
    meta       = json.load(open(meta_path))
    feature_cols = [f["col"] for f in meta["features"]]

    df = df_feat.merge(
        df_cluster[["customer_id", "cluster_label", "cluster_likely"]],
        on="customer_id", how="inner"
    )
    print(f"  {len(df):,} customers with features + cluster labels")
    return df, feature_cols, meta


# ══════════════════════════════════════════════════════════════════════════
# TimeSeriesSplit CV
# ══════════════════════════════════════════════════════════════════════════
def run_cv(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    TimeSeriesSplit with N_CV_FOLDS folds.
    Returns cross-validated predictions and per-fold accuracy scores.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    rfc  = RandomForestClassifier(**RFC_PARAMS)

    fold_scores = []
    all_preds   = np.empty(len(y), dtype=object)
    all_probs   = np.zeros((len(y), len(LABEL_ORDER)))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Only train if we have all labels (or at least 2)
        if len(set(y_train)) < 2:
            continue

        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        probs = rfc.predict_proba(X_test)

        all_preds[test_idx] = preds
        # Align probabilities to LABEL_ORDER
        classes = list(rfc.classes_)
        for li, label in enumerate(LABEL_ORDER):
            if label in classes:
                all_probs[test_idx, li] = probs[:, classes.index(label)]

        acc = accuracy_score(y_test, preds)
        fold_scores.append(acc)
        print(f"    Fold {fold}: {len(train_idx):,} train → "
              f"{len(test_idx):,} test  accuracy={acc:.3f}")

    return all_preds, all_probs, fold_scores


# ══════════════════════════════════════════════════════════════════════════
# Final model (full training)
# ══════════════════════════════════════════════════════════════════════════
def train_final(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    rfc = RandomForestClassifier(**RFC_PARAMS)
    rfc.fit(X, y)
    return rfc


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(y: np.ndarray, cv_preds: np.ndarray, fold_scores: list,
           feature_cols: list, rfc: RandomForestClassifier,
           label_counts: dict) -> dict:

    # Only evaluate on rows that got a CV prediction
    mask = cv_preds != None  # noqa: E711
    y_eval    = y[mask]
    pred_eval = cv_preds[mask]

    acc = accuracy_score(y_eval, pred_eval)

    print(f"\n{'═' * 62}")
    print(f"  Supervised RFC — Cross-Validation Report")
    print(f"  (TimeSeriesSplit, {N_CV_FOLDS} folds, VS methodology)")
    print(f"{'═' * 62}")
    print(f"\n  Per-fold accuracy:")
    for i, s in enumerate(fold_scores, 1):
        bar = "█" * int(s * 30)
        print(f"    Fold {i}  {s:.3f}  {bar}")
    print(f"  Mean CV accuracy : {np.mean(fold_scores):.3f}")
    print(f"  Std              : {np.std(fold_scores):.3f}")
    print(f"  Overall accuracy : {acc:.3f}")

    print(f"\n  Classification report (CV predictions):")
    labels_present = [l for l in LABEL_ORDER if l in set(y_eval)]
    report_str = classification_report(
        y_eval, pred_eval,
        labels=labels_present,
        target_names=labels_present,
        zero_division=0
    )
    for line in report_str.split("\n"):
        print(f"    {line}")

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_eval, pred_eval, labels=labels_present)
    header = "  " + " ".join(f"{l[:8]:>10}" for l in labels_present)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels_present[i][:10]:<12}" +
              " ".join(f"{v:>10}" for v in row))

    # FN analysis (VS primary objective: minimize FNs)
    print(f"\n  False Negative analysis (VS primary objective):")
    for i, label in enumerate(labels_present):
        fn = int(cm[i].sum() - cm[i, i]) if i < len(cm) else 0
        tp = int(cm[i, i]) if i < len(cm) else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"    {label:<26}  TP={tp:>4}  FN={fn:>4}  "
              f"FN_rate={fn_rate:.1%}")

    # Feature importance
    if hasattr(rfc, "feature_importances_"):
        importances = rfc.feature_importances_
        ranked = sorted(zip(feature_cols, importances),
                       key=lambda x: x[1], reverse=True)
        print(f"\n  Feature importance (RFC):")
        for col, imp in ranked[:10]:
            bar = "█" * int(imp * 100)
            print(f"    {col:<38} {imp:.4f}  {bar}")

    print(f"{'═' * 62}\n")

    return {
        "fold_scores": fold_scores,
        "mean_cv_accuracy": round(float(np.mean(fold_scores)), 4),
        "overall_accuracy": round(float(acc), 4),
        "n_folds": N_CV_FOLDS,
    }


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train RFC on cluster labels, evaluate with TimeSeriesSplit CV.")
    ap.add_argument("--dry-run", action="store_true",
                    help="CV report only, do not write output files")
    ap.add_argument("--host", help="DB host")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    print("Loading features and cluster labels …")
    df, feature_cols, meta = load_data()

    X = df[feature_cols].values.astype(np.float64)
    y = df["cluster_label"].values

    print(f"\nLabel distribution (from KMeans++):")
    for label, n in Counter(y).most_common():
        print(f"  {label:<26} {n:>5}")

    print(f"\nRunning TimeSeriesSplit CV ({N_CV_FOLDS} folds) …")
    cv_preds, cv_probs, fold_scores = run_cv(X, y)

    print(f"\nTraining final model on full dataset …")
    rfc = train_final(X, y)

    # Final probabilities from the full model
    final_probs = np.zeros((len(df), len(LABEL_ORDER)))
    final_preds = rfc.predict(X)
    final_prob_raw = rfc.predict_proba(X)
    classes = list(rfc.classes_)
    for li, label in enumerate(LABEL_ORDER):
        if label in classes:
            final_probs[:, li] = final_prob_raw[:, classes.index(label)]

    metrics = report(y, cv_preds, fold_scores, feature_cols, rfc,
                     Counter(y))

    if args.dry_run:
        print("Dry run — no files written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save RFC probability output per customer
    result_df = df[["customer_id"]].copy()
    for li, label in enumerate(LABEL_ORDER):
        result_df[f"{label}_prob"] = final_probs[:, li].round(4)
    result_df["rfc_label"]      = final_preds
    result_df["rfc_confidence"] = final_probs.max(axis=1).round(4)

    out_path = OUTPUT_DIR / "rfc_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"✓ RFC results       : {out_path}")

    # Save metrics
    out_meta = OUTPUT_DIR / "rfc_meta.json"
    with open(out_meta, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ RFC metrics       : {out_meta}")


if __name__ == "__main__":
    main()
