#!/usr/bin/env python3
"""
pipeline_v2/01_feature_analysis.py

PURPOSE
-------
Produce the canonical feature set used by BOTH pipelines:
  - ml/pipeline_ml.py       (statistical ML pipeline)
  - agents/pipeline_claude.py  (Claude rule-based pipeline)

This script is run ONCE before either pipeline is built.
Its output — 02_feature_canonical.json — is the single source
of truth for feature selection across the entire v2 framework.

THREE-PASS FILTER
-----------------
Pass 1 — Pearson correlation matrix
  Drop one feature from any pair where |r| > CORR_THRESHOLD (0.75)
  Rule: keep the feature with higher RFC importance (computed in Pass 3)
  If RFC not yet available: keep the more interpretable feature

Pass 2 — Variance Inflation Factor (VIF)
  Drop any surviving feature with VIF > VIF_THRESHOLD (5.0)
  VIF > 5 indicates the feature is largely explained by the others
  — it adds multicollinearity noise, not independent signal

Pass 3 — RFC feature importance
  Train a quick RandomForestClassifier on surviving features
  using KMeans++ cluster labels as proxy target
  Drop features with importance < IMPORTANCE_THRESHOLD (0.02)
  These contribute < 2% of the model's decisions — noise

Pass 4 — Domain review output
  Print the shortlist with business descriptions
  Practitioner can override any drop decision
  Final canonical set written to 02_feature_canonical.json

CANDIDATE FEATURES
------------------
All features from customer_derivatives that are:
  - Numeric (continuous or count)
  - Not definitionally derived from another candidate
    (e.g. total_purchases_without_campaign = total_orders
     minus total_purchases_with_campaign — excluded as derivative)
  - Not a label or identifier
    (customer_type, attribution_segment excluded)

GROUND TRUTH FOR RFC (Pass 3)
------------------------------
KMeans++ cluster labels from ml/_v1_archive output are used
as a proxy target for RFC importance ranking.
If not available, RFC is trained on a binary likely/unlikely
derived from campaign_influence_rate threshold.

Usage
    python pipeline_v2/01_feature_analysis.py
    python pipeline_v2/01_feature_analysis.py --dry-run
    python pipeline_v2/01_feature_analysis.py --corr-threshold 0.80
    python pipeline_v2/01_feature_analysis.py --port 5433
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "data_generator"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Thresholds (all overridable via CLI) ──────────────────────────────────
CORR_THRESHOLD       = 0.75   # |r| above this → one of the pair is dropped
VIF_THRESHOLD        = 5.0    # VIF above this → feature dropped
IMPORTANCE_THRESHOLD = 0.02   # RFC importance below this → feature dropped

# ── Candidate features with business descriptions ─────────────────────────
# Format: (column_name, description, business_rationale)
CANDIDATES = [
    ("total_revenue",
     "Total lifetime revenue ($)",
     "Primary LTV signal — higher = more valuable customer"),

    ("total_orders",
     "Total number of orders placed",
     "Purchase frequency proxy — correlates with loyalty"),

    ("average_order_value",
     "Average order value ($)",
     "Spend per transaction — basket size signal"),

    ("recency_days",
     "Days since last purchase",
     "Strongest individual predictor in v1 RFC (25.6% importance)"),

    ("frequency_months",
     "Average months between purchases",
     "Purchase cadence — lower = more frequent buyer"),

    ("spend_last_3_months",
     "Revenue in last 3 months ($)",
     "Recent activity signal — detects reactivation or churn"),

    ("spend_last_12_months",
     "Revenue in last 12 months ($)",
     "Annual spend window — stabilises seasonal variation"),

    ("email_open_rate",
     "Email open rate (0-1)",
     "Engagement with campaigns — attention signal"),

    ("email_click_rate",
     "Email click rate (0-1)",
     "Active engagement — stronger intent signal than open rate"),

    ("campaign_influence_rate",
     "Fraction of orders with a campaign in 30d window (0-1)",
     "Core attribution signal — measures campaign dependency"),

    ("avg_campaigns_before_purchase",
     "Average campaigns received before each order",
     "Nurture requirement — how many touches before conversion"),

    ("total_purchases_with_campaign",
     "Count of orders with campaign in 30d window",
     "Absolute influenced order count"),

    ("total_campaigns_received",
     "Total campaigns received (lifetime)",
     "Campaign exposure volume"),

    ("days_since_last_campaign",
     "Days since last campaign was received",
     "Campaign recency — gap between last touch and now"),

    ("avg_days_campaign_to_purchase",
     "Avg days from campaign send to purchase (influenced orders only)",
     "Response speed — how quickly this customer converts after a campaign"),

    ("stddev_days_campaign_to_purchase",
     "Std deviation of campaign-to-purchase lag",
     "Response consistency — predictability of conversion timing"),

    ("months_since_customer",
     "Months since customer first created",
     "Customer tenure — longevity signal"),
]

CANDIDATE_COLS = [c[0] for c in CANDIDATES]
CANDIDATE_META = {c[0]: {"description": c[1], "rationale": c[2]}
                  for c in CANDIDATES}


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


def load_data(conn) -> pd.DataFrame:
    cols = ", ".join(CANDIDATE_COLS +
                     ["customer_id", "attribution_segment"])
    query = f"""
        SELECT DISTINCT ON (customer_id)
            {cols}
        FROM customer_derivatives
        ORDER BY customer_id, run_id DESC
    """
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df):,} customers, "
          f"{len(CANDIDATE_COLS)} candidate features")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation + clip extreme outliers at 99th percentile."""
    df = df.copy()
    for col in CANDIDATE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            continue
        null_n = df[col].isna().sum()
        if null_n > 0:
            fill = df[col].max() if col == "recency_days" else df[col].median()
            df[col] = df[col].fillna(fill)
        p99 = df[col].quantile(0.99)
        if p99 > 0:
            df[col] = df[col].clip(upper=p99)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Pass 1 — Correlation analysis
# ══════════════════════════════════════════════════════════════════════════
def correlation_pass(df: pd.DataFrame,
                     threshold: float) -> tuple[list, pd.DataFrame]:
    """
    Compute Pearson correlation matrix.
    For each pair with |r| > threshold, flag the pair.
    Decision on which to drop deferred to after RFC importance (Pass 3).
    Returns surviving features and the full correlation matrix.
    """
    corr = df[CANDIDATE_COLS].corr(method="pearson")

    flagged_pairs = []
    seen = set()
    for i, c1 in enumerate(CANDIDATE_COLS):
        for j, c2 in enumerate(CANDIDATE_COLS):
            if i >= j:
                continue
            r = corr.loc[c1, c2]
            if abs(r) > threshold:
                key = tuple(sorted([c1, c2]))
                if key not in seen:
                    flagged_pairs.append((c1, c2, round(r, 4)))
                    seen.add(key)

    return flagged_pairs, corr


# ══════════════════════════════════════════════════════════════════════════
# Pass 2 — VIF
# ══════════════════════════════════════════════════════════════════════════
def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.
    VIF_i = 1 / (1 - R²_i) where R²_i is from regressing
    feature i on all other features.
    Uses numpy OLS for speed (no statsmodels dependency).
    """
    from numpy.linalg import lstsq

    X = df[cols].values.astype(float)
    n_features = X.shape[1]
    vifs = []

    for i in range(n_features):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_with_const = np.column_stack([np.ones(len(y)), X_others])
        try:
            coeffs, _, _, _ = lstsq(X_with_const, y, rcond=None)
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        except Exception:
            vif = float("inf")
        vifs.append({"feature": cols[i], "vif": round(vif, 2)})

    return pd.DataFrame(vifs).sort_values("vif", ascending=False)


def vif_pass(df: pd.DataFrame, cols: list,
             threshold: float) -> tuple[list, pd.DataFrame]:
    """Iteratively drop highest VIF feature until all are below threshold."""
    remaining = list(cols)
    history = []

    while True:
        vif_df = compute_vif(df, remaining)
        max_row = vif_df.iloc[0]
        if max_row["vif"] <= threshold:
            break
        drop = max_row["feature"]
        history.append((drop, max_row["vif"]))
        remaining.remove(drop)

    final_vif = compute_vif(df, remaining)
    return remaining, final_vif, history


# ══════════════════════════════════════════════════════════════════════════
# Pass 3 — RFC importance
# ══════════════════════════════════════════════════════════════════════════
def get_proxy_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Generate proxy labels for RFC importance ranking.
    Tries to load v1 cluster labels; falls back to binary
    derived from campaign_influence_rate threshold.
    """
    cluster_path = ROOT_DIR / "ml" / "_v1_archive" / ".." / "output" / "cluster_results.csv"
    # Try ml/output first (v1 output may still be there)
    for candidate in [
        ROOT_DIR / "ml" / "output" / "cluster_results.csv",
        ROOT_DIR / "pipeline_v2" / "output" / "cluster_results.csv",
    ]:
        if candidate.exists():
            clusters = pd.read_csv(candidate)
            clusters["customer_id"] = clusters["customer_id"].astype(str)
            merged = df[["customer_id"]].astype({"customer_id": str}).merge(
                clusters[["customer_id", "cluster_label"]],
                on="customer_id", how="left"
            )
            if merged["cluster_label"].notna().sum() > len(df) * 0.8:
                print("  Using v1 cluster labels as RFC proxy target")
                return merged["cluster_label"].fillna("no_campaign_impact").values

    # Fallback: binary from campaign_influence_rate
    print("  No cluster labels found — using binary influence_rate proxy")
    labels = np.where(
        df["campaign_influence_rate"].fillna(0) >= 0.5,
        "likely", "unlikely"
    )
    return labels


def rfc_importance_pass(df: pd.DataFrame, cols: list,
                        threshold: float) -> tuple[list, pd.DataFrame]:
    """Train RFC on surviving features, rank by importance."""
    X = df[cols].values.astype(float)
    y = get_proxy_labels(df)

    rfc = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rfc.fit(X, y)

    importance_df = pd.DataFrame({
        "feature":    cols,
        "importance": rfc.feature_importances_.round(4),
    }).sort_values("importance", ascending=False)

    surviving = importance_df[
        importance_df["importance"] >= threshold
    ]["feature"].tolist()

    return surviving, importance_df, rfc


# ══════════════════════════════════════════════════════════════════════════
# Collinear pair resolution (after RFC importance)
# ══════════════════════════════════════════════════════════════════════════
def resolve_corr_pairs(flagged_pairs: list,
                       importance_df: pd.DataFrame,
                       surviving_after_vif: list) -> list:
    """
    For each flagged correlation pair where both survive VIF:
    Drop the one with lower RFC importance.
    Log the decision.
    """
    imp = dict(zip(importance_df["feature"], importance_df["importance"]))
    to_drop = set()
    decisions = []

    for c1, c2, r in flagged_pairs:
        if c1 not in surviving_after_vif or c2 not in surviving_after_vif:
            continue  # one already dropped by VIF
        if c1 in to_drop or c2 in to_drop:
            continue  # one already resolved

        imp1 = imp.get(c1, 0)
        imp2 = imp.get(c2, 0)
        drop = c1 if imp1 < imp2 else c2
        keep = c2 if drop == c1 else c1
        to_drop.add(drop)
        decisions.append({
            "pair": (c1, c2), "r": r,
            "drop": drop, "keep": keep,
            "imp_drop": round(imp.get(drop, 0), 4),
            "imp_keep": round(imp.get(keep, 0), 4),
        })

    return to_drop, decisions


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(flagged_pairs, corr, vif_dropped, vif_df,
           corr_decisions, importance_df, final_features,
           corr_threshold, vif_threshold, imp_threshold) -> None:

    print(f"\n{'═' * 66}")
    print(f"  Feature Analysis Report")
    print(f"  Thresholds: corr={corr_threshold} | VIF={vif_threshold} "
          f"| importance={imp_threshold}")
    print(f"{'═' * 66}")

    # Pass 1
    print(f"\n  PASS 1 — Correlation Analysis (|r| > {corr_threshold})")
    print(f"  {'─' * 60}")
    if not flagged_pairs:
        print(f"  No pairs exceed threshold.")
    else:
        for c1, c2, r in flagged_pairs:
            print(f"  {c1:<38} ↔  {c2:<38}  r={r:+.3f}")

    # Pass 2
    print(f"\n  PASS 2 — VIF Analysis (drop if VIF > {vif_threshold})")
    print(f"  {'─' * 60}")
    if not vif_dropped:
        print(f"  No features dropped by VIF.")
    else:
        for feat, vif in vif_dropped:
            print(f"  DROPPED  {feat:<38}  VIF={vif:.1f}")
    print(f"\n  Surviving VIF scores:")
    for _, row in vif_df.iterrows():
        flag = " ← borderline" if row["vif"] > vif_threshold * 0.8 else ""
        print(f"    {row['feature']:<38}  VIF={row['vif']:>6.2f}{flag}")

    # Pass 3
    print(f"\n  PASS 3 — RFC Importance (drop if importance < {imp_threshold})")
    print(f"  {'─' * 60}")
    for _, row in importance_df.iterrows():
        bar    = "█" * int(row["importance"] * 200)
        status = "DROP" if row["importance"] < imp_threshold else "    "
        print(f"  {status}  {row['feature']:<38}  "
              f"{row['importance']:.4f}  {bar}")

    # Correlation resolution
    if corr_decisions:
        print(f"\n  PASS 1 Resolution — Collinear pair decisions:")
        print(f"  {'─' * 60}")
        for d in corr_decisions:
            print(f"  Keep {d['keep']:<30} (imp={d['imp_keep']:.4f})")
            print(f"  Drop {d['drop']:<30} (imp={d['imp_drop']:.4f})"
                  f"  |r|={abs(d['r']):.3f}")
            print()

    # Final feature set
    print(f"\n  {'═' * 60}")
    print(f"  CANONICAL FEATURE SET — {len(final_features)} features")
    print(f"  {'═' * 60}")
    for feat in final_features:
        meta = CANDIDATE_META.get(feat, {})
        print(f"  ✓  {feat:<38}  {meta.get('description', '')}")

    dropped = [c for c in CANDIDATE_COLS if c not in final_features]
    if dropped:
        print(f"\n  Dropped ({len(dropped)}):")
        for feat in dropped:
            print(f"  ✗  {feat}")
    print(f"\n{'═' * 66}\n")


# ══════════════════════════════════════════════════════════════════════════
# Write canonical feature set
# ══════════════════════════════════════════════════════════════════════════
def write_canonical(final_features: list, importance_df: pd.DataFrame,
                    vif_df: pd.DataFrame, corr_decisions: list,
                    args) -> None:

    imp = dict(zip(importance_df["feature"], importance_df["importance"]))
    vif = dict(zip(vif_df["feature"], vif_df["vif"]))

    output = {
        "version":     "v2",
        "created":     "2026-09-19",
        "description": (
            "Canonical feature set for pipeline_v2. "
            "Used identically by ml/pipeline_ml.py and "
            "agents/pipeline_claude.py. "
            "Do not modify without re-running 01_feature_analysis.py."
        ),
        "thresholds": {
            "correlation": args.corr_threshold,
            "vif":         args.vif_threshold,
            "importance":  args.importance_threshold,
        },
        "features": [
            {
                "column":      feat,
                "description": CANDIDATE_META[feat]["description"],
                "rationale":   CANDIDATE_META[feat]["rationale"],
                "rfc_importance": round(imp.get(feat, 0), 4),
                "vif":            round(vif.get(feat, 0), 2),
            }
            for feat in final_features
        ],
        "dropped_features": [
            c for c in CANDIDATE_COLS if c not in final_features
        ],
        "corr_pair_decisions": [
            {
                "pair":     list(d["pair"]),
                "r":        d["r"],
                "kept":     d["keep"],
                "dropped":  d["drop"],
            }
            for d in corr_decisions
        ],
    }

    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = SCRIPT_DIR / "02_feature_canonical.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Canonical feature set → {out_path}")

    # Also save correlation matrix as CSV for inspection
    corr_path = out_dir / "correlation_matrix.csv"
    # recompute corr on all candidates for the saved matrix
    return out_path


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Feature analysis — produce canonical feature set for pipeline_v2.")
    ap.add_argument("--dry-run",
                    action="store_true",
                    help="Run analysis and report, do not write output files")
    ap.add_argument("--corr-threshold",
                    type=float, default=CORR_THRESHOLD,
                    help=f"Correlation drop threshold (default: {CORR_THRESHOLD})")
    ap.add_argument("--vif-threshold",
                    type=float, default=VIF_THRESHOLD,
                    help=f"VIF drop threshold (default: {VIF_THRESHOLD})")
    ap.add_argument("--importance-threshold",
                    type=float, default=IMPORTANCE_THRESHOLD,
                    help=f"RFC importance drop threshold (default: {IMPORTANCE_THRESHOLD})")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = get_conn(args)
    print(f"✓ Connected")
    print(f"\nLoading candidate features …")
    df_raw = load_data(conn)
    conn.close()

    print(f"Preprocessing (median imputation, 99th percentile clip) …")
    df = preprocess(df_raw)

    # Pass 1 — correlation
    print(f"\nPass 1 — Correlation matrix (threshold={args.corr_threshold}) …")
    flagged_pairs, corr_matrix = correlation_pass(df, args.corr_threshold)
    print(f"  {len(flagged_pairs)} correlated pairs flagged")

    # Save correlation matrix
    if not args.dry_run:
        out_dir = SCRIPT_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        corr_matrix.round(3).to_csv(out_dir / "correlation_matrix.csv")
        print(f"  Correlation matrix → pipeline_v2/output/correlation_matrix.csv")

    # Pass 2 — VIF (on all candidates first)
    print(f"\nPass 2 — VIF (threshold={args.vif_threshold}) …")
    surviving_vif, vif_df, vif_dropped = vif_pass(
        df, CANDIDATE_COLS, args.vif_threshold
    )
    print(f"  Dropped by VIF: {len(vif_dropped)} features")
    print(f"  Surviving VIF:  {len(surviving_vif)} features")

    # Pass 3 — RFC importance (on VIF survivors)
    print(f"\nPass 3 — RFC importance (threshold={args.importance_threshold}) …")
    surviving_imp, importance_df, rfc = rfc_importance_pass(
        df, surviving_vif, args.importance_threshold
    )
    print(f"  Dropped by importance: "
          f"{len(surviving_vif) - len(surviving_imp)} features")
    print(f"  Surviving importance:  {len(surviving_imp)} features")

    # Resolve correlation pairs using RFC importance
    corr_to_drop, corr_decisions = resolve_corr_pairs(
        flagged_pairs, importance_df, surviving_vif
    )
    final_features = [f for f in surviving_imp if f not in corr_to_drop]
    print(f"  Dropped by corr resolution: {len(corr_to_drop)} features")

    # Report
    report(flagged_pairs, corr_matrix, vif_dropped, vif_df,
           corr_decisions, importance_df, final_features,
           args.corr_threshold, args.vif_threshold,
           args.importance_threshold)

    # ── Domain override (Pass 4) ─────────────────────────────────────────
    # Two practitioner overrides applied after statistical filtering:
    #
    # Override 1: Swap avg_campaigns_before_purchase ↔ campaign_influence_rate
    #   avg_campaigns_before_purchase suffers from averaging noise — it masks
    #   the temporal distribution of campaigns within a customer's journey.
    #   campaign_influence_rate (normalised 0-1) is a cleaner signal.
    #   Both are correlated (r=0.865) but measure different things:
    #     avg_campaigns = how many touches needed (count, unnormalised)
    #     influence_rate = what fraction of orders were campaign-driven (rate)
    #
    # Override 2: Reinstate email_open_rate
    #   Importance 0.0176 — just below the 0.02 threshold.
    #   The canonical set has zero email engagement features without it.
    #   For a campaign response prediction problem, email engagement is
    #   essential signal. Reinstated via domain override.

    OVERRIDE_DROP    = {"avg_campaigns_before_purchase"}
    OVERRIDE_ADD     = {"campaign_influence_rate", "email_open_rate"}

    final_features = [f for f in final_features
                      if f not in OVERRIDE_DROP]
    for feat in OVERRIDE_ADD:
        if feat not in final_features:
            final_features.append(feat)

    # Sort by RFC importance for clean display
    imp_lookup = dict(zip(importance_df["feature"],
                          importance_df["importance"]))
    final_features.sort(key=lambda f: imp_lookup.get(f, 0), reverse=True)

    print(f"\n  PASS 4 — Domain Override")
    print(f"  {'─' * 60}")
    print(f"  Dropped  : avg_campaigns_before_purchase "
          f"(averaging noise, replaced by influence_rate)")
    print(f"  Reinstated: campaign_influence_rate "
          f"(normalised 0-1, cleaner signal)")
    print(f"  Reinstated: email_open_rate "
          f"(only email engagement signal, just below threshold)")
    print(f"\n  FINAL CANONICAL SET — {len(final_features)} features")
    print(f"  {'─' * 60}")
    for feat in final_features:
        meta = CANDIDATE_META.get(feat, {})
        imp  = imp_lookup.get(feat, 0)
        src  = " [override]" if feat in OVERRIDE_ADD else ""
        print(f"  ✓  {feat:<42} imp={imp:.4f}{src}")

    if args.dry_run:
        print("\nDry run — 02_feature_canonical.json not written.")
        return

    write_canonical(final_features, importance_df, vif_df,
                    corr_decisions, args)

    print(f"\nNext step:")
    print(f"  Review the canonical feature set above.")
    print(f"  If any feature should be reinstated (domain override),")
    print(f"  edit 02_feature_canonical.json directly before")
    print(f"  building pipeline_ml.py and pipeline_claude.py.")


if __name__ == "__main__":
    main()
