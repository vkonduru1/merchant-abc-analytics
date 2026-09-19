# ml/_v1_archive/

Files in this folder are the **v1 ML pipeline** built on 2026-09-18 —
the first working implementation of the KMeans++ → RFC → OR merge
recommendation pipeline.

## What these files do

| File | Description |
|---|---|
| `ml_feature_prep.py` | Loads customer_derivatives, applies VS grade weights, VS normalisation, runs silhouette grid. |
| `ml_cluster.py` | KMeans++ with VS 3-of-5 anchor interpretation logic. Produces cluster_likely per customer. |
| `ml_supervised.py` | RFC with TimeSeriesSplit CV. 98.5% CV accuracy on cluster labels. Produces prediction probabilities. |
| `ml_merge.py` | VS logical OR merge of cluster + RFC likelihoods. HYP segmentation. Writes to customer_recommendations. |
| `ml_backtest.py` | 4 temporal cuts (6-month windows). 64.8% OOT accuracy against purchase outcome ground truth. |

## Key findings from v1

- **Best k**: 2 (silhouette 0.441 — below 0.60 threshold, synthetic data limitation)
- **RFC CV accuracy**: 98.5% on KMeans++ cluster labels
- **OOT backtest accuracy**: 64.8% across 4 temporal cuts
- **Top RFC features**: recency_days (25.6%), campaign_influence_rate (21.9%), avg_campaigns_before_purchase (18.8%)
- **Claude vs ML agreement**: 23.8% — informative disagreement, not failure

## Why archived

Three methodological gaps identified post-v1:

1. **Feature sets not like-for-like** — Claude's classify_node received
   a different (larger) feature set than the ML pipeline. No formal
   multicollinearity filtering was applied before ML feature selection.

2. **No common ground truth baseline** — Claude's "confidence" is a
   data-volume proxy, not a calibrated probability. The two pipelines
   were never scored against the same holdout set independently.

3. **Drift between pipelines** — features were selected for each system
   independently rather than from a shared canonical set.

These are not implementation bugs — the pipeline runs correctly and the
methodology is sound. The archive reflects a decision to rebuild with
stricter comparability between the two pipelines.

## What's coming in ml/

`pipeline_ml.py` — the v2 ML pipeline:
- Features: canonical set from `pipeline_v2/02_feature_canonical.json`
  (post correlation analysis, VIF filtering, RFC importance ranking)
- Same temporal holdout sets as the Claude pipeline
- Accuracy measured independently against actual purchase outcomes
- Clean, single-file implementation replacing the 5-file v1 structure

See `pipeline_v2/` for the shared feature analysis and backtest framework.
