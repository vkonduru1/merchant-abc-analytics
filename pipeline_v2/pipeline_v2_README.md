# pipeline_v2/ — Comparable Pipeline Framework

This folder contains the shared scaffolding for a rigorous,
methodologically comparable evaluation of two classification approaches
for campaign send likelihood prediction.

## The problem this solves

v1 had two pipelines (Claude rule-based + ML statistical) that were
built independently with different feature sets, different validation
approaches, and no shared ground truth. This made comparison
descriptive rather than analytical.

v2 fixes this with a common foundation:

```
Shared canonical feature set
    ↓
┌───────────────┐    ┌───────────────┐
│ ML Pipeline   │    │ Claude Pipeline│
│ ml/pipeline   │    │ agents/pipeline│
│ _ml.py        │    │ _claude.py     │
└───────┬───────┘    └───────┬───────┘
        │                    │
        └─────────┬──────────┘
                  ↓
    pipeline_v2/backtest_compare.py
    Same holdout sets, same ground truth,
    accuracy measured independently
```

## Files in this folder

| File | Status | Description |
|---|---|---|
| `01_feature_analysis.py` | ✓ Built | Correlation matrix, VIF, RFC importance → canonical feature set |
| `02_feature_canonical.json` | Generated | Output of step 1 — locked feature set used by both pipelines |
| `03_backtest_compare.py` | Coming | Shared temporal cuts, independent scoring, comparison report |
| `README.md` | This file | |

## Methodology

### Feature selection (Step 1)
Three-pass filter applied to all candidate features from
`customer_derivatives`:

1. **Correlation analysis** — drop one from any pair with |r| > 0.75
2. **Variance Inflation Factor (VIF)** — drop features with VIF > 5
3. **RFC importance** — drop features with importance < 2%
4. **Domain review** — practitioner override to reinstate meaningful features

Output: `02_feature_canonical.json` — the canonical feature set.
Both `ml/pipeline_ml.py` and `agents/pipeline_claude.py` read from this file.

### Ground truth
Purchase outcome labels derived from `campaign_attribution`:
- Customer purchased within 30 days of campaign → `send_campaign`
- Customer purchased with no campaign in 30d window → `no_campaign_needed`
- Customer purchased very recently (recency < 14d) → `dont_send`
- No purchase, low engagement, ≥5 campaigns → `no_campaign_impact`

### Temporal holdout sets (same for both pipelines)
```
Cut 1: Train Jan 2022 – Jun 2022  | Test Jul – Dec 2022
Cut 2: Train Jan 2022 – Dec 2022  | Test Jan – Jun 2023
Cut 3: Train Jan 2022 – Jun 2023  | Test Jul – Dec 2023
Cut 4: Train Jan 2022 – Dec 2023  | Test Jan – Jun 2024
```

### Accuracy measurement
- Each pipeline scored independently against ground truth
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Primary objective: minimise FN (missed likely buyers)
- Comparison: descriptive only (sample too small for statistical significance)

## What this is NOT

This is an academic and educational exercise applied to synthetic data.
The findings are about methodology comparison — not about merchant-abc
specifically and not a reconstruction of the original VectorScient
PredictAlly product (which predicted purchase likelihood on real
merchant data with real outcomes).

The synthetic data was generated with known campaign-to-purchase
relationships built in. Both pipelines are validated against data
designed to have signal. Results should be interpreted accordingly.
