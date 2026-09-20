# ML Pipeline — Case Scenarios

Iterative refinement of the ML pipeline for campaign send likelihood prediction.
**Status: SATISFACTORY — paused at case_04 to move to frontend + deployment.**

---

## Final Results Summary

| Case | Change | All Cuts | Mature (3-4) | Cut 4 | Status |
|------|--------|----------|--------------|-------|--------|
| v2.0 | KMeans++ OR merge + HYP segmentation | 31.3% | 60.9% | 65.3% | ✗ HYP_FP inflation |
| case_01 | RFC argmax direct | 40.8% | 61.0% | 65.3% | ✓ Fixed distribution, found root cause |
| case_02 | Per-window feature recomputation | 58.3% | 70.8% | 75.2% | ✓ ACCEPTED — best single-pipeline ML |
| case_03 | Post-prediction recency override | 80.0% | 64.5% | 73.0% | ✗ REJECTED — Cut 4 worse |
| case_04 | Confidence-based ML + Claude ensemble | 65.0% | 71.5% | 79.8% | ✓ SATISFACTORY — final result |
| **Claude v2** | **Rule-based + reasoning** | **66.5%** | **64.6%** | **63.5%** | **Benchmark** |

**case_04 beats Claude on mature cuts (71.5% vs 64.6%) and Cut 4 (79.8% vs 63.5%).**

---

## The ML Journey — Chronological

```
v2.0  → 31.3%   HYP segmentation inflates no_campaign_needed (56.5% vs GT 29.8%)
         ↓ Fix: remove HYP segmentation
case_01 → 40.8%  Label distribution fixed (100% in-sample match)
                 Root cause found: static features vs dynamic window labels
         ↓ Fix: recompute features per window
case_02 → 70.8%  (mature) Walk-forward fixes mismatch
                 New finding: training label availability
                 Cut 4 = 75.2% — first production-realistic accuracy
         ↓ Fix: post-prediction recency override
case_03 → 64.5%  (mature) REJECTED — Cut 4 accuracy dropped (75.2% → 73.0%)
                 Override too blunt for well-trained walk-forward model
         ↓ Fix: confidence-based routing to Claude
case_04 → 71.5%  (mature) ENSEMBLE — best result across all cases
                 Cut 4 = 79.8% — best single-cut result
                 [PAUSED — satisfactory for this phase]
```

---

## Key Findings

### 1. Training label availability (case_02)
The most important structural finding:
```
Cuts 1-3 training windows have only 2 classes
(dont_send + no_campaign_impact) because the
campaign program was immature before mid-2023.

A 4-class classifier cannot be trained on 2-class data.
Cut 4 is the only production-realistic cut.
75.2% (case_02) / 79.8% (case_04) is the realistic
starting accuracy for a production deployment.
```

### 2. The orthogonality argument (case_04)
```
ML failure modes:   sparse training classes, dont_send FN=74%
Claude failure modes: send_campaign conservative, no rank ordering

These are LARGELY ORTHOGONAL.
Ensemble accuracy > either pipeline alone on mature cuts.
This is the architectural case for agentic ensembles.
```

### 3. Diminishing returns signal
```
case_01 → case_02: +9.8% mature accuracy (major)
case_02 → case_04: +0.7% mature accuracy (marginal)

Stopping criterion met. Further tuning requires
real merchant data, not more synthetic iterations.
```

### 4. Feature importance (stable across cases)
```
recency_days                     38.5%
campaign_influence_rate          22.6%
spend_last_12_months             11.3%
spend_last_3_months               7.3%
stddev_days_campaign_to_purchase  6.0%
total_campaigns_received          4.4%
avg_days_campaign_to_purchase     3.9%
email_open_rate                   2.9%
months_since_customer             1.6%
frequency_months                  1.6%
```

---

## FN Rate Comparison — Final

### All cuts average
```
Label               case_02    case_04    Claude    Best
─────────────────────────────────────────────────────────
send_campaign        0.0%      11.2%     23.8%     case_02
dont_send           74.0%      54.8%      9.7%     Claude
no_campaign_needed  51.0%      32.3%     37.5%     case_04
no_campaign_impact  24.0%      29.3%     26.5%     case_02
```

### Cut 4 only (most production-realistic)
```
Label               case_02    case_04    Claude    Best
─────────────────────────────────────────────────────────
send_campaign        0.0%      14.0%     48.0%     case_02
dont_send           77.0%      55.0%     10.0%     Claude
no_campaign_needed  51.0%      53.0%     74.0%     case_02
no_campaign_impact  23.0%       7.0%     19.0%     case_04
```

---

## What the Ensemble Routing Looks Like in Production

```
For each customer (496 total):
  187 (37.7%) → ML label directly (high confidence ≥ 0.80)
  125 (25.2%) → Claude label (low ML confidence < 0.60)
  184 (37.1%) → ML label + human review flag (middle band)

This is CCAR-F D1D HITL escalation in practice:
  high confidence → automated decision
  low confidence  → LLM reasoning
  middle band     → human review queue
```

---

## Next Optimisation Opportunities (when returning)

### Immediate (30 minutes)
```bash
# Threshold experiment — may recover send_campaign FN regression
python ml/cases/case_04/pipeline_ml_case04.py --dry-run --high 0.85 --low 0.55
```
Hypothesis: tighter thresholds reduce Claude routing
of true send_campaign customers. Expected send_campaign
FN drops from 11.2% → ~5% while mature accuracy holds.

### Case 05 — Threshold grid search
Try all [high=0.75-0.90] × [low=0.50-0.65] combinations.
Find Pareto-optimal threshold pair.

### Case 06 — Label-specific routing
```
IF ml_label == 'dont_send' → always Claude (Claude FN=9.7%)
IF ml_label == 'send_campaign' AND conf ≥ 0.75 → always ML
ELSE → confidence routing
```

### Case 07 — Real merchant data
All synthetic limitations resolved. Expected mature accuracy > 80%.

---

## Case Details Index

| Case | File | Notes |
|------|------|-------|
| case_01 | `ml/cases/case_01/pipeline_ml_case01.py` | `ml/cases/case_01/notes_case01.md` |
| case_02 | `ml/cases/case_02/pipeline_ml_case02.py` | `ml/cases/case_02/notes_case02.md` |
| case_03 | `ml/cases/case_03/pipeline_ml_case03.py` | `ml/cases/case_03/notes_case03.md` |
| case_04 | `ml/cases/case_04/pipeline_ml_case04.py` | `ml/cases/case_04/notes_case04.md` |

---

## Resuming This Work

When returning to ML optimisation:
1. Read this README top to bottom
2. Run case_04 dry-run with --high 0.85 --low 0.55 first
3. Check if that recovers send_campaign FN before building Case 05
4. All pipeline_v2/ scripts still work — ground truth derivation unchanged
5. backtest_compare.py auto-picks latest model_version from DB

**The architecture is sound. The data is the variable.**
