# ML Pipeline — Case Scenarios

Iterative refinement of `ml/pipeline_ml.py`.
Each case documents one change, its hypothesis, findings, and decision.
Claude v2 accuracy (66.5% all cuts / 64.6% mature cuts) is the benchmark.

---

## Results Summary

| Case | Change | ML Accuracy (all) | ML Accuracy (mature) | Claude Accuracy | Decision |
|------|--------|-------------------|----------------------|-----------------|----------|
| v2.0 | Baseline — KMeans++ OR merge + HYP segmentation | 31.3% | 60.9% | 66.5% | ✗ HYP_FP inflation — no_campaign_needed 56.5% vs GT 29.8% |
| case_01 | RFC argmax direct — remove HYP segmentation | 40.8% | 61.0% | 66.5% | ✓ Label distribution fixed. Root cause identified: static features vs dynamic window labels. Option C: exclude Cuts 1-2 |
| case_02 | Per-window feature recomputation (walk-forward) | TBD | TBD | 64.6% | TBD |
| case_03 | Rolling window retrain (concept drift) | TBD | TBD | TBD | TBD |
| case_04 | Confidence-based routing ML + Claude ensemble | TBD | TBD | TBD | TBD |

**Mature windows** = Cuts 3-4 only (Jul 2023 – Jun 2024).
Cuts 1-2 excluded because 93.8% of window ground truth is
no_campaign_impact (campaign program immature in 2022-H1 2023).

---

## Case Details

### v2.0 Baseline
**File:** `ml/pipeline_ml.py`
**Change:** Initial build — KMeans++ (k=2) OR merge + HYP segmentation
**Result:**
- ML accuracy: 31.3% all / 60.9% mature cuts
- no_campaign_needed 56.5% predicted vs 29.8% GT — HYP_FP inflation
- Root cause: OR merge too permissive (80.8% Likely), rank formula
  collapses Likely pool into no_campaign_needed without LTV signal

---

### Case 01 — RFC Argmax Direct
**File:** `ml/cases/case_01/pipeline_ml_case01.py`
**Notes:** `ml/cases/case_01/notes_case01.md`

**Change:** Remove HYP segmentation. RFC 4-class probabilities
decide label directly via argmax.

**Result:**
- ML accuracy: 40.8% all / 61.0% mature cuts
- Label distribution now matches ground truth perfectly (in-sample)
- RFC wins on 3 of 4 labels by FN rate vs Claude
- single failure: no_campaign_impact FN=78.1% (vs Claude 26.5%)

**Root cause identified:**
- RFC trains on lifetime labels, backtest evaluates on window labels
- Customers who are lifetime send_campaign buyers appear as
  no_campaign_impact in early windows (didn't purchase that period)
- Cuts 1-2: 93.8% window GT is no_campaign_impact — campaign immature
- Static features carry forward-looking signal into early windows

**Decision:** Option C — exclude Cuts 1-2 from primary metric.
Mature window accuracy: ML=61.0% vs Claude=64.6% (gap=3.6%).
Proceed to Case 02: per-window feature recomputation.

---

### Case 02 — Per-Window Feature Recomputation (planned)
**File:** `ml/cases/case_02/pipeline_ml_case02.py`
**Notes:** `ml/cases/case_02/notes_case02.md`

**Change:** True walk-forward evaluation.
For each temporal cut:
  1. Recompute all 10 canonical features from training window only
  2. Retrain RFC on windowed features + window ground truth labels
  3. Predict on test window using test-window-computed features

**Hypothesis:**
  Eliminates static feature / dynamic label mismatch.
  Cuts 1-2 accuracy should improve significantly.
  Overall mean accuracy rises, gap to Claude narrows or inverts.
  no_campaign_impact FN rate should drop.

**Expected:** ML mature accuracy 65%+ (vs Claude 64.6%)
**Result:** TBD

---

### Case 03 — Rolling Window Retrain (planned)
**File:** `ml/cases/case_03/pipeline_ml_case03.py`

**Change:** Retrain RFC on rolling 12-month window per cut
rather than full historical data. Recent behaviour weighted more.

**Hypothesis:** Reduces concept drift across temporal cuts.
**Expected:** Accuracy degradation from Cut 1→4 reduced.
**Result:** TBD

---

### Case 04 — Confidence-Based Routing Ensemble (planned)
**File:** `ml/cases/case_04/pipeline_ml_case04.py`

**Change:** Route low-confidence ML predictions to Claude.
  RFC confidence > 0.80 → ML label
  RFC confidence < 0.60 → Claude label
  Middle band → ML label + flag for review

**Hypothesis:** Ensemble outperforms either pipeline alone.
**Expected:** Combined accuracy > 66.5% (Claude benchmark)
**Result:** TBD

---

## Key Findings Across Cases

### Feature importance (stable across cases)
```
recency_days                    38.5%  ← dominant signal
campaign_influence_rate         22.6%  ← campaign dependency
spend_last_12_months            11.3%  ← annual spend window
spend_last_3_months              7.3%  ← recent activity
stddev_days_campaign_to_purchase 6.0%  ← response consistency (new)
total_campaigns_received         4.4%  ← campaign exposure
avg_days_campaign_to_purchase    3.9%  ← response speed (new)
email_open_rate                  2.9%  ← email engagement
months_since_customer            1.6%  ← customer tenure
frequency_months                 1.6%  ← purchase cadence
```

### ML vs Claude FN rates (case_01, mature cuts)
```
Label               ML FN    Claude FN    Winner
────────────────────────────────────────────────
send_campaign        0.0%      23.8%      ML
dont_send            0.0%       9.7%      ML
no_campaign_needed   9.2%      37.5%      ML
no_campaign_impact  78.1%      26.5%      Claude
```

ML wins on 3 of 4 labels. Claude wins on no_campaign_impact.
Combined ensemble (Case 04) is the logical next step.

### The honest summary
```
ML is better at identifying WHO customers are (lifetime behaviour).
Claude is better at predicting WHAT they will do in a given window.
Neither is complete. Case 02 attempts to close that gap for ML.
```
