# Case 04 — Confidence-Based ML + Claude Ensemble (HITL Routing)

## What this is
Not a new ML model. A routing layer that decides — per customer —
whether the ML prediction or Claude prediction is more reliable,
using RFC confidence score as the routing signal.

This is the agentic ensemble: statistical prediction (ML) +
contextual reasoning (Claude), routed by confidence.

## The orthogonality argument (thesis)

Traditional ML ensembles reduce error by combining multiple
statistical models trained on the same data. The error floor
is bounded by what the training data supports.

With agentic AI, the ensemble gains a fundamentally different
kind of judge — one that reasons about feature combinations
in context, applies business intuition that no training label
can encode, and explains its decisions in language a merchant
can act on.

ML failure modes:
  - Distributional edge cases not seen in training
  - Sparse training classes (Cuts 1-3 with only 2 classes)
  - dont_send FN=74% (lifetime signal overrides recency)
  - Conservative Claude routing suppresses some send_campaign

Claude failure modes:
  - Statistical patterns it cannot infer from rules
  - No LTV-based rank ordering
  - send_campaign FN=48% on Cut 4
  - no_campaign_needed over-suppressed

These failure modes are LARGELY ORTHOGONAL.
This is exactly what makes the ensemble powerful.

## Routing logic (default thresholds)
```
RFC confidence >= 0.80  → ML label    (trust the model)
RFC confidence <  0.60  → Claude label (trust the reasoning)
0.60 <= conf < 0.80     → ML label + review flag
```

## Routing distribution (496 customers)
```
ml_high_conf (≥0.80)        : 187  (37.7%)  ML decides
claude_low_ml_conf (<0.60)  : 125  (25.2%)  Claude decides
ml_middle_band_flagged      : 184  (37.1%)  ML + review flag
ML ↔ Claude agreement       : 282/496 (56.9%)
```

## Results

### Accuracy per cut
```
Cut 1 (Jul-Dec 2022): Ensemble=0.585  ML=0.458  Claude=0.683  Winner=Claude
Cut 2 (Jan-Jun 2023): Ensemble=0.585  ML=0.458  Claude=0.683  Winner=Claude
Cut 3 (Jul-Dec 2023): Ensemble=0.631  ML=0.663  Claude=0.657  Winner=ML
Cut 4 (Jan-Jun 2024): Ensemble=0.798  ML=0.752  Claude=0.635  Winner=ENSEMBLE

All cuts mean : 65.0%
Mature (3-4)  : 71.5%  ← BEST across all cases, beats Claude 64.6%
Cut 4         : 79.8%  ← BEST single-cut result across entire journey
```

### FN rates (all cuts average)
```
Label               case_02    case_04    Claude    Best
─────────────────────────────────────────────────────────
send_campaign        0.0%      11.2%     23.8%     case_02
dont_send           74.0%      54.8%      9.7%     Claude
no_campaign_needed  51.0%      32.3%     37.5%     case_04 ✓
no_campaign_impact  24.0%      29.3%     26.5%     case_02
```

### Notable regression
send_campaign FN: 0% (case_02) → 11.2% (case_04)
Cause: 25% of customers routed to Claude include some true
send_campaign customers that Claude over-suppresses to
no_campaign_impact. The routing threshold may be set
too low (0.60) — some ML-uncertain send_campaign customers
are incorrectly handed to Claude.

### The case for stopping here (satisfactory)
```
Journey from case_01 → case_04:
  case_01 mature: 61.0%
  case_02 mature: 70.8%  (+9.8% — major improvement)
  case_04 mature: 71.5%  (+0.7% — marginal improvement)

Marginal gain from case_02 → case_04 = 0.7%
This is the diminishing returns signal.

Cut 4 tells a better story (75.2% → 79.8%, +4.6%)
but that is one cut — insufficient to declare breakthrough.
```

## Decision — SATISFACTORY ✓

**Declare case_04 as the final ML pipeline for this project.**

The ensemble is the right architecture. The gains are real.
Further tuning would chase marginal gains on 496 synthetic
customers — not worth the effort until real merchant data
is available.

---

## Next optimisation opportunities (for future return)

### Immediate next experiment
```
python ml/cases/case_04/pipeline_ml_case04.py --dry-run --high 0.85 --low 0.55
```
Tighter thresholds. Hypothesis: fewer customers routed to Claude
for dont_send and send_campaign → reduces send_campaign FN
regression while keeping dont_send improvement.

Expected: send_campaign FN drops from 11.2% back toward 5%
while mature accuracy stays above 71%.

### Case 05 — Threshold grid search (planned)
```
Try all combinations of:
  high_threshold: [0.75, 0.80, 0.85, 0.90]
  low_threshold:  [0.50, 0.55, 0.60, 0.65]

Find the Pareto-optimal threshold pair that maximises
mature accuracy while minimising send_campaign FN.
This is a 16-combination grid — 2 minutes to run.
```

### Case 06 — Feature-specific routing (planned)
```
Current routing: based on RFC confidence alone.
Better routing: IF ml_label == 'dont_send' → always Claude
               IF ml_label == 'send_campaign' AND conf >= 0.75 → ML
               ELSE → ensemble routing

This exploits the known label-specific strengths:
  Claude is always better on dont_send
  ML is always better on send_campaign (when confident)
```

### Case 07 — Real merchant data (the real test)
```
All synthetic data limitations resolved:
  - 3+ years of campaign history → all 4 classes in all cuts
  - Thousands of customers → stable distributions
  - Real email event timestamps → window-accurate email rates
  - Real purchase patterns → no synthetic correlation artifacts

Expected: mature accuracy > 80%, dont_send FN < 20%
```

---

## What to do when returning

1. Run the threshold experiment first:
   ```
   python ml/cases/case_04/pipeline_ml_case04.py --dry-run --high 0.85 --low 0.55
   ```
   If mature accuracy ≥ 71.5% AND send_campaign FN < 8%:
   → adopt new thresholds, run full, update README

2. If threshold tuning is insufficient:
   → build Case 05 (threshold grid search)

3. If returning with real merchant data:
   → rebuild from compute_derivatives.py with real data
   → all cases replay cleanly — the architecture is sound

---

## Known limitations at this stopping point

1. Synthetic data — 496 customers, 3-year window
   Real data would produce more stable class distributions

2. Email rates from customer_derivatives (lifetime, not windowed)
   email_events table only starts Jul 2023

3. Cuts 1-3 have only 2 training classes (campaign program immature)
   Cut 4 is the only production-realistic cut

4. dont_send FN remains high (54.8%) — the hardest label
   Claude is better (9.7%) but receives only 25% of customers

5. send_campaign FN regressed to 11.2% from 0% (case_02)
   Threshold tuning likely recovers most of this
