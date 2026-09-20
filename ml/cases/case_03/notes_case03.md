# Case 03 — Post-Prediction Recency Override

## Problem from Case 02
dont_send FN = 77.4% across all 4 cuts.
Same 18 customers misclassified in every cut:
  Actual=dont_send → Predicted=send_campaign

These 18 customers have:
  - High campaign_influence_rate → RFC says send_campaign
  - recency_days < 14 → business rule says dont_send

The RFC weights campaign signal (22.6% importance) too
heavily against recency signal for these edge cases.

## Change
Post-prediction deterministic override AFTER RFC argmax:
  IF predicted_label = 'send_campaign'
  AND recency_days < 14
  → override to 'dont_send'

This is a business rule, not a learned threshold.
Always correct by definition — a recent purchaser
should never receive a campaign regardless of
their lifetime campaign response pattern.

## Hypothesis
  dont_send FN drops from 77.4% to near 0%
  send_campaign recall unchanged for all other customers
  Overall accuracy improves ~3-4%
  Mature cuts (3-4) accuracy improves from 70.8%

## Actual result
TBD — fill in after running

## Decision
TBD — update cases/README.md with findings
