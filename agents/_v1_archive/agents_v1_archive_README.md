# agents/_v1_archive/

Files in this folder are the **Pass 1, 2, and 3** iterations of the
recommendation agent — the first working version of the Claude-based
classification pipeline built on 2026-09-17/18.

## What these files do

| File | Description |
|---|---|
| `recommendation_agent_pass1.py` | Graph skeleton — no DB, no Claude. Demonstrates LangGraph state flow. |
| `recommendation_agent_pass2.py` | Real DB fetch + real Claude API call. write_node still stub. |
| `recommendation_agent.py` | Pass 3 — full pipeline. 4-label schema, batch runner, DB writes. |

## Why archived

The v1 agent uses a **rule-based classification system**:
- Claude applies hard-coded threshold rules from a system prompt
- "Confidence" is a data-volume proxy, not a calibrated probability
- Feature set not formally reconciled with the ML pipeline
- No backtesting or cross-validation against actual purchase outcomes

These are not flaws in the implementation — the agent works correctly
and the reasoning output is genuinely useful. The archive reflects a
methodological decision to build a more rigorous v2 where:

1. Feature set is the same for both Claude and ML pipelines
2. Both pipelines are backtested against the same holdout sets
3. Accuracy is measured independently per pipeline against ground truth
4. Claude's role is reasoning and explainability, not classification

## What's coming in agents/

`pipeline_claude.py` — the v2 Claude pipeline:
- Uses only the canonical feature set from `pipeline_v2/02_feature_canonical.json`
- Rules derived from those features only (no extras)
- Same temporal holdout sets as the ML pipeline
- Accuracy measured against actual purchase outcomes
- Claude's reasoning layer sits on top of ML prediction likelihoods

See `pipeline_v2/` for the shared feature analysis and backtest framework.
