#!/usr/bin/env python3
"""
agents/pipeline_claude.py — Campaign Send Likelihood: Claude Pipeline (v2)

PURPOSE
-------
Predict campaign send likelihood for each customer using Claude AI
with directional rules derived from the canonical feature set.
Parallel to ml/pipeline_ml.py for direct comparison.

This pipeline is one half of a parallel comparison:
  ml/pipeline_ml.py          ← statistical predictions (RFC + KMeans++)
  agents/pipeline_claude.py  ← this file (Claude rule-based + reasoning)

FEATURE CONTRACT
----------------
Uses ONLY the 10 canonical features from
pipeline_v2/02_feature_canonical.json — identical to pipeline_ml.py.
No additional features. No extra context.
This ensures the comparison is feature-for-feature like-for-like.

CLAUDE'S ROLE
-------------
Claude receives:
  - The 10 canonical feature values for one customer
  - Directional guidance on what each feature signals
    (not hard thresholds — direction and context only)
  - The 4 label definitions and their business meaning
  - Instructions to reason about feature combinations

Claude decides:
  - Which label fits best
  - How confident it is (based on data richness and signal clarity)
  - Why — a 2-3 sentence reasoning string referencing specific values
  - Which feature combinations drove the decision

This gives Claude full autonomy on:
  - Threshold calibration (it sets its own from the data presented)
  - Feature combination logic (it reasons about interactions)
  - Confidence scoring (it weights certainty by data richness)
  - Edge case handling (low data, conflicting signals)

GROUND TRUTH (for backtest_compare.py)
---------------------------------------
Same derivation as ml/pipeline_ml.py:
  dont_send          → recency_days < 14
  send_campaign      → ≥50% of campaign-era orders were influenced
  no_campaign_needed → has orders but 0 were campaign-influenced
  no_campaign_impact → no campaign-era orders

Both pipelines evaluated against identical holdout sets in
pipeline_v2/backtest_compare.py.

Usage
    python agents/pipeline_claude.py                # full run (all customers)
    python agents/pipeline_claude.py --dry-run      # report, no DB writes
    python agents/pipeline_claude.py --limit 20     # first N customers
    python agents/pipeline_claude.py --customer-id X # one customer
    python agents/pipeline_claude.py --port 5433    # override DB port
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

SCRIPT_DIR     = Path(__file__).resolve().parent
ROOT_DIR       = SCRIPT_DIR.parent
CANONICAL_PATH = ROOT_DIR / "pipeline_v2" / "02_feature_canonical.json"

sys.path.insert(0, str(ROOT_DIR / "data_generator"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import anthropic
import pandas as pd
import numpy as np

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    sys.exit("✗ ANTHROPIC_API_KEY not set. Add it to your .env file.")

VALID_LABELS = {
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
}

UPSERT_SQL = """
INSERT INTO customer_recommendations_claude (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
) VALUES (%s, %s, %s, %s, %s, NOW(), 'claude-v2-pipeline')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""


# ══════════════════════════════════════════════════════════════════════════
# Helpers
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


def load_canonical() -> list[str]:
    if not CANONICAL_PATH.exists():
        sys.exit(f"✗ {CANONICAL_PATH} not found. "
                 "Run pipeline_v2/01_feature_analysis.py first.")
    meta  = json.load(open(CANONICAL_PATH))
    cols  = [f["column"] for f in meta["features"]]
    descs = {f["column"]: f["description"] for f in meta["features"]}
    return cols, descs


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════
def load_data(conn, feature_cols: list,
              customer_id: str = None) -> pd.DataFrame:
    extra = ["customer_id", "customer_type", "attribution_segment",
             "total_orders", "total_revenue"]
    all_cols = list(dict.fromkeys(feature_cols + extra))
    where    = f"AND customer_id = '{customer_id}'" if customer_id else ""
    query    = f"""
        SELECT DISTINCT ON (customer_id)
            {', '.join(all_cols)}
        FROM customer_derivatives
        WHERE TRUE {where}
        ORDER BY customer_id, run_id DESC
    """
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df):,} customers")
    return df


# ══════════════════════════════════════════════════════════════════════════
# System prompt
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a campaign send likelihood classifier for a specialty coffee and beverage merchant.

Your task: analyse a customer's behavioural features and assign ONE of four labels.

LABEL DEFINITIONS:
  send_campaign      Customer's purchases are demonstrably driven by campaigns.
                     Sending the next scheduled campaign is likely to generate a purchase.

  dont_send          Customer purchased very recently.
                     Interrupting with a campaign risks fatigue and unsubscribe.
                     Re-evaluate in 2 weeks.

  no_campaign_needed Customer purchases organically — campaigns do not appear to drive their orders.
                     They are brand-loyal. Protect them from regular cadence.
                     Merchant should send NPIs, exclusive loyalty offers, seasonal launches only.

  no_campaign_impact Customer shows low or zero response to campaigns despite receiving many.
                     Sending more campaigns is unlikely to generate a purchase.
                     Suppress or route to 90-day re-engagement flow.

FEATURE SIGNALS (directional guidance — use your judgment on thresholds):
  recency_days
    Very low (< 7)  → strong dont_send signal regardless of other features
    Low (7-14)      → consider dont_send
    High (> 90)     → customer is drifting, consider no_campaign_impact

  campaign_influence_rate (0-1)
    High (> 0.60)   → strong send_campaign signal
    Low (< 0.15)    → suggests organic buyer → no_campaign_needed if purchasing
    Zero            → no campaign correlation at all

  avg_days_campaign_to_purchase
    Very low (< 3 days)  → fast, responsive buyer → confirms send_campaign
    High (> 20 days)     → slow converter — campaigns work but timing matters
    NULL                 → no campaign-influenced orders (organic or ghost)

  stddev_days_campaign_to_purchase
    Low or zero     → consistent, predictable responder → higher confidence
    High (> 8)      → erratic response timing → lower confidence
    NULL            → same as above

  total_campaigns_received
    High with low influence_rate → campaigns not working → no_campaign_impact
    Low (< 5)       → insufficient data → be conservative with confidence

  recency + lag combination (KEY INTERACTION):
    If recency_days > avg_days_campaign_to_purchase:
      Customer is approaching their next expected purchase window
      → stronger case for send_campaign
    If recency_days < avg_days_campaign_to_purchase:
      Too early in their cycle → dont_send or hold

  spend trends (spend_last_3_months vs spend_last_12_months):
    spend_3m > spend_last_12m/4  → accelerating spend → send_campaign signal
    spend_3m ≈ 0 but spend_12m > 0 → recent inactivity → check recency

  frequency_months
    Low (< 1.5 months) → very frequent buyer → dont_send risk higher
    High (> 6 months)  → infrequent → don't over-send

CONFIDENCE GUIDANCE:
  High (0.80-0.99):
    Clear signal from 2+ features pointing the same direction
    ≥ 3 campaign-era orders with consistent pattern
    Both influence_rate AND lag features confirm the label

  Medium (0.60-0.79):
    Mixed signals OR sparse data (< 3 campaign-era orders)
    One strong feature but others are ambiguous

  Low (0.40-0.59):
    Conflicting signals OR very few campaigns received (< 5)
    Cannot determine pattern from available data

IMPORTANT:
  - Reason about COMBINATIONS of features, not single thresholds
  - Reference specific numeric values in your reasoning
  - The lag features (avg_days, stddev_days) are new — use them
    to add precision to send_campaign vs dont_send decisions
  - You MUST call submit_recommendation. Do not respond with free text."""

# ══════════════════════════════════════════════════════════════════════════
# Tool schema
# ══════════════════════════════════════════════════════════════════════════
RECOMMENDATION_TOOL = {
    "name": "submit_recommendation",
    "description": (
        "Submit your campaign send likelihood classification. "
        "You MUST call this tool with all required fields."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type":        "string",
                "enum":        list(VALID_LABELS),
                "description": "Your classification label.",
            },
            "confidence": {
                "type":        "number",
                "minimum":     0.0,
                "maximum":     1.0,
                "description": "Confidence in this label (0=uncertain, 1=certain).",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2-3 sentences referencing specific feature values. "
                    "Explain which feature combinations drove the decision. "
                    "This text appears in the merchant dashboard."
                ),
            },
            "key_signals": {
                "type":  "array",
                "items": {"type": "string"},
                "description": "2-4 bullet points naming the decisive signals.",
            },
        },
        "required": ["label", "confidence", "reasoning", "key_signals"],
    },
}


# ══════════════════════════════════════════════════════════════════════════
# User message builder — canonical features only
# ══════════════════════════════════════════════════════════════════════════
def build_user_message(row: pd.Series, feature_cols: list,
                       feature_descs: dict) -> str:
    """
    Pass ONLY the 10 canonical features to Claude.
    Also pass non-feature context (orders, customer type) so Claude
    can reason about data richness and confidence.
    """
    def fmt(v, pct=False, dollar=False, days=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if pct:
            return f"{float(v) * 100:.1f}%"
        if dollar:
            return f"${float(v):.2f}"
        if days:
            return f"{float(v):.1f} days"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [
        "## Customer context (for confidence calibration only)",
        f"- Customer type    : {row.get('customer_type', 'N/A')}",
        f"- Total orders     : {fmt(row.get('total_orders'))}",
        f"- Attribution segment: {row.get('attribution_segment', 'N/A')}",
        "",
        "## Canonical features (10) — basis for classification",
    ]

    # Feature-by-feature with description
    for col in feature_cols:
        val  = row.get(col)
        desc = feature_descs.get(col, col)
        if col in ("campaign_influence_rate", "email_open_rate"):
            lines.append(f"- {col:<44}: {fmt(val, pct=True)}")
        elif "days" in col or "recency" in col:
            lines.append(f"- {col:<44}: {fmt(val, days=True)}")
        elif "spend" in col or "revenue" in col:
            lines.append(f"- {col:<44}: {fmt(val, dollar=True)}")
        else:
            lines.append(f"- {col:<44}: {fmt(val)}")

    # Key combination hint
    avg_lag  = row.get("avg_days_campaign_to_purchase")
    recency  = row.get("recency_days")
    if avg_lag and recency and not np.isnan(float(avg_lag or 0)):
        diff = float(recency) - float(avg_lag)
        if diff > 0:
            lines.append(
                f"\n  Note: recency ({fmt(recency, days=True)}) > "
                f"avg response lag ({fmt(avg_lag, days=True)}) "
                f"— customer may be in purchase window (+{diff:.0f} days past expected)"
            )
        else:
            lines.append(
                f"\n  Note: recency ({fmt(recency, days=True)}) < "
                f"avg response lag ({fmt(avg_lag, days=True)}) "
                f"— too early in purchase cycle ({abs(diff):.0f} days before expected)"
            )

    lines += [
        "",
        "Based on these features, call submit_recommendation with your "
        "label, confidence, reasoning (referencing specific values), "
        "and key_signals."
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Claude classification
# ══════════════════════════════════════════════════════════════════════════
def classify_one(client: anthropic.Anthropic,
                 row: pd.Series,
                 feature_cols: list,
                 feature_descs: dict) -> dict:
    """Classify a single customer. Returns result dict."""
    user_msg = build_user_message(row, feature_cols, feature_descs)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[RECOMMENDATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_recommendation"},
            messages=[{"role": "user", "content": user_msg}],
        )
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_block:
            return {"error": "Claude did not call the tool"}

        result = tool_block.input
        label  = result.get("label", "")
        if label not in VALID_LABELS:
            return {"error": f"Invalid label: {label!r}"}

        return {
            "label":       label,
            "confidence":  float(result["confidence"]),
            "reasoning":   result["reasoning"],
            "key_signals": result.get("key_signals", []),
        }

    except anthropic.APIError as exc:
        return {"error": f"API error: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Batch run
# ══════════════════════════════════════════════════════════════════════════
def run_batch(df: pd.DataFrame, feature_cols: list,
              feature_descs: dict, dry_run: bool,
              conn=None, delay: float = 0.35) -> pd.DataFrame:
    client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    errors  = []
    counts  = Counter()

    print(f"\nClassifying {len(df):,} customers …")
    print(f"{'─' * 64}")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        cid = row["customer_id"]
        print(f"  [{i:>3}/{len(df)}] {cid}  ", end="", flush=True)

        result = classify_one(client, row, feature_cols, feature_descs)

        if "error" in result:
            print(f"✗ {result['error'][:60]}")
            errors.append((cid, result["error"]))
            counts["error"] += 1
            results.append({
                "customer_id": cid,
                "label": None,
                "confidence": None,
                "reasoning": None,
                "key_signals": [],
                "error": result["error"],
            })
        else:
            label = result["label"]
            conf  = result["confidence"]
            print(f"→ {label:<22}  conf={conf:.2f}")
            counts[label] += 1
            results.append({
                "customer_id": cid,
                "label":       label,
                "confidence":  conf,
                "reasoning":   result["reasoning"],
                "key_signals": result["key_signals"],
                "error":       None,
            })

            if not dry_run and conn:
                import psycopg2.extras
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(UPSERT_SQL, (
                            cid, label, conf,
                            result["reasoning"],
                            json.dumps(result["key_signals"]),
                        ))

        if i < len(df):
            time.sleep(delay)

    result_df = pd.DataFrame(results)

    # Summary
    total = len(df)
    print(f"\n{'═' * 64}")
    print(f"  Claude Pipeline v2 — Results ({total} customers)")
    print(f"{'═' * 64}")
    label_order = ["send_campaign", "dont_send",
                   "no_campaign_needed", "no_campaign_impact", "error"]
    for label in label_order:
        n = counts.get(label, 0)
        if n:
            pct = n * 100 / total
            bar = "█" * int(pct / 2)
            print(f"  {label:<26} {n:>5}  ({pct:.1f}%)  {bar}")

    successful = result_df[result_df["error"].isna()]
    if len(successful) > 0:
        avg_conf = successful["confidence"].mean()
        print(f"\n  Avg confidence (successful): {avg_conf:.3f}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for cid, err in errors[:5]:
            print(f"    {cid}: {err[:70]}")

    print(f"{'═' * 64}\n")
    return result_df


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Claude pipeline v2 — campaign send likelihood.")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Classify and report, no DB writes")
    ap.add_argument("--resume",      action="store_true",
                    help="Skip customers already in customer_recommendations_claude")
    ap.add_argument("--limit",       type=int,
                    help="Classify first N customers only")
    ap.add_argument("--customer-id", help="Classify one specific customer")
    ap.add_argument("--host",        help="DB host (default: localhost)")
    ap.add_argument("--port",        help="DB port (default: 5433)")
    args = ap.parse_args()

    print(f"\n{'═' * 64}")
    print(f"  Claude Pipeline v2 — Campaign Send Likelihood")
    print(f"{'═' * 64}")

    print("\nLoading canonical feature set …")
    feature_cols, feature_descs = load_canonical()
    print(f"  {len(feature_cols)} features: {', '.join(feature_cols)}")

    conn = get_conn(args)
    print(f"✓ Connected\n")

    print("Loading customer data …")
    df = load_data(conn, feature_cols, args.customer_id)

    if args.resume:
        done_df = pd.read_sql(
            "SELECT customer_id::text FROM customer_recommendations_claude",
            conn
        )
        done_ids = set(done_df["customer_id"].astype(str).values)
        before = len(df)
        df = df[~df["customer_id"].astype(str).isin(done_ids)]
        print(f"  Resume mode: {before - len(df)} already done, "
              f"{len(df)} remaining")

    if args.limit:
        df = df.head(args.limit)
        print(f"  Limited to {args.limit} customers")

    result_df = run_batch(
        df, feature_cols, feature_descs,
        dry_run=args.dry_run,
        conn=None if args.dry_run else conn,
    )

    # Save results
    out_dir = ROOT_DIR / "ml" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "claude_v2_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"✓ Results saved → {out_path}")

    if args.dry_run:
        print("Dry run — no DB writes.")

    conn.close()
    print("\nDone. Run pipeline_v2/backtest_compare.py to evaluate accuracy.")


if __name__ == "__main__":
    main()
