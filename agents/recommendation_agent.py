#!/usr/bin/env python3
"""
agents/recommendation_agent.py — PASS 3 (production): full agent.

Changes from Pass 2:
  - Four-label schema: send_campaign | dont_send |
                       no_campaign_needed | no_campaign_impact
  - write_node: real INSERT into customer_recommendations (idempotent)
  - Batch mode: run all customers in customer_derivatives
  - Progress reporting + summary table at the end

Label semantics (settled in design session 2026-09-18):
  send_campaign      Campaign-driven buyer. Send on regular cadence.
  dont_send          Just purchased (recency < 7d). Give them space.
  no_campaign_needed Organic / brand-loyal buyer. Buys without campaigns.
                     Merchant action: NPIs, loyalty offers, seasonal only —
                     NOT the regular newsletter cadence.
  no_campaign_impact Ghost subscriber or prospect. Campaigns not converting.
                     Merchant action: suppress or route to re-engagement (90d).

Usage
    python agents/recommendation_agent.py                    # one customer (top revenue)
    python agents/recommendation_agent.py --customer-id <id> # one specific customer
    python agents/recommendation_agent.py --batch            # all 496 customers
    python agents/recommendation_agent.py --batch --limit 20 # first N customers
    python agents/recommendation_agent.py --list-customers   # browse the pool
    python agents/recommendation_agent.py --summary          # show results table

CCAR-F domains demonstrated:
    D1A  Agentic architecture: StateGraph, typed state, node contracts
    D1B  Orchestration: linear graph, error propagation via state
    D1D  Guardrails: label validation, error state, skip-on-error pattern
    D2A  Structured output: tool_use forces valid JSON schema
    D4A  System prompt: role, four-label rules, output format
    D4B  User message: 12-field context, not 39-column raw row
    D5A  Context window discipline: only decision-relevant fields to Claude
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import TypedDict, Optional
from pathlib import Path

from langgraph.graph import StateGraph, START, END

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "data_generator"))

try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR.parent / ".env")
except ImportError:
    pass

import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    sys.exit("✗ ANTHROPIC_API_KEY not set. Add it to your .env file.")

# ── Four valid labels ─────────────────────────────────────────────────────
VALID_LABELS = {
    "send_campaign",
    "dont_send",
    "no_campaign_needed",
    "no_campaign_impact",
}


# ══════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════
class RecommendationState(TypedDict):
    customer_id:   str
    derivatives:   Optional[dict]
    attribution:   Optional[list]
    label:         Optional[str]
    confidence:    Optional[float]
    reasoning:     Optional[str]
    key_signals:   Optional[list]
    written:       Optional[bool]
    error:         Optional[str]


# ══════════════════════════════════════════════════════════════════════════
# DB helper
# ══════════════════════════════════════════════════════════════════════════
def get_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )


# ══════════════════════════════════════════════════════════════════════════
# NODE 1 — fetch_node  (unchanged from Pass 2)
# ══════════════════════════════════════════════════════════════════════════
DERIVATIVES_SQL = """
    SELECT
        customer_id, customer_type, total_orders, total_revenue,
        average_order_value, recency_days, frequency_months,
        total_campaigns_received, email_open_rate, email_click_rate,
        avg_campaigns_before_purchase, campaign_influence_rate,
        attribution_segment, days_since_last_campaign,
        spend_last_3_months, spend_last_12_months,
        total_purchases_with_campaign, total_purchases_without_campaign
    FROM customer_derivatives
    WHERE customer_id = %s
    ORDER BY run_id DESC LIMIT 1
"""
ATTRIBUTION_SQL = """
    SELECT order_id, order_date, order_revenue,
           order_number_in_lifecycle, campaigns_in_30d_window,
           campaign_influenced, attribution_bucket
    FROM campaign_attribution
    WHERE customer_id = %s
      AND attribution_bucket != 'pre_campaign_era'
    ORDER BY order_date DESC LIMIT 5
"""

def fetch_node(state: RecommendationState) -> dict:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(DERIVATIVES_SQL, (state["customer_id"],))
            cols = [d[0] for d in cur.description]
            row  = cur.fetchone()
            if not row:
                return {"error": f"No derivatives row for {state['customer_id']}"}
            derivatives = dict(zip(cols, row))
            cur.execute(ATTRIBUTION_SQL, (state["customer_id"],))
            cols2      = [d[0] for d in cur.description]
            attribution = [dict(zip(cols2, r)) for r in cur.fetchall()]
        conn.close()
        return {"derivatives": derivatives, "attribution": attribution}
    except Exception as exc:
        return {"error": f"fetch_node: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# NODE 2 — classify_node  (four-label schema)
# ══════════════════════════════════════════════════════════════════════════
RECOMMENDATION_TOOL = {
    "name": "submit_recommendation",
    "description": (
        "Submit your campaign recommendation for this customer. "
        "You MUST call this tool. Do not respond with free text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "send_campaign",
                    "dont_send",
                    "no_campaign_needed",
                    "no_campaign_impact",
                ],
                "description": (
                    "send_campaign     : campaigns demonstrably drive this customer's purchases. "
                    "dont_send         : customer purchased very recently — give them space. "
                    "no_campaign_needed: organic/brand-loyal buyer — buys without campaigns. "
                    "                    Merchant will send NPIs, loyalty offers, seasonal only. "
                    "no_campaign_impact: ghost subscriber or prospect — campaigns not converting. "
                    "                    Merchant will suppress or route to re-engagement (90d)."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Confidence in this label. "
                    "Higher when more orders + more campaign history are available. "
                    "Lower when data is sparse (< 3 campaigns or < 2 orders)."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2-3 sentences explaining your recommendation. "
                    "Reference specific numbers from the customer data. "
                    "This text will appear in the merchant dashboard."
                ),
            },
            "key_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 bullet points naming the data signals that drove your decision.",
            },
        },
        "required": ["label", "confidence", "reasoning", "key_signals"],
    },
}

SYSTEM_PROMPT = """You are a campaign recommendation engine for a specialty coffee and beverage merchant (abc Roasters).

Your job: analyse one customer's purchase and email engagement history and assign one of four labels.

LABEL DEFINITIONS:
  send_campaign      Campaigns demonstrably drive this customer's purchases.
                     Use when campaign_influence_rate >= 0.6 AND avg_campaigns_before_purchase <= 5.
  dont_send          Customer purchased very recently. Give them space.
                     Use when recency_days < 7 AND total_orders > 1.
  no_campaign_needed Customer is an organic / brand-loyal buyer.
                     They purchase WITHOUT needing a campaign nudge.
                     Merchant action: send NPIs, exclusive loyalty offers, seasonal launches only.
                     NOT the regular newsletter cadence.
                     Use when total_orders > 0 AND campaign_influence_rate < 0.15
                     AND total_purchases_without_campaign >= total_purchases_with_campaign.
  no_campaign_impact Ghost subscriber or prospect. Campaigns are not converting.
                     Merchant action: suppress or route to 90-day re-engagement flow.
                     Use when total_orders = 0
                     OR (total_campaigns_received >= 10 AND email_open_rate < 0.10)
                     OR (total_campaigns_received >= 5 AND campaign_influence_rate = 0 AND total_orders < 2).

DECISION ORDER (apply top to bottom, stop at first match):
  1. recency_days < 7 AND total_orders > 1                           → dont_send
  2. total_orders = 0                                                 → no_campaign_impact
  3. total_orders > 0 AND campaign_influence_rate IS NULL             → no_campaign_needed
  4. total_campaigns_received >= 10 AND email_open_rate < 0.10       → no_campaign_impact
  5. campaign_influence_rate >= 0.6 AND avg_campaigns_before_purchase <= 5 → send_campaign
  6. total_orders > 0 AND campaign_influence_rate < 0.15
     AND purchases_without >= purchases_with                          → no_campaign_needed
  7. Use judgment for all remaining cases, referencing the data signals.

CONFIDENCE GUIDANCE:
  High (0.80-0.99): >= 5 campaign-era orders, clear influence pattern
  Medium (0.60-0.79): 2-4 campaign-era orders, mixed signals
  Low (0.40-0.59): < 2 campaign-era orders, data too sparse for certainty

IMPORTANT:
  - Be specific. Reference actual numbers, not vague statements.
  - For no_campaign_needed customers, acknowledge their loyalty value explicitly.
  - You must call submit_recommendation. Do not respond with free text."""


def build_user_message(derivatives: dict, attribution: list) -> str:
    d = derivatives

    def fmt(v, pct=False, dollar=False):
        if v is None:
            return "N/A"
        if pct:
            return f"{float(v) * 100:.1f}%"
        if dollar:
            return f"${float(v):.2f}"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [
        "## Customer profile",
        f"- Customer type           : {d.get('customer_type', 'N/A')}",
        f"- Attribution segment     : {d.get('attribution_segment', 'N/A')}",
        f"- Total orders            : {fmt(d.get('total_orders'))}",
        f"- Total revenue (LTV)     : {fmt(d.get('total_revenue'), dollar=True)}",
        f"- Average order value     : {fmt(d.get('average_order_value'), dollar=True)}",
        f"- Recency (days since last purchase): {fmt(d.get('recency_days'))} days",
        f"- Purchase frequency      : every {fmt(d.get('frequency_months'))} months",
        "",
        "## Email engagement",
        f"- Campaigns received      : {fmt(d.get('total_campaigns_received'))}",
        f"- Email open rate         : {fmt(d.get('email_open_rate'), pct=True)}",
        f"- Email click rate        : {fmt(d.get('email_click_rate'), pct=True)}",
        f"- Days since last campaign: {fmt(d.get('days_since_last_campaign'))} days",
        "",
        "## Campaign attribution",
        f"- Campaign influence rate : {fmt(d.get('campaign_influence_rate'), pct=True)}",
        f"- Avg campaigns before purchase   : {fmt(d.get('avg_campaigns_before_purchase'))}",
        f"- Orders WITH campaign in 30d window  : {fmt(d.get('total_purchases_with_campaign'))}",
        f"- Orders WITHOUT campaign in 30d window: {fmt(d.get('total_purchases_without_campaign'))}",
        "",
        "## Spend trend",
        f"- Spend last 3 months     : {fmt(d.get('spend_last_3_months'), dollar=True)}",
        f"- Spend last 12 months    : {fmt(d.get('spend_last_12_months'), dollar=True)}",
    ]

    if attribution:
        lines += ["", "## Recent campaign-era orders (newest first)"]
        for a in attribution:
            rev  = f"${float(a.get('order_revenue') or 0):.2f}"
            camp = a.get('campaigns_in_30d_window', 0)
            inf  = "influenced" if a.get('campaign_influenced') else "organic"
            bkt  = a.get('attribution_bucket', 'N/A')
            n    = a.get('order_number_in_lifecycle', '?')
            lines.append(
                f"  Order #{n}: {rev} | {inf} | "
                f"{camp} campaigns in 30d window | bucket={bkt}"
            )

    lines += [
        "",
        "Based on this data, call submit_recommendation with your label, "
        "confidence, reasoning, and key signals.",
    ]
    return "\n".join(lines)


def classify_node(state: RecommendationState) -> dict:
    if state.get("error"):
        return {"error": state.get("error")}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_message = build_user_message(
        state["derivatives"], state["attribution"] or []
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[RECOMMENDATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_recommendation"},
            messages=[{"role": "user", "content": user_message}],
        )
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_block:
            return {"error": "classify_node: Claude did not call the tool"}

        result = tool_block.input
        label  = result.get("label", "")

        # ── Guardrail: validate label (D1D) ──────────────────────────────
        if label not in VALID_LABELS:
            return {"error": f"classify_node: invalid label {label!r}"}

        return {
            "label":       label,
            "confidence":  float(result["confidence"]),
            "reasoning":   result["reasoning"],
            "key_signals": result.get("key_signals", []),
        }

    except anthropic.APIError as exc:
        return {"error": f"classify_node: API error — {exc}"}
    except Exception as exc:
        return {"error": f"classify_node: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# NODE 3 — write_node  (real DB write)
# ══════════════════════════════════════════════════════════════════════════
UPSERT_SQL = """
INSERT INTO customer_recommendations (
    customer_id, recommendation_label, confidence_score,
    reasoning, key_signals, recommended_at, model_version
)
VALUES (%s, %s, %s, %s, %s, NOW(), 'claude-sonnet-4-6-pass3')
ON CONFLICT (customer_id) DO UPDATE SET
    recommendation_label = EXCLUDED.recommendation_label,
    confidence_score     = EXCLUDED.confidence_score,
    reasoning            = EXCLUDED.reasoning,
    key_signals          = EXCLUDED.key_signals,
    recommended_at       = NOW(),
    model_version        = EXCLUDED.model_version
"""

def write_node(state: RecommendationState) -> dict:
    if state.get("error"):
        return {"error": state.get("error")}
    try:
        import json
        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(UPSERT_SQL, (
                    state["customer_id"],
                    state["label"],
                    state["confidence"],
                    state["reasoning"],
                    json.dumps(state.get("key_signals") or []),
                ))
        conn.close()
        return {"written": True}
    except Exception as exc:
        return {"error": f"write_node: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(RecommendationState)
    graph.add_node("fetch",    fetch_node)
    graph.add_node("classify", classify_node)
    graph.add_node("write",    write_node)
    graph.add_edge(START,      "fetch")
    graph.add_edge("fetch",    "classify")
    graph.add_edge("classify", "write")
    graph.add_edge("write",    END)
    return graph.compile()


def run_one(app, customer_id: str, verbose: bool = True) -> dict:
    initial: RecommendationState = {
        "customer_id": customer_id,
        "derivatives": None,
        "attribution": None,
        "label":       None,
        "confidence":  None,
        "reasoning":   None,
        "key_signals": None,
        "written":     None,
        "error":       None,
    }
    final = app.invoke(initial)
    if verbose:
        _print_result(final)
    return final


def _print_result(final: dict) -> None:
    print("\n" + "═" * 60)
    print("  RESULT")
    print("═" * 60)
    if final.get("error"):
        print(f"  ✗ Error: {final['error']}")
        return
    d = final["derivatives"]
    print(f"  Customer     : {final['customer_id']}")
    print(f"  Type/Segment : {d.get('customer_type')} / {d.get('attribution_segment')}")
    print(f"  Orders / LTV : {d.get('total_orders')} / ${float(d.get('total_revenue') or 0):.2f}")
    print(f"  Label        : {final['label']}")
    print(f"  Confidence   : {final['confidence']:.2f}")
    print(f"  Reasoning    :")
    for line in (final.get('reasoning') or '').split('. '):
        if line.strip():
            print(f"    {line.strip()}.")
    if final.get("key_signals"):
        print(f"  Key signals  :")
        for s in final["key_signals"]:
            print(f"    • {s}")
    print(f"  Written to DB: {final.get('written', False)}")


# ══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════
def run_batch(app, limit: int | None = None, delay: float = 0.3) -> None:
    """
    Run the agent for every customer in customer_derivatives.
    delay: seconds between API calls — avoids rate-limit bursts.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        q = """
            SELECT customer_id, attribution_segment, total_orders
            FROM customer_derivatives
            ORDER BY total_revenue DESC NULLS LAST
        """
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        rows = cur.fetchall()
    conn.close()

    total   = len(rows)
    results = Counter()
    errors  = []

    print(f"\nBatch run: {total} customers")
    print("─" * 60)

    for i, (customer_id, segment, orders) in enumerate(rows, start=1):
        print(f"  [{i:>3}/{total}] {customer_id}  seg={segment:<26}", end="  ", flush=True)
        final = run_one(app, customer_id, verbose=False)
        if final.get("error"):
            print(f"✗ {final['error'][:60]}")
            errors.append((customer_id, final["error"]))
            results["error"] += 1
        else:
            label = final.get("label", "unknown")
            conf  = final.get("confidence", 0)
            print(f"→ {label:<22} conf={conf:.2f}")
            results[label] += 1
        if i < total:
            time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Batch complete — {total} customers")
    print("═" * 60)
    label_order = ["send_campaign", "dont_send",
                   "no_campaign_needed", "no_campaign_impact", "error"]
    for label in label_order:
        n = results.get(label, 0)
        if n:
            pct = n * 100 / total
            print(f"  {label:<24} {n:>4}  ({pct:.1f}%)")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for cid, err in errors[:5]:
            print(f"    {cid}: {err[:80]}")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY VIEW
# ══════════════════════════════════════════════════════════════════════════
def show_summary() -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.recommendation_label,
                COUNT(*)                            AS customers,
                ROUND(AVG(d.total_revenue)::numeric, 2)  AS avg_ltv,
                ROUND(AVG(d.total_orders)::numeric, 1)   AS avg_orders,
                ROUND(AVG(r.confidence_score)::numeric, 2) AS avg_conf,
                ROUND(AVG(d.email_open_rate)::numeric * 100, 1) AS avg_open_pct
            FROM customer_recommendations r
            JOIN customer_derivatives d ON d.customer_id = r.customer_id
            GROUP BY r.recommendation_label
            ORDER BY avg_ltv DESC NULLS LAST
        """)
        rows = cur.fetchall()

        cur.execute("SELECT COUNT(*) FROM customer_recommendations")
        total = cur.fetchone()[0]
    conn.close()

    print(f"\n{'═' * 72}")
    print(f"  customer_recommendations — {total} rows")
    print(f"{'═' * 72}")
    print(f"  {'label':<24} {'cust':>5}  {'avg LTV':>8}  "
          f"{'avg ord':>7}  {'avg conf':>8}  {'open%':>6}")
    print(f"  {'─' * 65}")
    for label, n, ltv, orders, conf, opn in rows:
        print(f"  {label:<24} {n:>5}  ${ltv or 0:>7.2f}  "
              f"{orders or 0:>7.1f}  {conf or 0:>8.2f}  {opn or 0:>5.1f}%")
    print(f"{'═' * 72}\n")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def list_customers() -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT d.customer_id, d.customer_type, d.attribution_segment,
                   d.total_orders, ROUND(d.total_revenue::numeric, 2) rev
            FROM customer_derivatives d
            ORDER BY d.total_revenue DESC NULLS LAST
            LIMIT 15
        """)
        rows = cur.fetchall()
    conn.close()
    print(f"\n{'customer_id':<22} {'type':<22} {'segment':<26} "
          f"{'orders':>6} {'rev':>8}")
    print("─" * 90)
    for cid, ctype, seg, orders, rev in rows:
        print(f"{cid:<22} {ctype:<22} {seg:<26} {orders:>6} ${rev:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="merchant-abc recommendation agent (Pass 3 — production)")
    ap.add_argument("--customer-id",    help="Run for one specific customer_id")
    ap.add_argument("--batch",          action="store_true",
                    help="Run for all customers in customer_derivatives")
    ap.add_argument("--limit",          type=int,
                    help="Cap batch size (useful for testing)")
    ap.add_argument("--list-customers", action="store_true",
                    help="Print top-15 customers by revenue")
    ap.add_argument("--summary",        action="store_true",
                    help="Show customer_recommendations summary table")
    ap.add_argument("--port",           help="DB port (default: 5433)")
    args = ap.parse_args()

    if args.port:
        os.environ["SYNTHETIC_DB_PORT"] = args.port

    if args.list_customers:
        list_customers()
        return

    if args.summary:
        show_summary()
        return

    app = build_graph()

    if args.batch:
        run_batch(app, limit=args.limit)
        show_summary()
        return

    # Single customer
    customer_id = args.customer_id
    if not customer_id:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT customer_id FROM customer_derivatives
                WHERE attribution_segment = 'campaign_3_5'
                ORDER BY total_revenue DESC LIMIT 1
            """)
            row = cur.fetchone()
        conn.close()
        if not row:
            sys.exit("✗ No rows in customer_derivatives.")
        customer_id = row[0]
        print(f"No --customer-id given. Using top campaign_3_5 customer: {customer_id}")

    print("\n" + "═" * 60)
    print(f"  merchant-abc recommendation agent — Pass 3")
    print(f"  customer: {customer_id}")
    print("═" * 60)
    run_one(app, customer_id, verbose=True)


if __name__ == "__main__":
    main()
