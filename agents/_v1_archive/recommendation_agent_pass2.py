#!/usr/bin/env python3
"""
agents/recommendation_agent_pass2.py — PASS 2: Real DB + Real Claude API.

What's new vs Pass 1:
  - fetch_node   : real Postgres query on customer_derivatives + campaign_attribution
  - classify_node: real Anthropic API call, structured JSON output via tool_use pattern
  - write_node   : still a stub (prints result, no DB write — that's Pass 3)

Run:
    python agents/recommendation_agent_pass2.py
        → runs against the first customer in customer_derivatives

    python agents/recommendation_agent_pass2.py --customer-id <id>
        → runs against a specific customer

    python agents/recommendation_agent_pass2.py --list-customers
        → prints 10 customer IDs you can test with

CCAR-F domains demonstrated in this pass:
    D4A — System prompt design: role, constraints, output schema
    D4B — User message construction: structured data, not raw SQL rows
    D2A — Structured output: tool_use pattern forces valid JSON schema
    D5A — Context window discipline: 12 fields passed, not 39
    D1D — Guardrails: label validation before state update
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

VALID_LABELS = {"send_campaign", "dont_send", "no_campaign_impact"}


# ══════════════════════════════════════════════════════════════════════════
# STATE  (identical to Pass 1 — the graph contract doesn't change)
# ══════════════════════════════════════════════════════════════════════════
class RecommendationState(TypedDict):
    customer_id:   str
    derivatives:   Optional[dict]
    attribution:   Optional[list]
    label:         Optional[str]
    confidence:    Optional[float]
    reasoning:     Optional[str]
    written:       Optional[bool]
    error:         Optional[str]


# ══════════════════════════════════════════════════════════════════════════
# DB helper  (shared by fetch_node and write_node)
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
# NODE 1 — fetch_node
# Reads the two tables that matter for classification.
# Context window discipline: we fetch all columns but pass only 12 to Claude.
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
    ORDER BY run_id DESC
    LIMIT 1
"""

ATTRIBUTION_SQL = """
    SELECT
        order_id, order_date, order_revenue,
        order_number_in_lifecycle, campaigns_in_30d_window,
        campaign_influenced, attribution_bucket
    FROM campaign_attribution
    WHERE customer_id = %s
      AND attribution_bucket != 'pre_campaign_era'
    ORDER BY order_date DESC
    LIMIT 5
"""


def fetch_node(state: RecommendationState) -> dict:
    print(f"\n[fetch_node] customer_id={state['customer_id']}")
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            # customer_derivatives row
            cur.execute(DERIVATIVES_SQL, (state["customer_id"],))
            cols = [d[0] for d in cur.description]
            row  = cur.fetchone()
            if not row:
                return {"error": f"No derivatives row found for {state['customer_id']}"}
            derivatives = dict(zip(cols, row))

            # Last 5 campaign-era orders with attribution context
            cur.execute(ATTRIBUTION_SQL, (state["customer_id"],))
            cols2 = [d[0] for d in cur.description]
            attribution = [dict(zip(cols2, r)) for r in cur.fetchall()]
        conn.close()

        print(f"[fetch_node] type={derivatives['customer_type']} | "
              f"segment={derivatives['attribution_segment']} | "
              f"orders={derivatives['total_orders']} | "
              f"recency={derivatives['recency_days']}d | "
              f"influence={float(derivatives['campaign_influence_rate'] or 0):.0%}")
        return {"derivatives": derivatives, "attribution": attribution}

    except Exception as exc:
        return {"error": f"fetch_node failed: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# NODE 2 — classify_node
# Calls Claude with a structured prompt and parses the JSON response.
#
# Structured output pattern (D2A, D4A, D4B):
#   We define a "tool" with a JSON schema. Claude is forced to call it —
#   meaning it cannot respond with free text, only valid structured JSON.
#   This is the safest way to get reliable structured output from Claude.
# ══════════════════════════════════════════════════════════════════════════

# ── Tool schema — the JSON structure Claude must return ───────────────────
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
                "enum": ["send_campaign", "dont_send", "no_campaign_impact"],
                "description": (
                    "send_campaign: evidence shows campaigns drive purchases for this customer. "
                    "dont_send: customer is active/recent enough that a campaign would be noise "
                    "or they are unresponsive to campaigns. "
                    "no_campaign_impact: insufficient campaign-era data to determine influence."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Your confidence in this label (0.0 = uncertain, 1.0 = certain).",
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

# ── System prompt — Claude's role and constraints (D4A) ───────────────────
SYSTEM_PROMPT = """You are a campaign recommendation engine for a specialty coffee and beverage merchant.

Your job is to analyse one customer's purchase and email engagement history and decide whether sending them the next marketing campaign is likely to drive a purchase.

The merchant runs 2–5 email campaigns per month: promotional discounts, seasonal launches, newsletters, reactivation sequences, and loyalty rewards.

DECISION RULES (apply in this order):
1. If campaign_influence_rate is NULL or total_campaigns_received < 3 → label = no_campaign_impact
2. If attribution_segment = 'never_purchased' → label = no_campaign_impact  
3. If recency_days < 7 AND total_orders > 1 → label = dont_send (just bought, give them space)
4. If campaign_influence_rate >= 0.6 AND avg_campaigns_before_purchase <= 5 → label = send_campaign
5. If email_open_rate < 0.10 AND total_campaigns_received >= 10 → label = dont_send (disengaged)
6. Use your judgment for all other cases, referencing the data signals

IMPORTANT:
- Be specific. Reference actual numbers, not vague statements.
- Confidence reflects data quality: more orders + more campaign history = higher confidence.
- You must call the submit_recommendation tool. Do not respond with free text."""

# ── User message builder — context window discipline (D5A) ────────────────
def build_user_message(derivatives: dict, attribution: list) -> str:
    """
    Pass only the 12 most decision-relevant fields to Claude.
    Passing all 39 columns would bloat the context window with nulls,
    internal IDs, and fields Claude cannot reason about (e.g. run_id).
    """
    d = derivatives

    def fmt(v, suffix=""):
        if v is None: return "N/A"
        if isinstance(v, float): return f"{v:.2f}{suffix}"
        return f"{v}{suffix}"

    lines = [
        "## Customer profile",
        f"- Customer type          : {d.get('customer_type', 'N/A')}",
        f"- Attribution segment    : {d.get('attribution_segment', 'N/A')}",
        f"- Total orders           : {fmt(d.get('total_orders'))}",
        f"- Total revenue (LTV)    : ${fmt(d.get('total_revenue'))}",
        f"- Average order value    : ${fmt(d.get('average_order_value'))}",
        f"- Recency (days since last purchase) : {fmt(d.get('recency_days'))} days",
        f"- Purchase frequency     : every {fmt(d.get('frequency_months'))} months",
        "",
        "## Email engagement",
        f"- Campaigns received     : {fmt(d.get('total_campaigns_received'))}",
        f"- Email open rate        : {fmt(float(d.get('email_open_rate') or 0) * 100)}%",
        f"- Email click rate       : {fmt(float(d.get('email_click_rate') or 0) * 100)}%",
        f"- Days since last campaign: {fmt(d.get('days_since_last_campaign'))} days",
        "",
        "## Campaign attribution (campaign-era orders only)",
        f"- Campaign influence rate : {fmt(float(d.get('campaign_influence_rate') or 0) * 100)}%",
        f"- Avg campaigns before purchase: {fmt(d.get('avg_campaigns_before_purchase'))}",
        f"- Orders WITH campaign in 30d window  : {fmt(d.get('total_purchases_with_campaign'))}",
        f"- Orders WITHOUT campaign in 30d window: {fmt(d.get('total_purchases_without_campaign'))}",
        "",
        "## Spend trend",
        f"- Spend last 3 months    : ${fmt(d.get('spend_last_3_months'))}",
        f"- Spend last 12 months   : ${fmt(d.get('spend_last_12_months'))}",
    ]

    if attribution:
        lines += ["", "## Recent order attribution (last 5 campaign-era orders, newest first)"]
        for a in attribution:
            rev  = f"${float(a.get('order_revenue') or 0):.2f}"
            camp = a.get('campaigns_in_30d_window', 0)
            inf  = "influenced" if a.get('campaign_influenced') else "organic"
            bkt  = a.get('attribution_bucket', 'N/A')
            n    = a.get('order_number_in_lifecycle', '?')
            lines.append(f"  Order #{n}: {rev} | {inf} | {camp} campaigns in window | bucket={bkt}")

    lines += [
        "",
        "Based on this data, call submit_recommendation with your label, "
        "confidence score, reasoning, and key signals.",
    ]
    return "\n".join(lines)


def classify_node(state: RecommendationState) -> dict:
    if state.get("error"):
        print(f"\n[classify_node] skipping — upstream error: {state['error']}")
        return {}

    print(f"\n[classify_node] calling Claude API …")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = build_user_message(
        state["derivatives"],
        state["attribution"] or [],
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

        # Extract the tool_use block — guaranteed by tool_choice="tool"
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_block:
            return {"error": "classify_node: Claude did not call the tool"}

        result = tool_block.input   # already a dict — no JSON parsing needed

        # ── Guardrail: validate label (D1D) ──────────────────────────────
        label = result.get("label", "")
        if label not in VALID_LABELS:
            return {"error": f"classify_node: invalid label {label!r} returned by Claude"}

        print(f"[classify_node] label={label!r}  confidence={result.get('confidence'):.2f}")
        print(f"[classify_node] reasoning: {result.get('reasoning', '')[:120]} …")

        return {
            "label":      label,
            "confidence": float(result["confidence"]),
            "reasoning":  result["reasoning"],
        }

    except anthropic.APIError as exc:
        return {"error": f"classify_node: Anthropic API error — {exc}"}
    except Exception as exc:
        return {"error": f"classify_node: unexpected error — {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# NODE 3 — write_node  (STUB — real DB write comes in Pass 3)
# ══════════════════════════════════════════════════════════════════════════
def write_node(state: RecommendationState) -> dict:
    if state.get("error"):
        print(f"\n[write_node] skipping — upstream error: {state['error']}")
        return {}

    print(f"\n[write_node] STUB — would write to customer_recommendations:")
    print(f"  customer_id : {state['customer_id']}")
    print(f"  label       : {state['label']}")
    print(f"  confidence  : {state['confidence']:.2f}")
    print(f"  reasoning   : {state['reasoning'][:100]} …")
    return {"written": True}


# ══════════════════════════════════════════════════════════════════════════
# GRAPH  (identical structure to Pass 1 — only node internals changed)
# ══════════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(RecommendationState)
    graph.add_node("fetch",    fetch_node)
    graph.add_node("classify", classify_node)
    graph.add_node("write",    write_node)
    graph.add_edge(START,    "fetch")
    graph.add_edge("fetch",  "classify")
    graph.add_edge("classify", "write")
    graph.add_edge("write",  END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def list_customers():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT d.customer_id, d.customer_type, d.attribution_segment,
                   d.total_orders, ROUND(d.total_revenue::numeric, 2)
            FROM customer_derivatives d
            ORDER BY d.total_revenue DESC NULLS LAST
            LIMIT 15
        """)
        rows = cur.fetchall()
    conn.close()
    print(f"\n{'customer_id':<22} {'type':<22} {'segment':<26} {'orders':>6} {'rev':>8}")
    print("─" * 90)
    for cid, ctype, seg, orders, rev in rows:
        print(f"{cid:<22} {ctype:<22} {seg:<26} {orders:>6} ${rev:>7}")


def main():
    ap = argparse.ArgumentParser(
        description="Pass 2: real DB fetch + real Claude classification.")
    ap.add_argument("--customer-id",     help="Specific customer_id to classify")
    ap.add_argument("--list-customers",  action="store_true",
                    help="Print 15 customer IDs ranked by revenue")
    ap.add_argument("--port",            help="DB port (default: 5433)")
    args = ap.parse_args()

    if args.port:
        os.environ["SYNTHETIC_DB_PORT"] = args.port

    if args.list_customers:
        list_customers()
        return

    # Resolve customer_id
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
            sys.exit("✗ No rows in customer_derivatives. Run the pipeline first.")
        customer_id = row[0]
        print(f"No --customer-id given. Using highest-revenue campaign_3_5 customer: {customer_id}")

    app = build_graph()
    print("\n" + "═" * 57)
    print(f"  merchant-abc recommendation agent — Pass 2")
    print(f"  customer: {customer_id}")
    print("═" * 57)

    initial_state: RecommendationState = {
        "customer_id": customer_id,
        "derivatives": None,
        "attribution": None,
        "label":       None,
        "confidence":  None,
        "reasoning":   None,
        "written":     None,
        "error":       None,
    }

    final = app.invoke(initial_state)

    print("\n" + "═" * 57)
    print("  RESULT")
    print("═" * 57)
    if final.get("error"):
        print(f"  ✗ Error: {final['error']}")
    else:
        d = final["derivatives"]
        print(f"  Customer     : {final['customer_id']}")
        print(f"  Segment      : {d.get('attribution_segment')}")
        print(f"  Orders / LTV : {d.get('total_orders')} / ${float(d.get('total_revenue') or 0):.2f}")
        print(f"  Label        : {final['label']}")
        print(f"  Confidence   : {final['confidence']:.2f}")
        print(f"  Reasoning    :")
        for line in (final.get('reasoning') or '').split('. '):
            if line.strip():
                print(f"    {line.strip()}.")

    print("\nTo test other customers:")
    print("  python agents/recommendation_agent_pass2.py --list-customers")
    print("  python agents/recommendation_agent_pass2.py --customer-id <id>")


if __name__ == "__main__":
    main()
