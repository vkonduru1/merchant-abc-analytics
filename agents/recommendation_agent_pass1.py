#!/usr/bin/env python3
"""
agents/recommendation_agent.py — PASS 1: Graph skeleton.

No database. No Claude API call.
Three placeholder nodes that print what they receive and return fake data.
Goal: watch the execution model — state flows through nodes, each node
      sees what the previous one produced, END terminates cleanly.

Run:
    python agents/recommendation_agent_pass1.py

What you should see:
    [fetch_node]    received state keys: customer_id
    [fetch_node]    loaded derivatives and attribution for customer abc123
    [classify_node] received state keys: customer_id, derivatives, attribution
    [classify_node] would call Claude here — returning stub label
    [write_node]    received state keys: customer_id, derivatives, attribution, label, confidence, reasoning
    [write_node]    would write to customer_recommendations — stub complete
    ─────────────────────────────────────────────────────
    Final state:
      customer_id : abc123
      label       : send_campaign
      confidence  : 0.87
      reasoning   : STUB — Claude would explain here
      error       : None
    ─────────────────────────────────────────────────────

CCAR-F domains demonstrated in this pass:
    D1A — Agentic architecture: StateGraph, nodes, edges
    D1B — Orchestration: START → fetch → classify → write → END
    D5  — Context management: State typed dict carries context window across nodes
"""
from __future__ import annotations

from typing import TypedDict, Optional

# LangGraph imports — the only three you need for a basic graph
from langgraph.graph import StateGraph, START, END


# ══════════════════════════════════════════════════════════════════════════
# 1. STATE
# The typed dict that travels through every node.
# Every field that any node reads or writes must be declared here.
# ══════════════════════════════════════════════════════════════════════════
class RecommendationState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    customer_id:        str

    # ── Populated by fetch_node ────────────────────────────────────────────
    derivatives:        Optional[dict]   # one row from customer_derivatives
    attribution:        Optional[list]   # rows from campaign_attribution

    # ── Populated by classify_node ─────────────────────────────────────────
    label:              Optional[str]    # send_campaign | dont_send | no_campaign_impact
    confidence:         Optional[float]  # 0.0 → 1.0
    reasoning:          Optional[str]    # Claude's explanation (shown in dashboard)

    # ── Populated by write_node ────────────────────────────────────────────
    written:            Optional[bool]   # True once the row is in the DB

    # ── Set by any node on failure ─────────────────────────────────────────
    error:              Optional[str]


# ══════════════════════════════════════════════════════════════════════════
# 2. NODES
# Each node is a plain Python function:
#   - receives the current state (full dict)
#   - returns ONLY the fields it wants to update (LangGraph merges the rest)
# ══════════════════════════════════════════════════════════════════════════

def fetch_node(state: RecommendationState) -> dict:
    """
    PASS 1 STUB — In Pass 2 this will:
      - Connect to Postgres
      - SELECT * FROM customer_derivatives WHERE customer_id = state["customer_id"]
      - SELECT * FROM campaign_attribution   WHERE customer_id = state["customer_id"]
    """
    print(f"\n[fetch_node]    received state keys: {', '.join(k for k, v in state.items() if v is not None)}")

    # Fake data shaped like the real rows — same field names as our schema
    fake_derivatives = {
        "customer_id":                   state["customer_id"],
        "customer_type":                 "Repeat Purchaser",
        "total_orders":                  5,
        "total_revenue":                 276.49,
        "average_order_value":           55.30,
        "recency_days":                  18,
        "frequency_months":              2.1,
        "total_campaigns_received":      12,
        "email_open_rate":               0.58,
        "email_click_rate":              0.27,
        "avg_campaigns_before_purchase": 3.2,
        "campaign_influence_rate":       0.80,
        "attribution_segment":           "campaign_3_5",
        "days_since_last_campaign":      5,
        "spend_last_3_months":           89.99,
    }
    fake_attribution = [
        {"order_id": "ord_001", "attribution_bucket": "campaign_3_5",
         "campaigns_in_30d_window": 3, "campaign_influenced": True},
        {"order_id": "ord_002", "attribution_bucket": "campaign_1_2",
         "campaigns_in_30d_window": 2, "campaign_influenced": True},
    ]

    print(f"[fetch_node]    loaded derivatives and attribution for customer {state['customer_id']}")

    # Return ONLY the fields this node produced — customer_id stays in state automatically
    return {
        "derivatives": fake_derivatives,
        "attribution":  fake_attribution,
    }


def classify_node(state: RecommendationState) -> dict:
    """
    PASS 1 STUB — In Pass 2 this will:
      - Build a structured prompt from state["derivatives"] + state["attribution"]
      - Call the Anthropic API (claude-sonnet-4-6)
      - Parse JSON response into label + confidence + reasoning
      - Validate that label is one of the three allowed values

    This is the node that demonstrates D4 (prompt engineering) and
    D2 (structured output). Those concepts come alive in Pass 2.
    """
    print(f"\n[classify_node] received state keys: {', '.join(k for k, v in state.items() if v is not None)}")

    # Show what the node can see from previous nodes
    d = state["derivatives"]
    print(f"[classify_node] customer_type={d['customer_type']}, "
          f"segment={d['attribution_segment']}, "
          f"recency={d['recency_days']}d, "
          f"influence_rate={d['campaign_influence_rate']:.0%}")
    print(f"[classify_node] would call Claude here — returning stub label")

    # Stub output — Pass 2 replaces this with a real API call
    return {
        "label":      "send_campaign",
        "confidence": 0.87,
        "reasoning":  "STUB — Claude would explain here: customer is in campaign_3_5 "
                      "segment with 80% campaign influence rate, 18-day recency, "
                      "and active email engagement (58% open rate). Send recommended.",
    }


def write_node(state: RecommendationState) -> dict:
    """
    PASS 1 STUB — In Pass 2 this will:
      - INSERT INTO customer_recommendations (customer_id, label, confidence, reasoning, ...)
      - ON CONFLICT DO UPDATE (idempotent)
    """
    print(f"\n[write_node]    received state keys: {', '.join(k for k, v in state.items() if v is not None)}")
    print(f"[write_node]    would write to customer_recommendations — stub complete")
    print(f"[write_node]    label={state['label']!r}  confidence={state['confidence']}")

    return {"written": True}


# ══════════════════════════════════════════════════════════════════════════
# 3. GRAPH
# Build it, add nodes, add edges, compile.
# Compilation validates the graph (no dangling nodes, no missing edges).
# ══════════════════════════════════════════════════════════════════════════

def build_graph() -> object:
    # Create a StateGraph parameterised on our state type
    graph = StateGraph(RecommendationState)

    # Add nodes — name (string) + function
    graph.add_node("fetch",    fetch_node)
    graph.add_node("classify", classify_node)
    graph.add_node("write",    write_node)

    # Add edges — execution order
    graph.add_edge(START,      "fetch")     # START is the entry point
    graph.add_edge("fetch",    "classify")
    graph.add_edge("classify", "write")
    graph.add_edge("write",    END)         # END terminates the graph

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════
# 4. RUN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_graph()

    # Initial state — only the fields we know at invocation time.
    # All other fields start as None (TypedDict allows Optional fields
    # to be absent from the initial dict).
    initial_state: RecommendationState = {
        "customer_id":  "abc123",
        "derivatives":  None,
        "attribution":  None,
        "label":        None,
        "confidence":   None,
        "reasoning":    None,
        "written":      None,
        "error":        None,
    }

    print("═" * 57)
    print("  merchant-abc recommendation agent — Pass 1 skeleton")
    print("═" * 57)

    # .invoke() runs the full graph synchronously and returns the final state
    final_state = app.invoke(initial_state)

    print("\n" + "─" * 57)
    print("Final state:")
    for key in ("customer_id", "label", "confidence", "reasoning", "written", "error"):
        print(f"  {key:<14} {final_state.get(key)}")
    print("─" * 57)

    print("""
What just happened:
  1. invoke() put initial_state into START
  2. fetch_node ran → returned derivatives + attribution → merged into state
  3. classify_node ran → saw derivatives + attribution → returned label, confidence, reasoning
  4. write_node ran → saw everything → returned written=True
  5. Graph hit END → returned final state to you

Nothing was hardcoded in the flow control.
You defined the graph. LangGraph ran it.
""")
