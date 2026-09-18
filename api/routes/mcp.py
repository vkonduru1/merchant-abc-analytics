"""
routes/mcp.py — MCP (Model Context Protocol) server endpoint.

Exposes the merchant-abc recommendation engine as MCP tools so any
MCP-compatible client (Claude Desktop, Claude Code, other agents) can
call your API with natural language.

Tools exposed
─────────────
  get_recommendation_scorecard   → summary table (all 4 labels + LTV)
  get_customer_recommendation    → full reasoning for one customer
  get_attribution_summary        → revenue by AIIR attribution segment
  get_campaign_performance       → top campaigns by influenced revenue
  list_customers_by_label        → browse customers in a segment

MCP transport: HTTP/SSE (works over the internet, not just stdio)

Usage from Claude Desktop (add to claude_desktop_config.json):
  {
    "mcpServers": {
      "merchant-abc": {
        "url": "https://demo.vedakonduru.ai/mcp/sse"
      }
    }
  }

Locally:
  {
    "mcpServers": {
      "merchant-abc": {
        "url": "http://localhost:8000/mcp/sse"
      }
    }
  }

Install dependency:
  pip install mcp
"""
import json
from typing import Any, List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio

router = APIRouter()

# ── Try to import MCP SDK ─────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp import types as mcp_types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
# Tool definitions — what MCP clients see
# ══════════════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "name": "get_recommendation_scorecard",
        "description": (
            "Get the campaign recommendation scorecard for all customers. "
            "Returns counts, average LTV, and merchant action for each of the "
            "four labels: send_campaign, dont_send, no_campaign_needed, no_campaign_impact."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_customer_recommendation",
        "description": (
            "Get the full recommendation for a specific customer — label, confidence score, "
            "Claude's reasoning, key signals, and the underlying purchase + campaign data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The Shopify customer_id (numeric string, e.g. '7300615917143')",
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_attribution_summary",
        "description": (
            "Get revenue and engagement metrics grouped by AIIR attribution segment "
            "(campaign_3_5, campaign_1_2, no_campaign_influence, never_purchased). "
            "Shows which segment generates the most lifetime revenue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_campaign_performance",
        "description": (
            "Get performance metrics for recent campaigns: open rate, click rate, "
            "and orders/revenue influenced within 30 days of the send."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of campaigns to return (default 10)",
                    "default": 10,
                }
            },
            "required": [],
        },
    },
    {
        "name": "list_customers_by_label",
        "description": (
            "List customers filtered by recommendation label. "
            "Use this to browse who is in each segment."
        ),
        "inputSchema": {
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
                    "description": "Recommendation label to filter by",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of customers to return (default 20)",
                    "default": 20,
                },
            },
            "required": ["label"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════
# Tool execution — calls the internal FastAPI endpoints
# ══════════════════════════════════════════════════════════════════════════
async def execute_tool(name: str, arguments: dict) -> str:
    """
    Executes an MCP tool by calling the corresponding FastAPI route internally.
    Returns a JSON string — MCP tools always return text content.
    """
    import httpx

    base = "http://localhost:8000"   # internal call to ourselves

    async with httpx.AsyncClient(timeout=30.0) as client:
        if name == "get_recommendation_scorecard":
            r = await client.get(f"{base}/recommendations/scorecard")

        elif name == "get_customer_recommendation":
            cid = arguments.get("customer_id", "")
            if not cid:
                return json.dumps({"error": "customer_id is required"})
            r = await client.get(f"{base}/recommendations/{cid}")

        elif name == "get_attribution_summary":
            r = await client.get(f"{base}/analytics/attribution-summary")

        elif name == "get_campaign_performance":
            limit = arguments.get("limit", 10)
            r = await client.get(f"{base}/analytics/campaign-performance?limit={limit}")

        elif name == "list_customers_by_label":
            label = arguments.get("label", "")
            limit = arguments.get("limit", 20)
            r = await client.get(
                f"{base}/recommendations/?label={label}&limit={limit}"
            )
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        if r.status_code == 200:
            return json.dumps(r.json(), indent=2, default=str)
        else:
            return json.dumps({"error": f"HTTP {r.status_code}", "detail": r.text})


# ══════════════════════════════════════════════════════════════════════════
# MCP endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.get("/tools")
async def list_tools():
    """List available MCP tools — used by MCP clients to discover capabilities."""
    return {"tools": TOOLS}


@router.post("/tools/{tool_name}")
async def call_tool(tool_name: str, body: dict = {}):
    """
    Direct HTTP tool call (non-SSE).
    Useful for testing: POST /mcp/tools/get_recommendation_scorecard
    """
    result = await execute_tool(tool_name, body)
    return {"content": [{"type": "text", "text": result}]}


@router.get("/sse")
async def mcp_sse():
    """
    SSE transport endpoint for MCP clients (Claude Desktop, Claude Code).
    Streams the MCP protocol over Server-Sent Events.

    If the mcp package is not installed, returns a JSON capability list instead
    so you can still test the tools via /mcp/tools/{name}.
    """
    if not MCP_AVAILABLE:
        # Graceful fallback — list tools as JSON so the API is still useful
        return {
            "note": "MCP SSE transport not available. Install: pip install mcp",
            "tools": TOOLS,
            "direct_call_endpoint": "POST /mcp/tools/{tool_name}",
        }

    server = Server("merchant-abc-analytics")

    @server.list_tools()
    async def handle_list_tools() -> List[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict
    ) -> List[mcp_types.TextContent]:
        result = await execute_tool(name, arguments)
        return [mcp_types.TextContent(type="text", text=result)]

    transport = SseServerTransport("/mcp/messages/")

    async def event_generator():
        async with transport.connect_sse(
            scope={}, receive=asyncio.Queue().get, send=asyncio.Queue().put
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options()
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/health")
async def mcp_health():
    return {
        "status": "ok",
        "mcp_sdk_available": MCP_AVAILABLE,
        "tools_count": len(TOOLS),
        "tools": [t["name"] for t in TOOLS],
    }
