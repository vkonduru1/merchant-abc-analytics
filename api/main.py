"""
main.py — FastAPI application entrypoint for merchant-abc-analytics.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from routes import customers, campaigns, analytics, recommendations, mcp


# ── Startup / shutdown ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify DB connection on startup
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    print("✓ Database connection verified")
    yield
    await engine.dispose()


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="merchant-abc Analytics API",
    description=(
        "E-Commerce Campaign Analytics & Recommendation Engine. "
        "Built on the AIIR framework: Attribution → Insights → "
        "Identity Resolution → Recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
# Local dev + production domain — update ALLOWED_ORIGINS in .env for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",    # Vite dev server
    "http://localhost",
    "https://demo.vedakonduru.ai",
    "https://vedakonduru.ai",
]
origins = list(set(DEFAULT_ORIGINS + [o.strip() for o in ALLOWED_ORIGINS if o.strip()]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(customers.router,        prefix="/customers",        tags=["Customers"])
app.include_router(campaigns.router,        prefix="/campaigns",        tags=["Campaigns"])
app.include_router(analytics.router,        prefix="/analytics",        tags=["Analytics"])
app.include_router(recommendations.router,  prefix="/recommendations",  tags=["Recommendations"])
app.include_router(mcp.router,              prefix="/mcp",              tags=["MCP"])


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "merchant-abc Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "mcp_tools": "/mcp/tools",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
