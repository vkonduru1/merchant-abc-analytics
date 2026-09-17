"""
main.py — FastAPI application entrypoint for merchant-abc-analytics.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import customers, campaigns, analytics, recommendations

app = FastAPI(
    title="merchant-abc Analytics API",
    description="E-Commerce Campaign Analytics & Recommendation Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────
app.include_router(customers.router,       prefix="/customers",       tags=["Customers"])
app.include_router(campaigns.router,       prefix="/campaigns",       tags=["Campaigns"])
app.include_router(analytics.router,       prefix="/analytics",       tags=["Analytics"])
app.include_router(recommendations.router, prefix="/recommendations",  tags=["Recommendations"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "merchant-abc Analytics API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
