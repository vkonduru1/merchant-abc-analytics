# merchant-abc-analytics — Claude Code Context

## What this project is
E-Commerce Marketing Campaign Analytics & Recommendation Engine.
Merchant "abc" sells specialty coffee and beverages. Two data sources:
- **Shopify** — orders, customers, products (ERP)
- **Klaviyo** — campaigns, email events (CRM)

Core question: *How many campaigns does it take before each customer buys — and should we send them the next one?*

## Architecture: AIIR framework
```
Shopify + Klaviyo → Identity Resolution → customer_events (timeline) →
customer_derivatives (features) → campaign_attribution → Recommendations
```

## Tech stack
- PostgreSQL 15 (primary DB)
- FastAPI + Python 3.11 (backend)
- React + Vite + Tailwind (frontend, JSX files)
- LangGraph (agent orchestration — Phase 2)
- Docker Compose (full stack)
- Nginx (reverse proxy)
- Anthropic Claude API + OpenAI placeholder

## Key design decisions
1. `customer_events` table = the centrepiece. Every metric is derived from it.
2. Identity resolution via email: Shopify uses numeric customer_id, Klaviyo uses profile_id. customer_identity_map normalises both.
3. LEFT JOINs throughout — never INNER JOINs on order sub-tables (drops orders without shipping/tax/discount rows).
4. Dual-key join on product_variants: `variant_id AND sku` both required.
5. Attribution window = 30 days lookback before each purchase event.

## File structure
- `db/schema.sql` — complete DDL, run once
- `api/main.py` — FastAPI entrypoint
- `api/database.py` — SQLAlchemy async engine
- `api/routes/` — one file per domain
- `data_generator/generate_synthetic_data.py` — creates test data
- `transformation/` — pipeline scripts run in sequence
- `frontend/src/` — React JSX components

## Environment
All secrets in `.env` (copied from `.env.example`). Never commit `.env`.

## Current phase
Phase 1 — Foundation: schema, synthetic data, transformation pipeline, basic API and dashboard.
