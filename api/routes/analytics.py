"""
routes/analytics.py — Attribution and KPI endpoints.

Endpoints
  GET /analytics/overview               → top-level dashboard numbers
  GET /analytics/attribution-summary    → revenue by attribution segment
  GET /analytics/campaign-performance   → per-campaign open/click rates
  GET /analytics/monthly-revenue        → monthly order revenue trend
  GET /analytics/customer-timeline      → chronological events (replaces view)
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════
# GET /analytics/overview
# ══════════════════════════════════════════════════════════════════════════
@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)):
    """Top-level numbers for the dashboard hero row."""
    result = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM customers)                            AS total_customers,
            (SELECT COUNT(*) FROM orders
             WHERE financial_status IN ('paid','pending'))              AS total_orders,
            (SELECT ROUND(COALESCE(SUM(total_price_usd),0)::numeric,2)
             FROM orders
             WHERE financial_status IN ('paid','pending'))              AS total_revenue,
            (SELECT COUNT(*) FROM campaigns)                            AS total_campaigns,
            (SELECT COUNT(*) FROM email_events
             WHERE event_type = 'delivered')                            AS total_emails_delivered,
            (SELECT COUNT(*) FROM customer_recommendations)             AS customers_classified,
            (SELECT COUNT(*) FROM customer_recommendations
             WHERE recommendation_label = 'send_campaign')              AS send_campaign_count,
            (SELECT COUNT(*) FROM customer_recommendations
             WHERE recommendation_label = 'no_campaign_needed')         AS brand_loyal_count,
            (SELECT COUNT(*) FROM customer_recommendations
             WHERE recommendation_label = 'no_campaign_impact')         AS ghost_count,
            (SELECT ROUND(AVG(avg_campaigns_before_purchase)::numeric,2)
             FROM customer_derivatives
             WHERE avg_campaigns_before_purchase IS NOT NULL)           AS avg_campaigns_before_purchase,
            (SELECT ROUND(AVG(campaign_influence_rate)::numeric * 100, 1)
             FROM customer_derivatives
             WHERE campaign_influence_rate IS NOT NULL)                 AS avg_influence_rate_pct
    """))
    row = result.mappings().first()
    return dict(row) if row else {}


# ══════════════════════════════════════════════════════════════════════════
# GET /analytics/attribution-summary
# ══════════════════════════════════════════════════════════════════════════
@router.get("/attribution-summary")
async def attribution_summary(db: AsyncSession = Depends(get_db)):
    """Revenue and engagement by attribution segment — the AIIR business case."""
    result = await db.execute(text("""
        SELECT
            attribution_segment,
            COUNT(DISTINCT customer_id)                                 AS customer_count,
            ROUND(AVG(avg_campaigns_before_purchase)::numeric, 2)      AS avg_campaigns_before_purchase,
            ROUND(AVG(total_revenue)::numeric, 2)                      AS avg_lifetime_revenue,
            ROUND(AVG(average_order_value)::numeric, 2)                AS avg_order_value,
            ROUND(AVG(total_orders)::numeric, 2)                       AS avg_orders,
            ROUND(AVG(email_open_rate)::numeric * 100, 1)              AS avg_open_rate_pct,
            ROUND(AVG(email_click_rate)::numeric * 100, 1)             AS avg_click_rate_pct,
            ROUND(AVG(campaign_influence_rate)::numeric * 100, 1)      AS avg_influence_rate_pct,
            ROUND(SUM(total_revenue)::numeric, 2)                      AS total_segment_revenue
        FROM customer_derivatives
        GROUP BY attribution_segment
        ORDER BY avg_lifetime_revenue DESC NULLS LAST
    """))
    rows = result.mappings().all()
    return {"attribution_segments": [dict(r) for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# GET /analytics/campaign-performance
# ══════════════════════════════════════════════════════════════════════════
@router.get("/campaign-performance")
async def campaign_performance(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Per-campaign open/click/conversion rates, newest first."""
    result = await db.execute(text("""
        SELECT
            k.campaign_id,
            k.campaign_name,
            ct.type_name                                            AS campaign_type,
            k.send_time,
            k.total_delivered,
            k.total_opens,
            k.total_clicks,
            k.total_unsubscribes,
            k.has_discount_code,
            k.discount_pct,
            ROUND(k.total_opens::numeric /
                  NULLIF(k.total_delivered,0) * 100, 1)            AS open_rate_pct,
            ROUND(k.total_clicks::numeric /
                  NULLIF(k.total_delivered,0) * 100, 1)            AS click_rate_pct,
            -- Orders placed within 30 days after this campaign
            (SELECT COUNT(DISTINCT ca.order_id)
             FROM campaign_attribution ca
             WHERE ca.most_recent_campaign_id = k.campaign_id
               AND ca.campaign_influenced = TRUE)                   AS influenced_orders,
            (SELECT ROUND(COALESCE(SUM(ca.order_revenue),0)::numeric, 2)
             FROM campaign_attribution ca
             WHERE ca.most_recent_campaign_id = k.campaign_id
               AND ca.campaign_influenced = TRUE)                   AS influenced_revenue
        FROM campaigns k
        LEFT JOIN campaign_type ct ON ct.campaign_type_id = k.campaign_type_id
        ORDER BY k.send_time DESC
        LIMIT :limit
    """), {"limit": limit})
    rows = result.mappings().all()
    return {"campaigns": [dict(r) for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# GET /analytics/monthly-revenue
# ══════════════════════════════════════════════════════════════════════════
@router.get("/monthly-revenue")
async def monthly_revenue(db: AsyncSession = Depends(get_db)):
    """Monthly order revenue — powers the trend line on the dashboard."""
    result = await db.execute(text("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', order_created_at), 'YYYY-MM') AS month,
            COUNT(*)                                                    AS orders,
            ROUND(SUM(total_price_usd)::numeric, 2)                   AS revenue,
            ROUND(AVG(total_price_usd)::numeric, 2)                   AS avg_order_value,
            COUNT(DISTINCT customer_id)                                 AS unique_customers
        FROM orders
        WHERE financial_status IN ('paid', 'pending')
        GROUP BY DATE_TRUNC('month', order_created_at)
        ORDER BY DATE_TRUNC('month', order_created_at)
    """))
    rows = result.mappings().all()
    return {"monthly_revenue": [dict(r) for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# GET /analytics/customer-timeline
# Replaces the missing v_customer_timeline view — query directly
# ══════════════════════════════════════════════════════════════════════════
@router.get("/customer-timeline")
async def customer_timeline_sample(
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """
    Chronological event overlay for a sample of customers.
    Shows the AIIR centrepiece — orders and campaigns on one timeline.
    """
    result = await db.execute(text("""
        SELECT
            ce.customer_id,
            c.first_name || ' ' || c.last_name   AS customer_name,
            ce.event_date,
            ce.event_type,
            ce.event_source,
            ce.revenue_amount,
            ce.product_type,
            ce.campaign_name,
            ce.campaign_type,
            ce.has_discount,
            ce.campaign_id
        FROM customer_events ce
        JOIN customers c ON c.customer_id = ce.customer_id
        WHERE ce.customer_id IN (
            SELECT customer_id FROM customers
            ORDER BY customer_created_at
            LIMIT :limit
        )
        ORDER BY ce.customer_id, ce.event_date
    """), {"limit": limit})
    rows = result.mappings().all()
    return {
        "events": [dict(r) for r in rows],
        "sample_customers": limit,
        "note": "Chronological overlay of ORDER and CAMPAIGN_SENT events per customer",
    }
