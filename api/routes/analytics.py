"""analytics.py — Attribution and KPI endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

router = APIRouter()


@router.get("/attribution-summary")
async def attribution_summary(db: AsyncSession = Depends(get_db)):
    """
    The key business output: customer segments by campaign influence.
    Which segment generates the most revenue?
    """
    result = await db.execute(text("""
        SELECT
            attribution_segment,
            COUNT(DISTINCT customer_id)                                             AS customer_count,
            ROUND(AVG(avg_campaigns_before_purchase)::numeric, 2)                  AS avg_campaigns_before_purchase,
            ROUND(AVG(total_revenue)::numeric, 2)                                  AS avg_lifetime_revenue,
            ROUND(AVG(average_order_value)::numeric, 2)                            AS avg_order_value,
            ROUND(AVG(total_orders)::numeric, 2)                                   AS avg_orders,
            ROUND(AVG(email_open_rate)::numeric, 4)                                AS avg_open_rate,
            ROUND(AVG(email_click_rate)::numeric, 4)                               AS avg_click_rate,
            ROUND(AVG(campaign_influence_rate)::numeric, 4)                        AS avg_campaign_influence_rate
        FROM customer_derivatives
        GROUP BY attribution_segment
        ORDER BY avg_lifetime_revenue DESC NULLS LAST
    """))
    rows = result.mappings().all()
    return {"attribution_segments": [dict(r) for r in rows]}


@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)):
    """Top-level numbers for the dashboard."""
    result = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM customers)                        AS total_customers,
            (SELECT COUNT(*) FROM orders)                           AS total_orders,
            (SELECT COALESCE(SUM(total_price_usd),0) FROM orders)  AS total_revenue,
            (SELECT COUNT(*) FROM campaigns)                        AS total_campaigns,
            (SELECT COUNT(*) FROM email_events
             WHERE event_type = 'sent')                             AS total_emails_sent,
            (SELECT COUNT(*) FROM customer_events
             WHERE event_type = 'ORDER')                            AS total_purchase_events,
            (SELECT ROUND(AVG(avg_campaigns_before_purchase)::numeric, 2)
             FROM customer_derivatives
             WHERE avg_campaigns_before_purchase IS NOT NULL)       AS avg_campaigns_before_purchase
    """))
    row = result.mappings().first()
    return dict(row) if row else {}


@router.get("/customer-timeline-sample")
async def timeline_sample(limit: int = 5, db: AsyncSession = Depends(get_db)):
    """Sample of the v_customer_timeline view — shows the chronological overlay in action."""
    result = await db.execute(text("""
        SELECT * FROM v_customer_timeline
        WHERE customer_id IN (
            SELECT customer_id FROM customers
            ORDER BY customer_created_at
            LIMIT :limit
        )
        ORDER BY customer_id, event_date
    """), {"limit": limit})
    rows = result.mappings().all()
    return {"events": [dict(r) for r in rows], "sample_customers": limit}
