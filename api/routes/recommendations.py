"""
routes/recommendations.py — Recommendation endpoints.

All queries JOIN customer_recommendations → customer_derivatives → customers
so every response carries both the agent output AND the underlying features.

Endpoints
  GET /recommendations/                → paginated list, newest first
  GET /recommendations/summary         → count + avg LTV by label (dashboard scorecard)
  GET /recommendations/scorecard       → rich scorecard for the dashboard hero panel
  GET /recommendations/{customer_id}   → single customer full detail
"""
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

router = APIRouter()


# ── Helper: parse key_signals TEXT → list ─────────────────────────────────
def _parse_signals(raw) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return [raw]


# ══════════════════════════════════════════════════════════════════════════
# GET /recommendations/
# ══════════════════════════════════════════════════════════════════════════
@router.get("/")
async def list_recommendations(
    limit: int = 50,
    offset: int = 0,
    label: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated list of recommendations, newest first.
    Optional ?label=send_campaign filter.
    """
    where = "WHERE r.recommendation_label = :label" if label else ""
    result = await db.execute(text(f"""
        SELECT
            r.customer_id,
            c.first_name,
            c.last_name,
            r.recommendation_label,
            r.confidence_score,
            r.reasoning,
            r.key_signals,
            r.recommended_at,
            r.model_version,
            d.attribution_segment,
            d.customer_type,
            d.total_orders,
            ROUND(d.total_revenue::numeric, 2)            AS total_revenue,
            ROUND(d.average_order_value::numeric, 2)      AS average_order_value,
            d.recency_days,
            ROUND(d.email_open_rate::numeric * 100, 1)    AS open_rate_pct,
            ROUND(d.email_click_rate::numeric * 100, 1)   AS click_rate_pct,
            ROUND(d.campaign_influence_rate::numeric * 100, 1) AS influence_rate_pct
        FROM customer_recommendations r
        JOIN customers c         ON c.customer_id = r.customer_id
        LEFT JOIN (
            SELECT DISTINCT ON (customer_id) *
            FROM customer_derivatives
            ORDER BY customer_id, run_id DESC
        ) d ON d.customer_id = r.customer_id
        {where}
        ORDER BY r.recommended_at DESC, r.confidence_score DESC
        LIMIT :limit OFFSET :offset
    """), {"limit": limit, "offset": offset, "label": label})
    rows = result.mappings().all()
    out = []
    for r in rows:
        row = dict(r)
        row["key_signals"] = _parse_signals(row.get("key_signals"))
        out.append(row)
    return {"recommendations": out, "limit": limit, "offset": offset}


# ══════════════════════════════════════════════════════════════════════════
# GET /recommendations/summary
# ══════════════════════════════════════════════════════════════════════════
@router.get("/summary")
async def recommendation_summary(db: AsyncSession = Depends(get_db)):
    """Count + avg confidence by label."""
    result = await db.execute(text("""
        SELECT
            recommendation_label,
            COUNT(*)                                        AS customer_count,
            ROUND(AVG(confidence_score)::numeric, 2)       AS avg_confidence
        FROM customer_recommendations
        GROUP BY recommendation_label
        ORDER BY customer_count DESC
    """))
    rows = result.mappings().all()
    return {"summary": [dict(r) for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# GET /recommendations/scorecard
# The dashboard hero panel — one number per label with business context.
# ══════════════════════════════════════════════════════════════════════════
@router.get("/scorecard")
async def scorecard(db: AsyncSession = Depends(get_db)):
    """
    Rich scorecard for the dashboard.
    Returns per-label counts, avg LTV, avg orders, avg open rate,
    plus a total row and the top customer per label.
    """
    result = await db.execute(text("""
        SELECT
            r.recommendation_label                              AS label,
            COUNT(*)                                            AS customers,
            ROUND(AVG(d.total_revenue)::numeric, 2)            AS avg_ltv,
            ROUND(AVG(d.total_orders)::numeric, 1)             AS avg_orders,
            ROUND(AVG(r.confidence_score)::numeric, 2)         AS avg_confidence,
            ROUND(AVG(d.email_open_rate)::numeric * 100, 1)    AS avg_open_pct,
            ROUND(SUM(d.total_revenue)::numeric, 2)            AS total_segment_revenue,
            ROUND(AVG(d.campaign_influence_rate)::numeric * 100, 1) AS avg_influence_pct
        FROM customer_recommendations r
        LEFT JOIN (
            SELECT DISTINCT ON (customer_id) *
            FROM customer_derivatives
            ORDER BY customer_id, run_id DESC
        ) d ON d.customer_id = r.customer_id
        GROUP BY r.recommendation_label
        ORDER BY avg_ltv DESC NULLS LAST
    """))
    segments = [dict(r) for r in result.mappings().all()]

    # Overall totals
    totals = await db.execute(text("""
        SELECT
            COUNT(*)                                        AS total_customers,
            ROUND(SUM(d.total_revenue)::numeric, 2)        AS total_revenue,
            ROUND(AVG(d.total_revenue)::numeric, 2)        AS avg_ltv,
            ROUND(AVG(d.total_orders)::numeric, 1)         AS avg_orders
        FROM customer_recommendations r
        LEFT JOIN (
            SELECT DISTINCT ON (customer_id) *
            FROM customer_derivatives
            ORDER BY customer_id, run_id DESC
        ) d ON d.customer_id = r.customer_id
    """))
    total_row = dict(totals.mappings().first() or {})

    # Label descriptions for the dashboard
    label_meta = {
        "send_campaign": {
            "action": "Send next scheduled campaign",
            "description": "Campaigns demonstrably drive purchases for these customers.",
            "color": "#22c55e",
        },
        "dont_send": {
            "action": "Wait — re-evaluate in 2 weeks",
            "description": "Recently purchased. Don't interrupt the post-purchase experience.",
            "color": "#f59e0b",
        },
        "no_campaign_needed": {
            "action": "Send NPIs, loyalty offers & seasonal launches only",
            "description": "Organic brand-loyal buyers. Protect from regular cadence fatigue.",
            "color": "#3b82f6",
        },
        "no_campaign_impact": {
            "action": "Suppress or route to 90-day re-engagement flow",
            "description": "Ghost subscribers or prospects. Campaigns not converting.",
            "color": "#ef4444",
        },
    }
    for seg in segments:
        seg.update(label_meta.get(seg["label"], {}))

    return {
        "scorecard": segments,
        "totals": total_row,
    }


# ══════════════════════════════════════════════════════════════════════════
# GET /recommendations/{customer_id}
# ══════════════════════════════════════════════════════════════════════════
@router.get("/{customer_id}")
async def get_recommendation(
    customer_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Full recommendation detail for one customer.
    Includes agent output + all derivative features + last 5 attribution rows.
    """
    # Recommendation + derivatives
    result = await db.execute(text("""
        SELECT
            r.customer_id,
            c.first_name,
            c.last_name,
            c.email,
            r.recommendation_label,
            r.confidence_score,
            r.reasoning,
            r.key_signals,
            r.recommended_at,
            r.model_version,
            d.*
        FROM customer_recommendations r
        JOIN customers c ON c.customer_id = r.customer_id
        LEFT JOIN (
            SELECT DISTINCT ON (customer_id) *
            FROM customer_derivatives
            ORDER BY customer_id, run_id DESC
        ) d ON d.customer_id = r.customer_id
        WHERE r.customer_id = :cid
    """), {"cid": customer_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendation found for customer {customer_id}"
        )
    detail = dict(row)
    detail["key_signals"] = _parse_signals(detail.get("key_signals"))

    # Last 5 attribution rows
    attr = await db.execute(text("""
        SELECT
            order_id, order_date, order_revenue,
            order_number_in_lifecycle, campaigns_in_30d_window,
            campaign_influenced, attribution_bucket
        FROM campaign_attribution
        WHERE customer_id = :cid
          AND attribution_bucket != 'pre_campaign_era'
        ORDER BY order_date DESC
        LIMIT 5
    """), {"cid": customer_id})
    detail["recent_attribution"] = [dict(r) for r in attr.mappings().all()]

    return detail
