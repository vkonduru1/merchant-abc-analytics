"""customers.py — Customer endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database import get_db

router = APIRouter()


@router.get("/")
async def list_customers(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List all customers with basic stats."""
    result = await db.execute(text("""
        SELECT
            c.customer_id,
            c.email,
            c.first_name,
            c.last_name,
            c.customer_created_at,
            c.orders_count,
            c.total_spent_usd,
            c.accepts_marketing
        FROM customers c
        ORDER BY c.customer_created_at DESC
        LIMIT :limit OFFSET :offset
    """), {"limit": limit, "offset": offset})
    rows = result.mappings().all()
    return {"customers": [dict(r) for r in rows], "limit": limit, "offset": offset}


@router.get("/{customer_id}/timeline")
async def customer_timeline(
    customer_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Chronological event timeline for a single customer — the core view."""
    result = await db.execute(text("""
        SELECT
            event_id,
            customer_id,
            event_date,
            event_type,
            event_source,
            event_ref_id,
            revenue_amount,
            product_type,
            campaign_name,
            campaign_type,
            has_discount
        FROM customer_events
        WHERE customer_id = :customer_id
        ORDER BY event_date ASC
    """), {"customer_id": customer_id})
    rows = result.mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No events found for customer {customer_id}")

    return {
        "customer_id": customer_id,
        "total_events": len(rows),
        "events": [dict(r) for r in rows],
    }


@router.get("/{customer_id}/derivatives")
async def customer_derivatives(
    customer_id: str,
    db: AsyncSession = Depends(get_db)
):
    """First-level derived features for a customer."""
    result = await db.execute(text("""
        SELECT * FROM customer_derivatives
        WHERE customer_id = :customer_id
        ORDER BY run_date DESC
        LIMIT 1
    """), {"customer_id": customer_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="No derivatives computed yet for this customer.")
    return dict(row)
