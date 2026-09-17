"""campaigns.py — Campaign endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

router = APIRouter()


@router.get("/")
async def list_campaigns(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT c.*, ct.type_name as campaign_type_name
        FROM campaigns c
        LEFT JOIN campaign_type ct ON ct.campaign_type_id = c.campaign_type_id
        ORDER BY c.send_time DESC
        LIMIT :limit
    """), {"limit": limit})
    rows = result.mappings().all()
    return {"campaigns": [dict(r) for r in rows]}


@router.get("/{campaign_id}/stats")
async def campaign_stats(campaign_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT
            campaign_id,
            COUNT(*) FILTER (WHERE event_type = 'sent')         AS sent,
            COUNT(*) FILTER (WHERE event_type = 'delivered')    AS delivered,
            COUNT(*) FILTER (WHERE event_type = 'opened')       AS opened,
            COUNT(*) FILTER (WHERE event_type = 'clicked')      AS clicked,
            COUNT(*) FILTER (WHERE event_type = 'unsubscribed') AS unsubscribed,
            ROUND(COUNT(*) FILTER (WHERE event_type = 'opened')::numeric /
                  NULLIF(COUNT(*) FILTER (WHERE event_type = 'sent'), 0) * 100, 2) AS open_rate_pct,
            ROUND(COUNT(*) FILTER (WHERE event_type = 'clicked')::numeric /
                  NULLIF(COUNT(*) FILTER (WHERE event_type = 'sent'), 0) * 100, 2) AS click_rate_pct
        FROM email_events
        WHERE campaign_id = :campaign_id
        GROUP BY campaign_id
    """), {"campaign_id": campaign_id})
    row = result.mappings().first()
    return dict(row) if row else {}
