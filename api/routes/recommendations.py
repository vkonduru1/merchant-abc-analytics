"""recommendations.py — Recommendation engine endpoints (Phase 2: LangGraph agent)."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

router = APIRouter()


@router.get("/")
async def list_recommendations(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT
            r.customer_id,
            c.email,
            c.first_name,
            r.recommendation_label,
            r.confidence_score,
            r.attribution_segment,
            r.customer_type,
            r.recency_days,
            r.reasoning,
            r.run_date
        FROM customer_recommendations r
        JOIN customers c ON c.customer_id = r.customer_id
        ORDER BY r.run_date DESC, r.confidence_score DESC
        LIMIT :limit
    """), {"limit": limit})
    rows = result.mappings().all()
    return {"recommendations": [dict(r) for r in rows]}


@router.get("/summary")
async def recommendation_summary(db: AsyncSession = Depends(get_db)):
    """Count of send / dont_send / no_impact recommendations."""
    result = await db.execute(text("""
        SELECT
            recommendation_label,
            COUNT(*) AS customer_count,
            ROUND(AVG(confidence_score)::numeric, 4) AS avg_confidence
        FROM customer_recommendations
        GROUP BY recommendation_label
        ORDER BY customer_count DESC
    """))
    rows = result.mappings().all()
    return {"summary": [dict(r) for r in rows]}
