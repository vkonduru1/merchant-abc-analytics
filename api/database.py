"""
database.py — Async SQLAlchemy engine + session factory.
Reads connection params from environment variables.

Port note: Docker publishes Postgres on 5433 externally (5432 inside the container).
  Local dev  → POSTGRES_PORT=5433
  Inside Docker network → POSTGRES_PORT=5432  (service name: db)
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")   # 5433 = published Docker port
POSTGRES_DB   = os.getenv("POSTGRES_DB",   "merchant_abc")
POSTGRES_USER = os.getenv("POSTGRES_USER", "abc_user")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "abc_dev_password")

DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASS}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "false").lower() == "true",
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
