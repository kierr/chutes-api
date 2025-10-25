import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from api.config import settings
from typing import AsyncGenerator
from contextlib import asynccontextmanager

# XXX Legacy database, delete after migration.
legacy_engine = create_async_engine(
    settings.legacy_db_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_overflow,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
    pool_timeout=30,
    pool_recycle=900,
    pool_use_lifo=True,
)

# Read/write database engine.
engine = create_async_engine(
    settings.db_rw_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_overflow,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
    pool_timeout=30,
    pool_recycle=900,
    pool_use_lifo=True,
)

# Read-only engine.
ro_engine = create_async_engine(
    settings.db_ro_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_overflow,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
    pool_timeout=30,
    pool_recycle=900,
    pool_use_lifo=True,
)

# Session makers.
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
SessionLocalRead = sessionmaker(
    bind=ro_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
LegacySessionLocal = sessionmaker(
    bind=legacy_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


@asynccontextmanager
async def get_session(
    readonly: bool = False, legacy: bool = False
) -> AsyncGenerator[AsyncSession, None]:
    session_maker = (
        SessionLocalRead if readonly else (SessionLocal if not legacy else LegacySessionLocal)
    )
    async with session_maker() as session:
        try:
            yield session
            if not readonly:
                await session.commit()
        except Exception:
            if not readonly:
                try:
                    await session.rollback()
                except Exception:
                    pass
            raise


async def get_db_session():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                pass
            raise


async def get_db_ro_session():
    async with SessionLocalRead() as session:
        yield session


def generate_uuid():
    """
    Helper for uuid generation.
    """
    return str(uuid.uuid4())
