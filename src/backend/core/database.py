"""
Database configuration and session management
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from .config import settings

# Create async engine
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=NullPool if settings.ENVIRONMENT == "test" else None,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create sync engine for migrations
sync_engine = create_engine(
    settings.DATABASE_URL.replace("+asyncpg", ""),
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factories
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False
)

# Base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for async database sessions"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db() -> Session:
    """Dependency for sync database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db() -> None:
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections"""
    await async_engine.dispose()
    sync_engine.dispose()


class DatabaseManager:
    """Database manager for advanced operations"""
    
    def __init__(self):
        self.async_engine = async_engine
        self.sync_engine = sync_engine
    
    async def create_tables(self):
        """Create all tables"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all tables"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def execute_raw_sql(self, sql: str, params: dict = None):
        """Execute raw SQL query"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql, params or {})
            await session.commit()
            return result
    
    async def backup_database(self, backup_path: str):
        """Create database backup (PostgreSQL specific)"""
        import subprocess
        import os
        
        # Extract connection details
        db_url = str(settings.DATABASE_URL)
        # Implementation would depend on specific backup requirements
        pass
    
    async def restore_database(self, backup_path: str):
        """Restore database from backup"""
        # Implementation would depend on specific restore requirements
        pass


# Global database manager instance
db_manager = DatabaseManager()