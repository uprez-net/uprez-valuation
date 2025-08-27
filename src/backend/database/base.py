"""
Database Base Configuration
SQLAlchemy setup and base classes for the IPO Valuation Platform
"""
import asyncio
from typing import Any, Dict, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import settings

# SQLAlchemy Base
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

# Database engines
engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_connections - settings.database.pool_size,
    pool_timeout=settings.database.pool_timeout,
    echo=settings.debug,
)

# Async engine for high-performance operations
async_engine = create_async_engine(
    settings.database.url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_connections - settings.database.pool_size,
    pool_timeout=settings.database.pool_timeout,
    echo=settings.debug,
)

# Session makers
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncSession:
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        yield session


class DatabaseManager:
    """Database connection and lifecycle manager"""
    
    def __init__(self):
        self.engine = engine
        self.async_engine = async_engine
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables"""
        Base.metadata.drop_all(bind=self.engine)
    
    async def create_tables_async(self):
        """Create all tables asynchronously"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables_async(self):
        """Drop all tables asynchronously"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    def get_session(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    async def get_async_session(self) -> AsyncSession:
        """Get async database session"""
        return AsyncSessionLocal()
    
    async def close(self):
        """Close database connections"""
        await self.async_engine.dispose()
        self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()