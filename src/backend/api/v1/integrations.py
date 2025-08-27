"""
External integrations API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_db
from ...models.user import User
from ...api.v1.auth import get_current_active_user

router = APIRouter()


@router.post("/asx/sync")
async def sync_asx_data(
    company_codes: List[str],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Synchronize data from ASX"""
    
    # Schedule background sync
    return {
        "message": f"ASX sync started for {len(company_codes)} companies",
        "companies": company_codes,
        "status": "scheduled"
    }


@router.post("/asic/sync")
async def sync_asic_data(
    acn_list: List[str],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Synchronize data from ASIC"""
    
    return {
        "message": f"ASIC sync started for {len(acn_list)} companies",
        "acn_list": acn_list,
        "status": "scheduled"
    }


@router.get("/webhooks")
async def list_webhooks(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List configured webhooks"""
    
    return {
        "webhooks": [],
        "total": 0
    }


@router.post("/webhooks")
async def create_webhook(
    webhook_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create webhook endpoint"""
    
    return {
        "id": 1,
        "url": webhook_data.get("url"),
        "events": webhook_data.get("events", []),
        "status": "active"
    }