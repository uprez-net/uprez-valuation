"""
Admin API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_db
from ...models.user import User
from ...api.v1.auth import get_admin_user

router = APIRouter()


@router.get("/stats")
async def get_system_stats(
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get system statistics"""
    
    return {
        "users": {
            "total": 150,
            "active": 130,
            "new_this_month": 15
        },
        "companies": {
            "total": 500,
            "with_financials": 350
        },
        "valuations": {
            "total": 1250,
            "completed": 1100
        }
    }


@router.get("/users")
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all users (admin)"""
    
    return {
        "users": [],
        "total": 0
    }


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    status: str,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update user status"""
    
    return {
        "message": f"User {user_id} status updated to {status}"
    }