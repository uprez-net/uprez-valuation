"""
User management API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_db
from ...models.user import User, UserRole
from ...api.v1.auth import get_current_active_user, get_admin_user

router = APIRouter()


@router.get("/me")
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
):
    """Get current user profile"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "role": current_user.role,
        "status": current_user.status
    }


@router.put("/me")
async def update_user_profile(
    profile_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update current user profile"""
    
    # Update allowed fields
    allowed_fields = ['first_name', 'last_name', 'phone']
    for field, value in profile_data.items():
        if field in allowed_fields and hasattr(current_user, field):
            setattr(current_user, field, value)
    
    await db.commit()
    
    return {"message": "Profile updated successfully"}


@router.get("/")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all users (admin only)"""
    
    # Implementation would query users with pagination
    return {
        "users": [],
        "total": 0
    }


@router.get("/{user_id}")
async def get_user(
    user_id: int = Path(..., description="User ID"),
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get user by ID (admin only)"""
    
    # Implementation would get user by ID
    return {
        "id": user_id,
        "email": "user@example.com",
        "first_name": "John",
        "last_name": "Doe"
    }