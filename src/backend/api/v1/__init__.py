"""
API v1 Package
Main API router for version 1 endpoints
"""
from fastapi import APIRouter
from .auth import router as auth_router
from .valuations import router as valuations_router
from .companies import router as companies_router
from .documents import router as documents_router
from .users import router as users_router

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(companies_router, prefix="/companies", tags=["companies"])
api_router.include_router(valuations_router, prefix="/valuations", tags=["valuations"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])

__all__ = ["api_router"]