"""
Document upload and processing API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_db
from ...models.user import User
from ...api.v1.auth import get_current_active_user

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    company_id: int = Form(...),
    document_type: str = Form(...),
    description: Optional[str] = Form(None),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload company document"""
    
    # Validate file type and size
    if file.content_type not in ["application/pdf", "application/vnd.ms-excel"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type"
        )
    
    # Save file and create document record
    # Schedule background processing
    
    return {
        "document_id": "doc_123",
        "filename": file.filename,
        "status": "uploaded",
        "processing": True
    }


@router.get("/")
async def list_documents(
    company_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List documents"""
    
    return {
        "documents": [],
        "total": 0
    }


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get document details"""
    
    return {
        "id": document_id,
        "filename": "annual_report.pdf",
        "processed": True,
        "extracted_metrics": {}
    }