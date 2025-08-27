"""
Company management API endpoints
"""

from typing import List, Optional, Dict, Any
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload

from ...core.database import get_async_db
from ...models.user import User
from ...models.company import Company, CompanyFinancials, CompanyDocument, CompanyStatus, CompanyType, Industry
from ...schemas.company import (
    CompanyCreate,
    CompanyUpdate,
    CompanyResponse,
    CompanyListResponse,
    CompanyFinancialsCreate,
    CompanyFinancialsUpdate,
    CompanyFinancialsResponse,
    CompanySearchFilters,
    CompanyBulkUpdate
)
from ...models.base import PaginationParams, PaginationResponse, SortParams
from ...api.v1.auth import get_current_active_user
from ...services.company import CompanyService
from ...services.external_data import ASXService, ASICService

router = APIRouter()


@router.post("/", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
async def create_company(
    company_data: CompanyCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new company"""
    
    # Check if company already exists
    if company_data.asx_code:
        existing = await CompanyService.get_by_asx_code(db, company_data.asx_code)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Company with ASX code {company_data.asx_code} already exists"
            )
    
    if company_data.acn:
        existing = await CompanyService.get_by_acn(db, company_data.acn)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Company with ACN {company_data.acn} already exists"
            )
    
    # Create company
    company = await CompanyService.create(
        db=db,
        company_data=company_data,
        created_by=str(current_user.id)
    )
    
    return CompanyResponse.from_orm(company)


@router.get("/", response_model=CompanyListResponse)
async def list_companies(
    pagination: PaginationParams = Depends(),
    sort: SortParams = Depends(),
    filters: CompanySearchFilters = Depends(),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List companies with filtering, pagination and sorting"""
    
    # Build query
    query = select(Company).options(selectinload(Company.financials))
    
    # Apply filters
    if filters.name:
        query = query.where(Company.name.ilike(f"%{filters.name}%"))
    
    if filters.asx_code:
        query = query.where(Company.asx_code.ilike(f"%{filters.asx_code}%"))
    
    if filters.industry:
        query = query.where(Company.industry == filters.industry)
    
    if filters.status:
        query = query.where(Company.status == filters.status)
    
    if filters.company_type:
        query = query.where(Company.company_type == filters.company_type)
    
    if filters.listing_date_from:
        query = query.where(Company.listing_date >= filters.listing_date_from)
    
    if filters.listing_date_to:
        query = query.where(Company.listing_date <= filters.listing_date_to)
    
    if filters.market_cap_min:
        query = query.where(Company.market_cap >= filters.market_cap_min)
    
    if filters.market_cap_max:
        query = query.where(Company.market_cap <= filters.market_cap_max)
    
    if filters.search:
        search_term = f"%{filters.search}%"
        query = query.where(
            or_(
                Company.name.ilike(search_term),
                Company.asx_code.ilike(search_term),
                Company.description.ilike(search_term)
            )
        )
    
    # Apply sorting
    sort_column = getattr(Company, sort.sort_by, Company.id)
    if sort.sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    # Get total count
    count_query = query.with_only_columns(Company.id)
    total_result = await db.execute(count_query)
    total = len(total_result.all())
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.size)
    
    # Execute query
    result = await db.execute(query)
    companies = result.scalars().all()
    
    # Calculate pagination metadata
    pages = (total + pagination.size - 1) // pagination.size
    
    pagination_response = PaginationResponse(
        page=pagination.page,
        size=pagination.size,
        total=total,
        pages=pages,
        has_next=pagination.page < pages,
        has_prev=pagination.page > 1
    )
    
    return CompanyListResponse(
        companies=[CompanyResponse.from_orm(c) for c in companies],
        pagination=pagination_response
    )


@router.get("/{company_id}", response_model=CompanyResponse)
async def get_company(
    company_id: int = Path(..., description="Company ID"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get company by ID with full details"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    return CompanyResponse.from_orm(company)


@router.put("/{company_id}", response_model=CompanyResponse)
async def update_company(
    company_id: int,
    company_data: CompanyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update company information"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Check for conflicts if updating unique fields
    if company_data.asx_code and company_data.asx_code != company.asx_code:
        existing = await CompanyService.get_by_asx_code(db, company_data.asx_code)
        if existing and existing.id != company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Company with ASX code {company_data.asx_code} already exists"
            )
    
    # Update company
    updated_company = await CompanyService.update(
        db=db,
        company=company,
        company_data=company_data,
        updated_by=str(current_user.id)
    )
    
    return CompanyResponse.from_orm(updated_company)


@router.delete("/{company_id}")
async def delete_company(
    company_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Delete company (soft delete)"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Perform soft delete
    company.soft_delete()
    await db.commit()
    
    return {"message": "Company deleted successfully"}


# Company Financials endpoints
@router.post("/{company_id}/financials", response_model=CompanyFinancialsResponse)
async def create_company_financials(
    company_id: int,
    financials_data: CompanyFinancialsCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Add financial data to company"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Check for duplicate financial period
    existing = await CompanyService.get_financials_by_period(
        db, company_id, financials_data.period_end_date, financials_data.period_type
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Financial data for {financials_data.period_type} period ending {financials_data.period_end_date} already exists"
        )
    
    # Create financials
    financials = await CompanyService.create_financials(
        db=db,
        company_id=company_id,
        financials_data=financials_data,
        created_by=str(current_user.id)
    )
    
    return CompanyFinancialsResponse.from_orm(financials)


@router.get("/{company_id}/financials", response_model=List[CompanyFinancialsResponse])
async def get_company_financials(
    company_id: int,
    period_type: Optional[str] = Query(None, description="Filter by period type"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get company financial statements"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    financials = await CompanyService.get_financials_list(
        db, company_id, period_type, limit
    )
    
    return [CompanyFinancialsResponse.from_orm(f) for f in financials]


@router.put("/{company_id}/financials/{financials_id}", response_model=CompanyFinancialsResponse)
async def update_company_financials(
    company_id: int,
    financials_id: int,
    financials_data: CompanyFinancialsUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update company financial data"""
    
    financials = await CompanyService.get_financials_by_id(db, financials_id)
    if not financials or financials.company_id != company_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Financial data not found"
        )
    
    updated_financials = await CompanyService.update_financials(
        db=db,
        financials=financials,
        financials_data=financials_data,
        updated_by=str(current_user.id)
    )
    
    return CompanyFinancialsResponse.from_orm(updated_financials)


# External data synchronization
@router.post("/{company_id}/sync-asx")
async def sync_company_from_asx(
    company_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Synchronize company data from ASX"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    if not company.asx_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company must have ASX code for synchronization"
        )
    
    # Schedule background sync
    background_tasks.add_task(
        ASXService.sync_company_data,
        db, company.id, company.asx_code
    )
    
    return {"message": f"ASX data synchronization started for {company.asx_code}"}


@router.post("/{company_id}/sync-asic")
async def sync_company_from_asic(
    company_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Synchronize company data from ASIC"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    if not company.acn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company must have ACN for ASIC synchronization"
        )
    
    # Schedule background sync
    background_tasks.add_task(
        ASICService.sync_company_data,
        db, company.id, company.acn
    )
    
    return {"message": f"ASIC data synchronization started for ACN {company.acn}"}


# Bulk operations
@router.post("/bulk-update")
async def bulk_update_companies(
    bulk_data: CompanyBulkUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Bulk update multiple companies"""
    
    results = await CompanyService.bulk_update(
        db=db,
        company_ids=bulk_data.company_ids,
        update_data=bulk_data.update_data,
        updated_by=str(current_user.id)
    )
    
    return {
        "message": f"Bulk update completed",
        "success_count": results.success_count,
        "error_count": results.error_count,
        "errors": results.errors
    }


@router.get("/search/autocomplete")
async def company_autocomplete(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Company name and code autocomplete search"""
    
    suggestions = await CompanyService.autocomplete_search(db, query, limit)
    
    return {
        "suggestions": suggestions,
        "count": len(suggestions)
    }


@router.get("/statistics/overview")
async def companies_statistics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get companies statistics and overview"""
    
    stats = await CompanyService.get_statistics(db)
    
    return {
        "total_companies": stats["total"],
        "by_status": stats["by_status"],
        "by_industry": stats["by_industry"],
        "by_type": stats["by_type"],
        "recent_listings": stats["recent_listings"],
        "market_cap_distribution": stats["market_cap_distribution"]
    }


@router.get("/{company_id}/competitors")
async def get_company_competitors(
    company_id: int,
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get similar companies for competitive analysis"""
    
    company = await CompanyService.get_by_id(db, company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    competitors = await CompanyService.find_competitors(db, company, limit)
    
    return {
        "company": CompanyResponse.from_orm(company),
        "competitors": [CompanyResponse.from_orm(c) for c in competitors]
    }