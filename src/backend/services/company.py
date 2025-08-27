"""
Company service layer for business logic
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..models.company import Company, CompanyFinancials, CompanyDocument, Industry
from ..models.base import BulkOperationResult
from ..schemas.company import CompanyCreate, CompanyUpdate, CompanyFinancialsCreate, CompanyFinancialsUpdate


class CompanyService:
    """Service for company-related operations"""
    
    @staticmethod
    async def create(
        db: AsyncSession,
        company_data: CompanyCreate,
        created_by: str
    ) -> Company:
        """Create a new company"""
        
        # Convert Pydantic model to dict
        company_dict = company_data.dict(exclude_unset=True)
        
        # Create company instance
        company = Company(**company_dict)
        company.created_by = created_by
        
        db.add(company)
        await db.commit()
        await db.refresh(company)
        
        return company
    
    @staticmethod
    async def get_by_id(db: AsyncSession, company_id: int) -> Optional[Company]:
        """Get company by ID"""
        query = select(Company).options(
            selectinload(Company.financials),
            selectinload(Company.documents)
        ).where(
            and_(Company.id == company_id, Company.is_active == True)
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_asx_code(db: AsyncSession, asx_code: str) -> Optional[Company]:
        """Get company by ASX code"""
        query = select(Company).where(
            and_(Company.asx_code == asx_code.upper(), Company.is_active == True)
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_acn(db: AsyncSession, acn: str) -> Optional[Company]:
        """Get company by ACN"""
        query = select(Company).where(
            and_(Company.acn == acn, Company.is_active == True)
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update(
        db: AsyncSession,
        company: Company,
        company_data: CompanyUpdate,
        updated_by: str
    ) -> Company:
        """Update company information"""
        
        # Update fields
        update_dict = company_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(company, field, value)
        
        company.updated_by = updated_by
        company.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(company)
        
        return company
    
    @staticmethod
    async def create_financials(
        db: AsyncSession,
        company_id: int,
        financials_data: CompanyFinancialsCreate,
        created_by: str
    ) -> CompanyFinancials:
        """Create financial data for company"""
        
        financials_dict = financials_data.dict(exclude_unset=True)
        
        financials = CompanyFinancials(
            company_id=company_id,
            **financials_dict
        )
        financials.created_by = created_by
        
        # Calculate financial ratios
        financials.calculate_ratios()
        
        db.add(financials)
        await db.commit()
        await db.refresh(financials)
        
        return financials
    
    @staticmethod
    async def get_financials_by_period(
        db: AsyncSession,
        company_id: int,
        period_end_date: date,
        period_type: str
    ) -> Optional[CompanyFinancials]:
        """Get financial data by period"""
        
        query = select(CompanyFinancials).where(
            and_(
                CompanyFinancials.company_id == company_id,
                CompanyFinancials.period_end_date == period_end_date,
                CompanyFinancials.period_type == period_type,
                CompanyFinancials.is_active == True
            )
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_financials_by_id(
        db: AsyncSession,
        financials_id: int
    ) -> Optional[CompanyFinancials]:
        """Get financial data by ID"""
        
        query = select(CompanyFinancials).where(
            and_(
                CompanyFinancials.id == financials_id,
                CompanyFinancials.is_active == True
            )
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_financials_list(
        db: AsyncSession,
        company_id: int,
        period_type: Optional[str] = None,
        limit: int = 10
    ) -> List[CompanyFinancials]:
        """Get list of financial statements for company"""
        
        query = select(CompanyFinancials).where(
            and_(
                CompanyFinancials.company_id == company_id,
                CompanyFinancials.is_active == True
            )
        )
        
        if period_type:
            query = query.where(CompanyFinancials.period_type == period_type)
        
        query = query.order_by(desc(CompanyFinancials.period_end_date)).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def update_financials(
        db: AsyncSession,
        financials: CompanyFinancials,
        financials_data: CompanyFinancialsUpdate,
        updated_by: str
    ) -> CompanyFinancials:
        """Update financial data"""
        
        # Update fields
        update_dict = financials_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(financials, field, value)
        
        financials.updated_by = updated_by
        financials.updated_at = datetime.utcnow()
        
        # Recalculate ratios
        financials.calculate_ratios()
        
        await db.commit()
        await db.refresh(financials)
        
        return financials
    
    @staticmethod
    async def autocomplete_search(
        db: AsyncSession,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Autocomplete search for companies"""
        
        search_query = select(Company).where(
            and_(
                Company.is_active == True,
                or_(
                    Company.name.ilike(f"%{query}%"),
                    Company.asx_code.ilike(f"%{query}%")
                )
            )
        ).order_by(
            Company.name
        ).limit(limit)
        
        result = await db.execute(search_query)
        companies = result.scalars().all()
        
        return [
            {
                "id": company.id,
                "name": company.name,
                "asx_code": company.asx_code,
                "industry": company.industry.value if company.industry else None
            }
            for company in companies
        ]
    
    @staticmethod
    async def get_statistics(db: AsyncSession) -> Dict[str, Any]:
        """Get companies statistics"""
        
        # Total companies
        total_query = select(func.count(Company.id)).where(Company.is_active == True)
        total_result = await db.execute(total_query)
        total = total_result.scalar()
        
        # By status
        status_query = select(
            Company.status,
            func.count(Company.id)
        ).where(
            Company.is_active == True
        ).group_by(Company.status)
        
        status_result = await db.execute(status_query)
        by_status = {status.value: count for status, count in status_result.all()}
        
        # By industry
        industry_query = select(
            Company.industry,
            func.count(Company.id)
        ).where(
            and_(Company.is_active == True, Company.industry.isnot(None))
        ).group_by(Company.industry)
        
        industry_result = await db.execute(industry_query)
        by_industry = {industry.value: count for industry, count in industry_result.all()}
        
        # By type
        type_query = select(
            Company.company_type,
            func.count(Company.id)
        ).where(
            Company.is_active == True
        ).group_by(Company.company_type)
        
        type_result = await db.execute(type_query)
        by_type = {comp_type.value: count for comp_type, count in type_result.all()}
        
        # Recent listings (last 30 days)
        recent_listings_query = select(Company).where(
            and_(
                Company.is_active == True,
                Company.listing_date >= date.today() - datetime.timedelta(days=30)
            )
        ).order_by(desc(Company.listing_date)).limit(5)
        
        recent_result = await db.execute(recent_listings_query)
        recent_listings = recent_result.scalars().all()
        
        # Market cap distribution
        market_cap_ranges = [
            ("0-10M", 0, 10_000_000),
            ("10M-100M", 10_000_000, 100_000_000),
            ("100M-1B", 100_000_000, 1_000_000_000),
            ("1B+", 1_000_000_000, float('inf'))
        ]
        
        market_cap_distribution = {}
        for range_name, min_cap, max_cap in market_cap_ranges:
            if max_cap == float('inf'):
                cap_query = select(func.count(Company.id)).where(
                    and_(
                        Company.is_active == True,
                        Company.market_cap >= min_cap
                    )
                )
            else:
                cap_query = select(func.count(Company.id)).where(
                    and_(
                        Company.is_active == True,
                        Company.market_cap >= min_cap,
                        Company.market_cap < max_cap
                    )
                )
            
            cap_result = await db.execute(cap_query)
            market_cap_distribution[range_name] = cap_result.scalar()
        
        return {
            "total": total,
            "by_status": by_status,
            "by_industry": by_industry,
            "by_type": by_type,
            "recent_listings": recent_listings,
            "market_cap_distribution": market_cap_distribution
        }
    
    @staticmethod
    async def find_competitors(
        db: AsyncSession,
        company: Company,
        limit: int = 10
    ) -> List[Company]:
        """Find competitor companies based on industry and size"""
        
        query = select(Company).where(
            and_(
                Company.is_active == True,
                Company.id != company.id,
                Company.industry == company.industry
            )
        )
        
        # Order by market cap similarity if available
        if company.market_cap:
            query = query.order_by(
                func.abs(Company.market_cap - company.market_cap)
            )
        else:
            query = query.order_by(Company.name)
        
        query = query.limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def bulk_update(
        db: AsyncSession,
        company_ids: List[int],
        update_data: CompanyUpdate,
        updated_by: str
    ) -> BulkOperationResult:
        """Bulk update multiple companies"""
        
        success_count = 0
        error_count = 0
        errors = []
        
        for company_id in company_ids:
            try:
                company = await CompanyService.get_by_id(db, company_id)
                if company:
                    await CompanyService.update(db, company, update_data, updated_by)
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Company {company_id} not found")
            
            except Exception as e:
                error_count += 1
                errors.append(f"Company {company_id}: {str(e)}")
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
            total_processed=len(company_ids)
        )