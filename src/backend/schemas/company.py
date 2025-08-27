"""
Company-related Pydantic schemas for request/response validation
"""

from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator

from ..models.base import BaseSchema, TimestampSchema, PaginationResponse
from ..models.company import CompanyStatus, CompanyType, Industry, DocumentType


class CompanyBase(BaseModel):
    """Base company schema"""
    name: str = Field(..., min_length=1, max_length=255)
    legal_name: Optional[str] = Field(None, max_length=255)
    asx_code: Optional[str] = Field(None, max_length=10)
    acn: Optional[str] = Field(None, max_length=20)
    abn: Optional[str] = Field(None, max_length=20)
    company_type: CompanyType = CompanyType.PRIVATE
    industry: Optional[Industry] = None
    sector: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    website: Optional[str] = Field(None, max_length=255)


class CompanyCreate(CompanyBase):
    """Schema for creating a new company"""
    pass


class CompanyUpdate(BaseModel):
    """Schema for updating company information"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    legal_name: Optional[str] = Field(None, max_length=255)
    asx_code: Optional[str] = Field(None, max_length=10)
    acn: Optional[str] = Field(None, max_length=20)
    abn: Optional[str] = Field(None, max_length=20)
    company_type: Optional[CompanyType] = None
    status: Optional[CompanyStatus] = None
    industry: Optional[Industry] = None
    sector: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    website: Optional[str] = Field(None, max_length=255)
    address_line1: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=50)
    country: Optional[str] = Field(None, max_length=50)


class CompanyResponse(BaseSchema, CompanyBase):
    """Company response schema"""
    id: int
    status: CompanyStatus
    trading_code: Optional[str]
    incorporation_date: Optional[date]
    listing_date: Optional[date]
    market_cap: Optional[Decimal]
    shares_outstanding: Optional[Decimal]
    created_at: datetime
    updated_at: datetime
    
    # Latest financial metrics
    latest_revenue: Optional[Decimal] = None
    latest_ebitda: Optional[Decimal] = None
    latest_net_income: Optional[Decimal] = None


class CompanyFinancialsBase(BaseModel):
    """Base financial data schema"""
    period_type: str = Field(..., regex="^(annual|quarterly|half-yearly)$")
    period_start_date: date
    period_end_date: date
    fiscal_year: int
    currency: str = Field("AUD", max_length=3)
    
    # Income Statement
    revenue: Optional[Decimal] = None
    gross_profit: Optional[Decimal] = None
    operating_profit: Optional[Decimal] = None
    ebitda: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    
    # Balance Sheet
    total_assets: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    total_equity: Optional[Decimal] = None
    cash_and_equivalents: Optional[Decimal] = None
    total_debt: Optional[Decimal] = None
    
    # Cash Flow
    operating_cash_flow: Optional[Decimal] = None
    free_cash_flow: Optional[Decimal] = None
    capex: Optional[Decimal] = None
    
    # Share Data
    shares_outstanding: Optional[Decimal] = None
    eps_basic: Optional[Decimal] = None
    eps_diluted: Optional[Decimal] = None
    dps: Optional[Decimal] = None


class CompanyFinancialsCreate(CompanyFinancialsBase):
    """Schema for creating financial data"""
    pass


class CompanyFinancialsUpdate(BaseModel):
    """Schema for updating financial data"""
    revenue: Optional[Decimal] = None
    gross_profit: Optional[Decimal] = None
    operating_profit: Optional[Decimal] = None
    ebitda: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    total_assets: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    total_equity: Optional[Decimal] = None
    cash_and_equivalents: Optional[Decimal] = None
    total_debt: Optional[Decimal] = None
    operating_cash_flow: Optional[Decimal] = None
    free_cash_flow: Optional[Decimal] = None
    audited: Optional[bool] = None


class CompanyFinancialsResponse(BaseSchema, CompanyFinancialsBase):
    """Financial data response schema"""
    id: int
    company_id: int
    
    # Calculated ratios
    gross_margin: Optional[Decimal]
    operating_margin: Optional[Decimal]
    net_margin: Optional[Decimal]
    roa: Optional[Decimal]
    roe: Optional[Decimal]
    current_ratio: Optional[Decimal]
    debt_to_equity: Optional[Decimal]
    
    audited: bool
    data_source: Optional[str]
    created_at: datetime
    updated_at: datetime


class CompanyDocumentBase(BaseModel):
    """Base document schema"""
    title: str = Field(..., max_length=255)
    document_type: DocumentType
    description: Optional[str] = None
    period_start_date: Optional[date] = None
    period_end_date: Optional[date] = None
    filing_date: Optional[date] = None


class CompanyDocumentResponse(BaseSchema, CompanyDocumentBase):
    """Document response schema"""
    id: int
    company_id: int
    filename: str
    file_size: Optional[int]
    mime_type: Optional[str]
    processed: bool
    page_count: Optional[int]
    created_at: datetime


class CompanySearchFilters(BaseModel):
    """Filters for company search"""
    name: Optional[str] = None
    asx_code: Optional[str] = None
    industry: Optional[Industry] = None
    status: Optional[CompanyStatus] = None
    company_type: Optional[CompanyType] = None
    listing_date_from: Optional[date] = None
    listing_date_to: Optional[date] = None
    market_cap_min: Optional[Decimal] = None
    market_cap_max: Optional[Decimal] = None
    search: Optional[str] = None  # General search across multiple fields


class CompanyListResponse(BaseModel):
    """Paginated company list response"""
    companies: List[CompanyResponse]
    pagination: PaginationResponse


class CompanyBulkUpdate(BaseModel):
    """Bulk update schema"""
    company_ids: List[int] = Field(..., min_items=1, max_items=100)
    update_data: CompanyUpdate


class CompanyAutocompleteResponse(BaseModel):
    """Autocomplete suggestion"""
    id: int
    name: str
    asx_code: Optional[str]
    industry: Optional[str]


class CompanyStatistics(BaseModel):
    """Company statistics response"""
    total: int
    by_status: Dict[str, int]
    by_industry: Dict[str, int]
    by_type: Dict[str, int]
    recent_listings: List[CompanyResponse]
    market_cap_distribution: Dict[str, int]