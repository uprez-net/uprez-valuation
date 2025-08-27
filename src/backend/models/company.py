"""
Company and financial data models
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Numeric, Date, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin


class CompanyStatus(enum.Enum):
    """Company status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive" 
    DELISTED = "delisted"
    SUSPENDED = "suspended"
    PENDING_IPO = "pending_ipo"


class CompanyType(enum.Enum):
    """Company type enumeration"""
    PUBLIC = "public"
    PRIVATE = "private"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"


class Industry(enum.Enum):
    """Industry classification"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    TELECOMMUNICATIONS = "telecommunications"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class DocumentType(enum.Enum):
    """Document type enumeration"""
    PROSPECTUS = "prospectus"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    FINANCIAL_STATEMENTS = "financial_statements"
    PRESENTATION = "presentation"
    ANNOUNCEMENT = "announcement"
    OTHER = "other"


class Company(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """Company master data"""
    
    __tablename__ = "companies"
    
    # Basic company information
    name = Column(String(255), nullable=False, index=True)
    legal_name = Column(String(255), nullable=True)
    trading_code = Column(String(10), nullable=True, index=True)  # ASX code
    asx_code = Column(String(10), nullable=True, index=True)
    acn = Column(String(20), nullable=True, index=True)  # Australian Company Number
    abn = Column(String(20), nullable=True, index=True)  # Australian Business Number
    
    # Company details
    company_type = Column(Enum(CompanyType), default=CompanyType.PRIVATE, nullable=False)
    status = Column(Enum(CompanyStatus), default=CompanyStatus.ACTIVE, nullable=False)
    industry = Column(Enum(Industry), nullable=True)
    sector = Column(String(100), nullable=True)
    sub_sector = Column(String(100), nullable=True)
    
    # Business information
    description = Column(Text, nullable=True)
    business_model = Column(Text, nullable=True)
    competitive_advantages = Column(Text, nullable=True)
    risk_factors = Column(Text, nullable=True)
    
    # Contact information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(50), default="Australia", nullable=False)
    
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    website = Column(String(255), nullable=True)
    
    # Key dates
    incorporation_date = Column(Date, nullable=True)
    listing_date = Column(Date, nullable=True)
    ipo_date = Column(Date, nullable=True)
    delisting_date = Column(Date, nullable=True)
    
    # Market data
    market_cap = Column(Numeric(20, 2), nullable=True)
    shares_outstanding = Column(Numeric(20, 0), nullable=True)
    free_float = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Key personnel
    ceo_name = Column(String(100), nullable=True)
    cfo_name = Column(String(100), nullable=True)
    chairman_name = Column(String(100), nullable=True)
    
    # External identifiers
    reuters_code = Column(String(20), nullable=True)
    bloomberg_code = Column(String(20), nullable=True)
    isin = Column(String(20), nullable=True)
    gics_code = Column(String(20), nullable=True)
    
    # Relationships
    financials = relationship("CompanyFinancials", back_populates="company", cascade="all, delete-orphan")
    documents = relationship("CompanyDocument", back_populates="company", cascade="all, delete-orphan")
    valuation_results = relationship("ValuationResult", back_populates="company")
    
    @property
    def current_price(self) -> Optional[Decimal]:
        """Get current stock price"""
        # This would typically fetch from external data source
        return None
    
    @property
    def latest_financials(self):
        """Get most recent financial statements"""
        return (
            self.financials
            .filter(CompanyFinancials.is_active == True)
            .order_by(CompanyFinancials.period_end_date.desc())
            .first()
        )


class CompanyFinancials(Base, TimestampMixin, AuditMixin):
    """Company financial statements"""
    
    __tablename__ = "company_financials"
    
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # Period information
    period_type = Column(String(20), nullable=False)  # annual, quarterly, half-yearly
    period_start_date = Column(Date, nullable=False)
    period_end_date = Column(Date, nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    currency = Column(String(3), default="AUD", nullable=False)
    
    # Income Statement
    revenue = Column(Numeric(20, 2), nullable=True)
    gross_profit = Column(Numeric(20, 2), nullable=True)
    operating_profit = Column(Numeric(20, 2), nullable=True)
    ebitda = Column(Numeric(20, 2), nullable=True)
    ebit = Column(Numeric(20, 2), nullable=True)
    net_income = Column(Numeric(20, 2), nullable=True)
    
    # Per share metrics
    eps_basic = Column(Numeric(10, 4), nullable=True)
    eps_diluted = Column(Numeric(10, 4), nullable=True)
    dps = Column(Numeric(10, 4), nullable=True)  # Dividends per share
    
    # Balance Sheet - Assets
    total_assets = Column(Numeric(20, 2), nullable=True)
    current_assets = Column(Numeric(20, 2), nullable=True)
    cash_and_equivalents = Column(Numeric(20, 2), nullable=True)
    accounts_receivable = Column(Numeric(20, 2), nullable=True)
    inventory = Column(Numeric(20, 2), nullable=True)
    ppe_net = Column(Numeric(20, 2), nullable=True)  # Property, Plant & Equipment
    intangible_assets = Column(Numeric(20, 2), nullable=True)
    goodwill = Column(Numeric(20, 2), nullable=True)
    
    # Balance Sheet - Liabilities
    total_liabilities = Column(Numeric(20, 2), nullable=True)
    current_liabilities = Column(Numeric(20, 2), nullable=True)
    accounts_payable = Column(Numeric(20, 2), nullable=True)
    short_term_debt = Column(Numeric(20, 2), nullable=True)
    long_term_debt = Column(Numeric(20, 2), nullable=True)
    total_debt = Column(Numeric(20, 2), nullable=True)
    
    # Balance Sheet - Equity
    total_equity = Column(Numeric(20, 2), nullable=True)
    retained_earnings = Column(Numeric(20, 2), nullable=True)
    share_capital = Column(Numeric(20, 2), nullable=True)
    
    # Cash Flow Statement
    operating_cash_flow = Column(Numeric(20, 2), nullable=True)
    investing_cash_flow = Column(Numeric(20, 2), nullable=True)
    financing_cash_flow = Column(Numeric(20, 2), nullable=True)
    free_cash_flow = Column(Numeric(20, 2), nullable=True)
    capex = Column(Numeric(20, 2), nullable=True)
    
    # Share information
    shares_outstanding = Column(Numeric(20, 0), nullable=True)
    weighted_avg_shares = Column(Numeric(20, 0), nullable=True)
    weighted_avg_shares_diluted = Column(Numeric(20, 0), nullable=True)
    
    # Key ratios (calculated)
    gross_margin = Column(Numeric(5, 2), nullable=True)
    operating_margin = Column(Numeric(5, 2), nullable=True)
    net_margin = Column(Numeric(5, 2), nullable=True)
    roa = Column(Numeric(5, 2), nullable=True)  # Return on Assets
    roe = Column(Numeric(5, 2), nullable=True)  # Return on Equity
    current_ratio = Column(Numeric(5, 2), nullable=True)
    debt_to_equity = Column(Numeric(5, 2), nullable=True)
    
    # Data quality
    audited = Column(Boolean, default=False, nullable=False)
    restated = Column(Boolean, default=False, nullable=False)
    data_source = Column(String(50), nullable=True)
    filing_date = Column(Date, nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="financials")
    
    def calculate_ratios(self):
        """Calculate financial ratios from raw data"""
        if self.revenue and self.revenue > 0:
            self.gross_margin = (self.gross_profit / self.revenue * 100) if self.gross_profit else None
            self.operating_margin = (self.operating_profit / self.revenue * 100) if self.operating_profit else None
            self.net_margin = (self.net_income / self.revenue * 100) if self.net_income else None
        
        if self.total_assets and self.total_assets > 0:
            self.roa = (self.net_income / self.total_assets * 100) if self.net_income else None
        
        if self.total_equity and self.total_equity > 0:
            self.roe = (self.net_income / self.total_equity * 100) if self.net_income else None
            self.debt_to_equity = (self.total_debt / self.total_equity) if self.total_debt else None
        
        if self.current_liabilities and self.current_liabilities > 0:
            self.current_ratio = (self.current_assets / self.current_liabilities) if self.current_assets else None


class CompanyDocument(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Company documents and filings"""
    
    __tablename__ = "company_documents"
    
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # Document details
    title = Column(String(255), nullable=False)
    document_type = Column(Enum(DocumentType), nullable=False)
    description = Column(Text, nullable=True)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Document metadata
    period_start_date = Column(Date, nullable=True)
    period_end_date = Column(Date, nullable=True)
    filing_date = Column(Date, nullable=True)
    published_date = Column(Date, nullable=True)
    
    # Processing status
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Content analysis
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    extracted_text = Column(Text, nullable=True)
    key_metrics = Column(Text, nullable=True)  # JSON of extracted metrics
    
    # External references
    asx_announcement_id = Column(String(50), nullable=True)
    asic_document_id = Column(String(50), nullable=True)
    external_url = Column(String(500), nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    
    @property
    def file_extension(self) -> str:
        """Get file extension"""
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ''
    
    @property
    def is_financial_document(self) -> bool:
        """Check if document contains financial data"""
        return self.document_type in [
            DocumentType.ANNUAL_REPORT,
            DocumentType.QUARTERLY_REPORT,
            DocumentType.FINANCIAL_STATEMENTS
        ]