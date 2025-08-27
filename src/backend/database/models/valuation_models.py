"""
Valuation Models
Core models for IPO valuations, companies, and financial data
"""
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, Enum as SQLEnum, Integer, JSON, Date, DECIMAL, Float
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base_model import BaseModel, VersionedModel


class CompanyStatus(str, Enum):
    """Company status enumeration"""
    PRIVATE = "private"
    PRE_IPO = "pre_ipo" 
    IPO_FILED = "ipo_filed"
    IPO_PRICING = "ipo_pricing"
    PUBLIC = "public"
    DELISTED = "delisted"


class ValuationStatus(str, Enum):
    """Valuation status enumeration"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ValuationMethod(str, Enum):
    """Valuation method enumeration"""
    DCF = "dcf"  # Discounted Cash Flow
    CCA = "cca"  # Comparable Company Analysis
    DTA = "dta"  # Discounted Transaction Analysis
    SUM_OF_PARTS = "sum_of_parts"
    ASSET_BASED = "asset_based"
    OPTION_PRICING = "option_pricing"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Company(BaseModel):
    """Company model"""
    
    __tablename__ = "companies"
    
    # Basic Information
    name = Column(String(200), nullable=False, index=True)
    ticker_symbol = Column(String(10), nullable=True, unique=True, index=True)
    exchange = Column(String(10), nullable=True)
    sector = Column(String(100), nullable=True, index=True)
    industry = Column(String(100), nullable=True, index=True)
    
    # Company Details
    description = Column(Text, nullable=True)
    founded_date = Column(Date, nullable=True)
    headquarters = Column(String(200), nullable=True)
    website = Column(String(500), nullable=True)
    employee_count = Column(Integer, nullable=True)
    
    # Financial Information
    market_cap = Column(DECIMAL(20, 2), nullable=True)
    revenue = Column(DECIMAL(20, 2), nullable=True)
    revenue_currency = Column(String(3), default="USD", nullable=False)
    
    # IPO Information
    status = Column(SQLEnum(CompanyStatus), default=CompanyStatus.PRIVATE, nullable=False)
    ipo_date = Column(Date, nullable=True)
    ipo_price_low = Column(DECIMAL(10, 2), nullable=True)
    ipo_price_high = Column(DECIMAL(10, 2), nullable=True)
    ipo_final_price = Column(DECIMAL(10, 2), nullable=True)
    shares_offered = Column(Integer, nullable=True)
    
    # External IDs
    cik_number = Column(String(10), nullable=True, unique=True)  # SEC CIK
    lei_code = Column(String(20), nullable=True, unique=True)    # Legal Entity Identifier
    
    # Metadata
    tags = Column(ARRAY(String), default=[], nullable=False)
    external_data_sources = Column(JSON, default={}, nullable=False)
    
    # Relationships
    valuations = relationship("Valuation", back_populates="company", cascade="all, delete-orphan")
    financial_statements = relationship("FinancialStatement", back_populates="company", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="company", cascade="all, delete-orphan")
    comparables = relationship("ComparableCompany", foreign_keys="ComparableCompany.target_company_id", back_populates="target_company")
    
    @validates('ticker_symbol')
    def validate_ticker(self, key, ticker):
        """Validate ticker symbol format"""
        if ticker:
            return ticker.upper().strip()
        return ticker
    
    @property
    def is_public(self) -> bool:
        """Check if company is public"""
        return self.status == CompanyStatus.PUBLIC
    
    @property
    def is_pre_ipo(self) -> bool:
        """Check if company is pre-IPO"""
        return self.status in [CompanyStatus.PRIVATE, CompanyStatus.PRE_IPO, CompanyStatus.IPO_FILED, CompanyStatus.IPO_PRICING]


class Valuation(VersionedModel):
    """Valuation model with versioning"""
    
    __tablename__ = "valuations"
    
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    analyst_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Valuation Details
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    valuation_date = Column(Date, default=date.today, nullable=False)
    status = Column(SQLEnum(ValuationStatus), default=ValuationStatus.DRAFT, nullable=False)
    
    # Results
    base_case_value = Column(DECIMAL(20, 2), nullable=True)
    bull_case_value = Column(DECIMAL(20, 2), nullable=True)
    bear_case_value = Column(DECIMAL(20, 2), nullable=True)
    target_price = Column(DECIMAL(10, 2), nullable=True)
    
    # Risk Assessment
    overall_risk_level = Column(SQLEnum(RiskLevel), default=RiskLevel.MEDIUM, nullable=False)
    risk_factors = Column(JSON, default=[], nullable=False)
    risk_score = Column(Float, nullable=True)  # 0-100 scale
    
    # Methodology
    primary_method = Column(SQLEnum(ValuationMethod), nullable=False)
    methods_used = Column(ARRAY(SQLEnum(ValuationMethod)), default=[], nullable=False)
    assumptions = Column(JSON, default={}, nullable=False)
    
    # Market Data
    market_conditions = Column(JSON, default={}, nullable=False)
    comparable_metrics = Column(JSON, default={}, nullable=False)
    
    # Confidence and Quality
    confidence_level = Column(Float, nullable=True)  # 0-100 scale
    data_quality_score = Column(Float, nullable=True)  # 0-100 scale
    model_accuracy = Column(Float, nullable=True)  # 0-100 scale
    
    # Review
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="valuations")
    analyst = relationship("User", foreign_keys=[analyst_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    dcf_models = relationship("DCFModel", back_populates="valuation", cascade="all, delete-orphan")
    cca_analyses = relationship("CCAAnalysis", back_populates="valuation", cascade="all, delete-orphan")
    
    @property
    def valuation_range(self) -> Dict[str, Optional[Decimal]]:
        """Get valuation range"""
        return {
            "bear": self.bear_case_value,
            "base": self.base_case_value,
            "bull": self.bull_case_value
        }
    
    @property
    def days_since_valuation(self) -> int:
        """Days since valuation was created"""
        return (date.today() - self.valuation_date).days
    
    def calculate_weighted_average(self, weights: Dict[str, float] = None) -> Optional[Decimal]:
        """Calculate weighted average of scenarios"""
        if not all([self.bear_case_value, self.base_case_value, self.bull_case_value]):
            return None
        
        weights = weights or {"bear": 0.2, "base": 0.6, "bull": 0.2}
        
        return (
            self.bear_case_value * Decimal(str(weights["bear"])) +
            self.base_case_value * Decimal(str(weights["base"])) +
            self.bull_case_value * Decimal(str(weights["bull"]))
        )


class DCFModel(BaseModel):
    """Discounted Cash Flow model"""
    
    __tablename__ = "dcf_models"
    
    valuation_id = Column(UUID(as_uuid=True), ForeignKey("valuations.id"), nullable=False)
    
    # Model Parameters
    projection_years = Column(Integer, default=5, nullable=False)
    terminal_growth_rate = Column(Float, nullable=False)
    discount_rate = Column(Float, nullable=False)  # WACC
    
    # Cash Flow Projections (JSON arrays for each year)
    revenue_projections = Column(JSON, default=[], nullable=False)
    ebitda_projections = Column(JSON, default=[], nullable=False)
    fcf_projections = Column(JSON, default=[], nullable=False)
    
    # Terminal Value
    terminal_value = Column(DECIMAL(20, 2), nullable=True)
    terminal_value_method = Column(String(50), default="perpetual_growth", nullable=False)
    
    # Results
    enterprise_value = Column(DECIMAL(20, 2), nullable=True)
    equity_value = Column(DECIMAL(20, 2), nullable=True)
    value_per_share = Column(DECIMAL(10, 2), nullable=True)
    
    # Adjustments
    cash_and_equivalents = Column(DECIMAL(20, 2), default=0, nullable=False)
    total_debt = Column(DECIMAL(20, 2), default=0, nullable=False)
    minority_interest = Column(DECIMAL(20, 2), default=0, nullable=False)
    
    # Sensitivity Analysis
    sensitivity_analysis = Column(JSON, default={}, nullable=False)
    
    # Relationships
    valuation = relationship("Valuation", back_populates="dcf_models")


class CCAAnalysis(BaseModel):
    """Comparable Company Analysis model"""
    
    __tablename__ = "cca_analyses"
    
    valuation_id = Column(UUID(as_uuid=True), ForeignKey("valuations.id"), nullable=False)
    
    # Analysis Parameters
    peer_selection_criteria = Column(JSON, default={}, nullable=False)
    multiples_used = Column(ARRAY(String), default=[], nullable=False)
    
    # Results by Multiple
    ev_revenue_median = Column(Float, nullable=True)
    ev_ebitda_median = Column(Float, nullable=True)
    pe_ratio_median = Column(Float, nullable=True)
    peg_ratio_median = Column(Float, nullable=True)
    
    # Valuation Results
    implied_valuation_low = Column(DECIMAL(20, 2), nullable=True)
    implied_valuation_high = Column(DECIMAL(20, 2), nullable=True)
    implied_valuation_median = Column(DECIMAL(20, 2), nullable=True)
    
    # Quality Metrics
    peer_correlation_score = Column(Float, nullable=True)
    data_completeness_score = Column(Float, nullable=True)
    
    # Relationships
    valuation = relationship("Valuation", back_populates="cca_analyses")
    comparable_companies = relationship("ComparableCompany", back_populates="cca_analysis", cascade="all, delete-orphan")


class ComparableCompany(BaseModel):
    """Comparable company model"""
    
    __tablename__ = "comparable_companies"
    
    cca_analysis_id = Column(UUID(as_uuid=True), ForeignKey("cca_analyses.id"), nullable=False)
    target_company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    comparable_company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    
    # Similarity Metrics
    sector_match = Column(Boolean, default=False, nullable=False)
    size_similarity = Column(Float, nullable=True)  # 0-1 scale
    geography_similarity = Column(Float, nullable=True)  # 0-1 scale
    business_model_similarity = Column(Float, nullable=True)  # 0-1 scale
    overall_similarity_score = Column(Float, nullable=True)  # 0-1 scale
    
    # Financial Multiples
    ev_revenue = Column(Float, nullable=True)
    ev_ebitda = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    peg_ratio = Column(Float, nullable=True)
    
    # Selection Rationale
    inclusion_reason = Column(Text, nullable=True)
    exclusion_flags = Column(JSON, default=[], nullable=False)
    weight_in_analysis = Column(Float, default=1.0, nullable=False)
    
    # Relationships
    cca_analysis = relationship("CCAAnalysis", back_populates="comparable_companies")
    target_company = relationship("Company", foreign_keys=[target_company_id], back_populates="comparables")
    comparable_company = relationship("Company", foreign_keys=[comparable_company_id])