"""
Valuation models and calculations
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Numeric, Date, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin


class ValuationMethod(enum.Enum):
    """Valuation methodology enumeration"""
    DCF = "dcf"  # Discounted Cash Flow
    COMPARABLE_COMPANIES = "comparable_companies"
    PRECEDENT_TRANSACTIONS = "precedent_transactions"
    ASSET_BASED = "asset_based"
    SUM_OF_PARTS = "sum_of_parts"
    DIVIDEND_DISCOUNT = "dividend_discount"
    RESIDUAL_INCOME = "residual_income"
    REAL_OPTIONS = "real_options"


class ValuationStatus(enum.Enum):
    """Valuation calculation status"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    ARCHIVED = "archived"


class ScenarioType(enum.Enum):
    """Valuation scenario types"""
    BASE = "base"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    STRESSED = "stressed"
    MONTE_CARLO = "monte_carlo"


class ValuationModel(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """Valuation model templates and configurations"""
    
    __tablename__ = "valuation_models"
    
    # Model identification
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    method = Column(Enum(ValuationMethod), nullable=False)
    version = Column(String(20), nullable=False, default="1.0")
    
    # Model configuration
    industry_specific = Column(String(100), nullable=True)
    complexity_level = Column(String(20), default="intermediate", nullable=False)  # basic, intermediate, advanced
    
    # Model parameters (JSON)
    parameters = Column(JSON, default=dict, nullable=False)
    assumptions = Column(JSON, default=dict, nullable=False)
    formulas = Column(JSON, default=dict, nullable=False)
    
    # Model metadata
    template = Column(Boolean, default=False, nullable=False)
    public = Column(Boolean, default=False, nullable=False)
    featured = Column(Boolean, default=False, nullable=False)
    
    # Usage statistics
    usage_count = Column(Integer, default=0, nullable=False)
    avg_rating = Column(Numeric(3, 2), nullable=True)
    
    # Documentation
    methodology_notes = Column(Text, nullable=True)
    limitations = Column(Text, nullable=True)
    references = Column(Text, nullable=True)  # JSON array of references
    
    # Relationships
    results = relationship("ValuationResult", back_populates="model")
    
    def increment_usage(self):
        """Increment usage counter"""
        self.usage_count += 1
    
    def get_parameter_value(self, key: str, default=None):
        """Get parameter value with fallback"""
        return self.parameters.get(key, default) if self.parameters else default
    
    def set_parameter(self, key: str, value: Any):
        """Set parameter value"""
        if not self.parameters:
            self.parameters = {}
        self.parameters[key] = value


class ValuationResult(Base, TimestampMixin, AuditMixin, UUIDMixin):
    """Valuation calculation results"""
    
    __tablename__ = "valuation_results"
    
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("valuation_models.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Calculation details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(ValuationStatus), default=ValuationStatus.DRAFT, nullable=False)
    
    # Valuation date and data
    valuation_date = Column(Date, default=date.today, nullable=False)
    data_as_of_date = Column(Date, nullable=True)  # Date of financial data used
    
    # Core valuation results
    enterprise_value = Column(Numeric(20, 2), nullable=True)
    equity_value = Column(Numeric(20, 2), nullable=True)
    value_per_share = Column(Numeric(10, 4), nullable=True)
    
    # Valuation range
    min_value_per_share = Column(Numeric(10, 4), nullable=True)
    max_value_per_share = Column(Numeric(10, 4), nullable=True)
    target_price = Column(Numeric(10, 4), nullable=True)
    
    # Market comparison
    current_price = Column(Numeric(10, 4), nullable=True)
    upside_downside = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Key assumptions (JSON)
    assumptions = Column(JSON, default=dict, nullable=False)
    inputs = Column(JSON, default=dict, nullable=False)
    calculations = Column(JSON, default=dict, nullable=False)
    
    # Sensitivity analysis
    sensitivity_analysis = Column(JSON, default=dict, nullable=True)
    key_drivers = Column(JSON, default=list, nullable=True)
    
    # Quality metrics
    confidence_level = Column(String(20), nullable=True)  # high, medium, low
    data_quality_score = Column(Numeric(3, 2), nullable=True)  # 0-5 scale
    model_accuracy = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Review and approval
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    
    # Notes and documentation
    methodology_notes = Column(Text, nullable=True)
    key_risks = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    
    # External benchmarks
    peer_comparison = Column(JSON, default=dict, nullable=True)
    industry_multiples = Column(JSON, default=dict, nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="valuation_results")
    model = relationship("ValuationModel", back_populates="results")
    user = relationship("User")
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    approver = relationship("User", foreign_keys=[approved_by])
    scenarios = relationship("ValuationScenario", back_populates="result", cascade="all, delete-orphan")
    
    @property
    def is_stale(self) -> bool:
        """Check if valuation is more than 30 days old"""
        return (date.today() - self.valuation_date).days > 30
    
    @property
    def recommendation(self) -> str:
        """Get investment recommendation based on upside/downside"""
        if not self.upside_downside:
            return "Hold"
        
        if self.upside_downside > 20:
            return "Strong Buy"
        elif self.upside_downside > 10:
            return "Buy"
        elif self.upside_downside > -10:
            return "Hold"
        elif self.upside_downside > -20:
            return "Sell"
        else:
            return "Strong Sell"
    
    def calculate_upside_downside(self):
        """Calculate upside/downside vs current price"""
        if self.target_price and self.current_price:
            self.upside_downside = ((self.target_price - self.current_price) / self.current_price) * 100
    
    def get_assumption(self, key: str, default=None):
        """Get assumption value with fallback"""
        return self.assumptions.get(key, default) if self.assumptions else default


class ValuationScenario(Base, TimestampMixin, AuditMixin):
    """Different valuation scenarios (base, optimistic, pessimistic)"""
    
    __tablename__ = "valuation_scenarios"
    
    result_id = Column(Integer, ForeignKey("valuation_results.id"), nullable=False)
    
    # Scenario details
    name = Column(String(100), nullable=False)
    scenario_type = Column(Enum(ScenarioType), nullable=False)
    probability = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Scenario-specific values
    enterprise_value = Column(Numeric(20, 2), nullable=True)
    equity_value = Column(Numeric(20, 2), nullable=True)
    value_per_share = Column(Numeric(10, 4), nullable=True)
    
    # Scenario assumptions
    assumptions = Column(JSON, default=dict, nullable=False)
    key_variables = Column(JSON, default=dict, nullable=False)
    
    # Risk adjustments
    risk_premium = Column(Numeric(5, 2), nullable=True)  # Percentage
    discount_rate = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Monte Carlo specific
    iterations = Column(Integer, nullable=True)
    confidence_intervals = Column(JSON, nullable=True)
    distribution_params = Column(JSON, nullable=True)
    
    # Relationships
    result = relationship("ValuationResult", back_populates="scenarios")
    
    def get_confidence_interval(self, confidence: float = 0.95):
        """Get confidence interval for Monte Carlo scenarios"""
        if self.scenario_type == ScenarioType.MONTE_CARLO and self.confidence_intervals:
            return self.confidence_intervals.get(str(confidence))
        return None


class ValuationComparable(Base, TimestampMixin, AuditMixin):
    """Comparable companies for valuation analysis"""
    
    __tablename__ = "valuation_comparables"
    
    result_id = Column(Integer, ForeignKey("valuation_results.id"), nullable=False)
    comparable_company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # Comparable selection criteria
    similarity_score = Column(Numeric(3, 2), nullable=True)  # 0-1 scale
    selection_rationale = Column(Text, nullable=True)
    
    # Multiples at time of valuation
    pe_ratio = Column(Numeric(8, 2), nullable=True)
    ev_ebitda = Column(Numeric(8, 2), nullable=True)
    ev_revenue = Column(Numeric(8, 2), nullable=True)
    price_book = Column(Numeric(8, 2), nullable=True)
    price_sales = Column(Numeric(8, 2), nullable=True)
    
    # Growth metrics
    revenue_growth = Column(Numeric(5, 2), nullable=True)  # Percentage
    ebitda_growth = Column(Numeric(5, 2), nullable=True)  # Percentage
    
    # Financial metrics
    market_cap = Column(Numeric(20, 2), nullable=True)
    enterprise_value = Column(Numeric(20, 2), nullable=True)
    
    # Adjustments
    size_adjustment = Column(Numeric(5, 2), nullable=True)  # Percentage
    liquidity_adjustment = Column(Numeric(5, 2), nullable=True)  # Percentage
    other_adjustments = Column(Text, nullable=True)  # JSON
    
    # Data as of date
    data_date = Column(Date, nullable=True)
    
    # Relationships
    result = relationship("ValuationResult")
    comparable_company = relationship("Company")


class ValuationAuditLog(Base, TimestampMixin):
    """Audit log for valuation changes"""
    
    __tablename__ = "valuation_audit_logs"
    
    result_id = Column(Integer, ForeignKey("valuation_results.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Change details
    action = Column(String(50), nullable=False)  # created, updated, reviewed, approved
    field_name = Column(String(100), nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    
    # Context
    reason = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    result = relationship("ValuationResult")
    user = relationship("User")