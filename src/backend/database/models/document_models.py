"""
Document Models
Models for document processing, storage, and analysis
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, Enum as SQLEnum, Integer, JSON, LargeBinary, Float
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base_model import BaseModel


class DocumentType(str, Enum):
    """Document type enumeration"""
    PROSPECTUS = "prospectus"
    FINANCIAL_STATEMENT = "financial_statement"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    RESEARCH_REPORT = "research_report"
    NEWS_ARTICLE = "news_article"
    REGULATORY_FILING = "regulatory_filing"
    PRESS_RELEASE = "press_release"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class Document(BaseModel):
    """Document storage and metadata model"""
    
    __tablename__ = "documents"
    
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Document Information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    document_type = Column(SQLEnum(DocumentType), nullable=False)
    
    # File Details
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_extension = Column(String(10), nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256
    
    # Storage Information
    storage_path = Column(String(1000), nullable=False)
    storage_bucket = Column(String(100), nullable=True)
    storage_provider = Column(String(50), default="gcp", nullable=False)
    
    # Processing Status
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.UPLOADED, nullable=False)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)
    processing_attempts = Column(Integer, default=0, nullable=False)
    
    # Document Metadata
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    
    # Classification and Tags
    tags = Column(ARRAY(String), default=[], nullable=False)
    categories = Column(ARRAY(String), default=[], nullable=False)
    confidence_scores = Column(JSON, default={}, nullable=False)
    
    # Access Control
    is_public = Column(Boolean, default=False, nullable=False)
    access_permissions = Column(JSON, default={}, nullable=False)
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    uploader = relationship("User")
    extracted_text = relationship("DocumentText", back_populates="document", uselist=False, cascade="all, delete-orphan")
    extracted_entities = relationship("ExtractedEntity", back_populates="document", cascade="all, delete-orphan")
    sentiment_analysis = relationship("SentimentAnalysis", back_populates="document", cascade="all, delete-orphan")
    
    @validates('file_extension')
    def validate_extension(self, key, extension):
        """Validate and normalize file extension"""
        if extension:
            return extension.lower().lstrip('.')
        return extension
    
    @property
    def is_processing_complete(self) -> bool:
        """Check if processing is complete"""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    @property
    def processing_duration_seconds(self) -> Optional[int]:
        """Calculate processing duration in seconds"""
        if self.processing_started_at and self.processing_completed_at:
            return int((self.processing_completed_at - self.processing_started_at).total_seconds())
        return None


class DocumentText(BaseModel):
    """Extracted text content from documents"""
    
    __tablename__ = "document_texts"
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, unique=True)
    
    # Extracted Content
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    structured_content = Column(JSON, default={}, nullable=False)
    
    # OCR Information
    ocr_engine = Column(String(50), nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    ocr_language = Column(String(10), nullable=True)
    
    # Text Analysis
    readability_score = Column(Float, nullable=True)
    complexity_score = Column(Float, nullable=True)
    
    # Processing Metadata
    extraction_method = Column(String(50), nullable=False)
    extraction_timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="extracted_text")
    
    @property
    def word_count(self) -> int:
        """Count words in cleaned text"""
        if self.cleaned_text:
            return len(self.cleaned_text.split())
        return 0
    
    @property
    def character_count(self) -> int:
        """Count characters in cleaned text"""
        if self.cleaned_text:
            return len(self.cleaned_text)
        return 0


class ExtractedEntity(BaseModel):
    """Named entities extracted from documents"""
    
    __tablename__ = "extracted_entities"
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Entity Information
    entity_text = Column(String(500), nullable=False)
    entity_type = Column(String(50), nullable=False)  # PERSON, ORG, MONEY, etc.
    confidence_score = Column(Float, nullable=True)
    
    # Position Information
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    
    # Context
    context_before = Column(String(200), nullable=True)
    context_after = Column(String(200), nullable=True)
    
    # Normalization
    normalized_value = Column(String(500), nullable=True)
    canonical_form = Column(String(500), nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="extracted_entities")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results for documents"""
    
    __tablename__ = "sentiment_analyses"
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Overall Sentiment
    overall_sentiment = Column(String(20), nullable=False)  # positive, negative, neutral
    overall_score = Column(Float, nullable=False)  # -1 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    
    # Detailed Scores
    positive_score = Column(Float, nullable=False)
    negative_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    
    # Aspect-based Sentiment
    aspect_sentiments = Column(JSON, default={}, nullable=False)  # {aspect: {sentiment, score}}
    
    # Risk Sentiment
    risk_sentiment_score = Column(Float, nullable=True)  # Higher = more risky
    opportunity_sentiment_score = Column(Float, nullable=True)  # Higher = more opportunity
    
    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="sentiment_analysis")


class FinancialStatement(BaseModel):
    """Financial statement model"""
    
    __tablename__ = "financial_statements"
    
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    
    # Statement Information
    statement_type = Column(String(50), nullable=False)  # income, balance, cash_flow
    period_end_date = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)  # annual, quarterly
    fiscal_year = Column(Integer, nullable=False)
    fiscal_quarter = Column(Integer, nullable=True)
    
    # Currency
    reporting_currency = Column(String(3), default="USD", nullable=False)
    
    # Financial Data (JSON structure for flexibility)
    financial_data = Column(JSON, nullable=False)
    
    # Data Quality
    data_completeness = Column(Float, nullable=True)  # 0-1
    data_confidence = Column(Float, nullable=True)   # 0-1
    extraction_method = Column(String(50), nullable=True)
    
    # Audit Information
    is_audited = Column(Boolean, nullable=True)
    auditor_name = Column(String(200), nullable=True)
    audit_opinion = Column(String(50), nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="financial_statements")
    source_document = relationship("Document")
    
    @property
    def is_annual(self) -> bool:
        """Check if this is an annual statement"""
        return self.period_type == "annual"
    
    @property
    def is_quarterly(self) -> bool:
        """Check if this is a quarterly statement"""
        return self.period_type == "quarterly"
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get a specific financial metric"""
        return self.financial_data.get(metric_name)
    
    def set_metric(self, metric_name: str, value: float):
        """Set a specific financial metric"""
        if self.financial_data is None:
            self.financial_data = {}
        self.financial_data[metric_name] = value