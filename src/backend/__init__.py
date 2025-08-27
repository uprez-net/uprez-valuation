"""
Uprez IPO Valuation Platform Backend
Advanced ML/AI-powered backend for IPO valuations with comprehensive services

Architecture Overview:
- FastAPI-based microservices with async processing
- ML/AI services for DCF, CCA, and risk assessment models
- NLP pipeline for document processing and sentiment analysis
- GCP integration for Vertex AI, Document AI, and BigQuery ML
- Redis-based caching and Celery task queues
- PostgreSQL for data persistence with SQLAlchemy ORM
- Prometheus metrics and structured logging
- Docker containerization with multi-service deployment
"""

__version__ = "1.0.0"
__author__ = "Uprez AI Team"
__description__ = "IPO Valuation Platform Backend Services"

# Service registry
SERVICES = {
    "api": "Core API service with authentication and REST endpoints",
    "ml_services": "Machine learning services for valuation models",
    "nlp_services": "Natural language processing for document analysis", 
    "gcp_integration": "Google Cloud Platform integrations",
    "messaging": "Celery task queues for async processing",
    "database": "Data persistence layer with models and repositories",
    "utils": "Utilities for logging, metrics, and configuration"
}

# API endpoints summary
API_ENDPOINTS = {
    "/api/v1/auth": "Authentication and user management",
    "/api/v1/companies": "Company data and management",
    "/api/v1/valuations": "Valuation analysis and results",
    "/api/v1/documents": "Document upload and processing",
    "/api/v1/users": "User profile management",
    "/health": "Health check endpoint",
    "/metrics": "Prometheus metrics",
    "/docs": "API documentation"
}

# ML Models included
ML_MODELS = {
    "dcf_model": "Discounted Cash Flow valuation with Monte Carlo simulation",
    "cca_model": "Comparable Company Analysis with ML peer selection", 
    "sentiment_analyzer": "Financial sentiment analysis with FinBERT",
    "risk_analyzer": "Investment risk assessment model",
    "ocr_processor": "Document OCR with GCP Document AI integration"
}

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "SERVICES",
    "API_ENDPOINTS",
    "ML_MODELS"
]