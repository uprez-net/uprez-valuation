"""
Application Configuration Settings
Comprehensive settings for the IPO Valuation Platform Backend
"""
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import BaseSettings, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(PydanticBaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "uprez_valuation"
    username: str = "postgres"
    password: str = ""
    max_connections: int = 20
    pool_size: int = 5
    pool_timeout: int = 30
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(PydanticBaseSettings):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


class GCPSettings(PydanticBaseSettings):
    """Google Cloud Platform configuration"""
    project_id: str = ""
    region: str = "us-central1"
    credentials_path: Optional[str] = None
    
    # Vertex AI
    vertex_ai_endpoint: Optional[str] = None
    vertex_ai_model_name: str = "valuation-model"
    
    # Document AI
    document_ai_processor_id: str = ""
    document_ai_location: str = "us"
    
    # BigQuery
    bigquery_dataset: str = "valuation_data"
    
    # Cloud Storage
    storage_bucket: str = "uprez-valuation-data"
    
    # Pub/Sub
    pubsub_topic_valuations: str = "valuation-requests"
    pubsub_topic_documents: str = "document-processing"


class MLSettings(PydanticBaseSettings):
    """Machine Learning configuration"""
    model_cache_size: int = 100
    model_cache_ttl: int = 3600  # 1 hour
    batch_size: int = 32
    max_prediction_time: int = 300  # 5 minutes
    
    # Model paths
    dcf_model_path: str = "models/dcf_model.joblib"
    cca_model_path: str = "models/cca_model.joblib"
    sentiment_model_path: str = "models/sentiment_model"
    risk_model_path: str = "models/risk_assessment_model.joblib"


class SecuritySettings(PydanticBaseSettings):
    """Security configuration"""
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 20
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "https://app.uprez.com"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError("SECRET_KEY must be set")
        return v


class APISettings(PydanticBaseSettings):
    """API configuration"""
    title: str = "Uprez Valuation Platform API"
    description: str = "Comprehensive IPO Valuation Platform with AI/ML Services"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100
    
    # File upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = ["pdf", "doc", "docx", "txt", "html"]


class MonitoringSettings(PydanticBaseSettings):
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Sentry
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "development"
    
    # Prometheus
    metrics_enabled: bool = True
    metrics_port: int = 8080
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10


class CelerySettings(PydanticBaseSettings):
    """Celery configuration"""
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"
    task_serializer: str = "json"
    accept_content: List[str] = ["json"]
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    
    # Task routing
    task_routes: Dict[str, Dict[str, Any]] = {
        "ml_services.valuation.*": {"queue": "valuation"},
        "nlp_services.document.*": {"queue": "document_processing"},
        "gcp_integration.*": {"queue": "gcp"},
    }


class Settings(PydanticBaseSettings):
    """Main application settings"""
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Service settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    gcp: GCPSettings = GCPSettings()
    ml: MLSettings = MLSettings()
    security: SecuritySettings = SecuritySettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    celery: CelerySettings = CelerySettings()
    
    # Service discovery
    service_registry: Dict[str, str] = {
        "valuation_service": "http://localhost:8001",
        "nlp_service": "http://localhost:8002",
        "document_service": "http://localhost:8003",
    }
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()