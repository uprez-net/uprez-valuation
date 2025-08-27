"""
External integration and data synchronization models
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Numeric, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin


class DataSourceType(enum.Enum):
    """External data source types"""
    ASX = "asx"
    ASIC = "asic"
    RBA = "rba"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    REFINITIV = "refinitiv"
    XERO = "xero"
    QUICKBOOKS = "quickbooks"
    MYOB = "myob"
    FACTSET = "factset"
    MORNINGSTAR = "morningstar"
    CUSTOM_API = "custom_api"


class SyncStatus(enum.Enum):
    """Data synchronization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class WebhookStatus(enum.Enum):
    """Webhook endpoint status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DISABLED = "disabled"


class ExternalDataSource(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """External data source configurations"""
    
    __tablename__ = "external_data_sources"
    
    # Data source identification
    name = Column(String(255), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    description = Column(Text, nullable=True)
    
    # Connection details
    base_url = Column(String(500), nullable=False)
    api_version = Column(String(20), nullable=True)
    
    # Authentication
    auth_type = Column(String(50), nullable=False)  # api_key, oauth, basic, custom
    api_key = Column(String(500), nullable=True)  # Encrypted
    oauth_token = Column(Text, nullable=True)  # Encrypted OAuth token
    oauth_refresh_token = Column(Text, nullable=True)  # Encrypted
    oauth_expires_at = Column(DateTime, nullable=True)
    username = Column(String(100), nullable=True)
    password = Column(String(255), nullable=True)  # Encrypted
    
    # Configuration
    config = Column(JSON, default=dict, nullable=False)
    headers = Column(JSON, default=dict, nullable=True)
    rate_limit = Column(Integer, nullable=True)  # Requests per minute
    
    # Status and monitoring
    is_active = Column(Boolean, default=True, nullable=False)
    last_sync = Column(DateTime, nullable=True)
    last_successful_sync = Column(DateTime, nullable=True)
    sync_frequency = Column(String(50), nullable=True)  # hourly, daily, weekly, etc.
    
    # Quality metrics
    success_rate = Column(Numeric(5, 2), nullable=True)  # Percentage
    avg_response_time = Column(Integer, nullable=True)  # Milliseconds
    error_count = Column(Integer, default=0, nullable=False)
    
    # Data mapping
    field_mappings = Column(JSON, default=dict, nullable=True)
    data_transformations = Column(JSON, default=list, nullable=True)
    
    # Relationships
    sync_logs = relationship("DataSync", back_populates="data_source", cascade="all, delete-orphan")
    webhook_endpoints = relationship("WebhookEndpoint", back_populates="data_source", cascade="all, delete-orphan")
    
    @property
    def is_oauth_expired(self) -> bool:
        """Check if OAuth token is expired"""
        return self.oauth_expires_at and self.oauth_expires_at < datetime.utcnow()
    
    def increment_error_count(self):
        """Increment error counter"""
        self.error_count += 1
    
    def reset_error_count(self):
        """Reset error counter after successful sync"""
        self.error_count = 0
    
    def update_success_rate(self, successful: bool):
        """Update success rate based on sync result"""
        # Implementation would calculate rolling success rate
        pass
    
    def get_config_value(self, key: str, default=None):
        """Get configuration value with fallback"""
        return self.config.get(key, default) if self.config else default


class DataSync(Base, TimestampMixin, AuditMixin):
    """Data synchronization logs and status"""
    
    __tablename__ = "data_syncs"
    
    data_source_id = Column(Integer, ForeignKey("external_data_sources.id"), nullable=False)
    
    # Sync details
    sync_type = Column(String(50), nullable=False)  # full, incremental, delta
    status = Column(Enum(SyncStatus), default=SyncStatus.PENDING, nullable=False)
    
    # Sync timeline
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)  # Seconds
    
    # Data statistics
    records_requested = Column(Integer, default=0, nullable=False)
    records_processed = Column(Integer, default=0, nullable=False)
    records_created = Column(Integer, default=0, nullable=False)
    records_updated = Column(Integer, default=0, nullable=False)
    records_failed = Column(Integer, default=0, nullable=False)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    
    # Sync metadata
    sync_params = Column(JSON, default=dict, nullable=True)
    response_metadata = Column(JSON, default=dict, nullable=True)
    
    # Data quality
    data_quality_score = Column(Numeric(3, 2), nullable=True)  # 0-5 scale
    validation_errors = Column(JSON, default=list, nullable=True)
    
    # Relationships
    data_source = relationship("ExternalDataSource", back_populates="sync_logs")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this sync"""
        if self.records_requested == 0:
            return 0.0
        return (self.records_processed - self.records_failed) / self.records_requested * 100
    
    @property
    def is_retryable(self) -> bool:
        """Check if sync can be retried"""
        return self.status == SyncStatus.FAILED and self.retry_count < self.max_retries
    
    def mark_completed(self):
        """Mark sync as completed"""
        self.status = SyncStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.duration = int((self.completed_at - self.started_at).total_seconds())
    
    def mark_failed(self, error_message: str, error_details: dict = None):
        """Mark sync as failed with error details"""
        self.status = SyncStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_details = error_details or {}
        self.duration = int((self.completed_at - self.started_at).total_seconds())


class WebhookEndpoint(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """Webhook endpoints for real-time data updates"""
    
    __tablename__ = "webhook_endpoints"
    
    data_source_id = Column(Integer, ForeignKey("external_data_sources.id"), nullable=True)
    
    # Endpoint details
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False)
    method = Column(String(10), default="POST", nullable=False)
    
    # Authentication
    secret_key = Column(String(255), nullable=True)  # For signature verification
    auth_header = Column(String(100), nullable=True)
    auth_token = Column(String(500), nullable=True)  # Encrypted
    
    # Configuration
    events = Column(JSON, default=list, nullable=False)  # Array of event types
    headers = Column(JSON, default=dict, nullable=True)
    timeout = Column(Integer, default=30, nullable=False)  # Seconds
    retry_policy = Column(JSON, default=dict, nullable=True)
    
    # Status and monitoring
    status = Column(Enum(WebhookStatus), default=WebhookStatus.ACTIVE, nullable=False)
    last_triggered = Column(DateTime, nullable=True)
    last_successful = Column(DateTime, nullable=True)
    
    # Statistics
    total_requests = Column(Integer, default=0, nullable=False)
    successful_requests = Column(Integer, default=0, nullable=False)
    failed_requests = Column(Integer, default=0, nullable=False)
    avg_response_time = Column(Integer, nullable=True)  # Milliseconds
    
    # Filtering
    filters = Column(JSON, default=dict, nullable=True)
    data_format = Column(String(20), default="json", nullable=False)  # json, xml, form
    
    # Relationships
    data_source = relationship("ExternalDataSource", back_populates="webhook_endpoints")
    deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")
    
    @property
    def success_rate(self) -> float:
        """Calculate webhook success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def record_success(self, response_time: int = None):
        """Record successful webhook delivery"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_successful = datetime.utcnow()
        self.last_triggered = datetime.utcnow()
        
        if response_time:
            # Update rolling average response time
            if self.avg_response_time:
                self.avg_response_time = int((self.avg_response_time + response_time) / 2)
            else:
                self.avg_response_time = response_time
    
    def record_failure(self):
        """Record failed webhook delivery"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_triggered = datetime.utcnow()


class WebhookDelivery(Base, TimestampMixin):
    """Individual webhook delivery attempts"""
    
    __tablename__ = "webhook_deliveries"
    
    webhook_id = Column(Integer, ForeignKey("webhook_endpoints.id"), nullable=False)
    
    # Delivery details
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    
    # Request details
    request_headers = Column(JSON, nullable=True)
    request_method = Column(String(10), nullable=False)
    request_url = Column(String(500), nullable=False)
    
    # Response details
    response_status = Column(Integer, nullable=True)
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)
    response_time = Column(Integer, nullable=True)  # Milliseconds
    
    # Delivery status
    success = Column(Boolean, default=False, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Retry information
    attempt_number = Column(Integer, default=1, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    next_retry_at = Column(DateTime, nullable=True)
    
    # Relationships
    webhook = relationship("WebhookEndpoint", back_populates="deliveries")
    
    @property
    def can_retry(self) -> bool:
        """Check if delivery can be retried"""
        return not self.success and self.attempt_number < self.max_attempts
    
    def schedule_retry(self, delay_minutes: int = 5):
        """Schedule retry for failed delivery"""
        if self.can_retry:
            self.next_retry_at = datetime.utcnow() + timedelta(minutes=delay_minutes)


class APIRateLimit(Base, TimestampMixin):
    """API rate limiting tracking"""
    
    __tablename__ = "api_rate_limits"
    
    # Rate limit identification
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    endpoint = Column(String(255), nullable=False)
    
    # Rate limiting
    requests_count = Column(Integer, default=0, nullable=False)
    window_start = Column(DateTime, default=datetime.utcnow, nullable=False)
    window_size = Column(Integer, default=3600, nullable=False)  # Seconds
    limit_per_window = Column(Integer, default=1000, nullable=False)
    
    # Status
    is_blocked = Column(Boolean, default=False, nullable=False)
    blocked_until = Column(DateTime, nullable=True)
    
    @property
    def remaining_requests(self) -> int:
        """Get remaining requests in current window"""
        return max(0, self.limit_per_window - self.requests_count)
    
    @property
    def is_limit_exceeded(self) -> bool:
        """Check if rate limit is exceeded"""
        return self.requests_count >= self.limit_per_window
    
    def reset_window(self):
        """Reset rate limiting window"""
        self.requests_count = 0
        self.window_start = datetime.utcnow()
        self.is_blocked = False
        self.blocked_until = None
    
    def increment_requests(self) -> bool:
        """Increment request counter and check if limit exceeded"""
        # Reset window if expired
        if datetime.utcnow() - self.window_start >= timedelta(seconds=self.window_size):
            self.reset_window()
        
        self.requests_count += 1
        
        if self.is_limit_exceeded:
            self.is_blocked = True
            self.blocked_until = self.window_start + timedelta(seconds=self.window_size)
            return False
        
        return True