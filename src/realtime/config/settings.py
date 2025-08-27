"""
Real-time Collaboration System Configuration
Centralized settings for WebSocket connections, Redis, performance, and scaling
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class WebSocketConfig:
    """WebSocket server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections_per_workspace: int = 100
    connection_timeout_seconds: int = 30
    heartbeat_interval_seconds: int = 30
    message_queue_size: int = 1000
    auto_reconnect: bool = True
    reconnect_interval_seconds: int = 3
    max_reconnect_attempts: int = 10
    
    # Message handling
    max_message_size_bytes: int = 1024 * 1024  # 1MB
    message_rate_limit: int = 100  # messages per minute
    batch_message_interval_ms: int = 50
    compression_enabled: bool = True
    compression_threshold_bytes: int = 1024

@dataclass
class RedisConfig:
    """Redis configuration for pub/sub and caching"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = None
    ssl: bool = False
    connection_pool_size: int = 20
    connection_timeout_seconds: int = 5
    socket_keepalive: bool = True
    
    # Pub/Sub settings
    pubsub_patterns: List[str] = None
    message_ttl_seconds: int = 3600
    max_message_history: int = 1000
    
    # Clustering (for production)
    cluster_enabled: bool = False
    cluster_nodes: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.pubsub_patterns is None:
            self.pubsub_patterns = ["workspace:*", "document:*", "user:*"]
        if self.cluster_nodes is None:
            self.cluster_nodes = []

@dataclass
class PerformanceConfig:
    """Performance monitoring and optimization settings"""
    # Monitoring
    metrics_collection_enabled: bool = True
    metrics_retention_seconds: int = 3600
    system_monitor_interval_seconds: float = 5.0
    performance_analysis_enabled: bool = True
    
    # Thresholds for bottleneck detection
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    response_time_threshold_ms: float = 1000.0
    error_rate_threshold_percent: float = 5.0
    connection_count_threshold: int = 10000
    message_queue_threshold: int = 1000
    
    # Optimization
    auto_scaling_enabled: bool = True
    scale_up_cpu_threshold: float = 70.0
    scale_down_cpu_threshold: float = 30.0
    scale_up_memory_threshold: float = 80.0
    scale_down_memory_threshold: float = 40.0
    min_instances: int = 1
    max_instances: int = 10
    
    # Cleanup
    cleanup_interval_seconds: int = 300
    stale_connection_timeout_seconds: int = 300
    old_metrics_cleanup_days: int = 7

@dataclass
class CollaborationConfig:
    """Collaboration features configuration"""
    # Comments
    max_comment_length: int = 5000
    max_comments_per_document: int = 1000
    comment_auto_resolve_days: int = 30
    
    # Annotations
    max_annotation_length: int = 2000
    max_annotations_per_document: int = 500
    supported_annotation_types: List[str] = None
    
    # Presence
    presence_update_interval_ms: int = 1000
    presence_timeout_seconds: int = 300
    cursor_debounce_ms: int = 100
    selection_debounce_ms: int = 150
    
    # Activity feed
    activity_feed_max_entries: int = 1000
    activity_retention_days: int = 30
    
    def __post_init__(self):
        if self.supported_annotation_types is None:
            self.supported_annotation_types = ["highlight", "note", "bookmark", "suggestion", "question"]

@dataclass
class OperationalTransformConfig:
    """Operational Transform engine configuration"""
    # Document handling
    max_document_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_operations_per_delta: int = 100
    version_history_limit: int = 1000
    conflict_resolution_strategy: str = "last_writer_wins"  # or "operational_transform"
    
    # Performance
    operation_batch_size: int = 50
    delta_merge_threshold: int = 10
    document_lock_timeout_seconds: int = 5
    
    # Validation
    validate_operations: bool = True
    strict_position_validation: bool = True
    allow_empty_operations: bool = False

@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    # JWT settings
    jwt_secret_key: str = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS settings
    cors_allowed_origins: List[str] = None
    cors_allow_credentials: bool = True
    cors_allowed_methods: List[str] = None
    cors_allowed_headers: List[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    # Input validation
    sanitize_input: bool = True
    max_input_length: int = 10000
    allowed_file_types: List[str] = None
    
    def __post_init__(self):
        if self.cors_allowed_origins is None:
            self.cors_allowed_origins = ["http://localhost:3000", "http://localhost:8000"]
        if self.cors_allowed_methods is None:
            self.cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.cors_allowed_headers is None:
            self.cors_allowed_headers = ["*"]
        if self.allowed_file_types is None:
            self.allowed_file_types = [".json", ".txt", ".md"]

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Component-specific logging
    websocket_debug: bool = False
    redis_debug: bool = False
    performance_debug: bool = False
    collaboration_debug: bool = False
    
    # Structured logging
    structured_logging: bool = True
    json_logging: bool = False
    include_request_id: bool = True

class Settings:
    """Main settings class that loads configuration from environment variables"""
    
    def __init__(self, environment: Environment = None):
        self.environment = environment or Environment(os.getenv("ENVIRONMENT", "development"))
        
        # Load configurations
        self.websocket = self._load_websocket_config()
        self.redis = self._load_redis_config()
        self.performance = self._load_performance_config()
        self.collaboration = self._load_collaboration_config()
        self.operational_transform = self._load_ot_config()
        self.security = self._load_security_config()
        self.logging = self._load_logging_config()
    
    def _load_websocket_config(self) -> WebSocketConfig:
        """Load WebSocket configuration from environment"""
        return WebSocketConfig(
            host=os.getenv("WS_HOST", "0.0.0.0"),
            port=int(os.getenv("WS_PORT", "8000")),
            max_connections_per_workspace=int(os.getenv("WS_MAX_CONNECTIONS_PER_WORKSPACE", "100")),
            connection_timeout_seconds=int(os.getenv("WS_CONNECTION_TIMEOUT", "30")),
            heartbeat_interval_seconds=int(os.getenv("WS_HEARTBEAT_INTERVAL", "30")),
            message_queue_size=int(os.getenv("WS_MESSAGE_QUEUE_SIZE", "1000")),
            auto_reconnect=os.getenv("WS_AUTO_RECONNECT", "true").lower() == "true",
            reconnect_interval_seconds=int(os.getenv("WS_RECONNECT_INTERVAL", "3")),
            max_reconnect_attempts=int(os.getenv("WS_MAX_RECONNECT_ATTEMPTS", "10")),
            max_message_size_bytes=int(os.getenv("WS_MAX_MESSAGE_SIZE", str(1024 * 1024))),
            message_rate_limit=int(os.getenv("WS_MESSAGE_RATE_LIMIT", "100")),
            batch_message_interval_ms=int(os.getenv("WS_BATCH_INTERVAL", "50")),
            compression_enabled=os.getenv("WS_COMPRESSION", "true").lower() == "true",
            compression_threshold_bytes=int(os.getenv("WS_COMPRESSION_THRESHOLD", "1024"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment"""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            connection_pool_size=int(os.getenv("REDIS_POOL_SIZE", "20")),
            connection_timeout_seconds=int(os.getenv("REDIS_CONNECTION_TIMEOUT", "5")),
            socket_keepalive=os.getenv("REDIS_KEEPALIVE", "true").lower() == "true",
            message_ttl_seconds=int(os.getenv("REDIS_MESSAGE_TTL", "3600")),
            max_message_history=int(os.getenv("REDIS_MAX_HISTORY", "1000")),
            cluster_enabled=os.getenv("REDIS_CLUSTER", "false").lower() == "true"
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration from environment"""
        return PerformanceConfig(
            metrics_collection_enabled=os.getenv("PERF_METRICS_ENABLED", "true").lower() == "true",
            metrics_retention_seconds=int(os.getenv("PERF_METRICS_RETENTION", "3600")),
            system_monitor_interval_seconds=float(os.getenv("PERF_MONITOR_INTERVAL", "5.0")),
            performance_analysis_enabled=os.getenv("PERF_ANALYSIS_ENABLED", "true").lower() == "true",
            cpu_threshold_percent=float(os.getenv("PERF_CPU_THRESHOLD", "80.0")),
            memory_threshold_percent=float(os.getenv("PERF_MEMORY_THRESHOLD", "85.0")),
            disk_threshold_percent=float(os.getenv("PERF_DISK_THRESHOLD", "90.0")),
            response_time_threshold_ms=float(os.getenv("PERF_RESPONSE_TIME_THRESHOLD", "1000.0")),
            error_rate_threshold_percent=float(os.getenv("PERF_ERROR_RATE_THRESHOLD", "5.0")),
            auto_scaling_enabled=os.getenv("AUTO_SCALING_ENABLED", "true").lower() == "true",
            scale_up_cpu_threshold=float(os.getenv("SCALE_UP_CPU_THRESHOLD", "70.0")),
            scale_down_cpu_threshold=float(os.getenv("SCALE_DOWN_CPU_THRESHOLD", "30.0")),
            min_instances=int(os.getenv("MIN_INSTANCES", "1")),
            max_instances=int(os.getenv("MAX_INSTANCES", "10")),
            cleanup_interval_seconds=int(os.getenv("CLEANUP_INTERVAL", "300")),
            stale_connection_timeout_seconds=int(os.getenv("STALE_CONNECTION_TIMEOUT", "300"))
        )
    
    def _load_collaboration_config(self) -> CollaborationConfig:
        """Load collaboration configuration from environment"""
        return CollaborationConfig(
            max_comment_length=int(os.getenv("COLLAB_MAX_COMMENT_LENGTH", "5000")),
            max_comments_per_document=int(os.getenv("COLLAB_MAX_COMMENTS_PER_DOC", "1000")),
            comment_auto_resolve_days=int(os.getenv("COLLAB_COMMENT_AUTO_RESOLVE_DAYS", "30")),
            max_annotation_length=int(os.getenv("COLLAB_MAX_ANNOTATION_LENGTH", "2000")),
            max_annotations_per_document=int(os.getenv("COLLAB_MAX_ANNOTATIONS_PER_DOC", "500")),
            presence_update_interval_ms=int(os.getenv("COLLAB_PRESENCE_INTERVAL", "1000")),
            presence_timeout_seconds=int(os.getenv("COLLAB_PRESENCE_TIMEOUT", "300")),
            cursor_debounce_ms=int(os.getenv("COLLAB_CURSOR_DEBOUNCE", "100")),
            selection_debounce_ms=int(os.getenv("COLLAB_SELECTION_DEBOUNCE", "150")),
            activity_feed_max_entries=int(os.getenv("COLLAB_ACTIVITY_MAX_ENTRIES", "1000")),
            activity_retention_days=int(os.getenv("COLLAB_ACTIVITY_RETENTION_DAYS", "30"))
        )
    
    def _load_ot_config(self) -> OperationalTransformConfig:
        """Load Operational Transform configuration from environment"""
        return OperationalTransformConfig(
            max_document_size_bytes=int(os.getenv("OT_MAX_DOC_SIZE", str(10 * 1024 * 1024))),
            max_operations_per_delta=int(os.getenv("OT_MAX_OPS_PER_DELTA", "100")),
            version_history_limit=int(os.getenv("OT_VERSION_HISTORY_LIMIT", "1000")),
            conflict_resolution_strategy=os.getenv("OT_CONFLICT_RESOLUTION", "operational_transform"),
            operation_batch_size=int(os.getenv("OT_BATCH_SIZE", "50")),
            delta_merge_threshold=int(os.getenv("OT_DELTA_MERGE_THRESHOLD", "10")),
            document_lock_timeout_seconds=int(os.getenv("OT_DOC_LOCK_TIMEOUT", "5")),
            validate_operations=os.getenv("OT_VALIDATE_OPERATIONS", "true").lower() == "true",
            strict_position_validation=os.getenv("OT_STRICT_VALIDATION", "true").lower() == "true",
            allow_empty_operations=os.getenv("OT_ALLOW_EMPTY_OPS", "false").lower() == "true"
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration from environment"""
        jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not jwt_secret and self.environment == Environment.PRODUCTION:
            raise ValueError("JWT_SECRET_KEY must be set in production")
        
        return SecurityConfig(
            jwt_secret_key=jwt_secret or "dev-secret-key",
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
            rate_limit_burst_size=int(os.getenv("RATE_LIMIT_BURST", "10")),
            sanitize_input=os.getenv("SANITIZE_INPUT", "true").lower() == "true",
            max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "10000"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment"""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE", "100")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            websocket_debug=os.getenv("LOG_WEBSOCKET_DEBUG", "false").lower() == "true",
            redis_debug=os.getenv("LOG_REDIS_DEBUG", "false").lower() == "true",
            performance_debug=os.getenv("LOG_PERFORMANCE_DEBUG", "false").lower() == "true",
            collaboration_debug=os.getenv("LOG_COLLABORATION_DEBUG", "false").lower() == "true",
            structured_logging=os.getenv("LOG_STRUCTURED", "true").lower() == "true",
            json_logging=os.getenv("LOG_JSON", "false").lower() == "true",
            include_request_id=os.getenv("LOG_INCLUDE_REQUEST_ID", "true").lower() == "true"
        )
    
    def get_database_url(self) -> str:
        """Get database URL for the environment"""
        if self.environment == Environment.PRODUCTION:
            return os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/uprez_prod")
        elif self.environment == Environment.STAGING:
            return os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/uprez_staging")
        else:
            return os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/uprez_dev")
    
    def get_redis_url(self) -> str:
        """Get Redis URL for the environment"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.database}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.database}"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary"""
        return {
            "environment": self.environment.value,
            "websocket": self.websocket.__dict__,
            "redis": self.redis.__dict__,
            "performance": self.performance.__dict__,
            "collaboration": self.collaboration.__dict__,
            "operational_transform": self.operational_transform.__dict__,
            "security": {
                **self.security.__dict__,
                "jwt_secret_key": "[REDACTED]" if self.security.jwt_secret_key else None
            },
            "logging": self.logging.__dict__
        }

# Global settings instance
settings = Settings()