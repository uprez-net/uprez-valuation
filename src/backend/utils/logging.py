"""
Logging Configuration
Structured logging setup for the application
"""
import logging
import logging.config
import sys
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

from ..config import settings


def setup_logging():
    """Setup structured logging configuration"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.JSONRenderer() if settings.monitoring.log_format == "json" else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.monitoring.log_level.upper())),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    
    # Configure standard library logging
    if settings.monitoring.log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Setup handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.getLevelName(settings.monitoring.log_level.upper()))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(settings.monitoring.log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    configure_logger_levels()
    
    # Setup Sentry if configured
    if settings.monitoring.sentry_dsn:
        setup_sentry()


def configure_logger_levels():
    """Configure specific logger levels"""
    logger_config = {
        'uvicorn': logging.INFO,
        'uvicorn.error': logging.INFO,
        'uvicorn.access': logging.WARNING if not settings.debug else logging.INFO,
        'sqlalchemy.engine': logging.WARNING if not settings.debug else logging.INFO,
        'sqlalchemy.pool': logging.WARNING,
        'alembic': logging.INFO,
        'redis': logging.WARNING,
        'celery': logging.INFO,
        'httpx': logging.WARNING,
        'google.cloud': logging.WARNING,
    }
    
    for logger_name, level in logger_config.items():
        logging.getLogger(logger_name).setLevel(level)


def setup_sentry():
    """Setup Sentry error tracking"""
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.redis import RedisIntegration
        
        sentry_sdk.init(
            dsn=settings.monitoring.sentry_dsn,
            environment=settings.monitoring.sentry_environment,
            integrations=[
                FastApiIntegration(auto_enable=True),
                SqlalchemyIntegration(),
                RedisIntegration(),
            ],
            traces_sample_rate=0.1 if settings.environment == "production" else 1.0,
            debug=settings.debug,
            attach_stacktrace=True,
            send_default_pii=False,
        )
        
        structlog.get_logger(__name__).info("Sentry initialized", environment=settings.monitoring.sentry_environment)
        
    except ImportError:
        structlog.get_logger(__name__).warning("Sentry SDK not installed")


class StructuredLogger:
    """Structured logger wrapper for consistent logging"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.name = name
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind context variables to logger"""
        new_logger = StructuredLogger(self.name)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name)


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records"""
    
    def filter(self, record):
        # Add request ID if available
        import contextvars
        
        try:
            request_id = contextvars.copy_context().get('request_id', None)
            if request_id:
                record.request_id = request_id
        except:
            pass
        
        return True


# Audit logging for security events
class AuditLogger:
    """Specialized logger for audit events"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_authentication(self, user_id: str, method: str, success: bool, ip_address: str, user_agent: str = None):
        """Log authentication attempt"""
        self.logger.info(
            "Authentication attempt",
            user_id=user_id,
            method=method,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            event_type="authentication"
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, granted: bool):
        """Log authorization decision"""
        self.logger.info(
            "Authorization check",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            event_type="authorization"
        )
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, action: str):
        """Log data access"""
        self.logger.info(
            "Data access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            event_type="data_access"
        )
    
    def log_admin_action(self, admin_id: str, action: str, target: str, details: Dict[str, Any] = None):
        """Log administrative action"""
        self.logger.info(
            "Administrative action",
            admin_id=admin_id,
            action=action,
            target=target,
            details=details or {},
            event_type="admin_action"
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security event"""
        self.logger.warning(
            f"Security event: {event_type}",
            event_type="security",
            security_event_type=event_type,
            severity=severity,
            **details
        )


# Global audit logger instance
audit_logger = AuditLogger()