"""
Metrics Collection
Prometheus metrics for monitoring and observability
"""
import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

# Database metrics
DB_CONNECTIONS = Gauge(
    'database_connections',
    'Number of active database connections'
)

DB_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type']
)

DB_QUERY_COUNT = Counter(
    'database_queries_total',
    'Total number of database queries',
    ['query_type', 'status']
)

# ML/AI metrics
ML_INFERENCE_COUNT = Counter(
    'ml_inference_total',
    'Total number of ML inferences',
    ['model_name', 'status']
)

ML_INFERENCE_DURATION = Histogram(
    'ml_inference_duration_seconds',
    'ML inference duration in seconds',
    ['model_name']
)

ML_MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current ML model accuracy',
    ['model_name', 'metric']
)

# Business metrics
VALUATIONS_CREATED = Counter(
    'valuations_created_total',
    'Total number of valuations created',
    ['method', 'status']
)

DOCUMENTS_PROCESSED = Counter(
    'documents_processed_total',
    'Total number of documents processed',
    ['document_type', 'status']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'severity']
)

# Application info
APP_INFO = Info(
    'application_info',
    'Application information'
)


def setup_metrics():
    """Setup and initialize metrics"""
    
    # Set application info
    APP_INFO.info({
        'version': settings.api.version,
        'environment': settings.environment,
        'name': settings.api.title
    })
    
    # Start metrics server if enabled
    if settings.monitoring.metrics_enabled:
        try:
            start_http_server(settings.monitoring.metrics_port)
            logger.info(f"Metrics server started on port {settings.monitoring.metrics_port}")
        except Exception as e:
            logger.warning("Failed to start metrics server", error=str(e))


def track_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Track HTTP request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def track_db_query(query_type: str, duration: float, success: bool = True):
    """Track database query metrics"""
    status = "success" if success else "error"
    DB_QUERY_COUNT.labels(query_type=query_type, status=status).inc()
    DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)


def track_ml_inference(model_name: str, duration: float, success: bool = True):
    """Track ML inference metrics"""
    status = "success" if success else "error"
    ML_INFERENCE_COUNT.labels(model_name=model_name, status=status).inc()
    ML_INFERENCE_DURATION.labels(model_name=model_name).observe(duration)


def track_error(error_type: str, severity: str = "error"):
    """Track error occurrence"""
    ERROR_COUNT.labels(error_type=error_type, severity=severity).inc()


def track_valuation_created(method: str, success: bool = True):
    """Track valuation creation"""
    status = "success" if success else "error"
    VALUATIONS_CREATED.labels(method=method, status=status).inc()


def track_document_processed(document_type: str, success: bool = True):
    """Track document processing"""
    status = "success" if success else "error"
    DOCUMENTS_PROCESSED.labels(document_type=document_type, status=status).inc()


def track_cache_operation(cache_type: str, hit: bool):
    """Track cache hit/miss"""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


# Decorators for automatic metrics tracking
def track_time(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Decorator to track execution time"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                raise
        
        return wrapper
    return decorator


def track_async_time(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Decorator to track async execution time"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                raise
        
        return wrapper
    return decorator


def track_counter(metric: Counter, labels: Optional[Dict[str, str]] = None):
    """Decorator to increment counter on function call"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
                return result
            except Exception:
                if labels:
                    error_labels = labels.copy()
                    error_labels['status'] = 'error'
                    metric.labels(**error_labels).inc()
                else:
                    metric.inc()
                raise
        
        return wrapper
    return decorator


class MetricsCollector:
    """Centralized metrics collector"""
    
    def __init__(self):
        self.custom_metrics = {}
    
    def create_counter(self, name: str, description: str, labels: list = None) -> Counter:
        """Create a new counter metric"""
        labels = labels or []
        metric = Counter(name, description, labels)
        self.custom_metrics[name] = metric
        return metric
    
    def create_histogram(self, name: str, description: str, labels: list = None) -> Histogram:
        """Create a new histogram metric"""
        labels = labels or []
        metric = Histogram(name, description, labels)
        self.custom_metrics[name] = metric
        return metric
    
    def create_gauge(self, name: str, description: str, labels: list = None) -> Gauge:
        """Create a new gauge metric"""
        labels = labels or []
        metric = Gauge(name, description, labels)
        self.custom_metrics[name] = metric
        return metric
    
    def get_metric(self, name: str):
        """Get a metric by name"""
        return self.custom_metrics.get(name)
    
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        import psutil
        
        # CPU usage
        cpu_gauge = self.create_gauge('system_cpu_usage_percent', 'CPU usage percentage')
        cpu_gauge.set(psutil.cpu_percent())
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_gauge = self.create_gauge('system_memory_usage_percent', 'Memory usage percentage')
        memory_gauge.set(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_gauge = self.create_gauge('system_disk_usage_percent', 'Disk usage percentage')
        disk_gauge.set((disk.used / disk.total) * 100)


# Global metrics collector
metrics_collector = MetricsCollector()


class BusinessMetrics:
    """Business-specific metrics tracking"""
    
    @staticmethod
    def track_user_activity(user_id: str, activity: str):
        """Track user activity"""
        user_activity = Counter(
            'user_activity_total',
            'User activity count',
            ['user_id', 'activity']
        )
        user_activity.labels(user_id=user_id, activity=activity).inc()
    
    @staticmethod
    def track_valuation_accuracy(method: str, accuracy: float):
        """Track valuation method accuracy"""
        ML_MODEL_ACCURACY.labels(model_name=method, metric="accuracy").set(accuracy)
    
    @staticmethod
    def track_processing_time(process_type: str, duration: float):
        """Track processing time for different operations"""
        processing_time = Histogram(
            'processing_duration_seconds',
            'Processing duration in seconds',
            ['process_type']
        )
        processing_time.labels(process_type=process_type).observe(duration)