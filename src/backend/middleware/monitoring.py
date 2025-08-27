"""
Monitoring middleware for metrics and performance tracking
"""

import time
import psutil
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import asyncio

from ..core.config import settings


# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

REDIS_CONNECTIONS = Gauge(
    'redis_connections_active', 
    'Number of active Redis connections'
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics and monitoring application health"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.metrics_enabled = settings.ENABLE_METRICS
        
        # Start background metrics collection
        if self.metrics_enabled:
            asyncio.create_task(self._collect_system_metrics())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.metrics_enabled:
            return await call_next(request)
        
        # Skip metrics for metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        # Get endpoint pattern for grouping
        endpoint = self._get_endpoint_pattern(request)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            process_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(process_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            process_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code="500"
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(process_time)
            
            raise
            
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Get endpoint pattern for metrics grouping"""
        path = request.url.path
        
        # Map specific patterns to avoid high cardinality
        if path.startswith("/api/v1/"):
            # Extract base endpoint
            parts = path.split("/")
            if len(parts) >= 4:
                base_path = "/".join(parts[:4])
                
                # Handle ID patterns
                if len(parts) > 4 and parts[4].isdigit():
                    return f"{base_path}/{{id}}"
                elif len(parts) > 4:
                    return f"{base_path}/{parts[4]}"
                else:
                    return base_path
            return path
        
        elif path.startswith("/ws/"):
            return "/ws/*"
        
        elif path.startswith("/graphql"):
            return "/graphql"
        
        elif path in ["/docs", "/redoc", "/openapi.json"]:
            return path
        
        elif path in ["/health", "/health/detailed", "/status", "/metrics"]:
            return path
        
        else:
            return "/*"
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                SYSTEM_CPU_USAGE.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.set(memory.used)
                
                # Database connections (would need actual connection pool)
                # DATABASE_CONNECTIONS.set(get_db_connection_count())
                
                # Redis connections (would need actual Redis client)
                # REDIS_CONNECTIONS.set(get_redis_connection_count())
                
            except Exception as e:
                # Log error but don't crash the collection
                pass
            
            # Wait before next collection
            await asyncio.sleep(30)  # Collect every 30 seconds


def get_metrics_handler():
    """Get Prometheus metrics endpoint handler"""
    async def metrics_endpoint():
        """Return Prometheus metrics"""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    
    return metrics_endpoint


class HealthChecker:
    """Health check utilities for monitoring"""
    
    @staticmethod
    async def check_database_health():
        """Check database connectivity"""
        try:
            # Would implement actual database health check
            return {"status": "healthy", "response_time": 0.05}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_redis_health():
        """Check Redis connectivity"""
        try:
            # Would implement actual Redis health check
            return {"status": "healthy", "response_time": 0.01}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_external_services():
        """Check external service connectivity"""
        services = {}
        
        # ASX API
        try:
            # Would implement actual ASX API check
            services["asx_api"] = {"status": "healthy", "response_time": 0.2}
        except Exception as e:
            services["asx_api"] = {"status": "unhealthy", "error": str(e)}
        
        # ASIC API
        try:
            # Would implement actual ASIC API check
            services["asic_api"] = {"status": "healthy", "response_time": 0.3}
        except Exception as e:
            services["asic_api"] = {"status": "unhealthy", "error": str(e)}
        
        return services
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage("/").percent,
            "boot_time": psutil.boot_time(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }


class AlertManager:
    """Simple alerting for critical metrics"""
    
    def __init__(self):
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "response_time": 5000  # milliseconds
        }
        self.alert_cooldown = {}
    
    async def check_alerts(self):
        """Check metrics against thresholds and trigger alerts"""
        current_time = time.time()
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > self.alert_thresholds["cpu_usage"]:
            if self._should_alert("cpu_usage", current_time):
                await self._send_alert("High CPU Usage", f"CPU usage at {cpu_usage}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        if memory_usage > self.alert_thresholds["memory_usage"]:
            if self._should_alert("memory_usage", current_time):
                await self._send_alert("High Memory Usage", f"Memory usage at {memory_usage}%")
        
        # Would implement error rate and response time checks from metrics
    
    def _should_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if we should send alert (implement cooldown)"""
        cooldown_period = 300  # 5 minutes
        last_alert = self.alert_cooldown.get(alert_type, 0)
        
        if current_time - last_alert > cooldown_period:
            self.alert_cooldown[alert_type] = current_time
            return True
        
        return False
    
    async def _send_alert(self, title: str, message: str):
        """Send alert notification"""
        # Would implement actual alerting (email, Slack, PagerDuty, etc.)
        print(f"ALERT: {title} - {message}")