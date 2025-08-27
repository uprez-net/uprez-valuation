"""
Logging middleware for request/response logging
"""

import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings


# Configure structured logging
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.sensitive_headers = {
            "authorization", "x-api-key", "cookie", "x-auth-token"
        }
        self.sensitive_paths = {
            "/api/v1/auth/login", "/api/v1/auth/register", 
            "/api/v1/auth/password/reset", "/api/v1/auth/password/change"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get request ID from security middleware
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log incoming request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, request_id, process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, request_id, process_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""
        
        # Get request body for POST/PUT requests (if not too large)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Only log body if content-type is JSON and size is reasonable
                content_type = request.headers.get("content-type", "")
                content_length = int(request.headers.get("content-length", 0))
                
                if "application/json" in content_type and content_length < 10000:
                    body = await request.body()
                    if body:
                        body = body.decode("utf-8")
                        # Don't log sensitive data
                        if request.url.path not in self.sensitive_paths:
                            try:
                                body_json = json.loads(body)
                                # Remove password fields
                                if isinstance(body_json, dict):
                                    body_json.pop("password", None)
                                    body_json.pop("current_password", None) 
                                    body_json.pop("new_password", None)
                                body = json.dumps(body_json)
                            except:
                                body = "[non-json body]"
                        else:
                            body = "[sensitive data hidden]"
                else:
                    body = f"[{content_type} - {content_length} bytes]"
                    
            except Exception:
                body = "[body read error]"
        
        # Prepare log data
        log_data = {
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "headers": self._filter_sensitive_headers(dict(request.headers)),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "body": body,
            "timestamp": time.time()
        }
        
        # Log as JSON for structured logging
        logger.info(
            f"Request {request_id} - {request.method} {request.url.path}",
            extra={"log_data": log_data}
        )
    
    async def _log_response(self, request: Request, response: Response, 
                           request_id: str, process_time: float):
        """Log response details"""
        
        # Get response body for errors or if explicitly requested
        body = None
        if response.status_code >= 400:
            # Try to get error response body
            try:
                if hasattr(response, 'body'):
                    body = response.body.decode("utf-8") if response.body else None
            except Exception:
                body = "[body read error]"
        
        log_data = {
            "event": "response",
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
            "process_time": round(process_time * 1000, 2),  # milliseconds
            "timestamp": time.time()
        }
        
        # Choose log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
            log_msg = f"Response {request_id} - {response.status_code} ERROR"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            log_msg = f"Response {request_id} - {response.status_code} CLIENT_ERROR"
        else:
            log_level = logging.INFO
            log_msg = f"Response {request_id} - {response.status_code} OK"
        
        logger.log(
            log_level,
            log_msg,
            extra={"log_data": log_data}
        )
    
    async def _log_error(self, request: Request, exception: Exception, 
                        request_id: str, process_time: float):
        """Log unhandled errors"""
        
        log_data = {
            "event": "error",
            "request_id": request_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "process_time": round(process_time * 1000, 2),
            "timestamp": time.time()
        }
        
        logger.error(
            f"Error {request_id} - {type(exception).__name__}: {str(exception)}",
            extra={"log_data": log_data},
            exc_info=True
        )
    
    def _filter_sensitive_headers(self, headers: dict) -> dict:
        """Remove sensitive headers from log data"""
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        return filtered
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check forwarded headers
        forwarded_headers = [
            "cf-connecting-ip",
            "x-forwarded-for",
            "x-real-ip",
            "x-client-ip"
        ]
        
        for header in forwarded_headers:
            ip = request.headers.get(header)
            if ip:
                return ip.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        # Base log structure
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add structured data if available
        if hasattr(record, 'log_data'):
            log_entry.update(record.log_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)