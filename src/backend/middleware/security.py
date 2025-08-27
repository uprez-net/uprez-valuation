"""
Security middleware for headers and general security
"""

import time
import secrets
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.config import settings


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for adding security headers and basic protection"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.server_header = f"{settings.PROJECT_NAME}/{settings.VERSION}"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not present
        if "X-Request-ID" not in request.headers:
            request_id = f"req_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
            # Add to request state for logging
            request.state.request_id = request_id
        else:
            request.state.request_id = request.headers["X-Request-ID"]
        
        # Basic security checks
        if await self._should_block_request(request):
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": 403,
                        "message": "Request blocked by security policy",
                        "type": "security_violation"
                    }
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, request)
        
        return response
    
    async def _should_block_request(self, request: Request) -> bool:
        """Check if request should be blocked for security reasons"""
        
        # Block requests with suspicious user agents
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan", "nessus"]
        if any(agent in user_agent for agent in suspicious_agents):
            return True
        
        # Block requests with suspicious paths
        suspicious_paths = [
            "/.env", "/wp-admin", "/phpmyadmin", "/admin/config.php",
            "/xmlrpc.php", "/wp-config.php", "/.git", "/config.json"
        ]
        if any(path in str(request.url.path).lower() for path in suspicious_paths):
            return True
        
        # Block requests with suspicious headers
        if "X-Real-IP" in request.headers or "X-Originating-IP" in request.headers:
            # Log potential header injection attempt
            pass
        
        # Check for SQL injection patterns in query parameters
        query_string = str(request.url.query).lower()
        sql_patterns = ["union select", "drop table", "insert into", "delete from", "update set"]
        if any(pattern in query_string for pattern in sql_patterns):
            return True
        
        return False
    
    def _add_security_headers(self, response: Response, request: Request):
        """Add security headers to response"""
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
        response.headers["Server"] = self.server_header
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Request-ID"] = request.state.request_id
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP header for HTML responses
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://unpkg.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
        
        # API specific headers
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # CORS headers are handled by CORSMiddleware
        
        # Add API version
        response.headers["X-API-Version"] = settings.VERSION