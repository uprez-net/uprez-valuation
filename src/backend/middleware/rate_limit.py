"""
Rate limiting middleware with Redis backend
"""

import time
import hashlib
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as aioredis

from ..core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis backend"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.redis: Optional[aioredis.Redis] = None
        self.default_limit = settings.RATE_LIMIT_PER_MINUTE
        self.burst_limit = settings.RATE_LIMIT_BURST
        self.window_size = 60  # 1 minute window
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Initialize Redis connection if not available
        if not self.redis:
            try:
                self.redis = aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception:
                # If Redis is not available, skip rate limiting
                return await call_next(request)
        
        # Skip rate limiting for certain endpoints
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        # Get rate limit key and limits
        rate_limit_key = self._get_rate_limit_key(request)
        limits = self._get_rate_limits(request)
        
        # Check rate limit
        is_allowed, headers = await self._check_rate_limit(
            rate_limit_key,
            limits["limit"],
            limits["window"]
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded",
                        "retry_after": headers.get("X-RateLimit-Reset", 60)
                    }
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header, value in headers.items():
            response.headers[header] = str(value)
        
        return response
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request"""
        
        # Skip for health checks
        if request.url.path in ["/health", "/health/detailed", "/status"]:
            return True
        
        # Skip for static files
        if request.url.path.startswith("/static/"):
            return True
        
        # Skip for documentation
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return True
        
        return False
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key for request"""
        
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"rate_limit:api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Use user ID if authenticated
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            return f"rate_limit:user:{token_hash}"
        
        # Fall back to IP address
        client_ip = self._get_client_ip(request)
        return f"rate_limit:ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        
        # Check forwarded headers (in order of preference)
        forwarded_headers = [
            "CF-Connecting-IP",  # Cloudflare
            "X-Forwarded-For",   # Standard
            "X-Real-IP",         # Nginx
            "X-Client-IP"        # Some proxies
        ]
        
        for header in forwarded_headers:
            ip = request.headers.get(header)
            if ip:
                # Take first IP if comma-separated list
                return ip.split(",")[0].strip()
        
        # Fall back to remote address
        return request.client.host if request.client else "unknown"
    
    def _get_rate_limits(self, request: Request) -> dict:
        """Get rate limits for request based on endpoint and authentication"""
        
        path = request.url.path
        
        # Different limits for different endpoints
        if path.startswith("/api/v1/auth/"):
            # Stricter limits for auth endpoints
            return {"limit": 10, "window": 60}  # 10 per minute
        
        elif path.startswith("/api/v1/documents/upload"):
            # More lenient for file uploads
            return {"limit": 5, "window": 60}   # 5 per minute
        
        elif path.startswith("/api/v1/admin/"):
            # Admin endpoints
            return {"limit": 100, "window": 60}  # 100 per minute
        
        elif request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            # Write operations
            return {"limit": 30, "window": 60}   # 30 per minute
        
        else:
            # Default limits for read operations
            return {"limit": self.default_limit, "window": self.window_size}
    
    async def _check_rate_limit(self, key: str, limit: int, window: int) -> tuple[bool, dict]:
        """Check rate limit using sliding window algorithm"""
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window * 2)
            
            results = await pipe.execute()
            current_count = results[1]  # Count after removing expired entries
            
            # Calculate remaining and reset time
            remaining = max(0, limit - current_count - 1)  # -1 for current request
            reset_time = current_time + window
            
            headers = {
                "X-RateLimit-Limit": limit,
                "X-RateLimit-Remaining": remaining,
                "X-RateLimit-Reset": reset_time,
                "X-RateLimit-Window": window
            }
            
            # Check if limit exceeded
            if current_count >= limit:
                headers["Retry-After"] = window
                return False, headers
            
            return True, headers
            
        except Exception as e:
            # If Redis fails, allow request but log error
            # In production, you might want to use fallback rate limiting
            headers = {
                "X-RateLimit-Limit": limit,
                "X-RateLimit-Remaining": limit,
                "X-RateLimit-Reset": int(time.time()) + window
            }
            return True, headers
    
    async def cleanup_expired_keys(self):
        """Background task to cleanup expired rate limit keys"""
        try:
            if self.redis:
                # Get all rate limit keys
                keys = await self.redis.keys("rate_limit:*")
                
                if keys:
                    # Remove keys with no members (expired)
                    pipe = self.redis.pipeline()
                    for key in keys:
                        pipe.zcard(key)
                    
                    results = await pipe.execute()
                    
                    # Delete empty keys
                    empty_keys = [key for key, count in zip(keys, results) if count == 0]
                    if empty_keys:
                        await self.redis.delete(*empty_keys)
                        
        except Exception as e:
            # Log cleanup error but don't crash
            pass