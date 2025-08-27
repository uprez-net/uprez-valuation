"""
Rate Limiting Middleware
API rate limiting using Redis for distributed rate limiting
"""
import json
import time
from typing import Optional, Dict, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import redis
import structlog

from ...config import settings

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm"""
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client or self._create_redis_client()
        self.default_per_minute = settings.security.rate_limit_per_minute
        self.default_burst = settings.security.rate_limit_burst
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client for rate limiting"""
        try:
            return redis.from_url(settings.redis.url, decode_responses=True)
        except Exception as e:
            logger.warning("Redis not available for rate limiting", error=str(e))
            return None
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/health/detailed", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limits for this client
        per_minute, burst = self._get_rate_limits(request)
        
        # Check rate limit
        allowed, retry_after = await self._is_allowed(client_id, per_minute, burst)
        
        if not allowed:
            logger.warning("Rate limit exceeded", client_id=client_id, path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate Limit Exceeded",
                    "detail": f"Too many requests. Try again in {retry_after} seconds.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id, per_minute, burst)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""
        # Try to get user ID if authenticated
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:16]}..."
        
        # Fallback to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host
        
        return f"ip:{client_ip}"
    
    def _get_rate_limits(self, request: Request) -> Tuple[int, int]:
        """Get rate limits for the current request"""
        # Check if user has custom API key limits
        if hasattr(request.state, 'user') and hasattr(request.state, 'auth_method'):
            if request.state.auth_method == "api_key":
                # Get API key limits from database
                # This would require a database query in a real implementation
                pass
        
        # Default limits
        return self.default_per_minute, self.default_burst
    
    async def _is_allowed(self, client_id: str, per_minute: int, burst: int) -> Tuple[bool, int]:
        """Check if request is allowed using sliding window algorithm"""
        if not self.redis_client:
            # If Redis is not available, allow all requests
            return True, 0
        
        try:
            current_time = time.time()
            window_start = current_time - 60  # 1 minute window
            
            # Redis key for this client
            key = f"rate_limit:{client_id}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries (outside the window)
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry for cleanup
            pipe.expire(key, 120)  # 2 minutes
            
            results = pipe.execute()
            current_requests = results[1]
            
            # Check limits
            if current_requests < per_minute:
                return True, 0
            else:
                # Calculate retry after time
                oldest_request = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_request:
                    retry_after = int(60 - (current_time - oldest_request[0][1]))
                    return False, max(retry_after, 1)
                else:
                    return False, 60
        
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Allow request if rate limiting fails
            return True, 0
    
    def _add_rate_limit_headers(self, response, client_id: str, per_minute: int, burst: int):
        """Add rate limiting headers to response"""
        if not self.redis_client:
            return
        
        try:
            key = f"rate_limit:{client_id}"
            current_requests = self.redis_client.zcard(key)
            remaining = max(0, per_minute - current_requests)
            
            response.headers["X-RateLimit-Limit"] = str(per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            
        except Exception as e:
            logger.error("Error adding rate limit headers", error=str(e))


class APIKeyRateLimiter:
    """Enhanced rate limiter for API keys with custom limits"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def check_api_key_limits(
        self, 
        api_key_id: str, 
        limits: Dict[str, int]
    ) -> Tuple[bool, Optional[str], int]:
        """
        Check API key specific rate limits
        
        Args:
            api_key_id: API key identifier
            limits: Dictionary with 'per_minute', 'per_hour', 'per_day' limits
        
        Returns:
            (allowed, period_exceeded, retry_after)
        """
        current_time = time.time()
        
        periods = {
            'per_minute': 60,
            'per_hour': 3600,
            'per_day': 86400
        }
        
        for period_name, period_seconds in periods.items():
            if period_name not in limits:
                continue
            
            limit = limits[period_name]
            window_start = current_time - period_seconds
            key = f"api_key_rate_limit:{api_key_id}:{period_name}"
            
            # Clean old entries and count current
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()
            
            current_count = results[1]
            
            if current_count >= limit:
                # Calculate retry after
                oldest_request = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_request:
                    retry_after = int(period_seconds - (current_time - oldest_request[0][1]))
                else:
                    retry_after = period_seconds
                
                return False, period_name, retry_after
        
        # If all limits pass, record the request
        for period_name in periods.keys():
            if period_name in limits:
                key = f"api_key_rate_limit:{api_key_id}:{period_name}"
                pipe = self.redis_client.pipeline()
                pipe.zadd(key, {str(current_time): current_time})
                pipe.expire(key, periods[period_name] * 2)
                pipe.execute()
        
        return True, None, 0