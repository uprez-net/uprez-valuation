"""
Authentication Middleware
JWT token validation and user authentication
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

from ...config import settings
from ...database.base import get_db
from ...database.models import User, UserSession

logger = structlog.get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer token
security = HTTPBearer(auto_error=False)


class AuthenticationService:
    """Authentication service for JWT tokens and user management"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
        
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.security.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.security.secret_key,
                algorithms=[settings.security.algorithm]
            )
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None or datetime.fromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def authenticate_user(db, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            return None
        
        if not AuthenticationService.verify_password(password, user.hashed_password):
            # Track failed login attempt
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            db.commit()
            return None
        
        # Reset failed attempts on successful login
        if user.failed_login_attempts > 0:
            user.failed_login_attempts = 0
            user.locked_until = None
        
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    @staticmethod
    def get_current_user_from_token(db, token: str) -> Optional[User]:
        """Get current user from JWT token"""
        payload = AuthenticationService.verify_token(token)
        
        if payload is None:
            return None
        
        user_id = payload.get("sub")
        if user_id is None:
            return None
        
        user = db.query(User).filter(User.id == user_id).first()
        
        if user is None:
            return None
        
        # Check if user is active and not locked
        if not user.is_active or user.is_locked():
            return None
        
        return user


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication"""
    
    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/health",
        "/health/detailed",
        "/metrics",
        "/info",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/forgot-password",
        "/api/v1/auth/reset-password",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    async def dispatch(self, request: Request, call_next):
        # Check if endpoint is public
        path = request.url.path
        if path in self.PUBLIC_ENDPOINTS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)
        
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        
        if not authorization or not authorization.startswith("Bearer "):
            # Check for API key in headers
            api_key = request.headers.get("X-API-Key")
            if api_key:
                # Handle API key authentication
                user = await self._authenticate_api_key(api_key)
                if user:
                    request.state.user = user
                    request.state.auth_method = "api_key"
                    return await call_next(request)
            
            logger.warning("No authentication provided", path=path)
            return self._create_auth_error("Authentication required")
        
        # Extract token
        token = authorization.split(" ")[1]
        
        # Verify token and get user
        try:
            with next(get_db()) as db:
                user = AuthenticationService.get_current_user_from_token(db, token)
                
                if user is None:
                    logger.warning("Invalid token", path=path)
                    return self._create_auth_error("Invalid or expired token")
                
                # Set user in request state
                request.state.user = user
                request.state.auth_method = "jwt"
                
                logger.info("User authenticated", user_id=str(user.id), path=path)
                
        except Exception as e:
            logger.error("Authentication error", error=str(e), path=path)
            return self._create_auth_error("Authentication failed")
        
        return await call_next(request)
    
    async def _authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        try:
            with next(get_db()) as db:
                from ...database.models import APIKey
                
                # Hash the API key for comparison
                api_key_hash = AuthenticationService.get_password_hash(api_key)
                
                # Find API key record
                api_key_record = db.query(APIKey).filter(
                    APIKey.key_hash == api_key_hash,
                    APIKey.is_active == True
                ).first()
                
                if not api_key_record or api_key_record.is_expired():
                    return None
                
                # Update usage
                api_key_record.increment_usage()
                db.commit()
                
                # Get associated user
                user = db.query(User).filter(User.id == api_key_record.user_id).first()
                
                if user and user.is_active and not user.is_locked():
                    return user
                
        except Exception as e:
            logger.error("API key authentication error", error=str(e))
        
        return None
    
    def _create_auth_error(self, detail: str) -> Response:
        """Create authentication error response"""
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Authentication Error",
                "detail": detail
            },
            headers={"WWW-Authenticate": "Bearer"}
        )