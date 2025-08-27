"""
Security utilities for JWT tokens, password hashing, and authentication
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, status
from pydantic import BaseModel

from .config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """Token payload data structure"""
    sub: Optional[str] = None
    scopes: list = []
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    jti: Optional[str] = None


class JWTTokens(BaseModel):
    """JWT token pair response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


def create_password_hash(password: str) -> str:
    """Create password hash using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    scopes: list = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """Create JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "exp": expire,
        "iat": datetime.utcnow(),
        "sub": str(subject),
        "scopes": scopes or [],
        "type": "access"
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """Create JWT refresh token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "exp": expire,
        "iat": datetime.utcnow(),
        "sub": str(subject),
        "type": "refresh"
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_token_pair(
    subject: Union[str, Any],
    scopes: list = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> JWTTokens:
    """Create access and refresh token pair"""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = create_access_token(
        subject=subject,
        expires_delta=access_token_expires,
        scopes=scopes,
        additional_claims=additional_claims
    )
    
    refresh_token = create_refresh_token(
        subject=subject,
        expires_delta=refresh_token_expires,
        additional_claims=additional_claims
    )
    
    return JWTTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds())
    )


def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        return TokenData(
            sub=payload.get("sub"),
            scopes=payload.get("scopes", []),
            exp=datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None,
            iat=datetime.fromtimestamp(payload.get("iat")) if payload.get("iat") else None,
            jti=payload.get("jti")
        )
    
    except PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_token_type(token: str, expected_type: str) -> bool:
    """Validate token type (access/refresh)"""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        return payload.get("type") == expected_type
    except PyJWTError:
        return False


def is_token_expired(token: str) -> bool:
    """Check if token is expired"""
    try:
        token_data = decode_token(token)
        if token_data.exp:
            return datetime.utcnow() > token_data.exp
        return True
    except HTTPException:
        return True


def generate_api_key(prefix: str = "uprez") -> str:
    """Generate API key for external integrations"""
    import secrets
    import string
    
    # Generate random string
    alphabet = string.ascii_letters + string.digits
    key_suffix = ''.join(secrets.choice(alphabet) for _ in range(32))
    
    return f"{prefix}_{key_suffix}"


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage"""
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify API key against hash"""
    return pwd_context.verify(plain_key, hashed_key)