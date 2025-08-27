"""
Authentication schemas for request/response validation
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator

from ..models.base import BaseSchema, TimestampSchema
from ..models.user import UserRole, UserStatus


class UserLogin(BaseModel):
    """User login request schema"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False


class UserRegister(BaseModel):
    """User registration request schema"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    phone: Optional[str] = Field(None, max_length=20)
    company: Optional[str] = Field(None, max_length=100)
    position: Optional[str] = Field(None, max_length=100)
    
    @validator('email')
    def email_must_be_valid(cls, v):
        # Additional email validation if needed
        return v.lower()
    
    @validator('phone')
    def phone_must_be_valid(cls, v):
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit():
            raise ValueError('Phone number contains invalid characters')
        return v


class UserResponse(BaseSchema):
    """User response schema"""
    id: int
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    role: UserRole
    status: UserStatus
    email_verified: bool
    last_login: Optional[datetime]
    created_at: datetime
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Optional[UserResponse] = None
    requires_2fa: bool = False


class RefreshToken(BaseModel):
    """Refresh token request schema"""
    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset request schema"""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)


class PasswordChange(BaseModel):
    """Password change request schema"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def new_password_must_be_different(cls, v, values):
        if 'current_password' in values and v == values['current_password']:
            raise ValueError('New password must be different from current password')
        return v


class OAuth2Token(BaseModel):
    """OAuth2 token exchange schema"""
    code: str
    state: Optional[str] = None


class TwoFactorAuth(BaseModel):
    """Two-factor authentication schema"""
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    
    @validator('code')
    def code_must_be_numeric(cls, v):
        if not v.isdigit():
            raise ValueError('2FA code must be numeric')
        return v


class TwoFactorSetup(BaseModel):
    """Two-factor authentication setup schema"""
    secret_key: str
    backup_codes: List[str]
    qr_code_url: str


class APIKeyCreate(BaseModel):
    """API key creation request schema"""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = []
    expires_days: Optional[int] = Field(None, gt=0, le=365)
    rate_limit: int = Field(1000, gt=0, le=10000)


class APIKeyResponse(BaseSchema):
    """API key response schema"""
    id: int
    name: str
    key_prefix: str
    scopes: List[str]
    rate_limit: int
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    is_active: bool


class APIKeyUpdate(BaseModel):
    """API key update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    scopes: Optional[List[str]] = None
    rate_limit: Optional[int] = Field(None, gt=0, le=10000)
    is_active: Optional[bool] = None