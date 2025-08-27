"""
User and Authentication Models
User management, authentication, and authorization models
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, Enum as SQLEnum, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base_model import BaseModel, TimestampMixin


class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    ANALYST = "analyst"
    INVESTOR = "investor"
    VIEWER = "viewer"
    API_USER = "api_user"


class UserStatus(str, Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class User(BaseModel):
    """User model"""
    
    __tablename__ = "users"
    
    # Basic Information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    status = Column(SQLEnum(UserStatus), default=UserStatus.PENDING_VERIFICATION, nullable=False)
    
    # Profile
    company = Column(String(200), nullable=True)
    position = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # Preferences
    timezone = Column(String(50), default="UTC", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    preferences = Column(JSON, default={}, nullable=False)
    
    # Security
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Verification
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    verification_token = Column(String(255), nullable=True)
    reset_password_token = Column(String(255), nullable=True)
    reset_password_expires = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return any(user_role.role == role for user_role in self.user_roles)
    
    def get_roles(self) -> List[UserRole]:
        """Get all user roles"""
        return [user_role.role for user_role in self.user_roles]
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def to_dict(self, include_sensitive=False):
        """Convert to dictionary, optionally excluding sensitive data"""
        exclude_fields = {'hashed_password', 'verification_token', 'reset_password_token'}
        if not include_sensitive:
            exclude_fields.update({'failed_login_attempts', 'locked_until'})
        
        result = super().to_dict(exclude_fields=exclude_fields)
        result['roles'] = self.get_roles()
        result['full_name'] = self.full_name
        return result


class UserRoleAssignment(BaseModel):
    """User role assignment model"""
    
    __tablename__ = "user_roles"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    granted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="user_roles")
    granter = relationship("User", foreign_keys=[granted_by])
    
    def is_expired(self) -> bool:
        """Check if role assignment is expired"""
        if self.expires_at and self.expires_at < datetime.utcnow():
            return True
        return False


class APIKey(BaseModel):
    """API key model for programmatic access"""
    
    __tablename__ = "api_keys"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    prefix = Column(String(10), nullable=False)
    
    # Permissions and limits
    permissions = Column(JSON, default=[], nullable=False)
    rate_limit_per_minute = Column(Integer, default=100, nullable=False)
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    rate_limit_per_day = Column(Integer, default=10000, nullable=False)
    
    # Status and expiry
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if self.expires_at and self.expires_at < datetime.utcnow():
            return True
        return False
    
    def increment_usage(self):
        """Increment usage counter"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()


class UserSession(BaseModel):
    """User session model for tracking login sessions"""
    
    __tablename__ = "user_sessions"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), nullable=False, unique=True)
    refresh_token = Column(String(255), nullable=True, unique=True)
    
    # Session details
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    device_fingerprint = Column(String(255), nullable=True)
    
    # Timestamps
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions", foreign_keys=[user_id])
    revoker = relationship("User", foreign_keys=[revoked_by])
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return self.expires_at < datetime.utcnow()
    
    def revoke(self, revoked_by_user_id: Optional[UUID] = None):
        """Revoke the session"""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by_user_id
    
    def refresh_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()