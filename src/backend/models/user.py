"""
User and authentication related models
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
import enum

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin


class UserRole(enum.Enum):
    """User roles enumeration"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    CLIENT = "client"


class UserStatus(enum.Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"


class User(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """User model for authentication and basic profile"""
    
    __tablename__ = "users"
    
    # Authentication fields
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile fields
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    phone = Column(String(20), nullable=True)
    
    # Status and role
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    
    # Authentication metadata
    email_verified = Column(Boolean, default=False, nullable=False)
    email_verified_at = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # Security
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    must_change_password = Column(Boolean, default=False, nullable=False)
    
    # Two-factor authentication
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_secret = Column(String(32), nullable=True)
    backup_codes = Column(Text, nullable=True)  # JSON array of backup codes
    
    # OAuth providers
    google_id = Column(String(100), nullable=True)
    microsoft_id = Column(String(100), nullable=True)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("ProjectMember", back_populates="user")
    comments = relationship("Comment", back_populates="user")
    activity_logs = relationship("ActivityLog", back_populates="user")
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        return self.locked_until and self.locked_until > datetime.utcnow()
    
    def increment_failed_login(self):
        """Increment failed login attempts"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)
    
    def reset_failed_login(self):
        """Reset failed login attempts"""
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def record_login(self):
        """Record successful login"""
        self.last_login = datetime.utcnow()
        self.login_count += 1
        self.reset_failed_login()


class UserProfile(Base, TimestampMixin, AuditMixin):
    """Extended user profile information"""
    
    __tablename__ = "user_profiles"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    # Professional information
    company = Column(String(100), nullable=True)
    position = Column(String(100), nullable=True)
    industry = Column(String(50), nullable=True)
    experience_years = Column(Integer, nullable=True)
    
    # Contact information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(50), nullable=True)
    
    # Profile details
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    timezone = Column(String(50), default="Australia/Sydney", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    
    # Preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    sms_notifications = Column(Boolean, default=False, nullable=False)
    newsletter_subscribed = Column(Boolean, default=False, nullable=False)
    
    # Professional credentials
    certifications = Column(Text, nullable=True)  # JSON array
    linkedin_url = Column(String(255), nullable=True)
    website_url = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="profile")


class APIKey(Base, TimestampMixin, SoftDeleteMixin):
    """API keys for external integrations"""
    
    __tablename__ = "api_keys"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Key details
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)
    key_prefix = Column(String(10), nullable=False)  # First few chars for identification
    
    # Permissions and limits
    scopes = Column(Text, nullable=False)  # JSON array of permitted scopes
    rate_limit = Column(Integer, default=1000, nullable=False)  # Requests per hour
    
    # Usage tracking
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Expiration
    expires_at = Column(DateTime, nullable=True)
    
    # Security
    allowed_ips = Column(Text, nullable=True)  # JSON array of allowed IP addresses
    webhook_url = Column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    @property
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        return self.expires_at and self.expires_at < datetime.utcnow()
    
    @property
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        return self.is_active and not self.is_expired and not self.is_deleted
    
    def record_usage(self):
        """Record API key usage"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()


class UserSession(Base, TimestampMixin):
    """User session tracking"""
    
    __tablename__ = "user_sessions"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    
    # Session details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    device_type = Column(String(50), nullable=True)
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)
    
    # Location data
    country = Column(String(50), nullable=True)
    city = Column(String(100), nullable=True)
    
    # Session status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return self.expires_at < datetime.utcnow()
    
    def extend_session(self, minutes: int = 30):
        """Extend session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(minutes=minutes)
        self.last_activity = datetime.utcnow()