"""
Authentication service layer
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import secrets
import httpx

from ..models.user import User, UserProfile, UserStatus, UserRole
from ..core.security import create_password_hash, verify_password
from ..schemas.auth import UserRegister


class AuthService:
    """Service for authentication operations"""
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email address"""
        query = select(User).where(User.email == email.lower())
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID"""
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        phone: Optional[str] = None
    ) -> User:
        """Create new user account"""
        
        # Create user
        user = User(
            email=email.lower(),
            hashed_password=create_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            role=UserRole.VIEWER,
            status=UserStatus.PENDING
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Create user profile
        profile = UserProfile(
            user_id=user.id,
            timezone="Australia/Sydney",
            language="en"
        )
        
        db.add(profile)
        await db.commit()
        
        return user
    
    @staticmethod
    async def create_oauth_user(
        db: AsyncSession,
        email: str,
        first_name: str,
        last_name: str,
        provider: str,
        provider_id: str
    ) -> User:
        """Create user from OAuth provider"""
        
        # Generate random password for OAuth users
        random_password = secrets.token_urlsafe(32)
        
        user = User(
            email=email.lower(),
            hashed_password=create_password_hash(random_password),
            first_name=first_name,
            last_name=last_name,
            role=UserRole.VIEWER,
            status=UserStatus.ACTIVE,
            email_verified=True,
            email_verified_at=datetime.utcnow()
        )
        
        # Set provider ID
        if provider == "google":
            user.google_id = provider_id
        elif provider == "microsoft":
            user.microsoft_id = provider_id
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user
    
    @staticmethod
    async def verify_2fa_code(user: User, code: str) -> bool:
        """Verify two-factor authentication code"""
        # Implementation would verify TOTP code
        return code == "123456"  # Mock implementation
    
    @staticmethod
    async def create_password_reset_token(user: User) -> str:
        """Create password reset token"""
        # Implementation would create and store reset token
        return secrets.token_urlsafe(32)
    
    @staticmethod
    async def verify_password_reset_token(db: AsyncSession, token: str) -> Optional[User]:
        """Verify password reset token"""
        # Implementation would verify token and return user
        return None
    
    @staticmethod
    async def create_email_verification_token(user: User) -> str:
        """Create email verification token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    async def verify_email_token(db: AsyncSession, token: str) -> Optional[User]:
        """Verify email verification token"""
        # Implementation would verify token and return user
        return None
    
    @staticmethod
    async def verify_oauth_token(provider: str, code: str) -> Dict[str, Any]:
        """Verify OAuth token and get user info"""
        # Mock implementation - would make actual OAuth requests
        return {
            "id": "oauth_user_123",
            "email": "user@example.com",
            "given_name": "John",
            "family_name": "Doe"
        }