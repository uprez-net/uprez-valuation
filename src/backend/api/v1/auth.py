"""
Authentication endpoints for user login, registration, and token management
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from ...core.database import get_async_db
from ...core.security import (
    create_token_pair,
    decode_token,
    verify_password,
    create_password_hash,
    validate_password_strength,
    validate_token_type
)
from ...core.config import settings
from ...models.user import User, UserProfile, UserStatus, UserRole
from ...schemas.auth import (
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    PasswordReset,
    PasswordChange,
    RefreshToken,
    OAuth2Token,
    TwoFactorAuth
)
from ...services.auth import AuthService
from ...services.email import EmailService

router = APIRouter()

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    scopes={
        "read": "Read access",
        "write": "Write access", 
        "admin": "Administrative access"
    }
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_async_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = decode_token(token)
        if token_data.sub is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    # Get user from database
    user = await AuthService.get_user_by_id(db, int(token_data.sub))
    if user is None:
        raise credentials_exception
    
    # Check user status
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not active"
        )
    
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is locked"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user with additional checks"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user with admin privileges"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """Register new user account"""
    
    # Check if user already exists
    existing_user = await AuthService.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Validate password strength
    is_strong, message = validate_password_strength(user_data.password)
    if not is_strong:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {message}"
        )
    
    # Create user
    user = await AuthService.create_user(
        db=db,
        email=user_data.email,
        password=user_data.password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        phone=user_data.phone
    )
    
    # Send welcome email
    try:
        await EmailService.send_welcome_email(user.email, user.first_name)
    except Exception as e:
        # Log error but don't fail registration
        pass
    
    return UserResponse.from_orm(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """Authenticate user and return access tokens"""
    
    # Get user by email
    user = await AuthService.get_user_by_email(db, form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check account status
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active"
        )
    
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is locked due to multiple failed login attempts"
        )
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        user.increment_failed_login()
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check 2FA if enabled
    if user.two_factor_enabled:
        # Return special response indicating 2FA required
        return TokenResponse(
            access_token="2fa_required",
            token_type="2fa",
            expires_in=300,  # 5 minutes to complete 2FA
            requires_2fa=True
        )
    
    # Create token pair
    scopes = ["read", "write"]
    if user.role == UserRole.ADMIN:
        scopes.append("admin")
    
    tokens = create_token_pair(
        subject=str(user.id),
        scopes=scopes,
        additional_claims={"role": user.role.value}
    )
    
    # Record successful login
    user.record_login()
    await db.commit()
    
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=UserResponse.from_orm(user)
    )


@router.post("/login/2fa", response_model=TokenResponse)
async def login_two_factor(
    two_factor_data: TwoFactorAuth,
    db: AsyncSession = Depends(get_async_db)
):
    """Complete two-factor authentication"""
    
    # Get user by email
    user = await AuthService.get_user_by_email(db, two_factor_data.email)
    if not user or not user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication not enabled"
        )
    
    # Verify 2FA code
    is_valid = await AuthService.verify_2fa_code(user, two_factor_data.code)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid two-factor authentication code"
        )
    
    # Create token pair
    scopes = ["read", "write"]
    if user.role == UserRole.ADMIN:
        scopes.append("admin")
    
    tokens = create_token_pair(
        subject=str(user.id),
        scopes=scopes,
        additional_claims={"role": user.role.value}
    )
    
    # Record successful login
    user.record_login()
    await db.commit()
    
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=UserResponse.from_orm(user)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshToken,
    db: AsyncSession = Depends(get_async_db)
):
    """Refresh access token using refresh token"""
    
    # Validate refresh token
    if not validate_token_type(refresh_data.refresh_token, "refresh"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    try:
        token_data = decode_token(refresh_data.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    user = await AuthService.get_user_by_id(db, int(token_data.sub))
    if not user or user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new token pair
    scopes = ["read", "write"]
    if user.role == UserRole.ADMIN:
        scopes.append("admin")
    
    tokens = create_token_pair(
        subject=str(user.id),
        scopes=scopes,
        additional_claims={"role": user.role.value}
    )
    
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=UserResponse.from_orm(user)
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Logout user and invalidate tokens"""
    
    # In a production system, you would typically:
    # 1. Add token to blacklist
    # 2. Remove from active sessions
    # 3. Clear refresh tokens
    
    # For now, just return success
    return {"message": "Successfully logged out"}


@router.post("/password/forgot")
async def forgot_password(
    email: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """Send password reset email"""
    
    user = await AuthService.get_user_by_email(db, email)
    if user:
        # Generate reset token
        reset_token = await AuthService.create_password_reset_token(user)
        
        # Send reset email
        try:
            await EmailService.send_password_reset_email(
                user.email,
                user.first_name,
                reset_token
            )
        except Exception as e:
            # Log error but don't reveal if user exists
            pass
    
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/password/reset")
async def reset_password(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_async_db)
):
    """Reset password using reset token"""
    
    # Verify reset token
    user = await AuthService.verify_password_reset_token(db, reset_data.token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Validate new password
    is_strong, message = validate_password_strength(reset_data.new_password)
    if not is_strong:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {message}"
        )
    
    # Update password
    user.hashed_password = create_password_hash(reset_data.new_password)
    user.password_changed_at = datetime.utcnow()
    user.must_change_password = False
    
    await db.commit()
    
    return {"message": "Password has been reset successfully"}


@router.post("/password/change")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Change user password"""
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    is_strong, message = validate_password_strength(password_data.new_password)
    if not is_strong:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {message}"
        )
    
    # Check if new password is different
    if verify_password(password_data.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Update password
    current_user.hashed_password = create_password_hash(password_data.new_password)
    current_user.password_changed_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Password changed successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return UserResponse.from_orm(current_user)


@router.post("/verify-email")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_async_db)
):
    """Verify user email address"""
    
    user = await AuthService.verify_email_token(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Update user email verification status
    user.email_verified = True
    user.email_verified_at = datetime.utcnow()
    user.status = UserStatus.ACTIVE
    
    await db.commit()
    
    return {"message": "Email verified successfully"}


@router.post("/resend-verification")
async def resend_verification_email(
    email: str,
    db: AsyncSession = Depends(get_async_db)
):
    """Resend email verification"""
    
    user = await AuthService.get_user_by_email(db, email)
    if user and not user.email_verified:
        try:
            verification_token = await AuthService.create_email_verification_token(user)
            await EmailService.send_verification_email(
                user.email,
                user.first_name,
                verification_token
            )
        except Exception:
            pass
    
    return {"message": "If the email exists and is unverified, a verification email has been sent"}


# OAuth2 endpoints for external providers
@router.get("/oauth/{provider}")
async def oauth_login(provider: str):
    """Initiate OAuth login with external provider"""
    
    if provider == "google":
        # Redirect to Google OAuth
        auth_url = f"https://accounts.google.com/oauth/authorize?client_id={settings.GOOGLE_CLIENT_ID}&redirect_uri=callback&scope=email profile&response_type=code"
        return {"auth_url": auth_url}
    
    elif provider == "microsoft":
        # Redirect to Microsoft OAuth
        auth_url = f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id={settings.MICROSOFT_CLIENT_ID}&redirect_uri=callback&scope=email profile&response_type=code"
        return {"auth_url": auth_url}
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported OAuth provider"
        )


@router.post("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    oauth_token: OAuth2Token,
    db: AsyncSession = Depends(get_async_db)
):
    """Handle OAuth callback and create user session"""
    
    try:
        # Verify OAuth token and get user info
        user_info = await AuthService.verify_oauth_token(provider, oauth_token.code)
        
        # Find or create user
        user = await AuthService.get_user_by_email(db, user_info["email"])
        if not user:
            # Create new user from OAuth profile
            user = await AuthService.create_oauth_user(
                db=db,
                email=user_info["email"],
                first_name=user_info.get("given_name", ""),
                last_name=user_info.get("family_name", ""),
                provider=provider,
                provider_id=user_info["id"]
            )
        else:
            # Link OAuth account if not already linked
            if provider == "google" and not user.google_id:
                user.google_id = user_info["id"]
            elif provider == "microsoft" and not user.microsoft_id:
                user.microsoft_id = user_info["id"]
            
            await db.commit()
        
        # Create token pair
        scopes = ["read", "write"]
        if user.role == UserRole.ADMIN:
            scopes.append("admin")
        
        tokens = create_token_pair(
            subject=str(user.id),
            scopes=scopes,
            additional_claims={"role": user.role.value}
        )
        
        # Record login
        user.record_login()
        await db.commit()
        
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            user=UserResponse.from_orm(user)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth authentication failed: {str(e)}"
        )