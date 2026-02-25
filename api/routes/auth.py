"""
Nexus Trading System - Authentication Routes
User authentication, registration, and token management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import logging

from database.session import get_database_session
from database.models import User, UserRole, UserStatus
from database.schemas import Token, UserCreate, UserResponse, Login
from api.auth import (
    UserManager, TokenManager, PasswordManager, SecurityUtils,
    get_current_user, get_current_active_user
)
from api.dependencies import RateLimitChecker, AuditLogger

router = APIRouter(tags=["Authentication"])
logger = logging.getLogger(__name__)

# Rate limiter
rate_limiter = RateLimitChecker()


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_database_session)
):
    """Register a new user"""
    try:
        # Validate input
        if not SecurityUtils.validate_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate password strength
        is_valid, message = SecurityUtils.validate_password_strength(user_data.password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(None, "register"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts"
            )
        
        # Create user
        user = UserManager.create_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role
        )
        
        # Log registration
        AuditLogger.log_security_event(
            event_type="user_registered",
            user=user,
            details={"role": user.role.value, "plan": user_data.subscription_plan}
        )
        
        return UserResponse.from_orm(user)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_database_session)
):
    """Login user and return access token"""
    try:
        # Check rate limit
        if not rate_limiter.check_rate_limit(form_data.username, "login"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts"
            )
        
        # Authenticate user
        user = UserManager.authenticate_user(db, form_data.username, form_data.password)
        if not user:
            # Log failed login attempt
            AuditLogger.log_security_event(
                event_type="login_failed",
                details={"username": form_data.username},
                severity="warning"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        UserManager.update_last_login(db, user)
        
        # Create access token
        access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
        access_token = TokenManager.create_access_token(
            data={"sub": user.id, "username": user.username, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        # Log successful login
        AuditLogger.log_security_event(
            event_type="login_success",
            user=user,
            severity="info"
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token"""
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
        access_token = TokenManager.create_access_token(
            data={"sub": current_user.id, "username": current_user.username, "role": current_user.role.value},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds()
        }
    
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """Logout user (client-side token removal)"""
    try:
        # Log logout
        AuditLogger.log_security_event(
            event_type="logout",
            user=current_user,
            severity="info"
        )
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    try:
        # Verify current password
        if not PasswordManager.verify_password(current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        is_valid, message = SecurityUtils.validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Update password
        current_user.hashed_password = PasswordManager.hash_password(new_password)
        
        with get_database_session() as db:
            db.merge(current_user)
            db.commit()
        
        # Log password change
        AuditLogger.log_security_event(
            event_type="password_changed",
            user=current_user,
            severity="info"
        )
        
        return {"message": "Password changed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/reset-password-request")
async def request_password_reset(
    email: str,
    db: Session = Depends(get_database_session)
):
    """Request password reset (sends email with reset token)"""
    try:
        # Find user by email
        user = UserManager.get_user_by_email(db, email)
        if not user:
            # Don't reveal if email exists or not
            return {"message": "If the email exists, a reset link will be sent"}
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(email, "password_reset"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset requests"
            )
        
        # Generate reset token
        reset_token = TokenManager.generate_api_key()
        reset_token_expires = timedelta(hours=1)
        
        # TODO: Store reset token in database and send email
        
        # Log password reset request
        AuditLogger.log_security_event(
            event_type="password_reset_requested",
            user=user,
            details={"email": email},
            severity="warning"
        )
        
        return {"message": "If the email exists, a reset link will be sent"}
    
    except Exception as e:
        logger.error(f"Password reset request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )


@router.post("/reset-password")
async def reset_password(
    token: str,
    new_password: str,
    db: Session = Depends(get_database_session)
):
    """Reset password with token"""
    try:
        # TODO: Validate reset token
        # For now, this is a placeholder
        
        # Validate new password
        is_valid, message = SecurityUtils.validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {"message": "Password reset successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.get("/verify-token")
async def verify_token(
    current_user: User = Depends(get_current_user)
):
    """Verify if token is valid"""
    try:
        return {
            "valid": True,
            "user_id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value,
            "expires_at": "30 days"  # TODO: Calculate actual expiry
        }
    
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token verification failed"
        )


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    try:
        return UserResponse.from_orm(current_user)
    
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )
