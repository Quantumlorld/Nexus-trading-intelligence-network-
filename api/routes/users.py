"""
Nexus Trading System - Users Routes
User management, profile updates, and user administration
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional
import logging

from database.session import get_database_session
from database.models import User, UserRole, UserStatus, UserRiskSettings
from database.schemas import (
    UserResponse, UserUpdate, RiskSettingsCreate, RiskSettingsUpdate, 
    RiskSettingsResponse, PaginatedResponse, PaginationParams
)
from api.auth import get_current_user, get_current_active_user, get_current_admin_user
from api.dependencies import UserPermissions, RateLimitChecker, AuditLogger

router = APIRouter(tags=["Users"])
logger = logging.getLogger(__name__)

# Rate limiter
rate_limiter = RateLimitChecker()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user profile"""
    try:
        return UserResponse.from_orm(current_user)
    
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update current user profile"""
    try:
        with get_database_session() as db:
            # Update user fields
            if user_update.full_name is not None:
                current_user.full_name = user_update.full_name
            
            if user_update.phone is not None:
                current_user.phone = user_update.phone
            
            if user_update.country is not None:
                current_user.country = user_update.country
            
            if user_update.auto_trade_enabled is not None:
                # Check if user can auto-trade
                if user_update.auto_trade_enabled and not UserPermissions.can_auto_trade(current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="User is not authorized to enable auto-trading"
                    )
                current_user.auto_trade_enabled = user_update.auto_trade_enabled
            
            # Save changes
            db.merge(current_user)
            db.commit()
            db.refresh(current_user)
            
            # Log profile update
            AuditLogger.log_user_action(
                user=current_user,
                action="profile_updated",
                details=user_update.dict(exclude_unset=True)
            )
            
            return UserResponse.from_orm(current_user)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.get("/me/risk-settings", response_model=RiskSettingsResponse)
async def get_current_user_risk_settings(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user risk settings"""
    try:
        with get_database_session() as db:
            # Get or create risk settings
            risk_settings = db.query(UserRiskSettings).filter(
                UserRiskSettings.user_id == current_user.id
            ).first()
            
            if not risk_settings:
                # Create default risk settings
                risk_settings = UserRiskSettings(
                    user_id=current_user.id,
                    default_risk_percent=current_user.max_risk_percent,
                    max_daily_loss=current_user.max_daily_loss,
                    max_daily_trades=current_user.max_daily_trades
                )
                db.add(risk_settings)
                db.commit()
                db.refresh(risk_settings)
            
            return RiskSettingsResponse.from_orm(risk_settings)
    
    except Exception as e:
        logger.error(f"Failed to get risk settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get risk settings"
        )


@router.put("/me/risk-settings", response_model=RiskSettingsResponse)
async def update_current_user_risk_settings(
    settings_update: RiskSettingsUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update current user risk settings"""
    try:
        with get_database_session() as db:
            # Get existing risk settings
            risk_settings = db.query(UserRiskSettings).filter(
                UserRiskSettings.user_id == current_user.id
            ).first()
            
            if not risk_settings:
                # Create new risk settings
                risk_settings = UserRiskSettings(user_id=current_user.id)
                db.add(risk_settings)
            
            # Update fields
            update_data = settings_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(risk_settings, field):
                    setattr(risk_settings, field, value)
            
            # Validate settings
            from api.dependencies import RiskValidator
            is_valid, message = RiskValidator.validate_risk_settings(current_user, update_data)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=message
                )
            
            # Save changes
            db.merge(risk_settings)
            db.commit()
            db.refresh(risk_settings)
            
            # Log risk settings update
            AuditLogger.log_user_action(
                user=current_user,
                action="risk_settings_updated",
                details=update_data
            )
            
            return RiskSettingsResponse.from_orm(risk_settings)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update risk settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update risk settings"
        )


@router.get("/me/statistics")
async def get_current_user_statistics(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user trading statistics"""
    try:
        with get_database_session() as db:
            # TODO: Implement comprehensive user statistics
            stats = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "current_balance": 10000.0,
                "daily_trades": 0,
                "daily_pnl": 0.0,
                "weekly_trades": 0,
                "weekly_pnl": 0.0,
                "monthly_trades": 0,
                "monthly_pnl": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "avg_trade_duration": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "risk_reward_ratio": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
            return stats
    
    except Exception as e:
        logger.error(f"Failed to get user statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


@router.get("/me/performance")
async def get_current_user_performance(
    current_user: User = Depends(get_current_active_user),
    period: str = Query("monthly", regex="^(daily|weekly|monthly|yearly)$")
):
    """Get current user performance metrics"""
    try:
        # TODO: Implement performance calculation
        performance = {
            "period": period,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "total_trades": 0,
            "avg_trade_duration": 0.0,
            "best_trade": None,
            "worst_trade": None
        }
        
        return performance
    
    except Exception as e:
        logger.error(f"Failed to get user performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user performance"
        )


# Admin routes
@router.get("/list", response_model=PaginatedResponse)
async def list_users(
    pagination: PaginationParams = Depends(),
    role: Optional[UserRole] = Query(None),
    status: Optional[UserStatus] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user)
):
    """List all users (admin only)"""
    try:
        with get_database_session() as db:
            query = db.query(User)
            
            # Apply filters
            if role:
                query = query.filter(User.role == role)
            
            if status:
                query = query.filter(User.status == status)
            
            if search:
                query = query.filter(
                    or_(
                        User.username.contains(search),
                        User.email.contains(search),
                        User.full_name.contains(search)
                    )
                )
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            users = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            user_responses = [UserResponse.from_orm(user) for user in users]
            
            return PaginatedResponse.create(
                items=user_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_admin_user)
):
    """Get user by ID (admin only)"""
    try:
        with get_database_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            return UserResponse.from_orm(user)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/{user_id}/status")
async def update_user_status(
    user_id: int,
    new_status: UserStatus,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Update user status (admin only)"""
    try:
        with get_database_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            old_status = user.status
            user.status = new_status
            
            db.merge(user)
            db.commit()
            
            # Log status change
            AuditLogger.log_user_action(
                user=current_user,
                action="user_status_updated",
                details={
                    "target_user_id": user_id,
                    "target_username": user.username,
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "reason": reason
                }
            )
            
            return {"message": f"User status updated to {new_status.value}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.put("/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Update user role (admin only)"""
    try:
        with get_database_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            old_role = user.role
            user.role = new_role
            
            db.merge(user)
            db.commit()
            
            # Log role change
            AuditLogger.log_user_action(
                user=current_user,
                action="user_role_updated",
                details={
                    "target_user_id": user_id,
                    "target_username": user.username,
                    "old_role": old_role.value,
                    "new_role": new_role.value,
                    "reason": reason
                }
            )
            
            return {"message": f"User role updated to {new_role.value}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Delete user (admin only)"""
    try:
        with get_database_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Don't allow deletion of admin users
            if user.role == UserRole.ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot delete admin users"
                )
            
            username = user.username
            db.delete(user)
            db.commit()
            
            # Log user deletion
            AuditLogger.log_user_action(
                user=current_user,
                action="user_deleted",
                details={
                    "target_user_id": user_id,
                    "target_username": username,
                    "reason": reason
                }
            )
            
            return {"message": f"User {username} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
