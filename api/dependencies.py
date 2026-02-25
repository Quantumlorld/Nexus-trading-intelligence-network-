"""
Nexus Trading System - API Dependencies
FastAPI dependencies, middleware, and common utilities
"""

from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from database.session import get_database_session
from database.models import User, UserRole, UserStatus
from api.auth import get_current_user, get_current_active_user, UserManager, SecurityUtils
from core.logger import get_logger

logger = get_logger()


class UserPermissions:
    """User permission checker"""
    
    @staticmethod
    def can_trade(user: User) -> bool:
        """Check if user can trade"""
        if user.status not in [UserStatus.ACTIVE, UserStatus.TRIAL]:
            return False
        
        if user.subscription_status == "expired":
            return False
        
        if user.role == UserRole.BANNED or user.role == UserRole.SUSPENDED:
            return False
        
        return True
    
    @staticmethod
    def can_auto_trade(user: User) -> bool:
        """Check if user can auto-trade"""
        if not UserPermissions.can_trade(user):
            return False
        
        if not user.auto_trade_enabled:
            return False
        
        # Only verified and elite users can auto-trade
        if user.role not in [UserRole.VERIFIED, UserRole.ELITE, UserRole.ADMIN]:
            return False
        
        return True
    
    @staticmethod
    def can_access_asset(user: User, symbol: str) -> bool:
        """Check if user can access specific asset"""
        if not UserPermissions.can_trade(user):
            return False
        
        if user.allowed_assets and symbol not in user.allowed_assets:
            return False
        
        return True
    
    @staticmethod
    def can_view_performance(user: User, target_user_id: Optional[int] = None) -> bool:
        """Check if user can view performance data"""
        # Users can always view their own performance
        if target_user_id is None or target_user_id == user.id:
            return True
        
        # Admins can view all performance
        if user.role == UserRole.ADMIN:
            return True
        
        # Elite users can view aggregated performance
        if user.role == UserRole.ELITE:
            return True
        
        return False


class RateLimitChecker:
    """Rate limiting checker"""
    
    def __init__(self):
        self.limits = {
            "free": {
                "signals": 100,  # per hour
                "trades": 10,    # per day
                "api_calls": 1000  # per hour
            },
            "verified": {
                "signals": 500,
                "trades": 50,
                "api_calls": 5000
            },
            "elite": {
                "signals": 2000,
                "trades": 200,
                "api_calls": 20000
            },
            "admin": {
                "signals": 10000,
                "trades": 1000,
                "api_calls": 100000
            }
        }
    
    def check_rate_limit(self, user: User, action: str) -> bool:
        """Check if user has exceeded rate limit"""
        user_limits = self.limits.get(user.role.value, self.limits["free"])
        limit = user_limits.get(action, 0)
        
        if limit == 0:
            return False
        
        # TODO: Implement actual rate limiting with Redis or database
        # For now, return True (no rate limiting)
        return True


class RiskValidator:
    """Risk validation utilities"""
    
    @staticmethod
    def validate_trade_request(user: User, trade_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate trade request against user's risk settings"""
        
        # Check if user can trade
        if not UserPermissions.can_trade(user):
            return False, "User is not authorized to trade"
        
        # Check asset permission
        symbol = trade_data.get("symbol", "")
        if not UserPermissions.can_access_asset(user, symbol):
            return False, f"User is not authorized to trade {symbol}"
        
        # Check trade size
        size = trade_data.get("size", 0)
        if size <= 0:
            return False, "Trade size must be greater than 0"
        
        # Check risk amount
        risk_amount = trade_data.get("risk_amount", 0)
        if risk_amount > user.max_dollar_risk:
            return False, f"Risk amount exceeds maximum of ${user.max_dollar_risk}"
        
        # Check daily loss limit
        # TODO: Implement daily loss tracking
        
        return True, "Trade request is valid"
    
    @staticmethod
    def validate_risk_settings(user: User, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate risk settings"""
        
        # Check risk percentage
        risk_percent = settings.get("default_risk_percent", 0)
        if risk_percent < 0.1 or risk_percent > 5.0:
            return False, "Risk percentage must be between 0.1% and 5%"
        
        # Check daily loss limit
        daily_loss = settings.get("max_daily_loss", 0)
        if daily_loss < 1 or daily_loss > 1000:
            return False, "Daily loss limit must be between $1 and $1000"
        
        # Check daily trades
        daily_trades = settings.get("max_daily_trades", 0)
        if daily_trades < 1 or daily_trades > 50:
            return False, "Daily trades must be between 1 and 50"
        
        return True, "Risk settings are valid"


class AuditLogger:
    """Audit logging for security and compliance"""
    
    @staticmethod
    def log_user_action(user: User, action: str, details: Dict[str, Any] = None,
                       ip_address: str = None, user_agent: str = None):
        """Log user action for audit"""
        log_entry = {
            "user_id": user.id,
            "username": user.username,
            "action": action,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"AUDIT: {action} by user {user.username} ({user.id})")
        
        # TODO: Store in database audit table
    
    @staticmethod
    def log_trade_action(user: User, trade_data: Dict[str, Any], action: str,
                         result: str = "success", error: str = None):
        """Log trade action"""
        log_entry = {
            "user_id": user.id,
            "username": user.username,
            "action": f"trade_{action}",
            "trade_data": trade_data,
            "result": result,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"TRADE_AUDIT: {action} {result} for user {user.username}")
        
        # TODO: Store in database audit table
    
    @staticmethod
    def log_security_event(event_type: str, user: User = None, details: Dict[str, Any] = None,
                           ip_address: str = None, severity: str = "info"):
        """Log security event"""
        log_entry = {
            "event_type": event_type,
            "user_id": user.id if user else None,
            "username": user.username if user else None,
            "details": details or {},
            "ip_address": ip_address,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        log_message = f"SECURITY: {event_type}"
        if user:
            log_message += f" by user {user.username}"
        if details:
            log_message += f" - {details}"
        
        if severity == "critical":
            logger.critical(log_message)
        elif severity == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # TODO: Store in database security log table


class RequestValidator:
    """Request validation utilities"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> tuple[bool, str]:
        """Validate trading symbol"""
        if not symbol or len(symbol) < 1 or len(symbol) > 20:
            return False, "Symbol must be between 1 and 20 characters"
        
        # Check if symbol contains only valid characters
        import re
        if not re.match(r'^[A-Z0-9_]+$', symbol):
            return False, "Symbol can only contain uppercase letters, numbers, and underscores"
        
        return True, "Symbol is valid"
    
    @staticmethod
    def validate_direction(direction: str) -> tuple[bool, str]:
        """Validate trade direction"""
        if direction not in ["BUY", "SELL"]:
            return False, "Direction must be either BUY or SELL"
        
        return True, "Direction is valid"
    
    @staticmethod
    def validate_price(price: float) -> tuple[bool, str]:
        """Validate price"""
        if price <= 0:
            return False, "Price must be greater than 0"
        
        if price > 1000000:  # Reasonable upper limit
            return False, "Price seems unreasonably high"
        
        return True, "Price is valid"
    
    @staticmethod
    def validate_size(size: float) -> tuple[bool, str]:
        """Validate trade size"""
        if size <= 0:
            return False, "Size must be greater than 0"
        
        if size > 1000:  # Reasonable upper limit
            return False, "Size seems unreasonably large"
        
        return True, "Size is valid"
    
    @staticmethod
    def validate_confidence(confidence: float) -> tuple[bool, str]:
        """Validate signal confidence"""
        if confidence < 0 or confidence > 1:
            return False, "Confidence must be between 0 and 1"
        
        return True, "Confidence is valid"


# Dependency functions
def get_current_user_with_permissions(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user with permission check"""
    if not UserPermissions.can_trade(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not authorized to trade"
        )
    
    return current_user


def get_auto_trade_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user with auto-trade permission check"""
    if not UserPermissions.can_auto_trade(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not authorized to auto-trade"
        )
    
    return current_user


def get_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current admin user"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


def validate_trade_request(
    trade_data: Dict[str, Any],
    current_user: User = Depends(get_current_user_with_permissions)
) -> tuple[bool, str]:
    """Validate trade request"""
    return RiskValidator.validate_trade_request(current_user, trade_data)


def validate_risk_settings(
    settings: Dict[str, Any],
    current_user: User = Depends(get_current_user)
) -> tuple[bool, str]:
    """Validate risk settings"""
    return RiskValidator.validate_risk_settings(current_user, settings)


def log_request(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Log API request"""
    AuditLogger.log_user_action(
        user=current_user,
        action=f"API_{request.method}_{request.url.path}",
        details={
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params)
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )


# Error handlers
def handle_database_error(error: Exception):
    """Handle database errors"""
    logger.error(f"Database error: {error}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Database operation failed"
    )


def handle_authentication_error(error: Exception):
    """Handle authentication errors"""
    logger.error(f"Authentication error: {error}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication failed"
    )


def handle_authorization_error(error: Exception):
    """Handle authorization errors"""
    logger.error(f"Authorization error: {error}")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied"
    )


def handle_validation_error(error: Exception):
    """Handle validation errors"""
    logger.error(f"Validation error: {error}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid request data"
    )


# Utility functions
def get_user_stats(user: User, db: Session) -> Dict[str, Any]:
    """Get user statistics"""
    # TODO: Implement user statistics calculation
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "current_balance": 10000.0,
        "daily_trades": 0,
        "daily_pnl": 0.0
    }


def check_user_subscription(user: User) -> Dict[str, Any]:
    """Check user subscription status"""
    return {
        "is_active": user.subscription_status == "active",
        "plan": user.subscription_plan,
        "expires_at": user.subscription_end,
        "features": get_subscription_features(user.role)
    }


def get_subscription_features(role: UserRole) -> Dict[str, bool]:
    """Get features available for subscription plan"""
    features = {
        "free": {
            "signals": True,
            "manual_trading": True,
            "auto_trading": False,
            "advanced_analytics": False,
            "api_access": False,
            "priority_support": False
        },
        "verified": {
            "signals": True,
            "manual_trading": True,
            "auto_trading": True,
            "advanced_analytics": True,
            "api_access": True,
            "priority_support": False
        },
        "elite": {
            "signals": True,
            "manual_trading": True,
            "auto_trading": True,
            "advanced_analytics": True,
            "api_access": True,
            "priority_support": True
        },
        "admin": {
            "signals": True,
            "manual_trading": True,
            "auto_trading": True,
            "advanced_analytics": True,
            "api_access": True,
            "priority_support": True
        }
    }
    
    return features.get(role.value, features["free"])
