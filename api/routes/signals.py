"""
Nexus Trading System - Signals Routes
Signal generation, distribution, and management
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database.session import get_database_session
from database.models import User, Signal, UserRole, UserStatus
from database.schemas import (
    SignalCreate, SignalResponse, PaginatedResponse, PaginationParams
)
from api.auth import get_current_user, get_current_active_user, get_current_admin_user
from api.dependencies import UserPermissions, RateLimitChecker, AuditLogger

router = APIRouter(tags=["Signals"])
logger = logging.getLogger(__name__)

# Rate limiter
rate_limiter = RateLimitChecker()


@router.get("/", response_model=PaginatedResponse)
async def get_signals(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    active_only: bool = Query(True),
    current_user: User = Depends(get_current_active_user)
):
    """Get available signals for current user"""
    try:
        with get_database_session() as db:
            query = db.query(Signal)
            
            # Filter by user permissions
            if current_user.allowed_assets:
                query = query.filter(Signal.symbol.in_(current_user.allowed_assets))
            
            # Apply filters
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            if strategy:
                query = query.filter(Signal.strategy == strategy)
            
            if min_confidence is not None:
                query = query.filter(Signal.confidence >= min_confidence)
            
            if active_only:
                query = query.filter(
                    and_(
                        Signal.expires_at > datetime.utcnow(),
                        Signal.generated_at > datetime.utcnow() - timedelta(hours=24)
                    )
                )
            
            # Order by confidence and generation time
            query = query.order_by(desc(Signal.confidence), desc(Signal.generated_at))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            signals = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            signal_responses = [SignalResponse.from_orm(signal) for signal in signals]
            
            return PaginatedResponse.create(
                items=signal_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get signals"
        )


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal_by_id(
    signal_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Get signal by ID"""
    try:
        with get_database_session() as db:
            signal = db.query(Signal).filter(Signal.id == signal_id).first()
            
            if not signal:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Signal not found"
                )
            
            # Check if user can access this signal
            if not UserPermissions.can_access_asset(current_user, signal.symbol):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User is not authorized to access {signal.symbol}"
                )
            
            return SignalResponse.from_orm(signal)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get signal"
        )


@router.post("/", response_model=SignalResponse)
async def create_signal(
    signal_data: SignalCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new signal (for testing/admin use)"""
    try:
        # Check if user can create signals
        if current_user.role not in [UserRole.ELITE, UserRole.ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only elite and admin users can create signals"
            )
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(current_user.id, "create_signal"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many signal creation attempts"
            )
        
        # Validate signal data
        if not UserPermissions.can_access_asset(current_user, signal_data.symbol):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User is not authorized to create signals for {signal_data.symbol}"
            )
        
        with get_database_session() as db:
            # Create signal
            signal = Signal(
                symbol=signal_data.symbol,
                direction=signal_data.direction,
                confidence=signal_data.confidence,
                strategy=signal_data.strategy,
                entry_price=signal_data.entry_price,
                stop_loss=signal_data.stop_loss,
                take_profit=signal_data.take_profit,
                market_regime=signal_data.market_regime,
                volatility_level=signal_data.volatility_level,
                session_time=signal_data.session_time,
                expires_at=signal_data.expires_at,
                generated_at=datetime.utcnow(),
                distributed_to_users=[current_user.id]
            )
            
            db.add(signal)
            db.commit()
            db.refresh(signal)
            
            # Log signal creation
            AuditLogger.log_user_action(
                user=current_user,
                action="signal_created",
                details={
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy
                }
            )
            
            return SignalResponse.from_orm(signal)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create signal"
        )


@router.post("/{signal_id}/follow")
async def follow_signal(
    signal_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Follow a signal (mark as used)"""
    try:
        with get_database_session() as db:
            signal = db.query(Signal).filter(Signal.id == signal_id).first()
            
            if not signal:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Signal not found"
                )
            
            # Check if user can access this signal
            if not UserPermissions.can_access_asset(current_user, signal.symbol):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User is not authorized to access {signal.symbol}"
                )
            
            # Check if signal is still valid
            if signal.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Signal has expired"
                )
            
            # Add user to distributed list
            if signal.distributed_to_users is None:
                signal.distributed_to_users = []
            
            if current_user.id not in signal.distributed_to_users:
                signal.distributed_to_users.append(current_user.id)
                
                # Update user responses
                if signal.user_responses is None:
                    signal.user_responses = {}
                
                signal.user_responses[str(current_user.id)] = {
                    "action": "followed",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                db.merge(signal)
                db.commit()
                db.refresh(signal)
                
                # Log signal follow
                AuditLogger.log_user_action(
                    user=current_user,
                    action="signal_followed",
                    details={
                        "signal_id": signal_id,
                        "symbol": signal.symbol,
                        "direction": signal.direction
                    }
                )
            
            return {
                "message": "Signal followed successfully",
                "signal": SignalResponse.from_orm(signal)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to follow signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to follow signal"
        )


@router.post("/{signal_id}/result")
async def update_signal_result(
    signal_id: int,
    result: str,  # WIN, LOSS, BREAKEVEN
    pnl: Optional[float] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Update signal result (for testing/admin use)"""
    try:
        # Check if user can update signals
        if current_user.role not in [UserRole.ELITE, UserRole.ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only elite and admin users can update signal results"
            )
        
        with get_database_session() as db:
            signal = db.query(Signal).filter(Signal.id == signal_id).first()
            
            if not signal:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Signal not found"
                )
            
            # Validate result
            if result not in ["WIN", "LOSS", "BREAKEVEN"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Result must be WIN, LOSS, or BREAKEVEN"
                )
            
            # Update signal
            signal.actual_result = result
            signal.actual_pnl = pnl if pnl is not None else 0.0
            
            db.merge(signal)
            db.commit()
            db.refresh(signal)
            
            # Log signal result update
            AuditLogger.log_user_action(
                user=current_user,
                action="signal_result_updated",
                details={
                    "signal_id": signal_id,
                    "result": result,
                    "pnl": pnl
                }
            )
            
            return {
                "message": "Signal result updated successfully",
                "signal": SignalResponse.from_orm(signal)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update signal result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update signal result"
        )


@router.get("/statistics")
async def get_signal_statistics(
    current_user: User = Depends(get_current_active_user),
    days: int = Query(30, ge=1, le=365)
):
    """Get signal statistics for the current user"""
    try:
        with get_database_session() as db:
            # TODO: Implement comprehensive signal statistics
            stats = {
                "total_signals": 0,
                "followed_signals": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "best_strategy": None,
                "best_asset": None,
                "daily_signals": {},
                "strategy_performance": {},
                "asset_performance": {},
                "confidence_distribution": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.5-0.8
                    "low": 0   # < 0.5
                },
                "time_distribution": {
                    "london": 0,
                    "new_york": 0,
                    "asian": 0,
                    "overlap": 0
                }
            }
            
            return stats
    
    except Exception as e:
        logger.error(f"Failed to get signal statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get signal statistics"
        )


@router.get("/performance")
async def get_signal_performance(
    current_user: User = Depends(get_current_active_user),
    days: int = Query(30, ge=1, le=365)
):
    """Get signal performance metrics"""
    try:
        # TODO: Implement signal performance calculation
        performance = {
            "period_days": days,
            "total_signals": 0,
            "successful_signals": 0,
            "win_rate": 0.0,
            "avg_pnl_per_signal": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "best_signal": None,
            "worst_signal": None,
            "strategy_performance": {},
            "asset_performance": {},
            "confidence_vs_performance": {
                "high_confidence_win_rate": 0.0,
                "medium_confidence_win_rate": 0.0,
                "low_confidence_win_rate": 0.0
            }
        }
        
        return performance
    
    except Exception as e:
        logger.error(f"Failed to get signal performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get signal performance"
        )


# Admin routes
@router.get("/admin/all", response_model=PaginatedResponse)
async def get_all_signals(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    result: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user)
):
    """Get all signals (admin only)"""
    try:
        with get_database_session() as db:
            query = db.query(Signal)
            
            # Apply filters
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            if strategy:
                query = query.filter(Signal.strategy == strategy)
            
            if result:
                query = query.filter(Signal.actual_result == result)
            
            # Order by generation time
            query = query.order_by(desc(Signal.generated_at))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            signals = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            signal_responses = [SignalResponse.from_orm(signal) for signal in signals]
            
            return PaginatedResponse.create(
                items=signal_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to get all signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all signals"
        )


@router.get("/admin/statistics")
async def get_all_signal_statistics(
    current_user: User = Depends(get_current_admin_user)
):
    """Get all signal statistics (admin only)"""
    try:
        # TODO: Implement comprehensive signal statistics
        stats = {
            "total_signals": 0,
            "active_signals": 0,
            "expired_signals": 0,
            "total_follows": 0,
            "overall_win_rate": 0.0,
            "total_pnl": 0.0,
            "best_performing_strategy": None,
            "worst_performing_strategy": None,
            "most_traded_asset": None,
            "least_traded_asset": None,
            "signal_distribution": {
                "by_strategy": {},
                "by_asset": {},
                "by_direction": {"BUY": 0, "SELL": 0},
                "by_confidence": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                }
            },
            "user_engagement": {
                "total_users": 0,
                "active_users": 0,
                "avg_signals_per_user": 0.0
            }
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get all signal statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all signal statistics"
        )
