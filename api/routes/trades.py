"""
Nexus Trading System - Trades Routes
Trade management, execution, and monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database.session import get_database_session
from database.models import User, Trade, UserRole, UserStatus
from database.schemas import (
    TradeCreate, TradeUpdate, TradeResponse, PaginatedResponse, PaginationParams
)
from api.auth import get_current_user, get_current_active_user, get_current_admin_user
from api.dependencies import UserPermissions, RateLimitChecker, AuditLogger

router = APIRouter(tags=["Trades"])
logger = logging.getLogger(__name__)

# Rate limiter
rate_limiter = RateLimitChecker()


@router.get("/", response_model=PaginatedResponse)
async def get_user_trades(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's trades"""
    try:
        with get_database_session() as db:
            query = db.query(Trade).filter(Trade.user_id == current_user.id)
            
            # Apply filters
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            if status:
                query = query.filter(Trade.status == status.upper())
            
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(Trade.entry_time >= cutoff_date)
            
            # Order by entry time
            query = query.order_by(desc(Trade.entry_time))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            trades = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            trade_responses = [TradeResponse.from_orm(trade) for trade in trades]
            
            return PaginatedResponse.create(
                items=trade_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to get user trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user trades"
        )


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade_by_id(
    trade_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Get trade by ID"""
    try:
        with get_database_session() as db:
            trade = db.query(Trade).filter(
                and_(Trade.id == trade_id, Trade.user_id == current_user.id)
            ).first()
            
            if not trade:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Trade not found"
                )
            
            return TradeResponse.from_orm(trade)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trade"
        )


@router.post("/", response_model=TradeResponse)
async def create_trade(
    trade_data: TradeCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new trade"""
    try:
        # Check if user can trade
        if not UserPermissions.can_trade(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized to trade"
            )
        
        # Check if user can access this asset
        if not UserPermissions.can_access_asset(current_user, trade_data.symbol):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User is not authorized to trade {trade_data.symbol}"
            )
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(current_user.id, "create_trade"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many trade attempts"
            )
        
        # Validate trade data
        from api.dependencies import RequestValidator, RiskValidator
        
        # Validate symbol
        is_valid, message = RequestValidator.validate_symbol(trade_data.symbol)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Validate direction
        is_valid, message = RequestValidator.validate_direction(trade_data.direction)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Validate price
        is_valid, message = RequestValidator.validate_price(trade_data.entry_price)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Validate size
        is_valid, message = RequestValidator.validate_size(trade_data.size)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Validate confidence
        if trade_data.signal_confidence is not None:
            is_valid, message = RequestValidator.validate_confidence(trade_data.signal_confidence)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=message
                )
        
        # Validate against user's risk settings
        is_valid, message = RiskValidator.validate_trade_request(
            current_user, trade_data.dict()
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        with get_database_session() as db:
            # Create trade
            trade = Trade(
                user_id=current_user.id,
                symbol=trade_data.symbol,
                direction=trade_data.direction,
                size=trade_data.size,
                entry_price=trade_data.entry_price,
                stop_loss=trade_data.stop_loss,
                take_profit=trade_data.take_profit,
                strategy=trade_data.strategy,
                signal_confidence=trade_data.signal_confidence,
                signal_time=datetime.utcnow(),
                entry_time=datetime.utcnow(),
                status="OPEN",
                commission=0.0,  # TODO: Calculate commission
                swap=0.0,      # TODO: Calculate swap
                sl_locked=False,
                tp_extended=False,
                runner_mode=False
            )
            
            db.add(trade)
            db.commit()
            db.refresh(trade)
            
            # Log trade creation
            AuditLogger.log_trade_action(
                user=current_user,
                trade_data=trade_data.dict(),
                action="trade_created",
                result="success"
            )
            
            return TradeResponse.from_orm(trade)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create trade"
        )


@router.put("/{trade_id}", response_model=TradeResponse)
async def update_trade(
    trade_id: int,
    trade_update: TradeUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update an existing trade"""
    try:
        with get_database_session() as db:
            trade = db.query(Trade).filter(
                and_(Trade.id == trade_id, Trade.user_id == current_user.id)
            ).first()
            
            if not trade:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Trade not found"
                )
            
            # Only allow updates for open trades
            if trade.status != "OPEN":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot update closed trade"
                )
            
            # Update fields
            update_data = trade_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(trade, field):
                    setattr(trade, field, value)
            
            # Calculate duration if exit time is provided
            if trade_update.exit_time and trade.entry_time:
                trade.duration_minutes = int(
                    (trade_update.exit_time - trade.entry_time).total_seconds() / 60
                )
            
            # Calculate P&L if exit price is provided
            if trade_update.exit_price and trade.entry_price:
                if trade.direction == "BUY":
                    trade.pnl = (trade_update.exit_price - trade.entry_price) * trade.size * 100000
                else:  # SELL
                    trade.pnl = (trade.entry_price - trade_update.exit_price) * trade.size * 100000
            
            # Update status if closing
            if trade_update.status and trade_update.status != trade.status:
                if trade_update.status == "CLOSED":
                    trade.exit_time = trade_update.exit_time or datetime.utcnow()
                    trade.exit_reason = trade_update.exit_reason or "MANUAL"
                trade.status = trade_update.status
            
            db.merge(trade)
            db.commit()
            db.refresh(trade)
            
            # Log trade update
            AuditLogger.log_trade_action(
                user=current_user,
                trade_data=trade_update.dict(),
                action="trade_updated",
                result="success"
            )
            
            return TradeResponse.from_orm(trade)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update trade"
        )


@router.delete("/{trade_id}")
async def close_trade(
    trade_id: int,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Close a trade"""
    try:
        with get_database_session() as db:
            trade = db.query(Trade).filter(
                and_(Trade.id == trade_id, Trade.user_id == current_user.id)
            ).first()
            
            if not trade:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Trade not found"
                )
            
            if trade.status != "OPEN":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Trade is already closed"
                )
            
            # Close the trade
            trade.status = "CLOSED"
            trade.exit_time = datetime.utcnow()
            trade.exit_reason = reason or "MANUAL"
            
            # Calculate P&L
            if trade.direction == "BUY":
                trade.pnl = (trade.exit_price or trade.entry_price - trade.stop_loss) * trade.size * 100000
            else:  # SELL
                trade.pnl = (trade.entry_price - (trade.exit_price or trade.take_profit)) * trade.size * 100000
            
            # Calculate duration
            trade.duration_minutes = int(
                (trade.exit_time - trade.entry_time).total_seconds() / 60
            )
            
            db.merge(trade)
            db.commit()
            db.refresh(trade)
            
            # Log trade closure
            AuditLogger.log_trade_action(
                user=current_user,
                trade_data={"trade_id": trade_id, "reason": reason},
                action="trade_closed",
                result="success"
            )
            
            return {
                "message": "Trade closed successfully",
                "trade": TradeResponse.from_orm(trade)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close trade: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close trade"
        )


@router.get("/open", response_model=PaginatedResponse)
async def get_open_trades(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's open trades"""
    try:
        with get_database_session() as db:
            query = db.query(Trade).filter(
                and_(Trade.user_id == current_user.id, Trade.status == "OPEN")
            )
            
            # Order by entry time
            query = query.order_by(desc(Trade.entry_time))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            trades = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            trade_responses = [TradeResponse.from_orm(trade) for trade in trades]
            
            return PaginatedResponse.create(
                items=trade_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to get open trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get open trades"
        )


@router.get("/statistics")
async def get_trade_statistics(
    current_user: User = Depends(get_current_active_user),
    days: int = Query(30, ge=1, le=365)
):
    """Get current user's trade statistics"""
    try:
        with get_database_session() as db:
            # TODO: Implement comprehensive trade statistics
            stats = {
                "total_trades": 0,
                "open_trades": 0,
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_trade_duration": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "daily_stats": {},
                "asset_performance": {},
                "strategy_performance": {},
                "risk_reward_ratio": 0.0
            }
            
            return stats
    
    except Exception as e:
        logger.error(f"Failed to get trade statistics: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trade statistics"
        )


@router.get("/performance")
async def get_trade_performance(
    current_user: User = Depends(get_current_active_user),
    days: int = Query(30, ge=1, le=365)
):
    """Get current user's trade performance metrics"""
    try:
        # TODO: Implement comprehensive performance calculation
        performance = {
            "period_days": days,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "var": 0.0,
            "cvar": 0.0,
            "downside_deviation": 0.0,
            "upside_deviation": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade_duration": 0.0,
            "avg_risk_per_trade": 0.0,
            "max_risk_per_trade": 0.0,
            "risk_reward_ratio": 0.0,
            "monthly_returns": [],
            "equity_curve": [],
            "drawdown_periods": []
        }
        
        return performance
    
    except Exception as e:
        logger.error(f"Failed to get trade performance: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trade performance"
        )


# Admin routes
@router.get("/admin/all", response_model=PaginatedResponse)
async def get_all_trades(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_admin_user)
):
    """Get all trades (admin only)"""
    try:
        with get_database_session() as db:
            query = db.query(Trade)
            
            # Apply filters
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            if status:
                query = query.filter(Trade.status == status.upper())
            
            if user_id:
                query = query.filter(Trade.user_id == user_id)
            
            # Order by entry time
            query = query.order_by(desc(Trade.entry_time))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            trades = query.offset(pagination.offset).limit(pagination.size).all()
            
            # Convert to response format
            trade_responses = [TradeResponse.from_orm(trade) for trade in trades]
            
            return PaginatedResponse.create(
                items=trade_responses,
                total=total,
                page=pagination.page,
                size=pagination.size
            )
    
    except Exception as e:
        logger.error(f"Failed to get all trades: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all trades"
        )


@router.get("/admin/statistics")
async def get_all_trade_statistics(
    current_user: User = Depends(get_current_admin_user)
):
    """Get all trade statistics (admin only)"""
    try:
        # TODO: Implement comprehensive trade statistics
        stats = {
            "total_trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "total_pnl": 0.0,
            "total_volume": 0.0,
            "best_performing_user": None,
            "worst_performing_user": None,
            "most_traded_asset": None,
            "least_traded_asset": None,
            "trade_distribution": {
                "by_symbol": {},
                "by_direction": {"BUY": 0, "SELL": 0},
                "by_status": {"OPEN": 0, "CLOSED": 0, "CANCELLED": 0},
                "by_strategy": {},
                "by_user_role": {}
            },
            "performance_metrics": {
                "overall_win_rate": 0.0,
                "overall_profit_factor": 0.0,
                "overall_sharpe_ratio": 0.0,
                "system_drawdown": 0.0
            },
            "risk_metrics": {
                "avg_risk_per_trade": 0.0,
                "max_risk_per_trade": 0.0,
                "risk_reward_ratio": 0.0,
                "stop_loss_hit_rate": 0.0,
                "take_profit_hit_rate": 0.0
            }
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get all trade statistics: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all trade statistics"
        )
