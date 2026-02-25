"""
Nexus Trading System - Production Trading API
Broker-safe trading endpoints with ledger integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from api.auth import get_current_user, TokenManager
from core.broker_safe_executor import broker_safe_executor
from core.reconciliation_service import reconciliation_service
from database.session import get_database_session
from database.models import User
from database.ledger_models import TradeLedger, TradeStatus

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/v1/trading", tags=["trading"])
security = HTTPBearer()

# Request models
class TradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY or SELL")
    order_type: str = Field(default="MARKET", description="MARKET or LIMIT")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    entry_price: Optional[float] = Field(None, description="Entry price for limit orders")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    timeframe: str = Field(default="H1", description="Trading timeframe")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TradeResponse(BaseModel):
    success: bool
    trade_uuid: Optional[str] = None
    ledger_id: Optional[int] = None
    message: Optional[str] = None
    execution_time: Optional[float] = None
    broker_result: Optional[Dict[str, Any]] = None

class ReconciliationResponse(BaseModel):
    success: bool
    message: str
    discrepancies_found: int
    trading_enabled: bool

class TradingStatusResponse(BaseModel):
    trading_enabled: bool
    last_reconciliation: Optional[datetime] = None
    consecutive_failures: int
    active_positions: int

@router.post("/execute", response_model=TradeResponse)
async def execute_trade(
    trade_request: TradeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Execute trade with broker-safe ledger management
    """
    try:
        # Validate trade request
        if trade_request.action not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Invalid action. Must be BUY or SELL")
        
        if trade_request.order_type not in ["MARKET", "LIMIT"]:
            raise HTTPException(status_code=400, detail="Invalid order type. Must be MARKET or LIMIT")
        
        if trade_request.order_type == "LIMIT" and not trade_request.entry_price:
            raise HTTPException(status_code=400, detail="Entry price required for limit orders")
        
        # Generate trade UUID
        import uuid
        trade_uuid = str(uuid.uuid4())
        
        # Prepare trade request for executor
        executor_request = {
            'trade_uuid': trade_uuid,
            'user_id': current_user.id,
            'symbol': trade_request.symbol,
            'action': trade_request.action,
            'order_type': trade_request.order_type,
            'quantity': trade_request.quantity,
            'entry_price': trade_request.entry_price,
            'stop_loss': trade_request.stop_loss,
            'take_profit': trade_request.take_profit,
            'timeframe': trade_request.timeframe,
            'metadata': trade_request.metadata or {}
        }
        
        # Execute trade
        result = await broker_safe_executor.execute_trade(
            current_user.id, 
            executor_request
        )
        
        if result['success']:
            # Schedule background tasks
            background_tasks.add_task(
                _post_trade_tasks, 
                current_user.id, 
                result['ledger_id']
            )
            
            return TradeResponse(
                success=True,
                trade_uuid=trade_uuid,
                ledger_id=result.get('ledger_id'),
                message="Trade executed successfully",
                execution_time=result.get('execution_time'),
                broker_result=result.get('broker_result')
            )
        else:
            return TradeResponse(
                success=False,
                trade_uuid=trade_uuid,
                message=result.get('error', 'Unknown error'),
                execution_time=result.get('execution_time')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status(current_user: User = Depends(get_current_user)):
    """
    Get current trading status
    """
    try:
        # Check if trading is enabled
        trading_enabled = await broker_safe_executor._is_trading_enabled()
        
        # Get reconciliation status
        with next(get_database_session()) as db:
            # Count active positions for user
            from database.ledger_models import BrokerPosition
            active_positions = db.query(BrokerPosition).filter(
                BrokerPosition.user_id == current_user.id,
                BrokerPosition.quantity != 0
            ).count()
            
            # Get trading control info
            from database.ledger_models import TradingControl
            control = db.query(TradingControl).first()
            
            return TradingStatusResponse(
                trading_enabled=trading_enabled,
                last_reconciliation=reconciliation_service.last_successful_reconciliation,
                consecutive_failures=reconciliation_service.consecutive_failures,
                active_positions=active_positions
            )
            
    except Exception as e:
        logger.error(f"Trading status error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/positions")
async def get_positions(current_user: User = Depends(get_current_user)):
    """
    Get current positions from broker
    """
    try:
        # Get positions from broker
        positions = broker_safe_executor.mt5_bridge.get_positions()
        
        # Filter positions for current user (if multi-user setup)
        user_positions = positions  # In production, filter by user
        
        return {
            'success': True,
            'positions': user_positions,
            'count': len(user_positions)
        }
        
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/ledger")
async def get_trade_ledger(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get trade ledger for user
    """
    try:
        with next(get_database_session()) as db:
            query = db.query(TradeLedger).filter(
                TradeLedger.user_id == current_user.id
            )
            
            if status:
                try:
                    status_enum = TradeStatus(status.upper())
                    query = query.filter(TradeLedger.status == status_enum)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
            
            # Order by creation time
            query = query.order_by(TradeLedger.created_at.desc())
            
            # Apply pagination
            total = query.count()
            trades = query.offset(offset).limit(limit).all()
            
            return {
                'success': True,
                'trades': [
                    {
                        'trade_uuid': trade.trade_uuid,
                        'symbol': trade.symbol,
                        'action': trade.action,
                        'order_type': trade.order_type,
                        'requested_quantity': trade.requested_quantity,
                        'filled_quantity': trade.filled_quantity,
                        'entry_price': trade.entry_price,
                        'execution_price': trade.execution_price,
                        'status': trade.status.value,
                        'created_at': trade.created_at.isoformat(),
                        'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                        'slippage': trade.slippage,
                        'potential_loss': trade.potential_loss,
                        'actual_loss': trade.actual_loss
                    }
                    for trade in trades
                ],
                'pagination': {
                    'total': total,
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + limit < total
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get ledger error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/reconcile", response_model=ReconciliationResponse)
async def force_reconciliation(
    user_id: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Force reconciliation (admin only or own user)
    """
    try:
        # Check permissions (admin can reconcile any user, users only themselves)
        target_user_id = current_user.id
        if user_id and current_user.role.value == 'admin':
            target_user_id = user_id
        elif user_id and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Can only reconcile own positions")
        
        # Force reconciliation
        await reconciliation_service.force_reconciliation(target_user_id)
        
        return ReconciliationResponse(
            success=True,
            message="Reconciliation completed",
            discrepancies_found=0,  # TODO: Get actual count
            trading_enabled=await broker_safe_executor._is_trading_enabled()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Force reconciliation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/reconciliation/report")
async def get_reconciliation_report(
    hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """
    Get reconciliation report
    """
    try:
        report = await reconciliation_service.get_reconciliation_report(
            user_id=current_user.id,
            hours=hours
        )
        
        return {
            'success': True,
            'report': report
        }
        
    except Exception as e:
        logger.error(f"Reconciliation report error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/trading/enable")
async def enable_trading(current_user: User = Depends(get_current_user)):
    """
    Enable trading (admin only)
    """
    try:
        if current_user.role.value != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        with next(get_database_session()) as db:
            from database.ledger_models import TradingControl
            control = db.query(TradingControl).first()
            if not control:
                control = TradingControl()
                db.add(control)
            
            control.trading_enabled = True
            control.stop_reason = None
            control.stopped_at = None
            control.stop_threshold_exceeded = False
            control.consecutive_failures = 0
            control.updated_at = datetime.utcnow()
            
            db.commit()
            
        return {
            'success': True,
            'message': 'Trading enabled'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enable trading error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/trading/disable")
async def disable_trading(
    reason: str = "Manual disable",
    current_user: User = Depends(get_current_user)
):
    """
    Disable trading (admin only)
    """
    try:
        if current_user.role.value != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        with next(get_database_session()) as db:
            from database.ledger_models import TradingControl
            control = db.query(TradingControl).first()
            if not control:
                control = TradingControl()
                db.add(control)
            
            control.trading_enabled = False
            control.stop_reason = reason
            control.stopped_at = datetime.utcnow()
            control.updated_at = datetime.utcnow()
            
            db.commit()
            
        return {
            'success': True,
            'message': f'Trading disabled: {reason}'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disable trading error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def trading_health():
    """
    Health check for trading system
    """
    try:
        # Check broker connection
        broker_connected = broker_safe_executor.mt5_bridge.is_connected
        
        # Check trading status
        trading_enabled = await broker_safe_executor._is_trading_enabled()
        
        # Check reconciliation service
        reconciliation_running = reconciliation_service.is_running
        
        return {
            'status': 'healthy' if broker_connected and trading_enabled else 'degraded',
            'broker_connected': broker_connected,
            'trading_enabled': trading_enabled,
            'reconciliation_running': reconciliation_running,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trading health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

async def _post_trade_tasks(user_id: int, ledger_id: int):
    """
    Background tasks after trade execution
    """
    try:
        # Update user statistics
        # Send notifications
        # Update analytics
        logger.info(f"Post-trade tasks completed for user {user_id}, ledger {ledger_id}")
        
    except Exception as e:
        logger.error(f"Post-trade tasks failed: {e}")

# Background task to start services
async def start_trading_services():
    """
    Start background trading services
    """
    try:
        # Start broker-safe executor background tasks
        await broker_safe_executor.start_background_tasks()
        
        # Start reconciliation service
        asyncio.create_task(reconciliation_service.start_reconciliation())
        
        logger.info("Trading services started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start trading services: {e}")

# Export router and services
__all__ = [
    'router',
    'start_trading_services',
    'broker_safe_executor',
    'reconciliation_service'
]
