"""
Nexus Trading System - System Control API
Admin endpoints for operational control
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from api.auth import get_current_user, TokenManager
from core.system_control import system_control_manager
from core.operational_metrics import operational_metrics
from database.models import User
from database.session import get_database_session
from database.ledger_models import SystemControl

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/v1/system", tags=["system_control"])

# Request models
class TradingControlRequest(BaseModel):
    enabled: bool = Field(..., description="Enable or disable trading")
    reason: str = Field(..., description="Reason for the change")
    updated_by: str = Field(default="admin", description="Who made the change")

class TradingControlResponse(BaseModel):
    success: bool
    message: str
    trading_enabled: bool
    reason: str
    updated_at: datetime
    updated_by: str

class SystemStatusResponse(BaseModel):
    trading_enabled: bool
    trading_reason: str
    broker_connected: bool
    db_connected: bool
    last_broker_check: datetime
    last_db_check: datetime
    consecutive_broker_failures: int
    consecutive_db_failures: int

@router.post("/trading/control", response_model=TradingControlResponse)
async def control_trading(
    request: TradingControlRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Enable or disable trading (admin only)
    """
    try:
        # Check if user has admin privileges
        if current_user.role != "ADMIN":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        if request.enabled:
            success = await system_control_manager.enable_trading(
                request.reason, 
                request.updated_by
            )
            message = "Trading enabled successfully" if success else "Failed to enable trading"
        else:
            success = await system_control_manager.disable_trading(
                request.reason, 
                request.updated_by
            )
            message = "Trading disabled successfully" if success else "Failed to disable trading"
        
        # Get current state
        with next(get_database_session()) as db:
            control = SystemControl.get_current_state(db)
            
            # Update metrics
            operational_metrics.set_trading_state(control.trading_enabled)
        
        return TradingControlResponse(
            success=success,
            message=message,
            trading_enabled=control.trading_enabled,
            reason=control.reason,
            updated_at=control.updated_at,
            updated_by=control.updated_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trading control error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/trading/status", response_model=SystemStatusResponse)
async def get_system_status(current_user: User = Depends(get_current_user)):
    """
    Get current system status
    """
    try:
        # Get trading status
        with next(get_database_session()) as db:
            control = SystemControl.get_current_state(db)
        
        return SystemStatusResponse(
            trading_enabled=control.trading_enabled,
            trading_reason=control.reason,
            broker_connected=operational_metrics.gauges['broker_connected'] > 0,
            db_connected=operational_metrics.gauges['db_connected'] > 0,
            last_broker_check=datetime.utcnow(),  # TODO: Track actual last check time
            last_db_check=datetime.utcnow(),     # TODO: Track actual last check time
            consecutive_broker_failures=system_control_manager.broker_consecutive_failures,
            consecutive_db_failures=system_control_manager.db_consecutive_failures
        )
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-compatible metrics
    """
    try:
        return operational_metrics.get_prometheus_metrics()
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/monitoring/start")
async def start_monitoring(current_user: User = Depends(get_current_user)):
    """
    Start system monitoring (admin only)
    """
    try:
        if current_user.role != "ADMIN":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        await system_control_manager.start_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring started"
        }
        
    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/monitoring/stop")
async def stop_monitoring(current_user: User = Depends(get_current_user)):
    """
    Stop system monitoring (admin only)
    """
    try:
        if current_user.role != "ADMIN":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        await system_control_manager.stop_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring stopped"
        }
        
    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
