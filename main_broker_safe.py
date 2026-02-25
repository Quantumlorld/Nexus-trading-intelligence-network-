"""
Nexus Trading System - Broker-Safe Main Application
Production-ready trading system with ledger and reconciliation
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import API routers
from api.auth import router as auth_router
from api.trading_api import router as trading_router, start_trading_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    logger.info("Starting Nexus Trading System - Broker Safe Edition")
    
    try:
        # Start trading services
        await start_trading_services()
        logger.info("Trading services started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start trading services: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Nexus Trading System")
    
    try:
        # Stop trading services
        from core.reconciliation_service import reconciliation_service
        from core.broker_safe_executor import broker_safe_executor
        
        reconciliation_service.stop_reconciliation()
        await broker_safe_executor.stop_background_tasks()
        
        logger.info("Trading services stopped successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Nexus Trading System - Broker Safe",
    description="Production-ready trading system with broker-safe ledger and reconciliation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(trading_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    System health check
    """
    try:
        from core.reconciliation_service import reconciliation_service
        from core.broker_safe_executor import broker_safe_executor
        
        # Check system components
        trading_enabled = await broker_safe_executor._is_trading_enabled()
        reconciliation_running = reconciliation_service.is_running
        broker_connected = broker_safe_executor.mt5_bridge.is_connected
        
        status = "healthy"
        if not broker_connected:
            status = "degraded"
        if not trading_enabled:
            status = "maintenance"
        
        return {
            "status": status,
            "components": {
                "broker_connected": broker_connected,
                "trading_enabled": trading_enabled,
                "reconciliation_running": reconciliation_running
            },
            "timestamp": "2024-01-01T00:00:00Z"  # Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"  # Use actual timestamp
            }
        )

# System status endpoint
@app.get("/api/v1/system/status")
async def system_status():
    """
    Detailed system status
    """
    try:
        from core.reconciliation_service import reconciliation_service
        from core.broker_safe_executor import broker_safe_executor
        
        # Get reconciliation report
        report = await reconciliation_service.get_reconciliation_report(hours=24)
        
        # Get trading control status
        from database.session import get_database_session
        from database.ledger_models import TradingControl
        
        with next(get_database_session()) as db:
            control = db.query(TradingControl).first()
            
        return {
            "trading": {
                "enabled": await broker_safe_executor._is_trading_enabled(),
                "last_reconciliation": reconciliation_service.last_successful_reconciliation,
                "consecutive_failures": reconciliation_service.consecutive_failures
            },
            "reconciliation": {
                "running": reconciliation_service.is_running,
                "report": report
            },
            "control": {
                "trading_enabled": control.trading_enabled if control else True,
                "stop_reason": control.stop_reason if control else None,
                "stopped_at": control.stopped_at.isoformat() if control and control.stopped_at else None
            }
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Emergency stop endpoint
@app.post("/api/v1/system/emergency-stop")
async def emergency_stop(reason: str = "Manual emergency stop"):
    """
    Emergency stop all trading
    """
    try:
        from core.broker_safe_executor import broker_safe_executor
        
        await broker_safe_executor._emergency_stop(reason)
        
        logger.critical(f"Emergency stop activated: {reason}")
        
        return {
            "success": True,
            "message": f"Emergency stop activated: {reason}",
            "timestamp": "2024-01-01T00:00:00Z"  # Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate emergency stop: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "name": "Nexus Trading System",
        "edition": "Broker Safe",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "auth": "/api/v1/auth",
            "trading": "/api/v1/trading",
            "system": "/api/v1/system"
        }
    }

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": "2024-01-01T00:00:00Z"  # Use actual timestamp
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main_broker_safe:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
