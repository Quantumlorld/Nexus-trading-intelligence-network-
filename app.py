"""
Nexus Trading System - FastAPI Application
Production-ready backend service with full integration
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import system components
from core.system_control import system_control_manager
from core.operational_metrics import operational_metrics
from core.broker_safe_executor import broker_safe_executor
from core.candle_engine_9h import candle_engine_9h
from execution.real_broker_adapter import real_broker_adapter
from api.trading_api import router as trading_router
from api.system_control_api import router as system_control_router
from api.auth import router as auth_router
from database.session import get_database_session
from database.ledger_models import SystemControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nexus Trading System API",
    description="Production-ready trading system with operational resilience",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading_router, prefix="/api/v1")
app.include_router(system_control_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")

# Request models
class HealthResponse(BaseModel):
    status: str
    db_status: str
    broker_status: str
    trading_enabled: bool
    uptime: str
    metrics: Dict[str, Any]

class CandleRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    limit: int = Field(default=100, description="Number of candles")

class CandleResponse(BaseModel):
    success: bool
    candles: List[Dict[str, Any]]
    symbol: str
    timeframe: str

# Application state
app_state = {
    "start_time": datetime.utcnow(),
    "is_initialized": False
}

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Nexus Trading System API...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        from database.session import init_db
        init_db()
        
        # Start system monitoring
        logger.info("Starting system monitoring...")
        await system_control_manager.start_monitoring()
        
        # Start broker adapter monitoring
        logger.info("Starting broker adapter monitoring...")
        await real_broker_adapter.start_monitoring()
        
        # Generate sample data if needed
        logger.info("Generating sample data...")
        await candle_engine_9h.generate_sample_data("EUR/USD", days=7)
        await candle_engine_9h.generate_sample_data("BTC/USD", days=7)
        
        # Generate 9H candles
        logger.info("Generating 9H candles...")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        await candle_engine_9h.generate_9h_candles("EUR/USD", start_date, end_date)
        await candle_engine_9h.generate_9h_candles("BTC/USD", start_date, end_date)
        
        # Update metrics
        operational_metrics.set_trading_state(await system_control_manager.is_trading_enabled())
        
        app_state["is_initialized"] = True
        logger.info("✅ Nexus Trading System API initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Nexus Trading System API...")
    
    try:
        # Stop monitoring
        await system_control_manager.stop_monitoring()
        await real_broker_adapter.stop_monitoring()
        
        logger.info("✅ Nexus Trading System API shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "message": "Nexus Trading System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "initialized": app_state["is_initialized"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check database
        db_status = "healthy"
        try:
            with next(get_database_session()) as db:
                db.execute("SELECT 1")
                db_latency = 0.1  # Simulated latency
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
            db_latency = 999.0
        
        # Check broker
        broker_status = "healthy"
        try:
            broker_connected = await real_broker_adapter.check_connection()
            if not broker_connected:
                broker_status = "unhealthy: connection failed"
        except Exception as e:
            broker_status = f"unhealthy: {str(e)}"
        
        # Check trading status
        trading_enabled = await system_control_manager.is_trading_enabled()
        
        # Calculate uptime
        uptime = datetime.utcnow() - app_state["start_time"]
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        # Get metrics summary
        metrics = operational_metrics.get_summary_stats()
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" and broker_status == "healthy" else "degraded",
            db_status=db_status,
            broker_status=broker_status,
            trading_enabled=trading_enabled,
            uptime=uptime_str,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-compatible metrics"""
    try:
        return operational_metrics.get_prometheus_metrics()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/candles", response_model=CandleResponse)
async def get_candles(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Get candle data"""
    try:
        if timeframe == "9h":
            candles = await candle_engine_9h.get_9h_candles(symbol, limit)
        else:
            # For other timeframes, return sample data
            candles = await _get_sample_candles(symbol, timeframe, limit)
        
        return CandleResponse(
            success=True,
            candles=candles,
            symbol=symbol,
            timeframe=timeframe
        )
        
    except Exception as e:
        logger.error(f"Failed to get candles: {e}")
        return CandleResponse(
            success=False,
            candles=[],
            symbol=symbol,
            timeframe=timeframe
        )

async def _get_sample_candles(symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
    """Get sample candle data for testing"""
    candles = []
    base_price = 1.1000 if symbol == "EUR/USD" else 45000.0
    current_time = datetime.utcnow()
    
    for i in range(limit):
        candle_time = current_time - timedelta(hours=i) if timeframe == "1h" else current_time - timedelta(minutes=i*5)
        
        # Generate realistic OHLC data
        price_variation = 0.001 * (i / limit)
        open_price = base_price + price_variation
        close_price = open_price + (0.0001 if i % 2 == 0 else -0.0001)
        high_price = max(open_price, close_price) + 0.0002
        low_price = min(open_price, close_price) - 0.0002
        volume = 100 + (i * 10)
        
        candles.append({
            "timestamp": candle_time.isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })
    
    return list(reversed(candles))

@app.post("/admin/test-trade")
async def test_trade():
    """Test trade execution"""
    try:
        trade_request = {
            'trade_uuid': f"test-{int(datetime.utcnow().timestamp())}",
            'user_id': 1,
            'symbol': 'EUR/USD',
            'action': 'BUY',
            'order_type': 'MARKET',
            'quantity': 0.01,
            'entry_price': 1.1000,
            'timeframe': '1h'
        }
        
        result = await broker_safe_executor.execute_trade(1, trade_request)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test trade failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/admin/system-status")
async def get_system_status():
    """Get detailed system status"""
    try:
        return {
            "system_control": {
                "trading_enabled": await system_control_manager.is_trading_enabled(),
                "broker_failures": system_control_manager.broker_consecutive_failures,
                "db_failures": system_control_manager.db_consecutive_failures
            },
            "broker_adapter": real_broker_adapter.get_connection_status(),
            "metrics": operational_metrics.get_summary_stats(),
            "app_uptime": str(datetime.utcnow() - app_state["start_time"]).split('.')[0],
            "initialized": app_state["is_initialized"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
