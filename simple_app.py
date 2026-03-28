"""
Nexus Trading System - Simple FastAPI Application
Minimal working version for system integration
"""

# Core imports
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Import MT5 integration
from mt5_integration import get_mt5_connector
from universal_mt5_connector import get_universal_connector
from mt5_bridge_client import BridgeMT5Connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MT5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.info("✅ MetaTrader5 module available")
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logger.warning("⚠️ MetaTrader5 module not available")

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
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:6000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class HealthResponse(BaseModel):
    status: str
    db_status: str
    broker_status: str
    trading_enabled: bool
    uptime: str

class CandleResponse(BaseModel):
    success: bool
    candles: List[Dict[str, Any]]
    symbol: str
    timeframe: str

class TradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    order_type: str = Field(default="MARKET", description="MARKET or LIMIT")
    price: Optional[float] = Field(None, description="Entry price for limit orders")

class ResonanceResult(BaseModel):
    signal_3h: str
    signal_6h: str
    signal_9h: str
    confidence: int

# Global state
app_state = {
    "trading_enabled": False,
    "broker_connected": False,
    "db_connected": True,  # Simulated
    "start_time": datetime.utcnow(),
    "mt5_account": None,
    "mt5_server": None,
    "demo_mode": True,
    "trade_count": 0,
    "adaptive_learning": False,
    "current_phase": "baseline"
}

# Initialize connectors
mt5_connector = get_mt5_connector()
universal_connector = get_universal_connector()
bridge_connector = BridgeMT5Connector()

# XM Broker Credentials (User's actual accounts)
XM_BROKER_ACCOUNT = "primeworld069"
XM_BROKER_PASSWORD = "REPLACE_WITH_ACTUAL_PASSWORD"
XM_MQL5_ACCOUNT = "Quantumlorld"
XM_MQL5_PASSWORD = "REPLACE_WITH_ACTUAL_PASSWORD"

# XM Global servers to try
XM_SERVERS = [
    "XMGlobal-MT5 10",
    "XMGlobal-MT5 5",
    "XMGlobal-MT5",
    "XMGlobal-MT5 2",
    "XMGlobal-MT5 9",
    "XMGlobal-MT5 7",
    "XMGlobal-MT5 8"
]

# Demo account (fallback)
MT5_DEMO_ACCOUNT = 103969793
MT5_DEMO_PASSWORD = "*d8qNgQq"
MT5_DEMO_SERVER = "MetaQuotes-Demo"

# Demo trading configuration
DEMO_PHASES = {
    "baseline": {"max_trades": 100, "risk_per_trade": 1.0},
    "learning": {"max_trades": 200, "adaptive_features": True},
    "optimization": {"max_trades": 200, "full_optimization": True}
}

# Get MT5 connectors with working configuration
mt5_connector = get_mt5_connector()
# Override with working path and method
mt5_connector.terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
universal_connector = get_universal_connector()

# Mock data
mock_candles = {
    "EUR/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 1.1000, "high": 1.1050, "low": 1.0980, "close": 1.1020, "volume": 1000},
        {"timestamp": "2026-03-01T01:00:00Z", "open": 1.1020, "high": 1.1070, "low": 1.1000, "close": 1.1040, "volume": 1200},
        {"timestamp": "2026-03-01T02:00:00Z", "open": 1.1040, "high": 1.1080, "low": 1.1020, "close": 1.1060, "volume": 900},
        {"timestamp": "2026-03-01T03:00:00Z", "open": 1.1060, "high": 1.1100, "low": 1.1040, "close": 1.1080, "volume": 1100},
        {"timestamp": "2026-03-01T04:00:00Z", "open": 1.1080, "high": 1.1120, "low": 1.1060, "close": 1.1100, "volume": 1300},
    ],
    "XAU/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 2050.50, "high": 2055.80, "low": 2048.20, "close": 2052.30, "volume": 800},
        {"timestamp": "2026-03-01T01:00:00Z", "open": 2052.30, "high": 2058.00, "low": 2050.50, "close": 2055.20, "volume": 950},
        {"timestamp": "2026-03-01T02:00:00Z", "open": 2055.20, "high": 2060.50, "low": 2052.30, "close": 2058.80, "volume": 700},
        {"timestamp": "2026-03-01T03:00:00Z", "open": 2058.80, "high": 2063.00, "low": 2055.20, "close": 2060.50, "volume": 850},
        {"timestamp": "2026-03-01T04:00:00Z", "open": 2060.50, "high": 2065.00, "low": 2058.80, "close": 2063.00, "volume": 900},
    ],
    "USDX": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 103.20, "high": 103.50, "low": 103.00, "close": 103.35, "volume": 500},
        {"timestamp": "2026-03-01T01:00:00Z", "open": 103.35, "high": 103.80, "low": 103.20, "close": 103.65, "volume": 600},
        {"timestamp": "2026-03-01T02:00:00Z", "open": 103.65, "high": 104.10, "low": 103.35, "close": 103.90, "volume": 450},
        {"timestamp": "2026-03-01T03:00:00Z", "open": 103.90, "high": 104.40, "low": 103.65, "close": 104.15, "volume": 550},
        {"timestamp": "2026-03-01T04:00:00Z", "open": 104.15, "high": 104.70, "low": 103.90, "close": 104.50, "volume": 500},
    ],
    "BTC/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 42500, "high": 43200, "low": 42100, "close": 42850, "volume": 120},
        {"timestamp": "2026-03-01T01:00:00Z", "open": 42850, "high": 43500, "low": 42500, "close": 43200, "volume": 150},
        {"timestamp": "2026-03-01T02:00:00Z", "open": 43200, "high": 43800, "low": 42850, "close": 43500, "volume": 100},
        {"timestamp": "2026-03-01T03:00:00Z", "open": 43500, "high": 44100, "low": 43200, "close": 43800, "volume": 130},
        {"timestamp": "2026-03-01T04:00:00Z", "open": 43800, "high": 44300, "low": 43500, "close": 44100, "volume": 140},
    ]
}

# 9H candles (generated from 1H data)
mock_9h_candles = {
    "EUR/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 1.1000, "high": 1.1120, "low": 1.0980, "close": 1.1100, "volume": 5500},
        {"timestamp": "2026-03-01T09:00:00Z", "open": 1.1100, "high": 1.1220, "low": 1.1080, "close": 1.1200, "volume": 5800},
        {"timestamp": "2026-03-01T18:00:00Z", "open": 1.1200, "high": 1.1320, "low": 1.1180, "close": 1.1300, "volume": 6200},
    ],
    "XAU/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 2050.50, "high": 2065.00, "low": 2048.20, "close": 2060.50, "volume": 3200},
        {"timestamp": "2026-03-01T09:00:00Z", "open": 2060.50, "high": 2075.00, "low": 2055.00, "close": 2070.00, "volume": 3400},
        {"timestamp": "2026-03-01T18:00:00Z", "open": 2070.00, "high": 2085.00, "low": 2065.00, "close": 2080.00, "volume": 3600},
    ],
    "USDX": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 103.20, "high": 104.70, "low": 103.00, "close": 104.50, "volume": 2000},
        {"timestamp": "2026-03-01T09:00:00Z", "open": 104.50, "high": 106.00, "low": 104.20, "close": 105.80, "volume": 2200},
        {"timestamp": "2026-03-01T18:00:00Z", "open": 105.80, "high": 107.30, "low": 105.50, "close": 107.00, "volume": 2400},
    ],
    "BTC/USD": [
        {"timestamp": "2026-03-01T00:00:00Z", "open": 45000, "high": 46200, "low": 44800, "close": 46000, "volume": 550},
        {"timestamp": "2026-03-01T09:00:00Z", "open": 46000, "high": 47200, "low": 45800, "close": 47000, "volume": 580},
        {"timestamp": "2026-03-01T18:00:00Z", "open": 47000, "high": 48200, "low": 46800, "close": 48000, "volume": 620},
    ]
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Nexus Trading System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "start_time": app_state["start_time"].isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Simulate database check
        db_status = "healthy" if app_state["db_connected"] else "unhealthy"
        
        # Simulate broker check
        broker_status = "healthy" if app_state["broker_connected"] else "unhealthy"
        
        # Check trading status
        trading_enabled = app_state["trading_enabled"]
        
        # Calculate uptime
        uptime = datetime.utcnow() - app_state["start_time"]
        uptime_str = str(uptime).split('.')[0]
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" and broker_status == "healthy" else "degraded",
            db_status=db_status,
            broker_status=broker_status,
            trading_enabled=trading_enabled,
            uptime=uptime_str
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-compatible metrics"""
    metrics_text = f"""
# HELP nexus_trading_enabled Trading enabled status
# TYPE nexus_trading_enabled gauge
nexus_trading_enabled {1 if app_state["trading_enabled"] else 0}

# HELP nexus_broker_connected Broker connection status
# TYPE nexus_broker_connected gauge
nexus_broker_connected {1 if app_state["broker_connected"] else 0}

# HELP nexus_db_connected Database connection status
# TYPE nexus_db_connected gauge
nexus_db_connected {1 if app_state["db_connected"] else 0}

# HELP nexus_uptime_seconds System uptime in seconds
# TYPE nexus_uptime_seconds counter
nexus_uptime_seconds {(datetime.utcnow() - app_state["start_time"]).total_seconds()}
"""
    return metrics_text

@app.get("/candles", response_model=CandleResponse)
async def get_candles(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Get candle data"""
    try:
        if timeframe == "9h":
            candles = mock_9h_candles.get(symbol, [])
        else:
            candles = mock_candles.get(symbol, [])
        
        return CandleResponse(
            success=True,
            candles=candles[:limit],
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

def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if s.upper() in {"EURUSD", "EUR/USD"}:
        return "EURUSD"
    return s.replace("/", "")

def _simple_timeframe_signal(timeframe: str, symbol: str) -> str:
    sym = symbol
    if timeframe == "9h":
        candles = mock_9h_candles.get(sym, [])
    else:
        candles = mock_candles.get(sym, [])

    if not candles:
        return random.choice(["BUY", "SELL"])

    first = candles[-1] if len(candles) == 1 else candles[0]
    last = candles[-1]
    try:
        open_p = float(first.get("open", 0))
        close_p = float(last.get("close", 0))
    except Exception:
        return random.choice(["BUY", "SELL"])

    return "BUY" if close_p >= open_p else "SELL"

def resonance_validate(symbol: str, intended_action: str) -> ResonanceResult:
    signal_3h = _simple_timeframe_signal("3h", symbol)
    signal_6h = _simple_timeframe_signal("6h", symbol)
    signal_9h = _simple_timeframe_signal("9h", symbol)

    intended = (intended_action or "").upper().strip()
    aligned = (signal_3h == intended) and (signal_6h == intended) and (signal_9h == intended)
    confidence = 90 if aligned else 50

    return ResonanceResult(
        signal_3h=signal_3h,
        signal_6h=signal_6h,
        signal_9h=signal_9h,
        confidence=confidence,
    )


@app.post("/trade")
async def execute_trade(trade_request: TradeRequest):
    """Queue a real MT5 trade via the MQL5 bridge."""
    try:
        if not app_state["trading_enabled"]:
            raise HTTPException(status_code=403, detail="Trading is currently disabled")

        if not app_state["broker_connected"]:
            raise HTTPException(status_code=503, detail="Broker is not connected")

        action = (trade_request.action or "").upper().strip()
        if action not in {"BUY", "SELL"}:
            raise HTTPException(status_code=400, detail="Invalid action; must be BUY or SELL")

        resonance = resonance_validate(trade_request.symbol, action)
        if resonance.confidence < 80:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Resonance validation failed",
                    "resonance": resonance.model_dump(),
                },
            )

        order_id = f"CMD_{int(datetime.utcnow().timestamp())}_{random.randint(1000,9999)}"
        cmd = {
            "command": "place_order",
            "id": order_id,
            "symbol": _normalize_symbol(trade_request.symbol),
            "action": action,
            "volume": float(trade_request.quantity),
            "order_type": trade_request.order_type,
            "price": trade_request.price,
            "resonance": resonance.model_dump(),
            "timestamp": time.time(),
        }
        bridge_commands.append(cmd)

        logger.info(f"Queued MT5 bridge command {order_id}: {cmd['symbol']} {action} {cmd['volume']}")
        return {
            "success": True,
            "status": "QUEUED",
            "command_id": order_id,
            "command": cmd,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade queue failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trade queue failed: {str(e)}")

@app.post("/admin/mt5-connect-xm")
async def connect_mt5_xm():
    """Connect to XM Global MT5 with user credentials"""
    try:
        logger.info("🔑 Attempting to connect to XM Global MT5...")
        
        # Try XM broker account first
        for server in XM_SERVERS:
            logger.info(f"🔍 Trying XM broker account on {server}")
            
            try:
                # Initialize MT5
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    logger.error(f"❌ MT5 initialization failed: {error_code}")
                    continue
                
                # Try XM broker login
                login_result = mt5.login(
                    login=XM_BROKER_ACCOUNT,
                    password=XM_BROKER_PASSWORD,
                    server=server
                )
                
                if login_result:
                    logger.info(f"✅ XM broker login successful on {server}")
                    
                    # Get account info
                    account_info = mt5.account_info()
                    
                    # Update app state
                    app_state["mt5_connected"] = True
                    app_state["mt5_account"] = account_info.login
                    app_state["mt5_server"] = account_info.server
                    
                    # Start demo trading
                    app_state["demo_mode"] = True
                    app_state["trade_count"] = 0
                    app_state["adaptive_learning"] = False
                    app_state["current_phase"] = "baseline"
                    
                    return {
                        "success": True,
                        "message": f"Connected to XM Global MT5 on {server}",
                        "account": account_info.login,
                        "server": account_info.server,
                        "balance": account_info.balance,
                        "equity": account_info.equity,
                        "broker": account_info.company
                    }
                else:
                    error_code = mt5.last_error()
                    logger.warning(f"⚠️ XM broker login failed for {server}: {error_code}")
                    mt5.shutdown()
                    
            except Exception as e:
                logger.error(f"❌ XM broker login exception for {server}: {e}")
                try:
                    mt5.shutdown()
                except:
                    pass
        
        # Try MQL5 account
        for server in XM_SERVERS:
            logger.info(f"🔍 Trying MQL5 account on {server}")
            
            try:
                # Initialize MT5
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    logger.error(f"❌ MT5 initialization failed: {error_code}")
                    continue
                
                # Try MQL5 login
                login_result = mt5.login(
                    login=XM_MQL5_ACCOUNT,
                    password=XM_MQL5_PASSWORD,
                    server=server
                )
                
                if login_result:
                    logger.info(f"✅ MQL5 login successful on {server}")
                    
                    # Get account info
                    account_info = mt5.account_info()
                    
                    # Update app state
                    app_state["mt5_connected"] = True
                    app_state["mt5_account"] = account_info.login
                    app_state["mt5_server"] = account_info.server
                    
                    # Start demo trading
                    app_state["demo_mode"] = True
                    app_state["trade_count"] = 0
                    app_state["adaptive_learning"] = False
                    app_state["current_phase"] = "baseline"
                    
                    return {
                        "success": True,
                        "message": f"Connected to MQL5 account on {server}",
                        "account": account_info.login,
                        "server": account_info.server,
                        "balance": account_info.balance,
                        "equity": account_info.equity,
                        "broker": account_info.company
                    }
                else:
                    error_code = mt5.last_error()
                    logger.warning(f"⚠️ MQL5 login failed for {server}: {error_code}")
                    mt5.shutdown()
                    
            except Exception as e:
                logger.error(f"❌ MQL5 login exception for {server}: {e}")
                try:
                    mt5.shutdown()
                except:
                    pass
        
        # Fallback to demo account
        logger.info("🔄 Falling back to demo account...")
        
        if not mt5.initialize():
            error_code = mt5.last_error()
            return {"success": False, "message": f"MT5 initialization failed: {error_code}"}
        
        login_result = mt5.login(
            login=MT5_DEMO_ACCOUNT,
            password=MT5_DEMO_PASSWORD,
            server=MT5_DEMO_SERVER
        )
        
        if login_result:
            logger.info("✅ Demo account login successful")
            
            # Get account info
            account_info = mt5.account_info()
            
            # Update app state
            app_state["mt5_connected"] = True
            app_state["mt5_account"] = account_info.login
            app_state["mt5_server"] = account_info.server
            
            # Start demo trading
            app_state["demo_mode"] = True
            app_state["trade_count"] = 0
            app_state["adaptive_learning"] = False
            app_state["current_phase"] = "baseline"
            
            return {
                "success": True,
                "message": f"Connected to demo account on {account_info.server}",
                "account": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "broker": account_info.company
            }
        else:
            error_code = mt5.last_error()
            return {"success": False, "message": f"All login attempts failed: {error_code}"}
        
    except Exception as e:
        logger.error(f"❌ MT5 connect exception: {e}")
        return {"success": False, "message": f"Connection failed: {str(e)}"}
    finally:
        try:
            mt5.shutdown()
        except:
            pass

@app.post("/admin/mt5-connect")
async def connect_mt5(account: int, password: str, server: str):
    """Connect to MT5 account and auto-start demo trading"""
    try:
        success = await asyncio.create_task(asyncio.to_thread(mt5_connector.login, account, password, server))
        
        if success:
            app_state["mt5_account"] = account
            app_state["mt5_server"] = server
            app_state["broker_connected"] = True
            logger.info(f"✅ Connected to MT5 account {account} on {server}")
            
            # AUTO-START DEMO TRADING AFTER MT5 CONNECTION
            logger.info("🚀 Auto-starting demo trading...")
            app_state["demo_mode"] = True
            app_state["trade_count"] = 0
            app_state["adaptive_learning"] = False
            app_state["current_phase"] = "baseline"
            
            logger.info("✅ Demo trading auto-started!")
            
            return {
                "success": True,
                "message": f"Connected to MT5 account {account} and auto-started demo trading!",
                "account": account,
                "server": server,
                "demo_mode": True,
                "current_phase": "baseline",
                "trade_count": 0
            }
        else:
            logger.error("❌ MT5 connection failed")
            return {"success": False, "message": "Failed to connect to MT5"}
            
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/admin/mt5-status")
async def get_mt5_status():
    """Get MT5 connection status and account info"""
    try:
        # Prefer MQL5 bridge connectivity over Python MT5 IPC connectivity
        if app_state.get("broker_connected"):
            last_hb = app_state.get("last_bridge_heartbeat")
            hb_age_sec = None
            if last_hb is not None:
                try:
                    hb_age_sec = time.time() - float(last_hb)
                except Exception:
                    hb_age_sec = None

            return {
                "connected": True,
                "message": "MT5 connected via MQL5 bridge",
                "bridge": True,
                "server": app_state.get("mt5_server", "Unknown"),
                "account": app_state.get("mt5_account", "Unknown"),
                "demo_mode": app_state.get("demo_mode", False),
                "trade_count": app_state.get("trade_count", 0),
                "current_phase": app_state.get("current_phase", "baseline"),
                "adaptive_learning": app_state.get("adaptive_learning", False),
                "last_bridge_heartbeat_age_sec": hb_age_sec,
            }

        if not mt5_connector.connected:
            return {
                "connected": False, 
                "message": "MT5 not connected",
                "demo_mode": app_state.get("demo_mode", False),
                "trade_count": app_state.get("trade_count", 0),
                "current_phase": app_state.get("current_phase", "baseline")
            }
        
        account_info = mt5_connector.get_account_info()
        return {
            "connected": True,
            "account_info": account_info,
            "server": app_state.get("mt5_server", "Unknown"),
            "account": app_state.get("mt5_account", "Unknown"),
            "demo_mode": app_state.get("demo_mode", False),
            "trade_count": app_state.get("trade_count", 0),
            "current_phase": app_state.get("current_phase", "baseline"),
            "adaptive_learning": app_state.get("adaptive_learning", False)
        }
    except Exception as e:
        logger.error(f"MT5 status error: {e}")
        return {"connected": False, "message": f"Error: {str(e)}"}


@app.get("/mt5/status")
async def mt5_status_public():
    """Public MT5 status for dashboard compatibility."""
    try:
        health = mt5_connector.mt5_health() if hasattr(mt5_connector, "mt5_health") else {"connected": False}
        status = mt5_connector.status() if hasattr(mt5_connector, "status") else {}

        connection_status = status.get("connection_status") or health.get("connection_status") or ("CONNECTED" if health.get("connected") else "DISCONNECTED")

        return {
            "connection_status": connection_status,
            "account_login": health.get("account"),
            "account_balance": health.get("balance"),
            "connected": bool(health.get("connected")),
        }
    except Exception as e:
        logger.error(f"/mt5/status error: {e}")
        return {
            "connection_status": "IPC_ERROR",
            "account_login": None,
            "account_balance": None,
            "connected": False,
        }

@app.post("/admin/demo/start")
async def start_demo_trading():
    """Start demo trading with 500 trade learning plan"""
    try:
        app_state["demo_mode"] = True
        app_state["trade_count"] = 0
        app_state["current_phase"] = "baseline"
        app_state["adaptive_learning"] = False
        
        logger.info("🚀 Demo trading started - 500 trade learning plan")
        return {
            "success": True,
            "message": "Demo trading started",
            "plan": {
                "phase_1": "100 trades - Baseline establishment",
                "phase_2": "200 trades - Adaptive learning", 
                "phase_3": "200 trades - Full optimization",
                "total": "500 trades"
            },
            "current_phase": "baseline",
            "features": ["Safe risk management", "Performance tracking"]
        }
    except Exception as e:
        logger.error(f"Failed to start demo trading: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/admin/demo/progress")
async def get_demo_progress():
    """Get demo trading progress and learning status"""
    try:
        trade_count = app_state.get("trade_count", 0)
        current_phase = app_state.get("current_phase", "baseline")
        
        # Determine phase based on trade count
        if trade_count <= 100:
            phase_name = "baseline"
            phase_progress = trade_count / 100
        elif trade_count <= 300:
            phase_name = "learning"  
            phase_progress = (trade_count - 100) / 200
        else:
            phase_name = "optimization"
            phase_progress = (trade_count - 300) / 200
            
        return {
            "trade_count": trade_count,
            "current_phase": current_phase,
            "phase_name": phase_name,
            "phase_progress": min(phase_progress, 1.0),
            "adaptive_learning": app_state.get("adaptive_learning", False),
            "demo_mode": app_state.get("demo_mode", False),
            "next_phase_features": get_next_phase_features(phase_name)
        }
    except Exception as e:
        logger.error(f"Failed to get demo progress: {e}")
        return {"error": str(e)}

def get_next_phase_features(current_phase: str) -> List[str]:
    """Get features for next phase"""
    phase_features = {
        "baseline": ["Adaptive learning", "Behavioral analysis", "Strategy optimization"],
        "learning": ["Outlier detection", "Dynamic position sizing", "Confidence scoring"],
        "optimization": ["Advanced risk management", "Performance adaptation", "Full automation"]
    }
    return phase_features.get(current_phase, [])

@app.get("/admin/mt5-universal-status")
async def get_universal_mt5_status():
    """Get universal MT5 connector status"""
    try:
        if not universal_connector.connected:
            return {"connected": False, "message": "Universal MT5 not connected"}
        
        account_summary = universal_connector.get_account_summary()
        return {
            "connected": True,
            "connector_type": "Universal",
            "account_summary": account_summary,
            "available_symbols": universal_connector.get_available_symbols(),
            "open_positions": universal_connector.get_positions(),
            "demo_mode": app_state.get("demo_mode", False),
            "trade_count": app_state.get("trade_count", 0),
            "current_phase": app_state.get("current_phase", "baseline")
        }
    except Exception as e:
        logger.error(f"Universal MT5 status error: {e}")
        return {"connected": False, "message": f"Error: {str(e)}"}

@app.post("/admin/mt5-universal-connect")
async def connect_universal_mt5():
    """Connect to universal MT5 (any broker)"""
    try:
        from universal_mt5_connector import sync_auto_connect_to_mt5
        success = sync_auto_connect_to_mt5()
        
        if success:
            app_state["broker_connected"] = True
            logger.info("✅ Universal MT5 connection successful")
            return {
                "success": True,
                "message": "Connected to MT5 terminal",
                "connector_type": "Universal",
                "account_summary": universal_connector.get_account_summary()
            }
        else:
            logger.error("❌ Universal MT5 connection failed")
            return {
                "success": False,
                "message": "Failed to connect to MT5 terminal",
                "connector_type": "Universal"
            }
    except Exception as e:
        logger.error(f"Universal MT5 connection error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/admin/mt5-universal-order")
async def place_universal_mt5_order(
    symbol: str,
    volume: float,
    order_type: str,
    price: float = 0.0
):
    """Place order using universal MT5 connector"""
    try:
        if not universal_connector.connected:
            raise HTTPException(status_code=400, detail="Universal MT5 not connected")
        
        order_id = universal_connector.place_market_order(symbol, volume, order_type, price)
        
        if order_id:
            app_state["trade_count"] = app_state.get("trade_count", 0) + 1
            logger.info(f"✅ Universal MT5 order placed: {order_id}")
            return {
                "success": True,
                "message": "Order placed successfully",
                "order_id": order_id,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price
            }
        else:
            logger.error("❌ Universal MT5 order failed")
            return {"success": False, "message": "Failed to place order"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Universal MT5 order error: {e}")
        raise HTTPException(status_code=500, detail=f"Order placement error: {str(e)}")

@app.get("/admin/mt5-universal-positions")
async def get_universal_mt5_positions():
    """Get all positions from universal MT5 connector"""
    try:
        if not universal_connector.connected:
            raise HTTPException(status_code=400, detail="Universal MT5 not connected")
        
        positions = universal_connector.get_positions()
        return {
            "success": True,
            "positions": positions,
            "count": len(positions),
            "connector_type": "Universal"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Universal MT5 positions error: {e}")
        raise HTTPException(status_code=500, detail=f"Positions error: {str(e)}")

@app.post("/admin/enable-trading")
async def enable_trading():
    """Enable trading"""
    app_state["trading_enabled"] = True
    logger.info("Trading enabled by admin")
    return {"success": True, "message": "Trading enabled", "timestamp": datetime.utcnow().isoformat()}

@app.post("/admin/disable-trading")
async def disable_trading():
    """Disable trading"""
    app_state["trading_enabled"] = False
    logger.warning("Trading disabled by admin")
    return {"success": True, "message": "Trading disabled", "timestamp": datetime.utcnow().isoformat()}

@app.post("/admin/simulate-broker-failure")
async def simulate_broker_failure():
    """Simulate broker connection failure"""
    app_state["broker_connected"] = False
    logger.critical("Broker connection failure simulated")
    return {"success": True, "message": "Broker failure simulated", "timestamp": datetime.utcnow().isoformat()}

@app.post("/admin/simulate-broker-recovery")
async def simulate_broker_recovery():
    """Simulate broker connection recovery"""
    app_state["broker_connected"] = True
    logger.info("Broker connection recovery simulated")
    return {"success": True, "message": "Broker recovery simulated", "timestamp": datetime.utcnow().isoformat()}

@app.post("/admin/start-demo")
async def start_immediate_demo():
    """Start immediate demo trading"""
    global app_state
    try:
        app_state["demo_trading"] = True
        app_state["demo_trades_executed"] = 0
        app_state["demo_start_time"] = datetime.now().isoformat()
        
        logger.info("🚀 Immediate demo trading STARTED")
        return {"success": True, "message": "Demo trading started"}
    except Exception as e:
        logger.error(f"Demo trading start error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/admin/system-status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "trading_enabled": app_state["trading_enabled"],
        "broker_connected": app_state["broker_connected"],
        "db_connected": app_state["db_connected"],
        "uptime": str(datetime.utcnow() - app_state["start_time"]).split('.')[0],
        "start_time": app_state["start_time"].isoformat(),
        "supported_symbols": ["EUR/USD", "BTC/USD"],
        "supported_timeframes": ["1h", "5m", "9h"],
        "9h_candles_available": {symbol: len(candles) for symbol, candles in mock_9h_candles.items()}
    }

# Bridge endpoints
class BridgeCommand(BaseModel):
    command: str
    params: Dict[str, Any] = {}
    timestamp: Optional[float] = None

# Configure Pydantic
class Config:
    json_encoders = {
        'dict': lambda x: x
    }

# Store bridge commands and responses
bridge_commands = deque()
bridge_responses = []

@app.get("/mt5/bridge/commands")
async def get_bridge_commands():
    """Get ONE pending command for MQL5 bridge (FIFO queue)."""
    global bridge_commands
    if not bridge_commands:
        return {"command": None}
    cmd = bridge_commands.popleft()
    return cmd

@app.post("/mt5/bridge/data")
async def receive_bridge_data(data: Dict[str, Any]):
    """Receive data from MQL5 bridge"""
    global bridge_responses
    try:
        # Update connection state from bridge heartbeats
        if isinstance(data, dict) and data.get("command") == "account_info":
            app_state["broker_connected"] = True
            app_state["mt5_account"] = data.get("login")
            app_state["mt5_server"] = data.get("server")
            app_state["last_bridge_heartbeat"] = time.time()

        # Record order execution results from the EA
        if isinstance(data, dict) and data.get("command") == "order_result":
            app_state["last_order_result"] = data
            app_state["trade_count"] = app_state.get("trade_count", 0) + 1
            logger.info(f"✅ order_result received: {json.dumps(data, ensure_ascii=False)}")

        logger.info(f"[BRIDGE_DATA] {json.dumps(data, ensure_ascii=False)}")
        bridge_responses.append(data)
        return {"success": True}
    except Exception as e:
        logger.error(f"Bridge data error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/mt5/bridge/responses")
async def get_bridge_responses():
    """Get bridge responses"""
    global bridge_responses
    responses = bridge_responses.copy()
    bridge_responses.clear()
    return {"responses": responses}

@app.post("/mt5/bridge/command")
async def send_bridge_command(command: BridgeCommand):
    """Send command to bridge"""
    global bridge_commands
    if command.timestamp is None:
        command.timestamp = time.time()
    bridge_commands.append(command.dict())
    return {"success": True, "message": "Command queued"}

@app.post("/mt5/bridge/connect")
async def connect_bridge():
    """Connect to MT5 via bridge"""
    try:
        success = bridge_connector.initialize()
        if success:
            app_state["broker_connected"] = True
            account_info = bridge_connector.get_account_info()
            if account_info:
                app_state["mt5_account"] = account_info.get("login")
                app_state["mt5_server"] = account_info.get("server")
            return {
                "success": True,
                "message": "Bridge connected successfully",
                "account_info": account_info
            }
        else:
            return {"success": False, "message": "Failed to connect via bridge"}
    except Exception as e:
        logger.error(f"Bridge connection error: {e}")
        return {"success": False, "message": str(e)}

@app.get("/mt5/bridge/status")
async def get_bridge_status():
    """Get bridge connection status"""
    return {
        "connected": bridge_connector.connected,
        "account_info": bridge_connector.account_info,
        "broker_connected": app_state["broker_connected"]
    }

if __name__ == "__main__":
    # Run the application
    logger.info("🚀 Starting Nexus Trading System API...")
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
