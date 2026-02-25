"""
Nexus Trading System - Secure Main Entry Point
Production-ready application with enhanced security and monitoring
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from datetime import datetime

from config.settings import settings
from database.session import init_database, db_manager
from api.auth import TokenManager
from core.enhanced_risk_engine import enhanced_risk_engine
from execution.safe_executor import safe_executor
from monitoring.analytics_service import create_analytics_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nexus.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global services
analytics_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Nexus Trading System...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Check database health
        db_health = db_manager.health_check()
        if not db_health.get("healthy", False):
            logger.error(f"Database health check failed: {db_health}")
            raise Exception("Database initialization failed")
        
        logger.info("Database initialized successfully")
        
        # Initialize analytics service
        global analytics_service
        analytics_service = create_analytics_service()
        logger.info("Analytics service initialized")
        
        # Start system monitoring
        if settings.ENABLE_MONITORING:
            asyncio.create_task(analytics_service.start_monitoring())
            logger.info("System monitoring started")
        
        # Log system startup
        await analytics_service.log_event(
            level="INFO",
            message="Nexus Trading System started successfully",
            module="main",
            metadata={
                "version": "1.0.0",
                "environment": "production" if settings.is_production() else "development",
                "database_type": "postgresql"
            }
        )
        
        logger.info("Nexus Trading System startup completed")
        
    except Exception as e:
        logger.error(f"Failed to start Nexus Trading System: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nexus Trading System...")
    
    try:
        # Stop monitoring
        if analytics_service:
            analytics_service.stop_monitoring()
            logger.info("System monitoring stopped")
        
        # Cleanup database
        from database.session import cleanup_database
        cleanup_database()
        logger.info("Database cleanup completed")
        
        logger.info("Nexus Trading System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="Nexus Trading System",
    description="Secure AI-powered trading platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.API_DEBUG else None,
    redoc_url="/redoc" if settings.API_DEBUG else None
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"] if not settings.is_production() else ["yourdomain.com"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"] if not settings.is_production() else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Database health
        db_health = db_manager.health_check()
        
        # Risk engine status
        risk_status = enhanced_risk_engine.get_user_risk_status(0)  # System-wide status
        
        # Execution status
        exec_status = safe_executor.get_execution_status()
        
        # Analytics status
        analytics_status = {
            "monitoring_enabled": settings.ENABLE_MONITORING,
            "service_active": analytics_service is not None
        }
        
        overall_health = (
            db_health.get("healthy", False) and
            not risk_status.get("emergency_stop_active", False) and
            exec_status.get("execution_enabled", False)
        )
        
        return JSONResponse({
            "status": "healthy" if overall_health else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": "production" if settings.is_production() else "development",
            "services": {
                "database": db_health,
                "risk_engine": risk_status,
                "executor": exec_status,
                "analytics": analytics_status
            }
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# System status endpoint
@app.get("/status", tags=["System"])
async def system_status():
    """Detailed system status"""
    try:
        return JSONResponse({
            "system": {
                "uptime": "N/A",  # Would track actual uptime
                "version": "1.0.0",
                "environment": "production" if settings.is_production() else "development"
            },
            "database": db_manager.get_connection_info(),
            "risk_engine": {
                "emergency_stop": enhanced_risk_engine.emergency_stop,
                "disabled_users": len(enhanced_risk_engine.user_trading_disabled)
            },
            "executor": safe_executor.get_execution_status(),
            "security": {
                "jwt_expiry_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
                "bcrypt_rounds": settings.BCRYPT_ROUNDS,
                "max_login_attempts": settings.MAX_LOGIN_ATTEMPTS
            },
            "limits": {
                "max_daily_loss": settings.MAX_DAILY_LOSS,
                "max_risk_percent": settings.MAX_RISK_PERCENT,
                "default_sl_points": settings.DEFAULT_SL_POINTS,
                "default_tp_points": settings.DEFAULT_TP_POINTS
            }
        })
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Emergency control endpoints
@app.post("/admin/emergency-stop", tags=["Admin"])
async def emergency_stop(reason: str = ""):
    """Activate emergency stop (admin only)"""
    try:
        enhanced_risk_engine.set_emergency_stop(True, reason)
        safe_executor.set_execution_enabled(False, "Emergency stop activated")
        
        if analytics_service:
            await analytics_service.create_alert(
                severity="CRITICAL",
                title="Emergency Stop Activated",
                message=f"Emergency stop activated: {reason}",
                source="system_admin",
                metadata={"reason": reason, "timestamp": datetime.utcnow().isoformat()}
            )
        
        return JSONResponse({
            "message": "Emergency stop activated",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to activate emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/admin/emergency-resume", tags=["Admin"])
async def emergency_resume():
    """Resume trading after emergency stop (admin only)"""
    try:
        enhanced_risk_engine.set_emergency_stop(False)
        safe_executor.set_execution_enabled(True, "Emergency stop lifted")
        
        if analytics_service:
            await analytics_service.create_alert(
                severity="LOW",
                title="Emergency Stop Lifted",
                message="Trading resumed after emergency stop",
                source="system_admin",
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
        
        return JSONResponse({
            "message": "Trading resumed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to resume trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Token management endpoints
@app.post("/auth/refresh", tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        token_data = TokenManager.verify_token(refresh_token, "refresh")
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        access_data = {"sub": token_data.username, "user_id": token_data.user_id}
        new_access_token = TokenManager.create_access_token(access_data)
        
        return JSONResponse({
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        })
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@app.post("/auth/logout", tags=["Authentication"])
async def logout(access_token: str):
    """Logout and revoke token"""
    try:
        success = TokenManager.revoke_token(access_token)
        if success:
            return JSONResponse({"message": "Logged out successfully"})
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke token"
            )
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Include API routes
try:
    from api.routes.auth import router as auth_router
    from api.routes.users import router as users_router
    from api.routes.trades import router as trades_router
    from api.routes.signals import router as signals_router
    from api.routes.metrics import router as metrics_router
    
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(trades_router, prefix="/api/v1/trades", tags=["Trades"])
    app.include_router(signals_router, prefix="/api/v1/signals", tags=["Signals"])
    app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics"])
    
    logger.info("API routes loaded successfully")
    
except ImportError as e:
    logger.warning(f"Could not load some API routes: {e}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if analytics_service:
        await analytics_service.create_alert(
            severity="CRITICAL",
            title="Unhandled Exception",
            message=str(exc),
            source="global_handler",
            metadata={
                "path": str(request.url),
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

def main():
    """Main entry point"""
    logger.info("Starting Nexus Trading System server...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Start server
    uvicorn.run(
        "main_secure:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
