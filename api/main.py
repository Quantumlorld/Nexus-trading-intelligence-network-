"""
Nexus Trading System - FastAPI Main Application
Main API server with authentication, routes, and middleware
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uvicorn
import logging
from datetime import datetime
import os
from typing import Dict, Any

# Import database and authentication
from database.session import init_database, get_database_session
from database.models import User, UserRole, UserStatus
from api.auth import get_current_user, get_current_active_user, UserManager, TokenManager
from api.dependencies import log_request

# Import routes
from api.routes.users import router as users_router
from api.routes.signals import router as signals_router
from api.routes.trades import router as trades_router
from api.routes.metrics import router as metrics_router
from api.routes.adaptive import router as adaptive_router
from api.routes.auth import router as auth_router

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Nexus Trading System API",
    description="Professional trading system with multi-user support and adaptive learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Nexus Trading Team",
        "email": "support@nexustrading.com"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"API Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"API Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
app.include_router(signals_router, prefix="/api/v1/signals", tags=["Signals"])
app.include_router(trades_router, prefix="/api/v1/trades", tags=["Trades"])
app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics"])
app.include_router(adaptive_router, prefix="/api/v1/adaptive", tags=["Adaptive Learning"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Nexus Trading System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        from database.session import db_manager
        db_health = db_manager.health_check()
        
        return {
            "status": "healthy" if db_health else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected" if db_health else "disconnected",
            "version": "1.0.0",
            "uptime": "0s"  # TODO: Implement uptime tracking
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

# System status endpoint
@app.get("/api/v1/status")
async def get_system_status(current_user: User = Depends(get_current_active_user)):
    """Get system status"""
    try:
        with get_database_session() as db:
            # Get system statistics
            total_users = db.query(User).count()
            active_users = db.query(User).filter(
                User.status.in_([UserStatus.ACTIVE, UserStatus.TRIAL])
            ).count()
            
            # Get user statistics
            user_stats = get_user_statistics(db)
            
            return {
                "system": {
                    "status": "running",
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "users": {
                    "total": total_users,
                    "active": active_users,
                    "online": active_users  # TODO: Implement online tracking
                },
                "trading": {
                    "total_trades": user_stats["total_trades"],
                    "active_positions": user_stats["active_positions"],
                    "total_pnl": user_stats["total_pnl"]
                },
                "performance": {
                    "avg_win_rate": user_stats["avg_win_rate"],
                    "best_performer": user_stats["best_performer"],
                    "worst_performer": user_stats["worst_performer"]
                }
            }
    
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )

# User profile endpoint
@app.get("/api/v1/me")
async def get_current_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Get current user profile"""
    try:
        # Log request
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
        
        with get_database_session() as db:
            # Refresh user data
            db.refresh(current_user)
            
            # Get user statistics
            stats = get_user_statistics(db, current_user.id)
            
            # Get subscription info
            subscription = check_user_subscription(current_user)
            
            return {
                "user": {
                    "id": current_user.id,
                    "uuid": current_user.uuid,
                    "username": current_user.username,
                    "email": current_user.email,
                    "full_name": current_user.full_name,
                    "role": current_user.role.value,
                    "status": current_user.status.value,
                    "created_at": current_user.created_at.isoformat(),
                    "last_login": current_user.last_login.isoformat() if current_user.last_login else None
                },
                "subscription": subscription,
                "statistics": stats,
                "permissions": {
                    "can_trade": UserPermissions.can_trade(current_user),
                    "can_auto_trade": UserPermissions.can_auto_trade(current_user),
                    "allowed_assets": current_user.allowed_assets or []
                }
            }
    
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )

# Utility functions
def get_user_statistics(db: Session, user_id: int = None) -> Dict[str, Any]:
    """Get user statistics"""
    # TODO: Implement comprehensive user statistics
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "active_positions": 0,
        "avg_win_rate": 0.0,
        "best_performer": None,
        "worst_performer": None
    }

def check_user_subscription(user: User) -> Dict[str, Any]:
    """Check user subscription status"""
    return {
        "status": user.subscription_status.value,
        "plan": user.subscription_plan,
        "expires_at": user.subscription_end.isoformat() if user.subscription_end else None,
        "auto_trade_enabled": user.auto_trade_enabled
    }

# User permissions (moved from dependencies for circular import)
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Nexus Trading System API...")
    
    # Initialize database
    if not init_database():
        logger.error("Failed to initialize database")
        raise Exception("Database initialization failed")
    
    logger.info("Nexus Trading System API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Nexus Trading System API...")
    
    # Cleanup database connections
    from database.session import cleanup_database
    cleanup_database()
    
    logger.info("Nexus Trading System API shutdown complete")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
