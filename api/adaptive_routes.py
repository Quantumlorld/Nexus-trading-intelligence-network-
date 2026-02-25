"""
Adaptive AI API Routes
Integration with adaptive learning system
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from database.session import get_database_session
from adaptive import (
    AdaptiveEngine, 
    UserPerformanceTracker, 
    OutlierDetector, 
    StrategyWeightManager,
    create_adaptive_layer,
    create_performance_tracker,
    create_outlier_detector,
    create_strategy_weight_manager
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/adaptive", tags=["Adaptive AI"])

# Initialize adaptive components
adaptive_engine = create_adaptive_layer()
performance_tracker = create_performance_tracker()
outlier_detector = create_outlier_detector()
weight_manager = create_strategy_weight_manager()

@router.get("/status")
async def get_adaptive_status():
    """Get the status of the adaptive AI system"""
    return {
        "status": "active",
        "components": {
            "adaptive_engine": "running",
            "performance_tracker": "running", 
            "outlier_detector": "running",
            "weight_manager": "running"
        },
        "last_updated": datetime.utcnow().isoformat(),
        "models_loaded": 4
    }

@router.get("/signals")
async def get_adaptive_signals():
    """Get AI-powered trading signals"""
    try:
        # Generate signals using adaptive engine
        signals = adaptive_engine.generate_signals()
        
        return {
            "status": "success",
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": "high"
        }
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return {
            "status": "error",
            "message": str(e),
            "signals": [],
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics from the adaptive system"""
    try:
        metrics = performance_tracker.get_current_metrics()
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/outliers")
async def get_outlier_analysis():
    """Get outlier detection results"""
    try:
        outliers = outlier_detector.detect_outliers()
        
        return {
            "status": "success",
            "outliers": outliers,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        return {
            "status": "error",
            "message": str(e),
            "outliers": [],
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/weights")
async def get_strategy_weights():
    """Get current strategy weights"""
    try:
        weights = weight_manager.get_current_weights()
        
        return {
            "status": "success",
            "weights": weights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting strategy weights: {e}")
        return {
            "status": "error",
            "message": str(e),
            "weights": {},
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/train")
async def train_adaptive_model():
    """Train the adaptive model with new data"""
    try:
        # Start training process
        training_result = adaptive_engine.train_model()
        
        return {
            "status": "success",
            "training_result": training_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/analytics")
async def get_advanced_analytics():
    """Get advanced analytics from the adaptive system"""
    try:
        analytics = {
            "performance_trends": performance_tracker.get_performance_trends(),
            "risk_analysis": performance_tracker.get_risk_analysis(),
            "strategy_effectiveness": weight_manager.get_strategy_effectiveness(),
            "market_regime": adaptive_engine.detect_market_regime(),
            "prediction_accuracy": adaptive_engine.get_prediction_accuracy()
        }
        
        return {
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return {
            "status": "error",
            "message": str(e),
            "analytics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
