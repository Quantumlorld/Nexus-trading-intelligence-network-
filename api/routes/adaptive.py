"""
Nexus Trading System - Adaptive Layer API Routes
API endpoints for adaptive learning and performance tracking
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio

from database.session import get_database_session
from database.models import User, Trade, UserPerformance, AdaptiveWeight, Signal
from database.schemas import PaginatedResponse, PaginationParams
from api.auth import get_current_user, get_current_active_user, get_current_admin_user
from api.dependencies import UserPermissions, RateLimitChecker, AuditLogger

# Import adaptive components
from adaptive import (
    UserPerformanceTracker, PerformanceCategory, RiskProfile,
    OutlierDetector, OutlierType, OutlierSeverity,
    StrategyWeightManager, WeightUpdateMethod,
    create_performance_tracker, create_outlier_detector, create_strategy_weight_manager
)

router = APIRouter(tags=["Adaptive Learning"])
logger = logging.getLogger(__name__)

# Global adaptive components
performance_tracker = create_performance_tracker()
outlier_detector = create_outlier_detector()
weight_manager = create_strategy_weight_manager()

# Rate limiter
rate_limiter = RateLimitChecker()


@router.get("/performance/overview")
async def get_performance_overview(
    current_user: User = Depends(get_current_active_user)
):
    """Get performance overview for current user"""
    try:
        # Track user performance
        profile = performance_tracker.track_user_performance(current_user.id)
        
        if not profile:
            return {
                "message": "Insufficient trading data for analysis",
                "min_trades_required": performance_tracker.min_trades_for_analysis,
                "current_trades": 0
            }
        
        return {
            "user_profile": profile.to_dict(),
            "performance_category": profile.performance_category.value,
            "risk_profile": profile.risk_profile.value,
            "recommendations": performance_tracker.get_adaptive_recommendations(current_user.id)
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance overview"
        )


@router.get("/performance/history")
async def get_performance_history(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
):
    """Get performance history for current user"""
    try:
        history = performance_tracker.get_performance_history(current_user.id)
        
        # Filter by days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filtered_history = [
            entry for entry in history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        return {
            "history": filtered_history,
            "period_days": days,
            "total_entries": len(filtered_history)
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance history"
        )


@router.get("/performance/learning-progress")
async def get_learning_progress(
    current_user: User = Depends(get_current_active_user)
):
    """Get learning progress for current user"""
    try:
        progress = performance_tracker.get_user_learning_progress(current_user.id)
        
        return {
            "learning_progress": progress,
            "improvement_trends": {
                "performance_improvement": progress.get('performance_improvement', 0),
                "win_rate_improvement": progress.get('win_rate_improvement', 0),
                "profit_factor_improvement": progress.get('profit_factor_improvement', 0),
                "consistency_score": progress.get('consistency_score', 0),
                "learning_rate": progress.get('learning_rate', 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get learning progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning progress"
        )


@router.get("/performance/behavioral-patterns")
async def get_behavioral_patterns(
    current_user: User = Depends(get_current_active_user)
):
    """Get behavioral patterns for current user"""
    try:
        patterns = performance_tracker.get_user_behavioral_patterns(current_user.id)
        
        return {
            "behavioral_patterns": patterns,
            "analysis": {
                "trading_frequency": patterns.get('trading_frequency', 0),
                "avg_holding_time": patterns.get('avg_holding_time', 0),
                "risk_tolerance": patterns.get('avg_position_size', 0),
                "discipline": {
                    "sl_hit_rate": patterns.get('sl_hit_rate', 0),
                    "tp_hit_rate": patterns.get('tp_hit_rate', 0),
                    "manual_exit_rate": patterns.get('manual_exit_rate', 0)
                },
                "session_preferences": patterns.get('session_preferences', {}),
                "confidence_level": patterns.get('avg_confidence', 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get behavioral patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get behavioral patterns"
        )


@router.get("/outliers/detect")
async def detect_outliers(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin_user)
):
    """Detect outliers in user performance (admin only)"""
    try:
        # Check rate limit
        if not rate_limiter.check_rate_limit(current_user.id, "detect_outliers"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many outlier detection requests"
            )
        
        # Get all user profiles
        user_profiles = {}
        with get_database_session() as db:
            users = db.query(User).all()
            
            for user in users:
                profile = performance_tracker.track_user_performance(user.id)
                if profile:
                    user_profiles[user.id] = profile
        
        # Detect outliers
        outliers = outlier_detector.detect_outliers(user_profiles)
        
        # Update active outliers
        outlier_detector.active_outliers = {
            outlier.user_id: outlier for outlier in outliers
        }
        
        # Log outlier detection
        AuditLogger.log_user_action(
            user=current_user,
            action="outlier_detection",
            details={
                "total_users": len(user_profiles),
                "outliers_detected": len(outliers),
                "outlier_types": [o.outlier_type.value for o in outliers]
            }
        )
        
        return {
            "outliers": [outlier.to_dict() for outlier in outliers],
            "summary": outlier_detector.get_outlier_summary(),
            "detection_timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to detect outliers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect outliers"
        )


@router.get("/outliers/list")
async def list_outliers(
    pagination: PaginationParams = Depends(),
    severity: Optional[str] = Query(None),
    outlier_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user)
):
    """List detected outliers (admin only)"""
    try:
        outliers = list(outlier_detector.active_outliers.values())
        
        # Apply filters
        if severity:
            outliers = [o for o in outliers if o.severity.value == severity]
        
        if outlier_type:
            outliers = [o for o in outliers if o.outlier_type.value == outlier_type]
        
        # Sort by detection time
        outliers.sort(key=lambda x: x.detected_at, reverse=True)
        
        # Apply pagination
        total = len(outliers)
        start = (pagination.page - 1) * pagination.size
        end = start + pagination.size
        paginated_outliers = outliers[start:end]
        
        return PaginatedResponse.create(
            items=[outlier.to_dict() for outlier in paginated_outliers],
            total=total,
            page=pagination.page,
            size=pagination.size
        )
    
    except Exception as e:
        logger.error(f"Failed to list outliers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list outliers"
        )


@router.get("/outliers/{user_id}")
async def get_user_outlier_details(
    user_id: int,
    current_user: User = Depends(get_current_admin_user)
):
    """Get outlier details for a specific user (admin only)"""
    try:
        outlier = outlier_detector.active_outliers.get(user_id)
        
        if not outlier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Outlier not found for this user"
            )
        
        # Get additional user data
        profile = performance_tracker.get_user_profile(user_id)
        history = outlier_detector.get_user_outlier_history(user_id)
        recommendations = outlier_detector.get_outlier_recommendations(user_id)
        
        return {
            "outlier": outlier.to_dict(),
            "user_profile": profile.to_dict() if profile else None,
            "detection_history": [o.to_dict() for o in history],
            "recommendations": recommendations
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get outlier details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get outlier details"
        )


@router.post("/outliers/{user_id}/remove")
async def remove_outlier(
    user_id: int,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Remove user from outlier list (admin only)"""
    try:
        success = outlier_detector.remove_outlier(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in outlier list"
            )
        
        # Log removal
        AuditLogger.log_user_action(
            user=current_user,
            action="outlier_removed",
            details={
                "target_user_id": user_id,
                "reason": reason
            }
        )
        
        return {
            "message": f"User {user_id} removed from outlier list",
            "removed_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove outlier: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove outlier"
        )


@router.get("/weights/current")
async def get_current_strategy_weights(
    current_user: User = Depends(get_current_active_user)
):
    """Get current strategy weights"""
    try:
        weights = weight_manager.get_strategy_weights()
        
        return {
            "strategy_weights": weights,
            "summary": weight_manager.get_performance_summary(),
            "last_updated": weight_manager.last_update.isoformat(),
            "update_method": weight_manager.update_method.value
        }
    
    except Exception as e:
        logger.error(f"Failed to get strategy weights: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get strategy weights"
        )


@router.get("/weights/history")
async def get_weight_history(
    strategy: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user)
):
    """Get weight history for strategies"""
    try:
        if strategy:
            history = weight_manager.get_weight_history(strategy, limit)
            return {
                "strategy": strategy,
                "history": [w.to_dict() for w in history],
                "limit": limit
            }
        else:
            all_history = {}
            for strategy_name in weight_manager.strategy_weights.keys():
                all_history[strategy_name] = weight_manager.get_weight_history(strategy_name, limit)
            
            return {
                "all_history": all_history,
                "limit": limit
            }
    
    except Exception as e:
        logger.error(f"Failed to get weight history: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weight history"
        )


@router.get("/weights/ranking")
async def get_strategy_ranking(
    current_user: User = Depends(get_current_active_user)
):
    """Get strategy ranking by weight"""
    try:
        ranking = weight_manager.get_strategy_ranking()
        top_performers = weight_manager.get_top_strategies()
        
        return {
            "current_ranking": ranking,
            "top_performers": top_performers,
            "total_strategies": len(ranking)
        }
    
    except Exception as e:
        logger.error(f"Failed to get strategy ranking: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get strategy ranking"
        )


@router.post("/weights/update")
async def update_strategy_weights(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin_user)
):
    """Update strategy weights based on user performance (admin only)"""
    try:
        # Check rate limit
        if not rate_limiter.check_rate_limit(current_user.id, "update_weights"):
            raise HTTPException(
                status=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many weight update requests"
            )
        
        # Get all user profiles
        user_profiles = {}
        with get_database_session() as db:
            users = db.query(User).all()
            
            for user in users:
                profile = performance_tracker.track_user_performance(user.id)
                if profile:
                    user_profiles[user.id] = profile
        
        # Update weights
        updated_weights = weight_manager.update_strategy_weights(user_profiles, outlier_detector)
        
        # Log weight update
        AuditLogger.log_user_action(
            user=current_user,
            action="strategy_weights_updated",
            details={
                "total_users": len(user_profiles),
                "updated_strategies": len(updated_weights),
                "outlier_users_removed": len(user_profiles) - len([p for p in user_profiles.values() if not outlier_detector.is_user_outlier(p.user_id)])
            }
        )
        
        return {
            "updated_weights": {
                name: weight.to_dict() for name, weight in updated_weights.items()
            },
            "summary": weight_manager.get_performance_summary(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to update strategy weights: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strategy weights"
        )


@router.get("/performance/summary")
async def get_performance_summary(
    current_user: User = Depends(get_current_admin_user)
):
    """Get performance summary for all users (admin only)"""
    try:
        summary = performance_tracker.get_performance_summary()
        
        return {
            "summary": summary,
            "outlier_summary": outlier_detector.get_outlier_summary(),
            "weight_summary": weight_manager.get_performance_summary(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance summary"
        )


@router.get("/performance/categories")
async def get_performance_categories(
    current_user: User = Depends(get_current_active_user)
):
    """Get users by performance categories"""
    try:
        categories = {}
        
        for category in PerformanceCategory:
            users = performance_tracker.get_users_by_category(category)
            categories[category.value] = [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "performance_score": user.metrics.performance_score,
                    "win_rate": user.metrics.win_rate,
                    "total_trades": user.metrics.total_trades
                }
                for user in users
            ]
        
        return {
            "categories": categories,
            "total_categories": len(categories),
            "user_distribution": {
                category.value: len(users) for category, users in categories.items()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance categories: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance categories"
        )


@router.get("/performance/risk-profiles")
async def get_risk_profiles(
    current_user: User = Depends(get_current_active_user)
):
    """Get users by risk profiles"""
    try:
        profiles = {}
        
        for risk_profile in RiskProfile:
            users = performance_tracker.get_users_by_risk_profile(risk_profile)
            profiles[risk_profile.value] = [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "risk_reward_ratio": user.metrics.risk_reward_ratio,
                    "max_drawdown": user.metrics.max_drawdown,
                    "avg_position_size": user.behavioral_patterns.get('avg_position_size', 0)
                }
                for user in users
            ]
        
        return {
            "risk_profiles": profiles,
            "total_profiles": len(profiles),
            "user_distribution": {
                profile.value: len(users) for profile, users in profiles.items()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get risk profiles: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get risk profiles"
        )


@router.get("/performance/top-performers")
async def get_top_performers(
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """Get top performing users"""
    try:
        top_performers = performance_tracker.get_top_performers(limit)
        
        return {
            "top_performers": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "performance_score": user.metrics.performance_score,
                    "win_rate": user.metrics.win_rate,
                    "profit_factor": user.metrics.profit_factor,
                    "sharpe_ratio": user.metrics.sharpe_ratio,
                    "total_trades": user.metrics.total_trades,
                    "performance_category": user.performance_category.value,
                    "risk_profile": user.risk_profile.value,
                    "influence_score": user.influence_score
                }
                for user in top_performers
            ],
            "limit": limit,
            "total_analyzed": len(performance_tracker.user_profiles)
        }
    
    except Exception as e:
        logger.error(f"Failed to get top performers: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get top performers"
        )


@router.get("/performance/influence-ranking")
async def get_influence_ranking(
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """Get users by influence score"""
    try:
        influence_ranking = performance_tracker.get_influence_ranking(limit)
        
        return {
            "influence_ranking": [
                {
                    "user_id": user_id,
                    "influence_score": score,
                    "profile": performance_tracker.get_user_profile(user_id).to_dict() if performance_tracker.get_user_profile(user_id) else None
                }
                for user_id, score in influence_ranking
            ],
            "limit": limit,
            "total_analyzed": len(performance_tracker.influence_scores)
        }
    
    except Exception as e:
        logger.error(f"Failed to get influence ranking: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get influence ranking"
        )


@router.get("/adaptive/recommendations")
async def get_adaptive_recommendations(
    current_user: User = Depends(get_current_active_user)
):
    """Get adaptive recommendations for current user"""
    try:
        recommendations = performance_tracker.get_adaptive_recommendations(current_user.id)
        
        return {
            "recommendations": recommendations,
            "user_profile": performance_tracker.get_user_profile(current_user.id).to_dict() if performance_tracker.get_user_profile(current_user.id) else None,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get adaptive recommendations: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get adaptive recommendations"
        )


@router.get("/adaptive/statistics")
async def get_adaptive_statistics(
    current_user: User = Depends(get_current_admin_user)
):
    """Get adaptive learning statistics (admin only)"""
    try:
        return {
            "performance_tracker": {
                "total_users": len(performance_tracker.user_profiles),
                "performance_categories": {
                    category.value: len(performance_tracker.get_users_by_category(category))
                    for category in PerformanceCategory
                },
                "risk_profiles": {
                    profile.value: len(performance_tracker.get_users_by_risk_profile(profile))
                    for profile in RiskProfile
                }
            },
            "outlier_detector": outlier_detector.get_outlier_summary(),
            "weight_manager": weight_manager.get_weight_statistics(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get adaptive statistics: {e}")
        raise HTTPException(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get adaptive statistics"
        )
