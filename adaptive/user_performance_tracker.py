"""
Nexus Trading System - User Performance Tracker
Tracks and analyzes user performance for adaptive learning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

from core.logger import get_logger

logger = get_logger()
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from database.session import get_database_session
from database.models import User, Trade, UserPerformance, AdaptiveWeight, Signal
from database.schemas import UserRole, UserStatus
from core.logger import get_logger
from adaptive.adaptive_engine import PerformanceMetrics

logger = get_logger()


class PerformanceCategory(Enum):
    """Performance categories for user classification"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    PROFESSIONAL = "professional"
    MASTER = "master"


class RiskProfile(Enum):
    """Risk profile categories"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class UserPerformanceProfile:
    """Complete user performance profile"""
    user_id: int
    username: str
    role: UserRole
    performance_category: PerformanceCategory
    risk_profile: RiskProfile
    metrics: PerformanceMetrics
    strategy_preferences: Dict[str, float]
    asset_preferences: Dict[str, float]
    timeframe_preferences: Dict[str, float]
    behavioral_patterns: Dict[str, Any]
    learning_progress: Dict[str, float]
    influence_score: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'performance_category': self.performance_category.value,
            'risk_profile': self.risk_profile.value,
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'win_rate': self.metrics.win_rate,
                'total_pnl': self.metrics.total_pnl,
                'max_drawdown': self.metrics.max_drawdown,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'profit_factor': self.metrics.profit_factor,
                'avg_trade_duration': self.metrics.avg_trade_duration,
                'risk_reward_ratio': self.metrics.risk_reward_ratio,
                'consistency_score': self.metrics.consistency_score,
                'performance_score': self.metrics.performance_score
            },
            'strategy_preferences': self.strategy_preferences,
            'asset_preferences': self.asset_preferences,
            'timeframe_preferences': self.timeframe_preferences,
            'behavioral_patterns': self.behavioral_patterns,
            'learning_progress': self.learning_progress,
            'influence_score': self.influence_score,
            'last_updated': self.last_updated.isoformat()
        }


class UserPerformanceTracker:
    """Tracks and analyzes user performance for adaptive learning"""
    
    def __init__(self, min_trades_for_analysis: int = 10, performance_window_days: int = 30):
        self.min_trades_for_analysis = min_trades_for_analysis
        self.performance_window_days = performance_window_days
        self.user_profiles = {}
        self.performance_history = defaultdict(list)
        self.behavioral_patterns = defaultdict(dict)
        self.learning_progress = defaultdict(dict)
        self.influence_scores = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            PerformanceCategory.BEGINNER: {'min_score': 0.0, 'max_score': 0.3},
            PerformanceCategory.INTERMEDIATE: {'min_score': 0.3, 'max_score': 0.5},
            PerformanceCategory.ADVANCED: {'min_score': 0.5, 'max_score': 0.7},
            PerformanceCategory.EXPERT: {'min_score': 0.7, 'max_score': 0.85},
            PerformanceCategory.PROFESSIONAL: {'min_score': 0.85, 'max_score': 0.95},
            PerformanceCategory.MASTER: {'min_score': 0.95, 'max_score': 1.0}
        }
        
        # Risk profile thresholds
        self.risk_thresholds = {
            RiskProfile.CONSERVATIVE: {'min_rr': 1.5, 'max_rr': 2.5},
            RiskProfile.MODERATE: {'min_rr': 1.0, 'max_rr': 1.5},
            RiskProfile.AGGRESSIVE: {'min_rr': 0.5, 'max_rr': 1.0},
            RiskProfile.VERY_AGGRESSIVE: {'min_rr': 0.2, 'max_rr': 0.5}
        }
        
        logger.system_logger.info("User Performance Tracker initialized")
    
    def track_user_performance(self, user_id: int) -> Optional[UserPerformanceProfile]:
        """Track and analyze user performance"""
        try:
            with get_database_session() as db:
                # Get user information
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    logger.system_logger.warning(f"User {user_id} not found")
                    return None
                
                # Get user's trades within performance window
                cutoff_date = datetime.utcnow() - timedelta(days=self.performance_window_days)
                trades = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.entry_time >= cutoff_date
                ).all()
                
                if len(trades) < self.min_trades_for_analysis:
                    logger.system_logger.warning(f"User {user_id} has insufficient trades for analysis: {len(trades)}")
                    return None
                
                # Calculate performance metrics
                metrics = PerformanceMetrics()
                metrics.calculate_metrics([{
                    'pnl': t.get('pnl', 0),
                    'duration_minutes': t.get('duration_minutes', 0),
                    'symbol': t.get('symbol', ''),
                    'strategy': t.get('strategy', ''),
                    'signal_confidence': t.get('signal_confidence', 0),
                    'entry_time': t.get('entry_time'),
                    'exit_time': t.get('exit_time'),
                    'stop_loss': t.get('stop_loss'),
                    'take_profit': t.get('take_profit'),
                    'size': t.get('size', 0),
                    'entry_price': t.get('entry_price'),
                    'exit_price': t.get('exit_price')
                } for t in trades])
                
                # Analyze preferences
                strategy_preferences = self._analyze_strategy_preferences(trades)
                asset_preferences = self._analyze_asset_preferences(trades)
                timeframe_preferences = self._analyze_timeframe_preferences(trades)
                
                # Analyze behavioral patterns
                behavioral_patterns = self._analyze_behavioral_patterns(trades, user)
                
                # Analyze learning progress
                learning_progress = self._analyze_learning_progress(user_id, trades)
                
                # Calculate influence score
                influence_score = self._calculate_influence_score(user, metrics, trades)
                
                # Determine performance category
                performance_category = self._determine_performance_category(metrics.performance_score)
                
                # Determine risk profile
                risk_profile = self._determine_risk_profile(metrics.risk_reward_ratio)
                
                # Create user profile
                profile = UserPerformanceProfile(
                    user_id=user_id,
                    username=user.username,
                    role=user.role,
                    performance_category=performance_category,
                    risk_profile=risk_profile,
                    metrics=metrics,
                    strategy_preferences=strategy_preferences,
                    asset_preferences=asset_preferences,
                    timeframe_preferences=timeframe_preferences,
                    behavioral_patterns=behavioral_patterns,
                    learning_progress=learning_progress,
                    influence_score=influence_score,
                    last_updated=datetime.utcnow()
                )
                
                # Store profile
                self.user_profiles[user_id] = profile
                
                # Update performance history
                self.performance_history[user_id].append({
                    'timestamp': datetime.utcnow(),
                    'performance_score': metrics.performance_score,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio
                })
                
                # Update behavioral patterns
                self.behavioral_patterns[user_id] = behavioral_patterns
                
                # Update learning progress
                self.learning_progress[user_id] = learning_progress
                
                # Update influence scores
                self.influence_scores[user_id] = influence_score
                
                logger.system_logger.info(f"User {user_id} performance tracked successfully")
                return profile
                
        except Exception as e:
            logger.error(f"Error tracking user {user_id} performance: {e}")
            return None
    
    def _analyze_strategy_preferences(self, trades: List[Trade]) -> Dict[str, float]:
        """Analyze user's strategy preferences"""
        strategy_counts = defaultdict(int)
        total_trades = len(trades)
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            strategy_counts[strategy] += 1
        
        # Calculate preferences as percentages
        strategy_preferences = {
            strategy: count / total_trades 
            for strategy, count in strategy_counts.items()
        }
        
        return strategy_preferences
    
    def _analyze_asset_preferences(self, trades: List[Trade]) -> Dict[str, float]:
        """Analyze user's asset preferences"""
        asset_counts = defaultdict(int)
        total_trades = len(trades)
        
        for trade in trades:
            symbol = trade.get('symbol', 'unknown')
            asset_counts[symbol] += 1
        
        # Calculate preferences as percentages
        asset_preferences = {
            symbol: count / total_trades 
            for symbol, count in asset_counts.items()
        }
        
        return asset_preferences
    
    def _analyze_timeframe_preferences(self, trades: List[Trade]) -> Dict[str, float]:
        """Analyze user's timeframe preferences"""
        timeframe_counts = defaultdict(int)
        total_trades = len(trades)
        
        for trade in trades:
            # Estimate timeframe based on trade duration
            duration = trade.get('duration_minutes', 0)
            if duration < 60:
                timeframe = "5M"
            elif duration < 240:
                timeframe = "15M"
            elif duration < 720:
                timeframe = "1H"
            elif duration < 2160:
                timeframe = "3H"
            elif duration < 4320:
                timeframe = "6H"
            else:
                timeframe = "9H"
            
            timeframe_counts[timeframe] += 1
        
        # Calculate preferences as percentages
        timeframe_preferences = {
            timeframe: count / total_trades 
            for timeframe, count in timeframe_counts.items()
        }
        
        return timeframe_preferences
    
    def _analyze_behavioral_patterns(self, trades: List[Trade], user: User) -> Dict[str, Any]:
        """Analyze user's behavioral patterns"""
        patterns = {}
        
        # Trading frequency
        trading_days = len(set(trade.get('entry_time').date() for trade in trades if trade.get('entry_time')))
        patterns['trading_frequency'] = len(trades) / max(trading_days, 1)
        
        # Average holding time
        durations = [trade.get('duration_minutes', 0) for trade in trades]
        patterns['avg_holding_time'] = np.mean(durations) if durations else 0
        
        # Risk tolerance
        risk_sizes = [trade.get('size', 0) for trade in trades]
        patterns['avg_position_size'] = np.mean(risk_sizes) if risk_sizes else 0
        patterns['max_position_size'] = max(risk_sizes) if risk_sizes else 0
        
        # Stop loss discipline
        sl_hits = len([t for t in trades if t.get('exit_reason') == 'SL'])
        patterns['sl_hit_rate'] = sl_hits / len(trades) if trades else 0
        
        # Take profit discipline
        tp_hits = len([t for t in trades if t.get('exit_reason') == 'TP'])
        patterns['tp_hit_rate'] = tp_hits / len(trades) if trades else 0
        
        # Manual exit tendency
        manual_exits = len([t for t in trades if t.get('exit_reason') == 'MANUAL'])
        patterns['manual_exit_rate'] = manual_exits / len(trades) if trades else 0
        
        # Trading session preference
        session_counts = defaultdict(int)
        for trade in trades:
            entry_time = trade.get('entry_time')
            if entry_time:
                hour = entry_time.hour
                if 8 <= hour < 16:
                    session = "london"
                elif 13 <= hour < 21:
                    session = "new_york"
                elif 21 <= hour or hour < 5:
                    session = "asian"
                else:
                    session = "overlap"
                session_counts[session] += 1
        
        patterns['session_preferences'] = {
            session: count / len(trades) 
            for session, count in session_counts.items()
        }
        
        # Confidence level
        confidences = [trade.get('signal_confidence', 0) for trade in trades]
        patterns['avg_confidence'] = np.mean(confidences) if confidences else 0
        
        # Auto-trading usage
        patterns['auto_trade_enabled'] = user.auto_trade_enabled
        
        return patterns
    
    def _analyze_learning_progress(self, user_id: int, trades: List[Trade]) -> Dict[str, float]:
        """Analyze user's learning progress over time"""
        progress = {}
        
        # Get historical performance data
        with get_database_session() as db:
            historical_performance = db.query(UserPerformance).filter(
                UserPerformance.user_id == user_id
            ).order_by(UserPerformance.period_start.desc()).limit(4).all()
        
        if len(historical_performance) >= 2:
            # Calculate improvement trends
            recent_performance = historical_performance[0]
            older_performance = historical_performance[-1]
            
            progress['performance_improvement'] = (
                recent_performance.performance_score - older_performance.performance_score
            ) / max(older_performance.performance_score, 0.01)
            
            progress['win_rate_improvement'] = (
                recent_performance.win_rate - older_performance.win_rate
            ) / max(older_performance.win_rate, 0.01)
            
            progress['profit_factor_improvement'] = (
                recent_performance.profit_factor - older_performance.profit_factor
            ) / max(older_performance.profit_factor, 0.01)
        else:
            progress['performance_improvement'] = 0.0
            progress['win_rate_improvement'] = 0.0
            progress['profit_factor_improvement'] = 0.0
        
        # Calculate consistency
        if len(trades) >= 10:
            returns = [trade.get('pnl', 0) for trade in trades]
            returns_series = pd.Series(returns)
            rolling_returns = returns_series.rolling(window=5)
            consistency = (rolling_returns > 0).mean()
            progress['consistency_score'] = consistency
        else:
            progress['consistency_score'] = 0.5
        
        # Calculate learning rate
        progress['learning_rate'] = progress['performance_improvement'] * progress['consistency_score']
        
        return progress
    
    def _calculate_influence_score(self, user: User, metrics: PerformanceMetrics, trades: List[Trade]) -> float:
        """Calculate user influence score"""
        # Base influence from performance
        performance_influence = metrics.performance_score
        
        # Trade frequency influence
        trade_frequency = len(trades) / max(self.performance_window_days, 1)
        frequency_influence = min(trade_frequency / 10, 1.0)  # Normalize to 0-1
        
        # Consistency influence
        consistency_influence = metrics.consistency_score
        
        # Role influence
        role_weights = {
            UserRole.FREE: 0.5,
            UserRole.VERIFIED: 0.7,
            UserRole.ELITE: 0.9,
            UserRole.ADMIN: 1.0
        }
        role_influence = role_weights.get(user.role, 0.5)
        
        # Calculate total influence
        total_influence = (
            0.4 * performance_influence +
            0.3 * frequency_influence +
            0.2 * consistency_influence +
            0.1 * role_influence
        )
        
        return min(total_influence, 1.0)
    
    def _determine_performance_category(self, performance_score: float) -> PerformanceCategory:
        """Determine user performance category"""
        for category, thresholds in self.performance_thresholds.items():
            if thresholds['min_score'] <= performance_score <= thresholds['max_score']:
                return category
        return PerformanceCategory.BEGINNER
    
    def _determine_risk_profile(self, risk_reward_ratio: float) -> RiskProfile:
        """Determine user risk profile"""
        for profile, thresholds in self.risk_thresholds.items():
            if thresholds['min_rr'] <= risk_reward_ratio <= thresholds['max_rr']:
                return profile
        return RiskProfile.MODERATE
    
    def get_user_profile(self, user_id: int) -> Optional[UserPerformanceProfile]:
        """Get user performance profile"""
        return self.user_profiles.get(user_id)
    
    def get_top_performers(self, limit: int = 10) -> List[UserPerformanceProfile]:
        """Get top performing users"""
        sorted_profiles = sorted(
            self.user_profiles.values(),
            key=lambda p: p.metrics.performance_score,
            reverse=True
        )
        return sorted_profiles[:limit]
    
    def get_users_by_category(self, category: PerformanceCategory) -> List[UserPerformanceProfile]:
        """Get users by performance category"""
        return [
            profile for profile in self.user_profiles.values()
            if profile.performance_category == category
        ]
    
    def get_users_by_risk_profile(self, risk_profile: RiskProfile) -> List[UserPerformanceProfile]:
        """Get users by risk profile"""
        return [
            profile for profile in self.user_profiles.values()
            if profile.risk_profile == risk_profile
        ]
    
    def get_influence_ranking(self, limit: int = 10) -> List[Tuple[int, float]]:
        """Get users by influence score"""
        sorted_users = sorted(
            self.influence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_users[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.user_profiles:
            return {}
        
        profiles = list(self.user_profiles.values())
        
        summary = {
            'total_users': len(profiles),
            'performance_distribution': {
                category.value: len([p for p in profiles if p.performance_category == category])
                for category in PerformanceCategory
            },
            'risk_profile_distribution': {
                profile.value: len([p for p in profiles if p.risk_profile == profile])
                for profile in RiskProfile
            },
            'avg_performance_score': np.mean([p.metrics.performance_score for p in profiles]),
            'avg_win_rate': np.mean([p.metrics.win_rate for p in profiles]),
            'avg_profit_factor': np.mean([p.metrics.profit_factor for p in profiles]),
            'avg_sharpe_ratio': np.mean([p.metrics.sharpe_ratio for p in profiles]),
            'avg_influence_score': np.mean([p.influence_score for p in profiles]),
            'top_performers': self.get_top_performers(5),
            'most_influential': self.get_influence_ranking(5)
        }
        
        return summary
    
    def update_user_profile(self, user_id: int) -> Optional[UserPerformanceProfile]:
        """Update user performance profile"""
        return self.track_user_performance(user_id)
    
    def get_user_learning_progress(self, user_id: int) -> Dict[str, float]:
        """Get user learning progress"""
        return self.learning_progress.get(user_id, {})
    
    def get_user_behavioral_patterns(self, user_id: int) -> Dict[str, Any]:
        """Get user behavioral patterns"""
        return self.behavioral_patterns.get(user_id, {})
    
    def get_performance_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user performance history"""
        return self.performance_history.get(user_id, [])
    
    def export_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Export user data for analysis"""
        profile = self.get_user_profile(user_id)
        if not profile:
            return None
        
        return {
            'profile': profile.to_dict(),
            'performance_history': self.get_performance_history(user_id),
            'behavioral_patterns': self.get_user_behavioral_patterns(user_id),
            'learning_progress': self.get_user_learning_progress(user_id)
        }
    
    def batch_update_profiles(self, user_ids: List[int]) -> Dict[int, Optional[UserPerformanceProfile]]:
        """Batch update user profiles"""
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.update_user_profile(user_id)
        
        return results
    
    def get_adaptive_recommendations(self, user_id: int) -> Dict[str, Any]:
        """Get adaptive recommendations for user"""
        profile = self.get_user_profile(user_id)
        if not profile:
            return {}
        
        recommendations = {}
        
        # Strategy recommendations
        if profile.strategy_preferences:
            best_strategy = max(profile.strategy_preferences.items(), key=lambda x: x[1])
            recommendations['strategy'] = {
                'focus_on': best_strategy[0],
                'reason': f"You perform best with {best_strategy[0]} strategy ({best_strategy[1]:.1%} win rate)"
            }
        
        # Risk management recommendations
        if profile.risk_profile == RiskProfile.VERY_AGGRESSIVE:
            recommendations['risk_management'] = {
                'action': 'reduce_risk',
                'reason': 'Your risk profile is very aggressive. Consider reducing position sizes.'
            }
        elif profile.risk_profile == RiskProfile.CONSERVATIVE:
            recommendations['risk_management'] = {
                'action': 'increase_risk',
                'reason': 'Your risk profile is conservative. Consider taking calculated risks for better returns.'
            }
        
        # Learning recommendations
        if profile.learning_progress.get('learning_rate', 0) < 0.1:
            recommendations['learning'] = {
                'action': 'focus_on_learning',
                'reason': 'Your learning rate is low. Consider more practice and education.'
            }
        
        # Trading frequency recommendations
        if profile.behavioral_patterns.get('trading_frequency', 0) > 10:
            recommendations['frequency'] = {
                'action': 'reduce_frequency',
                'reason': 'You trade very frequently. Consider quality over quantity.'
            }
        elif profile.behavioral_patterns.get('trading_frequency', 0) < 1:
            recommendations['frequency'] = {
                'action': 'increase_frequency',
                'reason': 'You trade infrequently. Consider more regular trading.'
            }
        
        return recommendations


# Factory function
def create_performance_tracker(min_trades_for_analysis: int = 10, performance_window_days: int = 30) -> UserPerformanceTracker:
    """Create and return performance tracker instance"""
    return UserPerformanceTracker(min_trades_for_analysis, performance_window_days)
