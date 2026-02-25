"""
Nexus Trading System - Strategy Weight Manager
Manages and updates strategy weights based on user performance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from database.session import get_database_session
from database.models import User, Trade, UserPerformance, AdaptiveWeight, Signal
from database.schemas import UserRole, UserStatus
from core.logger import get_logger
from adaptive.user_performance_tracker import UserPerformanceProfile, PerformanceMetrics
from adaptive.outlier_detector import OutlierDetector

logger = get_logger()


class WeightUpdateMethod(Enum):
    """Methods for updating strategy weights"""
    PERFORMANCE_BASED = "performance_based"
    RISK_ADJUSTED = "risk_adjusted"
    REGIME_AWARE = "regime_aware"
    HYBRID = "hybrid"
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"


@dataclass
class StrategyWeight:
    """Strategy weight configuration"""
    strategy_name: str
    base_weight: float
    adaptive_weight: float
    final_weight: float
    performance_score: float
    confidence: float
    contributing_users: List[int]
    sample_size: int
    last_updated: datetime
    update_method: WeightUpdateMethod
    regime_adjustments: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'base_weight': self.base_weight,
            'adaptive_weight': self.adaptive_weight,
            'final_weight': self.final_weight,
            'performance_score': self.performance_score,
            'confidence': self.confidence,
            'contributing_users': self.contributing_users,
            'sample_size': self.sample_size,
            'last_updated': self.last_updated.isoformat(),
            'update_method': self.update_method.value,
            'regime_adjustments': self.regime_adjustments
        }


class StrategyWeightManager:
    """Manages strategy weights based on user performance"""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95, 
                 min_sample_size: int = 3, weight_smoothing: float = 0.8):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_sample_size = min_sample_size
        self.weight_smoothing = weight_smoothing
        
        # Strategy weights storage
        self.strategy_weights = {}
        self.weight_history = defaultdict(list)
        self.performance_cache = {}
        self.last_update = datetime.utcnow()
        
        # Weight update configuration
        self.update_method = WeightUpdateMethod.HYBRID
        self.regime_weights = defaultdict(dict)
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'average': 0.4,
            'poor': 0.2
        }
        
        logger.system_logger.info("Strategy Weight Manager initialized")
    
    def update_strategy_weights(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                 outlier_detector: Optional[OutlierDetector] = None) -> Dict[str, StrategyWeight]:
        """Update strategy weights based on user performance"""
        try:
            # Filter out outliers if detector provided
            if outlier_detector:
                filtered_profiles = {
                    uid: profile for uid, profile in user_profiles.items()
                    if not outlier_detector.is_user_outlier(uid)
                }
                logger.system_logger.info(f"Filtered out {len(user_profiles) - len(filtered_profiles)} outlier users")
                user_profiles = filtered_profiles
            
            # Group users by strategy performance
            strategy_performance = self._group_by_strategy_performance(user_profiles)
            
            # Update weights for each strategy
            updated_weights = {}
            
            for strategy_name, performance_data in strategy_performance.items():
                if len(performance_data['users']) >= self.min_sample_size:
                    # Calculate new weight
                    new_weight = self._calculate_strategy_weight(strategy_name, performance_data)
                    
                    # Apply regime adjustments
                    regime_adjusted_weight = self._apply_regime_adjustments(strategy_name, new_weight)
                    
                    # Apply smoothing
                    smoothed_weight = self._apply_smoothing(strategy_name, regime_adjusted_weight)
                    
                    # Create weight object
                    weight_obj = StrategyWeight(
                        strategy_name=strategy_name,
                        base_weight=1.0,
                        adaptive_weight=new_weight,
                        final_weight=smoothed_weight,
                        performance_score=performance_data['avg_performance'],
                        confidence=performance_data['confidence'],
                        contributing_users=performance_data['users'],
                        sample_size=len(performance_data['users']),
                        last_updated=datetime.utcnow(),
                        update_method=self.update_method,
                        regime_adjustments=self.regime_weights.get(strategy_name, {})
                    )
                    
                    updated_weights[strategy_name] = weight_obj
                    
                    # Update storage
                    self.strategy_weights[strategy_name] = weight_obj
                    self.weight_history[strategy_name].append(weight_obj)
                    
                    # Update cache
                    self.performance_cache[strategy_name] = performance_data
            
            # Normalize weights
            self._normalize_weights(updated_weights)
            
            # Update last update time
            self.last_update = datetime.utcnow()
            
            logger.system_logger.info(f"Updated weights for {len(updated_weights)} strategies")
            return updated_weights
            
        except Exception as e:
            logger.system_logger.error(f"Error updating strategy weights: {e}")
            return {}
    
    def _group_by_strategy_performance(self, user_profiles: Dict[int, UserPerformanceProfile]) -> Dict[str, Dict[str, Any]]:
        """Group users by strategy performance"""
        strategy_performance = defaultdict(lambda: {
            'users': [],
            'performance_scores': [],
            'win_rates': [],
            'profit_factors': [],
            'sharpe_ratios': [],
            'sample_sizes': [],
            'influence_scores': []
        })
        
        for user_id, profile in user_profiles.items():
            # Get user's strategy preferences
            for strategy, preference in profile.strategy_preferences.items():
                if preference > 0.1:  # Only include strategies with meaningful preference
                    strategy_performance[strategy]['users'].append(user_id)
                    strategy_performance[strategy]['performance_scores'].append(profile.metrics.performance_score)
                    strategy_performance[strategy]['win_rates'].append(profile.metrics.win_rate)
                    strategy_performance[strategy]['profit_factors'].append(profile.metrics.profit_factor)
                    strategy_performance[strategy]['sharpe_ratios'].append(profile.metrics.sharpe_ratio)
                    strategy_performance[strategy]['sample_sizes'].append(profile.metrics.total_trades)
                    strategy_performance[strategy]['influence_scores'].append(profile.influence_score)
        
        # Calculate aggregated metrics
        for strategy_name, data in strategy_performance.items():
            if data['performance_scores']:
                data['avg_performance'] = np.mean(data['performance_scores'])
                data['avg_win_rate'] = np.mean(data['win_rates'])
                data['avg_profit_factor'] = np.mean(data['profit_factors'])
                data['avg_sharpe_ratio'] = np.mean(data['sharpe_ratios'])
                data['total_trades'] = sum(data['sample_sizes'])
                data['avg_influence'] = np.mean(data['influence_scores'])
                
                # Calculate confidence based on sample size and variance
                sample_size = len(data['users'])
                variance = np.var(data['performance_scores'])
                
                # Higher confidence for larger samples and lower variance
                size_confidence = min(sample_size / 10, 1.0)
                variance_confidence = max(1.0 - variance / 0.25, 0.0)  # Normalize variance to 0-1 range
                data['confidence'] = (size_confidence + variance_confidence) / 2
            else:
                data['avg_performance'] = 0.0
                data['avg_win_rate'] = 0.0
                data['avg_profit_factor'] = 0.0
                data['avg_sharpe_ratio'] = 0.0
                data['total_trades'] = 0
                data['avg_influence'] = 0.0
                data['confidence'] = 0.0
        
        return dict(strategy_performance)
    
    def _calculate_strategy_weight(self, strategy_name: str, performance_data: Dict[str, Any]) -> float:
        """Calculate strategy weight based on performance data"""
        avg_performance = performance_data['avg_performance']
        confidence = performance_data['confidence']
        avg_influence = performance_data['avg_influence']
        
        # Base weight from performance
        if self.update_method == WeightUpdateMethod.PERFORMANCE_BASED:
            weight = self._performance_based_weight(avg_performance, confidence)
        elif self.update_method == WeightUpdateMethod.RISK_ADJUSTED:
            weight = self._risk_adjusted_weight(avg_performance, performance_data)
        elif self.update_method == WeightUpdateMethod.REGIME_AWARE:
            weight = self._regime_aware_weight(strategy_name, avg_performance, confidence)
        elif self.update_method == WeightUpdateMethod.EXPONENTIAL_MOVING_AVERAGE:
            weight = self._exponential_moving_average_weight(strategy_name, avg_performance, confidence)
        else:  # HYBRID
            weight = self._hybrid_weight(avg_performance, performance_data)
        
        # Apply learning rate
        if strategy_name in self.strategy_weights:
            current_weight = self.strategy_weights[strategy_name].adaptive_weight
            weight = current_weight * (1 - self.learning_rate) + weight * self.learning_rate
        
        return max(0.01, min(weight, 1.0))  # Ensure weight is between 0.01 and 1.0
    
    def _performance_based_weight(self, avg_performance: float, confidence: float) -> float:
        """Calculate weight based on performance only"""
        # Normalize performance to 0-1 range
        normalized_performance = max(0, min(avg_performance, 1.0))
        
        # Apply confidence weighting
        confidence_adjusted = normalized_performance * confidence + 0.5 * (1 - confidence)
        
        return confidence_adjusted
    
    def _risk_adjusted_weight(self, avg_performance: float, performance_data: Dict[str, Any]) -> float:
        """Calculate weight with risk adjustment"""
        base_weight = self._performance_based_weight(avg_performance, performance_data['confidence'])
        
        # Adjust based on risk metrics
        avg_sharpe = performance_data['avg_sharpe_ratio']
        avg_profit_factor = performance_data['avg_profit_factor']
        
        # Risk adjustment factor
        risk_adjustment = 1.0
        
        if avg_sharpe > 2.0:
            risk_adjustment *= 1.2  # Boost for high Sharpe ratio
        elif avg_sharpe < 0.5:
            risk_adjustment *= 0.8  # Reduce for low Sharpe ratio
        
        if avg_profit_factor > 2.0:
            risk_adjustment *= 1.1  # Boost for high profit factor
        elif avg_profit_factor < 1.0:
            risk_adjustment *= 0.9  # Reduce for low profit factor
        
        return base_weight * risk_adjustment
    
    def _regime_aware_weight(self, strategy_name: str, avg_performance: float, confidence: float) -> float:
        """Calculate weight with regime awareness"""
        base_weight = self._performance_based_weight(avg_performance, confidence)
        
        # Apply regime adjustments if available
        if strategy_name in self.regime_weights:
            current_regime = self._get_current_market_regime()
            regime_adjustment = self.regime_weights[strategy_name].get(current_regime, 1.0)
            base_weight *= regime_adjustment
        
        return base_weight
    
    def _exponential_moving_average_weight(self, strategy_name: str, avg_performance: float, confidence: float) -> float:
        """Calculate weight using exponential moving average"""
        base_weight = self._performance_based_weight(avg_performance, confidence)
        
        # Apply EMA smoothing
        if strategy_name in self.strategy_weights:
            current_weight = self.strategy_weights[strategy_name].adaptive_weight
            ema_weight = current_weight * self.weight_smoothing + base_weight * (1 - self.weight_smoothing)
            return ema_weight
        else:
            return base_weight
    
    def _hybrid_weight(self, avg_performance: float, performance_data: Dict[str, Any]) -> float:
        """Calculate hybrid weight combining multiple factors"""
        # Performance component
        performance_weight = self._performance_based_weight(avg_performance, performance_data['confidence'])
        
        # Risk adjustment component
        risk_weight = self._risk_adjusted_weight(avg_performance, performance_data)
        
        # Influence component
        influence_weight = performance_data['avg_influence']
        
        # Combine weights
        hybrid_weight = (
            0.5 * performance_weight +
            0.3 * risk_weight +
            0.2 * influence_weight
        )
        
        return hybrid_weight
    
    def _apply_regime_adjustments(self, strategy_name: str, weight: float) -> float:
        """Apply regime-specific adjustments"""
        current_regime = self._get_current_market_regime()
        
        if strategy_name in self.regime_weights and current_regime in self.regime_weights[strategy_name]:
            regime_adjustment = self.regime_weights[strategy_name][current_regime]
            return weight * regime_adjustment
        
        return weight
    
    def _apply_smoothing(self, strategy_name: str, weight: float) -> float:
        """Apply weight smoothing"""
        if strategy_name in self.strategy_weights:
            current_weight = self.strategy_weights[strategy_name].final_weight
            smoothed_weight = current_weight * self.weight_smoothing + weight * (1 - self.weight_smoothing)
            return smoothed_weight
        else:
            return weight
    
    def _normalize_weights(self, weights: Dict[str, StrategyWeight]) -> None:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(weight.final_weight for weight in weights.values())
        
        if total_weight > 0:
            for weight in weights.values():
                weight.final_weight = weight.final_weight / total_weight
    
    def _get_current_market_regime(self) -> str:
        """Get current market regime"""
        # This would typically be determined by the regime detector
        # For now, return a default
        return "neutral"
    
    def update_regime_weights(self, regime: str, strategy_performance: Dict[str, float]) -> None:
        """Update regime-specific weights"""
        for strategy_name, performance in strategy_performance.items():
            if strategy_name not in self.regime_weights:
                self.regime_weights[strategy_name] = {}
            
            self.regime_weights[strategy_name][regime] = performance
        
        logger.system_logger.info(f"Updated regime weights for {len(strategy_performance)} strategies in {regime} regime")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return {
            strategy_name: weight.final_weight
            for strategy_name, weight in self.strategy_weights.items()
        }
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for a specific strategy"""
        if strategy_name in self.strategy_weights:
            return self.strategy_weights[strategy_name].final_weight
        return 1.0  # Default weight
    
    def get_weight_history(self, strategy_name: str, limit: int = 100) -> List[StrategyWeight]:
        """Get weight history for a strategy"""
        history = self.weight_history.get(strategy_name, [])
        return history[-limit:] if limit else history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        summary = {
            'total_strategies': len(self.strategy_weights),
            'last_update': self.last_update.isoformat(),
            'update_method': self.update_method.value,
            'strategies': {}
        }
        
        for strategy_name, weight in self.strategy_weights.items():
            summary['strategies'][strategy_name] = {
                'final_weight': weight.final_weight,
                'adaptive_weight': weight.adaptive_weight,
                'performance_score': weight.performance_score,
                'confidence': weight.confidence,
                'sample_size': weight.sample_size,
                'contributing_users': len(weight.contributing_users)
            }
        
        return summary
    
    def get_top_strategies(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top performing strategies"""
        sorted_strategies = sorted(
            [(name, weight.performance_score) for name, weight in self.strategy_weights.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_strategies[:limit]
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """Get strategy ranking by final weight"""
        sorted_strategies = sorted(
            [(name, weight.final_weight) for name, weight in self.strategy_weights.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_strategies
    
    def set_update_method(self, method: WeightUpdateMethod) -> None:
        """Set weight update method"""
        self.update_method = method
        logger.system_logger.info(f"Weight update method changed to {method.value}")
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate for weight updates"""
        self.learning_rate = max(0.01, min(learning_rate, 1.0))
        logger.system_logger.info(f"Learning rate set to {self.learning_rate}")
    
    def set_decay_factor(self, decay_factor: float) -> None:
        """Set decay factor for weight updates"""
        self.decay_factor = max(0.8, min(decay_factor, 0.99))
        logger.system_logger.info(f"Decay factor set to {self.decay_factor}")
    
    def reset_weights(self) -> None:
        """Reset all weights to default values"""
        for weight in self.strategy_weights.values():
            weight.adaptive_weight = 1.0
            weight.final_weight = 1.0
            weight.performance_score = 0.5
            weight.confidence = 0.0
            weight.last_updated = datetime.utcnow()
        
        self._normalize_weights(self.strategy_weights)
        logger.system_logger.info("All strategy weights reset to default values")
    
    def export_weights(self) -> Dict[str, Any]:
        """Export weights for backup or analysis"""
        return {
            'strategy_weights': {
                name: weight.to_dict() for name, weight in self.strategy_weights.items()
            },
            'weight_history': {
                name: [w.to_dict() for w in history] 
                for name, history in self.weight_history.items()
            },
            'regime_weights': dict(self.regime_weights),
            'configuration': {
                'learning_rate': self.learning_rate,
                'decay_factor': self.decay_factor,
                'min_sample_size': self.min_sample_size,
                'weight_smoothing': self.weight_smoothing,
                'update_method': self.update_method.value
            },
            'last_update': self.last_update.isoformat()
        }
    
    def import_weights(self, weights_data: Dict[str, Any]) -> bool:
        """Import weights from backup"""
        try:
            # Import strategy weights
            if 'strategy_weights' in weights_data:
                for name, data in weights_data['strategy_weights'].items():
                    weight = StrategyWeight(
                        strategy_name=data['strategy_name'],
                        base_weight=data['base_weight'],
                        adaptive_weight=data['adaptive_weight'],
                        final_weight=data['final_weight'],
                        performance_score=data['performance_score'],
                        confidence=data['confidence'],
                        contributing_users=data['contributing_users'],
                        sample_size=data['sample_size'],
                        last_updated=datetime.fromisoformat(data['last_updated']),
                        update_method=WeightUpdateMethod(data['update_method']),
                        regime_adjustments=data.get('regime_adjustments', {})
                    )
                    self.strategy_weights[name] = weight
            
            # Import regime weights
            if 'regime_weights' in weights_data:
                self.regime_weights = defaultdict(dict, weights_data['regime_weights'])
            
            # Import configuration
            if 'configuration' in weights_data:
                config = weights_data['configuration']
                self.learning_rate = config.get('learning_rate', self.learning_rate)
                self.decay_factor = config.get('decay_factor', self.decay_factor)
                self.min_sample_size = config.get('min_sample_size', self.min_sample_size)
                self.weight_smoothing = config.get('weight_smoothing', self.weight_smoothing)
                self.update_method = WeightUpdateMethod(config.get('update_method', 'hybrid'))
            
            logger.system_logger.info("Weights imported successfully")
            return True
            
        except Exception as e:
            logger.system_logger.error(f"Error importing weights: {e}")
            return False
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get weight statistics"""
        if not self.strategy_weights:
            return {}
        
        weights = [weight.final_weight for weight in self.strategy_weights.values()]
        performances = [weight.performance_score for weight in self.strategy_weights.values()]
        confidences = [weight.confidence for weight in self.strategy_weights.values()]
        
        return {
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'median': np.median(weights)
            },
            'performance_stats': {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'min': np.min(performances),
                'max': np.max(performances),
                'median': np.median(performances)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            },
            'total_samples': sum(weight.sample_size for weight in self.strategy_weights.values()),
            'avg_confidence': np.mean(confidences)
        }


# Factory function
def create_strategy_weight_manager(learning_rate: float = 0.1, decay_factor: float = 0.95, 
                                   min_sample_size: int = 3, weight_smoothing: float = 0.8) -> StrategyWeightManager:
    """Create and return strategy weight manager instance"""
    return StrategyWeightManager(learning_rate, decay_factor, min_sample_size, weight_smoothing)
