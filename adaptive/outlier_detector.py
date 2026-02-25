"""
Nexus Trading System - Outlier Detection System
Detects and filters out outlier users to prevent manipulation
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
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

from database.session import get_database_session
from database.models import User, Trade, UserPerformance, AdaptiveWeight, Signal
from database.schemas import UserRole, UserStatus
from core.logger import get_logger
from adaptive.user_performance_tracker import UserPerformanceProfile, PerformanceMetrics

logger = get_logger()


class OutlierType(Enum):
    """Types of outliers"""
    PERFORMANCE = "performance"
    FREQUENCY = "frequency"
    RISK = "risk"
    CONSISTENCY = "consistency"
    BEHAVIOR = "behavior"
    MANIPULATION = "manipulation"
    ANOMALY = "anomaly"


class OutlierSeverity(Enum):
    """Outlier severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OutlierDetection:
    """Outlier detection result"""
    user_id: int
    username: str
    outlier_type: OutlierType
    severity: OutlierSeverity
    z_score: float
    p_value: float
    confidence: float
    reasons: List[str]
    metrics: Dict[str, float]
    detected_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'outlier_type': self.outlier_type.value,
            'severity': self.severity.value,
            'z_score': self.z_score,
            'p_value': self.p_value,
            'confidence': self.confidence,
            'reasons': self.reasons,
            'metrics': self.metrics,
            'detected_at': self.detected_at.isoformat(),
            'is_active': self.is_active
        }


class OutlierDetector:
    """Advanced outlier detection system"""
    
    def __init__(self, z_threshold: float = 2.5, iqr_multiplier: float = 1.5, 
                 contamination_rate: float = 0.1, min_samples_for_detection: int = 20):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.contamination_rate = contamination_rate
        self.min_samples_for_detection = min_samples_for_detection
        
        # Detection methods
        self.isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
        self.scaler = StandardScaler()
        
        # Detection history
        self.detection_history = defaultdict(list)
        self.outlier_scores = defaultdict(float)
        self.outlier_counts = defaultdict(int)
        self.active_outliers = {}
        
        # Detection thresholds
        self.detection_thresholds = {
            OutlierType.PERFORMANCE: {'z_score': 3.0, 'p_value': 0.01},
            OutlierType.FREQUENCY: {'z_score': 2.5, 'p_value': 0.05},
            OutlierType.RISK: {'z_score': 2.0, 'p_value': 0.05},
            OutlierType.CONSISTENCY: {'z_score': 2.5, 'p_value': 0.05},
            OutlierType.BEHAVIOR: {'z_score': 3.0, 'p_value': 0.01},
            OutlierType.MANIPULATION: {'z_score': 4.0, 'p_value': 0.001},
            OutlierType.ANOMALY: {'z_score': 3.5, 'p_value': 0.005}
        }
        
        logger.system_logger.info("Outlier Detector initialized")
    
    def detect_outliers(self, user_profiles: Dict[int, UserPerformanceProfile]) -> List[OutlierDetection]:
        """Detect outliers in user performance data"""
        if len(user_profiles) < self.min_samples_for_detection:
            logger.system_logger.warning(f"Insufficient samples for outlier detection: {len(user_profiles)}")
            return []
        
        outliers = []
        
        # Prepare data for analysis
        user_metrics = self._prepare_user_metrics(user_profiles)
        
        # Detect different types of outliers
        outliers.extend(self._detect_performance_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_frequency_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_risk_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_consistency_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_behavioral_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_manipulation_outliers(user_profiles, user_metrics))
        outliers.extend(self._detect_statistical_outliers(user_profiles, user_metrics))
        
        # Update detection history
        for outlier in outliers:
            self.detection_history[outlier.user_id].append(outlier)
            self.outlier_scores[outlier.user_id] = max(self.outlier_scores[outlier.user_id], outlier.z_score)
            self.outlier_counts[outlier.user_id] += 1
            self.active_outliers[outlier.user_id] = outlier
        
        logger.system_logger(f"Detected {len(outliers)} outliers out of {len(user_profiles)} users")
        return outliers
    
    def _prepare_user_metrics(self, user_profiles: Dict[int, UserPerformanceProfile]) -> Dict[int, Dict[str, float]]:
        """Prepare user metrics for analysis"""
        user_metrics = {}
        
        for user_id, profile in user_profiles.items():
            metrics = {
                'performance_score': profile.metrics.performance_score,
                'win_rate': profile.metrics.win_rate,
                'profit_factor': profile.metrics.profit_factor,
                'sharpe_ratio': profile.metrics.sharpe_ratio,
                'max_drawdown': profile.metrics.max_drawdown,
                'avg_trade_duration': profile.metrics.avg_trade_duration,
                'risk_reward_ratio': profile.metrics.risk_reward_ratio,
                'consistency_score': profile.metrics.consistency_score,
                'total_trades': profile.metrics.total_trades,
                'influence_score': profile.influence_score,
                'avg_position_size': profile.behavioral_patterns.get('avg_position_size', 0),
                'trading_frequency': profile.behavioral_patterns.get('trading_frequency', 0),
                'sl_hit_rate': profile.behavioral_patterns.get('sl_hit_rate', 0),
                'tp_hit_rate': profile.behavioral_patterns.get('tp_hit_rate', 0),
                'manual_exit_rate': profile.behavioral_patterns.get('manual_exit_rate', 0),
                'avg_confidence': profile.behavioral_patterns.get('avg_confidence', 0)
            }
            
            user_metrics[user_id] = metrics
        
        return user_metrics
    
    def _detect_performance_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                   user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect performance outliers"""
        outliers = []
        
        performance_scores = [m['performance_score'] for m in user_metrics.values()]
        
        if len(performance_scores) < 10:
            return outliers
        
        # Calculate z-scores
        mean_score = np.mean(performance_scores)
        std_score = np.std(performance_scores)
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            score = metrics['performance_score']
            
            if std_score > 0:
                z_score = (score - mean_score) / std_score
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                threshold = self.detection_thresholds[OutlierType.PERFORMANCE]
                
                if abs(z_score) > threshold['z_score'] or p_value < threshold['p_value']:
                    severity = self._determine_severity(abs(z_score))
                    reasons = []
                    
                    if z_score > 0:
                        reasons.append(f"Exceptionally high performance (z-score: {z_score:.2f})")
                    else:
                        reasons.append(f"Exceptionally low performance (z-score: {z_score:.2f})")
                    
                    if metrics['win_rate'] > 0.8:
                        reasons.append(f"Unusually high win rate: {metrics['win_rate']:.2f}")
                    elif metrics['win_rate'] < 0.2:
                        reasons.append(f"Unusually low win rate: {metrics['win_rate']:.2f}")
                    
                    if metrics['profit_factor'] > 3.0:
                        reasons.append(f"Unusually high profit factor: {metrics['profit_factor']:.2f}")
                    elif metrics['profit_factor'] < 0.5:
                        reasons.append(f"Unusually low profit factor: {metrics['profit_factor']:.2f}")
                    
                    outlier = OutlierDetection(
                        user_id=user_id,
                        username=profile.username,
                        outlier_type=OutlierType.PERFORMANCE,
                        severity=severity,
                        z_score=abs(z_score),
                        p_value=p_value,
                        confidence=min(abs(z_score) / 3.0, 1.0),
                        reasons=reasons,
                        metrics={'performance_score': score, 'win_rate': metrics['win_rate']},
                        detected_at=datetime.utcnow()
                    )
                    
                    outliers.append(outlier)
        
        return outliers
    
    def _detect_frequency_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                 user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect frequency outliers"""
        outliers = []
        
        frequencies = [m['trading_frequency'] for m in user_metrics.values()]
        
        if len(frequencies) < 10:
            return outliers
        
        # Calculate IQR
        q1 = np.percentile(frequencies, 25)
        q3 = np.percentile(frequencies, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            frequency = metrics['trading_frequency']
            
            if frequency < lower_bound or frequency > upper_bound:
                # Calculate z-score
                mean_freq = np.mean(frequencies)
                std_freq = np.std(frequencies)
                
                if std_freq > 0:
                    z_score = (frequency - mean_freq) / std_freq
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    
                    threshold = self.detection_thresholds[OutlierType.FREQUENCY]
                    
                    if abs(z_score) > threshold['z_score'] or p_value < threshold['p_value']:
                        severity = self._determine_severity(abs(z_score))
                        reasons = []
                        
                        if frequency > upper_bound:
                            reasons.append(f"Unusually high trading frequency: {frequency:.1f} trades/day")
                        else:
                            reasons.append(f"Unusually low trading frequency: {frequency:.1f} trades/day")
                        
                        if metrics['total_trades'] > 1000:
                            reasons.append(f"Very high total trades: {metrics['total_trades']}")
                        elif metrics['total_trades'] < 10:
                            reasons.append(f"Very low total trades: {metrics['total_trades']}")
                        
                        outlier = OutlierDetection(
                            user_id=user_id,
                            username=profile.username,
                            outlier_type=OutlierType.FREQUENCY,
                            severity=severity,
                            z_score=abs(z_score),
                            p_value=p_value,
                            confidence=min(abs(z_score) / 2.5, 1.0),
                            reasons=reasons,
                            metrics={'trading_frequency': frequency, 'total_trades': metrics['total_trades']},
                            detected_at=datetime.utcnow()
                        )
                        
                        outliers.append(outlier)
        
        return outliers
    
    def _detect_risk_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                             user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect risk outliers"""
        outliers = []
        
        risk_ratios = [m['risk_reward_ratio'] for m in user_metrics.values() if m['risk_reward_ratio'] > 0]
        
        if len(risk_ratios) < 10:
            return outliers
        
        # Calculate z-scores
        mean_ratio = np.mean(risk_ratios)
        std_ratio = np.std(risk_ratios)
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            risk_ratio = metrics['risk_reward_ratio']
            
            if risk_ratio > 0 and std_ratio > 0:
                z_score = (risk_ratio - mean_ratio) / std_ratio
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                threshold = self.detection_thresholds[OutlierType.RISK]
                
                if abs(z_score) > threshold['z_score'] or p_value < threshold['p_value']:
                    severity = self._determine_severity(abs(z_score))
                    reasons = []
                    
                    if risk_ratio > 5.0:
                        reasons.append(f"Extremely high risk-reward ratio: {risk_ratio:.2f}")
                    elif risk_ratio < 0.5:
                        reasons.append(f"Extremely low risk-reward ratio: {risk_ratio:.2f}")
                    
                    if metrics['max_drawdown'] > 0.5:
                        reasons.append(f"Very high max drawdown: {metrics['max_drawdown']:.2f}")
                    
                    if metrics['avg_position_size'] > 0.1:
                        reasons.append(f"Very large position sizes: {metrics['avg_position_size']:.2f}")
                    
                    outlier = OutlierDetection(
                        user_id=user_id,
                        username=profile.username,
                        outlier_type=OutlierType.RISK,
                        severity=severity,
                        z_score=abs(z_score),
                        p_value=p_value,
                        confidence=min(abs(z_score) / 2.0, 1.0),
                        reasons=reasons,
                        metrics={'risk_reward_ratio': risk_ratio, 'max_drawdown': metrics['max_drawdown']},
                        detected_at=datetime.utcnow()
                    )
                    
                    outliers.append(outlier)
        
        return outliers
    
    def _detect_consistency_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                    user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect consistency outliers"""
        outliers = []
        
        consistency_scores = [m['consistency_score'] for m in user_metrics.values()]
        
        if len(consistency_scores) < 10:
            return outliers
        
        # Calculate z-scores
        mean_consistency = np.mean(consistency_scores)
        std_consistency = np.std(consistency_scores)
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            consistency = metrics['consistency_score']
            
            if std_consistency > 0:
                z_score = (consistency - mean_consistency) / std_consistency
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                threshold = self.detection_thresholds[OutlierType.CONSISTENCY]
                
                if abs(z_score) > threshold['z_score'] or p_value < threshold['p_value']:
                    severity = self._determine_severity(abs(z_score))
                    reasons = []
                    
                    if consistency > 0.9:
                        reasons.append(f"Unusually high consistency: {consistency:.2f}")
                    elif consistency < 0.3:
                        reasons.append(f"Unusually low consistency: {consistency:.2f}")
                    
                    if metrics['sharpe_ratio'] > 3.0:
                        reasons.append(f"Very high Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                    elif metrics['sharpe_ratio'] < -1.0:
                        reasons.append(f"Very low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                    
                    outlier = OutlierDetection(
                        user_id=user_id,
                        username=profile.username,
                        outlier_type=OutlierType.CONSISTENCY,
                        severity=severity,
                        z_score=abs(z_score),
                        p_value=p_value,
                        confidence=min(abs(z_score) / 2.5, 1.0),
                        reasons=reasons,
                        metrics={'consistency_score': consistency, 'sharpe_ratio': metrics['sharpe_ratio']},
                        detected_at=datetime.utcnow()
                    )
                    
                    outliers.append(outlier)
        
        return outliers
    
    def _detect_behavioral_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                  user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect behavioral outliers"""
        outliers = []
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            reasons = []
            z_scores = []
            
            # Check for unusual manual exit rate
            manual_exit_rate = metrics['manual_exit_rate']
            if manual_exit_rate > 0.8:
                reasons.append(f"Very high manual exit rate: {manual_exit_rate:.2f}")
                z_scores.append(3.0)
            elif manual_exit_rate < 0.1:
                reasons.append(f"Very low manual exit rate: {manual_exit_rate:.2f}")
                z_scores.append(2.5)
            
            # Check for unusual stop loss hit rate
            sl_hit_rate = metrics['sl_hit_rate']
            if sl_hit_rate > 0.8:
                reasons.append(f"Very high stop loss hit rate: {sl_hit_rate:.2f}")
                z_scores.append(2.0)
            elif sl_hit_rate < 0.1:
                reasons.append(f"Very low stop loss hit rate: {sl_hit_rate:.2f}")
                z_scores.append(2.5)
            
            # Check for unusual take profit hit rate
            tp_hit_rate = metrics['tp_hit_rate']
            if tp_hit_rate > 0.8:
                reasons.append(f"Very high take profit hit rate: {tp_hit_rate:.2f}")
                z_scores.append(2.0)
            elif tp_hit_rate < 0.1:
                reasons.append(f"Very low take profit hit rate: {tp_hit_rate:.2f}")
                z_scores.append(2.5)
            
            # Check for unusual confidence levels
            avg_confidence = metrics['avg_confidence']
            if avg_confidence > 0.95:
                reasons.append(f"Unusually high confidence: {avg_confidence:.2f}")
                z_scores.append(2.0)
            elif avg_confidence < 0.3:
                reasons.append(f"Unusually low confidence: {avg_confidence:.2f}")
                z_scores.append(2.0)
            
            if reasons and z_scores:
                max_z_score = max(z_scores)
                p_value = 2 * (1 - stats.norm.cdf(max_z_score))
                
                threshold = self.detection_thresholds[OutlierType.BEHAVIOR]
                
                if max_z_score > threshold['z_score'] or p_value < threshold['p_value']:
                    severity = self._determine_severity(max_z_score)
                    
                    outlier = OutlierDetection(
                        user_id=user_id,
                        username=profile.username,
                        outlier_type=OutlierType.BEHAVIOR,
                        severity=severity,
                        z_score=max_z_score,
                        p_value=p_value,
                        confidence=min(max_z_score / 3.0, 1.0),
                        reasons=reasons,
                        metrics={
                            'manual_exit_rate': manual_exit_rate,
                            'sl_hit_rate': sl_hit_rate,
                            'tp_hit_rate': tp_hit_rate,
                            'avg_confidence': avg_confidence
                        },
                        detected_at=datetime.utcnow()
                    )
                    
                    outliers.append(outlier)
        
        return outliers
    
    def _detect_manipulation_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                     user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect potential manipulation outliers"""
        outliers = []
        
        for user_id, profile in user_profiles.items():
            metrics = user_metrics[user_id]
            reasons = []
            z_scores = []
            
            # Check for perfect win rate
            if metrics['win_rate'] >= 1.0 and metrics['total_trades'] > 10:
                reasons.append(f"Perfect win rate with {metrics['total_trades']} trades")
                z_scores.append(4.0)
            
            # Check for impossible performance
            if metrics['sharpe_ratio'] > 10.0:
                reasons.append(f"Unrealistically high Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                z_scores.append(3.5)
            
            # Check for zero drawdown with many trades
            if metrics['max_drawdown'] == 0.0 and metrics['total_trades'] > 50:
                reasons.append(f"Zero drawdown with {metrics['total_trades']} trades")
                z_scores.append(3.0)
            
            # Check for suspiciously consistent performance
            if metrics['consistency_score'] > 0.95 and metrics['total_trades'] > 100:
                reasons.append(f"Suspiciously consistent performance: {metrics['consistency_score']:.2f}")
                z_scores.append(3.0)
            
            # Check for unusual trading patterns
            if metrics['trading_frequency'] > 100 and metrics['win_rate'] > 0.9:
                reasons.append(f"High frequency with perfect performance")
                z_scores.append(3.5)
            
            if reasons and z_scores:
                max_z_score = max(z_scores)
                p_value = 2 * (1 - stats.norm.cdf(max_z_score))
                
                threshold = self.detection_thresholds[OutlierType.MANIPULATION]
                
                if max_z_score > threshold['z_score'] or p_value < threshold['p_value']:
                    severity = OutlierSeverity.CRITICAL
                    
                    outlier = OutlierDetection(
                        user_id=user_id,
                        username=profile.username,
                        outlier_type=OutlierType.MANIPULATION,
                        severity=severity,
                        z_score=max_z_score,
                        p_value=p_value,
                        confidence=min(max_z_score / 4.0, 1.0),
                        reasons=reasons,
                        metrics={
                            'win_rate': metrics['win_rate'],
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'max_drawdown': metrics['max_drawdown'],
                            'consistency_score': metrics['consistency_score'],
                            'trading_frequency': metrics['trading_frequency']
                        },
                        detected_at=datetime.utcnow()
                    )
                    
                    outliers.append(outlier)
        
        return outliers
    
    def _detect_statistical_outliers(self, user_profiles: Dict[int, UserPerformanceProfile], 
                                    user_metrics: Dict[int, Dict[str, float]]) -> List[OutlierDetection]:
        """Detect statistical outliers using machine learning"""
        outliers = []
        
        # Prepare feature matrix
        features = []
        user_ids = []
        
        for user_id, metrics in user_metrics.items():
            feature_vector = [
                metrics['performance_score'],
                metrics['win_rate'],
                metrics['profit_factor'],
                metrics['sharpe_ratio'],
                metrics['max_drawdown'],
                metrics['consistency_score'],
                metrics['trading_frequency'],
                metrics['risk_reward_ratio'],
                metrics['influence_score']
            ]
            features.append(feature_vector)
            user_ids.append(user_id)
        
        if len(features) < 20:
            return outliers
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit isolation forest
        outlier_labels = self.isolation_forest.fit_predict(features_scaled)
        outlier_scores = self.isolation_forest.decision_function(features_scaled)
        
        for i, (user_id, label, score) in enumerate(zip(user_ids, outlier_labels, outlier_scores)):
            if label == -1:  # Outlier
                profile = user_profiles[user_id]
                metrics = user_metrics[user_id]
                
                # Calculate confidence
                confidence = min(abs(score) / 0.5, 1.0)
                
                # Determine severity
                if confidence > 0.8:
                    severity = OutlierSeverity.HIGH
                elif confidence > 0.6:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW
                
                reasons = [
                    f"Statistical anomaly detected (score: {score:.3f})",
                    f"Multiple metrics deviate from normal distribution"
                ]
                
                outlier = OutlierDetection(
                    user_id=user_id,
                    username=profile.username,
                    outlier_type=OutlierType.ANOMALY,
                    severity=severity,
                    z_score=abs(score),
                    p_value=2 * (1 - stats.norm.cdf(abs(score))),
                    confidence=confidence,
                    reasons=reasons,
                    metrics={
                        'performance_score': metrics['performance_score'],
                        'win_rate': metrics['win_rate'],
                        'statistical_score': score
                    },
                    detected_at=datetime.utcnow()
                )
                
                outliers.append(outlier)
        
        return outliers
    
    def _determine_severity(self, z_score: float) -> OutlierSeverity:
        """Determine outlier severity based on z-score"""
        if z_score > 4.0:
            return OutlierSeverity.CRITICAL
        elif z_score > 3.0:
            return OutlierSeverity.HIGH
        elif z_score > 2.0:
            return OutlierSeverity.MEDIUM
        else:
            return OutlierSeverity.LOW
    
    def get_outlier_summary(self) -> Dict[str, Any]:
        """Get outlier detection summary"""
        total_outliers = len(self.active_outliers)
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for outlier in self.active_outliers.values():
            severity_counts[outlier.severity.value] += 1
            type_counts[outlier.outlier_type.value] += 1
        
        return {
            'total_outliers': total_outliers,
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'detection_history_size': sum(len(history) for history in self.detection_history.values()),
            'most_detected_users': sorted(self.outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_user_outlier_history(self, user_id: int) -> List[OutlierDetection]:
        """Get outlier detection history for a user"""
        return self.detection_history.get(user_id, [])
    
    def is_user_outlier(self, user_id: int) -> bool:
        """Check if user is currently an outlier"""
        return user_id in self.active_outliers
    
    def get_user_outlier_score(self, user_id: int) -> float:
        """Get user's outlier score"""
        return self.outlier_scores.get(user_id, 0.0)
    
    def remove_outlier(self, user_id: int) -> bool:
        """Remove user from active outliers"""
        if user_id in self.active_outliers:
            del self.active_outliers[user_id]
            logger.system_logger.info(f"Removed user {user_id} from active outliers")
            return True
        return False
    
    def get_outlier_recommendations(self, user_id: int) -> Dict[str, Any]:
        """Get recommendations for outlier users"""
        if user_id not in self.active_outliers:
            return {}
        
        outlier = self.active_outliers[user_id]
        recommendations = {}
        
        if outlier.outlier_type == OutlierType.PERFORMANCE:
            if outlier.z_score > 0:
                recommendations['action'] = 'investigate_high_performance'
                recommendations['reason'] = 'User shows exceptionally high performance - verify authenticity'
            else:
                recommendations['action'] = 'provide_training'
                recommendations['reason'] = 'User shows exceptionally low performance - provide education'
        
        elif outlier.outlier_type == OutlierType.MANIPULATION:
            recommendations['action'] = 'investigate_manipulation'
            recommendations['reason'] = 'Potential manipulation detected - investigate thoroughly'
            recommendations['priority'] = 'high'
        
        elif outlier.outlier_type == OutlierType.BEHAVIOR:
            recommendations['action'] = 'behavioral_analysis'
            recommendations['reason'] = 'Unusual trading patterns detected - analyze behavior'
        
        return recommendations


# Factory function
def create_outlier_detector(z_threshold: float = 2.5, iqr_multiplier: float = 1.5, 
                          contamination_rate: float = 0.1, min_samples_for_detection: int = 20) -> OutlierDetector:
    """Create and return outlier detector instance"""
    return OutlierDetector(z_threshold, iqr_multiplier, contamination_rate, min_samples_for_detection)
