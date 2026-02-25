"""
Nexus Trading System - Adaptive AI Engine
Hybrid adaptive learning system that learns from user performance
and adjusts strategy weights based on top performers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os

from database.session import get_database_session
from database.models import User, Trade, UserPerformance, AdaptiveWeight, Signal
from database.schemas import UserRole, UserStatus
from core.logger import get_logger
from strategy.base_strategy import BaseStrategy, TradingSignal, SignalType
from core.trade_manager import TradeManager
from core.risk_engine import RiskEngine
from core.position_sizer import PositionSizer
from core.regime_detector import RegimeDetector
from data.loaders import DataManager

logger = get_logger()


class PerformanceMetrics:
    """Performance metrics for user evaluation"""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.profit_factor = 0.0
        self.avg_trade_duration = 0.0
        self.risk_reward_ratio = 0.0
        self.consistency_score = 0.0
        self.performance_score = 0.0
        self.last_updated = datetime.utcnow()
        self.avg_win = 0.0
        self.avg_loss = 0.0
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]) -> None:
        """Calculate performance metrics from trade data"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        self.winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        self.losing_trades = len([t for t in trades if t.get('pnl', 0) <= 0])
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            self.total_pnl = sum(t.get('pnl', 0) for t in trades)
            self.avg_trade_duration = np.mean([t.get('duration_minutes', 0) for t in trades])
            
            # Calculate drawdown
            equity_curve = np.cumsum([t.get('pnl', 0) for t in trades])
            peak = np.maximum(equity_curve)
            drawdown = (peak - equity_curve.min()) / peak if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Calculate Sharpe ratio
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve)
                excess_returns = returns[returns > 0]
                if len(excess_returns) > 0:
                    self.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
                    self.sortino_ratio = np.mean(returns[returns < 0]) / np.std(returns[returns < 0]) * np.sqrt(252)  # Annualized
            
            # Calculate profit factor
            gross_profits = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
            gross_losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) <= 0]
            if gross_losses:
                self.profit_factor = abs(sum(gross_profits) / sum(gross_losses))
            
            # Calculate risk-reward ratio
            if self.winning_trades > 0:
                self.avg_win = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
            else:
                self.avg_win = 0.0
                
            if self.losing_trades > 0:
                self.avg_loss = abs(np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) <= 0]))
            else:
                self.avg_loss = 0.0
            
            self.risk_reward_ratio = self.avg_win / abs(self.avg_loss) if self.avg_loss != 0 else 0
            
            # Calculate consistency score
            if len(trades) > 10:
                returns_series = pd.Series([t.get('pnl', 0) for t in trades])
                rolling_returns = returns_series.rolling(window=10)
                consistency = (rolling_returns > 0).mean()
                self.consistency_score = consistency
            else:
                self.consistency_score = 0.5  # Default for insufficient data
            
            # Calculate performance score (weighted combination)
            self.performance_score = (
                0.4 * self.win_rate +
                0.3 * self.profit_factor +
                0.2 * self.sharpe_ratio +
                0.1 * self.consistency_score
            )
        
        self.last_updated = datetime.utcnow()


class OutlierDetector:
    """Detect and filter out outlier users"""
    
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
    
    def detect_outliers(self, user_metrics: Dict[int, PerformanceMetrics]) -> List[int]:
        """Detect outlier users based on performance metrics"""
        outlier_users = []
        
        if len(user_metrics) < 10:
            return outlier_users
        
        # Calculate z-scores for each metric
        win_rates = [m.win_rate for m in user_metrics.values()]
        profit_factors = [m.profit_factor for m in user_metrics.values()]
        sharpe_ratios = [m.sharpe_ratio for m in user_metrics.values()]
        
        # Detect outliers using z-score
        for user_id, metrics in user_metrics.items():
            z_scores = []
            
            if metrics.win_rate > 0:
                z_scores.append((metrics.win_rate - np.mean(win_rates)) / np.std(win_rates))
            if metrics.profit_factor > 0:
                z_scores.append((np.log(metrics.profit_factor) - np.mean(profit_factors)) / np.std(profit_factors))
            if metrics.sharpe_ratio > 0:
                z_scores.append((metrics.sharpe_ratio - np.mean(sharpe_ratios)) / np.std(sharpe_ratios))
            
            # User is outlier if any z-score exceeds threshold
            max_z = max(z_scores) if z_scores else 0
            if max_z > self.z_threshold:
                outlier_users.append(user_id)
        
        return outlier_users
    
    def remove_outliers(self, user_metrics: Dict[int, PerformanceMetrics]) -> Dict[int, PerformanceMetrics]:
        """Remove outlier users from metrics"""
        outlier_users = self.detect_outliers(user_metrics)
        clean_metrics = {uid: metrics for uid, metrics in user_metrics.items() if uid not in outlier_users}
        return clean_metrics


class StrategyWeightManager:
    """Manages strategy weights based on user performance"""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.strategy_weights = {}
        self.weight_history = defaultdict(list)
        self.performance_cache = {}
        self.last_update = datetime.utcnow()
    
    def update_weights(self, user_performance: Dict[int, PerformanceMetrics]) -> Dict[str, float]:
        """Update strategy weights based on user performance"""
        current_time = datetime.utcnow()
        
        # Clear old cache
        if (current_time - self.last_update).days > 7:
            self.performance_cache.clear()
            self.weight_history.clear()
        
        # Calculate performance scores for each strategy
        strategy_performance = defaultdict(list)
        
        for user_id, metrics in user_performance.items():
            # Get user's trade history
            with get_database_session() as db:
                trades = db.query(Trade).filter(Trade.user_id == user_id).all()
                
                # Group trades by strategy
                strategy_trades = defaultdict(list)
                for trade in trades:
                    strategy = trade.get('strategy', 'unknown')
                    strategy_trades[strategy].append(trade)
                
                # Calculate strategy-specific performance
                for strategy, trades in strategy_trades.items():
                    if len(trades) > 0:
                        strategy_metrics[strategy].append(PerformanceMetrics())
                        strategy_metrics[strategy].calculate_metrics(trades)
        
        # Update weights based on top performers
        for strategy, metrics_list in strategy_performance.items():
            if len(metrics_list) >= 3:  # Need at least 3 users
                avg_performance = np.mean([m.performance_score for m in metrics_list])
                self.strategy_weights[strategy] = avg_performance
                
                # Add to history
                self.weight_history[strategy].append(avg_performance)
        
        # Apply decay to old weights
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] *= self.decay_factor
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
        
        self.last_update = current_time
        
        return self.strategy_weights
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.strategy_weights.copy()
    
    def get_weight_history(self, strategy: str) -> List[float]:
        """Get weight history for a specific strategy"""
        return self.weight_history.get(strategy, [])


class AdaptiveEngine:
    """Main adaptive AI engine that learns from user performance"""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95, 
                 outlier_threshold: float = 2.5, retrain_interval_hours: int = 168):  # 1 week
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.outlier_threshold = outlier_threshold
        self.retrain_interval_hours = retrain_interval_hours
        self.weight_manager = StrategyWeightManager(learning_rate, decay_factor)
        self.outlier_detector = OutlierDetector(outlier_threshold)
        
        # Performance tracking
        self.user_performance = {}
        self.strategy_performance = defaultdict(list)
        self.adaptive_weights = {}
        self.last_retrain = datetime.utcnow()
        
        # Database connection
        self.db_session = None
        
        print("Adaptive AI Engine initialized")
    
    def set_database_session(self, db_session):
        """Set database session"""
        self.db_session = db_session
    
    def start_adaptive_learning(self):
        """Start the adaptive learning process"""
        print("Starting adaptive learning process")
        
        # Run in background
        asyncio.create_task(self._adaptive_learning_loop(), name="adaptive_learning")
    
    async def _adaptive_learning_loop(self):
        """Background task for adaptive learning"""
        while True:
            try:
                # Wait for next retrain interval
                await asyncio.sleep(self.retrain_interval_hours * 3600)  # Convert hours to seconds
                
                logger.info("Running adaptive learning cycle")
                
                # Collect user performance data
                await self._collect_performance_data()
                
                # Remove outliers
                clean_performance = self.outlier_detector.remove_outliers(self.user_performance)
                self.user_performance = clean_performance
                
                # Update strategy weights
                self.weight_manager.update_weights(self.user_performance)
                self.adaptive_weights = self.weight_manager.get_strategy_weights()
                
                # Update adaptive weights in database
                await self._save_adaptive_weights()
                
                # Update strategy performance cache
                for strategy, metrics_list in self.strategy_performance.items():
                    if len(metrics_list) >= 3:
                        avg_performance = np.mean([m.performance_score for m in metrics_list])
                        self.strategy_performance[strategy] = avg_performance
                
                logger.info(f"Adaptive learning cycle completed. Updated weights for {len(self.adaptive_weights)} strategies")
                
            except Exception as e:
                logger.error(f"Error in adaptive learning: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _collect_performance_data(self):
        """Collect user performance data from database"""
        with get_database_session() as db:
            # Get all users with trades
            users = db.query(User).all()
            
            for user in users:
                # Get user's trades
                trades = db.query(Trade).filter(Trade.user_id == user.id).all()
                
                if trades:
                    # Calculate user performance metrics
                    metrics = PerformanceMetrics()
                    metrics.calculate_metrics([{
                        'pnl': t.get('pnl', 0),
                        'duration_minutes': t.get('duration_minutes', 0),
                        'symbol': t.get('symbol', ''),
                        'strategy': t.get('strategy', ''),
                        'signal_confidence': t.get('signal_confidence', 0),
                        'entry_time': t.get('entry_time'),
                        'exit_time': t.get('exit_time')
                    } for t in trades])
                    
                    self.user_performance[user.id] = metrics
                    
                    # Update strategy performance
                    strategy = trades[0].get('strategy', 'unknown')
                    self.strategy_performance[strategy].append(metrics)
            
            logger.info(f"Collected performance data for {len(self.user_performance)} users")
    
    async def _save_adaptive_weights(self):
        """Save adaptive weights to database"""
        with get_database_session() as db:
            for strategy, weight in self.adaptive_weights.items():
                # Check if weight already exists
                existing = db.query(AdaptiveWeight).filter(
                    AdaptiveWeight.strategy_name == strategy,
                    AdaptiveWeight.asset == "ALL",
                    AdaptiveWeight.timeframe == "ALL"
                ).first()
                
                if existing:
                    # Update existing weight
                    existing.adaptive_weight = weight
                    existing.last_updated = datetime.utcnow()
                    db.merge(existing)
                else:
                    # Create new weight record
                    weight_record = AdaptiveWeight(
                        strategy_name=strategy,
                        asset="ALL",
                        timeframe="ALL",
                        base_weight=1.0,
                        adaptive_weight=weight,
                        final_weight=weight,
                        contributing_users=[uid for uid in self.user_performance.keys()],
                        performance_score=np.mean([m.performance_score for m in self.user_performance.values()]),
                        sample_size=len(self.user_performance),
                        calculation_period_start=datetime.utcnow() - timedelta(days=7),
                        calculation_period_end=datetime.utcnow(),
                        last_updated=datetime.utcnow()
                    )
                    db.add(weight_record)
            
            db.commit()
        
        logger.info(f"Saved {len(self.adaptive_weights)} adaptive weights to database")
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return self.weight_manager.get_strategy_weights()
    
    def get_strategy_performance(self, strategy: str) -> List[PerformanceMetrics]:
        """Get performance metrics for a specific strategy"""
        return self.strategy_performance.get(strategy, [])
    
    def get_user_performance(self, user_id: int) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific user"""
        return self.user_performance.get(user_id)
    
    def get_top_performers(self, limit: int = 10) -> List[Tuple[int, float]]:
        """Get top performing users"""
        sorted_users = sorted(
            [(uid, metrics.performance_score) for uid, metrics in self.user_performance.items()],
            key=lambda x: x[1], reverse=True
        )
        return sorted_users[:limit]
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """Get strategy ranking by performance"""
        return sorted(
            [(strategy, weight) for strategy, weight in self.adaptive_weights.items()],
            key=lambda x: x[1], reverse=True
        )
    
    def calculate_user_influence(self, user_id: int) -> float:
        """Calculate user influence score"""
        user_metrics = self.user_performance.get(user_id)
        if not user_metrics:
            return 0.0
        
        # Influence based on performance and trade frequency
        performance_score = user_metrics.performance_score
        trade_frequency = user_metrics.total_trades
        
        # Normalize trade frequency (0-100 trades)
        normalized_frequency = min(trade_frequency / 100, 1.0)
        
        # Calculate influence
        influence = performance_score * normalized_frequency
        return min(influence, 1.0)
    
    def get_market_regime_adjustments(self, current_regime: str) -> Dict[str, float]:
        """Get strategy weight adjustments based on market regime"""
        regime_adjustments = {}
        
        # Adjust weights based on regime performance
        regime_performance = defaultdict(float)
        
        for strategy, metrics_list in self.strategy_performance.items():
            if metrics_list:
                # Calculate average performance in this regime
                regime_performance[strategy] = np.mean([m.performance_score for m in metrics_list])
        
        # Normalize regime performance
        total_performance = sum(regime_performance.values())
        if total_performance > 0:
            for strategy, perf in regime_performance.items():
                regime_adjustments[strategy] = perf / total_performance
        
        return regime_adjustments
    
    def should_update_weights(self) -> bool:
        """Check if weights should be updated"""
        time_since_update = (datetime.utcnow() - self.last_retrain).total_seconds()
        return time_since_update >= self.retrain_interval_hours * 3600
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "total_users": len(self.user_performance),
            "active_strategies": len(self.adaptive_weights),
            "outlier_users": len(self.outlier_detector.detect_outliers(self.user_performance)),
            "learning_rate": self.learning_rate,
            "decay_factor": self.decay_factor,
            "outlier_threshold": self.outlier_threshold,
            "last_update": self.last_update.isoformat(),
            "retrain_interval_hours": self.retrain_interval_hours,
            "top_performers": self.get_top_performers(5),
            "strategy_ranking": self.get_strategy_ranking()
        }


class SignalOptimizer:
    """Optimizes signal generation based on adaptive weights"""
    
    def __init__(self, adaptive_engine: AdaptiveEngine):
        self.adaptive_engine = adaptive_engine
        self.signal_cache = {}
        self.optimization_history = []
    
    def optimize_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> TradingSignal:
        """Optimize a trading signal based on adaptive weights"""
        strategy_name = signal.strategy_name
        
        # Get adaptive weights for this strategy
        strategy_weights = self.adaptive_engine.get_strategy_weights()
        strategy_weight = strategy_weights.get(strategy_name, 1.0)
        
        # Get regime adjustments
        current_regime = market_data.get('regime', 'unknown')
        regime_adjustments = self.adaptive_engine.get_market_regime_adjustments(current_regime)
        regime_adjustment = regime_adjustments.get(strategy_name, 1.0)
        
        # Calculate optimized confidence
        base_confidence = signal.confidence
        adaptive_weight = strategy_weight * regime_adjustment
        optimized_confidence = min(base_confidence * adaptive_weight, 1.0)
        
        # Create optimized signal
        optimized_signal = TradingSignal(
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=optimized_confidence,
            strategy=signal.strategy_name,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            market_regime=current_regime,
            timestamp=signal.timestamp
        )
        
        # Cache the optimized signal
        signal_key = f"{signal.symbol}_{signal.direction}_{signal.strategy_name}_{signal.timestamp}"
        self.signal_cache[signal_key] = optimized_signal
        
        return optimized_signal
    
    def get_signal_performance(self, signal: TradingSignal) -> Dict[str, Any]:
        """Get performance data for a signal"""
        signal_key = f"{signal.symbol}_{signal.direction}_{signal.strategy_name}_{signal.timestamp}"
        
        if signal_key in self.signal_cache:
            cached_signal = self.signal_cache[signal_key]
            return {
                'original_confidence': signal.confidence,
                'optimized_confidence': cached_signal.confidence,
                'weight_adjustment': cached_signal.confidence / signal.confidence if signal.confidence > 0 else 1.0,
                'regime_adjustment': market_data.get('regime', 'unknown'),
                'strategy_weight': self.adaptive_engine.get_strategy_weights().get(signal.strategy_name, 1.0),
                'regime_adjustment': self.adaptive_engine.get_market_regime_adjustments(market_data.get('regime', 'unknown')).get(signal.strategy_name, 1.0)
            }
        
        return {
            'original_confidence': signal.confidence,
            'optimized_confidence': signal.confidence,
            'weight_adjustment': 1.0,
            'regime_adjustment': 1.0,
            'strategy_weight': 1.0
        }


class HybridAdaptiveLayer:
    """Main hybrid adaptive layer that integrates with the trading system"""
    
    def __init__(self, adaptive_engine: AdaptiveEngine, signal_optimizer: SignalOptimizer):
        self.adaptive_engine = adaptive_engine
        self.signal_optimizer = signal_optimizer
        self.is_enabled = True
        
        # Connect to trading system
        self.trade_manager = None
        self.risk_engine = None
        self.position_sizer = None
        self.regime_detector = None
        
        print("Hybrid Adaptive Layer initialized")
    
    def connect_to_trading_system(self, trade_manager, risk_engine, position_sizer, regime_detector):
        """Connect to trading system components"""
        self.trade_manager = trade_manager
        self.risk_engine = risk_engine
        self.position_sizer = position_sizer
        self.regime_detector = regime_detector
        
        print("Hybrid Adaptive Layer connected to trading system")
    
    def enable_adaptive_mode(self):
        """Enable adaptive learning mode"""
        self.is_enabled = True
        self.adaptive_engine.start_adaptive_learning()
        print("Adaptive learning mode enabled")
    
    def disable_adaptive_mode(self):
        """Disable adaptive learning mode"""
        self.is_enabled = False
        print("Adaptive learning mode disabled")
    
    def is_adaptive_mode_enabled(self) -> bool:
        """Check if adaptive mode is enabled"""
        return self.is_enabled
    
    def optimize_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> TradingSignal:
        """Optimize a signal using adaptive learning"""
        if not self.is_enabled:
            return signal
        
        return self.signal_optimizer.optimize_signal(signal, market_data)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.adaptive_engine.get_strategy_weights()
    
    def get_user_influence(self, user_id: int) -> float:
        """Get user influence score"""
        return self.adaptive_engine.calculate_user_influence(user_id)
    
    def get_top_performers(self, limit: int = 10) -> List[Tuple[int, float]]:
        """Get top performing users"""
        return self.adaptive_engine.get_top_performers(limit)
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """Get strategy ranking by performance"""
        return self.adaptive_engine.get_strategy_ranking()
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get adaptive learning statistics"""
        return self.adaptive_engine.get_learning_statistics()


# Factory function to create adaptive layer
def create_adaptive_layer(learning_rate: float = 0.1, decay_factor: float = 0.95, 
                         outlier_threshold: float = 2.5, retrain_interval_hours: int = 168) -> HybridAdaptiveLayer:
    """Create and return adaptive layer instance"""
    adaptive_engine = AdaptiveEngine(learning_rate, decay_factor, outlier_threshold, retrain_interval_hours)
    signal_optimizer = SignalOptimizer(adaptive_engine)
    
    return HybridAdaptiveLayer(adaptive_engine, signal_optimizer)


# Integration with main system
def integrate_adaptive_layer(trading_engine, adaptive_layer: HybridAdaptiveLayer):
    """Integrate adaptive layer with trading engine"""
    adaptive_layer.connect_to_trading_system(
        trading_engine.trade_manager,
        trading_engine.risk_engine,
        trading_engine.position_sizer,
        trading_engine.regime_detector
    )
    
    # Replace signal generation with adaptive optimization
    original_generate_signal = trading_engine.generate_signal
    trading_engine.generate_signal = lambda symbol, direction, confidence, strategy, market_data: adaptive_layer.optimize_signal(
        TradingSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strategy=strategy,
            entry_price=market_data.get('close'),
            stop_loss=None,
            take_profit=None,
            timestamp=datetime.utcnow()
        ),
        market_data
    )
    
    # Enable adaptive mode
    adaptive_layer.enable_adaptive_mode()
    
    print("Adaptive layer integrated with trading engine")
    
    return adaptive_layer
