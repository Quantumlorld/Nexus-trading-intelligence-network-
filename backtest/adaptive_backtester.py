"""
Adaptive Backtester - Integrates AI learning with strategy testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json

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

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveBacktestConfig:
    """Configuration for adaptive backtesting"""
    base_config: BacktestConfig
    enable_adaptive_learning: bool = True
    enable_outlier_detection: bool = True
    enable_weight_optimization: bool = True
    performance_window_days: int = 30
    min_trades_for_adaptation: int = 10
    adaptation_frequency: str = "daily"  # daily, weekly, monthly
    
class AdaptiveBacktester:
    """Enhanced backtester with adaptive AI integration"""
    
    def __init__(self, config: AdaptiveBacktestConfig):
        self.config = config
        self.base_engine = BacktestEngine(config.base_config)
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize adaptive components
        self.adaptive_engine = create_adaptive_layer()
        self.performance_tracker = create_performance_tracker()
        self.outlier_detector = create_outlier_detector()
        self.weight_manager = create_strategy_weight_manager()
        
        # Backtest state
        self.current_results = []
        self.adaptation_history = []
        self.performance_history = []
        
        logger.info("Adaptive Backtester initialized with AI components")
    
    def run_adaptive_backtest(self, 
                            strategy_func: callable,
                            data: pd.DataFrame,
                            initial_capital: float = 10000) -> Dict[str, Any]:
        """Run backtest with adaptive learning"""
        
        logger.info("Starting adaptive backtest...")
        
        # Initialize results tracking
        backtest_results = {
            "base_results": None,
            "adaptive_results": [],
            "performance_metrics": {},
            "adaptation_history": [],
            "final_weights": {},
            "improvement_analysis": {}
        }
        
        # Run base backtest for comparison
        logger.info("Running base backtest for comparison...")
        
        # Create a simple strategy wrapper for testing
        class TestStrategy:
            def __init__(self, strategy_func):
                self.strategy_func = strategy_func
                self.name = "TestStrategy"
            
            def generate_signals(self, data):
                return self.strategy_func(data)
        
        test_strategy = TestStrategy(strategy_func)
        base_result = self.base_engine.run_backtest([test_strategy], {"EURUSD": data})
        backtest_results["base_results"] = base_result
        
        # Run adaptive backtest
        if self.config.enable_adaptive_learning:
            logger.info("Running adaptive backtest with AI learning...")
            adaptive_results = self._run_adaptive_strategy(strategy_func, data, initial_capital)
            backtest_results["adaptive_results"] = adaptive_results
        else:
            logger.info("Adaptive learning disabled, using base results only")
            backtest_results["adaptive_results"] = [base_result]
        
        # Calculate performance metrics
        backtest_results["performance_metrics"] = self._calculate_comprehensive_metrics(
            backtest_results["base_results"],
            backtest_results["adaptive_results"]
        )
        
        # Generate improvement analysis
        backtest_results["improvement_analysis"] = self._analyze_improvements(
            backtest_results["base_results"],
            backtest_results["adaptive_results"]
        )
        
        # Get final strategy weights
        backtest_results["final_weights"] = self.weight_manager.get_current_weights()
        
        logger.info("Adaptive backtest completed successfully")
        return backtest_results
    
    def _run_adaptive_strategy(self, 
                             strategy_func: callable,
                             data: pd.DataFrame,
                             initial_capital: float) -> List[BacktestResult]:
        """Run strategy with adaptive learning"""
        
        results = []
        current_capital = initial_capital
        trades = []
        
        # Split data into chunks for adaptation
        chunk_size = self._calculate_adaptation_chunk_size(data)
        data_chunks = self._split_data_for_adaptation(data, chunk_size)
        
        for i, chunk in enumerate(data_chunks):
            logger.info(f"Processing chunk {i+1}/{len(data_chunks)} with {len(chunk)} periods")
            
            # Run strategy on current chunk
            test_strategy = TestStrategy(strategy_func)
            chunk_result = self.base_engine.run_backtest([test_strategy], {"EURUSD": chunk})
            results.append(chunk_result)
            
            # Collect trades for learning
            trades.extend(chunk_result.trades)
            
            # Adapt strategy based on performance
            if len(trades) >= self.config.min_trades_for_adaptation:
                self._adapt_strategy(trades[-self.config.min_trades_for_adaptation:])
                current_capital = chunk_result.final_capital
        
        return results
    
    def _adapt_strategy(self, recent_trades: List[Dict[str, Any]]):
        """Adapt strategy based on recent performance"""
        
        logger.info(f"Adapting strategy based on {len(recent_trades)} recent trades")
        
        # Update performance tracker
        self.performance_tracker.update_performance(recent_trades)
        
        # Detect outliers
        if self.config.enable_outlier_detection:
            outliers = self.outlier_detector.detect_outliers(recent_trades)
            if outliers:
                logger.warning(f"Detected {len(outliers)} outliers in recent trades")
                self._handle_outliers(outliers)
        
        # Update strategy weights
        if self.config.enable_weight_optimization:
            performance_metrics = self.performance_tracker.get_current_metrics()
            self.weight_manager.update_weights(performance_metrics)
            
            # Record adaptation
            adaptation_record = {
                "timestamp": datetime.utcnow(),
                "performance_metrics": performance_metrics,
                "new_weights": self.weight_manager.get_current_weights(),
                "outliers_detected": len(outliers) if self.config.enable_outlier_detection else 0
            }
            self.adaptation_history.append(adaptation_record)
        
        logger.info("Strategy adaptation completed")
    
    def _handle_outliers(self, outliers: List[Dict[str, Any]]):
        """Handle detected outliers"""
        
        for outlier in outliers:
            if outlier["severity"] in ["high", "critical"]:
                logger.warning(f"Critical outlier detected: {outlier}")
                # Implement outlier handling logic
                # Could adjust strategy parameters, reduce position size, etc.
    
    def _calculate_comprehensive_metrics(self, 
                                       base_result: BacktestResult,
                                       adaptive_results: List[BacktestResult]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {
            "base_metrics": self.metrics_calculator.calculate_all_metrics(base_result),
            "adaptive_metrics": [],
            "improvement_scores": {}
        }
        
        # Calculate metrics for each adaptive result
        for result in adaptive_results:
            metrics["adaptive_metrics"].append(
                self.metrics_calculator.calculate_all_metrics(result)
            )
        
        # Calculate improvement scores
        if adaptive_results:
            best_adaptive = max(adaptive_results, key=lambda x: x.total_return)
            base_return = base_result.total_return
            adaptive_return = best_adaptive.total_return
            
            metrics["improvement_scores"] = {
                "return_improvement": ((adaptive_return - base_return) / abs(base_return)) * 100,
                "sharpe_improvement": best_adaptive.sharpe_ratio - base_result.sharpe_ratio,
                "max_drawdown_improvement": base_result.max_drawdown - best_adaptive.max_drawdown,
                "win_rate_improvement": best_adaptive.win_rate - base_result.win_rate
            }
        
        return metrics
    
    def _analyze_improvements(self, 
                            base_result: BacktestResult,
                            adaptive_results: List[BacktestResult]) -> Dict[str, Any]:
        """Analyze improvements from adaptive learning"""
        
        analysis = {
            "summary": "",
            "key_improvements": [],
            "areas_for_improvement": [],
            "adaptation_effectiveness": 0.0
        }
        
        if not adaptive_results:
            analysis["summary"] = "No adaptive results to analyze"
            return analysis
        
        # Find best adaptive result
        best_adaptive = max(adaptive_results, key=lambda x: x.total_return)
        
        # Calculate improvements
        return_improvement = ((best_adaptive.total_return - base_result.total_return) / 
                            abs(base_result.total_return)) * 100
        
        analysis["adaptation_effectiveness"] = max(0, return_improvement)
        
        # Generate summary
        if return_improvement > 5:
            analysis["summary"] = f"Adaptive learning improved returns by {return_improvement:.1f}%"
            analysis["key_improvements"].append(f"Return improvement: {return_improvement:.1f}%")
        elif return_improvement > 0:
            analysis["summary"] = f"Adaptive learning provided modest improvement of {return_improvement:.1f}%"
        else:
            analysis["summary"] = f"Adaptive learning did not improve performance (-{abs(return_improvement):.1f}%)"
            analysis["areas_for_improvement"].append("Consider adjusting adaptation parameters")
        
        # Analyze other metrics
        if best_adaptive.sharpe_ratio > base_result.sharpe_ratio:
            sharpe_improvement = best_adaptive.sharpe_ratio - base_result.sharpe_ratio
            analysis["key_improvements"].append(f"Sharpe ratio improvement: {sharpe_improvement:.2f}")
        
        if best_adaptive.max_drawdown < base_result.max_drawdown:
            drawdown_improvement = base_result.max_drawdown - best_adaptive.max_drawdown
            analysis["key_improvements"].append(f"Max drawdown reduction: {drawdown_improvement:.1f}%")
        
        return analysis
    
    def _calculate_adaptation_chunk_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal chunk size for adaptation"""
        
        total_periods = len(data)
        
        if self.config.adaptation_frequency == "daily":
            return max(1, total_periods // 252)  # Approximate trading days
        elif self.config.adaptation_frequency == "weekly":
            return max(1, total_periods // 52)   # Approximate weeks
        elif self.config.adaptation_frequency == "monthly":
            return max(1, total_periods // 12)   # Approximate months
        else:
            return max(1, total_periods // 100)  # Default chunk size
    
    def _split_data_for_adaptation(self, data: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split data into chunks for adaptive learning"""
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
    
    def generate_report(self, backtest_results: Dict[str, Any]) -> str:
        """Generate comprehensive backtest report"""
        
        report = []
        report.append("=" * 80)
        report.append("🤖 ADAPTIVE BACKTEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Base results
        base_result = backtest_results["base_results"]
        report.append("📊 BASE STRATEGY RESULTS:")
        report.append(f"   Total Return: {base_result.total_return:.2%}")
        report.append(f"   Sharpe Ratio: {base_result.sharpe_ratio:.2f}")
        report.append(f"   Max Drawdown: {base_result.max_drawdown:.2%}")
        report.append(f"   Win Rate: {base_result.win_rate:.2%}")
        report.append(f"   Total Trades: {len(base_result.trades)}")
        report.append("")
        
        # Adaptive results
        if backtest_results["adaptive_results"]:
            best_adaptive = max(backtest_results["adaptive_results"], key=lambda x: x.total_return)
            report.append("🤖 BEST ADAPTIVE STRATEGY RESULTS:")
            report.append(f"   Total Return: {best_adaptive.total_return:.2%}")
            report.append(f"   Sharpe Ratio: {best_adaptive.sharpe_ratio:.2f}")
            report.append(f"   Max Drawdown: {best_adaptive.max_drawdown:.2%}")
            report.append(f"   Win Rate: {best_adaptive.win_rate:.2%}")
            report.append(f"   Total Trades: {len(best_adaptive.trades)}")
            report.append("")
        
        # Improvement analysis
        improvement = backtest_results["improvement_analysis"]
        report.append("📈 IMPROVEMENT ANALYSIS:")
        report.append(f"   {improvement['summary']}")
        
        if improvement["key_improvements"]:
            report.append("   Key Improvements:")
            for imp in improvement["key_improvements"]:
                report.append(f"     • {imp}")
        
        if improvement["areas_for_improvement"]:
            report.append("   Areas for Improvement:")
            for area in improvement["areas_for_improvement"]:
                report.append(f"     • {area}")
        
        report.append("")
        
        # Performance metrics
        metrics = backtest_results["performance_metrics"]
        if "improvement_scores" in metrics:
            scores = metrics["improvement_scores"]
            report.append("🎯 PERFORMANCE SCORES:")
            report.append(f"   Return Improvement: {scores.get('return_improvement', 0):.1f}%")
            report.append(f"   Sharpe Improvement: {scores.get('sharpe_improvement', 0):.2f}")
            report.append(f"   Drawdown Improvement: {scores.get('max_drawdown_improvement', 0):.1f}%")
            report.append(f"   Win Rate Improvement: {scores.get('win_rate_improvement', 0):.1f}%")
            report.append("")
        
        # Final weights
        if backtest_results["final_weights"]:
            report.append("⚖️ FINAL STRATEGY WEIGHTS:")
            for strategy, weight in backtest_results["final_weights"].items():
                report.append(f"   {strategy}: {weight:.3f}")
            report.append("")
        
        # Adaptation history
        if self.adaptation_history:
            report.append("🔄 ADAPTATION HISTORY:")
            report.append(f"   Total Adaptations: {len(self.adaptation_history)}")
            report.append(f"   Performance Window: {self.config.performance_window_days} days")
            report.append(f"   Min Trades for Adaptation: {self.config.min_trades_for_adaptation}")
            report.append("")
        
        report.append("=" * 80)
        report.append("🎉 ADAPTIVE BACKTEST COMPLETED")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, backtest_results: Dict[str, Any], filepath: str):
        """Save backtest results to file"""
        
        results_data = {
            "config": {
                "enable_adaptive_learning": self.config.enable_adaptive_learning,
                "enable_outlier_detection": self.config.enable_outlier_detection,
                "enable_weight_optimization": self.config.enable_weight_optimization,
                "performance_window_days": self.config.performance_window_days,
                "min_trades_for_adaptation": self.config.min_trades_for_adaptation,
                "adaptation_frequency": self.config.adaptation_frequency
            },
            "base_results": backtest_results["base_results"].__dict__,
            "adaptive_results": [result.__dict__ for result in backtest_results["adaptive_results"]],
            "performance_metrics": backtest_results["performance_metrics"],
            "improvement_analysis": backtest_results["improvement_analysis"],
            "final_weights": backtest_results["final_weights"],
            "adaptation_history": self.adaptation_history,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {filepath}")

def create_adaptive_backtester(config: AdaptiveBacktestConfig) -> AdaptiveBacktester:
    """Create adaptive backtester instance"""
    return AdaptiveBacktester(config)
