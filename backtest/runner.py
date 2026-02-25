"""
Nexus Trading System - Backtest Runner
Handles dynamic TP extension and SL lock with comprehensive rule enforcement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import MetricsCalculator, PerformanceMetrics
from core.trade_manager import TradeManager, Position
from core.logger import get_logger
from core.position_sizer import PositionSizer
from core.regime_detector import RegimeDetector
from strategy.base_strategy import BaseStrategy
from data.loaders import DataManager


@dataclass
class RunnerConfig:
    """Configuration for backtest runner"""
    run_parallel: bool = True
    max_workers: int = 4
    save_intermediate_results: bool = True
    generate_reports: bool = True
    create_charts: bool = True
    validate_rules: bool = True
    log_level: str = "INFO"
    output_format: str = "json"  # json, csv, excel
    
    # Dynamic TP/SL settings
    enable_dynamic_management: bool = True
    lock_profit_threshold: float = 3.0  # $3
    extend_tp_threshold: float = 9.9   # $9.9
    extended_tp_target: float = 15.0   # $15
    runner_mode_enabled: bool = True
    momentum_threshold: float = 0.02
    
    # Risk enforcement settings
    enforce_daily_loss_limit: bool = True
    max_daily_loss: float = 9.99
    enforce_consecutive_losses: bool = True
    max_consecutive_losses: int = 3
    enforce_position_limits: bool = True
    max_positions_per_asset: int = 1


@dataclass
class BacktestJob:
    """Individual backtest job"""
    job_id: str
    strategies: List[BaseStrategy]
    market_data: Dict[str, pd.DataFrame]
    config: BacktestConfig
    priority: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[BacktestResult] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class DynamicPositionManager:
    """Manages dynamic TP/SL adjustments and runner mode"""
    
    def __init__(self, config: RunnerConfig):
        self.config = config
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Position tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # Dynamic management state
        self.locked_positions: Dict[str, float] = {}  # symbol -> locked SL price
        self.extended_positions: Dict[str, float] = {}  # symbol -> extended TP price
        self.runner_positions: Dict[str, Dict[str, Any]] = {}  # runner mode positions
        
    def register_position(self, position: Position, current_price: float):
        """Register a new position for dynamic management"""
        
        symbol = position.symbol
        
        self.active_positions[symbol] = {
            'position': position,
            'entry_price': position.entry_price,
            'current_price': current_price,
            'original_sl': position.sl_price,
            'original_tp': position.tp_price,
            'current_sl': position.sl_price,
            'current_tp': position.tp_price,
            'max_favorable_move': 0.0,
            'lock_threshold_hit': False,
            'extend_threshold_hit': False,
            'runner_mode_active': False,
            'adjustments': []
        }
        
        self.logger.debug(f"Position registered for dynamic management: {symbol}")
    
    def update_position(self, symbol: str, current_price: float, 
                       market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update position and check for dynamic adjustments"""
        
        if symbol not in self.active_positions:
            return None
        
        pos_info = self.active_positions[symbol]
        position = pos_info['position']
        pos_info['current_price'] = current_price
        
        # Calculate current P&L
        if position.direction == 'buy':
            pnl = (current_price - position.entry_price) * position.size
            favorable_move = current_price - position.entry_price
        else:
            pnl = (position.entry_price - current_price) * position.size
            favorable_move = position.entry_price - current_price
        
        # Track maximum favorable move
        pos_info['max_favorable_move'] = max(pos_info['max_favorable_move'], favorable_move)
        
        adjustments = []
        
        # Check for SL lock (lock profits at +$3)
        if self.config.enable_dynamic_management:
            lock_adjustment = self._check_profit_lock(symbol, pos_info, current_price)
            if lock_adjustment:
                adjustments.append(lock_adjustment)
        
        # Check for TP extension (extend to +$15 at +$9.9)
        if self.config.enable_dynamic_management:
            extend_adjustment = self._check_tp_extension(symbol, pos_info, current_price)
            if extend_adjustment:
                adjustments.append(extend_adjustment)
        
        # Check for runner mode (dynamic extension based on momentum)
        if self.config.runner_mode_enabled:
            runner_adjustment = self._check_runner_mode(symbol, pos_info, current_price, market_data)
            if runner_adjustment:
                adjustments.append(runner_adjustment)
        
        # Record adjustments
        if adjustments:
            pos_info['adjustments'].extend(adjustments)
            self.logger.info(f"Dynamic adjustments for {symbol}: {[adj['type'] for adj in adjustments]}")
        
        return {
            'symbol': symbol,
            'current_pnl': pnl,
            'adjustments': adjustments,
            'should_close': self._should_close_position(symbol, pos_info, current_price)
        }
    
    def _check_profit_lock(self, symbol: str, pos_info: Dict[str, Any], 
                          current_price: float) -> Optional[Dict[str, Any]]:
        """Check if we should lock profits at +$3"""
        
        if pos_info['lock_threshold_hit']:
            return None
        
        position = pos_info['position']
        point_value = self._get_point_value(symbol)
        lock_threshold = self.config.lock_profit_threshold
        lock_distance = lock_threshold / point_value
        
        if position.direction == 'buy':
            # Lock SL to +$3 when price moves +$3 from entry
            if current_price >= position.entry_price + lock_distance:
                new_sl = position.entry_price + lock_distance
                
                pos_info['current_sl'] = new_sl
                pos_info['lock_threshold_hit'] = True
                self.locked_positions[symbol] = new_sl
                
                adjustment = {
                    'type': 'profit_lock',
                    'timestamp': datetime.now(),
                    'old_sl': pos_info['original_sl'],
                    'new_sl': new_sl,
                    'reason': f"Lock profits at +${lock_threshold}"
                }
                
                self.logger.info(f"SL locked for {symbol}: {new_sl:.4f}")
                return adjustment
        
        else:  # sell position
            # Lock SL to +$3 when price moves +$3 from entry
            if current_price <= position.entry_price - lock_distance:
                new_sl = position.entry_price - lock_distance
                
                pos_info['current_sl'] = new_sl
                pos_info['lock_threshold_hit'] = True
                self.locked_positions[symbol] = new_sl
                
                adjustment = {
                    'type': 'profit_lock',
                    'timestamp': datetime.now(),
                    'old_sl': pos_info['original_sl'],
                    'new_sl': new_sl,
                    'reason': f"Lock profits at +${lock_threshold}"
                }
                
                self.logger.info(f"SL locked for {symbol}: {new_sl:.4f}")
                return adjustment
        
        return None
    
    def _check_tp_extension(self, symbol: str, pos_info: Dict[str, Any], 
                           current_price: float) -> Optional[Dict[str, Any]]:
        """Check if we should extend TP to +$15"""
        
        if pos_info['extend_threshold_hit']:
            return None
        
        position = pos_info['position']
        point_value = self._get_point_value(symbol)
        extend_threshold = self.config.extend_tp_threshold
        extended_target = self.config.extended_tp_target
        
        extend_distance = extend_threshold / point_value
        extended_distance = extended_target / point_value
        
        if position.direction == 'buy':
            # Extend TP to +$15 when price hits +$9.9
            if current_price >= position.entry_price + extend_distance:
                new_tp = position.entry_price + extended_distance
                
                pos_info['current_tp'] = new_tp
                pos_info['extend_threshold_hit'] = True
                self.extended_positions[symbol] = new_tp
                
                adjustment = {
                    'type': 'tp_extension',
                    'timestamp': datetime.now(),
                    'old_tp': pos_info['original_tp'],
                    'new_tp': new_tp,
                    'reason': f"Extend TP to +${extended_target}"
                }
                
                self.logger.info(f"TP extended for {symbol}: {new_tp:.4f}")
                return adjustment
        
        else:  # sell position
            # Extend TP to +$15 when price hits +$9.9
            if current_price <= position.entry_price - extend_distance:
                new_tp = position.entry_price - extended_distance
                
                pos_info['current_tp'] = new_tp
                pos_info['extend_threshold_hit'] = True
                self.extended_positions[symbol] = new_tp
                
                adjustment = {
                    'type': 'tp_extension',
                    'timestamp': datetime.now(),
                    'old_tp': pos_info['original_tp'],
                    'new_tp': new_tp,
                    'reason': f"Extend TP to +${extended_target}"
                }
                
                self.logger.info(f"TP extended for {symbol}: {new_tp:.4f}")
                return adjustment
        
        return None
    
    def _check_runner_mode(self, symbol: str, pos_info: Dict[str, Any], 
                          current_price: float, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if we should activate runner mode with dynamic TP extension"""
        
        if not pos_info['extend_threshold_hit']:
            return None
        
        position = pos_info['position']
        
        # Check for strong momentum
        momentum = self._calculate_momentum(symbol, market_data)
        
        if abs(momentum) < self.config.momentum_threshold:
            return None
        
        # Extend TP further based on momentum strength
        point_value = self._get_point_value(symbol)
        momentum_extension = abs(momentum) * 100 / point_value  # Convert momentum to price extension
        
        if position.direction == 'buy':
            if momentum > 0:  # Bullish momentum
                new_tp = pos_info['current_tp'] + momentum_extension
                
                pos_info['current_tp'] = new_tp
                pos_info['runner_mode_active'] = True
                self.runner_positions[symbol] = {
                    'active': True,
                    'momentum': momentum,
                    'extended_tp': new_tp
                }
                
                adjustment = {
                    'type': 'runner_mode',
                    'timestamp': datetime.now(),
                    'old_tp': pos_info['current_tp'],
                    'new_tp': new_tp,
                    'momentum': momentum,
                    'reason': f"Runner mode: momentum {momentum:.3f}"
                }
                
                self.logger.info(f"Runner mode activated for {symbol}: TP extended to {new_tp:.4f}")
                return adjustment
        
        else:  # sell position
            if momentum < 0:  # Bearish momentum
                new_tp = pos_info['current_tp'] - momentum_extension
                
                pos_info['current_tp'] = new_tp
                pos_info['runner_mode_active'] = True
                self.runner_positions[symbol] = {
                    'active': True,
                    'momentum': momentum,
                    'extended_tp': new_tp
                }
                
                adjustment = {
                    'type': 'runner_mode',
                    'timestamp': datetime.now(),
                    'old_tp': pos_info['current_tp'],
                    'new_tp': new_tp,
                    'momentum': momentum,
                    'reason': f"Runner mode: momentum {momentum:.3f}"
                }
                
                self.logger.info(f"Runner mode activated for {symbol}: TP extended to {new_tp:.4f}")
                return adjustment
        
        return None
    
    def _calculate_momentum(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate price momentum"""
        
        # Simple momentum calculation
        # In production, this would use more sophisticated indicators
        
        if 'price_history' not in market_data:
            return 0.0
        
        prices = market_data['price_history']
        
        if len(prices) < 10:
            return 0.0
        
        # Calculate momentum as rate of change
        recent_prices = prices[-10:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        return momentum
    
    def _should_close_position(self, symbol: str, pos_info: Dict[str, Any], 
                              current_price: float) -> bool:
        """Check if position should be closed based on SL/TP"""
        
        position = pos_info['position']
        current_sl = pos_info['current_sl']
        current_tp = pos_info['current_tp']
        
        if position.direction == 'buy':
            return current_price <= current_sl or current_price >= current_tp
        else:
            return current_price >= current_sl or current_price <= current_tp
    
    def close_position(self, symbol: str, close_price: float, reason: str):
        """Close position and record history"""
        
        if symbol not in self.active_positions:
            return
        
        pos_info = self.active_positions[symbol]
        position = pos_info['position']
        
        # Calculate final P&L
        if position.direction == 'buy':
            pnl = (close_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - close_price) * position.size
        
        # Record position history
        history_record = {
            'symbol': symbol,
            'position': position,
            'entry_price': position.entry_price,
            'exit_price': close_price,
            'pnl': pnl,
            'reason': reason,
            'original_sl': pos_info['original_sl'],
            'original_tp': pos_info['original_tp'],
            'final_sl': pos_info['current_sl'],
            'final_tp': pos_info['current_tp'],
            'max_favorable_move': pos_info['max_favorable_move'],
            'adjustments': pos_info['adjustments'],
            'lock_threshold_hit': pos_info['lock_threshold_hit'],
            'extend_threshold_hit': pos_info['extend_threshold_hit'],
            'runner_mode_active': pos_info['runner_mode_active'],
            'duration': datetime.now() - position.timestamp
        }
        
        self.position_history.append(history_record)
        
        # Clean up
        del self.active_positions[symbol]
        if symbol in self.locked_positions:
            del self.locked_positions[symbol]
        if symbol in self.extended_positions:
            del self.extended_positions[symbol]
        if symbol in self.runner_positions:
            del self.runner_positions[symbol]
        
        self.logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} Reason: {reason}")
    
    def _get_point_value(self, symbol: str) -> float:
        """Get point value for symbol"""
        point_values = {
            'XAUUSD': 100,
            'EURUSD': 100000,
            'USDX': 1000,
            'BTCUSD': 1
        }
        return point_values.get(symbol, 100)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all managed positions"""
        
        return {
            'active_positions': len(self.active_positions),
            'locked_positions': len(self.locked_positions),
            'extended_positions': len(self.extended_positions),
            'runner_positions': len(self.runner_positions),
            'total_adjustments': sum(len(pos['adjustments']) for pos in self.active_positions.values()),
            'position_history': len(self.position_history)
        }
    
    def export_position_history(self, filepath: str):
        """Export position history to file"""
        
        # Convert to DataFrame for easier export
        history_data = []
        
        for record in self.position_history:
            history_data.append({
                'symbol': record['symbol'],
                'entry_time': record['position'].timestamp,
                'exit_time': datetime.now(),
                'direction': record['position'].direction,
                'size': record['position'].size,
                'entry_price': record['entry_price'],
                'exit_price': record['exit_price'],
                'pnl': record['pnl'],
                'reason': record['reason'],
                'original_sl': record['original_sl'],
                'original_tp': record['original_tp'],
                'final_sl': record['final_sl'],
                'final_tp': record['final_tp'],
                'max_favorable_move': record['max_favorable_move'],
                'adjustments_count': len(record['adjustments']),
                'lock_threshold_hit': record['lock_threshold_hit'],
                'extend_threshold_hit': record['extend_threshold_hit'],
                'runner_mode_active': record['runner_mode_active'],
                'duration_minutes': record['duration'].total_seconds() / 60
            })
        
        df = pd.DataFrame(history_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Position history exported to {filepath}")


class BacktestRunner:
    """Advanced backtest runner with parallel execution and comprehensive reporting"""
    
    def __init__(self, config: RunnerConfig, output_dir: str = "backtest/py_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Components
        self.metrics_calculator = MetricsCalculator(str(self.output_dir))
        self.dynamic_manager = DynamicPositionManager(config)
        
        # Job queue and execution
        self.job_queue: List[BacktestJob] = []
        self.completed_jobs: List[BacktestJob] = []
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Results storage
        self.results_cache: Dict[str, BacktestResult] = {}
        
        self.logger.info("Backtest runner initialized")
    
    def add_backtest_job(self, strategies: List[BaseStrategy], 
                         market_data: Dict[str, pd.DataFrame],
                         backtest_config: BacktestConfig,
                         priority: int = 0) -> str:
        """Add a backtest job to the queue"""
        
        job_id = f"job_{len(self.job_queue) + 1}_{datetime.now().strftime('%H%M%S')}"
        
        job = BacktestJob(
            job_id=job_id,
            strategies=strategies,
            market_data=market_data,
            config=backtest_config,
            priority=priority
        )
        
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: x.priority, reverse=True)
        
        self.logger.info(f"Backtest job added: {job_id}")
        
        return job_id
    
    def run_single_backtest(self, strategies: List[BaseStrategy], 
                           market_data: Dict[str, pd.DataFrame],
                           config: BacktestConfig) -> BacktestResult:
        """Run a single backtest with enhanced rule enforcement"""
        
        self.logger.info(f"Running backtest from {config.start_date} to {config.end_date}")
        
        # Create enhanced backtest engine
        engine = BacktestEngine(config, str(self.output_dir))
        
        # Run backtest
        result = engine.run_backtest(strategies, market_data)
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            result.equity_curve, result.trades, config.initial_capital
        )
        
        # Validate rule compliance
        if self.config.validate_rules:
            compliance_report = self._validate_rule_compliance(result, config)
            result.compliance_report = compliance_report
        
        # Generate reports
        if self.config.generate_reports:
            self._generate_comprehensive_report(result, metrics)
        
        # Create charts
        if self.config.create_charts:
            self.metrics_calculator.create_performance_charts(
                result.equity_curve, result.trades, save_charts=True
            )
        
        return result
    
    def run_parallel_backtests(self) -> Dict[str, BacktestResult]:
        """Run multiple backtests in parallel"""
        
        self.logger.info(f"Running {len(self.job_queue)} backtests in parallel")
        
        if not self.config.run_parallel:
            # Sequential execution
            for job in self.job_queue:
                job.status = "running"
                job.start_time = datetime.now()
                
                try:
                    result = self.run_single_backtest(
                        job.strategies, job.market_data, job.config
                    )
                    job.result = result
                    job.status = "completed"
                    self.results_cache[job.job_id] = result
                    
                except Exception as e:
                    job.error = str(e)
                    job.status = "failed"
                    self.logger.error(f"Job {job.job_id} failed: {e}")
                
                finally:
                    job.end_time = datetime.now()
                    self.completed_jobs.append(job)
            
            return self.results_cache
        
        # Parallel execution
        futures = {}
        
        for job in self.job_queue:
            job.status = "running"
            job.start_time = datetime.now()
            
            future = self.executor.submit(
                self.run_single_backtest,
                job.strategies,
                job.market_data,
                job.config
            )
            
            futures[future] = job
        
        # Wait for completion
        for future in futures:
            job = futures[future]
            
            try:
                result = future.result()
                job.result = result
                job.status = "completed"
                self.results_cache[job.job_id] = result
                
            except Exception as e:
                job.error = str(e)
                job.status = "failed"
                self.logger.error(f"Job {job.job_id} failed: {e}")
            
            finally:
                job.end_time = datetime.now()
                self.completed_jobs.append(job)
        
        return self.results_cache
    
    def _validate_rule_compliance(self, result: BacktestResult, 
                                 config: BacktestConfig) -> Dict[str, Any]:
        """Validate compliance with all trading rules"""
        
        compliance_report = {
            'overall_compliance': 0.0,
            'rule_violations': [],
            'daily_loss_compliance': 0.0,
            'consecutive_losses_compliance': 0.0,
            'position_limits_compliance': 0.0,
            'risk_per_trade_compliance': 0.0
        }
        
        # Check daily loss limit
        daily_pnls = {}
        for trade in result.trades:
            if trade['action'] == 'CLOSE':
                date = trade['timestamp'].date()
                if date not in daily_pnls:
                    daily_pnls[date] = 0
                daily_pnls[date] += trade['pnl']
        
        max_daily_loss = max([abs(pnl) for pnl in daily_pnls.values() if pnl < 0], default=0)
        daily_loss_compliance = 1.0 if max_daily_loss <= 9.99 else max(0, 1 - (max_daily_loss - 9.99) / 9.99)
        compliance_report['daily_loss_compliance'] = daily_loss_compliance * 100
        
        if max_daily_loss > 9.99:
            compliance_report['rule_violations'].append(f"Daily loss limit exceeded: ${max_daily_loss:.2f}")
        
        # Check consecutive losses
        consecutive_losses = self._calculate_max_consecutive_losses(result.trades)
        consecutive_losses_compliance = 1.0 if consecutive_losses <= 3 else max(0, 1 - (consecutive_losses - 3) / 3)
        compliance_report['consecutive_losses_compliance'] = consecutive_losses_compliance * 100
        
        if consecutive_losses > 3:
            compliance_report['rule_violations'].append(f"Consecutive losses exceeded: {consecutive_losses}")
        
        # Check position limits
        position_violations = self._check_position_limits(result.trades)
        compliance_report['position_limits_compliance'] = position_violations['compliance'] * 100
        
        if position_violations['violations']:
            compliance_report['rule_violations'].extend(position_violations['violations'])
        
        # Calculate overall compliance
        compliance_scores = [
            compliance_report['daily_loss_compliance'],
            compliance_report['consecutive_losses_compliance'],
            compliance_report['position_limits_compliance']
        ]
        
        compliance_report['overall_compliance'] = np.mean(compliance_scores)
        
        return compliance_report
    
    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losses"""
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['action'] == 'CLOSE' and trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif trade['action'] == 'CLOSE' and trade['pnl'] > 0:
                current_consecutive = 0
        
        return max_consecutive
    
    def _check_position_limits(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check position limit compliance"""
        
        violations = []
        total_checks = 0
        compliant_checks = 0
        
        # Track active positions by date
        active_positions = {}
        
        for trade in trades:
            date = trade['timestamp'].date()
            symbol = trade['symbol']
            
            if trade['action'] == 'OPEN':
                if date not in active_positions:
                    active_positions[date] = set()
                
                if symbol in active_positions[date]:
                    violations.append(f"Multiple positions for {symbol} on {date}")
                else:
                    active_positions[date].add(symbol)
                    compliant_checks += 1
                
                total_checks += 1
            
            elif trade['action'] == 'CLOSE':
                if date in active_positions and symbol in active_positions[date]:
                    active_positions[date].remove(symbol)
        
        compliance = compliant_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'compliance': compliance,
            'violations': violations
        }
    
    def _generate_comprehensive_report(self, result: BacktestResult, 
                                      metrics: PerformanceMetrics):
        """Generate comprehensive backtest report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance report
        performance_report = self.metrics_calculator.generate_performance_report(metrics)
        
        # Risk report
        risk_report = self.metrics_calculator.generate_risk_report(metrics)
        
        # Strategy performance
        strategy_report = self._generate_strategy_performance_report(result)
        
        # Dynamic management report
        dynamic_report = self._generate_dynamic_management_report()
        
        # Combine reports
        full_report = f"""
{performance_report}

{risk_report}

# Strategy Performance
{strategy_report}

# Dynamic TP/SL Management
{dynamic_report}

# Backtest Configuration
- Start Date: {result.config.start_date}
- End Date: {result.config.end_date}
- Initial Capital: ${result.config.initial_capital:,.2f}
- Commission: {result.config.commission * 100:.3f}%
- Slippage: {result.config.slippage * 100:.3f}%
- Dynamic TP/SL: {'Enabled' if result.config.enable_dynamic_tp else 'Disabled'}
- Runner Mode: {'Enabled' if result.config.enable_runner_mode else 'Disabled'}

# Execution Summary
- Total Duration: {result.total_duration}
- Start Time: {result.start_time}
- End Time: {result.end_time}
- Jobs Processed: {len(self.completed_jobs)}
"""
        
        # Save report
        report_path = self.output_dir / f"comprehensive_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(full_report)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
    
    def _generate_strategy_performance_report(self, result: BacktestResult) -> str:
        """Generate strategy performance report"""
        
        if not result.strategy_performance:
            return "No strategy performance data available."
        
        report = "\n## Strategy Performance\n\n"
        
        for strategy_name, perf in result.strategy_performance.items():
            report += f"### {strategy_name}\n"
            report += f"- Signals: {perf['signals']}\n"
            report += f"- Trades: {perf['trades']}\n"
            report += f"- Wins: {perf['wins']}\n"
            report += f"- Losses: {perf['losses']}\n"
            report += f"- Win Rate: {perf['win_rate']:.1f}%\n"
            report += f"- P&L: ${perf['pnl']:.2f}\n\n"
        
        return report
    
    def _generate_dynamic_management_report(self) -> str:
        """Generate dynamic TP/SL management report"""
        
        summary = self.dynamic_manager.get_position_summary()
        
        report = f"""
## Dynamic TP/SL Management Summary
- Active Positions: {summary['active_positions']}
- Positions with Locked SL: {summary['locked_positions']}
- Positions with Extended TP: {summary['extended_positions']}
- Runner Mode Positions: {summary['runner_positions']}
- Total Adjustments: {summary['total_adjustments']}
- Historical Positions: {summary['position_history']}

### Management Rules Applied
- Profit Lock Threshold: +${self.config.lock_profit_threshold}
- TP Extension Threshold: +${self.config.extend_tp_threshold}
- Extended TP Target: +${self.config.extended_tp_target}
- Runner Mode: {'Enabled' if self.config.runner_mode_enabled else 'Disabled'}
- Momentum Threshold: {self.config.momentum_threshold:.3f}
"""
        
        return report
    
    def compare_results(self, result_ids: List[str] = None) -> pd.DataFrame:
        """Compare multiple backtest results"""
        
        if result_ids is None:
            result_ids = list(self.results_cache.keys())
        
        comparison_data = []
        
        for result_id in result_ids:
            if result_id not in self.results_cache:
                continue
            
            result = self.results_cache[result_id]
            metrics = result.metrics
            
            comparison_data.append({
                'Job ID': result_id,
                'Total Return (%)': metrics.get('total_return', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                'Win Rate (%)': metrics.get('win_rate', 0),
                'Profit Factor': metrics.get('profit_factor', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Annual Return (%)': metrics.get('annual_return', 0),
                'Volatility (%)': metrics.get('volatility', 0),
                'Compliance (%)': getattr(result, 'compliance_report', {}).get('overall_compliance', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = self.output_dir / f"results_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"Results comparison saved to {comparison_path}")
        
        return comparison_df
    
    def export_all_results(self, format: str = "json"):
        """Export all backtest results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            self._export_results_json(timestamp)
        elif format == "csv":
            self._export_results_csv(timestamp)
        elif format == "excel":
            self._export_results_excel(timestamp)
        
        self.logger.info(f"All results exported in {format} format")
    
    def _export_results_json(self, timestamp: str):
        """Export results in JSON format"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'runner_config': {
                'enable_dynamic_management': self.config.enable_dynamic_management,
                'lock_profit_threshold': self.config.lock_profit_threshold,
                'extend_tp_threshold': self.config.extend_tp_threshold,
                'runner_mode_enabled': self.config.runner_mode_enabled
            },
            'jobs_completed': len(self.completed_jobs),
            'results': {}
        }
        
        for result_id, result in self.results_cache.items():
            export_data['results'][result_id] = {
                'metrics': result.metrics,
                'total_trades': len(result.trades),
                'final_equity': result.equity_curve.iloc[-1],
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration': str(result.total_duration),
                'compliance': getattr(result, 'compliance_report', {})
            }
        
        with open(self.output_dir / f"all_results_{timestamp}.json", 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_results_csv(self, timestamp: str):
        """Export results in CSV format"""
        
        # Create summary CSV
        summary_data = []
        
        for result_id, result in self.results_cache.items():
            summary_data.append({
                'result_id': result_id,
                'total_return': result.metrics.get('total_return', 0),
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': result.metrics.get('max_drawdown', 0),
                'win_rate': result.metrics.get('win_rate', 0),
                'total_trades': len(result.trades),
                'final_equity': result.equity_curve.iloc[-1],
                'compliance': getattr(result, 'compliance_report', {}).get('overall_compliance', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / f"results_summary_{timestamp}.csv", index=False)
        
        # Export trades for each result
        for result_id, result in self.results_cache.items():
            if result.trades:
                trades_df = pd.DataFrame(result.trades)
                trades_df.to_csv(self.output_dir / f"trades_{result_id}_{timestamp}.csv", index=False)
    
    def _export_results_excel(self, timestamp: str):
        """Export results in Excel format"""
        
        try:
            import excelwriter
            
            with pd.ExcelWriter(self.output_dir / f"all_results_{timestamp}.xlsx", engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = []
                for result_id, result in self.results_cache.items():
                    summary_data.append({
                        'Result ID': result_id,
                        'Total Return (%)': result.metrics.get('total_return', 0),
                        'Sharpe Ratio': result.metrics.get('sharpe_ratio', 0),
                        'Max Drawdown (%)': result.metrics.get('max_drawdown', 0),
                        'Win Rate (%)': result.metrics.get('win_rate', 0),
                        'Total Trades': len(result.trades),
                        'Final Equity': result.equity_curve.iloc[-1],
                        'Compliance (%)': getattr(result, 'compliance_report', {}).get('overall_compliance', 0)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed results for each job
                for result_id, result in self.results_cache.items():
                    if result.trades:
                        trades_df = pd.DataFrame(result.trades)
                        trades_df.to_excel(writer, sheet_name=f'Trades_{result_id[:20]}', index=False)
            
            self.logger.info(f"Results exported to Excel: all_results_{timestamp}.xlsx")
            
        except ImportError:
            self.logger.warning("Excel export requires openpyxl package")
    
    def get_runner_status(self) -> Dict[str, Any]:
        """Get comprehensive runner status"""
        
        return {
            'config': {
                'run_parallel': self.config.run_parallel,
                'max_workers': self.config.max_workers,
                'enable_dynamic_management': self.config.enable_dynamic_management,
                'runner_mode_enabled': self.config.runner_mode_enabled
            },
            'job_queue': {
                'pending_jobs': len([j for j in self.job_queue if j.status == 'pending']),
                'running_jobs': len([j for j in self.job_queue if j.status == 'running']),
                'completed_jobs': len(self.completed_jobs),
                'failed_jobs': len([j for j in self.completed_jobs if j.status == 'failed'])
            },
            'results': {
                'cached_results': len(self.results_cache),
                'total_trades': sum(len(r.trades) for r in self.results_cache.values()),
                'avg_return': np.mean([r.metrics.get('total_return', 0) for r in self.results_cache.values()]) if self.results_cache else 0
            },
            'dynamic_management': self.dynamic_manager.get_position_summary()
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        self.executor.shutdown(wait=True)
        self.dynamic_manager = None
        
        self.logger.info("Backtest runner cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
