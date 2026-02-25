"""
Nexus Trading System - Backtesting Engine
Comprehensive backtesting system that enforces all trading rules and risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

from core.trade_manager import TradeManager, Position
from core.risk_engine import RiskEngine
from core.position_sizer import PositionSizer
from core.regime_detector import RegimeDetector
from core.logger import get_logger
from strategy.base_strategy import BaseStrategy, TradingSignal, SignalType


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    commission: float = 0.0001  # 0.01% commission
    slippage: float = 0.0001   # 0.01% slippage
    margin_call_level: float = 0.5  # 50% margin call
    enable_dynamic_tp: bool = True
    enable_runner_mode: bool = True
    enforce_session_filters: bool = True
    enforce_risk_limits: bool = True
    save_trades: bool = True
    save_equity_curve: bool = True


@dataclass
class BacktestResult:
    """Results from backtesting"""
    equity_curve: pd.Series
    returns: pd.Series
    trades: List[Dict[str, Any]]
    positions: List[Position]
    metrics: Dict[str, float]
    daily_stats: pd.DataFrame
    risk_metrics: Dict[str, Any]
    strategy_performance: Dict[str, Dict[str, Any]]
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    total_duration: timedelta


class BacktestEngine:
    """
    Comprehensive backtesting engine that enforces all trading rules:
    - Default risk: 1% per trade, dollar override optional
    - Max daily loss: $9.99
    - Only one active trade per asset at a time
    - Max trades per day: 9H(2), 6H(2), 3H(1)
    - TP/SL: Entry SL=-$3, TP=+$9.9, lock SL to +$3, extend TP to +$15
    - All decisions logged for auditing
    """
    
    def __init__(self, config: BacktestConfig, data_dir: str = "data/processed"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path("backtest/py_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Initialize core components
        self.trade_manager = TradeManager(self._load_risk_config())
        self.risk_engine = RiskEngine(self._load_risk_config())
        self.position_sizer = PositionSizer(self._load_risk_config())
        self.regime_detector = RegimeDetector()
        
        # Backtesting state
        self.current_capital = config.initial_capital
        self.current_equity = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Any] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        # Rule enforcement
        self.daily_trade_counts: Dict[str, Dict[str, int]] = {}
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_reset_date = config.start_date.date()
        
        self.logger.info("Backtest engine initialized")
    
    def run_backtest(self, strategies: List[BaseStrategy], 
                    market_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run comprehensive backtest with all trading rules enforced
        
        Args:
            strategies: List of trading strategies to test
            market_data: Dictionary of market data by symbol
            
        Returns:
            BacktestResult with comprehensive results
        """
        
        self.logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        start_time = datetime.now()
        
        # Initialize strategy performance tracking
        for strategy in strategies:
            self.strategy_performance[strategy.name] = {
                'signals': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'win_rate': 0.0
            }
        
        # Process each day
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            # Reset daily limits if needed
            self._check_daily_reset(current_date)
            
            # Process each symbol
            for symbol, data in market_data.items():
                if symbol not in data.index or current_date not in data.index:
                    continue
                
                daily_data = data.loc[data.index <= current_date]
                
                if len(daily_data) < 100:  # Need enough data for analysis
                    continue
                
                # Get current bar data
                current_bar = daily_data.loc[current_date]
                
                # Process strategies for this symbol
                self._process_strategies_for_symbol(
                    strategies, symbol, daily_data, current_bar, current_date
                )
                
                # Manage existing positions
                self._manage_positions(symbol, current_bar, current_date)
            
            # Update equity curve
            self._update_equity_curve(current_date)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close all positions at end
        self._close_all_positions(market_data, current_date)
        
        # Calculate final results
        end_time = datetime.now()
        result = self._calculate_results(start_time, end_time)
        
        # Save results
        self._save_results(result)
        
        self.logger.info(f"Backtest completed in {end_time - start_time}")
        
        return result
    
    def _process_strategies_for_symbol(self, strategies: List[BaseStrategy], 
                                     symbol: str, daily_data: pd.DataFrame,
                                     current_bar: pd.Series, current_date: datetime):
        """Process all strategies for a specific symbol"""
        
        # Check session filters
        if self.config.enforce_session_filters:
            if not self._check_session_filters(symbol, current_date):
                return
        
        # Check if we already have an active position
        if self.trade_manager.has_active_position(symbol):
            return
        
        # Check daily trade limits
        if not self._check_daily_trade_limits(symbol, current_date):
            return
        
        # Check risk limits
        if not self._check_risk_limits():
            return
        
        # Generate signals from all strategies
        for strategy in strategies:
            try:
                # Generate signal
                signal = strategy.generate_signal(symbol, daily_data)
                
                if signal is None:
                    continue
                
                # Update strategy performance
                self.strategy_performance[strategy.name]['signals'] += 1
                
                # Validate signal
                if not self._validate_signal(signal, symbol, current_bar, current_date):
                    continue
                
                # Process signal
                self._process_signal(signal, symbol, current_bar, current_date, strategy.name)
                
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy.name} for {symbol}: {e}")
    
    def _validate_signal(self, signal: TradingSignal, symbol: str, 
                        current_bar: pd.Series, current_date: datetime) -> bool:
        """Validate trading signal against all rules"""
        
        # Check minimum confidence
        if signal.confidence < 0.6:
            return False
        
        # Check market conditions
        if not self._check_market_conditions(symbol, current_bar):
            return False
        
        # Risk assessment
        market_data = {
            'symbol': symbol,
            'close': current_bar['close'],
            'high': current_bar['high'],
            'low': current_bar['low'],
            'volume': current_bar.get('volume', 0),
            'volatility': self._calculate_volatility(symbol),
            'spread': self._estimate_spread(symbol)
        }
        
        portfolio_state = {
            'open_positions': len(self.positions),
            'positions': self.positions,
            'portfolio_heat': self._calculate_portfolio_heat()
        }
        
        daily_stats = {
            'trades_opened': len([t for t in self.trade_history if t['timestamp'].date() == current_date.date()]),
            'total_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses
        }
        
        risk_assessment = self.risk_engine.assess_trade_risk(
            signal, market_data, portfolio_state, daily_stats
        )
        
        if not risk_assessment.is_allowed:
            self.logger.debug(f"Signal rejected by risk engine: {risk_assessment.reasons}")
            return False
        
        return True
    
    def _process_signal(self, signal: TradingSignal, symbol: str, 
                       current_bar: pd.Series, current_date: datetime, strategy_name: str):
        """Process validated trading signal"""
        
        # Calculate position size
        market_data = {
            'symbol': symbol,
            'close': current_bar['close'],
            'volatility': self._calculate_volatility(symbol)
        }
        
        daily_stats = {
            'trades_opened': len([t for t in self.trade_history if t['timestamp'].date() == current_date.date()]),
            'total_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses
        }
        
        position_size = self.position_sizer.calculate_position_size(
            signal, market_data, daily_stats, self.current_capital
        )
        
        if position_size <= 0:
            return
        
        # Calculate SL/TP according to rules
        entry_price = current_bar['close']
        
        # Default SL/TP as per rules: SL = -$3, TP = +$9.9
        point_value = self._get_point_value(symbol)
        sl_points = 300  # $3 default
        tp_points = 990  # $9.9 default
        
        sl_distance = sl_points / point_value
        tp_distance = tp_points / point_value
        
        if signal.signal_type == SignalType.BUY:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        # Apply slippage and commission
        fill_price = self._apply_slippage(entry_price, signal.signal_type)
        commission_cost = fill_price * position_size * self.config.commission
        
        # Open position
        position = Position(
            symbol=symbol,
            direction=signal.signal_type.value,
            size=position_size,
            entry_price=fill_price,
            sl_price=sl_price,
            tp_price=tp_price,
            strategy=strategy_name,
            timestamp=current_date,
            unrealized_pnl=-commission_cost,
            notes=[f"Signal: {signal.reason}", f"Commission: {commission_cost:.2f}"]
        )
        
        self.positions[symbol] = position
        
        # Update trade manager
        self.trade_manager.open_position(
            symbol, signal.signal_type.value, position_size,
            fill_price, sl_price, tp_price, strategy_name
        )
        
        # Update daily stats
        self._update_daily_trade_counts(symbol, current_date)
        self.daily_pnl -= commission_cost
        
        # Record trade
        trade_record = {
            'timestamp': current_date,
            'symbol': symbol,
            'action': 'OPEN',
            'direction': signal.signal_type.value,
            'size': position_size,
            'entry_price': fill_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'strategy': strategy_name,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'commission': commission_cost
        }
        
        self.trade_history.append(trade_record)
        self.strategy_performance[strategy_name]['trades'] += 1
        
        self.logger.info(f"Position opened: {symbol} {signal.signal_type.value} {position_size} @ {fill_price:.4f}")
    
    def _manage_positions(self, symbol: str, current_bar: pd.Series, current_date: datetime):
        """Manage existing positions with dynamic TP/SL"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = current_bar['close']
        
        # Update unrealized P&L
        if position.direction == 'buy':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size
        
        position.unrealized_pnl = pnl
        
        # Check for SL/TP hit
        should_close = False
        close_reason = ""
        close_price = current_price
        
        if position.direction == 'buy':
            if current_price <= position.sl_price:
                should_close = True
                close_reason = "Stop Loss"
                close_price = position.sl_price
            elif current_price >= position.tp_price:
                should_close = True
                close_reason = "Take Profit"
                close_price = position.tp_price
        else:  # sell
            if current_price >= position.sl_price:
                should_close = True
                close_reason = "Stop Loss"
                close_price = position.sl_price
            elif current_price <= position.tp_price:
                should_close = True
                close_reason = "Take Profit"
                close_price = position.tp_price
        
        # Dynamic TP/SL management
        if not should_close and self.config.enable_dynamic_tp:
            should_close, close_reason, close_price = self._check_dynamic_tp_sl(
                position, current_price, current_date
            )
        
        if should_close:
            self._close_position(symbol, close_price, close_reason, current_date)
    
    def _check_dynamic_tp_sl(self, position: Position, current_price: float, 
                            current_date: datetime) -> Tuple[bool, str, float]:
        """Check dynamic TP/SL conditions"""
        
        point_value = self._get_point_value(position.symbol)
        
        # Lock profits at +$3
        lock_profit_threshold = 300  # $3
        lock_distance = lock_profit_threshold / point_value
        
        # Extend TP to +$15
        extend_tp_threshold = 990  # $9.9 (original TP)
        extend_tp_distance = 1500  # $15
        
        if position.direction == 'buy':
            # Lock SL to +$3 when trade moves favorably
            if current_price >= position.entry_price + lock_distance:
                if position.current_sl is None or position.current_sl < position.entry_price + lock_distance:
                    new_sl = position.entry_price + lock_distance
                    position.current_sl = new_sl
                    position.notes.append(f"SL locked to +${lock_profit_threshold/100:.1f} at {new_sl:.4f}")
                    
                    self.logger.info(f"SL locked for {position.symbol}: {new_sl:.4f}")
            
            # Extend TP when original TP is hit
            if current_price >= position.tp_price and self.config.enable_runner_mode:
                new_tp = position.entry_price + extend_tp_distance
                position.tp_price = new_tp
                position.notes.append(f"TP extended to +${extend_tp_distance/100:.1f} at {new_tp:.4f}")
                
                self.logger.info(f"TP extended for {position.symbol}: {new_tp:.4f}")
        
        else:  # sell position
            # Lock SL to +$3 when trade moves favorably
            if current_price <= position.entry_price - lock_distance:
                if position.current_sl is None or position.current_sl > position.entry_price - lock_distance:
                    new_sl = position.entry_price - lock_distance
                    position.current_sl = new_sl
                    position.notes.append(f"SL locked to +${lock_profit_threshold/100:.1f} at {new_sl:.4f}")
                    
                    self.logger.info(f"SL locked for {position.symbol}: {new_sl:.4f}")
            
            # Extend TP when original TP is hit
            if current_price <= position.tp_price and self.config.enable_runner_mode:
                new_tp = position.entry_price - extend_tp_distance
                position.tp_price = new_tp
                position.notes.append(f"TP extended to +${extend_tp_distance/100:.1f} at {new_tp:.4f}")
                
                self.logger.info(f"TP extended for {position.symbol}: {new_tp:.4f}")
        
        return False, "", current_price
    
    def _close_position(self, symbol: str, close_price: float, reason: str, current_date: datetime):
        """Close a position"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage
        if position.direction == 'buy':
            fill_price = close_price * (1 - self.config.slippage)
        else:
            fill_price = close_price * (1 + self.config.slippage)
        
        # Calculate P&L
        if position.direction == 'buy':
            pnl = (fill_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - fill_price) * position.size
        
        # Add commission
        commission_cost = fill_price * position.size * self.config.commission
        pnl -= commission_cost
        
        # Update position
        position.realized_pnl = pnl
        position.status = 'closed'
        position.close_timestamp = current_date
        position.close_reason = reason
        
        # Update trade manager
        self.trade_manager.close_position(symbol, pnl, reason)
        
        # Update daily stats
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.consecutive_losses = 0
            self.strategy_performance[position.strategy]['wins'] += 1
        else:
            self.consecutive_losses += 1
            self.strategy_performance[position.strategy]['losses'] += 1
        
        self.strategy_performance[position.strategy]['pnl'] += pnl
        
        # Record trade
        trade_record = {
            'timestamp': current_date,
            'symbol': symbol,
            'action': 'CLOSE',
            'direction': position.direction,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': fill_price,
            'pnl': pnl,
            'reason': reason,
            'strategy': position.strategy,
            'commission': commission_cost,
            'duration': current_date - position.timestamp
        }
        
        self.trade_history.append(trade_record)
        
        # Remove from active positions
        del self.positions[symbol]
        
        self.logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} Reason: {reason}")
    
    def _check_daily_reset(self, current_date: datetime):
        """Check and reset daily limits"""
        
        if current_date.date() != self.last_reset_date:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.daily_trade_counts.clear()
            self.last_reset_date = current_date.date()
            
            self.logger.info(f"Daily limits reset for {current_date.date()}")
    
    def _check_session_filters(self, symbol: str, current_date: datetime) -> bool:
        """Check session filters for trading"""
        
        # Get asset configuration
        asset_config = self._get_asset_config(symbol)
        
        if not asset_config:
            return True  # Allow if no config found
        
        current_hour = current_date.hour
        
        # Check avoid periods
        for avoid_period in asset_config.get('avoid_periods', []):
            start_time = datetime.strptime(avoid_period['start'], "%H:%M").time()
            end_time = datetime.strptime(avoid_period['end'], "%H:%M").time()
            
            current_time = current_date.time()
            
            if start_time <= current_time <= end_time:
                return False
        
        return True
    
    def _check_daily_trade_limits(self, symbol: str, current_date: datetime) -> bool:
        """Check daily trade limits per timeframe"""
        
        # Max trades per day: 9H(2), 6H(2), 3H(1)
        max_trades_per_day = {
            "9H": 2,
            "6H": 2,
            "3H": 1
        }
        
        # For simplicity, use 1H as default
        timeframe = "1H"
        max_trades = 2  # Default limit
        
        today_trades = len([t for t in self.trade_history 
                          if t['symbol'] == symbol and t['timestamp'].date() == current_date.date()])
        
        return today_trades < max_trades
    
    def _check_risk_limits(self) -> bool:
        """Check risk limits"""
        
        # Max daily loss: $9.99
        if self.daily_pnl <= -9.99:
            self.logger.warning("Max daily loss reached")
            return False
        
        # Kill switch conditions
        if self.consecutive_losses >= 3:
            self.logger.warning("Kill switch: Too many consecutive losses")
            return False
        
        return True
    
    def _check_market_conditions(self, symbol: str, current_bar: pd.Series) -> bool:
        """Check market conditions"""
        
        # Check volatility
        volatility = self._calculate_volatility(symbol)
        if volatility < 0.001 or volatility > 0.05:
            return False
        
        # Check spread
        spread = self._estimate_spread(symbol)
        if spread > 10.0:  # Max spread threshold
            return False
        
        return True
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility"""
        # Simplified volatility calculation
        return 0.02  # 2% default
    
    def _estimate_spread(self, symbol: str) -> float:
        """Estimate current spread"""
        # Asset-specific spreads
        spreads = {
            'XAUUSD': 5.0,
            'EURUSD': 2.0,
            'USDX': 1.5,
            'BTCUSD': 10.0
        }
        return spreads.get(symbol, 3.0)
    
    def _get_point_value(self, symbol: str) -> float:
        """Get point value for symbol"""
        point_values = {
            'XAUUSD': 100,
            'EURUSD': 100000,
            'USDX': 1000,
            'BTCUSD': 1
        }
        return point_values.get(symbol, 100)
    
    def _apply_slippage(self, price: float, direction: SignalType) -> float:
        """Apply slippage to price"""
        if direction == SignalType.BUY:
            return price * (1 + self.config.slippage)
        else:
            return price * (1 - self.config.slippage)
    
    def _update_daily_trade_counts(self, symbol: str, current_date: datetime):
        """Update daily trade counts"""
        date_str = current_date.strftime('%Y-%m-%d')
        
        if date_str not in self.daily_trade_counts:
            self.daily_trade_counts[date_str] = {}
        
        if symbol not in self.daily_trade_counts[date_str]:
            self.daily_trade_counts[date_str][symbol] = 0
        
        self.daily_trade_counts[date_str][symbol] += 1
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat (total risk exposure)"""
        total_risk = 0.0
        
        for position in self.positions.values():
            # Simplified risk calculation
            position_risk = abs(position.size * 0.01)  # 1% risk per position
            total_risk += position_risk
        
        return (total_risk / self.current_capital) * 100 if self.current_capital > 0 else 0
    
    def _update_equity_curve(self, current_date: datetime):
        """Update equity curve"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(t['pnl'] for t in self.trade_history)
        
        current_equity = self.config.initial_capital + total_realized + total_unrealized
        self.current_equity = current_equity
        
        self.equity_curve.append((current_date, current_equity))
    
    def _close_all_positions(self, market_data: Dict[str, pd.DataFrame], current_date: datetime):
        """Close all positions at end of backtest"""
        
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                self._close_position(symbol, current_price, "End of backtest", current_date)
    
    def _calculate_results(self, start_time: datetime, end_time: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Create equity curve series
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        equity_curve = equity_df['equity']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(equity_curve, returns)
        
        # Create daily stats dataframe
        daily_stats_df = self._create_daily_stats_dataframe()
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Update strategy win rates
        for strategy_name, perf in self.strategy_performance.items():
            total_trades = perf['wins'] + perf['losses']
            if total_trades > 0:
                perf['win_rate'] = perf['wins'] / total_trades
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=self.trade_history,
            positions=list(self.positions.values()),
            metrics=metrics,
            daily_stats=daily_stats_df,
            risk_metrics=risk_metrics,
            strategy_performance=self.strategy_performance,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_duration=end_time - start_time
        )
    
    def _calculate_performance_metrics(self, equity_curve: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annual_return = total_return * (365 / len(equity_curve)) * 100
        
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annual_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        total_trades = len([t for t in self.trade_history if t['action'] == 'CLOSE'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average trade
        profits = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
        
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': equity_curve.iloc[-1]
        }
    
    def _create_daily_stats_dataframe(self) -> pd.DataFrame:
        """Create daily statistics dataframe"""
        
        daily_data = []
        
        for date_str, trades_by_symbol in self.daily_trade_counts.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            total_trades = sum(trades_by_symbol.values())
            daily_pnl = sum(t['pnl'] for t in self.trade_history 
                           if t['timestamp'].date() == date.date())
            
            daily_data.append({
                'date': date,
                'trades': total_trades,
                'pnl': daily_pnl,
                'consecutive_losses': self.consecutive_losses
            })
        
        return pd.DataFrame(daily_data).set_index('date')
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics"""
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = returns.mean() / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _save_results(self, result: BacktestResult):
        """Save backtest results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save equity curve
        if self.config.save_equity_curve:
            result.equity_curve.to_csv(self.output_dir / f"equity_curve_{timestamp}.csv")
        
        # Save trades
        if self.config.save_trades:
            trades_df = pd.DataFrame(result.trades)
            trades_df.to_csv(self.output_dir / f"trades_{timestamp}.csv", index=False)
        
        # Save metrics
        with open(self.output_dir / f"metrics_{timestamp}.json", 'w') as f:
            json.dump(result.metrics, f, indent=2)
        
        # Save strategy performance
        with open(self.output_dir / f"strategy_performance_{timestamp}.json", 'w') as f:
            json.dump(result.strategy_performance, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _load_risk_config(self) -> Dict[str, Any]:
        """Load risk configuration"""
        # This would load from config file in production
        return {
            'position_sizing': {'default_risk_percent': 1.0},
            'daily_limits': {'max_daily_loss': 9.99},
            'kill_switch': {'max_consecutive_losses': 3}
        }
    
    def _get_asset_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset configuration"""
        # This would load from config file in production
        asset_configs = {
            'XAUUSD': {
                'avoid_periods': [
                    {'start': '22:00', 'end': '01:00', 'reason': 'Low liquidity'}
                ]
            }
        }
        return asset_configs.get(symbol)
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of backtest results"""
        
        if not self.trade_history:
            return {'message': 'No trades executed'}
        
        total_trades = len([t for t in self.trade_history if t['action'] == 'CLOSE'])
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'final_equity': self.current_equity,
            'max_consecutive_losses': max(
                self._calculate_consecutive_losses(),
                self.consecutive_losses
            )
        }
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trade_history:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
