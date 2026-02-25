"""
Nexus Trading System - Core Engine
Main trading engine that coordinates all system components
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yaml
from pathlib import Path

from .trade_manager import TradeManager
from .risk_engine import RiskEngine
from .position_sizer import PositionSizer
from .regime_detector import RegimeDetector
from .logger import setup_logging
from data.loaders import DataManager
from strategy.base_strategy import BaseStrategy
from execution.order_executor import OrderExecutor


class TradingEngine:
    """Main trading engine that orchestrates the entire trading system"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs = self._load_all_configs()
        
        # Setup logging
        logger_instance = setup_logging("INFO")
        self.logger = logger_instance.system_logger
        
        # Initialize core components
        self.data_manager = DataManager(
            loader_type=self.configs['execution'].get('loader_type', 'csv')
        )
        self.risk_engine = RiskEngine(self.configs['risk'])
        self.position_sizer = PositionSizer(self.configs['risk'])
        self.regime_detector = RegimeDetector()
        self.trade_manager = TradeManager(self.configs['risk'])
        self.order_executor = OrderExecutor(self.configs['execution'])
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # Engine state
        self.is_running = False
        self.current_positions = {}
        self.daily_stats = {
            'trades_taken': 0,
            'profit_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Track daily limits
        self.last_reset_date = datetime.now().date()
        self._reset_daily_limits()
        
        self.logger.info("Trading Engine initialized")
    
    def _load_all_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        config_files = ['assets.yaml', 'risk.yaml', 'execution.yaml', 'model.yaml']
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            try:
                with open(config_path, 'r') as f:
                    config_name = config_file.replace('.yaml', '')
                    configs[config_name] = yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {config_file}: {e}")
                configs[config_file.replace('.yaml', '')] = {}
                
        return configs
    
    def _reset_daily_limits(self):
        """Reset daily trading limits"""
        self.daily_stats = {
            'trades_taken': 0,
            'profit_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'consecutive_losses': 0
        }
        self.last_reset_date = datetime.now().date()
        self.logger.info("Daily limits reset")
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a trading strategy"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Registered strategy: {strategy.name}")
    
    def async_start(self):
        """Start the trading engine asynchronously"""
        if self.is_running:
            self.logger.warning("Engine is already running")
            return
            
        self.is_running = True
        self.logger.info("Starting Trading Engine")
        
        # Start main trading loop
        asyncio.create_task(self._main_trading_loop())
    
    async def _main_trading_loop(self):
        """Main trading loop that runs continuously"""
        while self.is_running:
            try:
                # Check if we need to reset daily limits
                current_date = datetime.now().date()
                if current_date != self.last_reset_date:
                    self._reset_daily_limits()
                
                # Process each asset
                for asset_config in self.configs['assets']['assets'].values():
                    await self._process_asset(asset_config)
                
                # Wait for next iteration (check every minute)
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    async def _process_asset(self, asset_config: Dict[str, Any]):
        """Process a single asset for trading opportunities"""
        symbol = asset_config['symbol']
        
        try:
            # Check if we already have an active position
            if self.trade_manager.has_active_position(symbol):
                self._manage_existing_position(symbol)
                return
            
            # Check daily limits
            if not self._check_daily_limits():
                return
            
            # Check session filters
            if not self._check_session_filters(asset_config):
                return
            
            # Get latest data
            latest_data = self._get_latest_data(symbol)
            if latest_data is None or latest_data.empty:
                return
            
            # Check market conditions
            if not self._check_market_conditions(asset_config, latest_data):
                return
            
            # Generate signals from all strategies
            signals = []
            for strategy in self.strategies.values():
                try:
                    signal = strategy.generate_signal(symbol, latest_data)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy.name}: {e}")
            
            # Process signals
            if signals:
                await self._process_signals(symbol, signals, asset_config, latest_data)
                
        except Exception as e:
            self.logger.error(f"Error processing asset {symbol}: {e}")
    
    def _manage_existing_position(self, symbol: str):
        """Manage existing position (SL/TP adjustments, runner mode)"""
        position = self.trade_manager.get_position(symbol)
        if not position:
            return
        
        # Get current price
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return
        
        # Check if we need to adjust SL/TP
        if position['type'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - current_price) * position['size']
        
        # Lock profits if threshold reached
        if pnl >= self.configs['risk']['dynamic_management']['lock_profit_threshold']:
            self._lock_profits(symbol, current_price)
        
        # Extend TP if momentum continues
        if pnl >= self.configs['risk']['dynamic_management']['extend_tp_threshold']:
            self._extend_take_profit(symbol, current_price)
        
        # Check for stop loss or take profit
        if self._check_exit_conditions(position, current_price):
            self._close_position(symbol, reason="SL/TP hit")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are respected"""
        risk_config = self.configs['risk']
        
        # Check max daily loss
        if self.daily_stats['profit_loss'] <= -risk_config['daily_limits']['max_daily_loss']:
            self.logger.warning("Max daily loss reached - stopping trading")
            return False
        
        # Check max daily trades
        if self.daily_stats['trades_taken'] >= risk_config['daily_limits']['max_daily_trades']:
            self.logger.warning("Max daily trades reached - stopping trading")
            return False
        
        # Check kill switch conditions
        if self.daily_stats['consecutive_losses'] >= risk_config['kill_switch']['max_consecutive_losses']:
            self.logger.warning("Kill switch activated - too many consecutive losses")
            return False
        
        return True
    
    def _check_session_filters(self, asset_config: Dict[str, Any]) -> bool:
        """Check if current time is within allowed trading sessions"""
        if not self.configs['execution']['session_filters']['enabled']:
            return True
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_minutes = current_hour * 60 + current_minute
        
        # Check avoid periods
        for avoid_period in asset_config.get('avoid_periods', []):
            start_time = datetime.strptime(avoid_period['start'], "%H:%M")
            end_time = datetime.strptime(avoid_period['end'], "%H:%M")
            
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            
            if start_minutes <= current_time_minutes <= end_minutes:
                self.logger.debug(f"Outside trading hours for {asset_config['symbol']}: {avoid_period['reason']}")
                return False
        
        return True
    
    def _check_market_conditions(self, asset_config: Dict[str, Any], 
                               latest_data: Dict[str, Any]) -> bool:
        """Check if market conditions are suitable for trading"""
        risk_config = self.configs['risk']
        
        # Check volatility
        if risk_config['volatility_filter']['enabled']:
            volatility = latest_data.get('volatility', 0)
            min_vol = risk_config['volatility_filter']['min_volatility_threshold']
            max_vol = risk_config['volatility_filter']['max_volatility_threshold']
            
            if not (min_vol <= volatility <= max_vol):
                self.logger.debug(f"Volatility {volatility} outside range [{min_vol}, {max_vol}]")
                return False
        
        # Check spread
        if risk_config['liquidity_filter']['enabled']:
            spread = latest_data.get('spread', 0)
            max_spread = risk_config['liquidity_filter']['max_spread_threshold']
            
            if spread > max_spread:
                self.logger.debug(f"Spread {spread} exceeds maximum {max_spread}")
                return False
        
        return True
    
    def _get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for an asset"""
        try:
            # Get latest data from data manager
            df = self.data_manager.get_latest_data(symbol, "1H", periods=100)
            if df.empty:
                return None
            
            # Get the most recent candle
            latest_candle = df.iloc[-1]
            
            # Calculate additional metrics
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            return {
                'symbol': symbol,
                'timestamp': latest_candle.name,
                'open': latest_candle['open'],
                'high': latest_candle['high'],
                'low': latest_candle['low'],
                'close': latest_candle['close'],
                'volume': latest_candle['volume'],
                'volatility': volatility,
                'spread': self._get_current_spread(symbol)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for an asset"""
        try:
            latest_data = self._get_latest_data(symbol)
            return latest_data['close'] if latest_data else None
        except:
            return None
    
    def _get_current_spread(self, symbol: str) -> float:
        """Get current spread for an asset"""
        # This would be implemented based on data source
        # For now, return a default spread
        return 2.0
    
    async def _process_signals(self, symbol: str, signals: List[Dict[str, Any]], 
                             asset_config: Dict[str, Any], latest_data: Dict[str, Any]):
        """Process trading signals and execute trades"""
        for signal in signals:
            try:
                # Validate signal
                if not self._validate_signal(signal, asset_config):
                    continue
                
                # Calculate position size
                position_size = self.position_sizer.calculate_position_size(
                    signal, latest_data, self.daily_stats
                )
                
                if position_size <= 0:
                    continue
                
                # Calculate SL/TP
                sl_price, tp_price = self._calculate_sl_tp(signal, latest_data, asset_config)
                
                # Execute trade
                order_result = await self.order_executor.place_order(
                    symbol=symbol,
                    order_type=signal['direction'],
                    volume=position_size,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    comment=f"NEXUS_{signal['strategy']}"
                )
                
                if order_result['success']:
                    # Register the trade
                    self.trade_manager.open_position(
                        symbol=symbol,
                        direction=signal['direction'],
                        size=position_size,
                        entry_price=order_result['fill_price'],
                        sl_price=sl_price,
                        tp_price=tp_price,
                        strategy=signal['strategy'],
                        timestamp=datetime.now()
                    )
                    
                    # Update daily stats
                    self.daily_stats['trades_taken'] += 1
                    
                    self.logger.info(f"Trade opened: {symbol} {signal['direction']} {position_size}")
                
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
    
    def _validate_signal(self, signal: Dict[str, Any], asset_config: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        required_fields = ['direction', 'strategy', 'confidence']
        if not all(field in signal for field in required_fields):
            return False
        
        # Check confidence threshold
        min_confidence = self.configs['model'].get('signal_processing', {}).get('min_confidence', 0.6)
        if signal.get('confidence', 0) < min_confidence:
            return False
        
        return True
    
    def _calculate_sl_tp(self, signal: Dict[str, Any], latest_data: Dict[str, Any],
                        asset_config: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices"""
        current_price = latest_data['close']
        risk_config = self.configs['risk']
        
        # Default SL/TP in points (converted to price based on asset)
        sl_points = risk_config['stop_loss']['default_sl_points']
        tp_points = risk_config['take_profit']['default_tp_points']
        
        point_value = asset_config.get('point_value', 1)
        sl_price_diff = sl_points / point_value
        tp_price_diff = tp_points / point_value
        
        if signal['direction'] == 'buy':
            sl_price = current_price - sl_price_diff
            tp_price = current_price + tp_price_diff
        else:  # sell
            sl_price = current_price + sl_price_diff
            tp_price = current_price - tp_price_diff
        
        return sl_price, tp_price
    
    def _lock_profits(self, symbol: str, current_price: float):
        """Lock profits by moving stop loss to breakeven or profit"""
        position = self.trade_manager.get_position(symbol)
        if not position:
            return
        
        # Move SL to lock in +$3 profit
        lock_profit = self.configs['risk']['dynamic_management']['lock_profit_threshold']
        point_value = self.configs['assets']['assets'][symbol]['point_value']
        sl_price_diff = lock_profit / point_value
        
        if position['type'] == 'buy':
            new_sl = position['entry_price'] + sl_price_diff
        else:
            new_sl = position['entry_price'] - sl_price_diff
        
        # Update the order
        self.order_executor.modify_position(symbol, stop_loss=new_sl)
        position['sl_price'] = new_sl
        
        self.logger.info(f"Locked profits for {symbol} at {new_sl}")
    
    def _extend_take_profit(self, symbol: str, current_price: float):
        """Extend take profit in runner mode"""
        position = self.trade_manager.get_position(symbol)
        if not position:
            return
        
        # Extend TP to +$15
        extended_tp = self.configs['risk']['dynamic_management']['extend_tp_threshold']
        point_value = self.configs['assets']['assets'][symbol]['point_value']
        tp_price_diff = extended_tp / point_value
        
        if position['type'] == 'buy':
            new_tp = position['entry_price'] + tp_price_diff
        else:
            new_tp = position['entry_price'] - tp_price_diff
        
        # Update the order
        self.order_executor.modify_position(symbol, take_profit=new_tp)
        position['tp_price'] = new_tp
        
        self.logger.info(f"Extended TP for {symbol} to {new_tp}")
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if position should be closed"""
        sl_price = position['sl_price']
        tp_price = position['tp_price']
        
        if position['type'] == 'buy':
            return current_price <= sl_price or current_price >= tp_price
        else:
            return current_price >= sl_price or current_price <= tp_price
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        position = self.trade_manager.get_position(symbol)
        if not position:
            return
        
        # Close the order
        result = self.order_executor.close_position(symbol)
        
        if result['success']:
            # Calculate P&L
            current_price = result['close_price']
            if position['type'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
            
            # Update stats
            self.daily_stats['profit_loss'] += pnl
            if pnl < 0:
                self.daily_stats['consecutive_losses'] += 1
            else:
                self.daily_stats['consecutive_losses'] = 0
            
            # Close position in trade manager
            self.trade_manager.close_position(symbol, pnl, reason)
            
            self.logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} Reason: {reason}")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        
        # Close all open positions
        for symbol in list(self.trade_manager.get_all_positions().keys()):
            self._close_position(symbol, "Engine shutdown")
        
        self.logger.info("Trading Engine stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'active_positions': len(self.trade_manager.get_all_positions()),
            'daily_stats': self.daily_stats,
            'last_reset_date': self.last_reset_date,
            'registered_strategies': list(self.strategies.keys())
        }
