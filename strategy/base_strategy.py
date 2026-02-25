"""
Nexus Trading System - Base Strategy
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Trading signal with all required information"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    strategy: str
    timestamp: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    timeframe: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'timeframe': self.timeframe,
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass
class StrategyConfig:
    """Configuration for trading strategy"""
    name: str
    enabled: bool = True
    min_confidence: float = 0.6
    max_position_size: float = 1.0
    risk_per_trade: float = 0.01
    timeframes: List[str] = None
    assets: List[str] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1H", "4H", "1D"]
        if self.assets is None:
            self.assets = ["XAUUSD", "EURUSD", "USDX", "BTCUSD"]
        if self.parameters is None:
            self.parameters = {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Strategy state
        self.is_active = False
        self.last_signal_time = None
        self.signal_history: List[TradingSignal] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0,
            'win_rate': 0.0,
            'total_pnl': 0.0
        }
        
        # Market data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        self.logger.info(f"Strategy {self.name} initialized")
    
    @abstractmethod
    def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal based on market data
        
        Args:
            symbol: Trading symbol
            market_data: OHLCV market data
            
        Returns:
            TradingSignal or None if no signal
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """
        Validate if signal meets strategy criteria
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss price for signal
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Stop loss price
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate take profit price for signal
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Take profit price
        """
        pass
    
    def analyze_market_data(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and return technical indicators
        
        Args:
            symbol: Trading symbol
            market_data: OHLCV data
            
        Returns:
            Dictionary with technical indicators
        """
        
        # Cache data if not already cached
        if symbol not in self.data_cache or len(self.data_cache[symbol]) < len(market_data):
            self.data_cache[symbol] = market_data.copy()
        
        # Calculate basic indicators
        indicators = {}
        
        # Price indicators
        indicators['current_price'] = market_data['close'].iloc[-1]
        indicators['price_change'] = market_data['close'].pct_change().iloc[-1]
        indicators['high_low_ratio'] = market_data['high'].iloc[-1] / market_data['low'].iloc[-1]
        
        # Moving averages
        for period in [10, 20, 50, 200]:
            indicators[f'ma_{period}'] = market_data['close'].rolling(period).mean().iloc[-1]
            indicators[f'price_to_ma_{period}'] = indicators['current_price'] / indicators[f'ma_{period}']
        
        # Volatility
        returns = market_data['close'].pct_change().dropna()
        indicators['volatility'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        indicators['atr'] = self._calculate_atr(market_data, 14)
        
        # Momentum
        indicators['rsi'] = self._calculate_rsi(market_data, 14)
        indicators['momentum_5'] = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5] - 1) if len(market_data) >= 5 else 0
        
        # Volume (if available)
        if 'volume' in market_data.columns:
            indicators['volume'] = market_data['volume'].iloc[-1]
            indicators['volume_ma'] = market_data['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = indicators['volume'] / indicators['volume_ma']
        
        return indicators
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> float:
        """Calculate RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def should_generate_signal(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if strategy should generate a signal
        
        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            
        Returns:
            True if signal generation is allowed
        """
        
        # Check if strategy is enabled
        if not self.config.enabled:
            return False
        
        # Check if symbol is in allowed assets
        if symbol not in self.config.assets:
            return False
        
        # Check minimum time between signals (to avoid overtrading)
        if self.last_signal_time:
            time_diff = current_time - self.last_signal_time
            min_interval = pd.Timedelta(hours=1)  # Minimum 1 hour between signals
            
            if time_diff < min_interval:
                return False
        
        return True
    
    def process_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Process and enhance a trading signal
        
        Args:
            signal: Raw trading signal
            market_data: Current market data
            
        Returns:
            Enhanced trading signal or None if rejected
        """
        
        # Validate signal
        if not self.validate_signal(signal, market_data):
            self.logger.debug(f"Signal validation failed for {signal.symbol}")
            return None
        
        # Calculate SL/TP if not provided
        if signal.stop_loss is None:
            signal.stop_loss = self.calculate_stop_loss(signal, market_data)
        
        if signal.take_profit is None:
            signal.take_profit = self.calculate_take_profit(signal, market_data)
        
        # Set entry price if not provided
        if signal.entry_price is None:
            signal.entry_price = market_data['close'].iloc[-1]
        
        # Update signal metadata
        if signal.metadata is None:
            signal.metadata = {}
        
        signal.metadata.update({
            'strategy_version': self.get_version(),
            'indicators': self.analyze_market_data(signal.symbol, market_data),
            'market_conditions': self._assess_market_conditions(market_data)
        })
        
        # Record signal
        self.record_signal(signal)
        
        self.logger.info(f"Signal generated: {signal.signal_type.value} {signal.symbol} "
                        f"Confidence: {signal.confidence:.2f}")
        
        return signal
    
    def record_signal(self, signal: TradingSignal):
        """Record signal in history"""
        self.signal_history.append(signal)
        self.last_signal_time = signal.timestamp
        
        # Update performance metrics
        self.performance_metrics['total_signals'] += 1
        self.performance_metrics['avg_confidence'] = (
            (self.performance_metrics['avg_confidence'] * (self.performance_metrics['total_signals'] - 1) + 
             signal.confidence) / self.performance_metrics['total_signals']
        )
        
        # Keep history manageable
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]
    
    def update_signal_result(self, signal: TradingSignal, success: bool, pnl: float = 0.0):
        """Update signal result for performance tracking"""
        
        if success:
            self.performance_metrics['successful_signals'] += 1
        else:
            self.performance_metrics['failed_signals'] += 1
        
        self.performance_metrics['total_pnl'] += pnl
        
        # Update win rate
        total_trades = (self.performance_metrics['successful_signals'] + 
                       self.performance_metrics['failed_signals'])
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['successful_signals'] / total_trades
            )
        
        self.logger.debug(f"Signal result updated: {success}, P&L: {pnl:.2f}")
    
    def _assess_market_conditions(self, market_data: pd.DataFrame) -> str:
        """Assess current market conditions"""
        
        # Simple market condition assessment
        volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
        trend = market_data['close'].rolling(50).mean().iloc[-1] - market_data['close'].rolling(200).mean().iloc[-1]
        
        if volatility > 0.02:
            return "high_volatility"
        elif abs(trend) > market_data['close'].iloc[-1] * 0.05:
            return "strong_trend"
        else:
            return "sideways"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return self.performance_metrics.copy()
    
    def get_signal_history(self, symbol: Optional[str] = None, 
                          limit: int = 100) -> List[TradingSignal]:
        """Get signal history"""
        history = self.signal_history
        
        if symbol:
            history = [s for s in history if s.symbol == symbol]
        
        return history[-limit:] if history else []
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0,
            'win_rate': 0.0,
            'total_pnl': 0.0
        }
        
        self.logger.info("Performance metrics reset")
    
    def get_version(self) -> str:
        """Get strategy version"""
        return "1.0.0"
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            'name': self.name,
            'enabled': self.config.enabled,
            'active': self.is_active,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'total_signals': len(self.signal_history),
            'performance': self.performance_metrics,
            'config': {
                'min_confidence': self.config.min_confidence,
                'max_position_size': self.config.max_position_size,
                'risk_per_trade': self.config.risk_per_trade,
                'timeframes': self.config.timeframes,
                'assets': self.config.assets
            }
        }
    
    def start(self):
        """Start the strategy"""
        self.is_active = True
        self.logger.info(f"Strategy {self.name} started")
    
    def stop(self):
        """Stop the strategy"""
        self.is_active = False
        self.logger.info(f"Strategy {self.name} stopped")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update strategy configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {value}")
    
    def export_signals(self, filepath: str, symbol: Optional[str] = None):
        """Export signals to CSV file"""
        
        signals = self.get_signal_history(symbol)
        
        if not signals:
            self.logger.warning("No signals to export")
            return
        
        # Convert to DataFrame
        data = [signal.to_dict() for signal in signals]
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Signals exported to {filepath}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.config.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()
