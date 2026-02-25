"""
Nexus Trading System - EMA Crossover Strategy
Fast/slow EMA filter strategy for trend following
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig
from core.logger import get_logger


class EMACrossoverConfig(StrategyConfig):
    """Configuration for EMA crossover strategy"""
    
    def __init__(self, 
                 fast_ema_period: int = 12,
                 slow_ema_period: int = 26,
                 signal_ema_period: int = 9,
                 min_crossover_strength: float = 0.001,
                 confirmation_candles: int = 1,
                 **kwargs):
        
        super().__init__(name="EMA_Crossover", **kwargs)
        
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.signal_ema_period = signal_ema_period
        self.min_crossover_strength = min_crossover_strength
        self.confirmation_candles = confirmation_candles
        
        # Update parameters
        self.parameters.update({
            'fast_ema_period': fast_ema_period,
            'slow_ema_period': slow_ema_period,
            'signal_ema_period': signal_ema_period,
            'min_crossover_strength': min_crossover_strength,
            'confirmation_candles': confirmation_candles
        })


class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover Strategy - Fast/slow EMA filter for trend following"""
    
    def __init__(self, config: EMACrossoverConfig):
        super().__init__(config)
        self.config = config
        
        # EMA cache
        self.ema_cache: Dict[str, Dict[str, pd.Series]] = {}
        
        # Crossover tracking
        self.last_crossover: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("EMA Crossover strategy initialized")
    
    def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate EMA crossover signal
        
        Args:
            symbol: Trading symbol
            market_data: OHLCV market data
            
        Returns:
            TradingSignal or None if no signal
        """
        
        if not self.should_generate_signal(symbol, datetime.now()):
            return None
        
        # Calculate EMAs
        emas = self._calculate_emas(symbol, market_data)
        
        if emas is None:
            return None
        
        # Check for crossover
        crossover_signal = self._detect_crossover(symbol, emas, market_data)
        
        if crossover_signal is None:
            return None
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=crossover_signal['type'],
            confidence=crossover_signal['confidence'],
            strategy=self.name,
            timestamp=datetime.now(),
            entry_price=market_data['close'].iloc[-1],
            timeframe=self._get_current_timeframe(market_data),
            reason=crossover_signal['reason'],
            metadata={
                'emas': {
                    'fast_ema': emas['fast_ema'].iloc[-1],
                    'slow_ema': emas['slow_ema'].iloc[-1],
                    'signal_ema': emas['signal_ema'].iloc[-1]
                },
                'crossover_strength': crossover_signal['strength'],
                'crossover_direction': crossover_signal['direction']
            }
        )
        
        return self.process_signal(signal, market_data)
    
    def validate_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """
        Validate EMA crossover signal
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid
        """
        
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check if we have enough data
        if len(market_data) < max(self.config.fast_ema_period, self.config.slow_ema_period) + 10:
            return False
        
        # Check market conditions
        indicators = self.analyze_market_data(signal.symbol, market_data)
        
        # Avoid signals in extremely low volatility
        if indicators.get('volatility', 0) < 0.001:
            return False
        
        # Check for recent crossover (avoid duplicate signals)
        last_crossover_info = self.last_crossover.get(signal.symbol)
        if last_crossover_info:
            time_diff = datetime.now() - last_crossover_info['timestamp']
            if time_diff.total_seconds() < 3600:  # 1 hour minimum between signals
                return False
        
        # Validate crossover strength
        crossover_strength = signal.metadata.get('crossover_strength', 0)
        if abs(crossover_strength) < self.config.min_crossover_strength:
            return False
        
        return True
    
    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss based on ATR and EMA levels
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Stop loss price
        """
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        atr = self._calculate_atr(market_data, 14)
        
        # Get EMA levels for reference
        emas = self._calculate_emas(signal.symbol, market_data)
        fast_ema = emas['fast_ema'].iloc[-1] if emas else current_price
        
        # Calculate SL based on signal type
        if signal.signal_type == SignalType.BUY:
            # For buy signals, place SL below recent low or fast EMA
            recent_low = market_data['low'].rolling(14).min().iloc[-1]
            sl_atr = current_price - (atr * 2)  # 2x ATR below entry
            sl_ema = fast_ema - (atr * 0.5)  # Below fast EMA with buffer
            
            stop_loss = max(recent_low, sl_atr, sl_ema)
            
        else:  # SELL signal
            # For sell signals, place SL above recent high or fast EMA
            recent_high = market_data['high'].rolling(14).max().iloc[-1]
            sl_atr = current_price + (atr * 2)  # 2x ATR above entry
            sl_ema = fast_ema + (atr * 0.5)  # Above fast EMA with buffer
            
            stop_loss = min(recent_high, sl_atr, sl_ema)
        
        return stop_loss
    
    def calculate_take_profit(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate take profit based on risk/reward ratio and EMA levels
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Take profit price
        """
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        stop_loss = signal.stop_loss or self.calculate_stop_loss(signal, market_data)
        
        # Risk amount
        if signal.signal_type == SignalType.BUY:
            risk_amount = current_price - stop_loss
        else:
            risk_amount = stop_loss - current_price
        
        # Use 3:1 risk/reward ratio (configurable)
        reward_multiplier = self.config.parameters.get('reward_multiplier', 3.0)
        reward_amount = risk_amount * reward_multiplier
        
        # Calculate TP
        if signal.signal_type == SignalType.BUY:
            take_profit = current_price + reward_amount
        else:
            take_profit = current_price - reward_amount
        
        # Adjust for significant EMA levels
        emas = self._calculate_emas(signal.symbol, market_data)
        if emas:
            slow_ema = emas['slow_ema'].iloc[-1]
            
            if signal.signal_type == SignalType.BUY:
                # For buy signals, if TP is below slow EMA, extend it
                if take_profit < slow_ema:
                    take_profit = slow_ema + (reward_amount * 0.5)
            else:
                # For sell signals, if TP is above slow EMA, extend it
                if take_profit > slow_ema:
                    take_profit = slow_ema - (reward_amount * 0.5)
        
        return take_profit
    
    def _calculate_emas(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
        """Calculate EMA values for the symbol"""
        
        # Check cache
        if symbol in self.ema_cache:
            cached_emas = self.ema_cache[symbol]
            # Check if cache is up to date
            if len(cached_emas['fast_ema']) == len(market_data):
                return cached_emas
        
        # Calculate EMAs
        try:
            emas = {
                'fast_ema': market_data['close'].ewm(span=self.config.fast_ema_period).mean(),
                'slow_ema': market_data['close'].ewm(span=self.config.slow_ema_period).mean(),
                'signal_ema': market_data['close'].ewm(span=self.config.signal_ema_period).mean()
            }
            
            # Cache the results
            self.ema_cache[symbol] = emas
            
            return emas
            
        except Exception as e:
            self.logger.error(f"Error calculating EMAs for {symbol}: {e}")
            return None
    
    def _detect_crossover(self, symbol: str, emas: Dict[str, pd.Series], 
                         market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect EMA crossover signals"""
        
        if len(emas['fast_ema']) < 2 or len(emas['slow_ema']) < 2:
            return None
        
        # Get current and previous values
        current_fast = emas['fast_ema'].iloc[-1]
        current_slow = emas['slow_ema'].iloc[-1]
        current_signal = emas['signal_ema'].iloc[-1]
        
        prev_fast = emas['fast_ema'].iloc[-2]
        prev_slow = emas['slow_ema'].iloc[-2]
        prev_signal = emas['signal_ema'].iloc[-2]
        
        # Detect crossovers
        crossover_signal = None
        
        # Bullish crossover (fast EMA crosses above slow EMA)
        if prev_fast <= prev_slow and current_fast > current_slow:
            # Check for confirmation
            if self._confirm_crossover('bullish', emas, market_data):
                strength = (current_fast - current_slow) / current_slow
                confidence = min(abs(strength) * 100, 1.0)
                
                crossover_signal = {
                    'type': SignalType.BUY,
                    'direction': 'bullish',
                    'strength': strength,
                    'confidence': confidence,
                    'reason': f"Bullish EMA crossover: Fast({current_fast:.4f}) > Slow({current_slow:.4f})"
                }
        
        # Bearish crossover (fast EMA crosses below slow EMA)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            # Check for confirmation
            if self._confirm_crossover('bearish', emas, market_data):
                strength = (current_slow - current_fast) / current_slow
                confidence = min(abs(strength) * 100, 1.0)
                
                crossover_signal = {
                    'type': SignalType.SELL,
                    'direction': 'bearish',
                    'strength': strength,
                    'confidence': confidence,
                    'reason': f"Bearish EMA crossover: Fast({current_fast:.4f}) < Slow({current_slow:.4f})"
                }
        
        # Signal line crossover (MACD-style)
        elif prev_signal <= prev_slow and current_signal > current_slow:
            if current_fast > current_slow:  # Only if already in uptrend
                strength = (current_signal - current_slow) / current_slow
                confidence = min(abs(strength) * 80, 0.8)  # Slightly lower confidence
                
                crossover_signal = {
                    'type': SignalType.BUY,
                    'direction': 'signal_bullish',
                    'strength': strength,
                    'confidence': confidence,
                    'reason': f"Signal line bullish crossover: Signal({current_signal:.4f}) > Slow({current_slow:.4f})"
                }
        
        elif prev_signal >= prev_slow and current_signal < current_slow:
            if current_fast < current_slow:  # Only if already in downtrend
                strength = (current_slow - current_signal) / current_slow
                confidence = min(abs(strength) * 80, 0.8)
                
                crossover_signal = {
                    'type': SignalType.SELL,
                    'direction': 'signal_bearish',
                    'strength': strength,
                    'confidence': confidence,
                    'reason': f"Signal line bearish crossover: Signal({current_signal:.4f}) < Slow({current_slow:.4f})"
                }
        
        # Record crossover if detected
        if crossover_signal:
            self.last_crossover[symbol] = {
                'timestamp': datetime.now(),
                'type': crossover_signal['type'].value,
                'direction': crossover_signal['direction'],
                'strength': crossover_signal['strength']
            }
        
        return crossover_signal
    
    def _confirm_crossover(self, direction: str, emas: Dict[str, pd.Series], 
                          market_data: pd.DataFrame) -> bool:
        """Confirm crossover with additional conditions"""
        
        if self.config.confirmation_candles == 0:
            return True
        
        # Check recent price action
        recent_candles = min(self.config.confirmation_candles + 1, len(market_data))
        recent_data = market_data.tail(recent_candles)
        
        if direction == 'bullish':
            # For bullish crossover, look for confirming price action
            # - Price should be closing higher
            # - Volume should be increasing (if available)
            
            price_confirmation = recent_data['close'].is_monotonic_increasing()
            
            volume_confirmation = True
            if 'volume' in recent_data.columns:
                volume_confirmation = recent_data['volume'].is_monotonic_increasing()
            
            return price_confirmation and volume_confirmation
            
        else:  # bearish
            # For bearish crossover, look for confirming price action
            # - Price should be closing lower
            # - Volume should be increasing
            
            price_confirmation = recent_data['close'].is_monotonic_decreasing()
            
            volume_confirmation = True
            if 'volume' in recent_data.columns:
                volume_confirmation = recent_data['volume'].is_monotonic_increasing()
            
            return price_confirmation and volume_confirmation
    
    def _get_current_timeframe(self, market_data: pd.DataFrame) -> str:
        """Estimate current timeframe from data frequency"""
        
        if len(market_data) < 2:
            return "Unknown"
        
        # Calculate time difference between consecutive candles
        time_diff = market_data.index[1] - market_data.index[0]
        
        # Map to common timeframes
        timeframe_map = {
            pd.Timedelta(minutes=1): "1M",
            pd.Timedelta(minutes=5): "5M",
            pd.Timedelta(minutes=15): "15M",
            pd.Timedelta(minutes=30): "30M",
            pd.Timedelta(hours=1): "1H",
            pd.Timedelta(hours=4): "4H",
            pd.Timedelta(days=1): "1D",
            pd.Timedelta(weeks=1): "1W"
        }
        
        return timeframe_map.get(time_diff, "Custom")
    
    def get_ema_values(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get current EMA values for a symbol"""
        
        emas = self._calculate_emas(symbol, market_data)
        
        if emas is None:
            return None
        
        return {
            'fast_ema': emas['fast_ema'].iloc[-1],
            'slow_ema': emas['slow_ema'].iloc[-1],
            'signal_ema': emas['signal_ema'].iloc[-1]
        }
    
    def get_trend_direction(self, symbol: str, market_data: pd.DataFrame) -> str:
        """Get current trend direction based on EMA positions"""
        
        ema_values = self.get_ema_values(symbol, market_data)
        
        if ema_values is None:
            return "Unknown"
        
        fast_ema = ema_values['fast_ema']
        slow_ema = ema_values['slow_ema']
        
        if fast_ema > slow_ema:
            return "Bullish"
        elif fast_ema < slow_ema:
            return "Bearish"
        else:
            return "Sideways"
    
    def get_strategy_status(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed strategy status for a symbol"""
        
        base_status = self.get_status()
        
        # Add EMA-specific information
        ema_values = self.get_ema_values(symbol, market_data)
        trend_direction = self.get_trend_direction(symbol, market_data)
        
        ema_status = {
            'current_ema_values': ema_values,
            'trend_direction': trend_direction,
            'last_crossover': self.last_crossover.get(symbol),
            'ema_distance': (ema_values['fast_ema'] - ema_values['slow_ema']) / ema_values['slow_ema'] if ema_values else 0
        }
        
        base_status['ema_analysis'] = ema_status
        
        return base_status
    
    def reset_strategy(self):
        """Reset strategy state"""
        super().reset_performance_metrics()
        self.ema_cache.clear()
        self.last_crossover.clear()
        
        self.logger.info("EMA Crossover strategy reset")
    
    def get_version(self) -> str:
        """Get strategy version"""
        return "1.1.0"
