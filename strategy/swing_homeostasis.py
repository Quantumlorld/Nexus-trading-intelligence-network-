"""
Nexus Trading System - Swing Homeostasis Strategy
Detects liquidity sweeps, structure shifts, and equilibrium restoration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig
from core.logger import get_logger


@dataclass
class MarketStructure:
    """Market structure analysis results"""
    support_levels: List[float]
    resistance_levels: List[float]
    liquidity_zones: List[Dict[str, Any]]
    equilibrium_price: float
    structure_shift_points: List[Dict[str, Any]]
    sweep_zones: List[Dict[str, Any]]


class SwingHomeostasisConfig(StrategyConfig):
    """Configuration for Swing Homeostasis strategy"""
    
    def __init__(self,
                 lookback_period: int = 100,
                 structure_sensitivity: float = 0.02,
                 liquidity_threshold: float = 0.001,
                 equilibrium_window: int = 20,
                 min_sweep_strength: float = 0.003,
                 confirmation_bars: int = 3,
                 **kwargs):
        
        super().__init__(name="Swing_Homeostasis", **kwargs)
        
        self.lookback_period = lookback_period
        self.structure_sensitivity = structure_sensitivity
        self.liquidity_threshold = liquidity_threshold
        self.equilibrium_window = equilibrium_window
        self.min_sweep_strength = min_sweep_strength
        self.confirmation_bars = confirmation_bars
        
        # Update parameters
        self.parameters.update({
            'lookback_period': lookback_period,
            'structure_sensitivity': structure_sensitivity,
            'liquidity_threshold': liquidity_threshold,
            'equilibrium_window': equilibrium_window,
            'min_sweep_strength': min_sweep_strength,
            'confirmation_bars': confirmation_bars
        })


class SwingHomeostasisStrategy(BaseStrategy):
    """
    Swing Homeostasis Strategy - Detects liquidity sweeps, structure shifts, 
    and equilibrium restoration patterns
    """
    
    def __init__(self, config: SwingHomeostasisConfig):
        super().__init__(config)
        self.config = config
        
        # Market structure cache
        self.structure_cache: Dict[str, MarketStructure] = {}
        
        # Liquidity sweep tracking
        self.liquidity_sweeps: Dict[str, List[Dict[str, Any]]] = {}
        
        # Equilibrium tracking
        self.equilibrium_history: Dict[str, List[float]] = {}
        
        self.logger.info("Swing Homeostasis strategy initialized")
    
    def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate swing homeostasis signal
        
        Args:
            symbol: Trading symbol
            market_data: OHLCV market data
            
        Returns:
            TradingSignal or None if no signal
        """
        
        if not self.should_generate_signal(symbol, datetime.now()):
            return None
        
        # Analyze market structure
        structure = self._analyze_market_structure(symbol, market_data)
        
        if structure is None:
            return None
        
        # Detect liquidity sweeps
        sweep_signal = self._detect_liquidity_sweep(symbol, structure, market_data)
        
        if sweep_signal:
            return self.process_signal(sweep_signal, market_data)
        
        # Detect structure shifts
        shift_signal = self._detect_structure_shift(symbol, structure, market_data)
        
        if shift_signal:
            return self.process_signal(shift_signal, market_data)
        
        # Detect equilibrium restoration
        equilibrium_signal = self._detect_equilibrium_restoration(symbol, structure, market_data)
        
        if equilibrium_signal:
            return self.process_signal(equilibrium_signal, market_data)
        
        return None
    
    def validate_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """
        Validate swing homeostasis signal
        
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
        if len(market_data) < self.config.lookback_period:
            return False
        
        # Validate signal-specific conditions
        signal_type = signal.metadata.get('signal_type', '')
        
        if signal_type == 'liquidity_sweep':
            return self._validate_liquidity_sweep_signal(signal, market_data)
        elif signal_type == 'structure_shift':
            return self._validate_structure_shift_signal(signal, market_data)
        elif signal_type == 'equilibrium_restoration':
            return self._validate_equilibrium_restoration_signal(signal, market_data)
        
        return False
    
    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss based on market structure
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Stop loss price
        """
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        structure = self.structure_cache.get(signal.symbol)
        
        if structure is None:
            # Fallback to ATR-based SL
            atr = self._calculate_atr(market_data, 14)
            if signal.signal_type == SignalType.BUY:
                return current_price - (atr * 2)
            else:
                return current_price + (atr * 2)
        
        signal_type = signal.metadata.get('signal_type', '')
        
        if signal_type == 'liquidity_sweep':
            # For sweep signals, place SL beyond the sweep zone
            sweep_info = signal.metadata.get('sweep_info', {})
            sweep_level = sweep_info.get('level', current_price)
            
            if signal.signal_type == SignalType.BUY:
                # Buy after bullish sweep - SL below sweep low
                return sweep_level - (self._calculate_atr(market_data, 14) * 0.5)
            else:
                # Sell after bearish sweep - SL above sweep high
                return sweep_level + (self._calculate_atr(market_data, 14) * 0.5)
        
        elif signal_type == 'structure_shift':
            # For structure shift signals, use recent structure levels
            if signal.signal_type == SignalType.BUY:
                # Buy after bullish shift - SL below nearest support
                support_levels = structure.support_levels
                if support_levels:
                    nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                    return nearest_support - (self._calculate_atr(market_data, 14) * 0.3)
            else:
                # Sell after bearish shift - SL above nearest resistance
                resistance_levels = structure.resistance_levels
                if resistance_levels:
                    nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                    return nearest_resistance + (self._calculate_atr(market_data, 14) * 0.3)
        
        elif signal_type == 'equilibrium_restoration':
            # For equilibrium signals, use equilibrium-based SL
            equilibrium = structure.equilibrium_price
            
            if signal.signal_type == SignalType.BUY:
                # Buy returning to equilibrium - SL below equilibrium with buffer
                return equilibrium - (self._calculate_atr(market_data, 14) * 1.5)
            else:
                # Sell returning to equilibrium - SL above equilibrium with buffer
                return equilibrium + (self._calculate_atr(market_data, 14) * 1.5)
        
        # Fallback
        atr = self._calculate_atr(market_data, 14)
        if signal.signal_type == SignalType.BUY:
            return current_price - (atr * 2)
        else:
            return current_price + (atr * 2)
    
    def calculate_take_profit(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate take profit based on market structure
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Take profit price
        """
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        stop_loss = signal.stop_loss or self.calculate_stop_loss(signal, market_data)
        structure = self.structure_cache.get(signal.symbol)
        
        # Calculate risk amount
        if signal.signal_type == SignalType.BUY:
            risk_amount = current_price - stop_loss
        else:
            risk_amount = stop_loss - current_price
        
        signal_type = signal.metadata.get('signal_type', '')
        
        if signal_type == 'liquidity_sweep':
            # For sweep signals, target opposite structure level
            if structure:
                if signal.signal_type == SignalType.BUY:
                    # Target next resistance level
                    if structure.resistance_levels:
                        next_resistance = min([r for r in structure.resistance_levels if r > current_price],
                                            default=current_price + (risk_amount * 3))
                        return next_resistance
                else:
                    # Target next support level
                    if structure.support_levels:
                        next_support = max([s for s in structure.support_levels if s < current_price],
                                          default=current_price - (risk_amount * 3))
                        return next_support
        
        elif signal_type == 'structure_shift':
            # For structure shift signals, target measured move
            if structure and structure.structure_shift_points:
                last_shift = structure.structure_shift_points[-1]
                measured_move = last_shift.get('measured_move', risk_amount * 3)
                
                if signal.signal_type == SignalType.BUY:
                    return current_price + measured_move
                else:
                    return current_price - measured_move
        
        elif signal_type == 'equilibrium_restoration':
            # For equilibrium signals, target equilibrium restoration
            if structure:
                equilibrium = structure.equilibrium_price
                
                # Add profit target beyond equilibrium
                if signal.signal_type == SignalType.BUY:
                    return equilibrium + (risk_amount * 2)
                else:
                    return equilibrium - (risk_amount * 2)
        
        # Default 3:1 risk/reward
        reward_multiplier = 3.0
        if signal.signal_type == SignalType.BUY:
            return current_price + (risk_amount * reward_multiplier)
        else:
            return current_price - (risk_amount * reward_multiplier)
    
    def _analyze_market_structure(self, symbol: str, market_data: pd.DataFrame) -> Optional[MarketStructure]:
        """Analyze market structure for support, resistance, and liquidity zones"""
        
        # Check cache
        if symbol in self.structure_cache:
            cached_structure = self.structure_cache[symbol]
            # Update if new data available
            if len(market_data) > len(cached_structure.support_levels):
                return self._update_market_structure(symbol, market_data, cached_structure)
            return cached_structure
        
        # Perform full analysis
        try:
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance_levels(market_data)
            
            # Identify liquidity zones
            liquidity_zones = self._identify_liquidity_zones(market_data, support_levels, resistance_levels)
            
            # Calculate equilibrium price
            equilibrium_price = self._calculate_equilibrium_price(market_data)
            
            # Detect structure shift points
            structure_shifts = self._detect_structure_shift_points(market_data)
            
            # Identify sweep zones
            sweep_zones = self._identify_sweep_zones(market_data, support_levels, resistance_levels)
            
            structure = MarketStructure(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                liquidity_zones=liquidity_zones,
                equilibrium_price=equilibrium_price,
                structure_shift_points=structure_shifts,
                sweep_zones=sweep_zones
            )
            
            # Cache the structure
            self.structure_cache[symbol] = structure
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure for {symbol}: {e}")
            return None
    
    def _update_market_structure(self, symbol: str, market_data: pd.DataFrame, 
                               existing_structure: MarketStructure) -> MarketStructure:
        """Update existing market structure with new data"""
        
        # Re-analyze with updated data
        return self._analyze_market_structure(symbol, market_data)
    
    def _find_support_resistance_levels(self, market_data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find significant support and resistance levels"""
        
        highs = market_data['high']
        lows = market_data['low']
        closes = market_data['close']
        
        support_levels = []
        resistance_levels = []
        
        # Use pivot points and swing highs/lows
        window = 10
        
        for i in range(window, len(market_data) - window):
            # Check for potential resistance (swing high)
            current_high = highs.iloc[i]
            is_swing_high = all(highs.iloc[j] <= current_high for j in range(i - window, i + window + 1) if j != i)
            
            if is_swing_high:
                resistance_levels.append(current_high)
            
            # Check for potential support (swing low)
            current_low = lows.iloc[i]
            is_swing_low = all(lows.iloc[j] >= current_low for j in range(i - window, i + window + 1) if j != i)
            
            if is_swing_low:
                support_levels.append(current_low)
        
        # Filter and cluster levels
        support_levels = self._cluster_price_levels(support_levels)
        resistance_levels = self._cluster_price_levels(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _cluster_price_levels(self, levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Cluster similar price levels"""
        
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        clustered = []
        
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] <= tolerance:
                current_cluster.append(level)
            else:
                # Average the cluster and add to result
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _identify_liquidity_zones(self, market_data: pd.DataFrame, 
                                support_levels: List[float], 
                                resistance_levels: List[float]) -> List[Dict[str, Any]]:
        """Identify liquidity accumulation zones"""
        
        liquidity_zones = []
        
        # Analyze volume at support/resistance levels
        if 'volume' in market_data.columns:
            for level in support_levels + resistance_levels:
                # Find bars near this level
                tolerance = level * self.config.structure_sensitivity
                
                near_level_bars = market_data[
                    (abs(market_data['high'] - level) <= tolerance) |
                    (abs(market_data['low'] - level) <= tolerance)
                ]
                
                if len(near_level_bars) > 0:
                    avg_volume = near_level_bars['volume'].mean()
                    total_volume = near_level_bars['volume'].sum()
                    
                    # Determine if this is a significant liquidity zone
                    overall_avg_volume = market_data['volume'].mean()
                    volume_ratio = avg_volume / overall_avg_volume if overall_avg_volume > 0 else 0
                    
                    if volume_ratio > 1.5:  # 50% above average volume
                        liquidity_zones.append({
                            'price_level': level,
                            'type': 'support' if level in support_levels else 'resistance',
                            'volume_ratio': volume_ratio,
                            'total_volume': total_volume,
                            'bar_count': len(near_level_bars),
                            'strength': min(volume_ratio / 2, 1.0)
                        })
        
        return liquidity_zones
    
    def _calculate_equilibrium_price(self, market_data: pd.DataFrame) -> float:
        """Calculate market equilibrium price"""
        
        # Use recent price action to find equilibrium
        recent_data = market_data.tail(self.config.equilibrium_window)
        
        # Calculate volume-weighted average price
        if 'volume' in recent_data.columns:
            typical_prices = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
            vwap = (typical_prices * recent_data['volume']).sum() / recent_data['volume'].sum()
        else:
            # Simple average of high, low, close
            vwap = ((recent_data['high'] + recent_data['low'] + recent_data['close']) / 3).mean()
        
        return vwap
    
    def _detect_structure_shift_points(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect points where market structure shifted"""
        
        shift_points = []
        
        # Look for significant breaks of structure
        closes = market_data['close']
        
        for i in range(50, len(market_data)):  # Start from 50 to have enough history
            current_price = closes.iloc[i]
            historical_prices = closes.iloc[i-50:i]
            
            # Calculate historical range
            historical_high = historical_prices.max()
            historical_low = historical_prices.min()
            historical_range = historical_high - historical_low
            
            # Detect breakout/breakdown
            if historical_range > 0:
                breakout_strength = (current_price - historical_high) / historical_range
                breakdown_strength = (historical_low - current_price) / historical_range
                
                if breakout_strength > self.config.structure_sensitivity:
                    shift_points.append({
                        'timestamp': market_data.index[i],
                        'price': current_price,
                        'type': 'bullish_shift',
                        'strength': breakout_strength,
                        'broken_level': historical_high,
                        'measured_move': historical_range * 1.272  # Fibonacci extension
                    })
                
                elif breakdown_strength > self.config.structure_sensitivity:
                    shift_points.append({
                        'timestamp': market_data.index[i],
                        'price': current_price,
                        'type': 'bearish_shift',
                        'strength': breakdown_strength,
                        'broken_level': historical_low,
                        'measured_move': historical_range * 1.272
                    })
        
        return shift_points
    
    def _identify_sweep_zones(self, market_data: pd.DataFrame,
                            support_levels: List[float], 
                            resistance_levels: List[float]) -> List[Dict[str, Any]]:
        """Identify potential liquidity sweep zones"""
        
        sweep_zones = []
        
        # Look for price action that sweeps liquidity levels
        for level in support_levels + resistance_levels:
            # Find bars that touched or slightly beyond the level
            tolerance = level * self.config.liquidity_threshold
            
            touch_bars = market_data[
                ((market_data['low'] <= level + tolerance) & (market_data['low'] >= level - tolerance)) |
                ((market_data['high'] <= level + tolerance) & (market_data['high'] >= level - tolerance))
            ]
            
            if len(touch_bars) >= 2:  # Need at least 2 touches for sweep pattern
                # Analyze the pattern
                for i in range(1, len(touch_bars)):
                    prev_bar = touch_bars.iloc[i-1]
                    curr_bar = touch_bars.iloc[i]
                    
                    # Check for sweep pattern (push beyond level then reverse)
                    if level in support_levels:
                        # Bullish sweep: push below support then reverse up
                        if (curr_bar['low'] < level and 
                            curr_bar['close'] > prev_bar['close'] and
                            curr_bar['close'] > level):
                            
                            sweep_strength = (level - curr_bar['low']) / level
                            
                            if sweep_strength >= self.config.min_sweep_strength:
                                sweep_zones.append({
                                    'level': level,
                                    'type': 'bullish_sweep',
                                    'sweep_low': curr_bar['low'],
                                    'reversal_bar': curr_bar.name,
                                    'strength': sweep_strength,
                                    'volume': curr_bar.get('volume', 0)
                                })
                    
                    else:  # resistance level
                        # Bearish sweep: push above resistance then reverse down
                        if (curr_bar['high'] > level and 
                            curr_bar['close'] < prev_bar['close'] and
                            curr_bar['close'] < level):
                            
                            sweep_strength = (curr_bar['high'] - level) / level
                            
                            if sweep_strength >= self.config.min_sweep_strength:
                                sweep_zones.append({
                                    'level': level,
                                    'type': 'bearish_sweep',
                                    'sweep_high': curr_bar['high'],
                                    'reversal_bar': curr_bar.name,
                                    'strength': sweep_strength,
                                    'volume': curr_bar.get('volume', 0)
                                })
        
        return sweep_zones
    
    def _detect_liquidity_sweep(self, symbol: str, structure: MarketStructure, 
                               market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Detect recent liquidity sweep patterns"""
        
        recent_sweeps = [sweep for sweep in structure.sweep_zones 
                       if (datetime.now() - sweep['reversal_bar']).total_seconds() < 3600]
        
        if not recent_sweeps:
            return None
        
        # Get the most recent sweep
        latest_sweep = max(recent_sweeps, key=lambda x: x['reversal_bar'])
        
        current_price = market_data['close'].iloc[-1]
        
        # Generate signal based on sweep type
        if latest_sweep['type'] == 'bullish_sweep':
            # Buy signal after bullish sweep
            confidence = min(latest_sweep['strength'] * 100, 0.9)
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy=self.name,
                timestamp=datetime.now(),
                entry_price=current_price,
                reason=f"Bullish liquidity sweep at {latest_sweep['level']:.4f}",
                metadata={
                    'signal_type': 'liquidity_sweep',
                    'sweep_info': latest_sweep,
                    'sweep_level': latest_sweep['level']
                }
            )
            
            return signal
        
        elif latest_sweep['type'] == 'bearish_sweep':
            # Sell signal after bearish sweep
            confidence = min(latest_sweep['strength'] * 100, 0.9)
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy=self.name,
                timestamp=datetime.now(),
                entry_price=current_price,
                reason=f"Bearish liquidity sweep at {latest_sweep['level']:.4f}",
                metadata={
                    'signal_type': 'liquidity_sweep',
                    'sweep_info': latest_sweep,
                    'sweep_level': latest_sweep['level']
                }
            )
            
            return signal
        
        return None
    
    def _detect_structure_shift(self, symbol: str, structure: MarketStructure, 
                               market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Detect recent structure shift patterns"""
        
        if not structure.structure_shift_points:
            return None
        
        # Get recent shifts
        recent_shifts = [shift for shift in structure.structure_shift_points
                        if (datetime.now() - shift['timestamp']).total_seconds() < 7200]  # 2 hours
        
        if not recent_shifts:
            return None
        
        latest_shift = recent_shifts[-1]
        current_price = market_data['close'].iloc[-1]
        
        # Check if price is confirming the shift
        if latest_shift['type'] == 'bullish_shift':
            # Price should remain above broken level
            if current_price > latest_shift['broken_level']:
                confidence = min(latest_shift['strength'] * 80, 0.8)
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    reason=f"Bullish structure shift above {latest_shift['broken_level']:.4f}",
                    metadata={
                        'signal_type': 'structure_shift',
                        'shift_info': latest_shift,
                        'broken_level': latest_shift['broken_level']
                    }
                )
                
                return signal
        
        elif latest_shift['type'] == 'bearish_shift':
            # Price should remain below broken level
            if current_price < latest_shift['broken_level']:
                confidence = min(latest_shift['strength'] * 80, 0.8)
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    reason=f"Bearish structure shift below {latest_shift['broken_level']:.4f}",
                    metadata={
                        'signal_type': 'structure_shift',
                        'shift_info': latest_shift,
                        'broken_level': latest_shift['broken_level']
                    }
                )
                
                return signal
        
        return None
    
    def _detect_equilibrium_restoration(self, symbol: str, structure: MarketStructure, 
                                      market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Detect equilibrium restoration patterns"""
        
        current_price = market_data['close'].iloc[-1]
        equilibrium = structure.equilibrium_price
        
        # Calculate distance from equilibrium
        distance_pct = abs(current_price - equilibrium) / equilibrium
        
        # Only trade if significantly away from equilibrium
        if distance_pct < self.config.structure_sensitivity:
            return None
        
        # Check for mean reversion signals
        recent_data = market_data.tail(10)
        
        # Look for reversal patterns toward equilibrium
        if current_price > equilibrium:
            # Price above equilibrium, look for bearish reversal
            if (recent_data['close'].iloc[-1] < recent_data['close'].iloc[-2] and
                recent_data['close'].iloc[-1] < recent_data['high'].tail(5).max()):
                
                confidence = min(distance_pct * 50, 0.7)
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    reason=f"Equilibrium restoration from above (equilibrium: {equilibrium:.4f})",
                    metadata={
                        'signal_type': 'equilibrium_restoration',
                        'equilibrium_price': equilibrium,
                        'distance_from_equilibrium': distance_pct
                    }
                )
                
                return signal
        
        else:  # current_price < equilibrium
            # Price below equilibrium, look for bullish reversal
            if (recent_data['close'].iloc[-1] > recent_data['close'].iloc[-2] and
                recent_data['close'].iloc[-1] > recent_data['low'].tail(5).min()):
                
                confidence = min(distance_pct * 50, 0.7)
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    reason=f"Equilibrium restoration from below (equilibrium: {equilibrium:.4f})",
                    metadata={
                        'signal_type': 'equilibrium_restoration',
                        'equilibrium_price': equilibrium,
                        'distance_from_equilibrium': distance_pct
                    }
                )
                
                return signal
        
        return None
    
    def _validate_liquidity_sweep_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """Validate liquidity sweep signal"""
        
        sweep_info = signal.metadata.get('sweep_info', {})
        
        # Check if sweep is recent enough
        if not sweep_info:
            return False
        
        # Validate sweep strength
        if sweep_info.get('strength', 0) < self.config.min_sweep_strength:
            return False
        
        # Check volume confirmation
        if 'volume' in market_data.columns:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].tail(20).mean()
            
            if current_volume < avg_volume * 0.8:  # Need volume confirmation
                return False
        
        return True
    
    def _validate_structure_shift_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """Validate structure shift signal"""
        
        shift_info = signal.metadata.get('shift_info', {})
        
        if not shift_info:
            return False
        
        # Check if shift is still valid
        broken_level = shift_info.get('broken_level', 0)
        current_price = market_data['close'].iloc[-1]
        
        if signal.signal_type == SignalType.BUY:
            # For bullish shift, price should stay above broken level
            return current_price > broken_level
        else:
            # For bearish shift, price should stay below broken level
            return current_price < broken_level
    
    def _validate_equilibrium_restoration_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """Validate equilibrium restoration signal"""
        
        equilibrium = signal.metadata.get('equilibrium_price', 0)
        current_price = market_data['close'].iloc[-1]
        
        if equilibrium == 0:
            return False
        
        # Check if we're still significantly away from equilibrium
        distance_pct = abs(current_price - equilibrium) / equilibrium
        
        return distance_pct >= self.config.structure_sensitivity
    
    def get_market_structure(self, symbol: str, market_data: pd.DataFrame) -> Optional[MarketStructure]:
        """Get current market structure for a symbol"""
        return self._analyze_market_structure(symbol, market_data)
    
    def get_strategy_status(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed strategy status for a symbol"""
        
        base_status = self.get_status()
        
        # Add swing homeostasis specific information
        structure = self.get_market_structure(symbol, market_data)
        
        if structure:
            homeostasis_status = {
                'support_levels': structure.support_levels,
                'resistance_levels': structure.resistance_levels,
                'equilibrium_price': structure.equilibrium_price,
                'current_distance_from_equilibrium': abs(market_data['close'].iloc[-1] - structure.equilibrium_price) / structure.equilibrium_price,
                'recent_sweeps': len([s for s in structure.sweep_zones if (datetime.now() - s['reversal_bar']).total_seconds() < 3600]),
                'structure_shifts': len(structure.structure_shift_points),
                'liquidity_zones': len(structure.liquidity_zones)
            }
        else:
            homeostasis_status = {
                'error': 'Could not analyze market structure'
            }
        
        base_status['homeostasis_analysis'] = homeostasis_status
        
        return base_status
    
    def reset_strategy(self):
        """Reset strategy state"""
        super().reset_performance_metrics()
        self.structure_cache.clear()
        self.liquidity_sweeps.clear()
        self.equilibrium_history.clear()
        
        self.logger.info("Swing Homeostasis strategy reset")
    
    def get_version(self) -> str:
        """Get strategy version"""
        return "1.2.0"
