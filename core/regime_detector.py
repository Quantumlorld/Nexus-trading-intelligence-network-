"""
Nexus Trading System - Market Regime Detector
Identifies market conditions and regimes for adaptive trading
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"           # Strong uptrend
    BEAR = "bear"           # Strong downtrend  
    SIDEWAYS = "sideways"   # Range-bound
    VOLATILE = "volatile"   # High volatility, unclear direction
    TRANSITION = "transition" # Regime change in progress


@dataclass
class RegimeSignal:
    """Market regime detection signal"""
    regime: MarketRegime
    confidence: float  # 0-1
    strength: float    # 0-1, how strong the regime is
    volatility_level: str  # low, medium, high
    trend_strength: float  # 0-1
    expected_duration: timedelta
    key_indicators: Dict[str, float]
    timestamp: datetime


class RegimeDetector:
    """Detects and tracks market regimes for adaptive strategy selection"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        
        # Regime history
        self.regime_history: List[RegimeSignal] = []
        self.current_regime: Optional[RegimeSignal] = None
        
        # Indicator weights for regime detection
        self.indicator_weights = {
            'trend_strength': 0.3,
            'volatility': 0.2,
            'momentum': 0.2,
            'volume': 0.15,
            'price_action': 0.15
        }
        
        # Regime transition thresholds
        self.transition_threshold = 0.6  # Confidence needed to change regime
        self.min_regime_duration = timedelta(hours=4)  # Minimum time before regime change
        
    def detect_regime(self, symbol: str, price_data: pd.DataFrame, 
                     volume_data: Optional[pd.Series] = None) -> RegimeSignal:
        """
        Detect current market regime from price and volume data
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with OHLC data
            volume_data: Optional volume data
            
        Returns:
            RegimeSignal with current regime assessment
        """
        
        if len(price_data) < self.lookback_period:
            self.logger.warning(f"Insufficient data for regime detection: {len(price_data)} < {self.lookback_period}")
            return self._create_default_regime()
        
        # Calculate indicators
        indicators = self._calculate_regime_indicators(price_data, volume_data)
        
        # Determine regime based on indicators
        regime, confidence = self._determine_regime(indicators)
        
        # Calculate regime strength
        strength = self._calculate_regime_strength(indicators, regime)
        
        # Determine volatility level
        volatility_level = self._classify_volatility(indicators['volatility'])
        
        # Estimate expected duration
        expected_duration = self._estimate_regime_duration(regime, indicators)
        
        # Create regime signal
        regime_signal = RegimeSignal(
            regime=regime,
            confidence=confidence,
            strength=strength,
            volatility_level=volatility_level,
            trend_strength=indicators['trend_strength'],
            expected_duration=expected_duration,
            key_indicators=indicators,
            timestamp=datetime.now()
        )
        
        # Update regime history
        self._update_regime_history(regime_signal)
        
        self.logger.info(f"Regime detected for {symbol}: {regime.value} (confidence: {confidence:.2f})")
        
        return regime_signal
    
    def _calculate_regime_indicators(self, price_data: pd.DataFrame, 
                                   volume_data: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate indicators for regime detection"""
        
        indicators = {}
        
        # Trend strength using ADX-like calculation
        indicators['trend_strength'] = self._calculate_trend_strength(price_data)
        
        # Volatility
        indicators['volatility'] = self._calculate_volatility(price_data)
        
        # Momentum
        indicators['momentum'] = self._calculate_momentum(price_data)
        
        # Volume analysis
        indicators['volume_strength'] = self._calculate_volume_strength(price_data, volume_data)
        
        # Price action patterns
        indicators['price_action_score'] = self._calculate_price_action_score(price_data)
        
        # Directional bias
        indicators['directional_bias'] = self._calculate_directional_bias(price_data)
        
        # Support/resistance levels
        indicators['sr_strength'] = self._calculate_support_resistance_strength(price_data)
        
        return indicators
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        
        closes = price_data['close']
        
        # Calculate linear regression slope
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        
        # Normalize slope by price level
        avg_price = closes.mean()
        normalized_slope = abs(slope) / avg_price
        
        # Calculate R-squared for trend consistency
        y_pred = slope * x + closes.iloc[0]
        ss_res = np.sum((closes - y_pred) ** 2)
        ss_tot = np.sum((closes - closes.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Combine slope magnitude and consistency
        trend_strength = normalized_slope * 1000 * r_squared  # Scale factor
        return min(trend_strength, 1.0)
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility (0-1)"""
        
        returns = price_data['close'].pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Normalize to 0-1 range (typical range 0-5% daily vol)
        normalized_vol = min(volatility * np.sqrt(252) / 0.3, 1.0)  # 30% annual vol = max
        
        return normalized_vol
    
    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum strength (0-1)"""
        
        closes = price_data['close']
        
        # Multiple timeframe momentum
        momentum_5 = (closes.iloc[-1] / closes.iloc[-5] - 1) if len(closes) >= 5 else 0
        momentum_10 = (closes.iloc[-1] / closes.iloc[-10] - 1) if len(closes) >= 10 else 0
        momentum_20 = (closes.iloc[-1] / closes.iloc[-20] - 1) if len(closes) >= 20 else 0
        
        # Combine momenta
        combined_momentum = abs(momentum_5) * 0.5 + abs(momentum_10) * 0.3 + abs(momentum_20) * 0.2
        
        # Normalize to 0-1
        return min(combined_momentum * 10, 1.0)
    
    def _calculate_volume_strength(self, price_data: pd.DataFrame, 
                                 volume_data: Optional[pd.Series] = None) -> float:
        """Calculate volume strength (0-1)"""
        
        if volume_data is None:
            # Use tick volume from price data if available
            if 'volume' in price_data.columns:
                volume_data = price_data['volume']
            else:
                return 0.5  # Default if no volume data
        
        # Compare recent volume to historical average
        recent_volume = volume_data.tail(10).mean()
        avg_volume = volume_data.mean()
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Normalize to 0-1 (0.5x to 3x volume range)
        normalized_volume = (volume_ratio - 0.5) / 2.5
        return max(0, min(normalized_volume, 1.0))
    
    def _calculate_price_action_score(self, price_data: pd.DataFrame) -> float:
        """Calculate price action pattern score (0-1)"""
        
        # Look for strong price action patterns
        scores = []
        
        # Engulfing patterns
        for i in range(2, len(price_data)):
            current = price_data.iloc[i]
            prev = price_data.iloc[i-1]
            
            # Bullish engulfing
            if (prev['close'] < prev['open'] and 
                current['close'] > current['open'] and
                current['open'] < prev['close'] and
                current['close'] > prev['open']):
                scores.append(0.8)
            
            # Bearish engulfing
            elif (prev['close'] > prev['open'] and 
                  current['close'] < current['open'] and
                  current['open'] > prev['close'] and
                  current['close'] < prev['open']):
                scores.append(0.8)
        
        # Strong directional candles
        for i in range(1, len(price_data)):
            current = price_data.iloc[i]
            body_size = abs(current['close'] - current['open'])
            range_size = current['high'] - current['low']
            
            if range_size > 0:
                body_ratio = body_size / range_size
                if body_ratio > 0.7:  # Strong candle
                    scores.append(0.6)
        
        return np.mean(scores) if scores else 0.3
    
    def _calculate_directional_bias(self, price_data: pd.DataFrame) -> float:
        """Calculate directional bias (-1 to 1)"""
        
        closes = price_data['close']
        
        # Calculate short and long term moving averages
        short_ma = closes.tail(20).mean()
        long_ma = closes.tail(50).mean()
        
        # Calculate bias
        if long_ma > 0:
            bias = (short_ma - long_ma) / long_ma
        else:
            bias = 0
        
        return np.clip(bias * 10, -1, 1)  # Scale and clip
    
    def _calculate_support_resistance_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate support/resistance level strength (0-1)"""
        
        highs = price_data['high']
        lows = price_data['low']
        
        # Find recent highs and lows
        recent_highs = highs.tail(20).nlargest(3)
        recent_lows = lows.tail(20).nsmallest(3)
        
        # Calculate how many times price tested these levels
        current_price = price_data['close'].iloc[-1]
        
        # Count touches near resistance
        resistance_touches = sum(1 for h in recent_highs if abs(h - current_price) / current_price < 0.01)
        
        # Count touches near support  
        support_touches = sum(1 for l in recent_lows if abs(l - current_price) / current_price < 0.01)
        
        total_touches = resistance_touches + support_touches
        
        # Normalize to 0-1
        return min(total_touches / 6.0, 1.0)
    
    def _determine_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Determine market regime from indicators"""
        
        regime_scores = {}
        
        # Bull regime conditions
        bull_score = 0
        if indicators['directional_bias'] > 0.3:
            bull_score += 0.3
        if indicators['trend_strength'] > 0.6:
            bull_score += 0.3
        if indicators['momentum'] > 0.5:
            bull_score += 0.2
        if indicators['volume_strength'] > 0.6:
            bull_score += 0.2
        regime_scores[MarketRegime.BULL] = bull_score
        
        # Bear regime conditions
        bear_score = 0
        if indicators['directional_bias'] < -0.3:
            bear_score += 0.3
        if indicators['trend_strength'] > 0.6:
            bear_score += 0.3
        if indicators['momentum'] > 0.5:
            bear_score += 0.2
        if indicators['volume_strength'] > 0.6:
            bear_score += 0.2
        regime_scores[MarketRegime.BEAR] = bear_score
        
        # Sideways regime conditions
        sideways_score = 0
        if abs(indicators['directional_bias']) < 0.2:
            sideways_score += 0.4
        if indicators['trend_strength'] < 0.4:
            sideways_score += 0.3
        if indicators['sr_strength'] > 0.6:
            sideways_score += 0.3
        regime_scores[MarketRegime.SIDEWAYS] = sideways_score
        
        # Volatile regime conditions
        volatile_score = 0
        if indicators['volatility'] > 0.7:
            volatile_score += 0.4
        if indicators['trend_strength'] < 0.3:
            volatile_score += 0.3
        if indicators['price_action_score'] > 0.6:
            volatile_score += 0.3
        regime_scores[MarketRegime.VOLATILE] = volatile_score
        
        # Determine best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        return best_regime, confidence
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], 
                                 regime: MarketRegime) -> float:
        """Calculate how strong the current regime is"""
        
        strength_factors = []
        
        # Trend alignment
        if regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            strength_factors.append(indicators['trend_strength'])
            strength_factors.append(abs(indicators['directional_bias']))
        elif regime == MarketRegime.SIDEWAYS:
            strength_factors.append(1 - indicators['trend_strength'])
            strength_factors.append(indicators['sr_strength'])
        elif regime == MarketRegime.VOLATILE:
            strength_factors.append(indicators['volatility'])
            strength_factors.append(indicators['price_action_score'])
        
        # Volume confirmation
        strength_factors.append(indicators['volume_strength'])
        
        # Momentum confirmation
        if regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            strength_factors.append(indicators['momentum'])
        
        return np.mean(strength_factors)
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.3:
            return "low"
        elif volatility < 0.7:
            return "medium"
        else:
            return "high"
    
    def _estimate_regime_duration(self, regime: MarketRegime, 
                                indicators: Dict[str, float]) -> timedelta:
        """Estimate how long the current regime might last"""
        
        base_durations = {
            MarketRegime.BULL: timedelta(hours=12),
            MarketRegime.BEAR: timedelta(hours=12),
            MarketRegime.SIDEWAYS: timedelta(hours=8),
            MarketRegime.VOLATILE: timedelta(hours=4),
            MarketRegime.TRANSITION: timedelta(hours=2)
        }
        
        base_duration = base_durations.get(regime, timedelta(hours=6))
        
        # Adjust based on trend strength and volatility
        if indicators['trend_strength'] > 0.7:
            duration_multiplier = 1.5  # Strong trends last longer
        elif indicators['volatility'] > 0.8:
            duration_multiplier = 0.5  # High volatility means shorter duration
        else:
            duration_multiplier = 1.0
        
        adjusted_duration = timedelta(
            hours=int(base_duration.total_seconds() / 3600 * duration_multiplier)
        )
        
        return adjusted_duration
    
    def _create_default_regime(self) -> RegimeSignal:
        """Create default regime when insufficient data"""
        return RegimeSignal(
            regime=MarketRegime.TRANSITION,
            confidence=0.3,
            strength=0.3,
            volatility_level="medium",
            trend_strength=0.3,
            expected_duration=timedelta(hours=2),
            key_indicators={},
            timestamp=datetime.now()
        )
    
    def _update_regime_history(self, new_regime: RegimeSignal):
        """Update regime history and current regime"""
        
        # Check if this is a genuine regime change
        if (self.current_regime is None or 
            new_regime.regime != self.current_regime.regime or
            new_regime.confidence > self.current_regime.confidence + 0.2):
            
            # Check minimum duration constraint
            if (self.current_regime is None or 
                datetime.now() - self.current_regime.timestamp >= self.min_regime_duration or
                new_regime.confidence > self.transition_threshold):
                
                self.current_regime = new_regime
                self.regime_history.append(new_regime)
                
                # Keep history manageable
                if len(self.regime_history) > 100:
                    self.regime_history = self.regime_history[-50:]
    
    def get_current_regime(self) -> Optional[RegimeSignal]:
        """Get current market regime"""
        return self.current_regime
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection"""
        
        if not self.regime_history:
            return {
                'total_regimes': 0,
                'current_regime': None,
                'regime_distribution': {},
                'avg_confidence': 0,
                'avg_duration': timedelta(0)
            }
        
        # Regime distribution
        regime_counts = {}
        for signal in self.regime_history:
            regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
        
        # Calculate averages
        avg_confidence = np.mean([s.confidence for s in self.regime_history])
        
        # Calculate average duration
        durations = []
        for i in range(1, len(self.regime_history)):
            duration = self.regime_history[i].timestamp - self.regime_history[i-1].timestamp
            durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else timedelta(0)
        
        return {
            'total_regimes': len(self.regime_history),
            'current_regime': self.current_regime.regime.value if self.current_regime else None,
            'regime_distribution': regime_counts,
            'avg_confidence': avg_confidence,
            'avg_duration': avg_duration,
            'last_update': self.regime_history[-1].timestamp if self.regime_history else None
        }
    
    def is_regime_favorable_for_strategy(self, strategy_type: str) -> Tuple[bool, str]:
        """Check if current regime is favorable for a specific strategy type"""
        
        if not self.current_regime:
            return False, "No regime detected"
        
        regime = self.current_regime.regime
        confidence = self.current_regime.confidence
        
        # Strategy compatibility matrix
        strategy_compatibility = {
            'trend_following': [MarketRegime.BULL, MarketRegime.BEAR],
            'mean_reversion': [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE],
            'breakout': [MarketRegime.VOLATILE, MarketRegime.TRANSITION],
            'swing_trading': [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]
        }
        
        compatible_regimes = strategy_compatibility.get(strategy_type, [])
        
        if regime in compatible_regimes and confidence > 0.6:
            return True, f"Regime {regime.value} is favorable for {strategy_type}"
        elif regime in compatible_regimes:
            return True, f"Regime {regime.value} is moderately favorable for {strategy_type}"
        else:
            return False, f"Regime {regime.value} is not ideal for {strategy_type}"
