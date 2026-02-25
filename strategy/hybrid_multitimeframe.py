"""
Nexus Trading System - Hybrid Multi-Timeframe Strategy
Master strategy integrating weekly bias, daily structure, EMA filter, 
swing homeostasis, and multi-timeframe signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig
from .ema_crossover import EMACrossoverStrategy, EMACrossoverConfig
from .swing_homeostasis import SwingHomeostasisStrategy, SwingHomeostasisConfig
from core.logger import get_logger


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: str
    signal_type: SignalType
    confidence: float
    weight: float
    metadata: Dict[str, Any]


@dataclass
class MultiTimeframeAnalysis:
    """Multi-timeframe analysis results"""
    weekly_bias: str  # bullish, bearish, neutral
    daily_structure: Dict[str, Any]
    intraday_signals: List[TimeframeSignal]
    overall_sentiment: float  # -1 to 1
    confluence_score: float  # 0 to 1
    risk_assessment: Dict[str, Any]


class HybridMultiTimeframeConfig(StrategyConfig):
    """Configuration for Hybrid Multi-Timeframe strategy"""
    
    def __init__(self,
                 timeframes: List[str] = None,
                 timeframe_weights: Dict[str, float] = None,
                 weekly_bias_weight: float = 0.3,
                 daily_structure_weight: float = 0.25,
                 confluence_threshold: float = 0.7,
                 min_timeframes_agree: int = 2,
                 **kwargs):
        
        super().__init__(name="Hybrid_Multi_Timeframe", **kwargs)
        
        self.timeframes = timeframes or ["1W", "1D", "4H", "1H", "15M"]
        self.timeframe_weights = timeframe_weights or {
            "1W": 0.3,
            "1D": 0.25,
            "4H": 0.2,
            "1H": 0.15,
            "15M": 0.1
        }
        self.weekly_bias_weight = weekly_bias_weight
        self.daily_structure_weight = daily_structure_weight
        self.confluence_threshold = confluence_threshold
        self.min_timeframes_agree = min_timeframes_agree
        
        # Update parameters
        self.parameters.update({
            'timeframes': self.timeframes,
            'timeframe_weights': self.timeframe_weights,
            'weekly_bias_weight': weekly_bias_weight,
            'daily_structure_weight': daily_structure_weight,
            'confluence_threshold': confluence_threshold,
            'min_timeframes_agree': min_timeframes_agree
        })


class HybridMultiTimeframeStrategy(BaseStrategy):
    """
    Hybrid Multi-Timeframe Strategy - Master strategy integrating:
    - Weekly bias analysis
    - Daily structure analysis
    - EMA crossover filter
    - Swing homeostasis patterns
    - Multi-timeframe signal confluence
    """
    
    def __init__(self, config: HybridMultiTimeframeConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize sub-strategies
        self.ema_strategy = EMACrossoverStrategy(
            EMACrossoverConfig(
                fast_ema_period=12,
                slow_ema_period=26,
                signal_ema_period=9,
                min_confidence=0.6
            )
        )
        
        self.swing_strategy = SwingHomeostasisStrategy(
            SwingHomeostasisConfig(
                lookback_period=100,
                structure_sensitivity=0.02,
                min_confidence=0.6
            )
        )
        
        # Multi-timeframe data cache
        self.mtf_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, MultiTimeframeAnalysis] = {}
        
        # Signal history by timeframe
        self.timeframe_signals: Dict[str, List[TimeframeSignal]] = defaultdict(list)
        
        self.logger.info("Hybrid Multi-Timeframe strategy initialized")
    
    def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate hybrid multi-timeframe signal
        
        Args:
            symbol: Trading symbol
            market_data: Primary timeframe OHLCV data
            
        Returns:
            TradingSignal or None if no signal
        """
        
        if not self.should_generate_signal(symbol, datetime.now()):
            return None
        
        # Perform multi-timeframe analysis
        mtf_analysis = self._analyze_multi_timeframe(symbol, market_data)
        
        if mtf_analysis is None:
            return None
        
        # Check for signal confluence
        confluence_signal = self._check_signal_confluence(symbol, mtf_analysis, market_data)
        
        if confluence_signal is None:
            return None
        
        return self.process_signal(confluence_signal, market_data)
    
    def validate_signal(self, signal: TradingSignal, market_data: pd.DataFrame) -> bool:
        """
        Validate hybrid multi-timeframe signal
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid
        """
        
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check confluence score
        confluence_score = signal.metadata.get('confluence_score', 0)
        if confluence_score < self.config.confluence_threshold:
            return False
        
        # Validate multi-timeframe agreement
        agreeing_timeframes = signal.metadata.get('agreeing_timeframes', [])
        if len(agreeing_timeframes) < self.config.min_timeframes_agree:
            return False
        
        # Check weekly bias alignment
        weekly_bias = signal.metadata.get('weekly_bias', 'neutral')
        signal_direction = signal.signal_type.value
        
        if weekly_bias == 'bullish' and signal_direction == 'sell':
            return False
        elif weekly_bias == 'bearish' and signal_direction == 'buy':
            return False
        
        # Additional validation using sub-strategies
        ema_valid = self.ema_strategy.validate_signal(signal, market_data)
        swing_valid = self.swing_strategy.validate_signal(signal, market_data)
        
        # At least one sub-strategy must validate the signal
        return ema_valid or swing_valid
    
    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss using hybrid approach
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Stop loss price
        """
        
        # Get SL from both sub-strategies
        ema_sl = self.ema_strategy.calculate_stop_loss(signal, market_data)
        swing_sl = self.swing_strategy.calculate_stop_loss(signal, market_data)
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        
        # Choose the more conservative SL (further from entry)
        if signal.signal_type == SignalType.BUY:
            # For buy, choose lower SL (more conservative)
            hybrid_sl = min(ema_sl, swing_sl)
        else:
            # For sell, choose higher SL (more conservative)
            hybrid_sl = max(ema_sl, swing_sl)
        
        # Additional safety based on weekly structure
        weekly_structure = signal.metadata.get('weekly_structure', {})
        if weekly_structure:
            weekly_support = weekly_structure.get('support_level')
            weekly_resistance = weekly_structure.get('resistance_level')
            
            if signal.signal_type == SignalType.BUY and weekly_support:
                hybrid_sl = min(hybrid_sl, weekly_support)
            elif signal.signal_type == SignalType.SELL and weekly_resistance:
                hybrid_sl = max(hybrid_sl, weekly_resistance)
        
        return hybrid_sl
    
    def calculate_take_profit(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate take profit using hybrid approach
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Take profit price
        """
        
        # Get TP from both sub-strategies
        ema_tp = self.ema_strategy.calculate_take_profit(signal, market_data)
        swing_tp = self.swing_strategy.calculate_take_profit(signal, market_data)
        
        current_price = signal.entry_price or market_data['close'].iloc[-1]
        
        # Choose the more aggressive TP (closer to entry for higher probability)
        if signal.signal_type == SignalType.BUY:
            # For buy, choose higher TP (more aggressive)
            hybrid_tp = max(ema_tp, swing_tp)
        else:
            # For sell, choose lower TP (more aggressive)
            hybrid_tp = min(ema_tp, swing_tp)
        
        # Adjust based on weekly targets
        weekly_structure = signal.metadata.get('weekly_structure', {})
        if weekly_structure:
            weekly_target = weekly_structure.get('target_level')
            
            if weekly_target:
                if signal.signal_type == SignalType.BUY:
                    hybrid_tp = max(hybrid_tp, weekly_target)
                else:
                    hybrid_tp = min(hybrid_tp, weekly_target)
        
        return hybrid_tp
    
    def _analyze_multi_timeframe(self, symbol: str, market_data: pd.DataFrame) -> Optional[MultiTimeframeAnalysis]:
        """Perform comprehensive multi-timeframe analysis"""
        
        try:
            # Get multi-timeframe data
            mtf_data = self._get_multi_timeframe_data(symbol, market_data)
            
            # Analyze weekly bias
            weekly_bias = self._analyze_weekly_bias(symbol, mtf_data.get('1W'))
            
            # Analyze daily structure
            daily_structure = self._analyze_daily_structure(symbol, mtf_data.get('1D'))
            
            # Generate signals from each timeframe
            intraday_signals = []
            
            for timeframe in ['4H', '1H', '15M']:
                tf_data = mtf_data.get(timeframe)
                if tf_data is not None:
                    tf_signals = self._generate_timeframe_signals(symbol, timeframe, tf_data)
                    intraday_signals.extend(tf_signals)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(weekly_bias, daily_structure, intraday_signals)
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(intraday_signals)
            
            # Risk assessment
            risk_assessment = self._assess_multi_timeframe_risk(mtf_data, overall_sentiment)
            
            analysis = MultiTimeframeAnalysis(
                weekly_bias=weekly_bias,
                daily_structure=daily_structure,
                intraday_signals=intraday_signals,
                overall_sentiment=overall_sentiment,
                confluence_score=confluence_score,
                risk_assessment=risk_assessment
            )
            
            # Cache analysis
            self.analysis_cache[symbol] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return None
    
    def _get_multi_timeframe_data(self, symbol: str, primary_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get data for all configured timeframes"""
        
        # For now, simulate multi-timeframe data by resampling
        # In production, this would fetch actual data from multiple timeframes
        
        mtf_data = {}
        
        # Primary timeframe (assumed to be 1H or similar)
        primary_tf = self._detect_timeframe(primary_data)
        mtf_data[primary_tf] = primary_data
        
        # Generate other timeframes by resampling
        resample_rules = {
            '15M': '15T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            '1W': '1W'
        }
        
        for tf in self.config.timeframes:
            if tf != primary_tf and tf in resample_rules:
                try:
                    resampled = self._resample_data(primary_data, resample_rules[tf])
                    if resampled is not None:
                        mtf_data[tf] = resampled
                except Exception as e:
                    self.logger.warning(f"Failed to resample to {tf}: {e}")
        
        return mtf_data
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect the timeframe of the data"""
        
        if len(data) < 2:
            return "1H"  # Default
        
        time_diff = data.index[1] - data.index[0]
        
        timeframe_map = {
            pd.Timedelta(minutes=15): "15M",
            pd.Timedelta(hours=1): "1H",
            pd.Timedelta(hours=4): "4H",
            pd.Timedelta(days=1): "1D",
            pd.Timedelta(weeks=1): "1W"
        }
        
        return timeframe_map.get(time_diff, "1H")
    
    def _resample_data(self, data: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        """Resample data to different timeframe"""
        
        try:
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
            if 'volume' in data.columns:
                ohlc_dict['volume'] = 'sum'
            
            resampled = data.resample(rule).agg(ohlc_dict)
            return resampled.dropna()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return None
    
    def _analyze_weekly_bias(self, symbol: str, weekly_data: Optional[pd.DataFrame]) -> str:
        """Analyze weekly bias"""
        
        if weekly_data is None or len(weekly_data) < 10:
            return "neutral"
        
        # Use EMA crossover on weekly data
        weekly_ema_signal = self.ema_strategy.generate_signal(symbol, weekly_data)
        
        if weekly_ema_signal and weekly_ema_signal.signal_type == SignalType.BUY:
            return "bullish"
        elif weekly_ema_signal and weekly_ema_signal.signal_type == SignalType.SELL:
            return "bearish"
        
        # Fallback to simple trend analysis
        weekly_close = weekly_data['close']
        weekly_ma = weekly_close.rolling(20).mean()
        
        if weekly_close.iloc[-1] > weekly_ma.iloc[-1]:
            return "bullish"
        elif weekly_close.iloc[-1] < weekly_ma.iloc[-1]:
            return "bearish"
        
        return "neutral"
    
    def _analyze_daily_structure(self, symbol: str, daily_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze daily market structure"""
        
        if daily_data is None or len(daily_data) < 50:
            return {}
        
        # Use swing homeostasis for structure analysis
        structure = self.swing_strategy.get_market_structure(symbol, daily_data)
        
        if structure is None:
            return {}
        
        return {
            'support_levels': structure.support_levels,
            'resistance_levels': structure.resistance_levels,
            'equilibrium_price': structure.equilibrium_price,
            'current_position': self._get_current_structure_position(daily_data, structure)
        }
    
    def _get_current_structure_position(self, data: pd.DataFrame, structure) -> str:
        """Determine current position relative to structure"""
        
        current_price = data['close'].iloc[-1]
        
        if not structure.support_levels or not structure.resistance_levels:
            return "unknown"
        
        nearest_support = min(structure.support_levels, key=lambda x: abs(x - current_price))
        nearest_resistance = min(structure.resistance_levels, key=lambda x: abs(x - current_price))
        
        support_distance = abs(current_price - nearest_support) / current_price
        resistance_distance = abs(nearest_resistance - current_price) / current_price
        
        if support_distance < resistance_distance * 0.5:
            return "near_support"
        elif resistance_distance < support_distance * 0.5:
            return "near_resistance"
        else:
            return "middle_range"
    
    def _generate_timeframe_signals(self, symbol: str, timeframe: str, 
                                  data: pd.DataFrame) -> List[TimeframeSignal]:
        """Generate signals from a specific timeframe"""
        
        signals = []
        
        # EMA crossover signal
        ema_signal = self.ema_strategy.generate_signal(symbol, data)
        if ema_signal:
            signals.append(TimeframeSignal(
                timeframe=timeframe,
                signal_type=ema_signal.signal_type,
                confidence=ema_signal.confidence,
                weight=self.config.timeframe_weights.get(timeframe, 0.1),
                metadata={'source': 'ema', 'ema_values': ema_signal.metadata.get('emas')}
            ))
        
        # Swing homeostasis signal
        swing_signal = self.swing_strategy.generate_signal(symbol, data)
        if swing_signal:
            signals.append(TimeframeSignal(
                timeframe=timeframe,
                signal_type=swing_signal.signal_type,
                confidence=swing_signal.confidence,
                weight=self.config.timeframe_weights.get(timeframe, 0.1),
                metadata={'source': 'swing', 'structure': swing_signal.metadata}
            ))
        
        return signals
    
    def _calculate_overall_sentiment(self, weekly_bias: str, daily_structure: Dict[str, Any],
                                   intraday_signals: List[TimeframeSignal]) -> float:
        """Calculate overall market sentiment (-1 to 1)"""
        
        sentiment_score = 0.0
        
        # Weekly bias contribution
        weekly_bias_scores = {'bullish': 0.8, 'neutral': 0.0, 'bearish': -0.8}
        sentiment_score += weekly_bias_scores.get(weekly_bias, 0) * self.config.weekly_bias_weight
        
        # Daily structure contribution
        if daily_structure:
            structure_position = daily_structure.get('current_position', 'middle_range')
            structure_scores = {
                'near_support': 0.6,
                'near_resistance': -0.6,
                'middle_range': 0.0,
                'unknown': 0.0
            }
            sentiment_score += structure_scores.get(structure_position, 0) * self.config.daily_structure_weight
        
        # Intraday signals contribution
        for signal in intraday_signals:
            signal_score = 1.0 if signal.signal_type == SignalType.BUY else -1.0
            weighted_score = signal_score * signal.confidence * signal.weight
            sentiment_score += weighted_score
        
        # Normalize to -1 to 1 range
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _calculate_confluence_score(self, intraday_signals: List[TimeframeSignal]) -> float:
        """Calculate signal confluence score (0 to 1)"""
        
        if not intraday_signals:
            return 0.0
        
        # Group signals by type
        buy_signals = [s for s in intraday_signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in intraday_signals if s.signal_type == SignalType.SELL]
        
        # Calculate confluence based on agreement
        total_weight = sum(s.weight for s in intraday_signals)
        
        if total_weight == 0:
            return 0.0
        
        buy_weight = sum(s.weight * s.confidence for s in buy_signals)
        sell_weight = sum(s.weight * s.confidence for s in sell_signals)
        
        # Confluence is stronger when one direction dominates
        dominant_weight = max(buy_weight, sell_weight)
        confluence_score = dominant_weight / total_weight
        
        return confluence_score
    
    def _assess_multi_timeframe_risk(self, mtf_data: Dict[str, pd.DataFrame], 
                                   sentiment: float) -> Dict[str, Any]:
        """Assess risk across multiple timeframes"""
        
        risk_assessment = {
            'overall_risk': 'medium',
            'volatility_risk': 'medium',
            'timeframe_alignment': 'mixed',
            'risk_factors': []
        }
        
        # Check volatility across timeframes
        volatility_scores = []
        
        for tf, data in mtf_data.items():
            if len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                volatility_scores.append(volatility)
        
        if volatility_scores:
            avg_volatility = np.mean(volatility_scores)
            
            if avg_volatility > 0.3:
                risk_assessment['volatility_risk'] = 'high'
                risk_assessment['risk_factors'].append('high_volatility')
            elif avg_volatility < 0.1:
                risk_assessment['volatility_risk'] = 'low'
        
        # Check timeframe alignment
        if abs(sentiment) > 0.7:
            risk_assessment['timeframe_alignment'] = 'strong'
        elif abs(sentiment) > 0.3:
            risk_assessment['timeframe_alignment'] = 'moderate'
        else:
            risk_assessment['timeframe_alignment'] = 'weak'
            risk_assessment['risk_factors'].append('weak_timeframe_alignment')
        
        # Overall risk assessment
        high_risk_factors = len([f for f in risk_assessment['risk_factors'] if 'high' in f or 'weak' in f])
        
        if high_risk_factors >= 2:
            risk_assessment['overall_risk'] = 'high'
        elif high_risk_factors == 0:
            risk_assessment['overall_risk'] = 'low'
        
        return risk_assessment
    
    def _check_signal_confluence(self, symbol: str, analysis: MultiTimeframeAnalysis,
                               market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Check for signal confluence across timeframes"""
        
        # Check if confluence score meets threshold
        if analysis.confluence_score < self.config.confluence_threshold:
            return None
        
        # Check minimum timeframes agreement
        buy_signals = [s for s in analysis.intraday_signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in analysis.intraday_signals if s.signal_type == SignalType.SELL]
        
        # Determine dominant signal direction
        if len(buy_signals) >= len(sell_signals):
            if len(buy_signals) < self.config.min_timeframes_agree:
                return None
            
            dominant_signals = buy_signals
            signal_type = SignalType.BUY
        else:
            if len(sell_signals) < self.config.min_timeframes_agree:
                return None
            
            dominant_signals = sell_signals
            signal_type = SignalType.SELL
        
        # Calculate weighted confidence
        total_weighted_confidence = sum(s.confidence * s.weight for s in dominant_signals)
        total_weight = sum(s.weight for s in dominant_signals)
        weighted_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Boost confidence based on weekly bias alignment
        if (analysis.weekly_bias == 'bullish' and signal_type == SignalType.BUY) or \
           (analysis.weekly_bias == 'bearish' and signal_type == SignalType.SELL):
            weighted_confidence = min(weighted_confidence * 1.2, 1.0)
        elif (analysis.weekly_bias == 'bullish' and signal_type == SignalType.SELL) or \
             (analysis.weekly_bias == 'bearish' and signal_type == SignalType.BUY):
            weighted_confidence *= 0.7  # Reduce confidence if going against weekly bias
        
        # Create hybrid signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=weighted_confidence,
            strategy=self.name,
            timestamp=datetime.now(),
            entry_price=market_data['close'].iloc[-1],
            reason=f"Multi-timeframe confluence: {len(dominant_signals)} timeframes agree",
            metadata={
                'signal_type': 'multi_timeframe_confluence',
                'confluence_score': analysis.confluence_score,
                'weekly_bias': analysis.weekly_bias,
                'daily_structure': analysis.daily_structure,
                'agreeing_timeframes': [s.timeframe for s in dominant_signals],
                'overall_sentiment': analysis.overall_sentiment,
                'risk_assessment': analysis.risk_assessment,
                'timeframe_signals': [
                    {
                        'timeframe': s.timeframe,
                        'signal_type': s.signal_type.value,
                        'confidence': s.confidence,
                        'source': s.metadata.get('source', 'unknown')
                    }
                    for s in analysis.intraday_signals
                ]
            }
        )
        
        return signal
    
    def get_multi_timeframe_analysis(self, symbol: str, market_data: pd.DataFrame) -> Optional[MultiTimeframeAnalysis]:
        """Get comprehensive multi-timeframe analysis"""
        return self._analyze_multi_timeframe(symbol, market_data)
    
    def get_timeframe_signals(self, symbol: str, timeframe: str, limit: int = 10) -> List[TimeframeSignal]:
        """Get recent signals from a specific timeframe"""
        return self.timeframe_signals[timeframe][-limit:] if timeframe in self.timeframe_signals else []
    
    def get_strategy_status(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed strategy status for a symbol"""
        
        base_status = self.get_status()
        
        # Add multi-timeframe specific information
        mtf_analysis = self.get_multi_timeframe_analysis(symbol, market_data)
        
        if mtf_analysis:
            mtf_status = {
                'weekly_bias': mtf_analysis.weekly_bias,
                'daily_structure_position': mtf_analysis.daily_structure.get('current_position', 'unknown'),
                'overall_sentiment': mtf_analysis.overall_sentiment,
                'confluence_score': mtf_analysis.confluence_score,
                'active_timeframe_signals': len(mtf_analysis.intraday_signals),
                'risk_assessment': mtf_analysis.risk_assessment,
                'sub_strategies': {
                    'ema_strategy': self.ema_strategy.get_status(),
                    'swing_strategy': self.swing_strategy.get_status()
                }
            }
        else:
            mtf_status = {'error': 'Could not perform multi-timeframe analysis'}
        
        base_status['multi_timeframe_analysis'] = mtf_status
        
        return base_status
    
    def reset_strategy(self):
        """Reset strategy state"""
        super().reset_performance_metrics()
        
        # Reset sub-strategies
        self.ema_strategy.reset_performance_metrics()
        self.swing_strategy.reset_performance_metrics()
        
        # Clear caches
        self.mtf_data_cache.clear()
        self.analysis_cache.clear()
        self.timeframe_signals.clear()
        
        self.logger.info("Hybrid Multi-Timeframe strategy reset")
    
    def get_version(self) -> str:
        """Get strategy version"""
        return "2.0.0"
