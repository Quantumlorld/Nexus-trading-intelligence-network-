"""
Nexus Trading System - Risk Engine
Central risk management and validation system
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import numpy as np


@dataclass
class RiskAssessment:
    """Risk assessment result for a trade"""
    is_allowed: bool
    risk_score: float  # 0-100, higher = riskier
    reasons: List[str]
    recommendations: List[str]
    max_position_size: float
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None


class RiskEngine:
    """Central risk management engine"""
    
    def __init__(self, risk_config: Dict[str, Any]):
        self.risk_config = risk_config
        self.logger = logging.getLogger(__name__)
        
        # Track risk metrics over time
        self.risk_history = []
        self.volatility_history = {}
        self.correlation_matrix = {}
        
        # Risk thresholds
        self.max_risk_score = 70  # Maximum acceptable risk score
        self.critical_risk_score = 85  # Critical risk level
        
    def assess_trade_risk(self, signal: Dict[str, Any], market_data: Dict[str, Any],
                         portfolio_state: Dict[str, Any], daily_stats: Dict[str, Any]) -> RiskAssessment:
        """
        Comprehensive risk assessment for a potential trade
        
        Returns:
            RiskAssessment with decision and details
        """
        
        risk_factors = []
        risk_score = 0.0
        reasons = []
        recommendations = []
        
        # 1. Check basic trade validity
        basic_risk = self._check_basic_trade_validity(signal, market_data)
        risk_score += basic_risk['score']
        reasons.extend(basic_risk['reasons'])
        
        if not basic_risk['is_valid']:
            return RiskAssessment(
                is_allowed=False,
                risk_score=risk_score,
                reasons=reasons,
                recommendations=["Trade rejected: Basic validation failed"],
                max_position_size=0.0
            )
        
        # 2. Assess market conditions risk
        market_risk = self._assess_market_conditions_risk(market_data)
        risk_score += market_risk['score']
        reasons.extend(market_risk['reasons'])
        recommendations.extend(market_risk['recommendations'])
        
        # 3. Assess portfolio risk
        portfolio_risk = self._assess_portfolio_risk(signal, portfolio_state)
        risk_score += portfolio_risk['score']
        reasons.extend(portfolio_risk['reasons'])
        recommendations.extend(portfolio_risk['recommendations'])
        
        # 4. Assess daily performance risk
        daily_risk = self._assess_daily_performance_risk(daily_stats)
        risk_score += daily_risk['score']
        reasons.extend(daily_risk['reasons'])
        recommendations.extend(daily_risk['recommendations'])
        
        # 5. Assess correlation risk
        correlation_risk = self._assess_correlation_risk(signal, portfolio_state)
        risk_score += correlation_risk['score']
        reasons.extend(correlation_risk['reasons'])
        
        # 6. Calculate maximum position size
        max_position_size = self._calculate_max_position_size(risk_score, signal, market_data)
        
        # 7. Adjust SL/TP if needed
        adjusted_sl, adjusted_tp = self._adjust_sl_tp(signal, market_data, risk_score)
        
        # Make final decision
        is_allowed = risk_score < self.max_risk_score
        
        if risk_score > self.critical_risk_score:
            recommendations.append("CRITICAL RISK: Consider reducing exposure significantly")
        elif risk_score > self.max_risk_score * 0.8:
            recommendations.append("HIGH RISK: Reduce position size or wait for better conditions")
        
        # Log the assessment
        self.logger.info(f"Risk assessment for {signal.get('symbol', 'Unknown')}: "
                        f"Score={risk_score:.1f}, Allowed={is_allowed}")
        
        return RiskAssessment(
            is_allowed=is_allowed,
            risk_score=risk_score,
            reasons=reasons,
            recommendations=recommendations,
            max_position_size=max_position_size,
            adjusted_sl=adjusted_sl,
            adjusted_tp=adjusted_tp
        )
    
    def _check_basic_trade_validity(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check basic trade validity"""
        
        score = 0.0
        reasons = []
        is_valid = True
        
        # Check required signal fields
        required_fields = ['direction', 'strategy', 'symbol']
        for field in required_fields:
            if field not in signal:
                reasons.append(f"Missing required field: {field}")
                score += 20
                is_valid = False
        
        # Check market data validity
        if not market_data or 'close' not in market_data:
            reasons.append("Invalid or missing market data")
            score += 25
            is_valid = False
        
        # Check confidence level
        confidence = signal.get('confidence', 0)
        min_confidence = 0.6  # Default minimum confidence
        if confidence < min_confidence:
            reasons.append(f"Low confidence: {confidence:.2f} < {min_confidence}")
            score += 15
        
        return {
            'score': score,
            'reasons': reasons,
            'is_valid': is_valid
        }
    
    def _assess_market_conditions_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk from current market conditions"""
        
        score = 0.0
        reasons = []
        recommendations = []
        
        # Volatility risk
        if self.risk_config['volatility_filter']['enabled']:
            volatility = market_data.get('volatility', 0)
            min_vol = self.risk_config['volatility_filter']['min_volatility_threshold']
            max_vol = self.risk_config['volatility_filter']['max_volatility_threshold']
            
            if volatility < min_vol:
                reasons.append(f"Low volatility: {volatility:.3f} < {min_vol}")
                score += 15
                recommendations.append("Consider waiting for higher volatility")
            elif volatility > max_vol:
                reasons.append(f"High volatility: {volatility:.3f} > {max_vol}")
                score += 20
                recommendations.append("Reduce position size due to high volatility")
        
        # Spread risk
        if self.risk_config['liquidity_filter']['enabled']:
            spread = market_data.get('spread', 0)
            max_spread = self.risk_config['liquidity_filter']['max_spread_threshold']
            
            if spread > max_spread:
                reasons.append(f"High spread: {spread:.1f} > {max_spread}")
                score += 25
                recommendations.append("Avoid trading due to high spread")
        
        # Liquidity risk (time-based)
        current_hour = datetime.now().hour
        symbol = market_data.get('symbol', '')
        
        if symbol == 'XAUUSD':
            # Gold-specific liquidity risks
            if current_hour < 8 or current_hour > 22:
                reasons.append("Outside optimal gold trading hours")
                score += 10
                recommendations.append("Gold trading outside optimal hours")
        
        return {
            'score': score,
            'reasons': reasons,
            'recommendations': recommendations
        }
    
    def _assess_portfolio_risk(self, signal: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio-level risk"""
        
        score = 0.0
        reasons = []
        recommendations = []
        
        # Check number of open positions
        open_positions = portfolio_state.get('open_positions', 0)
        max_positions = self.risk_config['portfolio_risk']['max_open_positions']
        
        if open_positions >= max_positions:
            reasons.append(f"Max positions reached: {open_positions}/{max_positions}")
            score += 30
            recommendations.append("Cannot open new positions")
        
        elif open_positions >= max_positions * 0.8:
            reasons.append(f"High position count: {open_positions}/{max_positions}")
            score += 15
            recommendations.append("Consider reducing position size")
        
        # Check portfolio heat
        portfolio_heat = portfolio_state.get('portfolio_heat', 0)
        max_heat = self.risk_config['portfolio_risk']['max_portfolio_heat']
        
        if portfolio_heat >= max_heat:
            reasons.append(f"Max portfolio heat reached: {portfolio_heat:.1f}%")
            score += 35
            recommendations.append("Portfolio at maximum risk exposure")
        elif portfolio_heat >= max_heat * 0.8:
            reasons.append(f"High portfolio heat: {portfolio_heat:.1f}%")
            score += 20
            recommendations.append("Reduce position size due to portfolio heat")
        
        return {
            'score': score,
            'reasons': reasons,
            'recommendations': recommendations
        }
    
    def _assess_daily_performance_risk(self, daily_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk based on daily performance"""
        
        score = 0.0
        reasons = []
        recommendations = []
        
        # Daily loss limit
        daily_pnl = daily_stats.get('total_pnl', 0)
        max_daily_loss = self.risk_config['daily_limits']['max_daily_loss']
        
        if daily_pnl <= -max_daily_loss:
            reasons.append(f"Max daily loss reached: ${daily_pnl:.2f}")
            score += 40
            recommendations.append("Stop trading for the day")
        elif daily_pnl <= -max_daily_loss * 0.8:
            reasons.append(f"Approaching daily loss limit: ${daily_pnl:.2f}")
            score += 25
            recommendations.append("Reduce position sizes")
        
        # Consecutive losses
        consecutive_losses = daily_stats.get('consecutive_losses', 0)
        max_consecutive = self.risk_config['kill_switch']['max_consecutive_losses']
        
        if consecutive_losses >= max_consecutive:
            reasons.append(f"Max consecutive losses: {consecutive_losses}")
            score += 45
            recommendations.append("Kill switch activated")
        elif consecutive_losses >= max_consecutive * 0.6:
            reasons.append(f"Multiple consecutive losses: {consecutive_losses}")
            score += 20
            recommendations.append("Be cautious with new trades")
        
        # Daily trade count
        daily_trades = daily_stats.get('trades_opened', 0)
        max_daily_trades = self.risk_config['daily_limits']['max_daily_trades']
        
        if daily_trades >= max_daily_trades:
            reasons.append(f"Max daily trades reached: {daily_trades}")
            score += 30
            recommendations.append("Daily trade limit reached")
        
        return {
            'score': score,
            'reasons': reasons,
            'recommendations': recommendations
        }
    
    def _assess_correlation_risk(self, signal: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess correlation risk with existing positions"""
        
        score = 0.0
        reasons = []
        
        symbol = signal.get('symbol', '')
        open_positions = portfolio_state.get('positions', {})
        
        # Check for correlated positions
        correlated_positions = self._find_correlated_positions(symbol, open_positions)
        max_correlated = self.risk_config['portfolio_risk']['max_correlated_positions']
        
        if len(correlated_positions) >= max_correlated:
            reasons.append(f"Too many correlated positions: {len(correlated_positions)}")
            score += 25
        
        return {
            'score': score,
            'reasons': reasons
        }
    
    def _find_correlated_positions(self, symbol: str, open_positions: Dict[str, Any]) -> List[str]:
        """Find positions correlated with the given symbol"""
        
        # Define correlation groups (simplified)
        correlation_groups = {
            'USD_GROUP': ['EURUSD', 'USDX', 'XAUUSD'],  # USD-related assets
            'RISK_GROUP': ['XAUUSD', 'BTCUSD'],  # Risk assets
            'SAFE_GROUP': ['USDX']  # Safe haven assets
        }
        
        correlated = []
        
        for group_name, group_symbols in correlation_groups.items():
            if symbol in group_symbols:
                # Find other symbols from the same group that are in open positions
                for other_symbol in group_symbols:
                    if other_symbol != symbol and other_symbol in open_positions:
                        correlated.append(other_symbol)
        
        return correlated
    
    def _calculate_max_position_size(self, risk_score: float, signal: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> float:
        """Calculate maximum position size based on risk score"""
        
        base_max_size = self.risk_config['portfolio_risk'].get('max_position_size', 1.0)
        
        # Reduce position size based on risk score
        if risk_score < 30:
            size_multiplier = 1.0  # Low risk - full size
        elif risk_score < 50:
            size_multiplier = 0.8  # Medium risk - 80% size
        elif risk_score < 70:
            size_multiplier = 0.6  # High risk - 60% size
        else:
            size_multiplier = 0.3  # Very high risk - 30% size
        
        max_size = base_max_size * size_multiplier
        
        # Ensure minimum size
        min_size = 0.01
        max_size = max(max_size, min_size)
        
        return max_size
    
    def _adjust_sl_tp(self, signal: Dict[str, Any], market_data: Dict[str, Any], 
                     risk_score: float) -> Tuple[Optional[float], Optional[float]]:
        """Adjust stop loss and take profit based on risk assessment"""
        
        adjusted_sl = None
        adjusted_tp = None
        
        # If high risk, tighten stop loss
        if risk_score > 60:
            current_price = market_data.get('close', 0)
            direction = signal.get('direction', '')
            
            # Tighten SL by 20%
            sl_points = self.risk_config['stop_loss']['default_sl_points']
            tightened_sl_points = sl_points * 0.8
            
            point_value = self._get_point_value(market_data.get('symbol', ''))
            sl_distance = tightened_sl_points / point_value
            
            if direction == 'buy':
                adjusted_sl = current_price - sl_distance
            else:
                adjusted_sl = current_price + sl_distance
        
        # If very high risk, reduce take profit
        if risk_score > 75:
            current_price = market_data.get('close', 0)
            direction = signal.get('direction', '')
            
            # Reduce TP by 25%
            tp_points = self.risk_config['take_profit']['default_tp_points']
            reduced_tp_points = tp_points * 0.75
            
            point_value = self._get_point_value(market_data.get('symbol', ''))
            tp_distance = reduced_tp_points / point_value
            
            if direction == 'buy':
                adjusted_tp = current_price + tp_distance
            else:
                adjusted_tp = current_price - tp_distance
        
        return adjusted_sl, adjusted_tp
    
    def _get_point_value(self, symbol: str) -> float:
        """Get point value for a symbol"""
        point_values = {
            'XAUUSD': 100,
            'EURUSD': 100000,
            'USDX': 1000,
            'BTCUSD': 1
        }
        return point_values.get(symbol, 100)
    
    def update_volatility_history(self, symbol: str, volatility: float):
        """Update volatility history for a symbol"""
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        
        self.volatility_history[symbol].append({
            'timestamp': datetime.now(),
            'volatility': volatility
        })
        
        # Keep only last 100 entries
        if len(self.volatility_history[symbol]) > 100:
            self.volatility_history[symbol] = self.volatility_history[symbol][-100:]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        
        # Calculate average volatility across all symbols
        all_volatilities = []
        for symbol, history in self.volatility_history.items():
            if history:
                latest_vol = history[-1]['volatility']
                all_volatilities.append(latest_vol)
        
        avg_volatility = np.mean(all_volatilities) if all_volatilities else 0
        
        # Calculate risk score trend
        recent_scores = [assessment['risk_score'] for assessment in self.risk_history[-10:]]
        avg_risk_score = np.mean(recent_scores) if recent_scores else 0
        
        return {
            'current_risk_score': avg_risk_score,
            'average_volatility': avg_volatility,
            'symbols_tracked': len(self.volatility_history),
            'assessments_today': len([r for r in self.risk_history 
                                    if r['timestamp'].date() == datetime.now().date()])
        }
    
    def should_activate_kill_switch(self, daily_stats: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if kill switch should be activated"""
        
        reasons = []
        
        # Check consecutive losses
        consecutive_losses = daily_stats.get('consecutive_losses', 0)
        if consecutive_losses >= self.risk_config['kill_switch']['max_consecutive_losses']:
            reasons.append(f"Too many consecutive losses: {consecutive_losses}")
        
        # Check daily loss
        daily_pnl = daily_stats.get('total_pnl', 0)
        if daily_pnl <= -self.risk_config['daily_limits']['max_daily_loss']:
            reasons.append(f"Maximum daily loss reached: ${daily_pnl:.2f}")
        
        # Check drawdown
        max_drawdown = daily_stats.get('max_drawdown', 0)
        if max_drawdown >= self.risk_config['kill_switch']['max_drawdown_percent']:
            reasons.append(f"Maximum drawdown reached: {max_drawdown:.1f}%")
        
        should_activate = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else ""
        
        return should_activate, reason_text
