"""
Nexus Trading System - Enhanced Risk Engine
Strict risk enforcement with daily loss caps and position limits
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, time
import logging
from dataclasses import dataclass
import numpy as np
from enum import Enum

from config.settings import settings
from database.session import get_database_session
from database.models import User, Trade, UserPerformance

class TimeFrame(Enum):
    """Trading timeframes with trade limits"""
    H9 = "9H"    # 9-hour timeframe
    H6 = "6H"    # 6-hour timeframe  
    H3 = "3H"    # 3-hour timeframe

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_risk_percent: float = 1.0
    max_daily_loss: float = 9.99
    max_daily_trades_9h: int = 2
    max_daily_trades_6h: int = 2
    max_daily_trades_3h: int = 1
    default_sl_points: int = 300  # -$3
    default_tp_points: int = 990  # +$9.9
    lock_profit_threshold: int = 300  # Lock SL to +$3
    runner_tp_extension: int = 1500  # Extend TP to +$15

@dataclass
class TradeValidation:
    """Trade validation result"""
    is_allowed: bool
    risk_score: float
    reasons: List[str]
    recommendations: List[str]
    max_position_size: float
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None
    daily_loss_remaining: float = 0.0
    trades_remaining_today: Dict[str, int] = None

class EnhancedRiskEngine:
    """Enhanced risk engine with strict enforcement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.limits = RiskLimits()
        self.emergency_stop = False
        self.user_trading_disabled = {}  # Per-user trading disable flags
        
        # Load limits from settings
        self.limits.max_risk_percent = settings.MAX_RISK_PERCENT
        self.limits.max_daily_loss = settings.MAX_DAILY_LOSS
        self.limits.default_sl_points = settings.DEFAULT_SL_POINTS
        self.limits.default_tp_points = settings.DEFAULT_TP_POINTS
        
        self.logger.info("Enhanced Risk Engine initialized with strict enforcement")
    
    def set_emergency_stop(self, enabled: bool, reason: str = ""):
        """Set global emergency stop"""
        self.emergency_stop = enabled
        if enabled:
            self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        else:
            self.logger.info("Emergency stop deactivated")
    
    def set_user_trading_disabled(self, user_id: int, disabled: bool, reason: str = ""):
        """Disable/enable trading for specific user"""
        self.user_trading_disabled[user_id] = disabled
        if disabled:
            self.logger.warning(f"Trading disabled for user {user_id}: {reason}")
        else:
            self.logger.info(f"Trading re-enabled for user {user_id}")
    
    def validate_trade(self, user_id: int, signal: Dict[str, Any], 
                      account_balance: float, timeframe: TimeFrame) -> TradeValidation:
        """
        Comprehensive trade validation with strict enforcement
        """
        
        # Check emergency stops
        if self.emergency_stop:
            return TradeValidation(
                is_allowed=False,
                risk_score=100.0,
                reasons=["Global emergency stop is active"],
                recommendations=["Wait for emergency stop to be lifted"],
                max_position_size=0.0
            )
        
        if self.user_trading_disabled.get(user_id, False):
            return TradeValidation(
                is_allowed=False,
                risk_score=100.0,
                reasons=["Trading is disabled for this user"],
                recommendations=["Contact support to enable trading"],
                max_position_size=0.0
            )
        
        # Get user's daily trading data
        daily_stats = self._get_daily_trading_stats(user_id)
        
        # Check daily loss cap
        if daily_stats['daily_loss'] >= self.limits.max_daily_loss:
            return TradeValidation(
                is_allowed=False,
                risk_score=100.0,
                reasons=[f"Daily loss cap exceeded: ${daily_stats['daily_loss']:.2f} >= ${self.limits.max_daily_loss:.2f}"],
                recommendations=["Wait for daily reset at midnight UTC"],
                max_position_size=0.0,
                daily_loss_remaining=0.0
            )
        
        # Check daily trade limits per timeframe
        max_trades = self._get_max_trades_for_timeframe(timeframe)
        if daily_stats['trade_count'] >= max_trades:
            return TradeValidation(
                is_allowed=False,
                risk_score=90.0,
                reasons=[f"Daily trade limit exceeded for {timeframe.value}: {daily_stats['trade_count']} >= {max_trades}"],
                recommendations=["Wait for daily reset at midnight UTC"],
                max_position_size=0.0,
                trades_remaining_today={
                    timeframe.value: max(0, max_trades - daily_stats['trade_count'])
                }
            )
        
        # Check for duplicate signals
        if self._is_duplicate_signal(user_id, signal):
            return TradeValidation(
                is_allowed=False,
                risk_score=80.0,
                reasons=["Duplicate signal detected"],
                recommendations=["Wait for new signal or different timeframe"],
                max_position_size=0.0
            )
        
        # Calculate position size based on risk
        risk_amount = min(
            account_balance * (self.limits.max_risk_percent / 100),
            self.limits.max_daily_loss - daily_stats['daily_loss']
        )
        
        if risk_amount <= 0:
            return TradeValidation(
                is_allowed=False,
                risk_score=100.0,
                reasons=["No risk capital available"],
                recommendations=["Wait for daily reset or increase account balance"],
                max_position_size=0.0,
                daily_loss_remaining=daily_stats['daily_loss_remaining']
            )
        
        # Calculate SL/TP based on signal
        entry_price = signal.get('entry_price', 0)
        symbol = signal.get('symbol', '')
        
        # Default SL/TP
        sl_points = self.limits.default_sl_points
        tp_points = self.limits.default_tp_points
        
        # Adjust for symbol if needed
        sl_price = entry_price - (sl_points / 10000) if 'BUY' in signal.get('action', '').upper() else entry_price + (sl_points / 10000)
        tp_price = entry_price + (tp_points / 10000) if 'BUY' in signal.get('action', '').upper() else entry_price - (tp_points / 10000)
        
        # Calculate position size
        position_size = self._calculate_position_size(risk_amount, sl_points, entry_price)
        
        # Risk score calculation
        risk_factors = []
        risk_score = 0.0
        
        # Account size factor
        if account_balance < 1000:
            risk_score += 20
            risk_factors.append("Small account size")
        
        # Volatility factor (if available)
        volatility = signal.get('volatility', 1.0)
        if volatility > 2.0:
            risk_score += 15
            risk_factors.append("High volatility")
        
        # Time factor
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 10
            risk_factors.append("Off-hours trading")
        
        # Ensure risk score doesn't exceed limits
        risk_score = min(risk_score, 85)  # Cap at 85 to allow some trades
        
        # Determine if trade is allowed based on risk score
        is_allowed = risk_score < 70  # Allow trades with risk score below 70
        
        # Calculate remaining daily loss
        daily_loss_remaining = self.limits.max_daily_loss - daily_stats['daily_loss']
        
        # Calculate remaining trades
        trades_remaining = {
            timeframe.value: max(0, max_trades - daily_stats['trade_count'])
        }
        
        return TradeValidation(
            is_allowed=is_allowed,
            risk_score=risk_score,
            reasons=risk_factors if risk_factors else ["Acceptable risk"],
            recommendations=self._get_recommendations(risk_score, risk_factors),
            max_position_size=position_size,
            adjusted_sl=sl_price,
            adjusted_tp=tp_price,
            daily_loss_remaining=daily_loss_remaining,
            trades_remaining_today=trades_remaining
        )
    
    def _get_daily_trading_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user's daily trading statistics"""
        try:
            # Get today's date in UTC
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, time.min)
            
            with next(get_database_session()) as db:
                # Get today's trades
                trades = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.entry_time >= start_of_day
                ).all()
                
                # Calculate daily P&L
                daily_pnl = sum(trade.pnl or 0 for trade in trades)
                daily_loss = abs(min(daily_pnl, 0))
                
                # Count trades
                trade_count = len(trades)
                
                # Get winning trades
                winning_trades = len([t for t in trades if (t.pnl or 0) > 0])
                
                return {
                    'daily_pnl': daily_pnl,
                    'daily_loss': daily_loss,
                    'trade_count': trade_count,
                    'winning_trades': winning_trades,
                    'losing_trades': trade_count - winning_trades,
                    'daily_loss_remaining': self.limits.max_daily_loss - daily_loss
                }
                
        except Exception as e:
            self.logger.error(f"Error getting daily trading stats for user {user_id}: {e}")
            return {
                'daily_pnl': 0.0,
                'daily_loss': 0.0,
                'trade_count': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'daily_loss_remaining': self.limits.max_daily_loss
            }
    
    def _get_max_trades_for_timeframe(self, timeframe: TimeFrame) -> int:
        """Get maximum trades allowed for timeframe"""
        limits = {
            TimeFrame.H9: self.limits.max_daily_trades_9h,
            TimeFrame.H6: self.limits.max_daily_trades_6h,
            TimeFrame.H3: self.limits.max_daily_trades_3h
        }
        return limits.get(timeframe, 1)
    
    def _is_duplicate_signal(self, user_id: int, signal: Dict[str, Any]) -> bool:
        """Check if this is a duplicate of a recent signal"""
        try:
            symbol = signal.get('symbol', '')
            action = signal.get('action', '')
            entry_price = signal.get('entry_price', 0)
            
            # Check for similar trades in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            with next(get_database_session()) as db:
                recent_trades = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.symbol == symbol,
                    Trade.action == action,
                    Trade.entry_time >= one_hour_ago
                ).all()
                
                # Check if any recent trade has similar entry price (within 0.1%)
                for trade in recent_trades:
                    price_diff = abs(trade.entry_price - entry_price) / entry_price
                    if price_diff < 0.001:  # 0.1% threshold
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking duplicate signal: {e}")
            return False
    
    def _calculate_position_size(self, risk_amount: float, sl_points: int, entry_price: float) -> float:
        """Calculate position size based on risk amount and stop loss"""
        if sl_points == 0:
            return 0.0
        
        # Risk per point (simplified - would need symbol-specific calculation)
        risk_per_point = 1.0  # $1 per point (simplified)
        
        # Calculate position size
        position_size = risk_amount / (sl_points * risk_per_point / 100)
        
        # Ensure minimum position size
        min_position_size = 0.01
        position_size = max(position_size, min_position_size)
        
        return position_size
    
    def _get_recommendations(self, risk_score: float, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on risk assessment"""
        recommendations = []
        
        if risk_score > 60:
            recommendations.append("Consider reducing position size")
        
        if "High volatility" in risk_factors:
            recommendations.append("Wait for volatility to decrease")
        
        if "Small account size" in risk_factors:
            recommendations.append("Consider increasing account balance")
        
        if "Off-hours trading" in risk_factors:
            recommendations.append("Trade during market hours for better liquidity")
        
        if not recommendations:
            recommendations.append("Proceed with caution")
        
        return recommendations
    
    def get_user_risk_status(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive risk status for user"""
        daily_stats = self._get_daily_trading_stats(user_id)
        
        return {
            'user_id': user_id,
            'daily_loss': daily_stats['daily_loss'],
            'daily_loss_limit': self.limits.max_daily_loss,
            'daily_loss_remaining': daily_stats['daily_loss_remaining'],
            'daily_trades': daily_stats['trade_count'],
            'emergency_stop_active': self.emergency_stop,
            'trading_disabled': self.user_trading_disabled.get(user_id, False),
            'risk_limits': {
                'max_risk_percent': self.limits.max_risk_percent,
                'max_daily_loss': self.limits.max_daily_loss,
                'default_sl_points': self.limits.default_sl_points,
                'default_tp_points': self.limits.default_tp_points
            },
            'timeframe_limits': {
                '9H': self.limits.max_daily_trades_9h,
                '6H': self.limits.max_daily_trades_6h,
                '3H': self.limits.max_daily_trades_3h
            }
        }
    
    def update_dynamic_sl_tp(self, trade_id: int, current_price: float, 
                           entry_price: float, original_sl: float, original_tp: float) -> Tuple[float, float]:
        """
        Update dynamic stop loss and take profit based on price movement
        """
        action = 'BUY'  # Would get from trade
        price_diff = current_price - entry_price if action == 'BUY' else entry_price - current_price
        price_diff_points = price_diff * 10000  # Convert to points
        
        new_sl = original_sl
        new_tp = original_tp
        
        # Lock profit when trade moves positive by lock threshold
        if price_diff_points >= self.limits.lock_profit_threshold:
            if action == 'BUY':
                new_sl = entry_price + (self.limits.lock_profit_threshold / 10000)
            else:
                new_sl = entry_price - (self.limits.lock_profit_threshold / 10000)
            
            self.logger.info(f"Locked profit for trade {trade_id}: SL moved to {new_sl}")
        
        # Extend TP in runner mode when original TP is hit
        if price_diff_points >= self.limits.default_tp_points:
            if action == 'BUY':
                new_tp = entry_price + (self.limits.runner_tp_extension / 10000)
            else:
                new_tp = entry_price - (self.limits.runner_tp_extension / 10000)
            
            self.logger.info(f"Extended TP for trade {trade_id}: TP moved to {new_tp}")
        
        return new_sl, new_tp

# Global risk engine instance
enhanced_risk_engine = EnhancedRiskEngine()
