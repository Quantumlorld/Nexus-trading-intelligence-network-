"""
Nexus Trading System - Atomic Risk Engine
Thread-safe atomic risk enforcement with row-level locking
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import text
from database.session import get_database_session
from database.models import UserDailyStats, Trade
from config.settings import settings

logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class TradeValidation:
    """Trade validation result"""
    def __init__(self, is_allowed: bool, risk_score: float, reasons: list, 
                 recommendations: list, max_position_size: float, daily_loss_remaining: float = 0.0):
        self.is_allowed = is_allowed
        self.risk_score = risk_score
        self.reasons = reasons
        self.recommendations = recommendations
        self.max_position_size = max_position_size
        self.daily_loss_remaining = daily_loss_remaining

class AtomicRiskEngine:
    """Thread-safe atomic risk enforcement engine"""
    
    def __init__(self):
        self.max_daily_loss = settings.MAX_DAILY_LOSS
        self.max_risk_percent = settings.MAX_RISK_PERCENT
        self.max_daily_trades_9h = 2
        self.max_daily_trades_6h = 2
        self.max_daily_trades_3h = 1
        self.logger = logging.getLogger(__name__)
    
    def validate_trade_atomic(self, db: Session, user_id: int, signal: Dict[str, Any], 
                              account_balance: float, timeframe: TimeFrame) -> TradeValidation:
        """
        Atomic trade validation with row-level locking
        This method MUST be called within a transaction
        """
        try:
            # Get today's date
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, time.min)
            
            # Lock user's daily stats row for atomic operation
            daily_stats = db.query(UserDailyStats).filter(
                UserDailyStats.user_id == user_id,
                UserDailyStats.date == start_of_day
            ).with_for_update().first()
            
            if not daily_stats:
                # Create new daily stats record
                daily_stats = UserDailyStats(
                    user_id=user_id,
                    date=start_of_day,
                    trade_count=0,
                    daily_loss=0.0,
                    daily_pnl=0.0
                )
                db.add(daily_stats)
                db.flush()  # Get ID without committing
            
            # Check daily loss cap
            if daily_stats.daily_loss >= self.max_daily_loss:
                return TradeValidation(
                    is_allowed=False,
                    risk_score=100.0,
                    reasons=[f"Daily loss cap exceeded: ${daily_stats.daily_loss:.2f} >= ${self.max_daily_loss:.2f}"],
                    recommendations=["Wait for daily reset at midnight UTC"],
                    max_position_size=0.0,
                    daily_loss_remaining=0.0
                )
            
            # Check trade count limits
            max_trades = self._get_max_trades_for_timeframe(timeframe)
            if daily_stats.trade_count >= max_trades:
                return TradeValidation(
                    is_allowed=False,
                    risk_score=90.0,
                    reasons=[f"Daily trade limit exceeded for {timeframe.value}: {daily_stats.trade_count} >= {max_trades}"],
                    recommendations=["Wait for daily reset at midnight UTC"],
                    max_position_size=0.0,
                    daily_loss_remaining=self.max_daily_loss - daily_stats.daily_loss
                )
            
            # Calculate position size based on risk
            risk_amount = min(
                account_balance * (self.max_risk_percent / 100),
                self.max_daily_loss - daily_stats.daily_loss
            )
            
            if risk_amount <= 0:
                return TradeValidation(
                    is_allowed=False,
                    risk_score=100.0,
                    reasons=["No risk capital available"],
                    recommendations=["Wait for daily reset or increase account balance"],
                    max_position_size=0.0,
                    daily_loss_remaining=daily_stats.daily_loss_remaining
                )
            
            # Calculate position size (simplified)
            entry_price = signal.get('entry_price', 0)
            sl_points = 300  # Default SL points
            position_size = risk_amount / (sl_points / 10000)  # Simplified calculation
            
            # Update daily stats atomically
            daily_stats.trade_count += 1
            # Note: daily_loss will be updated after trade execution
            
            return TradeValidation(
                is_allowed=True,
                risk_score=30.0,
                reasons=["Acceptable risk"],
                recommendations=["Proceed with trade execution"],
                max_position_size=position_size,
                daily_loss_remaining=self.max_daily_loss - daily_stats.daily_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error in atomic trade validation: {e}")
            db.rollback()
            return TradeValidation(
                is_allowed=False,
                risk_score=100.0,
                reasons=["System error during validation"],
                recommendations=["Try again later"],
                max_position_size=0.0
            )
    
    def execute_trade_atomic(self, db: Session, user_id: int, trade_data: Dict[str, Any]) -> bool:
        """
        Execute trade with atomic risk enforcement
        This method wraps validation and execution in a single transaction
        """
        try:
            # Begin transaction
            # Validate trade atomically
            validation = self.validate_trade_atomic(
                db, user_id, trade_data, 
                trade_data.get('account_balance', 0),
                TimeFrame(trade_data.get('timeframe', 'H1'))
            )
            
            if not validation.is_allowed:
                db.rollback()
                return False
            
            # Create trade record
            trade = Trade(
                user_id=user_id,
                symbol=trade_data.get('symbol'),
                action=trade_data.get('action'),
                order_type=trade_data.get('order_type', 'MARKET'),
                quantity=trade_data.get('quantity'),
                entry_price=trade_data.get('entry_price'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit'),
                status='FILLED',
                entry_time=datetime.utcnow(),
                timeframe=trade_data.get('timeframe'),
                risk_score=validation.risk_score
            )
            
            db.add(trade)
            
            # Update daily stats with potential loss
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, time.min)
            
            daily_stats = db.query(UserDailyStats).filter(
                UserDailyStats.user_id == user_id,
                UserDailyStats.date == start_of_day
            ).with_for_update().first()
            
            if daily_stats:
                # Calculate potential loss (simplified)
                potential_loss = abs(trade.quantity * (trade.stop_loss - trade.entry_price))
                daily_stats.daily_loss += potential_loss
                daily_stats.last_updated = datetime.utcnow()
            
            # Commit all changes atomically
            db.commit()
            
            self.logger.info(f"Trade executed atomically: User {user_id}, Symbol {trade.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in atomic trade execution: {e}")
            db.rollback()
            return False
    
    def _get_max_trades_for_timeframe(self, timeframe: TimeFrame) -> int:
        """Get maximum trades allowed for timeframe"""
        limits = {
            TimeFrame.H9: self.max_daily_trades_9h,
            TimeFrame.H6: self.max_daily_trades_6h,
            TimeFrame.H3: self.max_daily_trades_3h
        }
        return limits.get(timeframe, 1)
    
    def get_user_daily_stats(self, db: Session, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user's daily trading statistics"""
        try:
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, time.min)
            
            stats = db.query(UserDailyStats).filter(
                UserDailyStats.user_id == user_id,
                UserDailyStats.date == start_of_day
            ).first()
            
            if not stats:
                return {
                    'trade_count': 0,
                    'daily_loss': 0.0,
                    'daily_pnl': 0.0,
                    'last_updated': datetime.utcnow()
                }
            
            return {
                'trade_count': stats.trade_count,
                'daily_loss': stats.daily_loss,
                'daily_pnl': stats.daily_pnl,
                'last_updated': stats.last_updated
            }
            
        except Exception as e:
            self.logger.error(f"Error getting daily stats: {e}")
            return None

# Global atomic risk engine instance
atomic_risk_engine = AtomicRiskEngine()
