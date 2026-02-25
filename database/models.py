"""
Nexus Trading System - Database Models
SQLAlchemy models for user management, trade logging, and adaptive learning
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from enum import Enum
import hashlib

Base = declarative_base()


class UserRole(str, Enum):
    """User roles for access control"""
    FREE = "free"
    VERIFIED = "verified"
    ELITE = "elite"
    ADMIN = "admin"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    BANNED = "banned"


class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    TRIAL = "trial"


class User(Base):
    """User model for multi-user support"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    full_name = Column(String(255))
    phone = Column(String(20))
    country = Column(String(50))
    
    # Roles and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.FREE)
    status = Column(SQLEnum(UserStatus), default=UserStatus.TRIAL)
    subscription_status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.TRIAL)
    subscription_start = Column(DateTime)
    subscription_end = Column(DateTime)
    subscription_plan = Column(String(50))  # free, monthly, yearly
    
    # Trading permissions
    allowed_assets = Column(JSON)  # List of allowed symbols
    max_daily_trades = Column(Integer, default=5)
    max_risk_percent = Column(Float, default=1.0)
    max_daily_loss = Column(Float, default=9.99)
    auto_trade_enabled = Column(Boolean, default=False)
    
    # API keys
    api_key = Column(String(255), unique=True, index=True)
    api_secret = Column(String(255))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    performance_metrics = relationship("UserPerformance", back_populates="user")
    risk_settings = relationship("UserRiskSettings", back_populates="user", uselist=False)


class UserRiskSettings(Base):
    """Per-user risk management settings"""
    __tablename__ = "user_risk_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Position sizing
    default_risk_percent = Column(Float, default=1.0)
    max_risk_percent = Column(Float, default=2.0)
    dollar_override_enabled = Column(Boolean, default=False)
    default_dollar_risk = Column(Float, default=100.0)
    max_dollar_risk = Column(Float, default=500.0)
    
    # Daily limits
    max_daily_loss = Column(Float, default=9.99)
    max_daily_trades = Column(Integer, default=5)
    max_consecutive_losses = Column(Integer, default=3)
    
    # Asset-specific limits
    asset_limits = Column(JSON)  # Per-asset trade limits
    
    # Time-based restrictions
    trading_hours_enabled = Column(Boolean, default=True)
    allowed_sessions = Column(JSON)  # Trading sessions allowed
    
    # Kill switch
    kill_switch_enabled = Column(Boolean, default=False)
    kill_switch_reason = Column(Text)
    kill_switch_until = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="risk_settings")


class Trade(Base):
    """Trade logging for all users"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # User and session
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(255), index=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # BUY/SELL
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Execution details
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Financials
    pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    
    # Strategy and signal
    strategy = Column(String(100))
    signal_confidence = Column(Float)
    signal_time = Column(DateTime)
    
    # TP/SL management
    sl_locked = Column(Boolean, default=False)
    sl_locked_price = Column(Float)
    tp_extended = Column(Boolean, default=False)
    tp_extended_price = Column(Float)
    runner_mode = Column(Boolean, default=False)
    
    # Signal status
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trades")


class UserPerformance(Base):
    """User performance metrics for adaptive learning"""
    __tablename__ = "user_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Performance period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20))  # daily, weekly, monthly
    
    # Trading metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Financial metrics
    total_pnl = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    # Risk metrics
    avg_risk_per_trade = Column(Float, default=0.0)
    max_risk_per_trade = Column(Float, default=0.0)
    risk_reward_ratio = Column(Float, default=0.0)
    
    # Strategy performance
    strategy_performance = Column(JSON)  # Per-strategy performance
    
    # Asset performance
    asset_performance = Column(JSON)  # Per-asset performance
    
    # Timeframe performance
    timeframe_performance = Column(JSON)  # Per-timeframe performance
    
    # Adaptive learning score
    performance_score = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    
    # Outlier detection
    is_outlier = Column(Boolean, default=False)
    outlier_reason = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="performance_metrics")


class Signal(Base):
    """Signal generation and distribution"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Signal details
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # BUY/SELL
    confidence = Column(Float, nullable=False)
    
    # Strategy and generation
    strategy = Column(String(100), nullable=False)
    strategy_weight = Column(Float, default=1.0)
    
    # Signal generation
    generated_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime)
    
    # Entry parameters
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Market context
    market_regime = Column(String(20))
    volatility_level = Column(String(20))
    session_time = Column(String(20))
    
    # Distribution
    distributed_to_users = Column(JSON)  # List of user IDs who received this signal
    user_responses = Column(JSON)  # How users responded to this signal
    
    # Performance tracking
    actual_result = Column(String(20))  # WIN/LOSS/BREAKEVEN
    actual_pnl = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdaptiveWeight(Base):
    """Adaptive strategy weights based on user performance"""
    __tablename__ = "adaptive_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Weight configuration
    strategy_name = Column(String(100), nullable=False)
    asset = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Current weights
    base_weight = Column(Float, default=1.0)
    adaptive_weight = Column(Float, default=1.0)
    final_weight = Column(Float, default=1.0)
    
    # Performance data
    contributing_users = Column(JSON)  # User IDs contributing to this weight
    performance_score = Column(Float, default=0.0)
    sample_size = Column(Integer, default=0)
    
    # Weight calculation
    calculation_period_start = Column(DateTime)
    calculation_period_end = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Outlier control
    outlier_removed = Column(Boolean, default=False)
    outlier_users = Column(JSON)  # Users removed as outliers
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemLog(Base):
    """System logging for audit and monitoring"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Log details
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    category = Column(String(50), nullable=False)  # AUTH, TRADE, RISK, SYSTEM
    message = Column(Text, nullable=False)
    
    # Context
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Additional data
    extra_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class RevokedToken(Base):
    """Revoked token storage for persistent token revocation"""
    __tablename__ = "revoked_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(String(255), nullable=True)
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Generate SHA256 hash of token"""
        return hashlib.sha256(token.encode()).hexdigest()


class UserDailyStats(Base):
    """User daily trading statistics for atomic risk enforcement"""
    __tablename__ = "user_daily_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    
    # Trading limits
    trade_count = Column(Integer, default=0, nullable=False)
    daily_loss = Column(Float, default=0.0, nullable=False)
    daily_pnl = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")


class UserLockout(Base):
    """User lockout tracking for failed login attempts"""
    __tablename__ = "user_lockouts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Lockout tracking
    failed_attempts = Column(Integer, default=0, nullable=False)
    last_attempt_at = Column(DateTime, nullable=False)
    is_locked = Column(Boolean, default=False, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    # Network information
    ip_address = Column(String(45), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class SystemLog(Base):
    """System log entries for monitoring and analytics"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Log details
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=False, index=True)
    
    # User context
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    request_id = Column(String(255), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User")


class Subscription(Base):
    """Subscription and payment tracking"""
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User and plan
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    plan_name = Column(String(50), nullable=False)
    
    # Subscription period
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Payment details
    payment_method = Column(String(50))  # BTC, USDT, STRIPE, MTN, ORANGE
    payment_amount = Column(Float)
    payment_currency = Column(String(10))
    transaction_id = Column(String(255))
    
    # Status
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)
    
    # Features
    features = Column(JSON)  # List of enabled features
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
