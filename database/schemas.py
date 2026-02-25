"""
Nexus Trading System - Database Schemas
Pydantic schemas for API validation and serialization
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    FREE = "free"
    VERIFIED = "verified"
    ELITE = "elite"
    ADMIN = "admin"


class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    BANNED = "banned"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"


class Token(BaseModel):
    """JWT token schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None


# User schemas
class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.FREE
    subscription_plan: str = "free"


class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    auto_trade_enabled: Optional[bool] = None


class UserResponse(UserBase):
    """User response schema"""
    id: int
    uuid: str
    role: UserRole
    status: UserStatus
    subscription_status: SubscriptionStatus
    subscription_plan: Optional[str]
    subscription_end: Optional[datetime]
    allowed_assets: List[str]
    max_daily_trades: int
    max_risk_percent: float
    max_daily_loss: float
    auto_trade_enabled: bool
    api_key: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


# Authentication schemas
class Token(BaseModel):
    """JWT token schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None


class Login(BaseModel):
    """Login schema"""
    username: str
    password: str


# Risk settings schemas
class RiskSettingsBase(BaseModel):
    """Base risk settings schema"""
    default_risk_percent: float = Field(default=1.0, ge=0.1, le=5.0)
    max_risk_percent: float = Field(default=2.0, ge=0.1, le=10.0)
    dollar_override_enabled: bool = False
    default_dollar_risk: Optional[float] = Field(default=100.0, ge=1.0)
    max_dollar_risk: Optional[float] = Field(default=500.0, ge=1.0)
    max_daily_loss: float = Field(default=9.99, ge=1.0, le=1000.0)
    max_daily_trades: int = Field(default=5, ge=1, le=50)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    trading_hours_enabled: bool = True


class RiskSettingsCreate(RiskSettingsBase):
    """Risk settings creation schema"""
    asset_limits: Optional[Dict[str, int]] = None
    allowed_sessions: Optional[List[str]] = None


class RiskSettingsUpdate(BaseModel):
    """Risk settings update schema"""
    default_risk_percent: Optional[float] = Field(None, ge=0.1, le=5.0)
    max_risk_percent: Optional[float] = Field(None, ge=0.1, le=10.0)
    dollar_override_enabled: Optional[bool] = None
    default_dollar_risk: Optional[float] = Field(None, ge=1.0)
    max_dollar_risk: Optional[float] = Field(None, ge=1.0)
    max_daily_loss: Optional[float] = Field(None, ge=1.0, le=1000.0)
    max_daily_trades: Optional[int] = Field(None, ge=1, le=50)
    max_consecutive_losses: Optional[int] = Field(None, ge=1, le=10)
    trading_hours_enabled: Optional[bool] = None
    asset_limits: Optional[Dict[str, int]] = None
    allowed_sessions: Optional[List[str]] = None


class RiskSettingsResponse(RiskSettingsBase):
    """Risk settings response schema"""
    id: int
    user_id: int
    asset_limits: Optional[Dict[str, int]]
    allowed_sessions: Optional[List[str]]
    kill_switch_enabled: bool
    kill_switch_reason: Optional[str]
    kill_switch_until: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Trade schemas
class TradeBase(BaseModel):
    """Base trade schema"""
    symbol: str = Field(..., min_length=1, max_length=20)
    direction: str = Field(..., pattern="^(BUY|SELL)$")
    size: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    strategy: Optional[str] = None
    signal_confidence: Optional[float] = Field(None, ge=0, le=1)


class TradeCreate(TradeBase):
    """Trade creation schema"""
    pass


class TradeUpdate(BaseModel):
    """Trade update schema"""
    exit_price: Optional[float] = Field(None, gt=0)
    exit_time: Optional[datetime] = None
    status: Optional[str] = Field(None, pattern="^(OPEN|CLOSED|CANCELLED)$")
    exit_reason: Optional[str] = None
    sl_locked: Optional[bool] = None
    sl_locked_price: Optional[float] = None
    tp_extended: Optional[bool] = None
    tp_extended_price: Optional[float] = None
    runner_mode: Optional[bool] = None


class TradeResponse(TradeBase):
    """Trade response schema"""
    id: int
    trade_uuid: str
    user_id: int
    session_id: Optional[str]
    entry_time: datetime
    exit_time: Optional[datetime]
    duration_minutes: Optional[int]
    pnl: Optional[float]
    commission: float
    swap: float
    signal_time: Optional[datetime]
    sl_locked: bool
    sl_locked_price: Optional[float]
    tp_extended: bool
    tp_extended_price: Optional[float]
    runner_mode: bool
    status: str
    exit_reason: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Signal schemas
class SignalBase(BaseModel):
    """Base signal schema"""
    symbol: str = Field(..., min_length=1, max_length=20)
    direction: str = Field(..., pattern="^(BUY|SELL)$")
    confidence: float = Field(..., ge=0, le=1)
    strategy: str = Field(..., min_length=1, max_length=100)
    entry_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    market_regime: Optional[str] = None
    volatility_level: Optional[str] = None
    session_time: Optional[str] = None


class SignalCreate(SignalBase):
    """Signal creation schema"""
    expires_at: Optional[datetime] = None


class SignalResponse(SignalBase):
    """Signal response schema"""
    id: int
    signal_uuid: str
    strategy_weight: float
    generated_at: datetime
    expires_at: Optional[datetime]
    distributed_to_users: Optional[List[int]]
    user_responses: Optional[Dict[str, Any]]
    actual_result: Optional[str]
    actual_pnl: Optional[float]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Performance schemas
class PerformanceMetrics(BaseModel):
    """Performance metrics schema"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_risk_per_trade: float
    max_risk_per_trade: float
    risk_reward_ratio: float
    performance_score: float
    consistency_score: float
    risk_score: float


class PerformanceResponse(BaseModel):
    """Performance response schema"""
    id: int
    user_id: int
    period_start: datetime
    period_end: datetime
    period_type: str
    metrics: PerformanceMetrics
    strategy_performance: Optional[Dict[str, Any]]
    asset_performance: Optional[Dict[str, Any]]
    timeframe_performance: Optional[Dict[str, Any]]
    is_outlier: bool
    outlier_reason: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Subscription schemas
class SubscriptionCreate(BaseModel):
    """Subscription creation schema"""
    plan_name: str = Field(..., min_length=1, max_length=50)
    payment_method: str = Field(..., min_length=1, max_length=50)
    payment_amount: float = Field(..., gt=0)
    payment_currency: str = Field(..., min_length=3, max_length=10)
    transaction_id: Optional[str] = None


class SubscriptionResponse(BaseModel):
    """Subscription response schema"""
    id: int
    user_id: int
    plan_name: str
    start_date: datetime
    end_date: datetime
    payment_method: str
    payment_amount: float
    payment_currency: str
    transaction_id: Optional[str]
    status: SubscriptionStatus
    features: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# API Key schemas
class APIKeyCreate(BaseModel):
    """API key creation schema"""
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default=["read_signals"])


class APIKeyResponse(BaseModel):
    """API key response schema"""
    id: int
    name: str
    api_key: str
    api_secret: str
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime]
    
    class Config:
        from_attributes = True


# System schemas
class SystemStatus(BaseModel):
    """System status schema"""
    status: str
    version: str
    uptime: float
    active_users: int
    total_trades: int
    total_pnl: float
    last_signal: Optional[datetime]
    kill_switch_active: bool


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Success response schema"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @classmethod
    def create(cls, items: List[Any], total: int, page: int, size: int):
        pages = (total + size - 1) // size
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages
        )


# Webhook schemas
class WebhookPayload(BaseModel):
    """Webhook payload schema"""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    signature: Optional[str] = None


# Dashboard schemas
class DashboardStats(BaseModel):
    """Dashboard statistics schema"""
    total_users: int
    active_users: int
    total_trades: int
    total_pnl: float
    best_performing_user: Optional[str]
    worst_performing_user: Optional[str]
    most_traded_asset: str
    current_regime: str
    system_health: str


class UserDashboard(BaseModel):
    """User dashboard schema"""
    user_info: UserResponse
    current_positions: List[TradeResponse]
    recent_trades: List[TradeResponse]
    performance_metrics: PerformanceMetrics
    risk_settings: RiskSettingsResponse
    subscription_info: SubscriptionResponse
    available_signals: List[SignalResponse]
