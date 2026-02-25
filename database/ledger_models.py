"""
Nexus Trading System - Broker-Safe Ledger Models
Production-ready ledger and reconciliation database schema
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum
import uuid

Base = declarative_base()

class TradeStatus(str, Enum):
    """Trade execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"

class ReconciliationAction(str, Enum):
    """Reconciliation action types"""
    NO_ACTION = "no_action"
    LEDGER_ADJUSTED = "ledger_adjusted"
    BROKER_SYNCED = "broker_synced"
    MANUAL_REVIEW = "manual_review"
    TRADING_STOPPED = "trading_stopped"

class TradeLedger(Base):
    """
    Production trade ledger for broker-safe execution
    Tracks all trade lifecycle events with atomic precision
    """
    __tablename__ = "trade_ledger"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Trade request details
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # BUY/SELL
    order_type = Column(String(10), nullable=False)  # MARKET/LIMIT
    requested_quantity = Column(Float, nullable=False)
    
    # Execution details
    filled_quantity = Column(Float, default=0.0, nullable=False)
    entry_price = Column(Float, nullable=False)
    execution_price = Column(Float, nullable=True)
    avg_execution_price = Column(Float, nullable=True)
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    slippage = Column(Float, default=0.0, nullable=False)
    potential_loss = Column(Float, nullable=False)
    actual_loss = Column(Float, nullable=True)
    
    # Status and timing
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.PENDING, nullable=False, index=True)
    broker_order_id = Column(String(100), nullable=True, index=True)
    broker_position_id = Column(String(100), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    submitted_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_ledger_user_status', 'user_id', 'status'),
        Index('idx_ledger_symbol_status', 'symbol', 'status'),
        Index('idx_ledger_created_at', 'created_at'),
    )

class BrokerPosition(Base):
    """
    Broker position tracking for reconciliation
    Stores authoritative broker state
    """
    __tablename__ = "broker_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    quantity = Column(Float, nullable=False)  # Positive for long, negative for short
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0, nullable=False)
    
    # Risk metrics
    margin_used = Column(Float, default=0.0, nullable=False)
    margin_free = Column(Float, nullable=False)
    
    # Reconciliation tracking
    last_reconciled = Column(DateTime, default=datetime.utcnow, nullable=False)
    reconciliation_count = Column(Integer, default=0, nullable=False)
    last_discrepancy = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Broker metadata
    broker_position_id = Column(String(100), nullable=True, index=True)
    broker_account = Column(String(50), nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Unique constraint and indexes
    __table_args__ = (
        Index('idx_broker_user_symbol', 'user_id', 'symbol'),
        Index('idx_broker_updated', 'updated_at'),
    )

class ReconciliationLog(Base):
    """
    Reconciliation audit log
    Tracks all discrepancies and corrective actions
    """
    __tablename__ = "reconciliation_log"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Discrepancy details
    ledger_quantity = Column(Float, nullable=False)
    broker_quantity = Column(Float, nullable=False)
    quantity_discrepancy = Column(Float, nullable=False)
    ledger_price = Column(Float, nullable=True)
    broker_price = Column(Float, nullable=True)
    price_discrepancy = Column(Float, nullable=True)
    
    # Action taken
    action = Column(SQLEnum(ReconciliationAction), nullable=False, index=True)
    action_details = Column(Text, nullable=True)
    
    # Risk impact
    risk_impact = Column(String(20), nullable=False, index=True)  # HIGH/MEDIUM/LOW
    trading_stopped = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    reconciliation_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Metadata
    broker_data = Column(JSON, nullable=True)
    ledger_data = Column(JSON, nullable=True)
    error_details = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_reconciliation_user_time', 'user_id', 'reconciliation_time'),
        Index('idx_reconciliation_action', 'action'),
        Index('idx_reconciliation_risk', 'risk_impact'),
    )

class RiskAdjustment(Base):
    """
    Risk adjustment tracking for partial fills and slippage
    """
    __tablename__ = "risk_adjustments"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_ledger_id = Column(Integer, ForeignKey("trade_ledger.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Adjustment details
    adjustment_type = Column(String(50), nullable=False, index=True)  # PARTIAL_FILL, SLIPPAGE, RECONCILIATION
    original_quantity = Column(Float, nullable=False)
    adjusted_quantity = Column(Float, nullable=False)
    original_loss = Column(Float, nullable=False)
    adjusted_loss = Column(Float, nullable=False)
    
    # Risk metrics
    daily_loss_before = Column(Float, nullable=False)
    daily_loss_after = Column(Float, nullable=False)
    risk_score_before = Column(Float, nullable=False)
    risk_score_after = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Metadata
    reason = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    trade_ledger = relationship("TradeLedger")
    user = relationship("User")

class TradingControl(Base):
    """
    Global trading control for emergency stops
    """
    __tablename__ = "trading_control"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Control flags
    trading_enabled = Column(Boolean, default=True, nullable=False, index=True)
    reconciliation_enabled = Column(Boolean, default=True, nullable=False)
    
    # Stop conditions
    stop_reason = Column(String(100), nullable=True)
    stop_threshold_exceeded = Column(Boolean, default=False, nullable=False)
    consecutive_failures = Column(Integer, default=0, nullable=False)
    
    # Thresholds
    max_discrepancy_threshold = Column(Float, default=1000.0, nullable=False)  # Max $ discrepancy
    max_consecutive_failures = Column(Integer, default=3, nullable=False)
    slippage_threshold = Column(Float, default=0.5, nullable=False)  # Max 0.5% slippage
    
    # Timestamps
    stopped_at = Column(DateTime, nullable=True)
    last_check = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    stop_details = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_trading_control_enabled', 'trading_enabled'),
        Index('idx_trading_control_updated', 'updated_at'),
    )
