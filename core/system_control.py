"""
Nexus Trading System - System Control Module
Global emergency kill switch and operational controls
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

from database.session import get_database_session

Base = declarative_base()
logger = logging.getLogger(__name__)

class SystemControl(Base):
    """System control table for operational state management"""
    __tablename__ = "system_control"
    
    id = Column(Integer, primary_key=True, index=True)
    trading_enabled = Column(Boolean, default=True, nullable=False)
    reason = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_by = Column(String(100), nullable=True)
    
    @classmethod
    def get_current_state(cls, db: Session) -> 'SystemControl':
        """Get current system control state"""
        control = db.query(cls).order_by(cls.id.desc()).first()
        if not control:
            # Create initial state
            control = cls(
                trading_enabled=True,
                reason="System initialized",
                updated_at=datetime.utcnow(),
                updated_by="system"
            )
            db.add(control)
            db.commit()
            db.refresh(control)
        return control
    
    @classmethod
    def set_trading_state(cls, db: Session, enabled: bool, reason: str, updated_by: str) -> 'SystemControl':
        """Set trading enabled/disabled state"""
        control = cls(
            trading_enabled=enabled,
            reason=reason,
            updated_at=datetime.utcnow(),
            updated_by=updated_by
        )
        db.add(control)
        db.commit()
        db.refresh(control)
        
        # Log the change
        action = "ENABLED" if enabled else "DISABLED"
        logger.warning(f"TRADING {action}: {reason} by {updated_by}")
        
        return control

class SystemControlManager:
    """Manages system control operations"""
    
    def __init__(self):
        self.broker_heartbeat_interval = 10  # seconds
        self.broker_failure_threshold = 3  # consecutive failures
        self.db_latency_threshold = 2.0  # seconds
        self.broker_consecutive_failures = 0
        self.db_consecutive_failures = 0
        self.monitoring_task = None
        self.is_monitoring = False
        
    async def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled"""
        try:
            with next(get_database_session()) as db:
                control = SystemControl.get_current_state(db)
                return control.trading_enabled
        except Exception as e:
            logger.error(f"Failed to check trading state: {e}")
            # Fail safe - disable trading on error
            return False
    
    async def disable_trading(self, reason: str, updated_by: str = "system") -> bool:
        """Disable trading immediately"""
        try:
            with next(get_database_session()) as db:
                SystemControl.set_trading_state(db, False, reason, updated_by)
                
                # Send alert
                await self._send_alert(f"TRADING DISABLED: {reason}", "CRITICAL")
                
                logger.critical(f"Trading disabled by {updated_by}: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to disable trading: {e}")
            return False
    
    async def enable_trading(self, reason: str, updated_by: str = "system") -> bool:
        """Enable trading"""
        try:
            with next(get_database_session()) as db:
                SystemControl.set_trading_state(db, True, reason, updated_by)
                
                # Send alert
                await self._send_alert(f"TRADING ENABLED: {reason}", "INFO")
                
                logger.info(f"Trading enabled by {updated_by}: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to enable trading: {e}")
            return False
    
    async def check_broker_connectivity(self) -> bool:
        """Check broker connectivity"""
        try:
            from execution.mt5_bridge import MT5Bridge
            bridge = MT5Bridge()
            
            # Simple ping - get account info
            account_info = bridge.get_account_info()
            
            if account_info:
                self.broker_consecutive_failures = 0
                logger.debug("Broker connectivity check passed")
                return True
            else:
                self.broker_consecutive_failures += 1
                logger.warning(f"Broker connectivity failed (attempt {self.broker_consecutive_failures})")
                
                if self.broker_consecutive_failures >= self.broker_failure_threshold:
                    await self.disable_trading(
                        f"Broker connectivity failure after {self.broker_consecutive_failures} attempts",
                        "system_monitor"
                    )
                
                return False
                
        except Exception as e:
            self.broker_consecutive_failures += 1
            logger.error(f"Broker connectivity error (attempt {self.broker_consecutive_failures}): {e}")
            
            if self.broker_consecutive_failures >= self.broker_failure_threshold:
                await self.disable_trading(
                    f"Broker connectivity error after {self.broker_consecutive_failures} attempts: {str(e)}",
                    "system_monitor"
                )
            
            return False
    
    async def check_database_health(self) -> bool:
        """Check database health and latency"""
        try:
            start_time = datetime.utcnow()
            
            # Simple ping query
            with next(get_database_session()) as db:
                db.execute("SELECT 1")
                db.commit()
            
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            if latency > self.db_latency_threshold:
                self.db_consecutive_failures += 1
                logger.warning(f"Database latency {latency:.2f}s exceeds threshold {self.db_latency_threshold}s (attempt {self.db_consecutive_failures})")
                
                if self.db_consecutive_failures >= 3:
                    await self.disable_trading(
                        f"Database latency {latency:.2f}s exceeds threshold after {self.db_consecutive_failures} attempts",
                        "system_monitor"
                    )
                
                return False
            else:
                self.db_consecutive_failures = 0
                logger.debug(f"Database health check passed (latency: {latency:.2f}s)")
                return True
                
        except Exception as e:
            self.db_consecutive_failures += 1
            logger.error(f"Database health check failed (attempt {self.db_consecutive_failures}): {e}")
            
            if self.db_consecutive_failures >= 3:
                await self.disable_trading(
                    f"Database connection failed after {self.db_consecutive_failures} attempts: {str(e)}",
                    "system_monitor"
                )
            
            return False
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Check broker connectivity
                await self.check_broker_connectivity()
                
                # Check database health
                await self.check_database_health()
                
                # Wait for next check
                await asyncio.sleep(self.broker_heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _send_alert(self, message: str, severity: str = "INFO"):
        """Send alert (placeholder for integration)"""
        logger.warning(f"ALERT [{severity}]: {message}")
        # TODO: Integrate with email/Slack/SMS services

# Global system control manager
system_control_manager = SystemControlManager()
