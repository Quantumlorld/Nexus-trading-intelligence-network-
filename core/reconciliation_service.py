"""
Nexus Trading System - Reconciliation Service
Production-ready reconciliation with automated discrepancy detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func

from database.session import get_database_session
from database.ledger_models import (
    BrokerPosition, ReconciliationLog, TradingControl, 
    ReconciliationAction, RiskAdjustment
)
from execution.mt5_bridge import MT5Bridge
from config.settings import settings

logger = logging.getLogger(__name__)

class ReconciliationService:
    """
    Production reconciliation service
    Ensures ledger and broker state consistency
    """
    
    def __init__(self):
        self.mt5_bridge = MT5Bridge()
        self.reconciliation_interval = 60  # seconds
        self.max_discrepancy_threshold = 1000.0  # dollars
        self.max_consecutive_failures = 3
        self.alert_thresholds = {
            'LOW': 100.0,
            'MEDIUM': 500.0,
            'HIGH': 1000.0
        }
        
        # State tracking
        self.consecutive_failures = 0
        self.last_successful_reconciliation = None
        self.is_running = False
        
        logger.info("Reconciliation Service initialized")
    
    async def start_reconciliation(self):
        """
        Start the reconciliation service
        """
        if self.is_running:
            logger.warning("Reconciliation service already running")
            return
        
        self.is_running = True
        logger.info("Starting reconciliation service")
        
        try:
            while self.is_running:
                await self._perform_full_reconciliation()
                await asyncio.sleep(self.reconciliation_interval)
                
        except Exception as e:
            logger.error(f"Reconciliation service error: {e}")
            await self._handle_reconciliation_failure(str(e))
        finally:
            self.is_running = False
            logger.info("Reconciliation service stopped")
    
    async def _perform_full_reconciliation(self):
        """
        Perform full reconciliation for all users
        """
        try:
            start_time = datetime.utcnow()
            total_discrepancies = 0
            high_risk_discrepancies = 0
            
            with next(get_database_session()) as db:
                # Get all users with positions
                users_with_positions = db.query(BrokerPosition.user_id).distinct().all()
                
                for user_tuple in users_with_positions:
                    user_id = user_tuple[0]
                    
                    # Reconcile user positions
                    user_discrepancies = await self._reconcile_user_positions(user_id, db)
                    total_discrepancies += user_discrepancies['total']
                    high_risk_discrepancies += user_discrepancies['high_risk']
                
                # Update reconciliation statistics
                await self._update_reconciliation_stats(db, total_discrepancies, high_risk_discrepancies)
                
                # Check if we need to stop trading
                if high_risk_discrepancies > 0:
                    await self._check_trading_stop_conditions(db, high_risk_discrepancies)
                
                self.consecutive_failures = 0
                self.last_successful_reconciliation = start_time
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Reconciliation completed: {total_discrepancies} discrepancies in {duration:.2f}s")
                
        except Exception as e:
            logger.error(f"Full reconciliation failed: {e}")
            await self._handle_reconciliation_failure(str(e))
    
    async def _reconcile_user_positions(self, user_id: int, db: Session) -> Dict[str, int]:
        """
        Reconcile positions for a specific user
        """
        discrepancies = {'total': 0, 'high_risk': 0}
        
        try:
            # Get live broker positions
            broker_positions = self.mt5_bridge.get_positions()
            user_broker_positions = [pos for pos in broker_positions]  # Filter by user if needed
            
            # Get ledger positions
            ledger_positions = db.query(BrokerPosition).filter(
                BrokerPosition.user_id == user_id
            ).all()
            
            # Create position maps for easy comparison
            broker_pos_map = {pos['symbol']: pos for pos in user_broker_positions}
            ledger_pos_map = {pos.symbol: pos for pos in ledger_positions}
            
            # Get all unique symbols
            all_symbols = set(broker_pos_map.keys()) | set(ledger_pos_map.keys())
            
            for symbol in all_symbols:
                broker_pos = broker_pos_map.get(symbol)
                ledger_pos = ledger_pos_map.get(symbol)
                
                # Reconcile this symbol position
                discrepancy = await self._reconcile_symbol_position(
                    user_id, symbol, broker_pos, ledger_pos, db
                )
                
                if discrepancy['has_discrepancy']:
                    discrepancies['total'] += 1
                    if discrepancy['risk_level'] == 'HIGH':
                        discrepancies['high_risk'] += 1
                
                # Update ledger position with broker data
                await self._update_ledger_position(user_id, symbol, broker_pos, db)
            
            return discrepancies
            
        except Exception as e:
            logger.error(f"User position reconciliation failed for user {user_id}: {e}")
            return discrepancies
    
    async def _reconcile_symbol_position(self, user_id: int, symbol: str,
                                     broker_pos: Optional[Dict], ledger_pos: Optional[BrokerPosition],
                                     db: Session) -> Dict[str, Any]:
        """
        Reconcile a specific symbol position
        """
        discrepancy_info = {
            'has_discrepancy': False,
            'risk_level': 'LOW',
            'quantity_diff': 0.0,
            'price_diff': 0.0,
            'pnl_diff': 0.0
        }
        
        try:
            # Extract position data
            broker_qty = broker_pos['volume'] if broker_pos else 0.0
            broker_qty = broker_qty if broker_pos['type'] == 'buy' else -broker_qty if broker_pos else 0.0
            broker_price = broker_pos['open_price'] if broker_pos else 0.0
            broker_pnl = broker_pos['profit'] if broker_pos else 0.0
            
            ledger_qty = ledger_pos.quantity if ledger_pos else 0.0
            ledger_price = ledger_pos.avg_price if ledger_pos else 0.0
            ledger_pnl = ledger_pos.unrealized_pnl if ledger_pos else 0.0
            
            # Calculate discrepancies
            quantity_diff = abs(ledger_qty - broker_qty)
            price_diff = abs(ledger_price - broker_price) if ledger_price and broker_price else 0.0
            pnl_diff = abs(ledger_pnl - broker_pnl)
            
            # Determine if there's a significant discrepancy
            has_quantity_discrepancy = quantity_diff > 0.01  # Minimum threshold
            has_price_discrepancy = price_diff > 0.0001  # Minimum threshold
            
            if has_quantity_discrepancy or has_price_discrepancy:
                discrepancy_info['has_discrepancy'] = True
                discrepancy_info['quantity_diff'] = quantity_diff
                discrepancy_info['price_diff'] = price_diff
                discrepancy_info['pnl_diff'] = pnl_diff
                
                # Determine risk level
                monetary_discrepancy = max(
                    quantity_diff * broker_price,
                    price_diff * abs(broker_qty)
                )
                
                if monetary_discrepancy >= self.alert_thresholds['HIGH']:
                    discrepancy_info['risk_level'] = 'HIGH'
                elif monetary_discrepancy >= self.alert_thresholds['MEDIUM']:
                    discrepancy_info['risk_level'] = 'MEDIUM'
                else:
                    discrepancy_info['risk_level'] = 'LOW'
                
                # Log the discrepancy
                await self._log_discrepancy(
                    user_id, symbol, ledger_qty, broker_qty,
                    ledger_price, broker_price, ledger_pnl, broker_pnl,
                    discrepancy_info, db
                )
                
                # Send alert if high risk
                if discrepancy_info['risk_level'] in ['MEDIUM', 'HIGH']:
                    await self._send_discrepancy_alert(
                        user_id, symbol, discrepancy_info
                    )
            
            return discrepancy_info
            
        except Exception as e:
            logger.error(f"Symbol position reconciliation failed for {symbol}: {e}")
            return discrepancy_info
    
    async def _update_ledger_position(self, user_id: int, symbol: str,
                                   broker_pos: Optional[Dict], db: Session):
        """
        Update ledger position with broker data
        """
        try:
            if not broker_pos:
                # No broker position, check if we need to close ledger position
                ledger_pos = db.query(BrokerPosition).filter(
                    BrokerPosition.user_id == user_id,
                    BrokerPosition.symbol == symbol
                ).first()
                
                if ledger_pos:
                    # Position closed in broker, update ledger
                    ledger_pos.quantity = 0.0
                    ledger_pos.last_reconciled = datetime.utcnow()
                    logger.info(f"Position {symbol} closed in broker for user {user_id}")
                
                return
            
            # Update or create ledger position
            ledger_pos = db.query(BrokerPosition).filter(
                BrokerPosition.user_id == user_id,
                BrokerPosition.symbol == symbol
            ).first()
            
            quantity = broker_pos['volume'] if broker_pos['type'] == 'buy' else -broker_pos['volume']
            
            if ledger_pos:
                # Update existing position
                ledger_pos.quantity = quantity
                ledger_pos.avg_price = broker_pos['open_price']
                ledger_pos.current_price = broker_pos['current_price']
                ledger_pos.unrealized_pnl = broker_pos['profit']
                ledger_pos.margin_used = broker_pos.get('margin', 0.0)
                ledger_pos.last_reconciled = datetime.utcnow()
                ledger_pos.reconciliation_count += 1
            else:
                # Create new position
                ledger_pos = BrokerPosition(
                    user_id=user_id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=broker_pos['open_price'],
                    current_price=broker_pos['current_price'],
                    unrealized_pnl=broker_pos['profit'],
                    margin_used=broker_pos.get('margin', 0.0),
                    broker_position_id=str(broker_pos['position_id']),
                    last_reconciled=datetime.utcnow()
                )
                db.add(ledger_pos)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update ledger position for {symbol}: {e}")
            db.rollback()
    
    async def _log_discrepancy(self, user_id: int, symbol: str,
                               ledger_qty: float, broker_qty: float,
                               ledger_price: float, broker_price: float,
                               ledger_pnl: float, broker_pnl: float,
                               discrepancy_info: Dict[str, Any], db: Session):
        """
        Log discrepancy to reconciliation log
        """
        try:
            # Determine action
            if discrepancy_info['risk_level'] == 'HIGH':
                action = ReconciliationAction.TRADING_STOPPED
            elif discrepancy_info['risk_level'] == 'MEDIUM':
                action = ReconciliationAction.MANUAL_REVIEW
            else:
                action = ReconciliationAction.BROKER_SYNCED
            
            log_entry = ReconciliationLog(
                user_id=user_id,
                symbol=symbol,
                ledger_quantity=ledger_qty,
                broker_quantity=broker_qty,
                quantity_discrepancy=discrepancy_info['quantity_diff'],
                ledger_price=ledger_price,
                broker_price=broker_price,
                price_discrepancy=discrepancy_info['price_diff'],
                action=action,
                risk_impact=discrepancy_info['risk_level'],
                trading_stopped=(discrepancy_info['risk_level'] == 'HIGH'),
                broker_data={
                    'quantity': broker_qty,
                    'price': broker_price,
                    'pnl': broker_pnl
                },
                ledger_data={
                    'quantity': ledger_qty,
                    'price': ledger_price,
                    'pnl': ledger_pnl
                }
            )
            
            db.add(log_entry)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log discrepancy: {e}")
    
    async def _update_reconciliation_stats(self, db: Session, total_discrepancies: int, high_risk_discrepancies: int):
        """
        Update reconciliation statistics
        """
        try:
            # Update trading control if needed
            control = db.query(TradingControl).first()
            if not control:
                control = TradingControl()
                db.add(control)
            
            control.last_check = datetime.utcnow()
            
            # Reset consecutive failures on success
            if total_discrepancies == 0:
                control.consecutive_failures = 0
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update reconciliation stats: {e}")
    
    async def _check_trading_stop_conditions(self, db: Session, high_risk_discrepancies: int):
        """
        Check if trading should be stopped
        """
        try:
            control = db.query(TradingControl).first()
            if not control:
                control = TradingControl()
                db.add(control)
            
            # Stop trading if high risk discrepancies exceed threshold
            if high_risk_discrepancies > 0:
                if control.trading_enabled:
                    control.trading_enabled = False
                    control.stop_reason = f"High risk discrepancies detected: {high_risk_discrepancies}"
                    control.stopped_at = datetime.utcnow()
                    control.stop_threshold_exceeded = True
                    
                    await self._send_trading_stop_alert(
                        "Trading stopped due to high-risk discrepancies",
                        high_risk_discrepancies
                    )
            
            # Stop trading if consecutive failures exceed limit
            elif self.consecutive_failures >= self.max_consecutive_failures:
                if control.trading_enabled:
                    control.trading_enabled = False
                    control.stop_reason = f"Consecutive reconciliation failures: {self.consecutive_failures}"
                    control.stopped_at = datetime.utcnow()
                    control.consecutive_failures = self.consecutive_failures
                    
                    await self._send_trading_stop_alert(
                        "Trading stopped due to consecutive reconciliation failures",
                        self.consecutive_failures
                    )
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to check trading stop conditions: {e}")
    
    async def _handle_reconciliation_failure(self, error: str):
        """
        Handle reconciliation failure
        """
        self.consecutive_failures += 1
        logger.error(f"Reconciliation failure #{self.consecutive_failures}: {error}")
        
        # Send alert if approaching threshold
        if self.consecutive_failures >= self.max_consecutive_failures - 1:
            await self._send_alert(
                f"Reconciliation failures approaching threshold: {self.consecutive_failures}/{self.max_consecutive_failures}",
                "HIGH"
            )
    
    async def _send_discrepancy_alert(self, user_id: int, symbol: str, discrepancy_info: Dict[str, Any]):
        """
        Send discrepancy alert
        """
        message = (
            f"Position discrepancy detected for user {user_id}, symbol {symbol}:\n"
            f"Quantity diff: {discrepancy_info['quantity_diff']}\n"
            f"Price diff: {discrepancy_info['price_diff']}\n"
            f"Risk level: {discrepancy_info['risk_level']}"
        )
        
        await self._send_alert(message, discrepancy_info['risk_level'])
    
    async def _send_trading_stop_alert(self, reason: str, metric: int):
        """
        Send trading stop alert
        """
        message = f"TRADING STOPPED: {reason} (Metric: {metric})"
        await self._send_alert(message, "HIGH")
    
    async def _send_alert(self, message: str, severity: str = "MEDIUM"):
        """
        Send alert (placeholder for email/Slack/SMS integration)
        """
        logger.warning(f"RECONCILIATION ALERT [{severity}]: {message}")
        
        # TODO: Integrate with alert services
        # - Email service
        # - Slack webhook
        # - SMS service
        # - Push notifications
    
    async def get_reconciliation_report(self, user_id: Optional[int] = None,
                                    hours: int = 24) -> Dict[str, Any]:
        """
        Get reconciliation report
        """
        try:
            with next(get_database_session()) as db:
                # Time filter
                since = datetime.utcnow() - timedelta(hours=hours)
                
                # Base query
                query = db.query(ReconciliationLog).filter(
                    ReconciliationLog.reconciliation_time >= since
                )
                
                if user_id:
                    query = query.filter(ReconciliationLog.user_id == user_id)
                
                # Get discrepancies
                discrepancies = query.all()
                
                # Calculate statistics
                total_discrepancies = len(discrepancies)
                high_risk_count = len([d for d in discrepancies if d.risk_impact == 'HIGH'])
                medium_risk_count = len([d for d in discrepancies if d.risk_impact == 'MEDIUM'])
                low_risk_count = len([d for d in discrepancies if d.risk_impact == 'LOW'])
                
                # Get trading control status
                control = db.query(TradingControl).first()
                trading_enabled = control.trading_enabled if control else True
                
                return {
                    'period_hours': hours,
                    'total_discrepancies': total_discrepancies,
                    'high_risk_discrepancies': high_risk_count,
                    'medium_risk_discrepancies': medium_risk_count,
                    'low_risk_discrepancies': low_risk_count,
                    'trading_enabled': trading_enabled,
                    'last_successful_reconciliation': self.last_successful_reconciliation,
                    'consecutive_failures': self.consecutive_failures,
                    'recent_discrepancies': [
                        {
                            'symbol': d.symbol,
                            'risk_impact': d.risk_impact,
                            'quantity_discrepancy': d.quantity_discrepancy,
                            'price_discrepancy': d.price_discrepancy,
                            'action': d.action.value,
                            'reconciliation_time': d.reconciliation_time.isoformat()
                        }
                        for d in discrepancies[-10:]  # Last 10 discrepancies
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to generate reconciliation report: {e}")
            return {'error': str(e)}
    
    async def force_reconciliation(self, user_id: Optional[int] = None):
        """
        Force immediate reconciliation
        """
        try:
            if user_id:
                with next(get_database_session()) as db:
                    discrepancies = await self._reconcile_user_positions(user_id, db)
                    logger.info(f"Force reconciliation for user {user_id}: {discrepancies}")
            else:
                await self._perform_full_reconciliation()
                logger.info("Force full reconciliation completed")
                
        except Exception as e:
            logger.error(f"Force reconciliation failed: {e}")
    
    def stop_reconciliation(self):
        """
        Stop the reconciliation service
        """
        self.is_running = False
        logger.info("Reconciliation service stop requested")

# Global reconciliation service instance
reconciliation_service = ReconciliationService()
