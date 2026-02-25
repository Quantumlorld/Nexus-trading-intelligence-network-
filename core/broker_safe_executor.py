"""
Nexus Trading System - Broker-Safe Trade Executor
Production-ready execution with ledger and reconciliation integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.session import get_database_session
from database.ledger_models import (
    TradeLedger, BrokerPosition, ReconciliationLog, 
    RiskAdjustment, TradingControl, TradeStatus, ReconciliationAction
)
from core.atomic_risk_engine import atomic_risk_engine
from execution.mt5_bridge import MT5Bridge
from config.settings import settings

logger = logging.getLogger(__name__)

class BrokerSafeExecutor:
    """
    Production-ready broker-safe trade executor
    Ensures capital safety through proper ledger management and reconciliation
    """
    
    def __init__(self):
        self.mt5_bridge = MT5Bridge()
        self.max_slippage_percent = 0.5  # Max 0.5% slippage
        self.max_discrepancy_threshold = 1000.0  # Max $1000 discrepancy
        self.reconciliation_interval = 60  # Reconcile every 60 seconds
        self.max_consecutive_failures = 3
        
        # Background tasks
        self.reconciliation_task = None
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("Broker-Safe Executor initialized")
    
    async def execute_trade(self, user_id: int, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade with broker-safe ledger management
        """
        trade_uuid = trade_request.get('trade_uuid')
        start_time = datetime.utcnow()
        
        try:
            # Check if trading is enabled
            if not await self._is_trading_enabled():
                return {
                    'success': False,
                    'error': 'Trading is currently disabled',
                    'trade_uuid': trade_uuid
                }
            
            # Get live broker balance for risk validation
            broker_balance = await self._get_broker_balance(user_id)
            if broker_balance is None:
                return {
                    'success': False,
                    'error': 'Failed to fetch broker balance',
                    'trade_uuid': trade_uuid
                }
            
            # Validate trade using atomic risk engine with live balance
            with next(get_database_session()) as db:
                validation = atomic_risk_engine.validate_trade_atomic(
                    db, user_id, trade_request, broker_balance, 
                    trade_request.get('timeframe')
                )
                
                if not validation.is_allowed:
                    return {
                        'success': False,
                        'error': f"Risk validation failed: {', '.join(validation.reasons)}",
                        'trade_uuid': trade_uuid,
                        'validation': validation
                    }
            
            # Record trade as PENDING in ledger
            with next(get_database_session()) as db:
                # Calculate potential loss
                potential_loss = self._calculate_potential_loss(trade_request)
                
                ledger_entry = TradeLedger(
                    trade_uuid=trade_uuid,
                    user_id=user_id,
                    symbol=trade_request['symbol'],
                    action=trade_request['action'],
                    order_type=trade_request['order_type'],
                    requested_quantity=trade_request['quantity'],
                    entry_price=trade_request['entry_price'],
                    stop_loss=trade_request.get('stop_loss'),
                    take_profit=trade_request.get('take_profit'),
                    potential_loss=potential_loss,
                    status=TradeStatus.PENDING,
                    created_at=start_time
                )
                
                db.add(ledger_entry)
                db.commit()
                db.refresh(ledger_entry)
                
                ledger_id = ledger_entry.id
            
            # Submit trade to broker
            broker_result = await self._submit_to_broker(trade_request, ledger_id)
            
            if not broker_result['success']:
                # Update ledger as REJECTED
                await self._update_ledger_status(ledger_id, TradeStatus.REJECTED, 
                                               broker_result.get('error'))
                
                # Rollback risk calculations
                await self._rollback_risk_calculation(user_id, potential_loss)
                
                return {
                    'success': False,
                    'error': f"Broker rejected: {broker_result.get('error')}",
                    'trade_uuid': trade_uuid,
                    'ledger_id': ledger_id
                }
            
            # Update ledger with broker response
            await self._update_ledger_broker_response(ledger_id, broker_result)
            
            # Handle partial fills
            if broker_result.get('partially_filled', False):
                await self._handle_partial_fill(ledger_id, user_id, broker_result)
            
            # Recalculate risk after execution
            await self._recalculate_risk_post_execution(user_id, broker_result)
            
            # Update broker positions
            await self._update_broker_positions(user_id, trade_request['symbol'])
            
            return {
                'success': True,
                'trade_uuid': trade_uuid,
                'ledger_id': ledger_id,
                'broker_result': broker_result,
                'execution_time': (datetime.utcnow() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            
            # Update ledger as FAILED if we have a ledger entry
            if 'ledger_id' in locals():
                await self._update_ledger_status(ledger_id, TradeStatus.FAILED, str(e))
            
            return {
                'success': False,
                'error': f"Execution failed: {str(e)}",
                'trade_uuid': trade_uuid
            }
    
    async def _submit_to_broker(self, trade_request: Dict[str, Any], ledger_id: int) -> Dict[str, Any]:
        """
        Submit trade to broker with proper error handling
        """
        try:
            # Prepare broker order
            broker_order = {
                'symbol': trade_request['symbol'],
                'action': trade_request['action'],
                'type': trade_request['order_type'],
                'volume': trade_request['quantity'],
                'price': trade_request['entry_price'],
                'sl': trade_request.get('stop_loss'),
                'tp': trade_request.get('take_profit'),
                'comment': f"NEXUS_LEDGER_{ledger_id}"
            }
            
            # Submit to MT5
            result = self.mt5_bridge.place_order(**broker_order)
            
            if result['success']:
                # Calculate slippage
                slippage = self._calculate_slippage(
                    trade_request['entry_price'], 
                    result.get('price', trade_request['entry_price'])
                )
                
                # Check slippage threshold
                if abs(slippage) > self.max_slippage_percent:
                    logger.warning(f"High slippage detected: {slippage:.2f}%")
                
                return {
                    'success': True,
                    'broker_order_id': result.get('order_id'),
                    'broker_position_id': result.get('position_id'),
                    'execution_price': result.get('price'),
                    'filled_quantity': result.get('volume'),
                    'slippage': slippage,
                    'partially_filled': result.get('filled_volume', result.get('volume')) < trade_request['quantity']
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown broker error')
                }
                
        except Exception as e:
            logger.error(f"Broker submission failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _handle_partial_fill(self, ledger_id: int, user_id: int, broker_result: Dict[str, Any]):
        """
        Handle partial fills with risk recalculation
        """
        try:
            with next(get_database_session()) as db:
                ledger = db.query(TradeLedger).filter(TradeLedger.id == ledger_id).first()
                if not ledger:
                    return
                
                # Update ledger with partial fill
                ledger.filled_quantity = broker_result['filled_quantity']
                ledger.execution_price = broker_result['execution_price']
                ledger.status = TradeStatus.PARTIALLY_FILLED
                ledger.executed_at = datetime.utcnow()
                
                # Recalculate potential loss for remaining quantity
                remaining_quantity = ledger.requested_quantity - ledger.filled_quantity
                if remaining_quantity > 0:
                    new_potential_loss = self._calculate_potential_loss({
                        'quantity': remaining_quantity,
                        'entry_price': ledger.execution_price,
                        'stop_loss': ledger.stop_loss,
                        'action': ledger.action
                    })
                    
                    # Record risk adjustment
                    adjustment = RiskAdjustment(
                        trade_ledger_id=ledger_id,
                        user_id=user_id,
                        adjustment_type='PARTIAL_FILL',
                        original_quantity=ledger.requested_quantity,
                        adjusted_quantity=ledger.filled_quantity,
                        original_loss=ledger.potential_loss,
                        adjusted_loss=new_potential_loss,
                        reason=f"Partial fill: {ledger.filled_quantity}/{ledger.requested_quantity}"
                    )
                    db.add(adjustment)
                    
                    # Update potential loss
                    ledger.potential_loss = new_potential_loss
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Partial fill handling failed: {e}")
    
    async def _recalculate_risk_post_execution(self, user_id: int, broker_result: Dict[str, Any]):
        """
        Recalculate risk metrics after execution
        """
        try:
            with next(get_database_session()) as db:
                # Get user's daily stats
                from database.models import UserDailyStats
                today = datetime.utcnow().date()
                start_of_day = datetime.combine(today, datetime.min.time())
                
                daily_stats = db.query(UserDailyStats).filter(
                    UserDailyStats.user_id == user_id,
                    UserDailyStats.date == start_of_day
                ).with_for_update().first()
                
                if daily_stats:
                    # Update actual loss based on execution
                    actual_loss = abs(broker_result.get('slippage', 0) * broker_result.get('filled_quantity', 0))
                    daily_stats.daily_loss += actual_loss
                    daily_stats.last_updated = datetime.utcnow()
                    
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Risk recalculation failed: {e}")
    
    async def _update_broker_positions(self, user_id: int, symbol: str):
        """
        Update broker positions table with current broker state
        """
        try:
            # Get positions from broker
            broker_positions = self.mt5_bridge.get_positions(symbol)
            
            with next(get_database_session()) as db:
                for pos in broker_positions:
                    # Check if position exists
                    db_position = db.query(BrokerPosition).filter(
                        BrokerPosition.user_id == user_id,
                        BrokerPosition.symbol == pos['symbol'],
                        BrokerPosition.broker_position_id == str(pos['position_id'])
                    ).first()
                    
                    if db_position:
                        # Update existing position
                        db_position.quantity = pos['volume'] if pos['type'] == 'buy' else -pos['volume']
                        db_position.avg_price = pos['open_price']
                        db_position.current_price = pos['current_price']
                        db_position.unrealized_pnl = pos['profit']
                        db_position.last_reconciled = datetime.utcnow()
                        db_position.reconciliation_count += 1
                    else:
                        # Create new position
                        db_position = BrokerPosition(
                            user_id=user_id,
                            symbol=pos['symbol'],
                            quantity=pos['volume'] if pos['type'] == 'buy' else -pos['volume'],
                            avg_price=pos['open_price'],
                            current_price=pos['current_price'],
                            unrealized_pnl=pos['profit'],
                            broker_position_id=str(pos['position_id']),
                            last_reconciled=datetime.utcnow()
                        )
                        db.add(db_position)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Broker position update failed: {e}")
    
    async def _reconciliation_loop(self):
        """
        Background reconciliation loop
        """
        while self.is_running:
            try:
                await self._perform_reconciliation()
                await asyncio.sleep(self.reconciliation_interval)
                
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _perform_reconciliation(self):
        """
        Perform reconciliation between ledger and broker
        """
        try:
            with next(get_database_session()) as db:
                # Get all users with active positions
                users = db.query(BrokerPosition.user_id).distinct().all()
                
                for user_tuple in users:
                    user_id = user_tuple[0]
                    await self._reconcile_user_positions(user_id, db)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
    
    async def _reconcile_user_positions(self, user_id: int, db: Session):
        """
        Reconcile positions for a specific user
        """
        try:
            # Get broker positions
            broker_positions = self.mt5_bridge.get_positions()
            
            # Get ledger positions
            ledger_positions = db.query(BrokerPosition).filter(
                BrokerPosition.user_id == user_id
            ).all()
            
            # Compare and detect discrepancies
            for broker_pos in broker_positions:
                symbol = broker_pos['symbol']
                broker_qty = broker_pos['volume'] if broker_pos['type'] == 'buy' else -broker_pos['volume']
                
                # Find corresponding ledger position
                ledger_pos = next((lp for lp in ledger_positions if lp.symbol == symbol), None)
                
                if ledger_pos:
                    discrepancy = abs(ledger_pos.quantity - broker_qty)
                    
                    if discrepancy > 0.01:  # Minimum discrepancy threshold
                        await self._handle_discrepancy(
                            user_id, symbol, ledger_pos.quantity, broker_qty,
                            ledger_pos.avg_price, broker_pos['open_price'], db
                        )
                        
                        # Update ledger position
                        ledger_pos.quantity = broker_qty
                        ledger_pos.avg_price = broker_pos['open_price']
                        ledger_pos.current_price = broker_pos['current_price']
                        ledger_pos.last_reconciled = datetime.utcnow()
                        ledger_pos.reconciliation_count += 1
                        ledger_pos.last_discrepancy = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"User reconciliation failed: {e}")
    
    async def _handle_discrepancy(self, user_id: int, symbol: str, 
                                ledger_qty: float, broker_qty: float,
                                ledger_price: float, broker_price: float, db: Session):
        """
        Handle position discrepancy
        """
        try:
            quantity_discrepancy = abs(ledger_qty - broker_qty)
            price_discrepancy = abs(ledger_price - broker_price) if ledger_price and broker_price else 0
            
            # Determine risk impact
            risk_impact = 'HIGH' if quantity_discrepancy > 10 else 'MEDIUM' if quantity_discrepancy > 1 else 'LOW'
            
            # Check if threshold exceeded
            if quantity_discrepancy * broker_price > self.max_discrepancy_threshold:
                # Stop trading
                await self._emergency_stop(f"Discrepancy threshold exceeded: {symbol}")
                trading_stopped = True
            else:
                trading_stopped = False
            
            # Log discrepancy
            log_entry = ReconciliationLog(
                user_id=user_id,
                symbol=symbol,
                ledger_quantity=ledger_qty,
                broker_quantity=broker_qty,
                quantity_discrepancy=quantity_discrepancy,
                ledger_price=ledger_price,
                broker_price=broker_price,
                price_discrepancy=price_discrepancy,
                action=ReconciliationAction.BROKER_SYNCED,
                risk_impact=risk_impact,
                trading_stopped=trading_stopped,
                broker_data={'quantity': broker_qty, 'price': broker_price},
                ledger_data={'quantity': ledger_qty, 'price': ledger_price}
            )
            
            db.add(log_entry)
            
            # Send alert
            await self._send_alert(
                f"Position discrepancy detected for {symbol}: "
                f"Ledger={ledger_qty}@{ledger_price}, Broker={broker_qty}@{broker_price}",
                risk_impact
            )
            
        except Exception as e:
            logger.error(f"Discrepancy handling failed: {e}")
    
    async def _emergency_stop(self, reason: str):
        """
        Emergency stop trading
        """
        try:
            with next(get_database_session()) as db:
                control = db.query(TradingControl).first()
                if not control:
                    control = TradingControl()
                    db.add(control)
                
                control.trading_enabled = False
                control.stop_reason = reason
                control.stopped_at = datetime.utcnow()
                control.stop_threshold_exceeded = True
                
                db.commit()
            
            await self._send_alert(f"EMERGENCY STOP: {reason}", "HIGH")
            logger.critical(f"Trading stopped: {reason}")
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
    
    async def _is_trading_enabled(self) -> bool:
        """
        Check if trading is enabled
        """
        try:
            with next(get_database_session()) as db:
                control = db.query(TradingControl).first()
                return control.trading_enabled if control else True
        except:
            return False
    
    async def _get_broker_balance(self, user_id: int) -> Optional[float]:
        """
        Get live broker balance
        """
        try:
            account_info = self.mt5_bridge.get_account_info()
            return account_info.balance if account_info else None
        except Exception as e:
            logger.error(f"Failed to get broker balance: {e}")
            return None
    
    def _calculate_potential_loss(self, trade_request: Dict[str, Any]) -> float:
        """
        Calculate potential loss for trade
        """
        if not trade_request.get('stop_loss'):
            return 0.0
        
        entry_price = trade_request['entry_price']
        stop_loss = trade_request['stop_loss']
        quantity = trade_request['quantity']
        
        if trade_request['action'] == 'BUY':
            return abs((entry_price - stop_loss) * quantity)
        else:
            return abs((stop_loss - entry_price) * quantity)
    
    def _calculate_slippage(self, requested_price: float, execution_price: float) -> float:
        """
        Calculate slippage percentage
        """
        if requested_price == 0:
            return 0.0
        
        return ((execution_price - requested_price) / requested_price) * 100
    
    async def _update_ledger_status(self, ledger_id: int, status: TradeStatus, error_message: str = None):
        """
        Update ledger status
        """
        try:
            with next(get_database_session()) as db:
                ledger = db.query(TradeLedger).filter(TradeLedger.id == ledger_id).first()
                if ledger:
                    ledger.status = status
                    ledger.updated_at = datetime.utcnow()
                    if error_message:
                        ledger.error_message = error_message
                    db.commit()
        except Exception as e:
            logger.error(f"Failed to update ledger status: {e}")
    
    async def _update_ledger_broker_response(self, ledger_id: int, broker_result: Dict[str, Any]):
        """
        Update ledger with broker response
        """
        try:
            with next(get_database_session()) as db:
                ledger = db.query(TradeLedger).filter(TradeLedger.id == ledger_id).first()
                if ledger:
                    ledger.broker_order_id = broker_result.get('broker_order_id')
                    ledger.broker_position_id = broker_result.get('broker_position_id')
                    ledger.execution_price = broker_result.get('execution_price')
                    ledger.filled_quantity = broker_result.get('filled_quantity', 0)
                    ledger.slippage = broker_result.get('slippage', 0)
                    ledger.submitted_at = datetime.utcnow()
                    
                    if broker_result.get('partially_filled'):
                        ledger.status = TradeStatus.PARTIALLY_FILLED
                    else:
                        ledger.status = TradeStatus.FILLED
                        ledger.executed_at = datetime.utcnow()
                    
                    db.commit()
        except Exception as e:
            logger.error(f"Failed to update ledger broker response: {e}")
    
    async def _rollback_risk_calculation(self, user_id: int, potential_loss: float):
        """
        Rollback risk calculation on rejected trade
        """
        try:
            with next(get_database_session()) as db:
                from database.models import UserDailyStats
                today = datetime.utcnow().date()
                start_of_day = datetime.combine(today, datetime.min.time())
                
                daily_stats = db.query(UserDailyStats).filter(
                    UserDailyStats.user_id == user_id,
                    UserDailyStats.date == start_of_day
                ).with_for_update().first()
                
                if daily_stats:
                    daily_stats.daily_loss = max(0, daily_stats.daily_loss - potential_loss)
                    daily_stats.last_updated = datetime.utcnow()
                    db.commit()
        except Exception as e:
            logger.error(f"Failed to rollback risk calculation: {e}")
    
    async def _send_alert(self, message: str, severity: str = "MEDIUM"):
        """
        Send alert (placeholder for email/Slack/SMS integration)
        """
        logger.warning(f"ALERT [{severity}]: {message}")
        # TODO: Integrate with email/Slack/SMS services
    
    async def start_background_tasks(self):
        """
        Start background reconciliation and monitoring tasks
        """
        if not self.is_running:
            self.is_running = True
            self.reconciliation_task = asyncio.create_task(self._reconciliation_loop())
            logger.info("Background tasks started")
    
    async def stop_background_tasks(self):
        """
        Stop background tasks
        """
        self.is_running = False
        if self.reconciliation_task:
            self.reconciliation_task.cancel()
        logger.info("Background tasks stopped")

# Global broker-safe executor instance
broker_safe_executor = BrokerSafeExecutor()
