"""
Nexus Trading System - Safe Order Executor
Enhanced execution with safety mechanisms and error handling
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
import json
from contextlib import asynccontextmanager

from config.settings import settings
from core.enhanced_risk_engine import enhanced_risk_engine, TimeFrame
from database.session import get_database_session
from database.models import Trade, User, TradeStatus

class ExecutionStatus(Enum):
    """Execution status types"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class OrderRequest:
    """Order request structure"""
    user_id: int
    symbol: str
    action: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT
    quantity: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timeframe: TimeFrame
    signal_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionResult:
    """Execution result structure"""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[int] = None
    status: ExecutionStatus = ExecutionStatus.FAILED
    message: str = ""
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_time: Optional[datetime] = None
    error_details: Optional[Dict[str, Any]] = None

class SafeOrderExecutor:
    """Safe order executor with comprehensive error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_enabled = True
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        self.order_timeout = 30  # seconds
        self.pending_orders = {}  # Track pending orders
        
        # Network and API health monitoring
        self.api_health_status = True
        self.last_api_check = datetime.utcnow()
        self.api_failure_count = 0
        self.max_api_failures = 5
        
        self.logger.info("Safe Order Executor initialized")
    
    async def execute_order(self, order_request: OrderRequest) -> ExecutionResult:
        """
        Execute order with atomic risk enforcement
        """
        start_time = datetime.utcnow()
        
        try:
            # Use atomic risk engine for validation and execution
            from core.atomic_risk_engine import atomic_risk_engine
            from database.session import get_database_session
            
            with next(get_database_session()) as db:
                # Prepare trade data
                trade_data = {
                    'symbol': order_request.symbol,
                    'action': order_request.action,
                    'order_type': order_request.order_type,
                    'quantity': order_request.quantity,
                    'entry_price': order_request.entry_price,
                    'stop_loss': order_request.stop_loss,
                    'take_profit': order_request.take_profit,
                    'timeframe': order_request.timeframe,
                    'account_balance': order_request.account_balance
                }
                
                # Execute trade atomically
                success = atomic_risk_engine.execute_trade_atomic(
                    db, order_request.user_id, trade_data
                )
                
                if not success:
                    return ExecutionResult(
                        success=False,
                        status=ExecutionStatus.REJECTED,
                        message="Trade rejected by atomic risk validation",
                        execution_time=start_time
                    )
                
                # Simulate order execution (replace with real broker API)
                execution_result = await self._execute_with_retry(order_request, None)
                
                return execution_result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in execute_order: {e}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                execution_time=start_time,
                error_details={"exception": str(e), "timestamp": start_time.isoformat()}
            )
    
    async def _validate_order_request(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Validate order request before execution"""
        
        # Get user account balance
        try:
            with next(get_database_session()) as db:
                user = db.query(User).filter(User.id == order_request.user_id).first()
                if not user:
                    return {"valid": False, "message": "User not found"}
                
                account_balance = user.balance or 0.0
                
        except Exception as e:
            return {"valid": False, "message": f"Failed to get user data: {str(e)}"}
        
        # Validate with risk engine
        signal_data = {
            "symbol": order_request.symbol,
            "action": order_request.action,
            "entry_price": order_request.entry_price,
            "volatility": 1.0  # Would get from market data
        }
        
        risk_validation = enhanced_risk_engine.validate_trade(
            order_request.user_id,
            signal_data,
            account_balance,
            order_request.timeframe
        )
        
        if not risk_validation.is_allowed:
            return {
                "valid": False,
                "message": f"Risk validation failed: {'; '.join(risk_validation.reasons)}"
            }
        
        # Validate order parameters
        if order_request.quantity <= 0:
            return {"valid": False, "message": "Invalid quantity"}
        
        if order_request.action not in ['BUY', 'SELL']:
            return {"valid": False, "message": "Invalid action"}
        
        if order_request.order_type not in ['MARKET', 'LIMIT']:
            return {"valid": False, "message": "Invalid order type"}
        
        if order_request.order_type == 'LIMIT' and not order_request.entry_price:
            return {"valid": False, "message": "Limit order requires entry price"}
        
        return {"valid": True, "message": "Order validation passed"}
    
    async def _check_api_health(self) -> bool:
        """Check API health status"""
        try:
            # Only check health every 30 seconds
            if (datetime.utcnow() - self.last_api_check).seconds < 30:
                return self.api_health_status
            
            # Simulate API health check (would be actual API call)
            # For now, assume healthy unless too many failures
            if self.api_failure_count >= self.max_api_failures:
                self.api_health_status = False
                self.logger.warning("API marked as unhealthy due to repeated failures")
            else:
                self.api_health_status = True
            
            self.last_api_check = datetime.utcnow()
            return self.api_health_status
            
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            self.api_failure_count += 1
            return False
    
    async def _create_trade_record(self, order_request: OrderRequest) -> Optional[int]:
        """Create trade record in database"""
        try:
            with next(get_database_session()) as db:
                trade = Trade(
                    user_id=order_request.user_id,
                    symbol=order_request.symbol,
                    action=order_request.action,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    entry_price=order_request.entry_price,
                    stop_loss=order_request.stop_loss,
                    take_profit=order_request.take_profit,
                    status=TradeStatus.PENDING,
                    entry_time=datetime.utcnow(),
                    timeframe=order_request.timeframe.value,
                    signal_id=order_request.signal_id,
                    metadata=json.dumps(order_request.metadata or {})
                )
                
                db.add(trade)
                db.commit()
                db.refresh(trade)
                
                return trade.id
                
        except Exception as e:
            self.logger.error(f"Failed to create trade record: {e}")
            return None
    
    async def _execute_with_retry(self, order_request: OrderRequest, trade_id: int) -> ExecutionResult:
        """Execute order with retry logic"""
        
        for attempt in range(self.retry_attempts):
            try:
                # Simulate order execution (would integrate with actual broker API)
                order_id = await self._submit_order_to_broker(order_request)
                
                if order_id:
                    # Wait for order execution
                    execution_result = await self._wait_for_execution(order_id, order_request)
                    
                    if execution_result.success:
                        return execution_result
                    elif execution_result.status == ExecutionStatus.REJECTED:
                        return execution_result
                    else:
                        # Failed but can retry
                        self.logger.warning(f"Order execution failed (attempt {attempt + 1}): {execution_result.message}")
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                else:
                    self.logger.error(f"Failed to submit order (attempt {attempt + 1})")
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                    
            except Exception as e:
                self.logger.error(f"Order execution exception (attempt {attempt + 1}): {e}")
                self.api_failure_count += 1
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue
        
        # All attempts failed
        return ExecutionResult(
            success=False,
            status=ExecutionStatus.FAILED,
            message=f"Order execution failed after {self.retry_attempts} attempts"
        )
    
    async def _submit_order_to_broker(self, order_request: OrderRequest) -> Optional[str]:
        """Submit order to broker (simulated)"""
        try:
            # Simulate broker API call
            # In production, this would integrate with MT5, crypto exchanges, etc.
            
            order_data = {
                "symbol": order_request.symbol,
                "action": order_request.action,
                "type": order_request.order_type,
                "quantity": order_request.quantity,
                "price": order_request.entry_price,
                "sl": order_request.stop_loss,
                "tp": order_request.take_profit
            }
            
            # Simulate order submission
            order_id = f"ORDER_{int(time.time())}_{order_request.user_id}"
            
            # Simulate some orders failing for testing
            import random
            if random.random() < 0.1:  # 10% failure rate
                return None
            
            self.logger.info(f"Order submitted to broker: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order to broker: {e}")
            return None
    
    async def _wait_for_execution(self, order_id: str, order_request: OrderRequest) -> ExecutionResult:
        """Wait for order execution with timeout"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < self.order_timeout:
            try:
                # Simulate checking order status
                # In production, this would check actual broker API
                
                import random
                execution_time = random.uniform(0.5, 3.0)  # Simulate execution time
                
                if (datetime.utcnow() - start_time).seconds >= execution_time:
                    # Simulate execution result
                    if random.random() < 0.8:  # 80% success rate
                        executed_price = order_request.entry_price or 1.0
                        executed_quantity = order_request.quantity
                        
                        return ExecutionResult(
                            success=True,
                            order_id=order_id,
                            status=ExecutionStatus.FILLED,
                            message="Order filled successfully",
                            executed_price=executed_price,
                            executed_quantity=executed_quantity,
                            execution_time=datetime.utcnow()
                        )
                    else:
                        return ExecutionResult(
                            success=False,
                            order_id=order_id,
                            status=ExecutionStatus.REJECTED,
                            message="Order rejected by broker",
                            execution_time=datetime.utcnow()
                        )
                
                await asyncio.sleep(0.5)  # Check every 0.5 seconds
                
            except Exception as e:
                self.logger.error(f"Error checking order status: {e}")
                await asyncio.sleep(1.0)
        
        # Timeout
        return ExecutionResult(
            success=False,
            order_id=order_id,
            status=ExecutionStatus.FAILED,
            message="Order execution timeout",
            execution_time=datetime.utcnow()
        )
    
    async def _update_trade_record(self, trade_id: int, execution_result: ExecutionResult):
        """Update trade record with execution result"""
        try:
            with next(get_database_session()) as db:
                trade = db.query(Trade).filter(Trade.id == trade_id).first()
                if trade:
                    if execution_result.success:
                        trade.status = TradeStatus.FILLED
                        trade.exit_price = execution_result.executed_price
                        trade.exit_time = execution_result.execution_time
                        trade.pnl = self._calculate_pnl(trade, execution_result)
                    else:
                        trade.status = TradeStatus.REJECTED if execution_result.status == ExecutionStatus.REJECTED else TradeStatus.FAILED
                    
                    trade.metadata = json.dumps({
                        "order_id": execution_result.order_id,
                        "execution_result": execution_result.message,
                        "executed_price": execution_result.executed_price,
                        "executed_quantity": execution_result.executed_quantity
                    })
                    
                    db.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to update trade record: {e}")
    
    def _calculate_pnl(self, trade: Trade, execution_result: ExecutionResult) -> float:
        """Calculate P&L for trade (simplified)"""
        try:
            if not execution_result.executed_price:
                return 0.0
            
            entry_price = trade.entry_price or 0.0
            exit_price = execution_result.executed_price
            quantity = execution_result.executed_quantity or trade.quantity
            
            if trade.action == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
            
            return pnl
            
        except Exception:
            return 0.0
    
    def _log_execution(self, order_request: OrderRequest, execution_result: ExecutionResult):
        """Log execution details"""
        log_data = {
            "user_id": order_request.user_id,
            "symbol": order_request.symbol,
            "action": order_request.action,
            "quantity": order_request.quantity,
            "success": execution_result.success,
            "status": execution_result.status.value,
            "message": execution_result.message,
            "executed_price": execution_result.executed_price,
            "executed_quantity": execution_result.executed_quantity,
            "execution_time": execution_result.execution_time.isoformat() if execution_result.execution_time else None
        }
        
        if execution_result.success:
            self.logger.info(f"Order executed successfully: {json.dumps(log_data)}")
        else:
            self.logger.error(f"Order execution failed: {json.dumps(log_data)}")
    
    def set_execution_enabled(self, enabled: bool, reason: str = ""):
        """Enable/disable order execution"""
        self.execution_enabled = enabled
        if enabled:
            self.logger.info("Order execution enabled")
        else:
            self.logger.warning(f"Order execution disabled: {reason}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            "execution_enabled": self.execution_enabled,
            "api_health_status": self.api_health_status,
            "api_failure_count": self.api_failure_count,
            "pending_orders": len(self.pending_orders),
            "last_api_check": self.last_api_check.isoformat(),
            "retry_attempts": self.retry_attempts,
            "order_timeout": self.order_timeout
        }
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            # Simulate order cancellation
            # In production, this would call broker API
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.warning(f"Order not found for cancellation: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_open_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get open positions for user"""
        try:
            with next(get_database_session()) as db:
                trades = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.status == TradeStatus.FILLED,
                    Trade.exit_time.is_(None)
                ).all()
                
                positions = []
                for trade in trades:
                    position = {
                        "trade_id": trade.id,
                        "symbol": trade.symbol,
                        "action": trade.action,
                        "quantity": trade.quantity,
                        "entry_price": trade.entry_price,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "entry_time": trade.entry_time.isoformat(),
                        "current_pnl": self._calculate_current_pnl(trade)
                    }
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            self.logger.error(f"Failed to get open positions for user {user_id}: {e}")
            return []
    
    def _calculate_current_pnl(self, trade: Trade) -> float:
        """Calculate current P&L for open position (simulated)"""
        # In production, this would get current market price
        # For now, simulate small P&L
        import random
        return random.uniform(-50, 100)

# Global executor instance
safe_executor = SafeOrderExecutor()
