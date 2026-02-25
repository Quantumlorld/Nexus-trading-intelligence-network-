"""
Nexus Trading System - Order Executor
Advanced order execution with risk management and session filtering
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .mt5_bridge import MT5Bridge
from .session_filter import SessionFilter
from core.trade_manager import TradeManager, Position
from core.risk_engine import RiskEngine
from core.position_sizer import PositionSizer
from core.logger import get_logger
from strategy.base_strategy import TradingSignal, SignalType


@dataclass
class OrderRequest:
    """Order request for execution"""
    symbol: str
    signal_type: SignalType
    volume: float
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    strategy: str = ""
    confidence: float = 0.0
    magic: int = 123456
    comment: str = "NEXUS"
    session_filter_enabled: bool = True
    risk_check_enabled: bool = True


@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[int] = None
    position_id: Optional[int] = None
    fill_price: Optional[float] = None
    filled_volume: Optional[float] = None
    error: Optional[str] = None
    execution_time: Optional[datetime] = None
    slippage: Optional[float] = None
    commission: Optional[float] = None
    retry_count: int = 0


class OrderExecutor:
    """
    Advanced order execution system with comprehensive risk management,
    session filtering, and execution optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Initialize components
        self.mt5_bridge = MT5Bridge(config)
        self.session_filter = SessionFilter(config)
        
        # Execution state
        self.is_running = False
        self.execution_queue: List[OrderRequest] = []
        self.active_orders: Dict[int, OrderRequest] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Thread safety
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Execution settings
        self.max_execution_time = config.get('order_management', {}).get('max_execution_time_ms', 5000) / 1000.0
        self.slippage_tolerance = config.get('order_management', {}).get('slippage_tolerance', 5.0)
        self.retry_on_timeout = config.get('error_handling', {}).get('retry_on_timeout', True)
        self.max_retries = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_ms', 1000) / 1000.0
        
        # Risk management
        self.enforce_risk_limits = True
        self.max_position_size = config.get('position_management', {}).get('max_position_size', 1.0)
        self.one_active_per_asset = config.get('position_management', {}).get('one_active_per_asset', True)
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_slippage': 0.0,
            'avg_execution_time': 0.0,
            'total_commission': 0.0
        }
        
        self.logger.info("Order Executor initialized")
    
    async def start(self):
        """Start the order executor"""
        
        if self.is_running:
            self.logger.warning("Order executor is already running")
            return
        
        # Connect to MT5
        if not self.mt5_bridge.connect():
            self.logger.error("Failed to connect to MT5")
            return
        
        self.is_running = True
        
        # Start execution loop
        asyncio.create_task(self._execution_loop())
        
        self.logger.info("Order executor started")
    
    async def stop(self):
        """Stop the order executor"""
        
        self.is_running = False
        
        # Cancel pending orders
        await self._cancel_all_pending_orders()
        
        # Disconnect from MT5
        self.mt5_bridge.disconnect()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Order executor stopped")
    
    async def place_order(self, request: OrderRequest) -> ExecutionResult:
        """
        Place an order with comprehensive validation and risk management
        
        Args:
            request: Order request with all parameters
            
        Returns:
            ExecutionResult with execution details
        """
        
        start_time = datetime.now()
        
        try:
            # Validate order request
            validation_result = self._validate_order_request(request)
            if not validation_result['valid']:
                return ExecutionResult(
                    success=False,
                    error=validation_result['error'],
                    execution_time=start_time
                )
            
            # Check session filters
            if request.session_filter_enabled:
                session_check = self.session_filter.is_trading_allowed(request.symbol, datetime.now())
                if not session_check['allowed']:
                    return ExecutionResult(
                        success=False,
                        error=f"Session filter: {session_check['reason']}",
                        execution_time=start_time
                    )
            
            # Risk assessment
            if request.risk_check_enabled:
                risk_check = self._assess_order_risk(request)
                if not risk_check['allowed']:
                    return ExecutionResult(
                        success=False,
                        error=f"Risk check: {risk_check['reason']}",
                        execution_time=start_time
                    )
            
            # Execute order with retry logic
            execution_result = await self._execute_order_with_retry(request, start_time)
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            # Log execution
            if execution_result.success:
                self.logger.info(f"Order executed: {request.symbol} {request.signal_type.value} "
                              f"{request.volume} @ {execution_result.fill_price}")
            else:
                self.logger.error(f"Order failed: {request.symbol} {request.signal_type.value} "
                               f"- {execution_result.error}")
            
            return execution_result
        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=start_time
            )
    
    async def modify_position(self, symbol: str, position_id: int,
                            sl_price: Optional[float] = None,
                            tp_price: Optional[float] = None) -> ExecutionResult:
        """Modify position SL/TP"""
        
        start_time = datetime.now()
        
        try:
            # Validate modification request
            if sl_price is None and tp_price is None:
                return ExecutionResult(
                    success=False,
                    error="No SL/TP values provided for modification",
                    execution_time=start_time
                )
            
            # Execute modification
            result = self.mt5_bridge.modify_position(position_id, sl_price, tp_price)
            
            if result['success']:
                execution_result = ExecutionResult(
                    success=True,
                    position_id=position_id,
                    execution_time=datetime.now(),
                    slippage=0.0  # No slippage on modifications
                )
                
                self.logger.info(f"Position modified: {position_id} SL={sl_price}, TP={tp_price}")
            else:
                execution_result = ExecutionResult(
                    success=False,
                    error=result['error'],
                    execution_time=start_time
                )
            
            return execution_result
        
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=start_time
            )
    
    async def close_position(self, symbol: str, position_id: int,
                           volume: Optional[float] = None,
                           reason: str = "Manual close") -> ExecutionResult:
        """Close a position"""
        
        start_time = datetime.now()
        
        try:
            # Execute close
            result = self.mt5_bridge.close_position(position_id, volume)
            
            if result['success']:
                execution_result = ExecutionResult(
                    success=True,
                    order_id=result.get('order_id'),
                    position_id=position_id,
                    fill_price=result.get('price'),
                    filled_volume=result.get('volume'),
                    execution_time=datetime.now(),
                    commission=result.get('commission', 0.0)
                )
                
                self.logger.info(f"Position closed: {position_id} - {reason}")
            else:
                execution_result = ExecutionResult(
                    success=False,
                    error=result['error'],
                    execution_time=start_time
                )
            
            return execution_result
        
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=start_time
            )
    
    async def _execution_loop(self):
        """Main execution loop for processing queued orders"""
        
        while self.is_running:
            try:
                if self.execution_queue:
                    # Get next order from queue
                    with self._lock:
                        if self.execution_queue:
                            request = self.execution_queue.pop(0)
                
                    # Execute order
                    await self.place_order(request)
                
                # Sleep briefly
                await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_order_with_retry(self, request: OrderRequest, start_time: datetime) -> ExecutionResult:
        """Execute order with retry logic"""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute order
                if request.signal_type == SignalType.BUY:
                    result = self.mt5_bridge.place_market_order(
                        symbol=request.symbol,
                        order_type='buy',
                        volume=request.volume,
                        sl=request.sl_price,
                        tp=request.tp_price,
                        magic=request.magic,
                        comment=request.comment
                    )
                else:  # SELL
                    result = self.mt5_bridge.place_market_order(
                        symbol=request.symbol,
                        order_type='sell',
                        volume=request.volume,
                        sl=request.sl_price,
                        tp=request.tp_price,
                        magic=request.magic,
                        comment=request.comment
                    )
                
                if result['success']:
                    # Calculate slippage
                    slippage = 0.0
                    if request.entry_price and result['price']:
                        if request.signal_type == SignalType.BUY:
                            slippage = (result['price'] - request.entry_price) / request.entry_price * 100
                        else:
                            slippage = (request.entry_price - result['price']) / request.entry_price * 100
                    
                    return ExecutionResult(
                        success=True,
                        order_id=result.get('order_id'),
                        position_id=result.get('position_id'),
                        fill_price=result.get('price'),
                        filled_volume=result.get('volume'),
                        execution_time=datetime.now(),
                        slippage=slippage,
                        commission=result.get('commission', 0.0),
                        retry_count=attempt
                    )
                
                else:
                    last_error = result['error']
                    
                    # Check if we should retry
                    if not self._should_retry(last_error, attempt):
                        break
                    
                    # Wait before retry
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
            
            except Exception as e:
                last_error = str(e)
                
                if not self._should_retry(last_error, attempt):
                    break
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        return ExecutionResult(
            success=False,
            error=last_error or "Max retries exceeded",
            execution_time=start_time,
            retry_count=self.max_retries
        )
    
    def _validate_order_request(self, request: OrderRequest) -> Dict[str, Any]:
        """Validate order request"""
        
        # Check required fields
        if not request.symbol or not request.signal_type or not request.volume:
            return {
                'valid': False,
                'error': 'Missing required fields (symbol, signal_type, volume)'
            }
        
        # Validate volume
        if request.volume <= 0:
            return {
                'valid': False,
                'error': 'Volume must be positive'
            }
        
        # Check maximum position size
        if request.volume > self.max_position_size:
            return {
                'valid': False,
                'error': f'Volume {request.volume} exceeds maximum {self.max_position_size}'
            }
        
        # Check if we already have an active position for this asset
        if self.one_active_per_asset:
            current_positions = self.mt5_bridge.get_positions(request.symbol)
            if current_positions:
                return {
                    'valid': False,
                    'error': f'Already have active position for {request.symbol}'
                }
        
        # Validate SL/TP
        if request.sl_price and request.tp_price:
            if request.signal_type == SignalType.BUY:
                if request.sl_price >= request.tp_price:
                    return {
                        'valid': False,
                        'error': 'For BUY orders, SL must be less than TP'
                    }
            else:  # SELL
                if request.sl_price <= request.tp_price:
                    return {
                        'valid': False,
                        'error': 'For SELL orders, SL must be greater than TP'
                    }
        
        return {'valid': True}
    
    def _assess_order_risk(self, request: OrderRequest) -> Dict[str, Any]:
        """Assess order risk"""
        
        # Get current account info
        account_info = self.mt5_bridge.get_account_info()
        
        if not account_info:
            return {
                'allowed': False,
                'reason': 'Unable to get account information'
            }
        
        # Check margin requirements
        symbol_info = self.mt5_bridge.get_symbol_info(request.symbol)
        
        if not symbol_info:
            return {
                'allowed': False,
                'reason': f'Unable to get symbol information for {request.symbol}'
            }
        
        # Calculate required margin (simplified)
        required_margin = request.volume * 1000 / account_info.leverage  # Rough estimate
        
        if required_margin > account_info.free_margin:
            return {
                'allowed': False,
                'reason': f'Insufficient margin. Required: {required_margin:.2f}, Available: {account_info.free_margin:.2f}'
            }
        
        # Check position size relative to account
        position_value = request.volume * 100000  # Rough estimate for standard lots
        account_risk = position_value / account_info.balance * 100
        
        if account_risk > 2.0:  # Max 2% risk per position
            return {
                'allowed': False,
                'reason': f'Position risk {account_risk:.2f}% exceeds maximum 2%'
            }
        
        return {'allowed': True}
    
    def _should_retry(self, error: str, attempt: int) -> bool:
        """Determine if we should retry based on error"""
        
        # Don't retry on certain errors
        no_retry_errors = [
            'Invalid symbol',
            'Insufficient margin',
            'Volume outside range',
            'Market closed',
            'Invalid SL/TP'
        ]
        
        for no_retry_error in no_retry_errors:
            if no_retry_error in error:
                return False
        
        # Retry on timeout and connection errors
        retry_errors = [
            'timeout',
            'connection',
            'server',
            'network'
        ]
        
        for retry_error in retry_errors:
            if retry_error.lower() in error.lower():
                return True
        
        # Default: retry if we haven't exhausted attempts
        return attempt < self.max_retries - 1
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        
        self.execution_stats['total_orders'] += 1
        
        if result.success:
            self.execution_stats['successful_orders'] += 1
            
            # Update slippage
            if result.slippage is not None:
                current_avg = self.execution_stats['avg_slippage']
                successful_count = self.execution_stats['successful_orders']
                self.execution_stats['avg_slippage'] = (current_avg * (successful_count - 1) + result.slippage) / successful_count
            
            # Update commission
            if result.commission is not None:
                self.execution_stats['total_commission'] += result.commission
        else:
            self.execution_stats['failed_orders'] += 1
        
        # Update execution time
        if result.execution_time:
            execution_time = (datetime.now() - result.execution_time).total_seconds()
            current_avg = self.execution_stats['avg_execution_time']
            total_orders = self.execution_stats['total_orders']
            self.execution_stats['avg_execution_time'] = (current_avg * (total_orders - 1) + execution_time) / total_orders
    
    async def _cancel_all_pending_orders(self):
        """Cancel all pending orders"""
        
        try:
            # Get all pending orders
            pending_orders = self.mt5_bridge.get_orders()
            
            for order in pending_orders:
                # Cancel order
                result = self.mt5_bridge.cancel_order(order['order_id'])
                
                if result['success']:
                    self.logger.info(f"Cancelled pending order: {order['order_id']}")
                else:
                    self.logger.error(f"Failed to cancel order {order['order_id']}: {result['error']}")
        
        except Exception as e:
            self.logger.error(f"Error cancelling pending orders: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        total_orders = self.execution_stats['total_orders']
        
        if total_orders > 0:
            success_rate = (self.execution_stats['successful_orders'] / total_orders) * 100
        else:
            success_rate = 0.0
        
        return {
            'total_orders': total_orders,
            'successful_orders': self.execution_stats['successful_orders'],
            'failed_orders': self.execution_stats['failed_orders'],
            'success_rate': success_rate,
            'avg_slippage': self.execution_stats['avg_slippage'],
            'avg_execution_time': self.execution_stats['avg_execution_time'],
            'total_commission': self.execution_stats['total_commission'],
            'queue_size': len(self.execution_queue),
            'active_orders': len(self.active_orders)
        }
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        
        return self.mt5_bridge.get_positions()
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        
        return self.mt5_bridge.get_orders()
    
    def export_execution_history(self, filepath: str):
        """Export execution history to CSV"""
        
        if not self.execution_history:
            self.logger.warning("No execution history to export")
            return
        
        try:
            import pandas as pd
            
            # Convert to DataFrame
            data = []
            for result in self.execution_history:
                data.append({
                    'timestamp': result.execution_time,
                    'success': result.success,
                    'order_id': result.order_id,
                    'position_id': result.position_id,
                    'fill_price': result.fill_price,
                    'filled_volume': result.filled_volume,
                    'error': result.error,
                    'slippage': result.slippage,
                    'commission': result.commission,
                    'retry_count': result.retry_count
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Execution history exported to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error exporting execution history: {e}")
    
    def reset_stats(self):
        """Reset execution statistics"""
        
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_slippage': 0.0,
            'avg_execution_time': 0.0,
            'total_commission': 0.0
        }
        
        self.logger.info("Execution statistics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current executor status"""
        
        return {
            'is_running': self.is_running,
            'mt5_connected': self.mt5_bridge.is_connected,
            'execution_stats': self.get_execution_stats(),
            'queue_size': len(self.execution_queue),
            'active_orders': len(self.active_orders),
            'session_filter_enabled': self.session_filter.is_enabled,
            'risk_check_enabled': self.enforce_risk_limits,
            'max_position_size': self.max_position_size,
            'one_active_per_asset': self.one_active_per_asset
        }
