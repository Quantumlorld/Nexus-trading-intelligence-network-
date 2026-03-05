"""
Nexus Trading System - Real Broker Adapter
Connects to local matching engine with real connection health checks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from core.local_matching_engine import local_matching_engine
from core.system_control import system_control_manager
from core.operational_metrics import operational_metrics

logger = logging.getLogger(__name__)

class RealBrokerAdapter:
    """Real broker adapter with connection health monitoring"""
    
    def __init__(self):
        self.engine = local_matching_engine
        self.is_connected = True
        self.last_ping_time = datetime.utcnow()
        self.consecutive_failures = 0
        self.max_failures = 3
        self.ping_interval = 10  # seconds
        self.connection_timeout = 5.0  # seconds
        
        # Initialize market with liquidity
        self.engine.initialize_market()
        
        # Start health monitoring
        self.monitoring_task = None
        self.is_monitoring = False
    
    async def start_monitoring(self):
        """Start connection health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Broker adapter monitoring started")
    
    async def stop_monitoring(self):
        """Stop connection health monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Broker adapter monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                await self._ping_connection()
                await asyncio.sleep(self.ping_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broker monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _ping_connection(self) -> bool:
        """Ping broker connection"""
        try:
            start_time = datetime.utcnow()
            
            # Attempt to get account info as ping
            account_info = self.engine.get_account_info()
            
            if account_info:
                latency = (datetime.utcnow() - start_time).total_seconds()
                operational_metrics.record_broker_latency(latency)
                
                self.consecutive_failures = 0
                self.is_connected = True
                self.last_ping_time = datetime.utcnow()
                
                if operational_metrics.gauges['broker_connected'] == 0:
                    operational_metrics.record_broker_recovery()
                
                logger.debug(f"Broker ping successful: {latency:.3f}s")
                return True
            else:
                raise Exception("No account info returned")
                
        except Exception as e:
            self.consecutive_failures += 1
            self.is_connected = False
            operational_metrics.record_broker_failure()
            
            logger.error(f"Broker ping failed (attempt {self.consecutive_failures}): {e}")
            
            # Trigger system control if threshold exceeded
            if self.consecutive_failures >= self.max_failures:
                await system_control_manager.disable_trading(
                    f"Broker connection failed after {self.consecutive_failures} attempts: {str(e)}",
                    "broker_monitor"
                )
            
            return False
    
    async def check_connection(self) -> bool:
        """Check broker connection status"""
        return await self._ping_connection()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.is_connected:
                raise Exception("Broker not connected")
            
            return self.engine.get_account_info()
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        try:
            if not self.is_connected:
                raise Exception("Broker not connected")
            
            positions = self.engine.get_positions(symbol)
            
            # Format positions to match expected structure
            formatted_positions = []
            for pos in positions:
                formatted_positions.append({
                    'position_id': pos.get('trade_id', ''),
                    'symbol': pos.get('symbol', ''),
                    'type': pos.get('side', 'buy'),
                    'volume': abs(pos.get('quantity', 0)),
                    'open_price': pos.get('price', 0),
                    'current_price': self.engine.get_market_price(pos.get('symbol', '')),
                    'profit': 0,  # Would calculate based on current price
                    'open_time': pos.get('timestamp', datetime.utcnow()),
                    'commission': pos.get('commission', 0)
                })
            
            return formatted_positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Place an order"""
        try:
            if not self.is_connected:
                raise Exception("Broker not connected")
            
            # Add user_id from kwargs or default
            if 'user_id' not in kwargs:
                kwargs['user_id'] = 1
            
            # Place order through matching engine
            execution = self.engine.place_order(**kwargs)
            
            if execution.success:
                return {
                    'success': True,
                    'order_id': execution.order_id,
                    'position_id': execution.position_id,
                    'price': execution.execution_price,
                    'volume': execution.filled_quantity,
                    'commission': execution.commission,
                    'filled_volume': execution.filled_quantity,
                    'slippage': execution.slippage,
                    'partially_filled': execution.partially_filled
                }
            else:
                return {
                    'success': False,
                    'error': execution.error
                }
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            if not self.is_connected:
                raise Exception("Broker not connected")
            
            return self.engine.get_market_price(symbol)
            
        except Exception as e:
            logger.error(f"Failed to get market price: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            'connected': self.is_connected,
            'last_ping': self.last_ping_time.isoformat(),
            'consecutive_failures': self.consecutive_failures,
            'monitoring_active': self.is_monitoring
        }

# Global broker adapter instance
real_broker_adapter = RealBrokerAdapter()
