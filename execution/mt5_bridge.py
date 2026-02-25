"""
Nexus Trading System - MT5 Bridge
MetaTrader 5 integration for real-time trading execution
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from core.logger import get_logger


@dataclass
class MT5AccountInfo:
    """MT5 account information"""
    login: int
    server: str
    name: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: int
    trade_mode: str
    limit_orders: int
    positions: int


@dataclass
class MT5SymbolInfo:
    """MT5 symbol information"""
    symbol: str
    description: str
    digits: int
    point: float
    tick_value: float
    tick_size: float
    contract_size: float
    min_lot: float
    max_lot: float
    lot_step: float
    spread: float
    swap_long: float
    swap_short: float
    starting_time: datetime
    expiration_time: Optional[datetime]
    trade_mode: str


@dataclass
class OrderRequest:
    """Order request for MT5"""
    symbol: str
    order_type: int  # mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    volume: float
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    deviation: int = 10
    magic: int = 123456
    comment: str = "NEXUS"
    type_time: int = None
    type_filling: int = None


class MT5Bridge:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Connection state
        self.is_connected = False
        self.connection_info: Optional[MT5AccountInfo] = None
        self.symbol_info_cache: Dict[str, MT5SymbolInfo] = {}
        
        # Order tracking
        self.pending_orders: Dict[int, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Retry configuration
        self.max_retries = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_ms', 1000) / 1000.0
        
        # Session management
        self.session_active = False
        self.last_heartbeat = datetime.now()
        
        self.logger.info("MT5 Bridge initialized")
    
    def connect(self, login: int = None, password: str = None, server: str = None) -> bool:
        """
        Connect to MetaTrader 5 terminal
        
        Args:
            login: Account login number
            password: Account password
            server: Server name
            
        Returns:
            True if connection successful
        """
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("MT5 initialize() failed")
                return False
            
            # Login to account
            account_config = self.config.get('broker', {})
            
            login = login or account_config.get('login', 0)
            password = password or account_config.get('password', "")
            server = server or account_config.get('server', "")
            
            if login == 0:
                self.logger.warning("No login credentials provided, using existing connection")
                if not mt5.terminal_info():
                    self.logger.error("No active MT5 terminal found")
                    return False
            else:
                authorized = mt5.login(login, password, server)
                if not authorized:
                    self.logger.error(f"Failed to login: {mt5.last_error()}")
                    return False
            
            # Get account information
            account_info = mt5.account_info()
            self.connection_info = MT5AccountInfo(
                login=account_info.login,
                server=account_info.server,
                name=account_info.name,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                leverage=account_info.leverage,
                trade_mode=self._get_trade_mode_name(account_info.trade_mode),
                limit_orders=account_info.limit_orders,
                positions=account_info.positions
            )
            
            self.is_connected = True
            self.session_active = True
            self.last_heartbeat = datetime.now()
            
            self.logger.info(f"Connected to MT5: {self.connection_info.name} ({self.connection_info.login})")
            
            # Start heartbeat monitoring
            self._start_heartbeat_monitor()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        
        try:
            self.session_active = False
            
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                self.connection_info = None
                
                self.logger.info("Disconnected from MT5")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from MT5: {e}")
    
    def get_account_info(self) -> Optional[MT5AccountInfo]:
        """Get current account information"""
        
        if not self.is_connected:
            return None
        
        try:
            account_info = mt5.account_info()
            
            return MT5AccountInfo(
                login=account_info.login,
                server=account_info.server,
                name=account_info.name,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                leverage=account_info.leverage,
                trade_mode=self._get_trade_mode_name(account_info.trade_mode),
                limit_orders=account_info.limit_orders,
                positions=account_info.positions
            )
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[MT5SymbolInfo]:
        """Get symbol information"""
        
        if not self.is_connected:
            return None
        
        # Check cache first
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None
            
            mt5_symbol = MT5SymbolInfo(
                symbol=symbol_info.name,
                description=symbol_info.description,
                digits=symbol_info.digits,
                point=symbol_info.point,
                tick_value=symbol_info.trade_tick_value,
                tick_size=symbol_info.trade_tick_size,
                contract_size=symbol_info.trade_contract_size,
                min_lot=symbol_info.volume_min,
                max_lot=symbol_info.volume_max,
                lot_step=symbol_info.volume_step,
                spread=symbol_info.spread,
                swap_long=symbol_info.swap_long,
                swap_short=symbol_info.swap_short,
                starting_time=datetime.fromtimestamp(symbol_info.time_start),
                expiration_time=datetime.fromtimestamp(symbol_info.time_expiration) if symbol_info.time_expiration > 0 else None,
                trade_mode=self._get_trade_mode_name(symbol_info.trade_mode)
            )
            
            # Cache the info
            self.symbol_info_cache[symbol] = mt5_symbol
            
            return mt5_symbol
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, order_type: str, volume: float,
                          sl: Optional[float] = None, tp: Optional[float] = None,
                          deviation: int = 10, magic: int = 123456,
                          comment: str = "NEXUS") -> Dict[str, Any]:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Position size
            sl: Stop loss price
            tp: Take profit price
            deviation: Maximum price deviation
            magic: Magic number
            comment: Order comment
            
        Returns:
            Order result dictionary
        """
        
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        # Validate parameters
        if order_type.lower() not in ['buy', 'sell']:
            return {'success': False, 'error': 'Invalid order type'}
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return {'success': False, 'error': f'Invalid symbol: {symbol}'}
        
        # Validate volume
        if volume < symbol_info.min_lot or volume > symbol_info.max_lot:
            return {'success': False, 'error': f'Volume {volume} outside range [{symbol_info.min_lot}, {symbol_info.max_lot}]'}
        
        # Normalize volume
        normalized_volume = self._normalize_volume(volume, symbol_info)
        
        # Create order request
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type.lower() == 'buy' else mt5.ORDER_TYPE_SELL
        
        request = OrderRequest(
            symbol=symbol,
            order_type=mt5_order_type,
            volume=normalized_volume,
            sl=sl,
            tp=tp,
            deviation=deviation,
            magic=magic,
            comment=comment,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=mt5.ORDER_FILLING_IOC
        )
        
        # Send order with retry logic
        for attempt in range(self.max_retries):
            try:
                result = mt5.order_send(request._asdict())
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Order successful
                    order_result = {
                        'success': True,
                        'order_id': result.order,
                        'position_id': result.position,
                        'price': result.price,
                        'volume': result.volume,
                        'symbol': result.symbol,
                        'type': order_type,
                        'sl': result.sl,
                        'tp': result.tp,
                        'comment': result.comment,
                        'magic': result.magic,
                        'timestamp': datetime.now()
                    }
                    
                    # Track order
                    self._track_order(order_result)
                    
                    self.logger.info(f"Order placed: {symbol} {order_type} {normalized_volume} @ {result.price}")
                    
                    return order_result
                
                else:
                    # Order failed
                    error_msg = f"Order failed: {result.retcode} - {result.comment}"
                    self.logger.error(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    
                    return {'success': False, 'error': error_msg, 'retcode': result.retcode}
            
            except Exception as e:
                self.logger.error(f"Error placing order (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def modify_position(self, position_id: int, sl: Optional[float] = None,
                       tp: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify position SL/TP
        
        Args:
            position_id: Position ID
            sl: New stop loss price
            tp: New take profit price
            
        Returns:
            Modification result
        """
        
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        if sl is None and tp is None:
            return {'success': False, 'error': 'No SL/TP values provided'}
        
        try:
            # Get current position
            positions = mt5.positions_get(position_id=position_id)
            
            if not positions:
                return {'success': False, 'error': f'Position {position_id} not found'}
            
            position = positions[0]
            
            # Create modification request
            request = {
                'position': position_id,
                'sl': sl if sl is not None else position.sl,
                'tp': tp if tp is not None else position.tp,
                'magic': position.magic,
                'comment': position.comment,
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            
            # Send modification
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                modification_result = {
                    'success': True,
                    'position_id': position_id,
                    'old_sl': position.sl,
                    'new_sl': result.sl,
                    'old_tp': position.tp,
                    'new_tp': result.tp,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"Position {position_id} modified: SL={result.sl}, TP={result.tp}")
                
                return modification_result
            
            else:
                error_msg = f"Modification failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'retcode': result.retcode}
        
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def close_position(self, position_id: int, volume: Optional[float] = None,
                      deviation: int = 10) -> Dict[str, Any]:
        """
        Close a position
        
        Args:
            position_id: Position ID
            volume: Volume to close (None for full close)
            deviation: Maximum price deviation
            
        Returns:
            Close result
        """
        
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            # Get current position
            positions = mt5.positions_get(position_id=position_id)
            
            if not positions:
                return {'success': False, 'error': f'Position {position_id} not found'}
            
            position = positions[0]
            
            # Determine close volume
            close_volume = volume if volume is not None else position.volume
            
            if close_volume > position.volume:
                close_volume = position.volume
            
            # Create close request
            request = {
                'position': position_id,
                'volume': close_volume,
                'type': mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                'price': mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                'deviation': deviation,
                'magic': position.magic,
                'comment': f"NEXUS_CLOSE_{position_id}",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                close_result = {
                    'success': True,
                    'position_id': position_id,
                    'order_id': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'symbol': position.symbol,
                    'type': 'close',
                    'profit': position.profit,
                    'swap': position.swap,
                    'commission': position.commission,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"Position {position_id} closed: {close_volume} @ {result.price}")
                
                return close_result
            
            else:
                error_msg = f"Close failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'retcode': result.retcode}
        
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        
        if not self.is_connected:
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is None:
                return []
            
            position_list = []
            
            for pos in positions:
                position_data = {
                    'position_id': pos.position,
                    'symbol': pos.symbol,
                    'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'commission': pos.commission,
                    'magic': pos.magic,
                    'comment': pos.comment,
                    'open_time': datetime.fromtimestamp(pos.time),
                    'ticket': pos.ticket
                }
                
                position_list.append(position_data)
            
            return position_list
        
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending orders"""
        
        if not self.is_connected:
            return []
        
        try:
            orders = mt5.orders_get(symbol=symbol)
            
            if orders is None:
                return []
            
            order_list = []
            
            for order in orders:
                order_data = {
                    'order_id': order.order,
                    'symbol': order.symbol,
                    'type': 'buy_stop' if order.type == mt5.ORDER_TYPE_BUY_STOP else 
                           'sell_stop' if order.type == mt5.ORDER_TYPE_SELL_STOP else
                           'buy_limit' if order.type == mt5.ORDER_TYPE_BUY_LIMIT else
                           'sell_limit',
                    'volume': order.volume_current,
                    'price_open': order.price_open,
                    'sl': order.sl,
                    'tp': order.tp,
                    'magic': order.magic,
                    'comment': order.comment,
                    'placed_time': datetime.fromtimestamp(order.time_setup),
                    'expiration': datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None
                }
                
                order_list.append(order_data)
            
            return order_list
        
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def get_account_history(self, from_date: datetime, to_date: datetime,
                           group_by_day: bool = False) -> pd.DataFrame:
        """Get account history"""
        
        if not self.is_connected:
            return pd.DataFrame()
        
        try:
            history = mt5.history_orders_get(from_date, to_date, group=mt5.DEAL_GROUP_ALL)
            
            if history is None:
                return pd.DataFrame()
            
            history_data = []
            
            for deal in history:
                deal_data = {
                    'ticket': deal.ticket,
                    'symbol': deal.symbol,
                    'type': 'buy' if deal.type == mt5.DEAL_TYPE_BUY else 'sell',
                    'volume': deal.volume,
                    'price': deal.price,
                    'commission': deal.commission,
                    'swap': deal.swap,
                    'profit': deal.profit,
                    'fee': deal.fee,
                    'time': datetime.fromtimestamp(deal.time),
                    'magic': deal.magic,
                    'comment': deal.comment,
                    'position_id': deal.position_id
                }
                
                history_data.append(deal_data)
            
            df = pd.DataFrame(history_data)
            
            if not df.empty and group_by_day:
                df['date'] = df['time'].dt.date
                df = df.groupby('date').agg({
                    'volume': 'sum',
                    'profit': 'sum',
                    'commission': 'sum',
                    'swap': 'sum',
                    'fee': 'sum'
                }).reset_index()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error getting account history: {e}")
            return pd.DataFrame()
    
    def get_symbol_ticks(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Get recent tick data for a symbol"""
        
        if not self.is_connected:
            return pd.DataFrame()
        
        try:
            ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(days=1), datetime.now(), mt5.COPY_TICKS_ALL, count)
            
            if ticks is None:
                return pd.DataFrame()
            
            tick_data = []
            
            for tick in ticks:
                tick_data.append({
                    'time': datetime.fromtimestamp(tick.time),
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'spread': tick.ask - tick.bid
                })
            
            return pd.DataFrame(tick_data)
        
        except Exception as e:
            self.logger.error(f"Error getting ticks for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth (order book)"""
        
        if not self.is_connected:
            return {}
        
        try:
            order_book = mt5.order_get(symbol)
            
            if order_book is None:
                return {}
            
            depth_data = {
                'symbol': symbol,
                'time': datetime.fromtimestamp(order_book.time),
                'bids': [],
                'asks': []
            }
            
            # Process bids
            for bid in order_book.bids:
                depth_data['bids'].append({
                    'price': bid.price,
                    'volume': bid.volume
                })
            
            # Process asks
            for ask in order_book.asks:
                depth_data['asks'].append({
                    'price': ask.price,
                    'volume': ask.volume
                })
            
            return depth_data
        
        except Exception as e:
            self.logger.error(f"Error getting market depth for {symbol}: {e}")
            return {}
    
    def _normalize_volume(self, volume: float, symbol_info: MT5SymbolInfo) -> float:
        """Normalize volume to symbol requirements"""
        
        # Round to nearest step
        steps = int(volume / symbol_info.lot_step)
        normalized = steps * symbol_info.lot_step
        
        # Ensure within limits
        normalized = max(symbol_info.min_lot, min(normalized, symbol_info.max_lot))
        
        return normalized
    
    def _get_trade_mode_name(self, trade_mode: int) -> str:
        """Convert trade mode number to name"""
        
        mode_names = {
            mt5.SYMBOL_TRADE_MODE_FULL: 'FULL',
            mt5.SYMBOL_TRADE_MODE_LONGONLY: 'LONGONLY',
            mt5.SYMBOL_TRADE_MODE_SHORTONLY: 'SHORTONLY',
            mt5.SYMBOL_TRADE_MODE_CLOSEONLY: 'CLOSEONLY'
        }
        
        return mode_names.get(trade_mode, 'UNKNOWN')
    
    def _track_order(self, order_result: Dict[str, Any]):
        """Track order for monitoring"""
        
        with self._lock:
            self.order_history.append(order_result)
            
            # Keep history manageable
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-500:]
    
    def _start_heartbeat_monitor(self):
        """Start heartbeat monitoring thread"""
        
        def monitor_heartbeat():
            while self.session_active:
                try:
                    # Check connection status
                    if not mt5.terminal_info():
                        self.logger.warning("MT5 terminal connection lost")
                        self.is_connected = False
                    
                    # Update heartbeat
                    self.last_heartbeat = datetime.now()
                    
                    # Sleep for 30 seconds
                    time.sleep(30)
                
                except Exception as e:
                    self.logger.error(f"Heartbeat monitor error: {e}")
                    time.sleep(30)
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=monitor_heartbeat, daemon=True)
        monitor_thread.start()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        
        return {
            'is_connected': self.is_connected,
            'session_active': self.session_active,
            'last_heartbeat': self.last_heartbeat,
            'account_info': self.connection_info._asdict() if self.connection_info else None,
            'cached_symbols': len(self.symbol_info_cache),
            'tracked_orders': len(self.order_history)
        }
    
    def export_order_history(self, filepath: str):
        """Export order history to CSV"""
        
        if not self.order_history:
            self.logger.warning("No order history to export")
            return
        
        try:
            df = pd.DataFrame(self.order_history)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Order history exported to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error exporting order history: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
