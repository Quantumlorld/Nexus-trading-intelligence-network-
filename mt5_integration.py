"""
MetaTrader 5 Integration for Nexus Trading System
Real broker connection and trading functionality
"""

import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class MT5Connector:
    """Real MetaTrader 5 broker connector"""
    
    def __init__(self, terminal_path: str = None):
        self.terminal_path = terminal_path or r"C:\Program Files\MetaTrader 5\terminal64.exe"  # Working path
        self.connected = False
        self.account_info = None
        self.symbols = []
        self.connection_status: str = "DISCONNECTED"
        self.last_error: Optional[Tuple[int, str]] = None

        self.max_attempts = int(os.environ.get("MT5_MAX_ATTEMPTS", "5"))
        self.retry_delay_sec = float(os.environ.get("MT5_RETRY_DELAY_SEC", "2.0"))
        
    def _terminal_exists(self) -> bool:
        return bool(self.terminal_path) and os.path.exists(self.terminal_path)

    def _ensure_terminal_running(self) -> bool:
        """Best-effort: launch terminal if it's not running. Always safe to call."""
        try:
            if not self._terminal_exists():
                return False

            # We don't strictly check process list here (permissions vary). We just attempt a launch.
            subprocess.Popen([self.terminal_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.error(f"MT5 terminal launch error: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize MT5 connection with explicit path"""
        for attempt in range(self.max_attempts):
            try:
                logger.info(f"MT5 initialize attempt {attempt + 1}/{self.max_attempts}")
                
                # Use explicit path parameter
                result = mt5.initialize(path=self.terminal_path)
                
                if result:
                    self.connected = True
                    self.connection_status = "CONNECTED"
                    logger.info("✅ MT5 initialized successfully with explicit path")
                    return True
                else:
                    error = mt5.last_error()
                    self.last_error = error
                    logger.warning(f"MT5 initialize failed (attempt {attempt + 1}/{self.max_attempts}): {error}")
                    
                    if error[0] == -10005:  # IPC timeout
                        self.connection_status = "IPC_TIMEOUT"
                    elif error[0] == -10003:  # IPC initialize failed  
                        self.connection_status = "IPC_ERROR"
                    else:
                        self.connection_status = "INITIALIZE_FAILED"
                    
                    if attempt < self.max_attempts - 1:
                        time.sleep(self.retry_delay_sec)
                        
            except Exception as e:
                logger.error(f"MT5 initialize exception (attempt {attempt + 1}): {e}")
                self.connection_status = "EXCEPTION"
                self.last_error = (0, str(e))
                
                if attempt < self.max_attempts - 1:
                    time.sleep(self.retry_delay_sec)
        
        logger.error("❌ All MT5 initialize attempts failed")
        return False
    
    def login(self, account: int, password: str, server: str) -> bool:
        """Login to MT5 account with retry + status codes."""
        if not self.connected:
            if not self.initialize():
                return False

        for attempt in range(1, self.max_attempts + 1):
            try:
                authorized = mt5.login(account, password=password, server=server)
                self.last_error = mt5.last_error()

                if authorized:
                    self.account_info = mt5.account_info()
                    self.connected = True
                    self.connection_status = "CONNECTED"
                    logger.info(f"Successfully logged into MT5 account {account}")
                    return True

                code = (self.last_error or (None, ""))[0]
                if code in (-10001, -10004, -10005):
                    self.connection_status = "IPC_ERROR"
                else:
                    self.connection_status = "LOGIN_FAILED"

                logger.error(
                    f"MT5 login failed (attempt {attempt}/{self.max_attempts}): {self.last_error}"
                )

                if attempt < self.max_attempts:
                    time.sleep(self.retry_delay_sec)
            except Exception as e:
                self.connection_status = "LOGIN_FAILED"
                logger.error(f"MT5 login error (attempt {attempt}/{self.max_attempts}): {e}")
                if attempt < self.max_attempts:
                    time.sleep(self.retry_delay_sec)

        self.connected = False
        return False

    def status(self) -> Dict[str, Any]:
        return {
            "connection_status": self.connection_status,
            "connected": bool(self.connected and self.connection_status == "CONNECTED"),
            "terminal_path": self.terminal_path,
            "last_error": self.last_error,
        }

    def mt5_health(self) -> Dict[str, Any]:
        """Health snapshot for API/status screens."""
        try:
            if not self.connected:
                return {
                    "connected": False,
                    "connection_status": self.connection_status,
                    "account": None,
                    "balance": None,
                }

            ai = mt5.account_info()
            if not ai:
                return {
                    "connected": False,
                    "connection_status": "LOGIN_FAILED",
                    "account": None,
                    "balance": None,
                }

            return {
                "connected": True,
                "connection_status": self.connection_status,
                "account": ai.login,
                "balance": ai.balance,
            }
        except Exception as e:
            logger.error(f"mt5_health error: {e}")
            return {
                "connected": False,
                "connection_status": "IPC_ERROR",
                "account": None,
                "balance": None,
            }
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        try:
            if not self.connected:
                return None
            
            account_info = mt5.account_info()
            if account_info:
                return {
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "profit": account_info.profit,
                    "leverage": account_info.leverage,
                    "account": account_info.login,
                    "server": account_info.server
                }
            return None
            
        except Exception as e:
            logger.error(f"Get account info error: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        try:
            if not self.connected:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return {
                    "symbol": symbol_info.name,
                    "bid": symbol_info.bid,
                    "ask": symbol_info.ask,
                    "point": symbol_info.point,
                    "trade_contract_size": symbol_info.trade_contract_size,
                    "volume_min": symbol_info.volume_min,
                    "volume_max": symbol_info.volume_max,
                    "spread": symbol_info.spread
                }
            return None
            
        except Exception as e:
            logger.error(f"Get symbol info error: {e}")
            return None
    
    def place_buy_order(self, symbol: str, volume: float, price: float = 0.0) -> Optional[int]:
        """Place a buy order"""
        try:
            if not self.connected:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            if price == 0.0:
                price = symbol_info.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Buy order placed successfully: {result.order}")
                return result.order
            else:
                logger.error(f"Buy order failed: {result.comment}")
                return None
                
        except Exception as e:
            logger.error(f"Place buy order error: {e}")
            return None
    
    def place_sell_order(self, symbol: str, volume: float, price: float = 0.0) -> Optional[int]:
        """Place a sell order"""
        try:
            if not self.connected:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            if price == 0.0:
                price = symbol_info.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Sell order placed successfully: {result.order}")
                return result.order
            else:
                logger.error(f"Sell order failed: {result.comment}")
                return None
                
        except Exception as e:
            logger.error(f"Place sell order error: {e}")
            return None
    
    def get_candles(self, symbol: str, timeframe: int, count: int = 100) -> Optional[list]:
        """Get candle data from MT5"""
        try:
            if not self.connected:
                return None
            
            candles = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if candles is not None:
                return [{
                    "time": datetime.fromtimestamp(candle['time']).isoformat(),
                    "open": candle['open'],
                    "high": candle['high'],
                    "low": candle['low'],
                    "close": candle['close'],
                    "volume": candle['tick_volume']
                } for candle in candles]
            return None
            
        except Exception as e:
            logger.error(f"Get candles error: {e}")
            return None
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("MT5 connection shutdown")
        except Exception as e:
            logger.error(f"MT5 shutdown error: {e}")

# Global MT5 connector instance
mt5_connector = MT5Connector()

def get_mt5_connector() -> MT5Connector:
    """Get the global MT5 connector instance"""
    return mt5_connector
