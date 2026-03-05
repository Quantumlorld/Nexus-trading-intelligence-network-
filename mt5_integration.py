"""
MetaTrader 5 Integration for Nexus Trading System
Real broker connection and trading functionality
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MT5Connector:
    """Real MetaTrader 5 broker connector"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.symbols = []
        
    def initialize(self) -> bool:
        """Initialize MT5 terminal"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            self.connected = True
            logger.info("MT5 terminal initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def login(self, account: int, password: str, server: str) -> bool:
        """Login to MT5 account"""
        try:
            if not self.connected:
                if not self.initialize():
                    return False
            
            authorized = mt5.login(account, password=password, server=server)
            if authorized:
                self.account_info = mt5.account_info()
                logger.info(f"Successfully logged into MT5 account {account}")
                return True
            else:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"MT5 login error: {e}")
            return False
    
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
