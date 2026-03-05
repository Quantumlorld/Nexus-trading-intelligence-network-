"""
Nexus Trading System - Universal MT5 Connector
Works with ANY MT5 broker - no restrictions!
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import time

logger = logging.getLogger(__name__)

class UniversalMT5Connector:
    """Universal MT5 connector that works with any broker"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.symbols = []
        self.positions = []
        self.connection_attempts = 0
        self.max_attempts = 5
        
    def initialize(self) -> bool:
        """Initialize MT5 terminal"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            self.connected = True
            logger.info("✅ MT5 terminal initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def connect_to_any_account(self) -> bool:
        """Connect to any available MT5 account"""
        try:
            if not self.connected:
                if not self.initialize():
                    return False
            
            # Try to get account info without login
            account_info = mt5.account_info()
            if account_info:
                self.account_info = {
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "profit": account_info.profit,
                    "leverage": account_info.leverage,
                    "account": account_info.login,
                    "server": "MT5 Terminal",
                    "connected": True
                }
                
                logger.info(f"✅ Connected to MT5 account {account_info.login}")
                logger.info(f"💰 Balance: ${account_info.balance:.2f}")
                logger.info(f"📊 Equity: ${account_info.equity:.2f}")
                
                return True
            else:
                logger.error("❌ No MT5 account found. Please login to MT5 terminal first.")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """Get all available symbols from MT5"""
        try:
            if not self.connected:
                return []
            
            symbols = mt5.symbols_get()
            if symbols:
                self.symbols = [symbol.name for symbol in symbols[:20]]  # Top 20 symbols
                logger.info(f"📈 Found {len(self.symbols)} symbols: {self.symbols[:5]}...")
                return self.symbols
            else:
                logger.warning("⚠️ No symbols found")
                return []
                
        except Exception as e:
            logger.error(f"Get symbols error: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed symbol information"""
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
                    "digits": symbol_info.digits,
                    "spread": symbol_info.ask - symbol_info.bid,
                    "volume_min": symbol_info.volume_min,
                    "volume_max": symbol_info.volume_max,
                    "volume_step": symbol_info.volume_step,
                    "swap_long": symbol_info.swap_long,
                    "swap_short": symbol_info.swap_short,
                    "trade_mode": symbol_info.trade_mode
                }
            return None
            
        except Exception as e:
            logger.error(f"Get symbol info error for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, volume: float, order_type: str, price: float = 0.0) -> Optional[int]:
        """Place a market order"""
        try:
            if not self.connected:
                logger.error("❌ MT5 not connected")
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Symbol {symbol} not found")
                return None
            
            # Determine order type
            if order_type.upper() == "BUY":
                trade_type = mt5.ORDER_TYPE_BUY
                price_to_use = symbol_info.ask if price == 0.0 else price
            else:
                trade_type = mt5.ORDER_TYPE_SELL
                price_to_use = symbol_info.bid if price == 0.0 else price
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "price": price_to_use,
                "deviation": 20,  # 20 point slippage
                "magic": 234000,  # Magic number for Nexus
                "comment": "Nexus Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Order executed: {result.order} - {order_type} {volume} {symbol} @ {price_to_use}")
                
                # Store position for tracking
                position = {
                    "order_id": result.order,
                    "symbol": symbol,
                    "type": order_type,
                    "volume": volume,
                    "price": price_to_use,
                    "time": datetime.now().isoformat(),
                    "status": "OPEN"
                }
                self.positions.append(position)
                
                return result.order
            else:
                logger.error(f"❌ Order failed: {result.comment}")
                return None
                
        except Exception as e:
            logger.error(f"Place order error: {e}")
            return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            if not self.connected:
                return []
            
            positions = mt5.positions_get()
            if positions:
                open_positions = []
                for pos in positions:
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        pos_type = "BUY"
                    else:
                        pos_type = "SELL"
                    
                    open_positions.append({
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "type": pos_type,
                        "volume": pos.volume,
                        "price_open": pos.price_open,
                        "price_current": pos.price_current,
                        "profit": pos.profit,
                        "time": datetime.fromtimestamp(pos.time).isoformat(),
                        "comment": pos.comment
                    })
                
                self.positions = open_positions
                return open_positions
            else:
                return []
                
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        try:
            if not self.connected:
                return False
            
            # Find position
            positions = mt5.positions_get()
            position_to_close = None
            
            for pos in positions:
                if pos.ticket == ticket:
                    position_to_close = pos
                    break
            
            if not position_to_close:
                logger.error(f"❌ Position {ticket} not found")
                return False
            
            # Close position
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position_to_close.symbol,
                "volume": position_to_close.volume,
                "type": mt5.ORDER_TYPE_CLOSE_BY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position_to_close.symbol).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "Closed by Nexus",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {ticket} closed - P&L: {position_to_close.profit}")
                
                # Remove from positions list
                self.positions = [p for p in self.positions if p.get("ticket") != ticket]
                return True
            else:
                logger.error(f"❌ Failed to close position {ticket}")
                return False
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary"""
        try:
            if not self.connected:
                return {"connected": False}
            
            account_info = mt5.account_info()
            positions = mt5.positions_get()
            
            if account_info:
                total_profit = sum(pos.profit for pos in positions if pos.profit > 0)
                total_loss = sum(abs(pos.profit) for pos in positions if pos.profit < 0)
                open_positions = len([pos for pos in positions if pos.profit != 0])
                
                # Get terminal info safely
                terminal_info = {}
                try:
                    term_info = mt5.terminal_info()
                    terminal_info = {
                        "version": str(getattr(term_info, 'version', 'Unknown')),
                        "build": str(getattr(term_info, 'build', 'Unknown')),
                        "company": str(getattr(term_info, 'name', 'Unknown'))
                    }
                except:
                    terminal_info = {"version": "Unknown", "build": "Unknown", "company": "Unknown"}
                
                return {
                    "connected": True,
                    "account": account_info.login,
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "profit": account_info.profit,
                    "leverage": account_info.leverage,
                    "open_positions": open_positions,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "net_profit": total_profit - total_loss,
                    "margin_level": (account_info.margin / account_info.equity * 100) if account_info.equity > 0 else 0,
                    "server": "MT5 Terminal",
                    "terminal_info": terminal_info
                }
            else:
                return {"connected": False}
                
        except Exception as e:
            logger.error(f"Account summary error: {e}")
            return {"connected": False, "error": str(e)}
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("🔌 MT5 connection shutdown")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global connector instance
universal_connector = UniversalMT5Connector()

def get_universal_connector() -> UniversalMT5Connector:
    """Get the universal MT5 connector instance"""
    return universal_connector

# Auto-connection function
async def auto_connect_to_mt5() -> bool:
    """Automatically connect to any available MT5 account"""
    logger.info("🔍 Attempting auto-connection to MT5...")
    
    for attempt in range(universal_connector.max_attempts):
        logger.info(f"🔄 Connection attempt {attempt + 1}/{universal_connector.max_attempts}")
        
        if universal_connector.connect_to_any_account():
            logger.info("✅ Successfully connected to MT5!")
            return True
        
        if attempt < universal_connector.max_attempts - 1:
            logger.info("⏳ Waiting 5 seconds before next attempt...")
            await asyncio.sleep(5)
    
    logger.error("❌ Failed to connect to MT5 after all attempts")
    return False

def sync_auto_connect_to_mt5() -> bool:
    """Synchronous version of auto-connection"""
    logger.info("🔍 Attempting sync auto-connection to MT5...")
    
    for attempt in range(universal_connector.max_attempts):
        logger.info(f"🔄 Connection attempt {attempt + 1}/{universal_connector.max_attempts}")
        
        if universal_connector.connect_to_any_account():
            logger.info("✅ Successfully connected to MT5!")
            return True
        
        if attempt < universal_connector.max_attempts - 1:
            logger.info("⏳ Waiting 5 seconds before next attempt...")
            time.sleep(5)
    
    logger.error("❌ Failed to connect to MT5 after all attempts")
    return False

# Quick setup guide
def print_setup_instructions():
    """Print setup instructions"""
    print("""
🎯 NEXUS UNIVERSAL MT5 SETUP INSTRUCTIONS
==========================================

📋 REQUIREMENTS:
1. ✅ Install MetaTrader 5 Terminal
2. ✅ Open MT5 Terminal (any broker)
3. ✅ Login to ANY MT5 Account (Demo or Live)
4. ✅ Enable "Allow DLL Imports" in MT5
5. ✅ Enable "Automated Trading" in MT5

🔗 CONNECTION OPTIONS:
• Option A: Login to MT5 manually, then start Nexus
• Option B: Use Nexus auto-connection (attempts 5 times)
• Option C: Connect to specific broker account

🚀 START TRADING:
1. Run Nexus Trading System
2. Go to: http://localhost:5173
3. Click "Connect MT5" in the interface
4. Start demo trading with 500 trades learning plan

💡 UNIVERSAL ADVANTAGES:
• Works with ANY MT5 broker worldwide
• No broker restrictions or approvals
• Direct API access to MT5 terminal
• Full trading functionality
• Real-time position management
• Account information access

⚠️ IMPORTANT NOTES:
• This bypasses broker-specific APIs
• Uses direct MT5 terminal connection
• Works with demo AND live accounts
• No additional broker setup required
• Full control over trading operations

Ready to connect to ANY MT5 broker! 🎯
    """)

if __name__ == "__main__":
    print_setup_instructions()
