#!/usr/bin/env python3
"""
🎯 NEXUS MT5 BROKER ADAPTER
Senior Quantitative Trading Systems Engineer - Production MT5 Integration
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_broker_adapter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MT5BrokerAdapter:
    """Production-ready MT5 broker adapter for Nexus Trading System"""
    
    def __init__(self):
        self.mt5_version = mt5.__version__
        self.connected = False
        self.account_info = None
        self.terminal_info = None
        self.login_credentials = None
        
        # XM Global MT5 Configuration
        self.xm_servers = [
            "XMGlobal-MT5 10",
            "XMGlobal-MT5 5",
            "XMGlobal-MT5",
            "XMGlobal-MT5 2",
            "XMGlobal-MT5 9"
        ]
        
        logger.info(f"🎯 MT5 Broker Adapter Initialized")
        logger.info(f"📦 MT5 Version: {self.mt5_version}")
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 terminal connection"""
        logger.info("🔍 Initializing MT5 terminal...")
        
        try:
            if mt5.initialize():
                logger.info("✅ MT5 terminal initialized successfully!")
                self.connected = True
                
                # Get terminal information
                self.terminal_info = mt5.terminal_info()
                if self.terminal_info:
                    logger.info("📊 MT5 Terminal Information:")
                    logger.info(f"  🏦 Name: {self.terminal_info.name}")
                    logger.info(f"  📊 Path: {self.terminal_info.path}")
                    logger.info(f"  📊 Version: {self.terminal_info.version}")
                    logger.info(f"  🏢 Company: {self.terminal_info.company}")
                
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                self.connected = False
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 initialization exception: {e}")
            self.connected = False
            return False
    
    def login_to_broker(self, login: int, password: str, server: str) -> bool:
        """Login to MT5 broker"""
        logger.info(f"🔑 Attempting login to {server}")
        logger.info(f"  👤 Account: {login}")
        logger.info(f"  🏦 Server: {server}")
        
        try:
            # Store credentials
            self.login_credentials = {
                'login': login,
                'password': password,
                'server': server
            }
            
            # Attempt login
            login_result = mt5.login(login=login, password=password, server=server)
            
            if login_result:
                logger.info("✅ Successfully logged into broker!")
                
                # Get account information
                self.account_info = mt5.account_info()
                if self.account_info:
                    logger.info("💰 Account Information:")
                    logger.info(f"  👤 Account: {self.account_info.login}")
                    logger.info(f"  🏦 Server: {self.account_info.server}")
                    logger.info(f"  💵 Balance: ${self.account_info.balance:.2f}")
                    logger.info(f"  📈 Equity: ${self.account_info.equity:.2f}")
                    logger.info(f"  🏦 Leverage: 1:{self.account_info.leverage}")
                    logger.info(f"  📊 Margin: ${self.account_info.margin:.2f}")
                    logger.info(f"  💸 Free Margin: ${self.account_info.margin_free:.2f}")
                    logger.info(f"  💰 Profit: ${self.account_info.profit:.2f}")
                    logger.info(f"  🏢 Broker: {self.account_info.company}")
                    logger.info(f"  📊 Account Type: {self.account_info.trade_mode_description}")
                
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ Login failed: {error_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login exception: {e}")
            return False
    
    def auto_login_to_xm(self, login: int, password: str) -> bool:
        """Auto-login to XM Global MT5 by trying all servers"""
        logger.info("🔄 Auto-login to XM Global MT5...")
        
        for server in self.xm_servers:
            logger.info(f"🔍 Trying server: {server}")
            
            if self.login_to_broker(login, password, server):
                logger.info(f"✅ Successfully connected to {server}")
                return True
            
            # Small delay between attempts
            import time
            time.sleep(1)
        
        logger.error("❌ Failed to connect to any XM Global server")
        return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return {
                    'symbol': symbol_info.name,
                    'bid': symbol_info.bid,
                    'ask': symbol_info.ask,
                    'spread': symbol_info.ask - symbol_info.bid,
                    'point': symbol_info.point,
                    'digits': symbol_info.digits,
                    'volume_min': symbol_info.volume_min,
                    'volume_max': symbol_info.volume_max,
                    'volume_step': symbol_info.volume_step,
                    'trade_mode': symbol_info.trade_mode_description,
                    'currency_base': symbol_info.currency_base,
                    'currency_profit': symbol_info.currency_profit,
                    'margin_required': symbol_info.margin_initial
                }
            else:
                logger.warning(f"⚠️ Symbol {symbol} not found")
                return None
        except Exception as e:
            logger.error(f"❌ Get symbol info failed for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': datetime.fromtimestamp(tick.time),
                    'spread': tick.ask - tick.bid
                }
            else:
                logger.warning(f"⚠️ No tick data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"❌ Get market data failed for {symbol}: {e}")
            return None
    
    def get_candle_data(self, symbol: str, timeframe: int, count: int = 100) -> List[Dict]:
        """Get candle data for specified timeframe"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is not None and len(rates) > 0:
                candle_data = []
                for rate in rates:
                    candle_data.append({
                        'time': datetime.fromtimestamp(rate['time']),
                        'open': rate['open'],
                        'high': rate['high'],
                        'low': rate['low'],
                        'close': rate['close'],
                        'volume': rate['tick_volume'],
                        'spread': rate['spread']
                    })
                
                logger.info(f"✅ Retrieved {len(candle_data)} candles for {symbol}")
                return candle_data
            else:
                logger.warning(f"⚠️ No candle data for {symbol}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Get candle data failed for {symbol}: {e}")
            return []
    
    def create_9h_candles(self, symbol: str, count: int = 100) -> List[Dict]:
        """Create 9H candles by aggregating H1 candles"""
        logger.info(f"🔨 Creating 9H candles for {symbol}...")
        
        # Get H1 candles
        h1_candles = self.get_candle_data(symbol, mt5.TIMEFRAME_H1, count * 9)
        
        if not h1_candles:
            return []
        
        # Aggregate into 9H candles
        candles_9h = []
        for i in range(0, len(h1_candles), 9):
            if i + 9 <= len(h1_candles):
                chunk = h1_candles[i:i+9]
                
                # Calculate OHLC for 9H period
                opens = [c['open'] for c in chunk]
                highs = [c['high'] for c in chunk]
                lows = [c['low'] for c in chunk]
                closes = [c['close'] for c in chunk]
                volumes = [c['volume'] for c in chunk]
                
                candle_9h = {
                    'time': chunk[0]['time'],
                    'open': opens[0],
                    'high': max(highs),
                    'low': min(lows),
                    'close': closes[-1],
                    'volume': sum(volumes),
                    'timeframe': '9H'
                }
                
                candles_9h.append(candle_9h)
        
        logger.info(f"✅ Created {len(candles_9h)} 9H candles for {symbol}")
        return candles_9h
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                  price: float = 0, sl: float = 0, tp: float = 0,
                  comment: str = "Nexus Trade") -> Dict:
        """Place trading order"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Determine price if not specified
            if price == 0:
                price = symbol_info.ask if order_type.upper() == 'BUY' else symbol_info.bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"🎉 Order executed successfully!")
                logger.info(f"  📈 Order #{result.order}")
                logger.info(f"  💰 Price: {result.price}")
                logger.info(f"  📊 Volume: {result.volume}")
                logger.info(f"  💸 Commission: ${result.commission}")
                logger.info(f"  📊 Swap: ${result.swap}")
                logger.info(f"  💰 Profit: ${result.profit}")
                
                return {
                    'success': True,
                    'order_id': result.order,
                    'price': result.price,
                    'volume': result.volume,
                    'commission': result.commission,
                    'swap': result.swap,
                    'profit': result.profit,
                    'symbol': symbol,
                    'type': order_type
                }
            else:
                logger.error(f"❌ Order failed: {result.retcode} - {result.comment}")
                return {
                    'success': False,
                    'error': f'Order failed: {result.retcode} - {result.comment}',
                    'retcode': result.retcode
                }
                
        except Exception as e:
            logger.error(f"❌ Place order exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            positions = mt5.positions_get()
            if positions:
                position_data = []
                for pos in positions:
                    position_data.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'swap': pos.swap,
                        'commission': pos.commission,
                        'time': datetime.fromtimestamp(pos.time),
                        'comment': pos.comment
                    })
                
                logger.info(f"📊 Retrieved {len(position_data)} open positions")
                return position_data
            else:
                logger.info("📊 No open positions")
                return []
                
        except Exception as e:
            logger.error(f"❌ Get positions failed: {e}")
            return []
    
    def close_position(self, ticket: int) -> Dict:
        """Close specific position"""
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': f'Position {ticket} not found'}
            
            position = position[0]
            
            # Create close request
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": position.symbol_info.bid if position.type == mt5.POSITION_TYPE_BUY else position.symbol_info.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(close_request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position #{ticket} closed successfully")
                return {'success': True, 'order_id': result.order}
            else:
                logger.error(f"❌ Close position failed: {result.retcode}")
                return {'success': False, 'error': f'Close failed: {result.retcode}'}
                
        except Exception as e:
            logger.error(f"❌ Close position exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        try:
            if not self.account_info:
                self.account_info = mt5.account_info()
            
            if self.account_info:
                positions = self.get_positions()
                
                summary = {
                    'account': self.account_info.login,
                    'server': self.account_info.server,
                    'balance': self.account_info.balance,
                    'equity': self.account_info.equity,
                    'margin': self.account_info.margin,
                    'free_margin': self.account_info.margin_free,
                    'profit': self.account_info.profit,
                    'leverage': self.account_info.leverage,
                    'broker': self.account_info.company,
                    'currency': self.account_info.currency,
                    'open_positions': len(positions),
                    'total_profit_loss': sum(pos['profit'] for pos in positions),
                    'last_update': datetime.now()
                }
                
                return summary
            else:
                return {'error': 'No account information available'}
                
        except Exception as e:
            logger.error(f"❌ Get account summary failed: {e}")
            return {'error': str(e)}
    
    def disconnect(self):
        """Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("🔌 Disconnected from MT5")
        except Exception as e:
            logger.error(f"❌ Disconnect failed: {e}")

def main():
    """Test MT5 broker adapter"""
    logger.info("🎯 NEXUS MT5 BROKER ADAPTER TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create adapter instance
    adapter = MT5BrokerAdapter()
    
    try:
        # Test initialization
        if not adapter.initialize_mt5():
            logger.error("❌ MT5 initialization failed")
            return
        
        # Test auto-login to XM Global
        login = 103969793
        password = "*d8qNgQq"
        
        if adapter.auto_login_to_xm(login, password):
            logger.info("✅ XM Global login successful!")
            
            # Test symbol info
            symbol_info = adapter.get_symbol_info("EURUSD")
            if symbol_info:
                logger.info("✅ Symbol info test passed")
            
            # Test market data
            market_data = adapter.get_market_data("EURUSD")
            if market_data:
                logger.info("✅ Market data test passed")
            
            # Test candle data
            candle_data = adapter.get_candle_data("EURUSD", mt5.TIMEFRAME_H1, 10)
            if candle_data:
                logger.info("✅ Candle data test passed")
            
            # Test 9H candles
            candles_9h = adapter.create_9h_candles("EURUSD", 10)
            if candles_9h:
                logger.info("✅ 9H candle creation test passed")
            
            # Test safe demo trade
            logger.info("🎯 Testing safe demo trade...")
            trade_result = adapter.place_order("EURUSD", "BUY", 0.01)
            if trade_result['success']:
                logger.info("✅ Demo trade test passed")
                
                # Get positions
                positions = adapter.get_positions()
                logger.info(f"📊 Current positions: {len(positions)}")
                
                # Test account summary
                summary = adapter.get_account_summary()
                if 'error' not in summary:
                    logger.info("✅ Account summary test passed")
                    logger.info(f"💰 Account balance: ${summary['balance']:.2f}")
                    logger.info(f"📈 Account equity: ${summary['equity']:.2f}")
                    logger.info(f"📊 Open positions: {summary['open_positions']}")
            
        else:
            logger.error("❌ XM Global login failed")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
    finally:
        adapter.disconnect()

if __name__ == "__main__":
    main()
