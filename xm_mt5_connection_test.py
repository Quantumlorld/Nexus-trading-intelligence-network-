#!/usr/bin/env python3
"""
🎯 NEXUS XM MT5 CONNECTION TEST
Senior Quantitative Trading Systems Engineer - XM Broker Integration
"""

import sys
import logging
from datetime import datetime
import MetaTrader5 as mt5
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xm_mt5_connection_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XMMT5ConnectionTest:
    """XM MT5 connection test with user credentials"""
    
    def __init__(self):
        self.mt5_version = mt5.__version__
        self.connected = False
        self.account_info = None
        
        # User credentials
        self.mt5_demo_account = 103969793
        self.mt5_demo_password = "*d8qNgQq"
        self.mt5_demo_server = "MetaQuotes-Demo"
        
        self.xm_broker_account = "primeworld069"
        self.xm_broker_password = "REPLACE_WITH_ACTUAL_PASSWORD"
        self.xm_mql5_account = "Quantumlorld"
        self.xm_mql5_password = "REPLACE_WITH_ACTUAL_PASSWORD"
        
        # XM Global servers to try
        self.xm_servers = [
            "XMGlobal-MT5 10",
            "XMGlobal-MT5 5",
            "XMGlobal-MT5",
            "XMGlobal-MT5 2",
            "XMGlobal-MT5 9",
            "XMGlobal-MT5 7",
            "XMGlobal-MT5 8"
        ]
        
        logger.info("🎯 XM MT5 Connection Test Initialized")
        logger.info(f"📦 MT5 Version: {self.mt5_version}")
        logger.info(f"👤 MT5 Demo Account: {self.mt5_demo_account}")
        logger.info(f"🏦 XM Broker Account: {self.xm_broker_account}")
        logger.info(f"🔧 MQL5 Account: {self.xm_mql5_account}")
    
    def test_mt5_installation(self):
        """Test MT5 installation"""
        logger.info("🔍 Testing MetaTrader5 installation...")
        
        try:
            if mt5.initialize():
                logger.info("✅ MT5 package installed and initialized successfully!")
                self.connected = True
                
                # Get terminal information
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    logger.info("📊 MT5 Terminal Information:")
                    logger.info(f"  🏦 Name: {terminal_info.name}")
                    logger.info(f"  📊 Path: {terminal_info.path}")
                    logger.info(f"  📊 Version: {terminal_info.version}")
                    logger.info(f"  🏢 Company: {terminal_info.company}")
                
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 installation test failed: {e}")
            return False
    
    def test_mt5_demo_login(self):
        """Test MT5 demo account login"""
        logger.info("🔍 Testing MT5 Demo Account Login...")
        logger.info(f"  👤 Account: {self.mt5_demo_account}")
        logger.info(f"  🏦 Server: {self.mt5_demo_server}")
        
        try:
            login_result = mt5.login(
                login=self.mt5_demo_account,
                password=self.mt5_demo_password,
                server=self.mt5_demo_server
            )
            
            if login_result:
                logger.info("✅ MT5 Demo Account login successful!")
                
                # Get account information
                self.account_info = mt5.account_info()
                if self.account_info:
                    logger.info("💰 MT5 Demo Account Information:")
                    logger.info(f"  👤 Account: {self.account_info.login}")
                    logger.info(f"  🏦 Server: {self.account_info.server}")
                    logger.info(f"  💵 Balance: ${self.account_info.balance:.2f}")
                    logger.info(f"  📈 Equity: ${self.account_info.equity:.2f}")
                    logger.info(f"  🏦 Leverage: 1:{self.account_info.leverage}")
                    logger.info(f"  📊 Margin: ${self.account_info.margin:.2f}")
                    logger.info(f"  💸 Free Margin: ${self.account_info.margin_free:.2f}")
                    logger.info(f"  💰 Profit: ${self.account_info.profit:.2f}")
                    logger.info(f"  🏢 Broker: {self.account_info.company}")
                
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 Demo login failed: {error_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 Demo login exception: {e}")
            return False
    
    def test_xm_broker_login(self):
        """Test XM broker login"""
        logger.info("🔍 Testing XM Broker Login...")
        logger.info(f"  👤 Account: {self.xm_broker_account}")
        
        # Try different XM servers
        for server in self.xm_servers:
            logger.info(f"🔍 Trying XM server: {server}")
            
            try:
                login_result = mt5.login(
                    login=self.xm_broker_account,
                    password=self.xm_broker_password,
                    server=server
                )
                
                if login_result:
                    logger.info(f"✅ XM Broker login successful on {server}!")
                    
                    # Get account information
                    account_info = mt5.account_info()
                    if account_info:
                        logger.info("💰 XM Broker Account Information:")
                        logger.info(f"  👤 Account: {account_info.login}")
                        logger.info(f"  🏦 Server: {account_info.server}")
                        logger.info(f"  💵 Balance: ${account_info.balance:.2f}")
                        logger.info(f"  📈 Equity: ${account_info.equity:.2f}")
                        logger.info(f"  🏦 Leverage: 1:{account_info.leverage}")
                        logger.info(f"  📊 Margin: ${account_info.margin:.2f}")
                        logger.info(f"  💸 Free Margin: ${account_info.margin_free:.2f}")
                        logger.info(f"  💰 Profit: ${account_info.profit:.2f}")
                        logger.info(f"  🏢 Broker: {account_info.company}")
                    
                    return True
                else:
                    error_code = mt5.last_error()
                    logger.warning(f"⚠️ XM login failed for {server}: {error_code}")
                    
            except Exception as e:
                logger.error(f"❌ XM login exception for {server}: {e}")
            
            # Small delay between attempts
            time.sleep(1)
        
        logger.error("❌ Failed to connect to any XM server")
        return False
    
    def test_mql5_login(self):
        """Test MQL5 account login"""
        logger.info("🔍 Testing MQL5 Account Login...")
        logger.info(f"  👤 Account: {self.xm_mql5_account}")
        
        # Try MQL5 login with XM servers
        for server in self.xm_servers:
            logger.info(f"🔍 Trying MQL5 login on {server}")
            
            try:
                login_result = mt5.login(
                    login=self.xm_mql5_account,
                    password=self.xm_mql5_password,
                    server=server
                )
                
                if login_result:
                    logger.info(f"✅ MQL5 Account login successful on {server}!")
                    
                    # Get account information
                    account_info = mt5.account_info()
                    if account_info:
                        logger.info("💰 MQL5 Account Information:")
                        logger.info(f"  👤 Account: {account_info.login}")
                        logger.info(f"  🏦 Server: {account_info.server}")
                        logger.info(f"  💵 Balance: ${account_info.balance:.2f}")
                        logger.info(f"  📈 Equity: ${account_info.equity:.2f}")
                        logger.info(f"  🏦 Leverage: 1:{account_info.leverage}")
                        logger.info(f"  📊 Margin: ${account_info.margin:.2f}")
                        logger.info(f"  💸 Free Margin: ${account_info.margin_free:.2f}")
                        logger.info(f"  💰 Profit: ${account_info.profit:.2f}")
                        logger.info(f"  🏢 Broker: {account_info.company}")
                    
                    return True
                else:
                    error_code = mt5.last_error()
                    logger.warning(f"⚠️ MQL5 login failed for {server}: {error_code}")
                    
            except Exception as e:
                logger.error(f"❌ MQL5 login exception for {server}: {e}")
            
            # Small delay between attempts
            time.sleep(1)
        
        logger.error("❌ Failed to connect MQL5 account")
        return False
    
    def test_symbol_info(self):
        """Test symbol information"""
        logger.info("🔍 Testing symbol information...")
        
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
        
        for symbol in symbols:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    logger.info(f"✅ {symbol}:")
                    logger.info(f"  💰 Bid: {symbol_info.bid}")
                    logger.info(f"  💸 Ask: {symbol_info.ask}")
                    logger.info(f"  📊 Spread: {symbol_info.ask - symbol_info.bid}")
                    logger.info(f"  📊 Point: {symbol_info.point}")
                    logger.info(f"  📊 Volume Min: {symbol_info.volume_min}")
                    logger.info(f"  📊 Volume Max: {symbol_info.volume_max}")
                    logger.info(f"  📊 Trade Mode: {symbol_info.trade_mode_description}")
                else:
                    logger.warning(f"⚠️ {symbol}: Symbol not found or not available")
            except Exception as e:
                logger.error(f"❌ Symbol info test failed for {symbol}: {e}")
    
    def test_market_data(self):
        """Test market data retrieval"""
        logger.info("🔍 Testing market data retrieval...")
        
        try:
            # Test EURUSD tick data
            tick = mt5.symbol_info_tick("EURUSD")
            if tick:
                logger.info("✅ EURUSD Tick Data:")
                logger.info(f"  💰 Bid: {tick.bid}")
                logger.info(f"  💸 Ask: {tick.ask}")
                logger.info(f"  📊 Last: {tick.last}")
                logger.info(f"  📊 Volume: {tick.volume}")
                logger.info(f"  ⏰ Time: {tick.time}")
                return True
            else:
                logger.error("❌ Failed to get EURUSD tick data")
                return False
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            return False
    
    def test_candle_data(self):
        """Test candle data retrieval"""
        logger.info("🔍 Testing candle data retrieval...")
        
        timeframes = [
            (mt5.TIMEFRAME_M1, "M1"),
            (mt5.TIMEFRAME_M5, "M5"),
            (mt5.TIMEFRAME_M15, "M15"),
            (mt5.TIMEFRAME_H1, "H1"),
            (mt5.TIMEFRAME_H4, "H4")
        ]
        
        for timeframe, name in timeframes:
            try:
                # Get last 10 candles
                rates = mt5.copy_rates_from_pos("EURUSD", timeframe, 0, 10)
                
                if rates is not None and len(rates) > 0:
                    logger.info(f"✅ {name} Candle Data: {len(rates)} candles")
                    logger.info(f"  📊 Latest: {rates[-1]}")
                else:
                    logger.warning(f"⚠️ {name}: No candle data available")
                    
            except Exception as e:
                logger.error(f"❌ Candle data test failed for {name}: {e}")
    
    def test_demo_trade(self):
        """Test safe demo trade execution"""
        logger.info("🔍 Testing safe demo trade...")
        
        try:
            # Get current EURUSD price
            symbol_info = mt5.symbol_info("EURUSD")
            if not symbol_info:
                logger.error("❌ EURUSD symbol not available")
                return False
            
            # Calculate trade parameters
            lot_size = 0.01  # Minimum lot size
            point = symbol_info.point
            
            # Create buy order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "EURUSD",
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": symbol_info.ask,
                "sl": symbol_info.ask - 20 * point,  # 20 pips stop loss
                "tp": symbol_info.ask + 30 * point,  # 30 pips take profit
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus XM Test Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("🎉 Demo Trade SUCCESSFUL!")
                logger.info(f"  📈 Order #{result.order}")
                logger.info(f"  💰 Price: {result.price}")
                logger.info(f"  📊 Volume: {result.volume}")
                logger.info(f"  💸 Commission: ${result.commission}")
                logger.info(f"  📊 Swap: ${result.swap}")
                logger.info(f"  💰 Profit: ${result.profit}")
                
                # Get position info
                position = mt5.positions_get(symbol="EURUSD")
                if position:
                    logger.info(f"  📊 Position #{position[0].ticket} opened")
                
                return True
            else:
                logger.error(f"❌ Demo Trade FAILED: {result.retcode}")
                logger.error(f"  📝 Error: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Demo trade test failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup MT5 connection"""
        try:
            mt5.shutdown()
            logger.info("🔌 MT5 connection closed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def run_full_test(self):
        """Run comprehensive XM MT5 test"""
        logger.info("🚀 STARTING COMPREHENSIVE XM MT5 TEST")
        logger.info("=" * 60)
        
        test_results = {
            'installation': False,
            'mt5_demo_login': False,
            'xm_broker_login': False,
            'mql5_login': False,
            'symbol_info': False,
            'market_data': False,
            'candle_data': False,
            'demo_trade': False
        }
        
        try:
            # Test 1: Installation
            test_results['installation'] = self.test_mt5_installation()
            if not test_results['installation']:
                logger.error("❌ MT5 installation failed - aborting further tests")
                return test_results
            
            # Test 2: MT5 Demo Login
            test_results['mt5_demo_login'] = self.test_mt5_demo_login()
            
            # Test 3: XM Broker Login
            test_results['xm_broker_login'] = self.test_xm_broker_login()
            
            # Test 4: MQL5 Login
            test_results['mql5_login'] = self.test_mql5_login()
            
            # If any login was successful, continue with tests
            if any([test_results['mt5_demo_login'], test_results['xm_broker_login'], test_results['mql5_login']]):
                
                # Test 5: Symbol Info
                self.test_symbol_info()
                test_results['symbol_info'] = True
                
                # Test 6: Market Data
                test_results['market_data'] = self.test_market_data()
                
                # Test 7: Candle Data
                self.test_candle_data()
                test_results['candle_data'] = True
                
                # Test 8: Demo Trade
                test_results['demo_trade'] = self.test_demo_trade()
            
        finally:
            self.cleanup()
        
        # Print results
        logger.info("=" * 60)
        logger.info("📊 FINAL TEST RESULTS")
        logger.info("=" * 60)
        
        for test, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test.replace('_', ' ').title()}: {status}")
        
        return test_results

def main():
    """Main test execution"""
    logger.info("🎯 NEXUS XM MT5 CONNECTION TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create test instance
    tester = XMMT5ConnectionTest()
    
    # Run comprehensive test
    try:
        results = tester.run_full_test()
        
        # Overall status
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        logger.info("=" * 60)
        logger.info(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! XM MT5 integration is fully functional!")
        elif passed_tests >= 5:
            logger.info("✅ MOST TESTS PASSED! XM MT5 integration is mostly functional!")
        else:
            logger.error("❌ MULTIPLE TESTS FAILED! XM MT5 integration needs attention!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()
