#!/usr/bin/env python3
"""
🎯 NEXUS MT5 CONNECTION TEST
Senior Quantitative Trading Systems Engineer - MT5 Integration Test
"""

import sys
import logging
from datetime import datetime
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MT5ConnectionTest:
    """Comprehensive MT5 connection and trading test"""
    
    def __init__(self):
        self.mt5_version = mt5.__version__
        self.connected = False
        self.account_info = None
        
    def test_mt5_installation(self):
        """Test MT5 package installation"""
        logger.info("🔍 Testing MetaTrader5 installation...")
        logger.info(f"📦 MT5 Version: {self.mt5_version}")
        
        try:
            # Test MT5 initialization
            if mt5.initialize():
                logger.info("✅ MT5 package installed and initialized successfully!")
                self.connected = True
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                return False
        except Exception as e:
            logger.error(f"❌ MT5 installation test failed: {e}")
            return False
    
    def test_terminal_info(self):
        """Test MT5 terminal information"""
        logger.info("🔍 Testing MT5 terminal information...")
        
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info("✅ MT5 Terminal Information:")
                logger.info(f"  📊 Name: {terminal_info.name}")
                logger.info(f"  📊 Path: {terminal_info.path}")
                logger.info(f"  📊 Data Path: {terminal_info.data_path}")
                logger.info(f"  📊 Version: {terminal_info.version}")
                logger.info(f"  📊 Build: {terminal_info.build}")
                logger.info(f"  📊 Company: {terminal_info.company}")
                return True
            else:
                logger.error("❌ No MT5 terminal information available")
                return False
        except Exception as e:
            logger.error(f"❌ Terminal info test failed: {e}")
            return False
    
    def test_login_to_xm_global(self):
        """Test login to XM Global MT5"""
        logger.info("🔍 Testing login to XM Global MT5...")
        
        # XM Global credentials
        login = 103969793
        password = "*d8qNgQq"
        server = "XMGlobal-MT5 10"
        
        try:
            # Attempt login
            login_result = mt5.login(login=login, password=password, server=server)
            
            if login_result:
                logger.info("✅ Successfully logged into XM Global MT5!")
                logger.info(f"  🔑 Account: {login}")
                logger.info(f"  🏦 Server: {server}")
                
                # Get account information
                self.account_info = mt5.account_info()
                if self.account_info:
                    logger.info("💰 Account Information:")
                    logger.info(f"  💵 Balance: ${self.account_info.balance:.2f}")
                    logger.info(f"  📈 Equity: ${self.account_info.equity:.2f}")
                    logger.info(f"  🏦 Leverage: 1:{self.account_info.leverage}")
                    logger.info(f"  📊 Margin: ${self.account_info.margin:.2f}")
                    logger.info(f"  💸 Free Margin: ${self.account_info.margin_free:.2f}")
                    logger.info(f"  📊 Profit: ${self.account_info.profit:.2f}")
                    logger.info(f"  🏦 Broker: {self.account_info.server}")
                
                return True
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ Login failed: {error_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login test failed: {e}")
            return False
    
    def test_symbol_info(self):
        """Test symbol information retrieval"""
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
                    logger.info(f"  📊 Trade Mode: {symbol_info.trade_mode}")
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
            tick_value = symbol_info.trade_tick_value
            
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
                "comment": "Nexus MT5 Test Trade",
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
        """Run comprehensive MT5 test"""
        logger.info("🚀 STARTING COMPREHENSIVE MT5 TEST")
        logger.info("=" * 60)
        
        test_results = {
            'installation': False,
            'terminal_info': False,
            'login': False,
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
            
            # Test 2: Terminal Info
            test_results['terminal_info'] = self.test_terminal_info()
            
            # Test 3: Login to XM Global
            test_results['login'] = self.test_login_to_xm_global()
            
            if test_results['login']:
                # Test 4: Symbol Info
                self.test_symbol_info()
                test_results['symbol_info'] = True
                
                # Test 5: Market Data
                test_results['market_data'] = self.test_market_data()
                
                # Test 6: Candle Data
                self.test_candle_data()
                test_results['candle_data'] = True
                
                # Test 7: Demo Trade
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
    logger.info("🎯 NEXUS MT5 CONNECTION TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create test instance
    tester = MT5ConnectionTest()
    
    # Run comprehensive test
    try:
        results = tester.run_full_test()
        
        # Overall status
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        logger.info("=" * 60)
        logger.info(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! MT5 integration is fully functional!")
        elif passed_tests >= 5:
            logger.info("✅ MOST TESTS PASSED! MT5 integration is mostly functional!")
        else:
            logger.error("❌ MULTIPLE TESTS FAILED! MT5 integration needs attention!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()
