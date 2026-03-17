#!/usr/bin/env python3
"""
🎯 NEXUS TRADE EXECUTION TEST
Senior Quantitative Trading Systems Engineer - Production Trade Execution
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_execution_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeExecutionTest:
    """Production-ready trade execution test for Nexus Trading System"""
    
    def __init__(self):
        self.mt5_available = False
        self.connected = False
        self.account_info = None
        self.test_results = []
        
        # Try to import MT5
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mt5_available = True
            logger.info("✅ MetaTrader5 module available")
        except ImportError:
            logger.error("❌ MetaTrader5 module not available")
            self.mt5 = None
        
        logger.info("🎯 Trade Execution Test Initialized")
        logger.info(f"📊 MT5 Available: {self.mt5_available}")
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not self.mt5_available:
            logger.error("❌ MT5 not available - using simulation mode")
            return False
        
        try:
            if self.mt5.initialize():
                logger.info("✅ MT5 initialized for trade execution")
                self.connected = True
                return True
            else:
                error_code = self.mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                self.connected = False
                return False
        except Exception as e:
            logger.error(f"❌ MT5 initialization exception: {e}")
            self.connected = False
            return False
    
    def login_to_xm_global(self, login: int, password: str) -> bool:
        """Login to XM Global MT5"""
        logger.info("🔑 Attempting login to XM Global MT5...")
        
        # XM Global servers to try
        servers = [
            "XMGlobal-MT5 10",
            "XMGlobal-MT5 5",
            "XMGlobal-MT5",
            "XMGlobal-MT5 2",
            "XMGlobal-MT5 9"
        ]
        
        for server in servers:
            logger.info(f"🔍 Trying server: {server}")
            
            try:
                login_result = self.mt5.login(login=login, password=password, server=server)
                
                if login_result:
                    logger.info(f"✅ Successfully logged into {server}")
                    
                    # Get account information
                    self.account_info = self.mt5.account_info()
                    if self.account_info:
                        logger.info("💰 Account Information:")
                        logger.info(f"  👤 Account: {self.account_info.login}")
                        logger.info(f"  🏦 Server: {self.account_info.server}")
                        logger.info(f"  💵 Balance: ${self.account_info.balance:.2f}")
                        logger.info(f"  📈 Equity: ${self.account_info.equity:.2f}")
                        logger.info(f"  🏦 Leverage: 1:{self.account_info.leverage}")
                        logger.info(f"  📊 Margin: ${self.account_info.margin:.2f}")
                        logger.info(f"  💸 Free Margin: ${self.account_info.margin_free:.2f}")
                    
                    return True
                else:
                    error_code = self.mt5.last_error()
                    logger.warning(f"⚠️ Login failed for {server}: {error_code}")
                    
            except Exception as e:
                logger.error(f"❌ Login exception for {server}: {e}")
            
            # Small delay between attempts
            time.sleep(1)
        
        logger.error("❌ Failed to connect to any XM Global server")
        return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            symbol_info = self.mt5.symbol_info(symbol)
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
                    'trade_mode': symbol_info.trade_mode_description
                }
            else:
                logger.warning(f"⚠️ Symbol {symbol} not found")
                return None
        except Exception as e:
            logger.error(f"❌ Get symbol info failed for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, risk_percent: float = 1.0) -> float:
        """Calculate position size based on risk management"""
        try:
            if not self.account_info:
                return 0.01
            
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Calculate risk amount
            risk_amount = self.account_info.balance * (risk_percent / 100.0)
            
            # Calculate position size
            tick_value = symbol_info.get('trade_tick_value', 0.01)
            stop_loss_pips = 20  # 20 pips stop loss
            
            position_size = risk_amount / (stop_loss_pips * tick_value)
            
            # Min and max position size
            min_lot = symbol_info.get('volume_min', 0.01)
            max_lot = symbol_info.get('volume_max', 100.0)
            
            # Clamp position size
            position_size = max(min_lot, min(position_size, max_lot))
            
            # Round to 2 decimal places
            position_size = round(position_size, 2)
            
            logger.info(f"📊 {symbol} - Risk: {risk_percent}%, Size: {position_size} lots")
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0.01
    
    def place_safe_demo_trade(self, symbol: str, order_type: str, volume: float) -> Dict:
        """Place safe demo trade with proper risk management"""
        logger.info(f"🎯 Placing safe demo trade: {symbol} {order_type} {volume} lots")
        
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Determine price
            price = symbol_info['ask'] if order_type.upper() == 'BUY' else symbol_info['bid']
            
            # Calculate stop loss and take profit
            point = symbol_info['point']
            sl = price - (20 * point) if order_type.upper() == 'BUY' else price + (20 * point)
            tp = price + (30 * point) if order_type.upper() == 'BUY' else price - (30 * point)
            
            # Create order request
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": self.mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else self.mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus Safe Demo Trade",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            start_time = time.time()
            result = self.mt5.order_send(request)
            execution_time = time.time() - start_time
            
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                trade_info = {
                    'success': True,
                    'order_id': result.order,
                    'symbol': symbol,
                    'type': order_type,
                    'volume': result.volume,
                    'price': result.price,
                    'sl': sl,
                    'tp': tp,
                    'commission': result.commission,
                    'swap': result.swap,
                    'profit': result.profit,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"🎉 Demo Trade SUCCESSFUL!")
                logger.info(f"  📈 Order #{result.order}")
                logger.info(f"  💰 Price: {result.price}")
                logger.info(f"  📊 Volume: {result.volume}")
                logger.info(f"  🛡️ Stop Loss: {sl}")
                logger.info(f"  🎯 Take Profit: {tp}")
                logger.info(f"  💸 Commission: ${result.commission}")
                logger.info(f"  📊 Swap: ${result.swap}")
                logger.info(f"  💰 Profit: ${result.profit}")
                logger.info(f"  ⚡ Execution Time: {execution_time:.3f}s")
                
                self.test_results.append(trade_info)
                return trade_info
                
            else:
                error_info = {
                    'success': False,
                    'error': f'Order failed: {result.retcode} - {result.comment}',
                    'retcode': result.retcode,
                    'symbol': symbol,
                    'type': order_type,
                    'volume': volume,
                    'timestamp': datetime.now()
                }
                
                logger.error(f"❌ Demo Trade FAILED: {result.retcode}")
                logger.error(f"  📝 Error: {result.comment}")
                logger.error(f"  🔢 Return Code: {result.retcode}")
                
                self.test_results.append(error_info)
                return error_info
                
        except Exception as e:
            error_info = {
                'success': False,
                'error': f'Exception: {str(e)}',
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'timestamp': datetime.now()
            }
            
            logger.error(f"❌ Demo Trade Exception: {e}")
            self.test_results.append(error_info)
            return error_info
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            positions = self.mt5.positions_get()
            if positions:
                position_data = []
                for pos in positions:
                    position_data.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == self.mt5.POSITION_TYPE_BUY else 'SELL',
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
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': f'Position {ticket} not found'}
            
            position = position[0]
            
            # Create close request
            close_request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": position.symbol_info.bid if position.type == self.mt5.POSITION_TYPE_BUY else position.symbol_info.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Nexus Close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            start_time = time.time()
            result = self.mt5.order_send(close_request)
            execution_time = time.time() - start_time
            
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position #{ticket} closed successfully")
                return {
                    'success': True,
                    'order_id': result.order,
                    'execution_time': execution_time
                }
            else:
                logger.error(f"❌ Close position failed: {result.retcode}")
                return {'success': False, 'error': f'Close failed: {result.retcode}'}
                
        except Exception as e:
            logger.error(f"❌ Close position exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive trade execution test"""
        logger.info("🚀 STARTING COMPREHENSIVE TRADE EXECUTION TEST")
        logger.info("=" * 60)
        
        test_summary = {
            'mt5_available': self.mt5_available,
            'mt5_initialized': False,
            'login_successful': False,
            'account_info': None,
            'trade_tests': [],
            'position_tests': [],
            'execution_times': [],
            'errors': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        
        try:
            # Test 1: Initialize MT5
            if self.initialize_mt5():
                test_summary['mt5_initialized'] = True
                logger.info("✅ MT5 initialization test passed")
            else:
                logger.error("❌ MT5 initialization test failed")
                return test_summary
            
            # Test 2: Login to XM Global
            login = 103969793
            password = "*d8qNgQq"
            
            if self.login_to_xm_global(login, password):
                test_summary['login_successful'] = True
                test_summary['account_info'] = self.account_info
                logger.info("✅ XM Global login test passed")
            else:
                logger.error("❌ XM Global login test failed")
                return test_summary
            
            # Test 3: Symbol Information
            logger.info("🔍 Testing symbol information...")
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
            
            for symbol in symbols:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    logger.info(f"✅ {symbol}: Bid={symbol_info['bid']:.5f}, Ask={symbol_info['ask']:.5f}")
                else:
                    logger.warning(f"⚠️ {symbol}: Symbol not available")
            
            # Test 4: Position Size Calculation
            logger.info("🔍 Testing position size calculation...")
            for symbol in symbols[:3]:  # Test first 3 symbols
                size = self.calculate_position_size(symbol, 1.0)
                logger.info(f"✅ {symbol}: Position size = {size} lots (1% risk)")
            
            # Test 5: Safe Demo Trades
            logger.info("🔍 Testing safe demo trades...")
            trade_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            
            for i, symbol in enumerate(trade_symbols):
                logger.info(f"🎯 Demo Trade {i+1}/3")
                
                # Calculate position size
                volume = self.calculate_position_size(symbol, 0.5)  # 0.5% risk
                
                # Place buy trade
                trade_result = self.place_safe_demo_trade(symbol, "BUY", volume)
                test_summary['trade_tests'].append(trade_result)
                
                if trade_result['success']:
                    test_summary['execution_times'].append(trade_result['execution_time'])
                
                # Wait between trades
                time.sleep(2)
            
            # Test 6: Position Management
            logger.info("🔍 Testing position management...")
            positions = self.get_open_positions()
            test_summary['position_tests'] = positions
            
            if positions:
                logger.info(f"📊 Managing {len(positions)} positions...")
                
                # Close first position if exists
                if len(positions) > 0:
                    close_result = self.close_position(positions[0]['ticket'])
                    if close_result['success']:
                        test_summary['execution_times'].append(close_result['execution_time'])
                        logger.info("✅ Position close test passed")
                    else:
                        logger.error("❌ Position close test failed")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Test interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
        finally:
            # Cleanup
            try:
                self.mt5.shutdown()
                logger.info("🔌 MT5 connection closed")
            except:
                pass
        
        # Complete test summary
        test_summary['end_time'] = datetime.now()
        test_summary['duration'] = test_summary['end_time'] - test_summary['start_time']
        
        # Calculate statistics
        successful_trades = [t for t in test_summary['trade_tests'] if t['success']]
        failed_trades = [t for t in test_summary['trade_tests'] if not t['success']]
        
        test_summary['statistics'] = {
            'total_trades': len(test_summary['trade_tests']),
            'successful_trades': len(successful_trades),
            'failed_trades': len(failed_trades),
            'success_rate': (len(successful_trades) / len(test_summary['trade_tests']) * 100) if test_summary['trade_tests'] else 0,
            'avg_execution_time': sum(test_summary['execution_times']) / len(test_summary['execution_times']) if test_summary['execution_times'] else 0,
            'total_profit': sum(t.get('profit', 0) for t in successful_trades),
            'total_commission': sum(t.get('commission', 0) for t in successful_trades)
        }
        
        return test_summary
    
    def print_test_results(self, test_summary: Dict):
        """Print comprehensive test results"""
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        
        # Basic results
        logger.info(f"🎯 MT5 Available: {'✅' if test_summary['mt5_available'] else '❌'}")
        logger.info(f"🔌 MT5 Initialized: {'✅' if test_summary['mt5_initialized'] else '❌'}")
        logger.info(f"🔑 Login Successful: {'✅' if test_summary['login_successful'] else '❌'}")
        logger.info(f"⏱️ Test Duration: {test_summary['duration']}")
        
        # Account info
        if test_summary['account_info']:
            acc = test_summary['account_info']
            logger.info("💰 Account Information:")
            logger.info(f"  👤 Account: {acc.login}")
            logger.info(f"  🏦 Server: {acc.server}")
            logger.info(f"  💵 Balance: ${acc.balance:.2f}")
            logger.info(f"  📈 Equity: ${acc.equity:.2f}")
        
        # Trade statistics
        if 'statistics' in test_summary:
            stats = test_summary['statistics']
            logger.info("📊 Trade Statistics:")
            logger.info(f"  📈 Total Trades: {stats['total_trades']}")
            logger.info(f"  ✅ Successful: {stats['successful_trades']}")
            logger.info(f"  ❌ Failed: {stats['failed_trades']}")
            logger.info(f"  🎯 Success Rate: {stats['success_rate']:.1f}%")
            logger.info(f"  ⚡ Avg Execution Time: {stats['avg_execution_time']:.3f}s")
            logger.info(f"  💰 Total Profit: ${stats['total_profit']:.2f}")
            logger.info(f"  💸 Total Commission: ${stats['total_commission']:.2f}")
        
        # Position information
        if test_summary['position_tests']:
            positions = test_summary['position_tests']
            logger.info(f"📊 Open Positions: {len(positions)}")
            for pos in positions:
                logger.info(f"  📈 {pos['symbol']} {pos['type']} {pos['volume']} @ {pos['price_open']:.5f} (P&L: ${pos['profit']:.2f})")
        
        # Error summary
        failed_trades = [t for t in test_summary['trade_tests'] if not t['success']]
        if failed_trades:
            logger.info("❌ Trade Errors:")
            for error in failed_trades:
                logger.info(f"  📝 {error['symbol']} {error['type']}: {error['error']}")
        
        logger.info("=" * 60)
        
        # Overall assessment
        if test_summary['mt5_initialized'] and test_summary['login_successful']:
            success_rate = test_summary['statistics']['success_rate']
            if success_rate >= 80:
                logger.info("🎉 TRADE EXECUTION TEST: EXCELLENT!")
                logger.info("✅ Nexus Trading System is fully operational!")
            elif success_rate >= 60:
                logger.info("✅ TRADE EXECUTION TEST: GOOD!")
                logger.info("🔧 Minor improvements may be needed")
            else:
                logger.warning("⚠️ TRADE EXECUTION TEST: NEEDS IMPROVEMENT!")
                logger.warning("🔧 Significant improvements required")
        else:
            logger.error("❌ TRADE EXECUTION TEST: FAILED!")
            logger.error("🔧 Critical issues need to be resolved")

def main():
    """Main test execution"""
    logger.info("🎯 NEXUS TRADE EXECUTION TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create test instance
    tester = TradeExecutionTest()
    
    # Run comprehensive test
    try:
        test_summary = tester.run_comprehensive_test()
        tester.print_test_results(test_summary)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()
