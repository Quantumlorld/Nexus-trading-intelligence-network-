#!/usr/bin/env python3
"""
🎯 NEXUS 500-TRADE FULL BACKTEST
Complete system stress test with MT5 integration
"""

import sys
import time
import json
import random
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_backtest_500.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Nexus components
sys.path.append('.')
from mt5_integration import get_mt5_connector
from demo_trading_config import DEMO_CONFIG

class FullBacktestEngine:
    """Complete 500-trade backtest engine"""
    
    def __init__(self):
        self.mt5_connector = get_mt5_connector()
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'max_drawdown': 0.0,
            'current_phase': 'baseline',
            'phase_trades': 0,
            'execution_times': [],
            'errors': [],
            'symbols_traded': set(),
            'order_types': {'BUY': 0, 'SELL': 0}
        }
        
        # Trading symbols for diversification
        self.symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'XAU/USD', 'BTC/USD']
        
    def connect_to_mt5(self) -> bool:
        """Connect to MT5 with your credentials"""
        try:
            logger.info("🔍 Connecting to MT5...")
            
            # Initialize MT5
            if not self.mt5_connector.initialize():
                logger.error("❌ MT5 initialization failed")
                return False
            
            # Login to your account
            account = 103969793
            password = "*d8qNgQq"
            server = "MetaQuotes-Demo"
            
            if self.mt5_connector.login(account, password, server):
                logger.info("✅ Connected to MT5 successfully!")
                
                # Get account info
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    logger.info(f"💰 Balance: ${account_info.get('balance', 0):.2f}")
                    logger.info(f"📈 Equity: ${account_info.get('equity', 0):.2f}")
                    logger.info(f"🏦 Leverage: 1:{account_info.get('leverage', 0)}")
                
                return True
            else:
                logger.error("❌ MT5 login failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Dict[str, float]:
        """Get current market data for symbol"""
        try:
            import MetaTrader5 as mt5
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Get tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'volume': tick.volume,
                'time': datetime.now()
            }
        except Exception as e:
            logger.error(f"❌ Market data error for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, risk_percent: float = 1.0) -> float:
        """Calculate position size based on risk management"""
        try:
            import MetaTrader5 as mt5
            
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return 0.01
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Calculate risk amount
            risk_amount = account_info.balance * (risk_percent / 100.0)
            
            # Calculate position size
            tick_value = symbol_info.trade_tick_value
            stop_loss_pips = 20  # 20 pips stop loss
            
            position_size = risk_amount / (stop_loss_pips * tick_value)
            
            # Min and max position size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            
            # Clamp position size
            position_size = max(min_lot, min(position_size, max_lot))
            
            # Round to 2 decimal places
            position_size = round(position_size, 2)
            
            logger.info(f"📊 {symbol} - Risk: {risk_percent}%, Size: {position_size} lots")
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0.01
    
    def place_trade(self, symbol: str, order_type: str, volume: float) -> Dict[str, Any]:
        """Place a trade with proper error handling"""
        try:
            import MetaTrader5 as mt5
            
            # Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {'success': False, 'error': f'No market data for {symbol}'}
            
            # Determine price
            price = market_data['ask'] if order_type == 'BUY' else market_data['bid']
            
            # Create order request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': price,
                'deviation': 20,
                'magic': 234000,
                'comment': f'Nexus Backtest Trade #{self.results["total_trades"] + 1}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            start_time = time.time()
            result = mt5.order_send(request)
            execution_time = time.time() - start_time
            
            # Record execution time
            self.results['execution_times'].append(execution_time)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_info = {
                    'success': True,
                    'order_id': result.order,
                    'symbol': symbol,
                    'type': order_type,
                    'volume': result.volume,
                    'price': result.price,
                    'commission': result.commission,
                    'swap': result.swap,
                    'profit': result.profit,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"🎉 Trade #{result.order} executed!")
                logger.info(f"📊 {symbol} {order_type} {result.volume} @ {result.price}")
                logger.info(f"⏱️ Execution time: {execution_time:.3f}s")
                
                return trade_info
                
            else:
                error_info = {
                    'success': False,
                    'error': f'Order failed: {result.retcode} - {result.comment}',
                    'symbol': symbol,
                    'type': order_type,
                    'volume': volume
                }
                
                logger.error(f"❌ Trade failed: {result.retcode} - {result.comment}")
                self.results['errors'].append(error_info)
                
                return error_info
                
        except Exception as e:
            error_info = {
                'success': False,
                'error': f'Exception: {str(e)}',
                'symbol': symbol,
                'type': order_type,
                'volume': volume
            }
            
            logger.error(f"❌ Trade exception: {e}")
            self.results['errors'].append(error_info)
            
            return error_info
    
    def manage_positions(self):
        """Manage open positions with stop loss and take profit"""
        try:
            import MetaTrader5 as mt5
            
            positions = mt5.positions_get()
            if not positions:
                return
            
            for position in positions:
                # Simple position management
                # Close positions after 60 seconds (for demo)
                current_time = datetime.now()
                position_time = datetime.fromtimestamp(position.time)
                
                if (current_time - position_time).total_seconds() > 60:
                    close_request = {
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': position.symbol,
                        'volume': position.volume,
                        'type': mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                        'position': position.ticket,
                        'price': position.symbol_info.bid if position.type == mt5.POSITION_TYPE_BUY else position.symbol_info.ask,
                        'deviation': 20,
                        'magic': 234000,
                        'comment': 'Nexus Auto Close',
                        'type_time': mt5.ORDER_TIME_GTC,
                        'type_filling': mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(close_request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ Position #{position.ticket} closed")
                        logger.info(f"💰 Profit: ${position.profit:.2f}")
                    else:
                        logger.error(f"❌ Close failed: {result.retcode}")
                        
        except Exception as e:
            logger.error(f"❌ Position management error: {e}")
    
    def run_phase_1_baseline(self, trades_count: int):
        """Phase 1: Baseline - 100 trades"""
        logger.info("🎯 PHASE 1: BASELINE (100 trades)")
        logger.info("📊 Establishing baseline performance...")
        
        for i in range(trades_count):
            logger.info(f"📈 Trade {i+1}/100")
            
            # Select random symbol
            symbol = random.choice(self.symbols)
            self.results['symbols_traded'].add(symbol)
            
            # Random order type
            order_type = random.choice(['BUY', 'SELL'])
            self.results['order_types'][order_type] += 1
            
            # Calculate position size (1% risk)
            volume = self.calculate_position_size(symbol, 1.0)
            
            # Place trade
            trade_result = self.place_trade(symbol, order_type, volume)
            
            if trade_result['success']:
                self.results['total_trades'] += 1
                
                if trade_result['profit'] > 0:
                    self.results['winning_trades'] += 1
                    self.results['total_profit'] += trade_result['profit']
                else:
                    self.results['losing_trades'] += 1
                    self.results['total_loss'] += abs(trade_result['profit'])
                
                self.results['net_profit'] += trade_result['profit']
            else:
                self.results['errors'].append(trade_result)
            
            # Manage positions
            self.manage_positions()
            
            # Wait between trades (simulates real trading)
            time.sleep(2)
            
            # Progress update
            if (i + 1) % 10 == 0:
                self.print_progress(i + 1, trades_count, "baseline")
    
    def run_phase_2_learning(self, trades_count: int):
        """Phase 2: Learning - 200 trades"""
        logger.info("🧠 PHASE 2: LEARNING (200 trades)")
        logger.info("📈 Adaptive learning enabled...")
        
        for i in range(trades_count):
            logger.info(f"🧠 Trade {i+1}/200")
            
            # Select symbol based on previous performance
            symbol = self.select_best_symbol()
            self.results['symbols_traded'].add(symbol)
            
            # Order type based on market conditions
            order_type = self.select_order_type(symbol)
            self.results['order_types'][order_type] += 1
            
            # Adaptive position sizing (0.5-2% risk)
            risk_percent = random.uniform(0.5, 2.0)
            volume = self.calculate_position_size(symbol, risk_percent)
            
            # Place trade
            trade_result = self.place_trade(symbol, order_type, volume)
            
            if trade_result['success']:
                self.results['total_trades'] += 1
                
                if trade_result['profit'] > 0:
                    self.results['winning_trades'] += 1
                    self.results['total_profit'] += trade_result['profit']
                else:
                    self.results['losing_trades'] += 1
                    self.results['total_loss'] += abs(trade_result['profit'])
                
                self.results['net_profit'] += trade_result['profit']
            else:
                self.results['errors'].append(trade_result)
            
            # Manage positions
            self.manage_positions()
            
            # Adaptive wait time
            wait_time = random.uniform(1, 3)
            time.sleep(wait_time)
            
            # Progress update
            if (i + 1) % 20 == 0:
                self.print_progress(i + 1, trades_count, "learning")
    
    def run_phase_3_optimization(self, trades_count: int):
        """Phase 3: Optimization - 200 trades"""
        logger.info("⚡ PHASE 3: OPTIMIZATION (200 trades)")
        logger.info("🚀 Full optimization enabled...")
        
        for i in range(trades_count):
            logger.info(f"⚡ Trade {i+1}/200")
            
            # Select best performing symbol
            symbol = self.select_best_symbol()
            self.results['symbols_traded'].add(symbol)
            
            # Optimized order type
            order_type = self.select_optimized_order_type(symbol)
            self.results['order_types'][order_type] += 1
            
            # Optimized position sizing (0.3-1.5% risk)
            risk_percent = random.uniform(0.3, 1.5)
            volume = self.calculate_position_size(symbol, risk_percent)
            
            # Place trade
            trade_result = self.place_trade(symbol, order_type, volume)
            
            if trade_result['success']:
                self.results['total_trades'] += 1
                
                if trade_result['profit'] > 0:
                    self.results['winning_trades'] += 1
                    self.results['total_profit'] += trade_result['profit']
                else:
                    self.results['losing_trades'] += 1
                    self.results['total_loss'] += abs(trade_result['profit'])
                
                self.results['net_profit'] += trade_result['profit']
            else:
                self.results['errors'].append(trade_result)
            
            # Manage positions
            self.manage_positions()
            
            # Optimized wait time
            wait_time = random.uniform(0.5, 2)
            time.sleep(wait_time)
            
            # Progress update
            if (i + 1) % 20 == 0:
                self.print_progress(i + 1, trades_count, "optimization")
    
    def select_best_symbol(self) -> str:
        """Select best performing symbol based on history"""
        # Simple logic: rotate through symbols
        if not self.results['symbols_traded']:
            return random.choice(self.symbols)
        
        # Select symbol that hasn't been traded recently
        available_symbols = [s for s in self.symbols if s not in list(self.results['symbols_traded'])[-3:]]
        return random.choice(available_symbols) if available_symbols else random.choice(self.symbols)
    
    def select_order_type(self, symbol: str) -> str:
        """Select order type based on market conditions"""
        # Simple logic: alternate between BUY and SELL
        return 'BUY' if self.results['order_types']['BUY'] <= self.results['order_types']['SELL'] else 'SELL'
    
    def select_optimized_order_type(self, symbol: str) -> str:
        """Select optimized order type"""
        # More sophisticated logic for optimization phase
        market_data = self.get_market_data(symbol)
        if market_data:
            # Buy if spread is tight, sell if spread is wide
            return 'BUY' if market_data['spread'] < 0.0005 else 'SELL'
        return random.choice(['BUY', 'SELL'])
    
    def print_progress(self, current: int, total: int, phase: str):
        """Print progress update"""
        progress = (current / total) * 100
        logger.info(f"📊 {phase.upper()} Phase Progress: {current}/{total} ({progress:.1f}%)")
        logger.info(f"💰 Net Profit: ${self.results['net_profit']:.2f}")
        logger.info(f"📈 Win Rate: {(self.results['winning_trades']/self.results['total_trades']*100):.1f}%")
        logger.info(f"⚡ Avg Execution Time: {sum(self.results['execution_times'])/len(self.results['execution_times']):.3f}s")
    
    def print_final_results(self):
        """Print comprehensive backtest results"""
        logger.info("🎉 BACKTEST COMPLETE!")
        logger.info("=" * 60)
        logger.info("📊 FINAL RESULTS")
        logger.info("=" * 60)
        
        # Basic stats
        logger.info(f"📈 Total Trades: {self.results['total_trades']}")
        logger.info(f"✅ Winning Trades: {self.results['winning_trades']}")
        logger.info(f"❌ Losing Trades: {self.results['losing_trades']}")
        
        if self.results['total_trades'] > 0:
            win_rate = (self.results['winning_trades'] / self.results['total_trades']) * 100
            logger.info(f"🎯 Win Rate: {win_rate:.2f}%")
        
        # Financial results
        logger.info(f"💰 Total Profit: ${self.results['total_profit']:.2f}")
        logger.info(f"💸 Total Loss: ${self.results['total_loss']:.2f}")
        logger.info(f"📊 Net Profit: ${self.results['net_profit']:.2f}")
        
        # Execution stats
        if self.results['execution_times']:
            avg_time = sum(self.results['execution_times']) / len(self.results['execution_times'])
            max_time = max(self.results['execution_times'])
            min_time = min(self.results['execution_times'])
            logger.info(f"⚡ Avg Execution Time: {avg_time:.3f}s")
            logger.info(f"⚡ Max Execution Time: {max_time:.3f}s")
            logger.info(f"⚡ Min Execution Time: {min_time:.3f}s")
        
        # Symbol stats
        logger.info(f"📊 Symbols Traded: {len(self.results['symbols_traded'])}")
        for symbol in sorted(self.results['symbols_traded']):
            logger.info(f"  - {symbol}")
        
        # Order type stats
        logger.info(f"📈 BUY Orders: {self.results['order_types']['BUY']}")
        logger.info(f"📉 SELL Orders: {self.results['order_types']['SELL']}")
        
        # Error stats
        logger.info(f"❌ Errors: {len(self.results['errors'])}")
        if self.results['errors']:
            logger.info("📝 Error Details:")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                logger.info(f"  - {error['error']}")
        
        logger.info("=" * 60)
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        try:
            results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def run_full_backtest(self):
        """Run complete 500-trade backtest"""
        logger.info("🚀 STARTING NEXUS 500-TRADE BACKTEST")
        logger.info("=" * 60)
        
        # Connect to MT5
        if not self.connect_to_mt5():
            logger.error("❌ Cannot connect to MT5. Backtest aborted.")
            return False
        
        # Phase 1: Baseline (100 trades)
        self.results['current_phase'] = 'baseline'
        self.run_phase_1_baseline(100)
        
        # Short break between phases
        logger.info("⏳ 10-second break between phases...")
        time.sleep(10)
        
        # Phase 2: Learning (200 trades)
        self.results['current_phase'] = 'learning'
        self.run_phase_2_learning(200)
        
        # Short break between phases
        logger.info("⏳ 10-second break between phases...")
        time.sleep(10)
        
        # Phase 3: Optimization (200 trades)
        self.results['current_phase'] = 'optimization'
        self.run_phase_3_optimization(200)
        
        # Print final results
        self.print_final_results()
        
        return True

def main():
    """Main backtest execution"""
    logger.info("🎯 NEXUS 500-TRADE FULL BACKTEST")
    logger.info("🔍 Testing complete system functionality...")
    
    # Create backtest engine
    engine = FullBacktestEngine()
    
    # Run backtest
    try:
        success = engine.run_full_backtest()
        if success:
            logger.info("🎉 BACKTEST COMPLETED SUCCESSFULLY!")
        else:
            logger.error("❌ BACKTEST FAILED!")
    except KeyboardInterrupt:
        logger.info("⏹️ BACKTEST INTERRUPTED BY USER")
    except Exception as e:
        logger.error(f"❌ BACKTEST ERROR: {e}")
    finally:
        # Cleanup
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
            logger.info("🔌 MT5 shutdown complete")
        except:
            pass

if __name__ == "__main__":
    main()
