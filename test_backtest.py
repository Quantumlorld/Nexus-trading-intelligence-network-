#!/usr/bin/env python3
"""
Test backtesting functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_simple_backtest():
    """Test simple backtesting functionality"""
    
    print("🚀 TESTING BACKTESTING FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Import required modules
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Load sample data
        print("📊 Loading sample data...")
        data_path = Path("data/raw/XAUUSD_daily.csv")
        
        if not data_path.exists():
            print("❌ Sample data not found. Run create_sample_data.py first!")
            return False
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(df)} records for XAUUSD")
        
        # Simple EMA crossover strategy test
        print("\n🔍 Testing EMA Crossover Strategy...")
        
        # Calculate EMAs
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['ema_12'] > df['ema_26'], 'signal'] = 1  # Buy
        df.loc[df['ema_12'] < df['ema_26'], 'signal'] = -1  # Sell
        
        # Find crossovers
        df['position'] = df['signal'].shift(1)
        df['crossover'] = df['signal'].diff()
        
        # Backtest parameters
        initial_capital = 10000.0
        risk_per_trade = 0.01  # 1%
        sl_points = 3.0  # $3
        tp_points = 9.9  # $9.9
        
        # Simulate trades
        trades = []
        capital = initial_capital
        position = None
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_signal = df['signal'].iloc[i]
            
            # Check for entry signals
            if current_signal == 1 and position is None:  # Buy signal
                position_size = (capital * risk_per_trade) / sl_points
                entry_price = current_price
                stop_loss = entry_price - sl_points
                take_profit = entry_price + tp_points
                
                position = {
                    'type': 'BUY',
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_date': df.index[i]
                }
                
            elif current_signal == -1 and position is None:  # Sell signal
                position_size = (capital * risk_per_trade) / sl_points
                entry_price = current_price
                stop_loss = entry_price + sl_points
                take_profit = entry_price - tp_points
                
                position = {
                    'type': 'SELL',
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_date': df.index[i]
                }
            
            # Check for exit conditions
            if position is not None:
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                
                exit_price = None
                exit_reason = None
                
                if position['type'] == 'BUY':
                    if high >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'TP'
                    elif low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'SL'
                else:  # SELL
                    if low <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'TP'
                    elif high >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'SL'
                
                # Exit position
                if exit_price is not None:
                    pnl = 0
                    if position['type'] == 'BUY':
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']
                    
                    capital += pnl
                    
                    trades.append({
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i]
                    })
                    
                    position = None
        
        # Calculate performance metrics
        if trades:
            total_pnl = sum(trade['pnl'] for trade in trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
            
            print(f"\n📈 BACKTEST RESULTS:")
            print(f"✅ Total Trades: {len(trades)}")
            print(f"✅ Total P&L: ${total_pnl:.2f}")
            print(f"✅ Win Rate: {win_rate:.1f}%")
            print(f"✅ Winning Trades: {len(winning_trades)}")
            print(f"✅ Losing Trades: {len(losing_trades)}")
            print(f"✅ Average Win: ${avg_win:.2f}")
            print(f"✅ Average Loss: ${avg_loss:.2f}")
            print(f"✅ Profit Factor: {profit_factor:.2f}")
            print(f"✅ Final Capital: ${capital:.2f}")
            print(f"✅ Return: {(capital - initial_capital) / initial_capital * 100:.1f}%")
            
            # Show recent trades
            print(f"\n📋 RECENT TRADES:")
            for trade in trades[-5:]:
                print(f"  {trade['type']} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | P&L: ${trade['pnl']:.2f} | {trade['exit_reason']}")
            
            return True
        else:
            print("❌ No trades generated")
            return False
            
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    success = test_simple_backtest()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 BACKTEST TEST PASSED!")
        print("📊 System is ready for advanced backtesting!")
        print("\n📋 NEXT STEPS:")
        print("1. Run full system backtest: python main.py --backtest --start 2024-01-01 --end 2024-12-31")
        print("2. Test different strategies")
        print("3. Configure risk parameters")
        print("4. Start live trading when ready")
    else:
        print("❌ BACKTEST TEST FAILED!")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()
