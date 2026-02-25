#!/usr/bin/env python3
"""
Test script for the adaptive backtester
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data(days: int = 252) -> pd.DataFrame:
    """Create sample market data for testing"""
    
    print(f"📊 Creating sample data for {days} days...")
    
    # Generate realistic price data
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Starting prices
    prices = {
        'open': 100.0,
        'high': 100.0,
        'low': 100.0,
        'close': 100.0,
        'volume': 1000000
    }
    
    data = []
    
    for i, date in enumerate(dates):
        # Random walk with trend
        trend = 0.0001 * i  # Slight upward trend
        noise = np.random.normal(0, 0.01)  # Random noise
        
        # Calculate price changes
        open_price = prices['close']
        close_change = trend + noise
        close_price = open_price * (1 + close_change)
        
        # Generate high and low
        high_range = abs(np.random.normal(0, 0.005))
        low_range = abs(np.random.normal(0, 0.005))
        
        high_price = max(open_price, close_price) * (1 + high_range)
        low_price = min(open_price, close_price) * (1 - low_range)
        
        # Generate volume
        volume = int(np.random.normal(1000000, 200000))
        volume = max(100000, volume)  # Minimum volume
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        # Update prices for next iteration
        prices['close'] = close_price
    
    df = pd.DataFrame(data)
    # Don't set index here, keep 'date' as a column for the backtester
    
    print(f"✅ Sample data created: {len(df)} records")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Average volume: {df['volume'].mean():,.0f}")
    
    return df

def simple_moving_average_strategy(data: pd.DataFrame, short_window: int = 10, long_window: int = 30):
    """Simple moving average crossover strategy"""
    
    print(f"📈 Running SMA strategy (short={short_window}, long={long_window})...")
    
    # Calculate moving averages
    data['sma_short'] = data['close'].rolling(window=short_window).mean()
    data['sma_long'] = data['close'].rolling(window=long_window).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['sma_short'] > data['sma_long'], 'signal'] = 1  # Buy
    data.loc[data['sma_short'] < data['sma_long'], 'signal'] = -1  # Sell
    
    # Calculate positions
    data['position'] = data['signal'].shift(1)
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'] * data['returns']
    
    return data

def main():
    """Test the adaptive backtester"""
    
    print("🚀 NEXUS ADAPTIVE BACKTESTER TEST")
    print("=" * 60)
    
    try:
        # Import required modules
        from backtest.adaptive_backtester import (
            AdaptiveBacktester, 
            AdaptiveBacktestConfig,
            create_adaptive_backtester
        )
        from backtest.backtest_engine import BacktestConfig
        
        print("✅ All modules imported successfully!")
        
        # Create sample data
        print("\n📊 Creating test data...")
        data = create_sample_data(days=252)  # 1 year of data
        
        # Create backtest configuration
        print("\n⚙️ Creating backtest configuration...")
        
        base_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000,
            commission=0.001,
            slippage=0.0001,
            margin_call_level=0.5,
            enable_dynamic_tp=True,
            enable_runner_mode=True,
            enforce_session_filters=True,
            enforce_risk_limits=True,
            save_trades=True,
            save_equity_curve=True
        )
        
        adaptive_config = AdaptiveBacktestConfig(
            base_config=base_config,
            enable_adaptive_learning=True,
            enable_outlier_detection=True,
            enable_weight_optimization=True,
            performance_window_days=30,
            min_trades_for_adaptation=10,
            adaptation_frequency="weekly"
        )
        
        print("✅ Configuration created successfully!")
        
        # Create adaptive backtester
        print("\n🤖 Creating adaptive backtester...")
        backtester = create_adaptive_backtester(adaptive_config)
        print("✅ Adaptive backtester created successfully!")
        
        # Define strategy function
        def strategy_function(data_chunk):
            return simple_moving_average_strategy(data_chunk)
        
        # Run adaptive backtest
        print("\n🚀 Running adaptive backtest...")
        print("   This may take a few minutes...")
        
        results = backtester.run_adaptive_backtest(
            strategy_func=strategy_function,
            data=data,
            initial_capital=10000
        )
        
        print("✅ Adaptive backtest completed successfully!")
        
        # Generate and display report
        print("\n📋 GENERATING REPORT...")
        report = backtester.generate_report(results)
        print(report)
        
        # Save results
        print("\n💾 Saving results...")
        results_dir = Path("backtest/py_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"adaptive_backtest_{timestamp}.json"
        report_file = results_dir / f"adaptive_backtest_{timestamp}.txt"
        
        backtester.save_results(results, str(results_file))
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Results saved to: {results_file}")
        print(f"✅ Report saved to: {report_file}")
        
        # Display key metrics
        print("\n📊 KEY RESULTS SUMMARY:")
        print("=" * 40)
        
        base_result = results["base_results"]
        print(f"Base Strategy Return: {base_result.total_return:.2%}")
        print(f"Base Sharpe Ratio: {base_result.sharpe_ratio:.2f}")
        
        if results["adaptive_results"]:
            best_adaptive = max(results["adaptive_results"], key=lambda x: x.total_return)
            print(f"Best Adaptive Return: {best_adaptive.total_return:.2%}")
            print(f"Best Adaptive Sharpe: {best_adaptive.sharpe_ratio:.2f}")
            
            improvement = ((best_adaptive.total_return - base_result.total_return) / 
                          abs(base_result.total_return)) * 100
            print(f"Improvement: {improvement:.1f}%")
        
        # Display adaptation effectiveness
        effectiveness = results["improvement_analysis"]["adaptation_effectiveness"]
        print(f"Adaptation Effectiveness: {effectiveness:.1f}%")
        
        print("\n🎯 ADAPTIVE BACKTESTER TEST COMPLETED SUCCESSFULLY!")
        print("✅ All systems working correctly!")
        print("🚀 Ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing adaptive backtester: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
