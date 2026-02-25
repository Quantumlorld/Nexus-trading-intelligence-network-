#!/usr/bin/env python3
"""
Create sample market data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_data(symbol="XAUUSD", days=365):
    """Create sample market data"""
    
    print(f"Creating sample data for {symbol}...")
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible data
    
    # Starting price based on symbol
    if symbol == "XAUUSD":
        base_price = 2000.0
        volatility = 0.02
    elif symbol == "BTCUSD":
        base_price = 50000.0
        volatility = 0.05
    elif symbol == "EURUSD":
        base_price = 1.1000
        volatility = 0.01
    else:
        base_price = 100.0
        volatility = 0.02
    
    # Generate price movements
    returns = np.random.normal(0, volatility, len(dates))
    prices = [base_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = prices[1:]  # Remove initial price
    
    # Create OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday variation
        daily_vol = volatility * close * 0.5
        
        high = close + abs(np.random.normal(0, daily_vol))
        low = close - abs(np.random.normal(0, daily_vol))
        
        # Ensure high >= close >= low
        high = max(high, close)
        low = min(low, close)
        
        # Generate open (close of previous day with some variation)
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.normal(0, daily_vol * 0.3)
        
        # Generate volume
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def main():
    """Create sample data files"""
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data for different symbols
    symbols = ["XAUUSD", "BTCUSD", "EURUSD", "USDX"]
    
    for symbol in symbols:
        df = create_sample_data(symbol, days=365)
        
        # Save to CSV
        file_path = data_dir / f"{symbol}_daily.csv"
        df.to_csv(file_path)
        print(f"✅ Created {file_path} with {len(df)} records")
        
        # Show sample
        print(f"\nSample data for {symbol}:")
        print(df.head(3))
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"Average daily change: {df['close'].pct_change().mean()*100:.2f}%")
        print("-" * 50)
    
    print(f"\n🎉 Sample data created successfully!")
    print(f"📁 Location: {data_dir}")
    print(f"📊 Ready for backtesting!")

if __name__ == "__main__":
    main()
