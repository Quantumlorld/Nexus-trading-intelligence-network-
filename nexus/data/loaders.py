"""
Nexus Trading System - Data Loaders
Handles loading, processing, and managing market data from various sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path

class DataLoader:
    """Base class for data loading operations"""
    
    def __init__(self, config_path: str = "config/assets.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict:
        """Load asset configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
    
    def load_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical data for a symbol and timeframe
        Returns DataFrame with OHLCV data
        """
        raise NotImplementedError("Subclasses must implement load_historical_data")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame has required columns and no obvious issues"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Found: {df.columns}")
            return False
            
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False
            
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] < 0).any().any():
            self.logger.error("Negative prices found")
            return False
            
        # Check for invalid OHLC relationships
        invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                      (df['high'] < df['close']) | (df['low'] > df['open']) | \
                      (df['low'] > df['close'])
        
        if invalid_ohlc.any():
            self.logger.error(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} rows")
            return False
            
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Forward fill missing values (limited)
        df = df.fillna(method='ffill', limit=5)
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        return df
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe"""
        timeframe_map = {
            '1M': '1T', '5M': '5T', '15M': '15T', '30M': '30T',
            '1H': '1H', '4H': '4H', 'D': '1D', 'W': '1W'
        }
        
        if target_timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
            
        resample_rule = timeframe_map[target_timeframe]
        
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = df.resample(resample_rule).agg(ohlc_dict)
        return resampled.dropna()


class MT5DataLoader(DataLoader):
    """Load data from MetaTrader 5"""
    
    def __init__(self, config_path: str = "config/assets.yaml"):
        super().__init__(config_path)
        self.mt5 = None
        self._connect_mt5()
    
    def _connect_mt5(self):
        """Connect to MT5 terminal"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            
            if not mt5.initialize():
                raise Exception("Failed to initialize MT5")
                
            self.logger.info("Connected to MT5")
            
        except ImportError:
            self.logger.error("MetaTrader5 package not installed")
            self.mt5 = None
    
    def load_historical_data(self, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load historical data from MT5"""
        if not self.mt5:
            raise Exception("MT5 not available")
            
        timeframe_map = {
            '1M': self.mt5.TIMEFRAME_M1,
            '5M': self.mt5.TIMEFRAME_M5,
            '15M': self.mt5.TIMEFRAME_M15,
            '30M': self.mt5.TIMEFRAME_M30,
            '1H': self.mt5.TIMEFRAME_H1,
            '4H': self.mt5.TIMEFRAME_H4,
            'D': self.mt5.TIMEFRAME_D1,
            'W': self.mt5.TIMEFRAME_W1
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        mt5_timeframe = timeframe_map[timeframe]
        
        # Get data from MT5
        rates = self.mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            raise Exception(f"No data retrieved for {symbol} {timeframe}")
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to standard format
        df.rename(columns={
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        # Select only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        if not self.validate_data(df):
            raise Exception("Data validation failed")
            
        return self.clean_data(df)


class CSVDataLoader(DataLoader):
    """Load data from CSV files"""
    
    def __init__(self, data_dir: str = "data/raw", config_path: str = "config/assets.yaml"):
        super().__init__(config_path)
        self.data_dir = Path(data_dir)
        
    def load_historical_data(self, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load historical data from CSV file"""
        csv_path = self.data_dir / f"{symbol}_{timeframe}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Load CSV
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Filter date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if not self.validate_data(df):
            raise Exception("Data validation failed")
            
        return self.clean_data(df)


class DataManager:
    """Manages data loading, caching, and serving for the trading system"""
    
    def __init__(self, loader_type: str = "csv", **kwargs):
        self.loader_type = loader_type
        self.data_cache = {}
        
        if loader_type == "mt5":
            self.loader = MT5DataLoader(**kwargs)
        elif loader_type == "csv":
            self.loader = CSVDataLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
            
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, symbol: str, timeframe: str, 
                start_date: datetime, end_date: datetime,
                use_cache: bool = True) -> pd.DataFrame:
        """Get data with optional caching"""
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self.data_cache:
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key].copy()
        
        # Load fresh data
        df = self.loader.load_historical_data(symbol, timeframe, start_date, end_date)
        
        # Cache the data
        if use_cache:
            self.data_cache[cache_key] = df.copy()
            
        return df
    
    def get_latest_data(self, symbol: str, timeframe: str, 
                       periods: int = 100) -> pd.DataFrame:
        """Get latest N periods of data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods * 2)  # Buffer for weekends
        
        return self.get_data(symbol, timeframe, start_date, end_date)
    
    def clear_cache(self):
        """Clear data cache"""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def preload_data(self, symbols: List[str], timeframes: List[str],
                     start_date: datetime, end_date: datetime):
        """Preload data for multiple symbols and timeframes"""
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    self.get_data(symbol, timeframe, start_date, end_date)
                    self.logger.info(f"Preloaded {symbol} {timeframe}")
                except Exception as e:
                    self.logger.error(f"Failed to preload {symbol} {timeframe}: {e}")
