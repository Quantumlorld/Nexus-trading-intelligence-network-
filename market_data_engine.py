#!/usr/bin/env python3
"""
🎯 NEXUS MARKET DATA ENGINE
Senior Quantitative Trading Systems Engineer - Production Market Data System
"""

import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_data_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataEngine:
    """Production-ready market data engine for Nexus Trading System"""
    
    def __init__(self):
        self.mt5_available = False
        self.cache = {}
        self.cache_timeout = 60  # 60 seconds cache
        
        # Try to import MT5
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mt5_available = True
            logger.info("✅ MetaTrader5 module available")
        except ImportError:
            logger.error("❌ MetaTrader5 module not available")
            self.mt5 = None
        
        # Timeframe mappings
        self.timeframes = {
            'M1': self.mt5.TIMEFRAME_M1 if self.mt5_available else None,
            'M5': self.mt5.TIMEFRAME_M5 if self.mt5_available else None,
            'M15': self.mt5.TIMEFRAME_M15 if self.mt5_available else None,
            'H1': self.mt5.TIMEFRAME_H1 if self.mt5_available else None,
            'H4': self.mt5.TIMEFRAME_H4 if self.mt5_available else None,
            '9H': '9H'  # Custom timeframe
        }
        
        logger.info("🎯 Market Data Engine Initialized")
        logger.info(f"📊 MT5 Available: {self.mt5_available}")
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not self.mt5_available:
            logger.error("❌ MT5 not available - using simulation mode")
            return False
        
        try:
            if self.mt5.initialize():
                logger.info("✅ MT5 initialized for market data")
                return True
            else:
                error_code = self.mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                return False
        except Exception as e:
            logger.error(f"❌ MT5 initialization exception: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        cache_key = f"symbol_info_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < self.cache_timeout:
                return cached_data['data']
        
        if not self.mt5_available:
            return self._simulate_symbol_info(symbol)
        
        try:
            symbol_info = self.mt5.symbol_info(symbol)
            if symbol_info:
                data = {
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
                    'margin_required': symbol_info.margin_initial,
                    'swap_long': symbol_info.swap_long,
                    'swap_short': symbol_info.swap_short,
                    'swap_3day': symbol_info.swap_3day
                }
                
                # Cache the data
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                
                return data
            else:
                logger.warning(f"⚠️ Symbol {symbol} not found")
                return None
        except Exception as e:
            logger.error(f"❌ Get symbol info failed for {symbol}: {e}")
            return None
    
    def _simulate_symbol_info(self, symbol: str) -> Dict:
        """Simulate symbol info when MT5 is not available"""
        import random
        
        # Generate realistic simulated data
        base_price = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'USDJPY': 148.50,
            'XAUUSD': 2035.50,
            'BTCUSD': 67500.00
        }.get(symbol, 1.0000)
        
        spread = random.uniform(0.0001, 0.0003)
        
        return {
            'symbol': symbol,
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'spread': spread,
            'point': 0.00001,
            'digits': 5,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
            'trade_mode': 'Full',
            'currency_base': symbol[:3],
            'currency_profit': symbol[-3:],
            'margin_required': 1000.0,
            'swap_long': random.uniform(-0.5, 0.5),
            'swap_short': random.uniform(-0.5, 0.5),
            'swap_3day': random.uniform(-1.5, 1.5),
            'simulated': True
        }
    
    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """Get current tick data"""
        cache_key = f"tick_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 5:  # 5 second cache
                return cached_data['data']
        
        if not self.mt5_available:
            return self._simulate_tick_data(symbol)
        
        try:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick:
                data = {
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': datetime.fromtimestamp(tick.time),
                    'spread': tick.ask - tick.bid
                }
                
                # Cache the data
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                
                return data
            else:
                logger.warning(f"⚠️ No tick data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"❌ Get tick data failed for {symbol}: {e}")
            return None
    
    def _simulate_tick_data(self, symbol: str) -> Dict:
        """Simulate tick data when MT5 is not available"""
        import random
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return {}
        
        # Add small random movement
        movement = random.uniform(-0.0001, 0.0001)
        
        return {
            'symbol': symbol,
            'bid': symbol_info['bid'] + movement,
            'ask': symbol_info['ask'] + movement,
            'last': symbol_info['bid'] + movement,
            'volume': random.uniform(10, 100),
            'time': datetime.now(),
            'spread': symbol_info['spread'],
            'simulated': True
        }
    
    def get_candle_data(self, symbol: str, timeframe: str, count: int = 100) -> List[Dict]:
        """Get candle data for specified timeframe"""
        cache_key = f"candles_{symbol}_{timeframe}_{count}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < self.cache_timeout:
                return cached_data['data']
        
        if not self.mt5_available:
            return self._simulate_candle_data(symbol, timeframe, count)
        
        try:
            mt5_timeframe = self.timeframes.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"❌ Invalid timeframe: {timeframe}")
                return []
            
            if timeframe == '9H':
                # Create 9H candles by aggregating H1
                return self.create_9h_candles(symbol, count)
            
            rates = self.mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
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
                        'spread': rate.get('spread', 0),
                        'timeframe': timeframe
                    })
                
                logger.info(f"✅ Retrieved {len(candle_data)} {timeframe} candles for {symbol}")
                
                # Cache the data
                self.cache[cache_key] = {
                    'data': candle_data,
                    'timestamp': datetime.now()
                }
                
                return candle_data
            else:
                logger.warning(f"⚠️ No candle data for {symbol} {timeframe}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Get candle data failed for {symbol}: {e}")
            return []
    
    def _simulate_candle_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Simulate candle data when MT5 is not available"""
        import random
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return []
        
        # Timeframe configurations
        timeframe_minutes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'H1': 60,
            'H4': 240,
            '9H': 540
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        candles = []
        current_price = symbol_info['bid']
        
        for i in range(count):
            # Generate OHLC data
            open_price = current_price
            high_price = current_price + random.uniform(0, 0.001)
            low_price = current_price - random.uniform(0, 0.001)
            close_price = current_price + random.uniform(-0.0005, 0.0005)
            
            candle = {
                'time': datetime.now() - timedelta(minutes=minutes * (count - i)),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(50, 200),
                'spread': symbol_info['spread'],
                'timeframe': timeframe,
                'simulated': True
            }
            
            candles.append(candle)
            current_price = close_price
        
        # Reverse to have oldest first
        candles.reverse()
        
        logger.info(f"✅ Generated {len(candles)} simulated {timeframe} candles for {symbol}")
        return candles
    
    def create_9h_candles(self, symbol: str, count: int = 100) -> List[Dict]:
        """Create 9H candles by aggregating H1 candles"""
        logger.info(f"🔨 Creating 9H candles for {symbol}...")
        
        if not self.mt5_available:
            return self._simulate_candle_data(symbol, '9H', count)
        
        # Get H1 candles (need 9x more for aggregation)
        h1_candles = self.get_candle_data(symbol, 'H1', count * 9)
        
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
                    'spread': chunk[0]['spread'],
                    'timeframe': '9H',
                    'aggregated': True
                }
                
                candles_9h.append(candle_9h)
        
        logger.info(f"✅ Created {len(candles_9h)} 9H candles for {symbol}")
        return candles_9h
    
    def get_market_snapshot(self, symbols: List[str]) -> Dict:
        """Get market snapshot for multiple symbols"""
        snapshot = {}
        
        for symbol in symbols:
            symbol_data = {
                'symbol': symbol,
                'tick': self.get_tick_data(symbol),
                'info': self.get_symbol_info(symbol),
                'candles': {
                    'M1': self.get_candle_data(symbol, 'M1', 20),
                    'M5': self.get_candle_data(symbol, 'M5', 20),
                    'M15': self.get_candle_data(symbol, 'M15', 20),
                    'H1': self.get_candle_data(symbol, 'H1', 24),
                    'H4': self.get_candle_data(symbol, 'H4', 24),
                    '9H': self.get_candle_data(symbol, '9H', 20)
                }
            }
            snapshot[symbol] = symbol_data
        
        return snapshot
    
    def calculate_technical_indicators(self, symbol: str, timeframe: str, count: int = 100) -> Dict:
        """Calculate technical indicators"""
        candles = self.get_candle_data(symbol, timeframe, count)
        
        if len(candles) < 20:
            return {'error': 'Insufficient data for indicators'}
        
        # Convert to DataFrame for easier calculations
        df = pd.DataFrame(candles)
        
        # Calculate indicators
        indicators = {
            'symbol': symbol,
            'timeframe': timeframe,
            'sma_20': df['close'].rolling(window=20).mean().iloc[-1],
            'sma_50': df['close'].rolling(window=50).mean().iloc[-1],
            'ema_12': df['close'].ewm(span=12).mean().iloc[-1],
            'ema_26': df['close'].ewm(span=26).mean().iloc[-1],
            'rsi': self._calculate_rsi(df['close']),
            'macd': self._calculate_macd(df['close']),
            'bollinger': self._calculate_bollinger_bands(df['close']),
            'atr': self._calculate_atr(df),
            'volume_sma': df['volume'].rolling(window=20).mean().iloc[-1],
            'price_change': (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100,
            'high_24h': df['high'].rolling(window=24).max().iloc[-1],
            'low_24h': df['low'].rolling(window=24).min().iloc[-1]
        }
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1],
            'width': upper_band.iloc[-1] - lower_band.iloc[-1]
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def clear_cache(self):
        """Clear market data cache"""
        self.cache.clear()
        logger.info("🗑️ Market data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_timeout': self.cache_timeout
        }

def main():
    """Test market data engine"""
    logger.info("🎯 NEXUS MARKET DATA ENGINE TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create engine instance
    engine = MarketDataEngine()
    
    try:
        # Initialize MT5 if available
        engine.initialize_mt5()
        
        # Test symbols
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
        
        # Test symbol info
        logger.info("🔍 Testing symbol information...")
        for symbol in symbols:
            info = engine.get_symbol_info(symbol)
            if info:
                logger.info(f"✅ {symbol}: Bid={info['bid']:.5f}, Ask={info['ask']:.5f}")
        
        # Test tick data
        logger.info("🔍 Testing tick data...")
        tick_data = engine.get_tick_data("EURUSD")
        if tick_data:
            logger.info(f"✅ EURUSD Tick: {tick_data['bid']:.5f} / {tick_data['ask']:.5f}")
        
        # Test candle data
        logger.info("🔍 Testing candle data...")
        for timeframe in ['M1', 'M5', 'M15', 'H1', 'H4', '9H']:
            candles = engine.get_candle_data("EURUSD", timeframe, 10)
            if candles:
                logger.info(f"✅ {timeframe}: {len(candles)} candles")
        
        # Test technical indicators
        logger.info("🔍 Testing technical indicators...")
        indicators = engine.calculate_technical_indicators("EURUSD", "H1", 100)
        if 'error' not in indicators:
            logger.info(f"✅ RSI: {indicators['rsi']:.2f}")
            logger.info(f"✅ SMA 20: {indicators['sma_20']:.5f}")
            logger.info(f"✅ Price Change: {indicators['price_change']:.4f}%")
        
        # Test market snapshot
        logger.info("🔍 Testing market snapshot...")
        snapshot = engine.get_market_snapshot(["EURUSD", "GBPUSD"])
        logger.info(f"✅ Market snapshot: {len(snapshot)} symbols")
        
        # Cache stats
        cache_stats = engine.get_cache_stats()
        logger.info(f"📊 Cache stats: {cache_stats['cache_size']} items")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()
