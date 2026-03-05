"""
Nexus Trading System - 9-Hour Candle Engine
Full implementation with proper aggregation and timestamp alignment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, func
from decimal import Decimal
import pytz

from database.session import get_database_session
from database.ledger_models import CandleData

logger = logging.getLogger(__name__)

class CandleEngine9H:
    """9-hour candle aggregation engine"""
    
    def __init__(self):
        self.utc = pytz.UTC
        self.candle_intervals = {
            '9h': timedelta(hours=9),
            '1h': timedelta(hours=1),
            '5m': timedelta(minutes=5)
        }
        
        # 9H candle start times (UTC)
        self.nine_hour_starts = [0, 9, 18]  # 00:00, 09:00, 18:00 UTC
    
    def get_next_candle_start(self, current_time: datetime) -> datetime:
        """Get the next 9H candle start time"""
        utc_time = current_time.astimezone(self.utc)
        hour = utc_time.hour
        
        # Find the next start time
        for start_hour in sorted(self.nine_hour_starts):
            if hour < start_hour:
                return utc_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        # If we're past all start times, go to next day's first start
        next_day = utc_time + timedelta(days=1)
        return next_day.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def get_current_candle_start(self, current_time: datetime) -> datetime:
        """Get the current 9H candle start time"""
        utc_time = current_time.astimezone(self.utc)
        hour = utc_time.hour
        
        # Find the most recent start time
        for start_hour in sorted(self.nine_hour_starts, reverse=True):
            if hour >= start_hour:
                return utc_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        # If we're before the first start time, use previous day's last start
        previous_day = utc_time - timedelta(days=1)
        return previous_day.replace(hour=18, minute=0, second=0, microsecond=0)
    
    def get_candle_period(self, candle_start: datetime) -> Tuple[datetime, datetime]:
        """Get the start and end time for a 9H candle period"""
        end_time = candle_start + timedelta(hours=9)
        return candle_start, end_time
    
    async def generate_9h_candles(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Generate 9H candles from base timeframe data"""
        try:
            logger.info(f"Generating 9H candles for {symbol} from {start_date} to {end_date}")
            
            with next(get_database_session()) as db:
                generated_candles = []
                
                # Start from the first 9H boundary
                current_candle_start = self.get_current_candle_start(start_date)
                
                while current_candle_start < end_date:
                    candle_end = current_candle_start + timedelta(hours=9)
                    
                    # Generate candle for this period
                    candle = await self._generate_single_9h_candle(
                        db, symbol, current_candle_start, candle_end
                    )
                    
                    if candle:
                        generated_candles.append(candle)
                        logger.debug(f"Generated 9H candle: {symbol} @ {candle['timestamp']} "
                                   f"O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}")
                    
                    # Move to next candle
                    current_candle_start = candle_end
                
                logger.info(f"Generated {len(generated_candles)} 9H candles for {symbol}")
                return generated_candles
                
        except Exception as e:
            logger.error(f"Failed to generate 9H candles: {e}")
            return []
    
    async def _generate_single_9h_candle(self, db: Session, symbol: str, 
                                      start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Generate a single 9H candle from base timeframe data"""
        try:
            # Try to get 1-hour candles first
            candle = await self._generate_from_1h(db, symbol, start_time, end_time)
            
            if not candle:
                # Fall back to 5-minute candles
                candle = await self._generate_from_5m(db, symbol, start_time, end_time)
            
            if candle:
                # Store in database
                await self._store_candle(db, candle)
                return candle
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate 9H candle for {symbol} at {start_time}: {e}")
            return None
    
    async def _generate_from_1h(self, db: Session, symbol: str, 
                              start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Generate 9H candle from 1-hour candles"""
        try:
            # Query 1-hour candles within the period
            query = text("""
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM candle_data 
                WHERE symbol = :symbol 
                AND timeframe = '1h'
                AND timestamp >= :start_time 
                AND timestamp < :end_time
                ORDER BY timestamp ASC
            """)
            
            result = db.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }).fetchall()
            
            if not result:
                return None
            
            # Aggregate to 9H candle
            return self._aggregate_candles(result, start_time, '9h', symbol)
            
        except Exception as e:
            logger.error(f"Failed to generate from 1h: {e}")
            return None
    
    async def _generate_from_5m(self, db: Session, symbol: str, 
                              start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Generate 9H candle from 5-minute candles"""
        try:
            # Query 5-minute candles within the period
            query = text("""
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM candle_data 
                WHERE symbol = :symbol 
                AND timeframe = '5m'
                AND timestamp >= :start_time 
                AND timestamp < :end_time
                ORDER BY timestamp ASC
            """)
            
            result = db.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }).fetchall()
            
            if not result:
                return None
            
            # Aggregate to 9H candle
            return self._aggregate_candles(result, start_time, '9h', symbol)
            
        except Exception as e:
            logger.error(f"Failed to generate from 5m: {e}")
            return None
    
    def _aggregate_candles(self, candles: List, timestamp: datetime, 
                          timeframe: str, symbol: str) -> Dict[str, Any]:
        """Aggregate multiple candles into a single candle"""
        try:
            if not candles:
                return None
            
            # Extract OHLCV data
            opens = [float(c[1]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) if c[5] else 0 for c in candles]
            
            # Calculate aggregated values
            open_price = opens[0]
            high_price = max(highs)
            low_price = min(lows)
            close_price = closes[-1]
            total_volume = sum(volumes)
            
            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'timeframe': timeframe,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': total_volume
            }
            
        except Exception as e:
            logger.error(f"Failed to aggregate candles: {e}")
            return None
    
    async def _store_candle(self, db: Session, candle: Dict[str, Any]):
        """Store candle in database"""
        try:
            # Check if candle already exists
            existing = db.query(CandleData).filter(
                and_(
                    CandleData.symbol == candle['symbol'],
                    CandleData.timeframe == candle['timeframe'],
                    CandleData.timestamp == candle['timestamp']
                )
            ).first()
            
            if existing:
                # Update existing candle
                existing.open = candle['open']
                existing.high = candle['high']
                existing.low = candle['low']
                existing.close = candle['close']
                existing.volume = candle['volume']
                existing.updated_at = datetime.utcnow()
            else:
                # Create new candle
                new_candle = CandleData(
                    symbol=candle['symbol'],
                    timeframe=candle['timeframe'],
                    timestamp=candle['timestamp'],
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume'],
                    created_at=datetime.utcnow()
                )
                db.add(new_candle)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store candle: {e}")
            db.rollback()
            raise
    
    async def get_9h_candles(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get 9H candles from database"""
        try:
            with next(get_database_session()) as db:
                query = text("""
                    SELECT 
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candle_data 
                    WHERE symbol = :symbol 
                    AND timeframe = '9h'
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = db.execute(query, {
                    'symbol': symbol,
                    'limit': limit
                }).fetchall()
                
                candles = []
                for row in result:
                    candles.append({
                        'timestamp': row[0],
                        'open': float(row[1]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'close': float(row[4]),
                        'volume': float(row[5]) if row[5] else 0
                    })
                
                # Return in chronological order
                return list(reversed(candles))
                
        except Exception as e:
            logger.error(f"Failed to get 9H candles: {e}")
            return []
    
    async def generate_sample_data(self, symbol: str, days: int = 7):
        """Generate sample 1H and 5M data for testing"""
        try:
            logger.info(f"Generating sample data for {symbol} for {days} days")
            
            with next(get_database_session()) as db:
                base_time = datetime.utcnow() - timedelta(days=days)
                current_time = base_time
                
                # Generate 1-hour candles
                for hour in range(days * 24):
                    candle_time = base_time + timedelta(hours=hour)
                    
                    # Generate realistic price movement
                    base_price = 1.1000 if symbol == 'EUR/USD' else 45000.0
                    price_variation = 0.01 * (hour / 24)  # Daily trend
                    random_noise = random.uniform(-0.002, 0.002)
                    
                    open_price = base_price + price_variation + random_noise
                    close_price = open_price + random.uniform(-0.001, 0.001)
                    high_price = max(open_price, close_price) + random.uniform(0, 0.001)
                    low_price = min(open_price, close_price) - random.uniform(0, 0.001)
                    volume = random.uniform(100, 1000)
                    
                    candle = CandleData(
                        symbol=symbol,
                        timeframe='1h',
                        timestamp=candle_time,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume,
                        created_at=datetime.utcnow()
                    )
                    
                    db.add(candle)
                
                db.commit()
                logger.info(f"Generated {days * 24} 1H candles for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to generate sample data: {e}")

# Global 9H candle engine instance
candle_engine_9h = CandleEngine9H()

# Import random for sample data generation
import random
