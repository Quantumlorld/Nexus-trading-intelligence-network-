"""
Nexus Trading System - Volatility Monitor
Real-time volatility analysis and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from database.session import get_database_session
from database.models import VolatilityMetric

from core.logger import get_logger
from core.regime_detector import RegimeDetector, MarketRegime


class VolatilityRegime(Enum):
    """Volatility regime types"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNSTABLE = "unstable"


@dataclass
class VolatilityMetrics:
    """Volatility metrics for a symbol"""
    symbol: str
    current_volatility: float
    volatility_percentile: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    regime: VolatilityRegime
    atr: float
    atr_percentile: float
    realized_vol_20d: float
    realized_vol_50d: float
    garch_volatility: float
    volatility_ratio: float  # Current vs historical average
    volatility_spike: bool
    volatility_contraction: bool
    last_updated: datetime
    confidence_level: float  # Confidence in volatility estimate


@dataclass
class VolatilityAlert:
    """Volatility alert information"""
    symbol: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    regime_change: bool = False
    previous_regime: Optional[str] = None


class VolatilityMonitor:
    """
    Advanced volatility monitoring system with regime detection
    Tracks volatility patterns, detects regime changes, and provides alerts
    """
    
    def __init__(self, db_path: str = "monitoring/volatility.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Database connection
        self.conn = None
        self._init_database()
        
        # Volatility tracking
        self.volatility_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.current_metrics: Dict[str, VolatilityMetrics] = {}
        self.regime_history: Dict[str, List[Tuple[datetime, MarketRegime]]] = {}
        
        # Alert system
        self.alerts: List[VolatilityAlert] = []
        self.alert_thresholds = {
            'volatility_spike': 2.0,  # 2x historical average
            'volatility_contraction': 0.5,  # 50% of historical average
            'extreme_volatility': 0.05,  # 5% daily volatility
            'regime_change': True
        }
        
        # Configuration
        self.update_interval = timedelta(minutes=5)
        self.history_window = timedelta(days=252)  # 1 year
        self.min_history_points = 20
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_running = False
        
        # Regime detector
        self.regime_detector = RegimeDetector()
        
        self.logger.info("Volatility monitor initialized")
    
    def _init_database(self):
        """Initialize database for volatility data"""
        # Database is handled by SQLAlchemy models
        self.logger.info("Volatility monitor initialized")
    
    def start(self):
        """Start volatility monitoring"""
        
        if self.is_running:
            self.logger.warning("Volatility monitor is already running")
            return
        
        self.is_running = True
        
        # Start background monitoring
        self.executor.submit(self._monitoring_loop)
        
        self.logger.info("Volatility monitor started")
    
    def stop(self):
        """Stop volatility monitoring"""
        
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close database connection
        if self.conn:
            self.conn.close()
        
        self.logger.info("Volatility monitor stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.is_running:
            try:
                # Update volatility for all tracked symbols
                symbols = list(self.volatility_history.keys())
                
                for symbol in symbols:
                    self._update_symbol_volatility(symbol)
                
                # Check for alerts
                self._check_volatility_alerts()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep until next update
                import time
                time.sleep(self.update_interval.total_seconds())
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def add_symbol(self, symbol: str, price_data: pd.DataFrame):
        """Add a symbol for volatility monitoring"""
        
        try:
            # Calculate initial volatility
            volatility = self._calculate_realized_volatility(price_data)
            
            with self._lock:
                # Initialize history
                self.volatility_history[symbol] = [(datetime.now(), volatility)]
                self.regime_history[symbol] = []
                
                # Calculate initial metrics
                self._calculate_volatility_metrics(symbol, price_data)
            
            self.logger.info(f"Added symbol to volatility monitoring: {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error adding symbol {symbol}: {e}")
    
    def update_symbol_data(self, symbol: str, price_data: pd.DataFrame):
        """Update price data for a symbol"""
        
        try:
            # Calculate new volatility
            volatility = self._calculate_realized_volatility(price_data)
            
            with self._lock:
                # Update history
                if symbol not in self.volatility_history:
                    self.volatility_history[symbol] = []
                
                self.volatility_history[symbol].append((datetime.now(), volatility))
                
                # Keep history manageable
                if len(self.volatility_history[symbol]) > 10000:
                    self.volatility_history[symbol] = self.volatility_history[symbol][-5000]
                
                # Update metrics
                self._calculate_volatility_metrics(symbol, price_data)
                
                # Detect regime changes
                self._detect_regime_change(symbol)
        
        except Exception as e:
            self.logger.error(f"Error updating symbol data {symbol}: {e}")
    
    def _calculate_realized_volatility(self, price_data: pd.DataFrame, window: int = 20) -> float:
        """Calculate realized volatility"""
        
        if len(price_data) < window:
            return 0.0
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < window:
            return 0.0
        
        # Calculate rolling volatility
        volatility = returns.tail(window).std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    def _calculate_volatility_metrics(self, symbol: str, price_data: pd.DataFrame):
        """Calculate comprehensive volatility metrics"""
        
        try:
            # Get historical volatility data
            if symbol not in self.volatility_history or len(self.volatility_history[symbol]) < self.min_history_points:
                return
            
            volatility_data = self.volatility_history[symbol]
            timestamps, volatilities = zip(*volatility_data)
            
            current_volatility = volatilities[-1]
            
            # Calculate percentiles
            historical_volatilities = [v for _, v in volatility_data]
            volatility_percentile = np.percentile(historical_volatilities, current_volatility * 100)
            
            # Calculate trend
            if len(volatilities) >= 10:
                recent_vol = np.mean(volatilities[-5:])
                older_vol = np.mean(volatilities[-10:-5])
                
                if recent_vol > older_vol * 1.1:
                    volatility_trend = 'increasing'
                elif recent_vol < older_vol * 0.9:
                    volatility_trend = 'decreasing'
                else:
                    volatility_trend = 'stable'
            else:
                volatility_trend = 'stable'
            
            # Determine regime
            regime = self._determine_volatility_regime(current_volatility, historical_volatilities)
            
            # Calculate ATR
            atr = self._calculate_atr(price_data)
            atr_percentile = np.percentile([self._calculate_atr(price_data.tail(i)) for i in range(10, min(50, len(price_data)))], atr * 100) if len(price_data) >= 50 else 50
            
            # Calculate different volatility measures
            realized_vol_20d = self._calculate_realized_volatility(price_data, 20)
            realized_vol_50d = self._calculate_realized_volatility(price_data, 50)
            garch_vol = self._calculate_garch_volatility(price_data)
            
            # Calculate volatility ratio
            historical_avg = np.mean(historical_volatilities)
            volatility_ratio = current_volatility / historical_avg if historical_avg > 0 else 1.0
            
            # Detect spikes and contractions
            volatility_spike = current_volatility > historical_avg * self.alert_thresholds['volatility_spike']
            volatility_contraction = current_volatility < historical_avg * self.alert_thresholds['volatility_contraction']
            
            # Calculate confidence level
            confidence_level = min(len(historical_volatilities) / 252, 1.0)  # More data = higher confidence
            
            # Create metrics object
            metrics = VolatilityMetrics(
                symbol=symbol,
                current_volatility=current_volatility,
                volatility_percentile=volatility_percentile,
                volatility_trend=volatility_trend,
                regime=regime,
                atr=atr,
                atr_percentile=atr_percentile,
                realized_vol_20d=realized_vol_20d,
                realized_vol_50d=realized_vol_50d,
                garch_volatility=garch_vol,
                volatility_ratio=volatility_ratio,
                volatility_spike=volatility_spike,
                volatility_contraction=volatility_contraction,
                last_updated=datetime.now(),
                confidence_level=confidence_level
            )
            
            self.current_metrics[symbol] = metrics
            
            # Save to database
            self._save_volatility_snapshot(metrics)
        
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
    
    def _determine_volatility_regime(self, current_vol: float, historical_vols: List[float]) -> VolatilityRegime:
        """Determine volatility regime"""
        
        if not historical_vols:
            return VolatilityRegime.NORMAL
        
        historical_avg = np.mean(historical_vols)
        historical_std = np.std(historical_vols)
        
        # Define regime thresholds
        if current_vol > historical_avg + 2 * historical_std:
            return VolatilityRegime.EXTREME
        elif current_vol > historical_avg + historical_std:
            return VolatilityRegime.HIGH
        elif current_vol < historical_avg - historical_std:
            return VolatilityRegime.LOW
        elif abs(current_vol - historical_avg) / historical_std < 0.5:
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.UNSTABLE
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        
        if len(price_data) < period:
            return 0.0
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # True range calculations
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _calculate_garch_volatility(self, price_data: pd.DataFrame) -> float:
        """Simple GARCH(1,1) volatility estimation"""
        
        try:
            returns = price_data['close'].pct_change().dropna()
            
            if len(returns) < 50:
                return 0.0
            
            # Simple GARCH(1,1) estimation
            # This is a simplified version - in production, use proper GARCH models
            
            # Calculate squared returns
            squared_returns = returns ** 2
            
            # Estimate parameters (simplified)
            omega = np.var(squared_returns) * 0.1
            alpha = np.corr(squared_returns[:-1], squared_returns[1:]) * 0.85
            beta = 1 - alpha
            
            # Calculate conditional variance
            conditional_var = omega + alpha * squared_returns.iloc[-1]**2 + beta * np.var(squared_returns)
            
            return np.sqrt(conditional_var) * np.sqrt(252)  # Annualized
        
        except Exception as e:
            self.logger.error(f"Error calculating GARCH volatility: {e}")
            return 0.0
    
    def _detect_regime_change(self, symbol: str):
        """Detect volatility regime changes"""
        
        if symbol not in self.current_metrics or symbol not in self.regime_history:
            return
        
        current_metrics = self.current_metrics[symbol]
        current_regime = current_metrics.regime
        
        # Get recent regime history
        regime_history = self.regime_history.get(symbol, [])
        
        if regime_history:
            last_regime = regime_history[-1][1]
            
            # Check for regime change
            if last_regime != current_regime:
                # Create alert
                alert = VolatilityAlert(
                    symbol=symbol,
                    alert_type='regime_change',
                    severity='medium',
                    message=f"Volatility regime changed from {last_regime.value} to {current_regime.value}",
                    current_value=current_metrics.current_volatility,
                    threshold_value=0.0,
                    timestamp=datetime.now(),
                    regime_change=True,
                    previous_regime=last_regime.value
                )
                
                self.alerts.append(alert)
                self._save_volatility_alert(alert)
                
                # Save regime change
                self._save_regime_change(symbol, last_regime, current_regime, current_metrics)
                
                self.logger.info(f"Regime change detected for {symbol}: {last_regime.value} -> {current_regime.value}")
        
        # Update regime history
        self.regime_history[symbol].append((datetime.now(), current_regime))
        
        # Keep history manageable
        if len(self.regime_history[symbol]) > 1000:
            self.regime_history[symbol] = self.regime_history[symbol][-500]
    
    def _check_volatility_alerts(self):
        """Check for volatility alerts"""
        
        for symbol, metrics in self.current_metrics.items():
            # Check for volatility spike
            if metrics.volatility_spike:
                alert = VolatilityAlert(
                    symbol=symbol,
                    alert_type='volatility_spike',
                    severity='high',
                    message=f"Volatility spike detected: {metrics.current_volatility:.3f} ({metrics.volatility_ratio:.1f}x historical average)",
                    current_value=metrics.current_volatility,
                    threshold_value=np.mean([v for _, v in self.volatility_history.get(symbol, [])]) * self.alert_thresholds['volatility_spike'],
                    timestamp=datetime.now()
                )
                
                self.alerts.append(alert)
                self._save_volatility_alert(alert)
            
            # Check for volatility contraction
            if metrics.volatility_contraction:
                alert = VolatilityAlert(
                    symbol=symbol,
                    alert_type='volatility_contraction',
                    severity='medium',
                    message=f"Volatility contraction detected: {metrics.current_volatility:.3f} ({metrics.volatility_ratio:.1f}x historical average)",
                    current_value=metrics.current_volatility,
                    threshold_value=np.mean([v for _, v in self.volatility_history.get(symbol, [])]) * self.alert_thresholds['volatility_contraction'],
                    timestamp=datetime.now()
                )
                
                self.alerts.append(alert)
                self._save_volatility_alert(alert)
            
            # Check for extreme volatility
            if metrics.regime == VolatilityRegime.EXTREME:
                alert = VolatilityAlert(
                    symbol=symbol,
                    alert_type='extreme_volatility',
                    severity='critical',
                    message=f"Extreme volatility detected: {metrics.current_volatility:.3f}",
                    current_value=metrics.current_volatility,
                    threshold_value=0.05,
                    timestamp=datetime.now()
                )
                
                self.alerts.append(alert)
                self._save_volatility_alert(alert)
        
        # Keep alerts manageable
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500]
    
    def _save_volatility_snapshot(self, metrics: VolatilityMetrics):
        """Save volatility snapshot to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO volatility_snapshots (
                timestamp, symbol, current_volatility, volatility_percentile,
                volatility_trend, regime, atr, atr_percentile, realized_vol_20d,
                realized_vol_50d, garch_volatility, volatility_ratio,
                volatility_spike, volatility_contraction, confidence_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.last_updated.isoformat(),
            metrics.symbol,
            metrics.current_volatility,
            metrics.volatility_percentile,
            metrics.volatility_trend,
            metrics.regime.value,
            metrics.atr,
            metrics.atr_percentile,
            metrics.realized_vol_20d,
            metrics.realized_vol_50d,
            metrics.garch_volatility,
            metrics.volatility_ratio,
            metrics.volatility_spike,
            metrics.volatility_contraction,
            metrics.confidence_level
        ))
        
        self.conn.commit()
    
    def _save_volatility_alert(self, alert: VolatilityAlert):
        """Save volatility alert to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO volatility_alerts (
                timestamp, symbol, alert_type, severity, message,
                current_value, threshold_value, regime_change, previous_regime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.timestamp.isoformat(),
            alert.symbol,
            alert.alert_type,
            alert.severity,
            alert.message,
            alert.current_value,
            alert.threshold_value,
            alert.regime_change,
            alert.previous_regime
        ))
        
        self.conn.commit()
    
    def _save_regime_change(self, symbol: str, old_regime: MarketRegime, 
                            new_regime: MarketRegime, metrics: VolatilityMetrics):
        """Save regime change to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO regime_changes (
                timestamp, symbol, old_regime, new_regime, confidence,
                volatility_level, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            old_regime.value,
            new_regime.value,
            metrics.confidence_level,
            metrics.current_volatility,
            f"Volatility regime changed from {old_regime.value} to {new_regime.value}"
        ))
        
        self.conn.commit()
    
    def get_current_metrics(self, symbol: Optional[str] = None) -> Dict[str, VolatilityMetrics]:
        """Get current volatility metrics"""
        
        with self._lock:
            if symbol:
                return {symbol: self.current_metrics.get(symbol)} if symbol in self.current_metrics else {}
            else:
                return self.current_metrics.copy()
    
    def get_volatility_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get volatility history for a symbol"""
        
        if symbol not in self.volatility_history:
            return pd.DataFrame(columns=['timestamp', 'volatility'])
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            filtered_history = [(ts, vol) for ts, vol in self.volatility_history[symbol] if ts >= cutoff_date]
        
        if not filtered_history:
            return pd.DataFrame(columns=['timestamp', 'volatility'])
        
        df = pd.DataFrame(filtered_history, columns=['timestamp', 'volatility'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_regime_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get regime history for a symbol"""
        
        if symbol not in self.regime_history:
            return pd.DataFrame(columns=['timestamp', 'regime'])
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            filtered_history = [(ts, regime) for ts, regime in self.regime_history[symbol] if ts >= cutoff_date]
        
        if not filtered_history:
            return pd.DataFrame(columns=['timestamp', 'regime'])
        
        df = pd.DataFrame(filtered_history, columns=['timestamp', 'regime'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_alerts(self, symbol: Optional[str] = None, severity: Optional[str] = None, 
                    hours: int = 24) -> List[VolatilityAlert]:
        """Get volatility alerts"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = []
        
        for alert in self.alerts:
            if alert.timestamp < cutoff_time:
                continue
            
            if symbol and alert.symbol != symbol:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_volatility_summary(self) -> Dict[str, Any]:
        """Get volatility monitoring summary"""
        
        with self._lock:
            total_symbols = len(self.current_metrics)
            
            regime_counts = {}
            for metrics in self.current_metrics.values():
                regime = metrics.regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            recent_alerts = self.get_alerts(hours=1)
            
            return {
                'monitored_symbols': total_symbols,
                'regime_distribution': regime_counts,
                'recent_alerts_count': len(recent_alerts),
                'alert_severity_distribution': {
                    'low': len([a for a in recent_alerts if a.severity == 'low']),
                    'medium': len([a for a in recent_alerts if a.severity == 'medium']),
                    'high': len([a for a in recent_alerts if a.severity == 'high']),
                    'critical': len([a for a in recent_alerts if a.severity == 'critical'])
                },
                'last_update': max([m.last_updated for m in self.current_metrics.values()]) if self.current_metrics else None
            }
    
    def export_volatility_data(self, filepath: str, format: str = 'csv'):
        """Export volatility data to file"""
        
        try:
            if format.lower() == 'csv':
                self._export_to_csv(filepath)
            elif format.lower() == 'json':
                self._export_to_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting volatility data: {e}")
    
    def _export_to_csv(self, filepath: str):
        """Export volatility data to CSV"""
        
        import pandas as pd
        
        # Export metrics
        metrics_data = []
        for symbol, metrics in self.current_metrics.items():
            metrics_data.append({
                'symbol': symbol,
                'current_volatility': metrics.current_volatility,
                'volatility_percentile': metrics.volatility_percentile,
                'volatility_trend': metrics.volatility_trend,
                'regime': metrics.regime.value,
                'atr': metrics.atr,
                'realized_vol_20d': metrics.realized_vol_20d,
                'realized_vol_50d': metrics.realized_vol_50d,
                'volatility_ratio': metrics.volatility_ratio,
                'volatility_spike': metrics.volatility_spike,
                'volatility_contraction': metrics.volatility_contraction,
                'confidence_level': metrics.confidence_level,
                'last_updated': metrics.last_updated.isoformat()
            })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(filepath.replace('.csv', '_metrics.csv'), index=False)
        
        self.logger.info(f"Volatility data exported to CSV: {filepath}")
    
    def _export_to_json(self, filepath: str):
        """Export volatility data to JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_metrics': {symbol: {
                'current_volatility': m.current_volatility,
                'volatility_percentile': m.volatility_percentile,
                'volatility_trend': m.volatility_trend,
                'regime': m.regime.value,
                'atr': m.atr,
                'realized_vol_20d': m.realized_vol_20d,
                'realized_vol_50d': m.realized_vol_50d,
                'volatility_ratio': m.volatility_ratio,
                'volatility_spike': m.volatility_spike,
                'volatility_contraction': m.volatility_contraction,
                'confidence_level': m.confidence_level,
                'last_updated': m.last_updated.isoformat()
            } for symbol, m in self.current_metrics.items()},
            'alert_thresholds': self.alert_thresholds,
            'total_alerts': len(self.alerts),
            'monitored_symbols': len(self.current_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Volatility data exported to JSON: {filepath}")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        
        try:
            # Delete data older than 1 year
            cutoff_date = datetime.now() - timedelta(days=365)
            
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM volatility_snapshots WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM volatility_alerts WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM regime_changes WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            self.conn.commit()
            
            deleted_rows = cursor.rowcount
            if deleted_rows > 0:
                self.logger.info(f"Cleaned up {deleted_rows} old volatility records")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        
        return {
            'is_running': self.is_running,
            'monitored_symbols': len(self.current_metrics),
            'total_alerts': len(self.alerts),
            'alert_thresholds': self.alert_thresholds,
            'update_interval': self.update_interval.total_seconds(),
            'database_path': str(self.db_path),
            'regime_detector_enabled': True
        }
