"""
Nexus Trading System - Performance Tracker
Real-time performance monitoring and analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from database.session import get_database_session
from database.models import PerformanceMetric

from core.logger import get_logger
from core.trade_manager import TradeManager, Position
from strategy.base_strategy import BaseStrategy


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    current_equity: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trading metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Position metrics
    open_positions: int = 0
    total_exposure: float = 0.0
    margin_used: float = 0.0
    
    # Strategy metrics
    strategy_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Time metrics
    last_trade_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    # Compliance metrics
    daily_loss_limit_compliance: float = 100.0
    consecutive_losses_compliance: float = 100.0


class PerformanceTracker:
    """
    Advanced performance tracking system for real-time monitoring
    Tracks all trading activities, calculates metrics, and provides analysis
    """
    
    def __init__(self, db_path: str = "monitoring/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Database connection
        self.conn = None
        self._init_database()
        
        # Performance state
        self.metrics = PerformanceMetrics()
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_running = False
        
        # Update frequency
        self.update_interval = timedelta(seconds=30)
        self.last_update = datetime.now()
        
        self.logger.info("Performance tracker initialized")
    
    def _init_database(self):
        """Initialize database for performance data"""
        # Database is handled by SQLAlchemy models
        self.logger.info("Performance tracker initialized")
    
    def start(self):
        """Start performance tracking"""
        
        if self.is_running:
            self.logger.warning("Performance tracker is already running")
            return
        
        self.is_running = True
        
        # Start background monitoring
        self.executor.submit(self._monitoring_loop)
        
        self.logger.info("Performance tracker started")
    
    def stop(self):
        """Stop performance tracking"""
        
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close database connection
        if self.conn:
            self.conn.close()
        
        self.logger.info("Performance tracker stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.is_running:
            try:
                # Update metrics
                self._update_metrics()
                
                # Save snapshot
                self._save_equity_snapshot()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep until next update
                time.sleep(self.update_interval.total_seconds())
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade event"""
        
        try:
            with self._lock:
                # Add to history
                self.trade_history.append(trade_data)
                
                # Keep history manageable
                if len(self.trade_history) > 10000:
                    self.trade_history = self.trade_history[-5000]
                
                # Save to database
                self._save_trade_to_db(trade_data)
                
                # Update metrics
                self._update_trade_metrics(trade_data)
            
            self.logger.debug(f"Trade recorded: {trade_data.get('symbol')} {trade_data.get('action')}")
        
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def _save_trade_to_db(self, trade_data: Dict[str, Any]):
        """Save trade data to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, action, direction, size, entry_price, exit_price,
                sl_price, tp_price, pnl, strategy, confidence, reason, session_time,
                candle_timeframe, ticket_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('timestamp', datetime.now().isoformat()),
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('direction'),
            trade_data.get('size'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('sl_price'),
            trade_data.get('tp_price'),
            trade_data.get('pnl'),
            trade_data.get('strategy'),
            trade_data.get('confidence'),
            trade_data.get('reason'),
            trade_data.get('session_time'),
            trade_data.get('candle_timeframe'),
            trade_data.get('ticket_id'),
            trade_data.get('notes')
        ))
        
        self.conn.commit()
    
    def _update_trade_metrics(self, trade_data: Dict[str, Any]):
        """Update trade-related metrics"""
        
        if trade_data.get('action') == 'CLOSE':
            self.metrics.total_trades += 1
            
            pnl = trade_data.get('pnl', 0)
            self.metrics.total_pnl += pnl
            
            if pnl > 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1
            
            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.total_trades) * 100
            
            # Update average win/loss
            if self.metrics.winning_trades > 0:
                winning_pnls = [t['pnl'] for t in self.trade_history 
                               if t.get('action') == 'CLOSE' and t.get('pnl', 0) > 0]
                self.metrics.avg_win = np.mean(winning_pnls)
            
            if self.metrics.losing_trades > 0:
                losing_pnls = [t['pnl'] for t in self.trade_history 
                              if t.get('action') == 'CLOSE' and t.get('pnl', 0) < 0]
                self.metrics.avg_loss = np.mean(losing_pnls)
            
            # Update profit factor
            total_wins = sum(t['pnl'] for t in self.trade_history 
                            if t.get('action') == 'CLOSE' and t.get('pnl', 0) > 0)
            total_losses = abs(sum(t['pnl'] for t in self.trade_history 
                               if t.get('action') == 'CLOSE' and t.get('pnl', 0) < 0))
            
            self.metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Update last trade time
            self.metrics.last_trade_time = datetime.fromisoformat(trade_data['timestamp'])
    
    def update_equity(self, equity: float, balance: float = None, margin_used: float = 0.0):
        """Update current equity"""
        
        try:
            with self._lock:
                self.metrics.current_equity = equity
                
                # Add to history
                self.equity_history.append((datetime.now(), equity))
                
                # Keep history manageable
                if len(self.equity_history) > 10000:
                    self.equity_history = self.equity_history[-5000]
                
                # Update drawdown
                self._update_drawdown()
                
                # Update risk ratios
                self._update_risk_ratios()
            
            self.logger.debug(f"Equity updated: {equity:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error updating equity: {e}")
    
    def _update_drawdown(self):
        """Update drawdown calculations"""
        
        if len(self.equity_history) < 2:
            return
        
        # Get equity series
        equity_series = [eq for _, eq in self.equity_history]
        
        # Calculate running maximum
        running_max = []
        current_max = equity_series[0]
        
        for equity in equity_series:
            if equity > current_max:
                current_max = equity
            running_max.append(current_max)
        
        # Calculate drawdown
        drawdowns = []
        for i, equity in enumerate(equity_series):
            drawdown = (equity - running_max[i]) / running_max[i] * 100
            drawdowns.append(drawdown)
        
        # Update metrics
        if drawdowns:
            self.metrics.current_drawdown = drawdowns[-1]
            self.metrics.max_drawdown = min(drawdowns)
    
    def _update_risk_ratios(self):
        """Update risk-adjusted performance ratios"""
        
        if len(self.equity_history) < 2:
            return
        
        # Calculate returns
        equity_series = [eq for _, eq in self.equity_history]
        returns = pd.Series(equity_series).pct_change().dropna()
        
        if len(returns) == 0:
            return
        
        # Sharpe ratio
        if returns.std() > 0:
            excess_returns = returns.mean() - 0.02 / 252  # Assuming 2% risk-free rate
            self.metrics.sharpe_ratio = excess_returns / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            excess_returns = returns.mean() - 0.02 / 252
            self.metrics.sortino_ratio = excess_returns / downside_returns.std() * np.sqrt(252)
    
    def update_positions(self, positions: List[Dict[str, Any]]):
        """Update position information"""
        
        try:
            with self._lock:
                self.metrics.open_positions = len(positions)
                
                # Calculate total exposure
                total_exposure = 0.0
                for pos in positions:
                    size = pos.get('size', 0)
                    current_price = pos.get('current_price', 0)
                    exposure = size * current_price * 100000  # Rough estimate
                    total_exposure += abs(exposure)
                
                self.metrics.total_exposure = total_exposure
                self.metrics.margin_used = sum(pos.get('margin', 0) for pos in positions)
            
            self.logger.debug(f"Positions updated: {len(positions)} open, exposure: {total_exposure:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def update_strategy_performance(self, strategy_name: str, performance_data: Dict[str, Any]):
        """Update strategy-specific performance"""
        
        try:
            with self._lock:
                if strategy_name not in self.metrics.strategy_performance:
                    self.metrics.strategy_performance[strategy_name] = {
                        'total_signals': 0,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'avg_confidence': 0.0,
                        'avg_trade_duration': 0.0
                    }
                
                # Update strategy metrics
                strategy_metrics = self.metrics.strategy_performance[strategy_name]
                
                for key, value in performance_data.items():
                    if key in strategy_metrics:
                        strategy_metrics[key] = value
                
                # Save to database
                self._save_strategy_performance_to_db(strategy_name, strategy_metrics)
            
            self.logger.debug(f"Strategy performance updated: {strategy_name}")
        
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def _save_strategy_performance_to_db(self, strategy_name: str, metrics: Dict[str, Any]):
        """Save strategy performance to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO strategy_performance (
                timestamp, strategy_name, total_signals, total_trades, winning_trades,
                losing_trades, total_pnl, win_rate, avg_confidence, avg_trade_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            strategy_name,
            metrics.get('total_signals', 0),
            metrics.get('total_trades', 0),
            metrics.get('winning_trades', 0),
            metrics.get('losing_trades', 0),
            metrics.get('total_pnl', 0.0),
            metrics.get('win_rate', 0.0),
            metrics.get('avg_confidence', 0.0),
            metrics.get('avg_trade_duration', 0.0)
        ))
        
        self.conn.commit()
    
    def _update_metrics(self):
        """Update all performance metrics"""
        
        try:
            # Update compliance metrics
            self._update_compliance_metrics()
            
            # Update last update time
            self.metrics.last_update_time = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _update_compliance_metrics(self):
        """Update compliance metrics"""
        
        try:
            # Daily loss limit compliance
            today = datetime.now().date()
            today_pnl = sum(t['pnl'] for t in self.trade_history 
                            if t.get('action') == 'CLOSE' and 
                            datetime.fromisoformat(t['timestamp']).date() == today)
            
            if today_pnl < -9.99:
                self.metrics.daily_loss_limit_compliance = max(0, 100 + (today_pnl + 9.99) / 9.99 * 100)
            else:
                self.metrics.daily_loss_limit_compliance = 100.0
            
            # Consecutive losses compliance
            consecutive_losses = self._calculate_consecutive_losses()
            if consecutive_losses <= 3:
                self.metrics.consecutive_losses_compliance = 100.0
            else:
                self.metrics.consecutive_losses_compliance = max(0, 100 - (consecutive_losses - 3) / 3 * 100)
        
        except Exception as e:
            self.logger.error(f"Error updating compliance metrics: {e}")
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate current consecutive losses"""
        
        consecutive = 0
        max_consecutive = 0
        
        # Sort trades by timestamp
        sorted_trades = sorted(self.trade_history, 
                              key=lambda x: datetime.fromisoformat(x['timestamp']) if 'timestamp' in x else datetime.min,
                              reverse=True)
        
        for trade in sorted_trades:
            if trade.get('action') == 'CLOSE':
                if trade.get('pnl', 0) < 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break
        
        return max_consecutive
    
    def _save_equity_snapshot(self):
        """Save current equity snapshot to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO equity_snapshots (
                timestamp, equity, balance, margin_used, open_positions, total_exposure,
                drawdown, sharpe_ratio, win_rate, total_trades, daily_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            self.metrics.current_equity,
            self.metrics.current_equity,  # Balance = equity for now
            self.metrics.margin_used,
            self.metrics.open_positions,
            self.metrics.total_exposure,
            self.metrics.current_drawdown,
            self.metrics.sharpe_ratio,
            self.metrics.win_rate,
            self.metrics.total_trades,
            self._calculate_daily_pnl()
        ))
        
        self.conn.commit()
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        
        today = datetime.now().date()
        
        return sum(t['pnl'] for t in self.trade_history 
                   if t.get('action') == 'CLOSE' and 
                   datetime.fromisoformat(t['timestamp']).date() == today)
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        
        try:
            # Delete trades older than 1 year
            cutoff_date = datetime.now() - timedelta(days=365)
            
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM equity_snapshots WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM strategy_performance WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            self.conn.commit()
            
            deleted_rows = cursor.rowcount
            if deleted_rows > 0:
                self.logger.info(f"Cleaned up {deleted_rows} old records")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        
        with self._lock:
            return PerformanceMetrics(
                total_trades=self.metrics.total_trades,
                winning_trades=self.metrics.winning_trades,
                losing_trades=self.metrics.losing_trades,
                win_rate=self.metrics.win_rate,
                total_pnl=self.metrics.total_pnl,
                current_equity=self.metrics.current_equity,
                max_drawdown=self.metrics.max_drawdown,
                current_drawdown=self.metrics.current_drawdown,
                sharpe_ratio=self.metrics.sharpe_ratio,
                sortino_ratio=self.metrics.sortino_ratio,
                avg_win=self.metrics.avg_win,
                avg_loss=self.metrics.avg_loss,
                profit_factor=self.metrics.profit_factor,
                open_positions=self.metrics.open_positions,
                total_exposure=self.metrics.total_exposure,
                margin_used=self.metrics.margin_used,
                strategy_performance=self.metrics.strategy_performance.copy(),
                last_trade_time=self.metrics.last_trade_time,
                last_update_time=self.metrics.last_update_time,
                daily_loss_limit_compliance=self.metrics.daily_loss_limit_compliance,
                consecutive_losses_compliance=self.metrics.consecutive_losses_compliance
            )
    
    def get_equity_curve(self, days: int = 30) -> pd.DataFrame:
        """Get equity curve for specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            filtered_history = [(ts, eq) for ts, eq in self.equity_history if ts >= cutoff_date]
        
        if not filtered_history:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        df = pd.DataFrame(filtered_history, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_trade_history(self, days: int = 30, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get trade history for specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            filtered_trades = [t for t in self.trade_history 
                             if datetime.fromisoformat(t['timestamp']) >= cutoff_date]
            
            if symbol:
                filtered_trades = [t for t in filtered_trades if t.get('symbol') == symbol]
        
        if not filtered_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered_trades)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy performance summary"""
        
        with self._lock:
            if strategy_name:
                return self.metrics.strategy_performance.get(strategy_name, {})
            else:
                return self.metrics.strategy_performance.copy()
    
    def export_performance_data(self, filepath: str, format: str = 'csv'):
        """Export performance data to file"""
        
        try:
            if format.lower() == 'csv':
                self._export_to_csv(filepath)
            elif format.lower() == 'json':
                self._export_to_json(filepath)
            elif format.lower() == 'excel':
                self._export_to_excel(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
    
    def _export_to_csv(self, filepath: str):
        """Export performance data to CSV files"""
        
        import pandas as pd
        
        # Export trades
        trades_df = self.get_trade_history(days=365)
        trades_df.to_csv(filepath.replace('.csv', '_trades.csv'), index=False)
        
        # Export equity curve
        equity_df = self.get_equity_curve(days=365)
        equity_df.to_csv(filepath.replace('.csv', '_equity.csv'))
        
        # Export strategy performance
        strategy_data = []
        for strategy, metrics in self.metrics.strategy_performance.items():
            strategy_data.append({
                'strategy': strategy,
                **metrics
            })
        
        if strategy_data:
            strategy_df = pd.DataFrame(strategy_data)
            strategy_df.to_csv(filepath.replace('.csv', '_strategies.csv'), index=False)
        
        self.logger.info(f"Performance data exported to CSV files")
    
    def _export_to_json(self, filepath: str):
        """Export performance data to JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_metrics': self.get_current_metrics().__dict__,
            'strategy_performance': self.metrics.strategy_performance,
            'equity_history': [(ts.isoformat(), eq) for ts, eq in self.equity_history[-1000:]],  # Last 1000 entries
            'total_trades': len(self.trade_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Performance data exported to JSON: {filepath}")
    
    def _export_to_excel(self, filepath: str):
        """Export performance data to Excel file"""
        
        try:
            import pandas as pd
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Trades sheet
                trades_df = self.get_trade_history(days=365)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Equity curve sheet
                equity_df = self.get_equity_curve(days=365)
                equity_df.to_excel(writer, sheet_name='Equity Curve')
                
                # Strategy performance sheet
                strategy_data = []
                for strategy, metrics in self.metrics.strategy_performance.items():
                    strategy_data.append({
                        'Strategy': strategy,
                        **metrics
                    })
                
                if strategy_data:
                    strategy_df = pd.DataFrame(strategy_data)
                    strategy_df.to_excel(writer, sheet_name='Strategy Performance', index=False)
            
            self.logger.info(f"Performance data exported to Excel: {filepath}")
        
        except ImportError:
            self.logger.error("Excel export requires openpyxl package")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        metrics = self.get_current_metrics()
        
        report = f"""
# Nexus Trading System - Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- **Current Equity**: ${metrics.current_equity:,.2f}
- **Total P&L**: ${metrics.total_pnl:,.2f}
- **Total Trades**: {metrics.total_trades}
- **Win Rate**: {metrics.win_rate:.1f}%
- **Profit Factor**: {metrics.profit_factor:.2f}

## Risk Metrics
- **Maximum Drawdown**: {metrics.max_drawdown:.2f}%
- **Current Drawdown**: {metrics.current_drawdown:.2f}%
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Sortino Ratio**: {metrics.sortino_ratio:.2f}

## Trading Statistics
- **Winning Trades**: {metrics.winning_trades}
- **Losing Trades**: {metrics.losing_trades}
- **Average Win**: ${metrics.avg_win:.2f}
- **Average Loss**: ${metrics.avg_loss:.2f}
- **Open Positions**: {metrics.open_positions}
- **Total Exposure**: ${metrics.total_exposure:,.2f}

## Compliance Metrics
- **Daily Loss Limit Compliance**: {metrics.daily_loss_limit_compliance:.1f}%
- **Consecutive Losses Compliance**: {metrics.consecutive_losses_compliance:.1f}%

## Strategy Performance
"""
        
        for strategy_name, strategy_metrics in metrics.strategy_performance.items():
            report += f"""
### {strategy_name}
- Total Signals: {strategy_metrics.get('total_signals', 0)}
- Total Trades: {strategy_metrics.get('total_trades', 0)}
- Win Rate: {strategy_metrics.get('win_rate', 0):.1f}%
- Total P&L: ${strategy_metrics.get('total_pnl', 0):.2f}
"""
        
        return report
    
    def reset_performance_data(self):
        """Reset all performance data"""
        
        with self._lock:
            # Reset metrics
            self.metrics = PerformanceMetrics()
            
            # Clear history
            self.equity_history.clear()
            self.trade_history.clear()
            
            # Clear database
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM trades')
            cursor.execute('DELETE FROM equity_snapshots')
            cursor.execute('DELETE FROM strategy_performance')
            self.conn.commit()
        
        self.logger.info("Performance data reset")
    
    def get_tracker_status(self) -> Dict[str, Any]:
        """Get current tracker status"""
        
        return {
            'is_running': self.is_running,
            'last_update': self.metrics.last_update_time,
            'total_trades_recorded': len(self.trade_history),
            'equity_history_points': len(self.equity_history),
            'strategies_tracked': len(self.metrics.strategy_performance),
            'database_path': str(self.db_path),
            'update_interval': self.update_interval.total_seconds()
        }
