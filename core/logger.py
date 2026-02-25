"""
Nexus Trading System - Professional Logger
Comprehensive logging system for trading operations, decisions, and audits
"""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys
import traceback


@dataclass
class TradeLogEntry:
    """Trade log entry for complete audit trail"""
    timestamp: str
    symbol: str
    action: str  # 'OPEN', 'CLOSE', 'MODIFY', 'SIGNAL'
    direction: Optional[str] = None  # 'buy'/'sell'
    size: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    pnl: Optional[float] = None
    strategy: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    regime: Optional[str] = None
    risk_score: Optional[float] = None
    session_time: Optional[str] = None
    candle_timeframe: Optional[str] = None
    ticket_id: Optional[int] = None
    notes: Optional[str] = None


class NexusLogger:
    """Professional logging system for Nexus Trading System"""
    
    def __init__(self, name: str = "nexus", log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logs directory
        self.logs_dir = Path("data/processed/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different loggers for different purposes
        self.setup_loggers()
        
        # Trade log storage
        self.trade_log_entries: list[TradeLogEntry] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'daily_stats': {}
        }
    
    def setup_loggers(self):
        """Setup multiple loggers for different purposes"""
        
        # Main system logger
        self.system_logger = logging.getLogger(f"{self.name}.system")
        self.system_logger.setLevel(self.log_level)
        
        # Trade logger
        self.trade_logger = logging.getLogger(f"{self.name}.trades")
        self.trade_logger.setLevel(self.log_level)
        
        # Risk logger
        self.risk_logger = logging.getLogger(f"{self.name}.risk")
        self.risk_logger.setLevel(self.log_level)
        
        # Performance logger
        self.perf_logger = logging.getLogger(f"{self.name}.performance")
        self.perf_logger.setLevel(self.log_level)
        
        # Error logger
        self.error_logger = logging.getLogger(f"{self.name}.errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # Setup handlers for all loggers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers"""
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler for system logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        self.system_logger.addHandler(console_handler)
        
        # File handlers
        today = datetime.now().strftime("%Y-%m-%d")
        
        # System log file
        system_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"system_{today}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        system_handler.setFormatter(detailed_formatter)
        self.system_logger.addHandler(system_handler)
        
        # Trade log file
        trade_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"trades_{today}.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        trade_handler.setFormatter(detailed_formatter)
        self.trade_logger.addHandler(trade_handler)
        
        # Risk log file
        risk_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"risk_{today}.log",
            maxBytes=5*1024*1024,
            backupCount=3
        )
        risk_handler.setFormatter(detailed_formatter)
        self.risk_logger.addHandler(risk_handler)
        
        # Performance log file
        perf_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"performance_{today}.log",
            maxBytes=5*1024*1024,
            backupCount=3
        )
        perf_handler.setFormatter(detailed_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"errors_{today}.log",
            maxBytes=5*1024*1024,
            backupCount=3
        )
        error_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_handler)
    
    def log_trade_signal(self, symbol: str, direction: str, confidence: float,
                        strategy: str, regime: str, risk_score: float,
                        candle_timeframe: str, reason: str):
        """Log a trading signal"""
        
        entry = TradeLogEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="SIGNAL",
            direction=direction,
            confidence=confidence,
            strategy=strategy,
            regime=regime,
            risk_score=risk_score,
            candle_timeframe=candle_timeframe,
            reason=reason
        )
        
        self.trade_log_entries.append(entry)
        
        message = (f"SIGNAL | {symbol} | {direction} | Conf:{confidence:.2f} | "
                  f"Strategy:{strategy} | Regime:{regime} | Risk:{risk_score:.1f} | "
                  f"TF:{candle_timeframe} | {reason}")
        
        self.trade_logger.info(message)
        self._save_trade_entry_json(entry)
    
    def log_trade_open(self, symbol: str, direction: str, size: float,
                       entry_price: float, sl_price: float, tp_price: float,
                       strategy: str, ticket_id: int, reason: str):
        """Log a trade opening"""
        
        entry = TradeLogEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="OPEN",
            direction=direction,
            size=size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            strategy=strategy,
            ticket_id=ticket_id,
            reason=reason,
            session_time=self._get_session_time()
        )
        
        self.trade_log_entries.append(entry)
        self.performance_metrics['total_trades'] += 1
        
        message = (f"OPEN | {symbol} | {direction} | Size:{size} | "
                  f"Entry:{entry_price} | SL:{sl_price} | TP:{tp_price} | "
                  f"Ticket:{ticket_id} | {reason}")
        
        self.trade_logger.info(message)
        self._save_trade_entry_json(entry)
    
    def log_trade_close(self, symbol: str, exit_price: float, pnl: float,
                        reason: str, ticket_id: Optional[int] = None):
        """Log a trade closing"""
        
        entry = TradeLogEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="CLOSE",
            exit_price=exit_price,
            pnl=pnl,
            ticket_id=ticket_id,
            reason=reason,
            session_time=self._get_session_time()
        )
        
        self.trade_log_entries.append(entry)
        
        # Update performance metrics
        self.performance_metrics['total_pnl'] += pnl
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        message = (f"CLOSE | {symbol} | Exit:{exit_price} | "
                  f"P&L:{pnl:+.2f} | Ticket:{ticket_id} | {reason}")
        
        self.trade_logger.info(message)
        self._save_trade_entry_json(entry)
    
    def log_trade_modify(self, symbol: str, sl_price: Optional[float] = None,
                        tp_price: Optional[float] = None, reason: str = "",
                        ticket_id: Optional[int] = None):
        """Log a trade modification (SL/TP adjustment)"""
        
        entry = TradeLogEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="MODIFY",
            sl_price=sl_price,
            tp_price=tp_price,
            ticket_id=ticket_id,
            reason=reason
        )
        
        self.trade_log_entries.append(entry)
        
        modify_parts = []
        if sl_price is not None:
            modify_parts.append(f"SL:{sl_price}")
        if tp_price is not None:
            modify_parts.append(f"TP:{tp_price}")
        
        modify_str = " | ".join(modify_parts) if modify_parts else "Unknown"
        
        message = (f"MODIFY | {symbol} | {modify_str} | "
                  f"Ticket:{ticket_id} | {reason}")
        
        self.trade_logger.info(message)
        self._save_trade_entry_json(entry)
    
    def log_risk_assessment(self, symbol: str, risk_score: float, decision: str,
                           reasons: list, recommendations: list):
        """Log risk assessment"""
        
        reasons_str = "; ".join(reasons) if reasons else "None"
        recs_str = "; ".join(recommendations) if recommendations else "None"
        
        message = (f"RISK_ASSESSMENT | {symbol} | Score:{risk_score:.1f} | "
                  f"Decision:{decision} | Reasons:{reasons_str} | "
                  f"Recommendations:{recs_str}")
        
        self.risk_logger.info(message)
    
    def log_performance_update(self, equity: float, daily_pnl: float,
                             drawdown: float, win_rate: float, sharpe: float):
        """Log performance metrics update"""
        
        self.performance_metrics.update({
            'current_equity': equity,
            'daily_pnl': daily_pnl,
            'max_drawdown': max(drawdown, self.performance_metrics['max_drawdown']),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe
        })
        
        message = (f"PERFORMANCE | Equity:{equity:.2f} | Daily P&L:{daily_pnl:+.2f} | "
                  f"DD:{drawdown:.2f}% | WinRate:{win_rate:.1f}% | Sharpe:{sharpe:.2f}")
        
        self.perf_logger.info(message)
    
    def log_system_event(self, level: str, event: str, details: str = ""):
        """Log system events"""
        
        log_method = getattr(self.system_logger, level.lower(), self.system_logger.info)
        
        message = f"SYSTEM | {event}"
        if details:
            message += f" | {details}"
        
        log_method(message)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with full traceback"""
        
        error_details = f"{type(error).__name__}: {str(error)}"
        if context:
            error_details = f"{context} | {error_details}"
        
        # Get full traceback
        tb_str = traceback.format_exc()
        
        message = f"ERROR | {error_details}\n{tb_str}"
        self.error_logger.error(message)
        
        # Also log to system logger
        self.system_logger.error(f"ERROR | {error_details}")
    
    def log_kill_switch(self, reason: str, daily_stats: dict):
        """Log kill switch activation"""
        
        stats_str = json.dumps(daily_stats, indent=2)
        
        message = f"KILL_SWITCH_ACTIVATED | Reason:{reason}\nDaily Stats:\n{stats_str}"
        
        self.risk_logger.critical(message)
        self.system_logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    def _save_trade_entry_json(self, entry: TradeLogEntry):
        """Save trade entry to JSON file for audit trail"""
        
        try:
            json_file = self.logs_dir / f"trade_audit_{datetime.now().strftime('%Y-%m-%d')}.json"
            
            # Convert entry to dict and save
            entry_dict = asdict(entry)
            
            # Append to JSON array file
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                data.append(entry_dict)
            else:
                data = [entry_dict]
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.error_logger.error(f"Failed to save trade entry JSON: {e}")
    
    def _get_session_time(self) -> str:
        """Get current session time"""
        
        hour = datetime.now().hour
        
        if 8 <= hour < 13:
            return "LONDON_MORNING"
        elif 13 <= hour < 17:
            return "LONDON_AFTERNOON"
        elif 17 <= hour < 22:
            return "NEW_YORK_SESSION"
        else:
            return "ASIAN/OFF_SESSION"
    
    def get_trade_log(self, symbol: Optional[str] = None, 
                     action: Optional[str] = None,
                     limit: int = 100) -> list[TradeLogEntry]:
        """Get filtered trade log entries"""
        
        filtered_entries = self.trade_log_entries
        
        if symbol:
            filtered_entries = [e for e in filtered_entries if e.symbol == symbol]
        
        if action:
            filtered_entries = [e for e in filtered_entries if e.action == action]
        
        # Return most recent entries
        return filtered_entries[-limit:] if filtered_entries else []
    
    def export_trade_log_csv(self, filepath: str):
        """Export trade log to CSV"""
        
        try:
            import pandas as pd
            
            # Convert entries to DataFrame
            data = [asdict(entry) for entry in self.trade_log_entries]
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            self.system_logger.info(f"Trade log exported to {filepath}")
            
        except Exception as e:
            self.error_logger.error(f"Failed to export trade log CSV: {e}")
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report"""
        
        total_trades = self.performance_metrics['total_trades']
        
        if total_trades > 0:
            win_rate = (self.performance_metrics['winning_trades'] / total_trades) * 100
            avg_win = self.performance_metrics['total_pnl'] / max(self.performance_metrics['winning_trades'], 1)
            avg_loss = self.performance_metrics['total_pnl'] / max(self.performance_metrics['losing_trades'], 1)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.performance_metrics['winning_trades'],
            'losing_trades': self.performance_metrics['losing_trades'],
            'win_rate': win_rate,
            'total_pnl': self.performance_metrics['total_pnl'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'current_equity': self.performance_metrics.get('current_equity', 0),
            'daily_pnl': self.performance_metrics.get('daily_pnl', 0)
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.logs_dir.glob("*.log"):
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    log_file.unlink()
                    self.system_logger.info(f"Deleted old log file: {log_file}")
            
            for json_file in self.logs_dir.glob("*.json"):
                file_date = datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_date < cutoff_date:
                    json_file.unlink()
                    self.system_logger.info(f"Deleted old JSON file: {json_file}")
                    
        except Exception as e:
            self.error_logger.error(f"Failed to cleanup old logs: {e}")


# Global logger instance
_nexus_logger = None

def get_logger(name: str = "nexus") -> NexusLogger:
    """Get global logger instance"""
    global _nexus_logger
    if _nexus_logger is None:
        _nexus_logger = NexusLogger(name)
    return _nexus_logger

def setup_logging(log_level: str = "INFO"):
    """Setup global logging configuration"""
    global _nexus_logger
    _nexus_logger = NexusLogger("nexus", log_level)
    return _nexus_logger
