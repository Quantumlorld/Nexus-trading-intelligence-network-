"""
Nexus Trading System - Operational Metrics
Prometheus-compatible metrics collection
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]

class OperationalMetrics:
    """Operational metrics collector"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 points
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Initialize key metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metric collectors"""
        # Counters
        self.counters['trades_total'] = 0
        self.counters['trades_successful'] = 0
        self.counters['trades_failed'] = 0
        self.counters['trades_timeout'] = 0
        self.counters['ledger_update_retries'] = 0
        self.counters['broker_failures'] = 0
        self.counters['db_failures'] = 0
        self.counters['reconciliation_jobs'] = 0
        
        # Gauges
        self.gauges['trading_enabled'] = 1.0
        self.gauges['broker_connected'] = 1.0
        self.gauges['db_connected'] = 1.0
        self.gauges['reconciliation_queue_size'] = 0.0
        self.gauges['active_positions'] = 0.0
        
        # Histograms (for latency distributions)
        self.histograms['trade_execution_time'] = deque(maxlen=100)
        self.histograms['broker_latency'] = deque(maxlen=100)
        self.histograms['db_latency'] = deque(maxlen=100)
    
    def record_trade_execution(self, execution_time: float, success: bool, timeout: bool = False):
        """Record trade execution metrics"""
        self.counters['trades_total'] += 1
        
        if success:
            self.counters['trades_successful'] += 1
        else:
            self.counters['trades_failed'] += 1
            
        if timeout:
            self.counters['trades_timeout'] += 1
        
        self.histograms['trade_execution_time'].append(execution_time)
        
        logger.debug(f"Trade execution recorded: {execution_time:.3f}s, success: {success}")
    
    def record_broker_latency(self, latency: float):
        """Record broker latency"""
        self.histograms['broker_latency'].append(latency)
        logger.debug(f"Broker latency recorded: {latency:.3f}s")
    
    def record_db_latency(self, latency: float):
        """Record database latency"""
        self.histograms['db_latency'].append(latency)
        logger.debug(f"DB latency recorded: {latency:.3f}s")
    
    def record_ledger_retry(self):
        """Record ledger update retry"""
        self.counters['ledger_update_retries'] += 1
        logger.debug("Ledger retry recorded")
    
    def record_broker_failure(self):
        """Record broker failure"""
        self.counters['broker_failures'] += 1
        self.gauges['broker_connected'] = 0.0
        logger.warning("Broker failure recorded")
    
    def record_broker_recovery(self):
        """Record broker recovery"""
        self.gauges['broker_connected'] = 1.0
        logger.info("Broker recovery recorded")
    
    def record_db_failure(self):
        """Record database failure"""
        self.counters['db_failures'] += 1
        self.gauges['db_connected'] = 0.0
        logger.warning("DB failure recorded")
    
    def record_db_recovery(self):
        """Record database recovery"""
        self.gauges['db_connected'] = 1.0
        logger.info("DB recovery recorded")
    
    def set_trading_state(self, enabled: bool):
        """Set trading enabled state"""
        self.gauges['trading_enabled'] = 1.0 if enabled else 0.0
        logger.info(f"Trading state set to: {enabled}")
    
    def set_reconciliation_queue_size(self, size: int):
        """Set reconciliation queue size"""
        self.gauges['reconciliation_queue_size'] = float(size)
        logger.debug(f"Reconciliation queue size: {size}")
    
    def set_active_positions(self, count: int):
        """Set active positions count"""
        self.gauges['active_positions'] = float(count)
        logger.debug(f"Active positions: {count}")
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output"""
        metrics_output = []
        
        # Counters
        metrics_output.append("# HELP nexus_trades_total Total number of trades")
        metrics_output.append("# TYPE nexus_trades_total counter")
        metrics_output.append(f"nexus_trades_total {self.counters['trades_total']}")
        
        metrics_output.append("# HELP nexus_trades_successful Total successful trades")
        metrics_output.append("# TYPE nexus_trades_successful counter")
        metrics_output.append(f"nexus_trades_successful {self.counters['trades_successful']}")
        
        metrics_output.append("# HELP nexus_trades_failed Total failed trades")
        metrics_output.append("# TYPE nexus_trades_failed counter")
        metrics_output.append(f"nexus_trades_failed {self.counters['trades_failed']}")
        
        metrics_output.append("# HELP nexus_trades_timeout Total timeout trades")
        metrics_output.append("# TYPE nexus_trades_timeout counter")
        metrics_output.append(f"nexus_trades_timeout {self.counters['trades_timeout']}")
        
        metrics_output.append("# HELP nexus_ledger_update_retries Total ledger update retries")
        metrics_output.append("# TYPE nexus_ledger_update_retries counter")
        metrics_output.append(f"nexus_ledger_update_retries {self.counters['ledger_update_retries']}")
        
        metrics_output.append("# HELP nexus_broker_failures Total broker failures")
        metrics_output.append("# TYPE nexus_broker_failures counter")
        metrics_output.append(f"nexus_broker_failures {self.counters['broker_failures']}")
        
        metrics_output.append("# HELP nexus_db_failures Total database failures")
        metrics_output.append("# TYPE nexus_db_failures counter")
        metrics_output.append(f"nexus_db_failures {self.counters['db_failures']}")
        
        # Gauges
        metrics_output.append("# HELP nexus_trading_enabled Trading enabled status")
        metrics_output.append("# TYPE nexus_trading_enabled gauge")
        metrics_output.append(f"nexus_trading_enabled {self.gauges['trading_enabled']}")
        
        metrics_output.append("# HELP nexus_broker_connected Broker connection status")
        metrics_output.append("# TYPE nexus_broker_connected gauge")
        metrics_output.append(f"nexus_broker_connected {self.gauges['broker_connected']}")
        
        metrics_output.append("# HELP nexus_db_connected Database connection status")
        metrics_output.append("# TYPE nexus_db_connected gauge")
        metrics_output.append(f"nexus_db_connected {self.gauges['db_connected']}")
        
        metrics_output.append("# HELP nexus_reconciliation_queue_size Reconciliation queue size")
        metrics_output.append("# TYPE nexus_reconciliation_queue_size gauge")
        metrics_output.append(f"nexus_reconciliation_queue_size {self.gauges['reconciliation_queue_size']}")
        
        metrics_output.append("# HELP nexus_active_positions Number of active positions")
        metrics_output.append("# TYPE nexus_active_positions gauge")
        metrics_output.append(f"nexus_active_positions {self.gauges['active_positions']}")
        
        # Histograms (summary statistics)
        if self.histograms['trade_execution_time']:
            exec_times = list(self.histograms['trade_execution_time'])
            if exec_times:
                avg_time = sum(exec_times) / len(exec_times)
                max_time = max(exec_times)
                min_time = min(exec_times)
                
                metrics_output.append("# HELP nexus_trade_execution_time_seconds Trade execution time")
                metrics_output.append("# TYPE nexus_trade_execution_time_seconds gauge")
                metrics_output.append(f"nexus_trade_execution_time_seconds_avg {avg_time:.3f}")
                metrics_output.append(f"nexus_trade_execution_time_seconds_max {max_time:.3f}")
                metrics_output.append(f"nexus_trade_execution_time_seconds_min {min_time:.3f}")
        
        if self.histograms['broker_latency']:
            latencies = list(self.histograms['broker_latency'])
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                
                metrics_output.append("# HELP nexus_broker_latency_seconds Broker latency")
                metrics_output.append("# TYPE nexus_broker_latency_seconds gauge")
                metrics_output.append(f"nexus_broker_latency_seconds_avg {avg_latency:.3f}")
                metrics_output.append(f"nexus_broker_latency_seconds_max {max_latency:.3f}")
        
        if self.histograms['db_latency']:
            db_latencies = list(self.histograms['db_latency'])
            if db_latencies:
                avg_db_latency = sum(db_latencies) / len(db_latencies)
                max_db_latency = max(db_latencies)
                
                metrics_output.append("# HELP nexus_db_latency_seconds Database latency")
                metrics_output.append("# TYPE nexus_db_latency_seconds gauge")
                metrics_output.append(f"nexus_db_latency_seconds_avg {avg_db_latency:.3f}")
                metrics_output.append(f"nexus_db_latency_seconds_max {max_db_latency:.3f}")
        
        return "\n".join(metrics_output)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {
                'trade_execution_time': {
                    'count': len(self.histograms['trade_execution_time']),
                    'avg': sum(self.histograms['trade_execution_time']) / len(self.histograms['trade_execution_time']) if self.histograms['trade_execution_time'] else 0,
                    'max': max(self.histograms['trade_execution_time']) if self.histograms['trade_execution_time'] else 0
                },
                'broker_latency': {
                    'count': len(self.histograms['broker_latency']),
                    'avg': sum(self.histograms['broker_latency']) / len(self.histograms['broker_latency']) if self.histograms['broker_latency'] else 0,
                    'max': max(self.histograms['broker_latency']) if self.histograms['broker_latency'] else 0
                },
                'db_latency': {
                    'count': len(self.histograms['db_latency']),
                    'avg': sum(self.histograms['db_latency']) / len(self.histograms['db_latency']) if self.histograms['db_latency'] else 0,
                    'max': max(self.histograms['db_latency']) if self.histograms['db_latency'] else 0
                }
            }
        }

# Global metrics instance
operational_metrics = OperationalMetrics()
