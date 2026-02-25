"""
Nexus Trading System - Monitoring & Analytics Service (PostgreSQL Version)
Comprehensive logging, monitoring, and system analytics
"""

import logging
import json
import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict, deque
import statistics
from sqlalchemy.orm import Session
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for monitoring"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """Metric types for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Log entry data structure"""
    timestamp: datetime
    level: LogLevel
    message: str
    module: str
    user_id: str = None
    session_id: str = None
    request_id: str = None
    metadata: Dict[str, Any] = None

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    unit: str = ""
    timestamp: datetime = None

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AnalyticsService:
    """Comprehensive monitoring and analytics service"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=1000)
        self.log_entries = deque(maxlen=10000)
        self.performance_data = deque(maxlen=1000)
        
        # System monitoring
        self.system_monitoring_enabled = True
        self.monitoring_interval = 60  # seconds
        
        logger.info("Analytics service initialized")
    
    async def log_event(self, level: LogLevel, message: str, module: str,
                       user_id: str = None, session_id: str = None,
                       request_id: str = None, metadata: Dict[str, Any] = None):
        """Log an event"""
        try:
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                message=message,
                module=module,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                metadata=metadata
            )
            
            # Add to memory
            self.log_entries.append(log_entry)
            
            # Store in database via session factory
            await self._store_log_entry(log_entry)
            
            # Check for alerts
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                await self._check_error_alerts(log_entry)
            
            logger.debug(f"Logged event: {level.value} - {message}")
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
    
    async def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database using session factory"""
        try:
            with self.db_session_factory() as db:
                # Store in database using SQLAlchemy models
                from database.models import SystemLog
                
                db_log = SystemLog(
                    level=log_entry.level.value,
                    message=log_entry.message,
                    module=log_entry.module,
                    user_id=log_entry.user_id,
                    session_id=log_entry.session_id,
                    request_id=log_entry.request_id,
                    metadata=json.dumps(log_entry.metadata) if log_entry.metadata else None,
                    created_at=log_entry.timestamp
                )
                
                db.add(db_log)
                db.commit()
            
        except Exception as e:
            logger.error(f"Error storing log entry: {e}")
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType,
                           labels: Dict[str, str] = None, unit: str = ""):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels,
                unit=unit,
                timestamp=datetime.utcnow()
            )
            
            # Add to memory
            self.metrics[name].append(metric)
            
            # Store in database
            await self._store_metric(metric)
            
            # Check for alerts
            await self._check_metric_alerts(metric)
            
            logger.debug(f"Recorded metric: {name} = {value} {unit}")
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def _store_metric(self, metric: Metric):
        """Store metric in database using session factory"""
        try:
            with self.db_session_factory() as db:
                # Store in database using SQLAlchemy models
                from database.models import Metric as DBMetric
                
                db_metric = DBMetric(
                    name=metric.name,
                    value=metric.value,
                    metric_type=metric.metric_type.value,
                    labels=json.dumps(metric.labels) if metric.labels else None,
                    unit=metric.unit,
                    created_at=metric.timestamp
                )
                
                db.add(db_metric)
                db.commit()
            
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    async def create_alert(self, severity: AlertSeverity, title: str, message: str,
                          source: str, metadata: Dict[str, Any] = None):
        """Create an alert"""
        try:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                severity=severity,
                title=title,
                message=message,
                source=source,
                timestamp=datetime.utcnow(),
                metadata=metadata
            )
            
            # Add to memory
            self.alerts.append(alert)
            
            # Store in database
            await self._store_alert(alert)
            
            logger.warning(f"Alert created: {severity.value} - {title}")
            
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database using session factory"""
        try:
            with self.db_session_factory() as db:
                # Store in database using SQLAlchemy models
                from database.models import Alert as DBAlert
                
                db_alert = DBAlert(
                    level=alert.severity.value,
                    title=alert.title,
                    message=alert.message,
                    source=alert.source,
                    metadata=json.dumps(alert.metadata) if alert.metadata else None,
                    resolved=alert.resolved,
                    resolved_at=alert.resolved_at,
                    created_at=alert.timestamp
                )
                
                db.add(db_alert)
                db.commit()
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            # Find alert in memory
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    break
            
            # Update database
            with self.db_session_factory() as db:
                from database.models import Alert as DBAlert
                
                db_alert = db.query(DBAlert).filter(
                    DBAlert.title == alert.title,
                    DBAlert.resolved == False
                ).first()
                
                if db_alert:
                    db_alert.resolved = True
                    db_alert.resolved_at = datetime.utcnow()
                    db.commit()
            
            logger.info(f"Alert resolved: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
    
    async def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("cpu_usage", cpu_percent, MetricType.GAUGE, unit="%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric("memory_usage", memory.percent, MetricType.GAUGE, unit="%")
            await self.record_metric("memory_available", memory.available, MetricType.GAUGE, unit="bytes")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            await self.record_metric("disk_usage", disk.percent, MetricType.GAUGE, unit="%")
            await self.record_metric("disk_free", disk.free, MetricType.GAUGE, unit="bytes")
            
            # Network I/O
            network = psutil.net_io_counters()
            await self.record_metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER, unit="bytes")
            await self.record_metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER, unit="bytes")
            
            # Performance data
            performance_data = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_usage": disk.percent,
                "network_io": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                "active_connections": len(psutil.net_connections()),
                "response_time": 0.0,  # Would be measured from API calls
                "error_rate": 0.0  # Would be calculated from error logs
            }
            
            # Add to memory
            self.performance_data.append(performance_data)
            await self._store_performance_data(performance_data)
            
            # Check for system alerts
            await self._check_system_alerts(cpu_percent, memory.percent, disk.usage)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _store_performance_data(self, data: Dict[str, Any]):
        """Store performance data using session factory"""
        try:
            with self.db_session_factory() as db:
                # Store in database using SQLAlchemy models
                from database.models import PerformanceData
                
                perf_data = PerformanceData(
                    cpu_percent=data.get('cpu_percent'),
                    memory_percent=data.get('memory_percent'),
                    disk_usage=data.get('disk_usage'),
                    network_io=json.dumps(data.get('network_io', {})),
                    active_connections=data.get('active_connections'),
                    response_time=data.get('response_time'),
                    error_rate=data.get('error_rate'),
                    created_at=datetime.utcnow()
                )
                
                db.add(perf_data)
                db.commit()
            
        except Exception as e:
            logger.error(f"Error storing performance data: {e}")
    
    async def get_logs(self, level: LogLevel = None, module: str = None, user_id: str = None,
                      start_time: datetime = None, end_time: datetime = None, limit: int = 100) -> List[LogEntry]:
        """Get logs with filtering"""
        try:
            with self.db_session_factory() as db:
                from database.models import SystemLog
                
                query = db.query(SystemLog)
                
                # Apply filters
                if level:
                    query = query.filter(SystemLog.level == level.value)
                if module:
                    query = query.filter(SystemLog.module == module)
                if user_id:
                    query = query.filter(SystemLog.user_id == user_id)
                if start_time:
                    query = query.filter(SystemLog.created_at >= start_time)
                if end_time:
                    query = query.filter(SystemLog.created_at <= end_time)
                
                # Order and limit
                logs = query.order_by(SystemLog.created_at.desc()).limit(limit).all()
                
                # Convert to LogEntry objects
                log_entries = []
                for log in logs:
                    log_entry = LogEntry(
                        timestamp=log.created_at,
                        level=LogLevel(log.level),
                        message=log.message,
                        module=log.module,
                        user_id=log.user_id,
                        session_id=log.session_id,
                        request_id=log.request_id,
                        metadata=json.loads(log.metadata) if log.metadata else None
                    )
                    log_entries.append(log_entry)
                
                return log_entries
            
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []
    
    async def get_metrics(self, name: str = None, start_time: datetime = None,
                         end_time: datetime = None, limit: int = 100) -> List[Metric]:
        """Get metrics with filtering"""
        try:
            with self.db_session_factory() as db:
                from database.models import Metric as DBMetric
                
                query = db.query(DBMetric)
                
                # Apply filters
                if name:
                    query = query.filter(DBMetric.name == name)
                if start_time:
                    query = query.filter(DBMetric.created_at >= start_time)
                if end_time:
                    query = query.filter(DBMetric.created_at <= end_time)
                
                # Order and limit
                metrics = query.order_by(DBMetric.created_at.desc()).limit(limit).all()
                
                # Convert to Metric objects
                metric_list = []
                for metric in metrics:
                    metric_obj = Metric(
                        name=metric.name,
                        value=metric.value,
                        metric_type=MetricType(metric.metric_type),
                        labels=json.loads(metric.labels) if metric.labels else None,
                        unit=metric.unit,
                        timestamp=metric.created_at
                    )
                    metric_list.append(metric_obj)
                
                return metric_list
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    async def get_alerts(self, severity: AlertSeverity = None, resolved: bool = None,
                        start_time: datetime = None, end_time: datetime = None,
                        limit: int = 100) -> List[Alert]:
        """Get alerts with filtering"""
        try:
            with self.db_session_factory() as db:
                from database.models import Alert as DBAlert
                
                query = db.query(DBAlert)
                
                # Apply filters
                if severity:
                    query = query.filter(DBAlert.level == severity.value)
                if resolved is not None:
                    query = query.filter(DBAlert.resolved == resolved)
                if start_time:
                    query = query.filter(DBAlert.created_at >= start_time)
                if end_time:
                    query = query.filter(DBAlert.created_at <= end_time)
                
                # Order and limit
                alerts = query.order_by(DBAlert.created_at.desc()).limit(limit).all()
                
                # Convert to Alert objects
                alert_list = []
                for alert in alerts:
                    alert_obj = Alert(
                        alert_id=str(alert.id),
                        severity=AlertSeverity(alert.level),
                        title=alert.title,
                        message=alert.message,
                        source=alert.source,
                        timestamp=alert.created_at,
                        metadata=json.loads(alert.metadata) if alert.metadata else None,
                        resolved=alert.resolved,
                        resolved_at=alert.resolved_at
                    )
                    alert_list.append(alert_obj)
                
                return alert_list
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Get recent metrics
            recent_metrics = await self.get_metrics(limit=100)
            
            # Calculate health score
            health_score = 100
            
            # CPU health
            cpu_metrics = [m for m in recent_metrics if m.name == "cpu_usage"]
            if cpu_metrics:
                avg_cpu = sum(m.value for m in cpu_metrics) / len(cpu_metrics)
                if avg_cpu > 80:
                    health_score -= 20
                elif avg_cpu > 60:
                    health_score -= 10
            
            # Memory health
            memory_metrics = [m for m in recent_metrics if m.name == "memory_usage"]
            if memory_metrics:
                avg_memory = sum(m.value for m in memory_metrics) / len(memory_metrics)
                if avg_memory > 80:
                    health_score -= 20
                elif avg_memory > 60:
                    health_score -= 10
            
            # Error rate health
            error_logs = await self.get_logs(level=LogLevel.ERROR, limit=100)
            if len(error_logs) > 10:
                health_score -= 15
            elif len(error_logs) > 5:
                health_score -= 5
            
            # Critical alerts
            critical_alerts = await self.get_alerts(severity=AlertSeverity.CRITICAL, resolved=False)
            if len(critical_alerts) > 0:
                health_score -= 25
            
            health_score = max(0, health_score)
            
            return {
                "health_score": health_score,
                "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "unhealthy",
                "cpu_usage": cpu_metrics[-1].value if cpu_metrics else 0,
                "memory_usage": memory_metrics[-1].value if memory_metrics else 0,
                "active_alerts": len(await self.get_alerts(resolved=False)),
                "error_count": len(error_logs),
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "health_score": 0,
                "status": "error",
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def _check_error_alerts(self, log_entry: LogEntry):
        """Check for error-based alerts"""
        # High error rate alert
        recent_errors = [l for l in self.log_entries 
                        if l.level == LogLevel.ERROR 
                        and l.timestamp > datetime.utcnow() - timedelta(minutes=5)]
        
        if len(recent_errors) > 5:
            await self.create_alert(
                AlertSeverity.HIGH,
                "High Error Rate",
                f"Detected {len(recent_errors)} errors in the last 5 minutes",
                "system_monitor",
                {"error_count": len(recent_errors), "time_window": "5 minutes"}
            )
    
    async def _check_metric_alerts(self, metric: Metric):
        """Check for metric-based alerts"""
        # Example: Error rate alerts
        if metric.name == "error_rate" and metric.value > 0.05:  # 5%
            await self.create_alert(
                AlertSeverity.HIGH,
                "High Error Rate",
                f"Error rate is {metric.value:.1%}",
                "api_monitor",
                {"error_rate": metric.value, "threshold": 0.05}
            )
    
    async def _check_system_alerts(self, cpu_percent: float, memory_percent: float, disk_usage: float):
        """Check for system alerts"""
        # CPU alerts
        if cpu_percent > 90:
            await self.create_alert(
                AlertSeverity.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "system_monitor",
                {"cpu_percent": cpu_percent, "threshold": 90}
            )
        elif cpu_percent > 80:
            await self.create_alert(
                AlertSeverity.HIGH,
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "system_monitor",
                {"cpu_percent": cpu_percent, "threshold": 80}
            )
        
        # Memory alerts
        if memory_percent > 90:
            await self.create_alert(
                AlertSeverity.CRITICAL,
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                "system_monitor",
                {"memory_percent": memory_percent, "threshold": 90}
            )
        elif memory_percent > 80:
            await self.create_alert(
                AlertSeverity.HIGH,
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                "system_monitor",
                {"memory_percent": memory_percent, "threshold": 80}
            )
        
        # Disk alerts
        if disk_usage > 90:
            await self.create_alert(
                AlertSeverity.CRITICAL,
                "Low Disk Space",
                f"Disk usage is {disk_usage:.1f}%",
                "system_monitor",
                {"disk_usage": disk_usage, "threshold": 90}
            )
        elif disk_usage > 80:
            await self.create_alert(
                AlertSeverity.HIGH,
                "Low Disk Space",
                f"Disk usage is {disk_usage:.1f}%",
                "system_monitor",
                {"disk_usage": disk_usage, "threshold": 80}
            )
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if not self.system_monitoring_enabled:
            return
        
        logger.info("Starting system monitoring")
        
        while self.system_monitoring_enabled:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.system_monitoring_enabled = False
        logger.info("System monitoring stopped")

# Factory function
def create_analytics_service(db_session_factory) -> AnalyticsService:
    """Create and return analytics service instance"""
    return AnalyticsService(db_session_factory)
