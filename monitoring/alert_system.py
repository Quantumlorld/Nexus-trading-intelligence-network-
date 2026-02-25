"""
Nexus Trading System - Alert System
Real-time alerting for system events and notifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from sqlalchemy.orm import Session

from database.session import get_database_session
from database.models import Alert, AlertLevel
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time

from core.logger import get_logger


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Alert categories"""
    TRADE = "trade"
    RISK = "risk"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    VOLATILITY = "volatility"
    COMPLIANCE = "compliance"
    EXECUTION = "execution"
    STRATEGY = "strategy"


@dataclass
class Alert:
    """Alert information"""
    id: str
    timestamp: datetime
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    notification_sent: bool = False


@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # Python expression
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 5
    escalation_enabled: bool = True
    max_escalations: int = 3
    notification_channels: List[str] = field(default_factory=list)


class AlertSystem:
    """
    Comprehensive alert system for trading operations
    Handles rule-based alerts, notifications, escalations, and acknowledgments
    """
    
    def __init__(self, config: Dict[str, Any], db_path: str = "monitoring/alerts.db"):
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Setup logging
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Database connection
        self.conn = None
        self._init_database()
        
        # Alert management
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_queue = queue.Queue()
        
        # Notification channels
        self.notification_channels = {}
        self._init_notification_channels()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.is_running = False
        
        # Alert processing
        self.processing_interval = timedelta(seconds=10)
        self.escalation_interval = timedelta(minutes=30)
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_category': {},
            'alerts_by_severity': {},
            'acknowledged_alerts': 0,
            'escalated_alerts': 0,
            'notification_failures': 0
        }
        
        # Default rules
        self._init_default_rules()
        
        self.logger.info("Alert system initialized")
    
    def _init_database(self):
        """Initialize database for alerts"""
        # Database is handled by SQLAlchemy models
        self.logger.info("Alert system initialized")
    
    def _init_notification_channels(self):
        """Initialize notification channels"""
        
        # Email channel
        email_config = self.config.get('notifications', {}).get('email', {})
        if email_config.get('enabled', False):
            self.notification_channels['email'] = {
                'type': 'email',
                'enabled': True,
                'config': email_config
            }
        
        # Log channel (always enabled)
        self.notification_channels['log'] = {
            'type': 'log',
            'enabled': True,
            'config': {}
        }
        
        # Console channel
        self.notification_channels['console'] = {
            'type': 'console',
            'enabled': True,
            'config': {}
        }
        
        # Webhook channel
        webhook_config = self.config.get('notifications', {}).get('webhook', {})
        if webhook_config.get('enabled', False):
            self.notification_channels['webhook'] = {
                'type': 'webhook',
                'enabled': True,
                'config': webhook_config
            }
    
    def _init_default_rules(self):
        """Initialize default alert rules"""
        
        default_rules = [
            # Risk management alerts
            AlertRule(
                id="daily_loss_limit",
                name="Daily Loss Limit Breached",
                category=AlertCategory.RISK,
                severity=AlertSeverity.HIGH,
                condition="daily_pnl <= -9.99",
                message_template="Daily loss limit breached: ${daily_pnl:.2f} (limit: $9.99)",
                escalation_enabled=True
            ),
            
            AlertRule(
                id="consecutive_losses",
                name="Consecutive Losses",
                category=AlertCategory.RISK,
                severity=AlertSeverity.MEDIUM,
                condition="consecutive_losses >= 3",
                message_template="Consecutive losses: ${consecutive_losses} (limit: 3)",
                escalation_enabled=True
            ),
            
            AlertRule(
                id="max_drawdown",
                name="Maximum Drawdown",
                category=AlertCategory.RISK,
                severity=AlertSeverity.HIGH,
                condition="max_drawdown <= -5.0",
                message_template="Maximum drawdown reached: ${max_drawdown:.2f}%",
                escalation_enabled=True
            ),
            
            # Performance alerts
            AlertRule(
                id="low_win_rate",
                name="Low Win Rate",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.MEDIUM,
                condition="win_rate < 40 and total_trades >= 10",
                message_template="Low win rate: ${win_rate:.1f}% (trades: ${total_trades})",
                escalation_enabled=False
            ),
            
            AlertRule(
                id="profit_factor_decline",
                name="Profit Factor Decline",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.LOW,
                condition="profit_factor < 1.0 and total_trades >= 20",
                message_template="Profit factor below 1.0: ${profit_factor:.2f}",
                escalation_enabled=False
            ),
            
            # System alerts
            AlertRule(
                id="connection_lost",
                name="Connection Lost",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.HIGH,
                condition="not connected",
                message_template="Connection to trading system lost",
                escalation_enabled=True
            ),
            
            AlertRule(
                id="high_latency",
                name="High Latency",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.MEDIUM,
                condition="execution_latency > 5000",
                message_template="High execution latency: ${execution_latency}ms",
                escalation_enabled=False
            ),
            
            # Volatility alerts
            AlertRule(
                id="extreme_volatility",
                name="Extreme Volatility",
                category=AlertCategory.VOLATILITY,
                severity=AlertSeverity.HIGH,
                condition="volatility_regime == 'extreme'",
                message_template="Extreme volatility detected for ${symbol}: ${current_volatility:.3f}",
                escalation_enabled=True
            ),
            
            # Compliance alerts
            AlertRule(
                id="position_limit_violation",
                name="Position Limit Violation",
                category=AlertCategory.COMPLIANCE,
                severity=AlertSeverity.HIGH,
                condition="multiple_positions_same_asset",
                message_template="Position limit violation: Multiple positions for ${symbol}",
                escalation_enabled=True
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def start(self):
        """Start alert system"""
        
        if self.is_running:
            self.logger.warning("Alert system is already running")
            return
        
        self.is_running = True
        
        # Start background processing
        self.executor.submit(self._processing_loop)
        
        self.logger.info("Alert system started")
    
    def stop(self):
        """Stop alert system"""
        
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close database connection
        if self.conn:
            self.conn.close()
        
        self.logger.info("Alert system stopped")
    
    def _processing_loop(self):
        """Background alert processing loop"""
        
        while self.is_running:
            try:
                # Process queued alerts
                while not self.alert_queue.empty():
                    try:
                        alert = self.alert_queue.get_nowait()
                        self._process_alert(alert)
                    except queue.Empty:
                        break
                
                # Check for rule-based alerts
                self._check_rule_based_alerts()
                
                # Check for escalations
                self._check_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Sleep until next processing cycle
                time.sleep(self.processing_interval.total_seconds())
            
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def create_alert(self, category: AlertCategory, severity: AlertSeverity,
                    title: str, message: str, symbol: Optional[str] = None,
                    strategy: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert"""
        
        alert_id = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            title=title,
            message=message,
            symbol=symbol,
            strategy=strategy,
            data=data or {},
            escalation_level=0
        )
        
        # Add to queue for processing
        self.alert_queue.put(alert)
        
        self.logger.info(f"Alert created: {title}")
        
        return alert_id
    
    def _process_alert(self, alert: Alert):
        """Process an alert"""
        
        try:
            with self._lock:
                # Add to alerts list
                self.alerts.append(alert)
                
                # Update statistics
                self._update_alert_stats(alert)
                
                # Check cooldown
                if not self._check_cooldown(alert):
                    # Send notifications
                    self._send_notifications(alert)
                    alert.notification_sent = True
                
                # Save to database
                self._save_alert(alert)
            
            self.logger.debug(f"Alert processed: {alert.title}")
        
        except Exception as e:
            self.logger.error(f"Error processing alert {alert.id}: {e}")
    
    def _check_rule_based_alerts(self):
        """Check for rule-based alerts"""
        
        # Get current system state
        system_state = self._get_system_state()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if not self._check_rule_cooldown(rule):
                continue
            
            # Evaluate condition
            try:
                # Create safe evaluation context
                context = {
                    'system_state': system_state,
                    'datetime': datetime,
                    'math': __import__('math'),
                    'np': __import__('numpy')
                }
                
                # Add system state variables to context
                for key, value in system_state.items():
                    context[key] = value
                
                # Evaluate condition
                if eval(rule.condition, {"__builtins__": {}, **context}):
                    # Create alert
                    alert = Alert(
                        id=f"{rule.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        category=rule.category,
                        severity=rule.severity,
                        title=rule.name,
                        message=self._format_message(rule.message_template, context),
                        data={'rule_id': rule.id, 'condition': rule.condition},
                        escalation_level=0
                    )
                    
                    self.alert_queue.put(alert)
            
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.id}: {e}")
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for rule evaluation"""
        
        # This would integrate with other system components
        # For now, return mock data
        
        return {
            'daily_pnl': -5.50,  # Example
            'consecutive_losses': 2,
            'max_drawdown': -3.2,
            'win_rate': 45.0,
            'total_trades': 15,
            'profit_factor': 1.2,
            'connected': True,
            'execution_latency': 1200,
            'volatility_regime': 'normal',
            'current_volatility': 0.015,
            'symbol': 'XAUUSD',
            'multiple_positions_same_asset': False
        }
    
    def _format_message(self, template: str, context: Dict[str, Any]) -> str:
        """Format message template with context variables"""
        
        try:
            return template.format(**context)
        except KeyError as e:
            self.logger.error(f"Error formatting message template: {e}")
            return template
    
    def _check_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period"""
        
        # Get recent alerts of same category
        recent_alerts = [a for a in self.alerts 
                          if a.category == alert.category and 
                          a.timestamp > datetime.now() - timedelta(minutes=30)]
        
        return len(recent_alerts) > 0
    
    def _check_rule_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        
        # Get recent alerts from this rule
        recent_alerts = [a for a in self.alerts 
                          if a.data.get('rule_id') == rule.id and 
                          a.timestamp > datetime.now() - timedelta(minutes=rule.cooldown_minutes)]
        
        return len(recent_alerts) > 0
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        
        for channel_name, channel in self.notification_channels.items():
            if not channel['enabled']:
                continue
            
            try:
                if channel['type'] == 'email':
                    self._send_email_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_notification(alert, channel['config'])
                elif channel['type'] == 'console':
                    self._send_console_notification(alert)
                elif channel['type'] == 'log':
                    self._send_log_notification(alert)
            
            except Exception as e:
                self.logger.error(f"Error sending {channel_name} notification: {e}")
                self.alert_stats['notification_failures'] += 1
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email')
            msg['To'] = config.get('to_email')
            msg['Subject'] = f"Nexus Alert: {alert.title}"
            
            body = f"""
Alert Details:
Title: {alert.title}
Severity: {alert.severity.value.upper()}
Category: {alert.category.value.upper()}
Time: {alert.timestamp}
Symbol: {alert.symbol or 'N/A'}
Strategy: {alert.strategy or 'N/A'}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config.get('smtp_server'), config.get('smtp_port', 587))
            server.starttls()
            server.login(config.get('username'), config.get('password'))
            
            text = msg.as_string()
            server.sendmail(config.get('from_email'), config.get('to_email'), text)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert: {alert.title}")
        
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        
        try:
            import requests
            
            webhook_url = config.get('url')
            if not webhook_url:
                return
            
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'category': alert.category.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'symbol': alert.symbol,
                'strategy': alert.strategy,
                'data': alert.data
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert: {alert.title}")
        
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")
    
    def _send_console_notification(self, alert: Alert):
        """Send console notification"""
        
        # Color code by severity
        colors = {
            AlertSeverity.INFO: '\033[94m',      # Blue
            AlertSeverity.LOW: '\033[92m',       # Green
            AlertSeverity.MEDIUM: '\033[93m',    # Yellow
            AlertSeverity.HIGH: '\033[91m',     # Red
            AlertSeverity.CRITICAL: '\033[95m',  # Magenta
            AlertSeverity.EMERGENCY: '\033[97m'  # White
        }
        
        color = colors.get(alert.severity, '')
        reset = '\033[0m'
        
        print(f"{color}ALERT: {alert.title} ({alert.severity.value.upper()}){reset}")
        print(f"{color}Category: {alert.category.value}{reset}")
        print(f"{color}Message: {alert.message}{reset}")
        if alert.symbol:
            print(f"{color}Symbol: {alert.symbol}{reset}")
        print(f"{color}Time: {alert.timestamp}{reset}")
        print("-" * 50)
    
    def _send_log_notification(self, alert: Alert):
        """Send log notification"""
        
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }
        
        level = log_level.get(alert.severity, logging.INFO)
        
        self.logger.log(level, f"ALERT: {alert.title} - {alert.message}")
    
    def _update_alert_stats(self, alert: Alert):
        """Update alert statistics"""
        
        self.alert_stats['total_alerts'] += 1
        
        # Category stats
        category = alert.category.value
        self.alert_stats['alerts_by_category'][category] = self.alert_stats['alerts_by_category'].get(category, 0) + 1
        
        # Severity stats
        severity = alert.severity.value
        self.alert_stats['alerts_by_severity'][severity] = self.alert_stats['alerts_by_severity'].get(severity, 0) + 1
    
    def _save_alert(self, alert: Alert):
        """Save alert to database"""
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (
                id, timestamp, category, severity, title, message, symbol,
                strategy, data, acknowledged, acknowledged_by, acknowledged_at,
                resolved, resolved_at, escalation_level, notification_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id,
            alert.timestamp.isoformat(),
            alert.category.value,
            alert.severity.value,
            alert.title,
            alert.message,
            alert.symbol,
            alert.strategy,
            json.dumps(alert.data),
            alert.acknowledged,
            alert.acknowledged_by,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            alert.resolved,
            alert.resolved_at.isoformat() if alert.resolved_at else None,
            alert.escalation_level,
            alert.notification_sent
        ))
        
        self.conn.commit()
    
    def _check_escalations(self):
        """Check for alert escalations"""
        
        for alert in self.alerts:
            if alert.escalation_level >= self.alert_rules.get(alert.data.get('rule_id', ''), {}).get('max_escalations', 3):
                continue
            
            # Check if alert should be escalated
            if self._should_escalate(alert):
                self._escalate_alert(alert)
    
    def _should_escalate(self, alert: Alert) -> bool:
        """Determine if alert should be escalated"""
        
        # Escalate if not acknowledged after certain time
        if not alert.acknowledged:
            time_since_alert = datetime.now() - alert.timestamp
            if time_since_alert > self.escalation_interval:
                return True
        
        # Escalate based on severity
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            return True
        
        return False
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert"""
        
        alert.escalation_level += 1
        
        # Update in database
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE alerts SET escalation_level = ? WHERE id = ?
        ''', (alert.escalation_level, alert.id))
        self.conn.commit()
        
        # Send escalation notification
        escalation_message = f"ESCALATION (Level {alert.escalation_level}): {alert.message}"
        
        # Create escalated alert
        escalated_alert = Alert(
            id=f"{alert.id}_escalated_{alert.escalation_level}",
            timestamp=datetime.now(),
            category=alert.category,
            severity=AlertSeverity.HIGH if alert.severity != AlertSeverity.EMERGENCY else AlertSeverity.EMERGENCY,
            title=f"ESCALATED: {alert.title}",
            message=escalation_message,
            symbol=alert.symbol,
            strategy=alert.strategy,
            data={**alert.data, 'escalated_from': alert.id, 'escalation_level': alert.escalation_level}
        )
        
        self.alert_queue.put(escalated_alert)
        self.alert_stats['escalated_alerts'] += 1
        
        self.logger.warning(f"Alert escalated: {alert.title} (Level {alert.escalation_level})")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = user
                    alert.acknowledged_at = datetime.now()
                    
                    # Update in database
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE alerts SET acknowledged = TRUE, acknowledged_by = ?, acknowledged_at = ?
                        WHERE id = ?
                    ''', (user, alert.acknowledged_at.isoformat(), alert_id))
                    self.conn.commit()
                    
                    # Add to history
                    cursor.execute('''
                        INSERT INTO alert_history (alert_id, timestamp, action, details, user)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (alert_id, datetime.now().isoformat(), 'acknowledged', f"Alert acknowledged by {user}", user))
                    self.conn.commit()
                    
                    self.logger.info(f"Alert acknowledged: {alert.title} by {user}")
                    return True
        
        return False
    
    def resolve_alert(self, alert_id: str, user: str, resolution: str) -> bool:
        """Resolve an alert"""
        
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    # Update in database
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE alerts SET resolved = TRUE, resolved_at = ?
                        WHERE id = ?
                    ''', (alert.resolved_at.isoformat(), alert_id))
                    self.conn.commit()
                    
                    # Add to history
                    cursor.execute('''
                        INSERT INTO alert_history (alert_id, timestamp, action, details, user)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (alert_id, datetime.now().isoformat(), 'resolved', resolution, user))
                    self.conn.commit()
                    
                    self.logger.info(f"Alert resolved: {alert.title} by {user}: {resolution}")
                    return True
        
        return False
    
    def get_alerts(self, category: Optional[AlertCategory] = None,
                    severity: Optional[AlertSeverity] = None,
                    symbol: Optional[str] = None,
                    acknowledged: Optional[bool] = None,
                    resolved: Optional[bool] = None,
                    hours: int = 24) -> List[Alert]:
        """Get filtered alerts"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = []
        
        for alert in self.alerts:
            if alert.timestamp < cutoff_time:
                continue
            
            if category and alert.category != category:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            if symbol and alert.symbol != symbol:
                continue
            
            if acknowledged is not None and alert.acknowledged != acknowledged:
                continue
            
            if resolved is not None and alert.resolved != resolved:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        
        return {
            'total_alerts': self.alert_stats['total_alerts'],
            'alerts_by_category': self.alert_stats['alerts_by_category'],
            'alerts_by_severity': self.alert_stats['alerts_by_severity'],
            'acknowledged_alerts': len([a for a in self.alerts if a.acknowledged]),
            'resolved_alerts': len([a for a in self.alerts if a.resolved]),
            'escalated_alerts': self.alert_stats['escalated_alerts'],
            'notification_failures': self.alert_stats['notification_failures'],
            'active_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'notification_channels': len([c for c in self.notification_channels.values() if c['enabled']])
        }
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        
        self.alert_rules[rule.id] = rule
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO alert_rules (
                id, name, category, severity, condition, message_template,
                enabled, cooldown_minutes, escalation_enabled, max_escalations, notification_channels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rule.id,
            rule.name,
            rule.category.value,
            rule.severity.value,
            rule.condition,
            rule.message_template,
            rule.enabled,
            rule.cooldown_minutes,
            rule.escalation_enabled,
            rule.max_escalations,
            json.dumps(rule.notification_channels)
        ))
        
        self.conn.commit()
        
        self.logger.info(f"Alert rule added: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            
            # Remove from database
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM alert_rules WHERE id = ?', (rule_id,))
            self.conn.commit()
            
            self.logger.info(f"Alert rule removed: {rule_id}")
    
    def export_alerts(self, filepath: str, format: str = 'csv'):
        """Export alerts to file"""
        
        try:
            if format.lower() == 'csv':
                self._export_to_csv(filepath)
            elif format.lower() == 'json':
                self._export_to_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting alerts: {e}")
    
    def _export_to_csv(self, filepath: str):
        """Export alerts to CSV"""
        
        import pandas as pd
        
        alert_data = []
        for alert in self.alerts:
            alert_data.append({
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'category': alert.category.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'symbol': alert.symbol,
                'strategy': alert.strategy,
                'acknowledged': alert.acknowledged,
                'acknowledged_by': alert.acknowledged_by,
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                'resolved': alert.resolved,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'escalation_level': alert.escalation_level,
                'notification_sent': alert.notification_sent
            })
        
        if alert_data:
            df = pd.DataFrame(alert_data)
            df.to_csv(filepath, index=False)
        
        self.logger.info(f"Alerts exported to CSV: {filepath}")
    
    def _export_to_json(self, filepath: str):
        """Export alerts to JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'alerts': [
                {
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'category': alert.category.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'symbol': alert.symbol,
                    'strategy': alert.strategy,
                    'data': alert.data,
                    'acknowledged': alert.acknowledged,
                    'acknowledged_by': alert.acknowledged_by,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'escalation_level': alert.escalation_level,
                    'notification_sent': alert.notification_sent
                }
                for alert in self.alerts
            ],
            'statistics': self.get_alert_statistics(),
            'rules': {
                rule.id: {
                    'name': rule.name,
                    'category': rule.category.value,
                    'severity': rule.severity.value,
                    'enabled': rule.enabled,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'escalation_enabled': rule.escalation_enabled
                }
                for rule in self.alert_rules.values()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Alerts exported to JSON: {filepath}")
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        
        try:
            # Delete alerts older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM alert_history WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            self.conn.commit()
            
            # Update in-memory alerts
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old alerts")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        
        return {
            'is_running': self.is_running,
            'total_alerts': len(self.alerts),
            'active_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'notification_channels': list(self.notification_channels.keys()),
            'queue_size': self.alert_queue.qsize(),
            'statistics': self.get_alert_statistics()
        }
