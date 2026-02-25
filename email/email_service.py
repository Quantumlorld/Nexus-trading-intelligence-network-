"""
Nexus Trading System - Email Service
Enterprise email notifications and user communication
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.html import MIMEText as HTMLMIMEText
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging
import asyncio
from jinja2 import Template

logger = logging.getLogger(__name__)

class EmailType(Enum):
    """Email notification types"""
    WELCOME = "welcome"
    TRADE_EXECUTED = "trade_executed"
    SIGNAL_GENERATED = "signal_generated"
    PAYMENT_SUCCESS = "payment_success"
    PAYMENT_FAILED = "payment_failed"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_REPORT = "performance_report"
    SYSTEM_MAINTENANCE = "system_maintenance"
    PASSWORD_RESET = "password_reset"
    ACCOUNT_VERIFICATION = "account_verification"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"

class EmailPriority(Enum):
    """Email priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class EmailTemplate:
    """Email template configuration"""
    template_id: str
    name: str
    subject: str
    html_template: str
    text_template: str
    variables: List[str]

class EmailService:
    """Enterprise email service for user notifications"""
    
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        
        # Initialize email templates
        self.templates = self._initialize_templates()
        
        # Email queue for batch processing
        self.email_queue = []
        self.is_processing = False
        
        logger.info("Email service initialized")
    
    def _initialize_templates(self) -> Dict[str, EmailTemplate]:
        """Initialize email templates"""
        return {
            EmailType.WELCOME.value: EmailTemplate(
                template_id="welcome",
                name="Welcome to Nexus Trading",
                subject="Welcome to Nexus Trading Intelligence Network",
                html_template=self._get_welcome_html_template(),
                text_template=self._get_welcome_text_template(),
                variables=["user_name", "account_type", "login_url"]
            ),
            EmailType.TRADE_EXECUTED.value: EmailTemplate(
                template_id="trade_executed",
                name="Trade Execution Notification",
                subject="Trade Executed - {{symbol}} {{action}}",
                html_template=self._get_trade_executed_html_template(),
                text_template=self._get_trade_executed_text_template(),
                variables=["user_name", "symbol", "action", "quantity", "price", "pnl"]
            ),
            EmailType.SIGNAL_GENERATED.value: EmailTemplate(
                template_id="signal_generated",
                name="Trading Signal Alert",
                subject="New Trading Signal - {{symbol}} {{signal}}",
                html_template=self._get_signal_generated_html_template(),
                text_template=self._get_signal_generated_text_template(),
                variables=["user_name", "symbol", "signal", "confidence", "strategy"]
            ),
            EmailType.PAYMENT_SUCCESS.value: EmailTemplate(
                template_id="payment_success",
                name="Payment Successful",
                subject="Payment Successful - {{plan_name}} Subscription",
                html_template=self._get_payment_success_html_template(),
                text_template=self._get_payment_success_text_template(),
                variables=["user_name", "plan_name", "amount", "next_billing_date"]
            ),
            EmailType.PAYMENT_FAILED.value: EmailTemplate(
                template_id="payment_failed",
                name="Payment Failed",
                subject="Payment Failed - Action Required",
                html_template=self._get_payment_failed_html_template(),
                text_template=self._get_payment_failed_text_template(),
                variables=["user_name", "plan_name", "amount", "retry_date"]
            ),
            EmailType.RISK_ALERT.value: EmailTemplate(
                template_id="risk_alert",
                name="Risk Management Alert",
                subject="Risk Alert - {{alert_type}}",
                html_template=self._get_risk_alert_html_template(),
                text_template=self._get_risk_alert_text_template(),
                variables=["user_name", "alert_type", "risk_level", "recommendation"]
            ),
            EmailType.PERFORMANCE_REPORT.value: EmailTemplate(
                template_id="performance_report",
                name="Performance Report",
                subject="Your Trading Performance Report - {{period}}",
                html_template=self._get_performance_report_html_template(),
                text_template=self._get_performance_report_text_template(),
                variables=["user_name", "period", "total_trades", "win_rate", "total_pnl", "sharpe_ratio"]
            ),
            EmailType.DAILY_SUMMARY.value: EmailTemplate(
                template_id="daily_summary",
                name="Daily Trading Summary",
                subject="Daily Trading Summary - {{date}}",
                html_template=self._get_daily_summary_html_template(),
                text_template=self._get_daily_summary_text_template(),
                variables=["user_name", "date", "trades_executed", "signals_generated", "account_balance"]
            )
        }
    
    async def send_email(self, to_email: str, email_type: str, variables: Dict[str, Any], 
                        priority: EmailPriority = EmailPriority.NORMAL) -> Dict[str, Any]:
        """Send an email notification"""
        try:
            template = self.templates.get(email_type)
            if not template:
                return {
                    "success": False,
                    "error": f"Email template not found: {email_type}"
                }
            
            # Render templates
            html_content = Template(template.html_template).render(**variables)
            text_content = Template(template.text_template).render(**variables)
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = Template(template.subject).render(**variables)
            message["From"] = self.email
            message["To"] = to_email
            
            # Add HTML and text parts
            message.attach(HTMLMIMEText(html_content, "html"))
            message.attach(MIMEText(text_content, "plain"))
            
            # Send email
            result = await self._send_email_message(message, to_email)
            
            logger.info(f"Email sent to {to_email}: {email_type}")
            
            return {
                "success": True,
                "email_type": email_type,
                "to_email": to_email,
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _send_email_message(self, message: MIMEMultipart, to_email: str) -> bool:
        """Send email message via SMTP"""
        try:
            # Create SMTP session
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                server.send_message(message, to_addrs=[to_email])
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            raise
    
    async def send_welcome_email(self, user_email: str, user_name: str, account_type: str) -> Dict[str, Any]:
        """Send welcome email to new user"""
        variables = {
            "user_name": user_name,
            "account_type": account_type,
            "login_url": "https://nexus-trading.com/login",
            "current_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        
        return await self.send_email(user_email, EmailType.WELCOME.value, variables)
    
    async def send_trade_notification(self, user_email: str, user_name: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send trade execution notification"""
        variables = {
            "user_name": user_name,
            "symbol": trade_data.get("symbol", "N/A"),
            "action": trade_data.get("action", "N/A"),
            "quantity": trade_data.get("quantity", 0),
            "price": trade_data.get("price", 0),
            "pnl": trade_data.get("pnl", 0),
            "timestamp": trade_data.get("timestamp", datetime.utcnow().isoformat())
        }
        
        return await self.send_email(user_email, EmailType.TRADE_EXECUTED.value, variables)
    
    async def send_signal_notification(self, user_email: str, user_name: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send trading signal notification"""
        variables = {
            "user_name": user_name,
            "symbol": signal_data.get("symbol", "N/A"),
            "signal": signal_data.get("signal", "N/A"),
            "confidence": signal_data.get("confidence", 0),
            "strategy": signal_data.get("strategy", "N/A"),
            "timestamp": signal_data.get("timestamp", datetime.utcnow().isoformat())
        }
        
        return await self.send_email(user_email, EmailType.SIGNAL_GENERATED.value, variables)
    
    async def send_payment_success(self, user_email: str, user_name: str, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send payment success notification"""
        variables = {
            "user_name": user_name,
            "plan_name": payment_data.get("plan_name", "N/A"),
            "amount": payment_data.get("amount", 0),
            "next_billing_date": payment_data.get("next_billing_date", "N/A"),
            "payment_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        
        return await self.send_email(user_email, EmailType.PAYMENT_SUCCESS.value, variables)
    
    async def send_risk_alert(self, user_email: str, user_name: str, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send risk management alert"""
        variables = {
            "user_name": user_name,
            "alert_type": alert_data.get("alert_type", "N/A"),
            "risk_level": alert_data.get("risk_level", "medium"),
            "recommendation": alert_data.get("recommendation", "Monitor your positions"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        priority = EmailPriority.HIGH if alert_data.get("risk_level") == "high" else EmailPriority.NORMAL
        
        return await self.send_email(user_email, EmailType.RISK_ALERT.value, variables, priority)
    
    async def send_performance_report(self, user_email: str, user_name: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send performance report"""
        variables = {
            "user_name": user_name,
            "period": performance_data.get("period", "monthly"),
            "total_trades": performance_data.get("total_trades", 0),
            "win_rate": performance_data.get("win_rate", 0),
            "total_pnl": performance_data.get("total_pnl", 0),
            "sharpe_ratio": performance_data.get("sharpe_ratio", 0),
            "report_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        
        return await self.send_email(user_email, EmailType.PERFORMANCE_REPORT.value, variables)
    
    async def send_daily_summary(self, user_email: str, user_name: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send daily trading summary"""
        variables = {
            "user_name": user_name,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "trades_executed": summary_data.get("trades_executed", 0),
            "signals_generated": summary_data.get("signals_generated", 0),
            "account_balance": summary_data.get("account_balance", 0),
            "daily_pnl": summary_data.get("daily_pnl", 0)
        }
        
        return await self.send_email(user_email, EmailType.DAILY_SUMMARY.value, variables)
    
    async def send_bulk_emails(self, email_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send bulk emails to multiple users"""
        results = {
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        for email_data in email_list:
            try:
                result = await self.send_email(
                    email_data["to_email"],
                    email_data["email_type"],
                    email_data["variables"],
                    email_data.get("priority", EmailPriority.NORMAL)
                )
                
                if result["success"]:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append(result)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "success": False,
                    "error": str(e),
                    "to_email": email_data.get("to_email", "N/A")
                })
        
        return results
    
    def get_template(self, email_type: str) -> Optional[EmailTemplate]:
        """Get email template by type"""
        return self.templates.get(email_type)
    
    def get_all_templates(self) -> Dict[str, EmailTemplate]:
        """Get all email templates"""
        return self.templates
    
    # Template methods
    def _get_welcome_html_template(self) -> str:
        """Welcome email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Welcome to Nexus Trading</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .logo { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .content { margin-bottom: 30px; }
                .button { display: inline-block; padding: 12px 24px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px; }
                .footer { text-align: center; color: #7f8c8d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🚀 Nexus Trading Intelligence Network</div>
                    <h1>Welcome to the Future of Trading!</h1>
                </div>
                <div class="content">
                    <p>Hi {{user_name}},</p>
                    <p>Welcome to Nexus Trading! Your {{account_type}} account has been successfully created.</p>
                    <p>You're now ready to experience AI-powered trading with:</p>
                    <ul>
                        <li>🤖 Advanced AI trading signals</li>
                        <li>📊 Real-time market analytics</li>
                        <li>⚡ Lightning-fast execution</li>
                        <li>🛡️ Enterprise-grade security</li>
                    </ul>
                    <p><a href="{{login_url}}" class="button">Login to Your Account</a></p>
                </div>
                <div class="footer">
                    <p>© 2026 Nexus Trading Intelligence Network. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_welcome_text_template(self) -> str:
        """Welcome email text template"""
        return """
        Welcome to Nexus Trading Intelligence Network
        
        Hi {{user_name}},
        
        Welcome to Nexus Trading! Your {{account_type}} account has been successfully created.
        
        You're now ready to experience AI-powered trading with:
        - Advanced AI trading signals
        - Real-time market analytics
        - Lightning-fast execution
        - Enterprise-grade security
        
        Login to your account: {{login_url}}
        
        © 2026 Nexus Trading Intelligence Network. All rights reserved.
        """
    
    def _get_trade_executed_html_template(self) -> str:
        """Trade executed email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Trade Executed</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .trade-info { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .profit { color: #27ae60; font-weight: bold; }
                .loss { color: #e74c3c; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Trade Executed Successfully</h1>
                <p>Hi {{user_name}},</p>
                <p>Your trade has been executed:</p>
                <div class="trade-info">
                    <p><strong>Symbol:</strong> {{symbol}}</p>
                    <p><strong>Action:</strong> {{action}}</p>
                    <p><strong>Quantity:</strong> {{quantity}}</p>
                    <p><strong>Price:</strong> ${{price}}</p>
                    <p><strong>P&L:</strong> <span class="{{'profit' if pnl > 0 else 'loss'}}">${{pnl}}</span></p>
                </div>
                <p>Check your dashboard for more details.</p>
            </div>
        </body>
        </html>
        """
    
    def _get_trade_executed_text_template(self) -> str:
        """Trade executed email text template"""
        return """
        Trade Executed Successfully
        
        Hi {{user_name}},
        
        Your trade has been executed:
        
        Symbol: {{symbol}}
        Action: {{action}}
        Quantity: {{quantity}}
        Price: ${{price}}
        P&L: ${{pnl}}
        
        Check your dashboard for more details.
        """
    
    def _get_signal_generated_html_template(self) -> str:
        """Signal generated email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>New Trading Signal</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .signal-info { background-color: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .confidence { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 New Trading Signal</h1>
                <p>Hi {{user_name}},</p>
                <p>A new trading signal has been generated:</p>
                <div class="signal-info">
                    <p><strong>Symbol:</strong> {{symbol}}</p>
                    <p><strong>Signal:</strong> {{signal}}</p>
                    <p><strong>Confidence:</strong> <span class="confidence">{{confidence}}%</span></p>
                    <p><strong>Strategy:</strong> {{strategy}}</p>
                </div>
                <p>Login to your dashboard to act on this signal.</p>
            </div>
        </body>
        </html>
        """
    
    def _get_signal_generated_text_template(self) -> str:
        """Signal generated email text template"""
        return """
        New Trading Signal
        
        Hi {{user_name}},
        
        A new trading signal has been generated:
        
        Symbol: {{symbol}}
        Signal: {{signal}}
        Confidence: {{confidence}}%
        Strategy: {{strategy}}
        
        Login to your dashboard to act on this signal.
        """
    
    def _get_payment_success_html_template(self) -> str:
        """Payment success email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Payment Successful</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .payment-info { background-color: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>💳 Payment Successful</h1>
                <p>Hi {{user_name}},</p>
                <p>Your payment has been processed successfully:</p>
                <div class="payment-info">
                    <p><strong>Plan:</strong> {{plan_name}}</p>
                    <p><strong>Amount:</strong> ${{amount}}</p>
                    <p><strong>Next Billing:</strong> {{next_billing_date}}</p>
                </div>
                <p>Thank you for your continued subscription!</p>
            </div>
        </body>
        </html>
        """
    
    def _get_payment_success_text_template(self) -> str:
        """Payment success email text template"""
        return """
        Payment Successful
        
        Hi {{user_name}},
        
        Your payment has been processed successfully:
        
        Plan: {{plan_name}}
        Amount: ${{amount}}
        Next Billing: {{next_billing_date}}
        
        Thank you for your continued subscription!
        """
    
    def _get_payment_failed_html_template(self) -> str:
        """Payment failed email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Payment Failed</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .payment-info { background-color: #ffe8e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .button { display: inline-block; padding: 12px 24px; background-color: #e74c3c; color: white; text-decoration: none; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠️ Payment Failed</h1>
                <p>Hi {{user_name}},</p>
                <p>We were unable to process your payment:</p>
                <div class="payment-info">
                    <p><strong>Plan:</strong> {{plan_name}}</p>
                    <p><strong>Amount:</strong> ${{amount}}</p>
                    <p><strong>Retry Date:</strong> {{retry_date}}</p>
                </div>
                <p><a href="#" class="button">Update Payment Method</a></p>
                <p>Please update your payment method to continue your subscription.</p>
            </div>
        </body>
        </html>
        """
    
    def _get_payment_failed_text_template(self) -> str:
        """Payment failed email text template"""
        return """
        Payment Failed - Action Required
        
        Hi {{user_name}},
        
        We were unable to process your payment:
        
        Plan: {{plan_name}}
        Amount: ${{amount}}
        Retry Date: {{retry_date}}
        
        Please update your payment method to continue your subscription.
        """
    
    def _get_risk_alert_html_template(self) -> str:
        """Risk alert email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Risk Alert</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .alert-info { background-color: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }
                .high-risk { border-left-color: #dc3545; background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠️ Risk Alert</h1>
                <p>Hi {{user_name}},</p>
                <div class="alert-info {{'high-risk' if risk_level == 'high' else ''}}">
                    <p><strong>Alert Type:</strong> {{alert_type}}</p>
                    <p><strong>Risk Level:</strong> {{risk_level}}</p>
                    <p><strong>Recommendation:</strong> {{recommendation}}</p>
                </div>
                <p>Please review your positions and take appropriate action.</p>
            </div>
        </body>
        </html>
        """
    
    def _get_risk_alert_text_template(self) -> str:
        """Risk alert email text template"""
        return """
        Risk Alert
        
        Hi {{user_name}},
        
        Alert Type: {{alert_type}}
        Risk Level: {{risk_level}}
        Recommendation: {{recommendation}}
        
        Please review your positions and take appropriate action.
        """
    
    def _get_performance_report_html_template(self) -> str:
        """Performance report email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .performance-info { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 12px; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Performance Report - {{period}}</h1>
                <p>Hi {{user_name}},</p>
                <p>Here's your trading performance summary:</p>
                <div class="performance-info">
                    <div class="metric">
                        <div class="metric-value">{{total_trades}}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{win_rate}}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{total_pnl}}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{sharpe_ratio}}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>
                <p>Check your dashboard for detailed analytics.</p>
            </div>
        </body>
        </html>
        """
    
    def _get_performance_report_text_template(self) -> str:
        """Performance report email text template"""
        return """
        Performance Report - {{period}}
        
        Hi {{user_name}},
        
        Here's your trading performance summary:
        
        Total Trades: {{total_trades}}
        Win Rate: {{win_rate}}%
        Total P&L: ${{total_pnl}}
        Sharpe Ratio: {{sharpe_ratio}}
        
        Check your dashboard for detailed analytics.
        """
    
    def _get_daily_summary_html_template(self) -> str:
        """Daily summary email HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Daily Trading Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }
                .summary-info { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Daily Trading Summary - {{date}}</h1>
                <p>Hi {{user_name}},</p>
                <p>Here's your daily trading summary:</p>
                <div class="summary-info">
                    <p><strong>Trades Executed:</strong> {{trades_executed}}</p>
                    <p><strong>Signals Generated:</strong> {{signals_generated}}</p>
                    <p><strong>Account Balance:</strong> ${{account_balance}}</p>
                    <p><strong>Daily P&L:</strong> ${{daily_pnl}}</p>
                </div>
                <p>Keep up the great work!</p>
            </div>
        </body>
        </html>
        """
    
    def _get_daily_summary_text_template(self) -> str:
        """Daily summary email text template"""
        return """
        Daily Trading Summary - {{date}}
        
        Hi {{user_name}},
        
        Here's your daily trading summary:
        
        Trades Executed: {{trades_executed}}
        Signals Generated: {{signals_generated}}
        Account Balance: ${{account_balance}}
        Daily P&L: ${{daily_pnl}}
        
        Keep up the great work!
        """

# Factory function
def create_email_service(smtp_server: str, smtp_port: int, email: str, password: str) -> EmailService:
    """Create and return email service instance"""
    return EmailService(smtp_server, smtp_port, email, password)
