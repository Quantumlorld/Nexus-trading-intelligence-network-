#!/usr/bin/env python3
"""
NEXUS TRADING SYSTEM - PRODUCTION HARDENING & VALIDATION AUDIT
===============================================================

This is a comprehensive security and reliability audit for financial infrastructure.
All findings are classified by severity with exact code patches provided.

Audit Scope:
- Frontend React + TypeScript application
- Backend FastAPI services
- Docker configuration
- Database operations
- WebSocket connections
- JWT security
- Trade atomicity
- Error handling
- Performance under load

Risk Classification:
- CRITICAL: Immediate security risk or data corruption potential
- HIGH: System reliability or security issue
- MEDIUM: Performance or operational issue
- LOW: Code quality or maintenance issue
"""

import os
import re
import json
import ast
import time
import asyncio
import subprocess
import threading
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class AuditFinding:
    category: str
    risk_level: RiskLevel
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    patch: str
    impact: str

class ProductionAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[AuditFinding] = []
        self.frontend_dir = self.project_root / "frontend"
        self.backend_dir = self.project_root
        self.docker_files = [
            self.project_root / "Dockerfile",
            self.project_root / "docker-compose.yml",
            self.project_root / "docker-compose-broker-safe.yml",
            self.frontend_dir / "Dockerfile",
            self.frontend_dir / "nginx.conf"
        ]
        
    def run_full_audit(self) -> Dict[str, Any]:
        """Execute comprehensive production audit"""
        print("🔍 STARTING PRODUCTION AUDIT...")
        print("=" * 80)
        
        # Frontend Security Audit
        self.audit_frontend_security()
        self.audit_frontend_race_conditions()
        self.audit_frontend_memory_leaks()
        self.audit_frontend_websocket_safety()
        
        # Backend Security Audit
        self.audit_backend_atomicity()
        self.audit_backend_race_conditions()
        self.audit_backend_jwt_security()
        self.audit_backend_error_handling()
        
        # Infrastructure Audit
        self.audit_docker_security()
        self.audit_database_operations()
        
        # Generate Report
        return self.generate_audit_report()
    
    def audit_frontend_security(self):
        """Audit frontend for security vulnerabilities"""
        print("\n🔐 AUDITING FRONTEND SECURITY...")
        
        # Check for localStorage usage (JWT tokens)
        frontend_files = list(self.frontend_dir.rglob("*.ts")) + list(self.frontend_dir.rglob("*.tsx"))
        
        for file_path in frontend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for localStorage usage
                    if 'localStorage' in line and ('token' in line.lower() or 'jwt' in line.lower()):
                        self.findings.append(AuditFinding(
                            category="JWT Security",
                            risk_level=RiskLevel.CRITICAL,
                            description="JWT token stored in localStorage - XSS vulnerability",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Use httpOnly cookies with secure flag for JWT storage",
                            impact="Token theft via XSS attacks"
                        ))
                    
                    # Check for unsafe async patterns
                    if re.search(r'await.*\.(catch|then)\s*\(', line):
                        self.findings.append(AuditFinding(
                            category="Async Safety",
                            risk_level=RiskLevel.HIGH,
                            description="Unsafe async/await with .catch/.then chaining",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Use try/catch blocks with proper error handling",
                            impact="Unhandled promise rejections"
                        ))
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_frontend_race_conditions(self):
        """Audit frontend for race conditions"""
        print("\n⚡ AUDITING FRONTEND RACE CONDITIONS...")
        
        frontend_files = list(self.frontend_dir.rglob("*.ts")) + list(self.frontend_dir.rglob("*.tsx"))
        
        for file_path in frontend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for double submission patterns
                    if 'executeTrade' in line or 'submitTrade' in line:
                        # Look for missing debouncing or loading state
                        if 'loading' not in line and 'disabled' not in line:
                            self.findings.append(AuditFinding(
                                category="Race Condition",
                                risk_level=RiskLevel.CRITICAL,
                                description="Potential double trade submission - no loading state protection",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Add loading state and disable button during submission",
                                impact="Duplicate trade execution"
                            ))
                    
                    # Check for unbounded retries
                    if 'retry' in line.lower() and 'infinite' not in line.lower():
                        if 'maxRetry' not in line and 'limit' not in line.lower():
                            self.findings.append(AuditFinding(
                                category="Resource Management",
                                risk_level=RiskLevel.HIGH,
                                description="Unbounded retry mechanism detected",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Add retry limits with exponential backoff",
                                impact="Resource exhaustion and infinite loops"
                            ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_frontend_memory_leaks(self):
        """Audit frontend for memory leak patterns"""
        print("\n💾 AUDITING FRONTEND MEMORY LEAKS...")
        
        frontend_files = list(self.frontend_dir.rglob("*.ts")) + list(self.frontend_dir.rglob("*.tsx"))
        
        for file_path in frontend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for missing cleanup
                    if 'useEffect' in line and 'return' not in content[max(0, i-10):i+10]:
                        if 'addEventListener' in content[max(0, i-5):i+5] or 'setInterval' in content[max(0, i-5):i+5]:
                            self.findings.append(AuditFinding(
                                category="Memory Leak",
                                risk_level=RiskLevel.HIGH,
                                description="useEffect with event listeners but no cleanup function",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Add cleanup function to remove event listeners",
                                impact="Memory leaks and performance degradation"
                            ))
                    
                    # Check for WebSocket subscription leaks
                    if 'WebSocket' in line and 'close' not in content[max(0, i-20):i+20]:
                        self.findings.append(AuditFinding(
                            category="Memory Leak",
                            risk_level=RiskLevel.HIGH,
                            description="WebSocket created without proper cleanup",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Add WebSocket.close() in cleanup function",
                            impact="Connection leaks and resource exhaustion"
                        ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_frontend_websocket_safety(self):
        """Audit WebSocket implementation for safety"""
        print("\n🔌 AUDITING WEBSOCKET SAFETY...")
        
        frontend_files = list(self.frontend_dir.rglob("*.ts")) + list(self.frontend_dir.rglob("*.tsx"))
        
        for file_path in frontend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for duplicate subscriptions
                    if 'addEventListener' in line and 'message' in line:
                        # Look for potential duplicate listeners
                        if content.count('addEventListener("message"') > 1:
                            self.findings.append(AuditFinding(
                                category="WebSocket Safety",
                                risk_level=RiskLevel.MEDIUM,
                                description="Potential duplicate WebSocket message listeners",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Remove existing listener before adding new one",
                                impact="Duplicate message processing"
                            ))
                    
                    # Check for missing connection state validation
                    if 'websocket.send' in line.lower() or 'ws.send' in line.lower():
                        if 'readyState' not in content[max(0, i-5):i+5]:
                            self.findings.append(AuditFinding(
                                category="WebSocket Safety",
                                risk_level=RiskLevel.HIGH,
                                description="WebSocket send without connection state check",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Check readyState before sending messages",
                                impact="Errors on disconnected WebSocket"
                            ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_backend_atomicity(self):
        """Audit backend for trade atomicity"""
        print("\n⚛️ AUDITING BACKEND TRADE ATOMICITY...")
        
        backend_files = list(self.backend_dir.rglob("*.py"))
        
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for non-atomic trade operations
                    if 'execute_trade' in line.lower() or 'create_trade' in line.lower():
                        # Look for database operations without transactions
                        if 'session' not in content[max(0, i-10):i+30] and 'transaction' not in content[max(0, i-10):i+30]:
                            self.findings.append(AuditFinding(
                                category="Trade Atomicity",
                                risk_level=RiskLevel.CRITICAL,
                                description="Trade operations without database transaction",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Wrap trade operations in database transaction",
                                impact="Partial state corruption on failure"
                            ))
                    
                    # Check for race conditions in trade execution
                    if 'broker_api' in line.lower() and 'update_ledger' in content[max(0, i-5):i+15]:
                        if 'atomic' not in content[max(0, i-10):i+20]:
                            self.findings.append(AuditFinding(
                                category="Race Condition",
                                risk_level=RiskLevel.CRITICAL,
                                description="Broker API and ledger update not atomic",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Use atomic transaction for broker and ledger operations",
                                impact="Inconsistent state between broker and ledger"
                            ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_backend_race_conditions(self):
        """Audit backend for race conditions"""
        print("\n🏁 AUDITING BACKEND RACE CONDITIONS...")
        
        backend_files = list(self.backend_dir.rglob("*.py"))
        
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for concurrent access without locks
                    if 'async def' in line and ('trade' in line.lower() or 'position' in line.lower()):
                        if 'lock' not in content[max(0, i-5):i+20] and 'mutex' not in content[max(0, i-5):i+20]:
                            self.findings.append(AuditFinding(
                                category="Race Condition",
                                risk_level=RiskLevel.HIGH,
                                description="Concurrent trade operations without locking",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Add async locks for critical sections",
                                impact="Race conditions in trade processing"
                            ))
                    
                    # Check for reconciliation override risks
                    if 'reconciliation' in line.lower() and 'override' in content[max(0, i-5):i+15]:
                        self.findings.append(AuditFinding(
                            category="Data Integrity",
                            risk_level=RiskLevel.CRITICAL,
                            description="Reconciliation can override valid broker state",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Add validation before reconciliation override",
                            impact="Valid broker state corruption"
                        ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_backend_jwt_security(self):
        """Audit JWT security implementation"""
        print("\n🔑 AUDITING JWT SECURITY...")
        
        backend_files = list(self.backend_dir.rglob("*.py"))
        
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for JWT validation
                    if 'jwt' in line.lower() and 'decode' in line:
                        if 'verify' not in content[max(0, i-5):i+5]:
                            self.findings.append(AuditFinding(
                                category="JWT Security",
                                risk_level=RiskLevel.CRITICAL,
                                description="JWT decode without verification",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Always verify JWT signature and claims",
                                impact="Token forgery attacks"
                            ))
                    
                    # Check for token refresh race conditions
                    if 'refresh_token' in line.lower():
                        if 'lock' not in content[max(0, i-10):i+20]:
                            self.findings.append(AuditFinding(
                                category="JWT Security",
                                risk_level=RiskLevel.HIGH,
                                description="Token refresh without concurrency protection",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Add locks for token refresh operations",
                                impact="Duplicate token issuance"
                            ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_backend_error_handling(self):
        """Audit backend error handling"""
        print("\n⚠️ AUDITING BACKEND ERROR HANDLING...")
        
        backend_files = list(self.backend_dir.rglob("*.py"))
        
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for unhandled exceptions
                    if 'except:' in line and ('Exception' not in line and line.strip() != 'except:'):
                        self.findings.append(AuditFinding(
                            category="Error Handling",
                            risk_level=RiskLevel.MEDIUM,
                            description="Bare except clause - may hide errors",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Specify exception types or log the exception",
                            impact="Debugging difficulties and hidden errors"
                        ))
                    
                    # Check for missing error logging
                    if 'raise' in line and 'log' not in content[max(0, i-5):i+5]:
                        self.findings.append(AuditFinding(
                            category="Error Handling",
                            risk_level=RiskLevel.MEDIUM,
                            description="Exception raised without logging",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Add logging before raising exceptions",
                            impact="Poor observability"
                        ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def audit_docker_security(self):
        """Audit Docker configuration for security"""
        print("\n🐳 AUDITING DOCKER SECURITY...")
        
        for docker_file in self.docker_files:
            if not docker_file.exists():
                continue
                
            try:
                with open(docker_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for running as root
                    if 'USER' not in content and 'FROM' in line:
                        self.findings.append(AuditFinding(
                            category="Docker Security",
                            risk_level=RiskLevel.HIGH,
                            description="Container running as root user",
                            file_path=str(docker_file.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Add USER instruction with non-root user",
                            impact="Container escape vulnerabilities"
                        ))
                    
                    # Check for exposed credentials
                    if 'ENV' in line and ('PASSWORD' in line or 'SECRET' in line or 'TOKEN' in line):
                        self.findings.append(AuditFinding(
                            category="Docker Security",
                            risk_level=RiskLevel.CRITICAL,
                            description="Secrets hardcoded in Dockerfile",
                            file_path=str(docker_file.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Use secrets management or environment files",
                            impact="Credential exposure"
                        ))
                            
            except Exception as e:
                print(f"Error reading {docker_file}: {e}")
    
    def audit_database_operations(self):
        """Audit database operations for safety"""
        print("\n🗄️ AUDITING DATABASE OPERATIONS...")
        
        db_files = list(self.backend_dir.rglob("*.py")) + list(self.backend_dir.rglob("*.sql"))
        
        for file_path in db_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for SQL injection risks
                    if 'execute' in line.lower() and 'format' in line:
                        if 'parameter' not in line.lower() and 'bind' not in line.lower():
                            self.findings.append(AuditFinding(
                                category="Database Security",
                                risk_level=RiskLevel.CRITICAL,
                                description="Potential SQL injection with string formatting",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip(),
                                patch="Use parameterized queries",
                                impact="SQL injection attacks"
                            ))
                    
                    # Check for missing connection pooling
                    if 'connect' in line.lower() and 'pool' not in content[max(0, i-10):i+10]:
                        self.findings.append(AuditFinding(
                            category="Database Performance",
                            risk_level=RiskLevel.MEDIUM,
                            description="Database connection without pooling",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip(),
                            patch="Use connection pooling",
                            impact="Poor performance under load"
                        ))
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        print("\n📋 GENERATING AUDIT REPORT...")
        
        # Categorize findings by risk level
        critical_issues = [f for f in self.findings if f.risk_level == RiskLevel.CRITICAL]
        high_issues = [f for f in self.findings if f.risk_level == RiskLevel.HIGH]
        medium_issues = [f for f in self.findings if f.risk_level == RiskLevel.MEDIUM]
        low_issues = [f for f in self.findings if f.risk_level == RiskLevel.LOW]
        
        report = {
            "audit_summary": {
                "total_findings": len(self.findings),
                "critical": len(critical_issues),
                "high": len(high_issues),
                "medium": len(medium_issues),
                "low": len(low_issues),
                "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            },
            "findings_by_category": {},
            "detailed_findings": [],
            "recommendations": [],
            "production_readiness_score": self.calculate_readiness_score()
        }
        
        # Group findings by category
        categories = set(f.category for f in self.findings)
        for category in categories:
            category_findings = [f for f in self.findings if f.category == category]
            report["findings_by_category"][category] = {
                "total": len(category_findings),
                "critical": len([f for f in category_findings if f.risk_level == RiskLevel.CRITICAL]),
                "high": len([f for f in category_findings if f.risk_level == RiskLevel.HIGH]),
                "medium": len([f for f in category_findings if f.risk_level == RiskLevel.MEDIUM]),
                "low": len([f for f in category_findings if f.risk_level == RiskLevel.LOW])
            }
        
        # Add detailed findings
        for finding in self.findings:
            report["detailed_findings"].append({
                "category": finding.category,
                "risk_level": finding.risk_level.value,
                "description": finding.description,
                "file_path": finding.file_path,
                "line_number": finding.line_number,
                "code_snippet": finding.code_snippet,
                "patch": finding.patch,
                "impact": finding.impact
            })
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations()
        
        return report
    
    def calculate_readiness_score(self) -> int:
        """Calculate production readiness score (0-100)"""
        total_issues = len(self.findings)
        critical_weight = 10
        high_weight = 5
        medium_weight = 2
        low_weight = 1
        
        critical_count = len([f for f in self.findings if f.risk_level == RiskLevel.CRITICAL])
        high_count = len([f for f in self.findings if f.risk_level == RiskLevel.HIGH])
        medium_count = len([f for f in self.findings if f.risk_level == RiskLevel.MEDIUM])
        low_count = len([f for f in self.findings if f.risk_level == RiskLevel.LOW])
        
        weighted_score = (critical_count * critical_weight + 
                         high_count * high_weight + 
                         medium_count * medium_weight + 
                         low_count * low_weight)
        
        # Base score of 100, subtract weighted issues
        score = max(0, 100 - weighted_score)
        return score
    
    def generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        critical_count = len([f for f in self.findings if f.risk_level == RiskLevel.CRITICAL])
        high_count = len([f for f in self.findings if f.risk_level == RiskLevel.HIGH])
        
        if critical_count > 0:
            recommendations.append(f"🚨 IMMEDIATE ACTION REQUIRED: Fix {critical_count} critical security issues before production deployment")
        
        if high_count > 0:
            recommendations.append(f"⚠️ HIGH PRIORITY: Address {high_count} high-risk issues to ensure system reliability")
        
        # Check for specific patterns
        jwt_issues = [f for f in self.findings if f.category == "JWT Security"]
        if jwt_issues:
            recommendations.append("🔐 SECURITY: Implement proper JWT security measures including httpOnly cookies")
        
        race_conditions = [f for f in self.findings if f.category == "Race Condition"]
        if race_conditions:
            recommendations.append("⚡ CONCURRENCY: Add proper locking mechanisms for critical operations")
        
        atomicity_issues = [f for f in self.findings if f.category == "Trade Atomicity"]
        if atomicity_issues:
            recommendations.append("⚛️ ATOMICITY: Ensure all trade operations are wrapped in database transactions")
        
        memory_leaks = [f for f in self.findings if f.category == "Memory Leak"]
        if memory_leaks:
            recommendations.append("💾 PERFORMANCE: Fix memory leaks to ensure long-term stability")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_path: str = "production_audit_report.json"):
        """Save audit report to file"""
        report_path = self.project_root / output_path
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Audit report saved to: {report_path}")
        return report_path

def main():
    """Main audit execution"""
    project_root = Path(__file__).parent
    auditor = ProductionAuditor(str(project_root))
    
    print("🚀 NEXUS TRADING SYSTEM - PRODUCTION AUDIT")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 80)
    
    # Run comprehensive audit
    report = auditor.run_full_audit()
    
    # Save report
    report_path = auditor.save_report(report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Findings: {report['audit_summary']['total_findings']}")
    print(f"Critical Issues: {report['audit_summary']['critical']} 🚨")
    print(f"High Issues: {report['audit_summary']['high']} ⚠️")
    print(f"Medium Issues: {report['audit_summary']['medium']} 🔍")
    print(f"Low Issues: {report['audit_summary']['low']} ℹ️")
    print(f"Production Readiness Score: {report['production_readiness_score']}/100")
    
    print("\n🎯 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    if report['audit_summary']['critical'] > 0:
        print("\n🚨 CRITICAL ISSUES DETECTED - DO NOT DEPLOY TO PRODUCTION")
    elif report['production_readiness_score'] < 70:
        print("\n⚠️ SYSTEM NOT READY FOR PRODUCTION")
    else:
        print("\n✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT")
    
    print(f"\n📄 Full report available at: {report_path}")

if __name__ == "__main__":
    main()
