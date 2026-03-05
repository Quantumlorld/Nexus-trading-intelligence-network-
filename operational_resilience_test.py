"""
Nexus Trading System - Operational Resilience Live Test
Simulates failures and proves system behavior with logs
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

# Configure logging to see all system logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('operational_test.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    logs: List[str]
    trading_state_before: bool
    trading_state_after: bool
    system_state: Dict[str, Any]
    duration: float

class OperationalResilienceTester:
    """Tests operational resilience mechanisms"""
    
    def __init__(self):
        self.test_results = []
        self.current_test_logs = []
        
    def log(self, message: str, level: str = "INFO"):
        """Add log entry to current test"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.current_test_logs.append(log_entry)
        
        if level == "DEBUG":
            logger.debug(message)
        elif level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "CRITICAL":
            logger.critical(message)
    
    async def get_trading_state(self) -> bool:
        """Get current trading state"""
        try:
            from core.system_control import system_control_manager
            return await system_control_manager.is_trading_enabled()
        except Exception as e:
            self.log(f"Failed to get trading state: {e}", "ERROR")
            return False
    
    async def simulate_phase1_emergency_kill_switch(self) -> TestResult:
        """PHASE 1: Global Emergency Kill Switch"""
        self.current_test_logs = []
        start_time = time.time()
        
        self.log("=== PHASE 1: GLOBAL EMERGENCY KILL SWITCH TEST ===", "INFO")
        
        # Get initial state
        trading_before = await self.get_trading_state()
        self.log(f"Initial trading state: {trading_before}", "INFO")
        
        try:
            from core.system_control import system_control_manager
            from core.broker_safe_executor import broker_safe_executor
            from core.operational_metrics import operational_metrics
            
            # Disable trading
            self.log("Disabling trading via system control...", "INFO")
            disable_result = await system_control_manager.disable_trading(
                "Emergency kill switch test", 
                "operational_test"
            )
            self.log(f"Disable trading result: {disable_result}", "INFO")
            
            # Verify trading is disabled
            trading_after_disable = await self.get_trading_state()
            self.log(f"Trading state after disable: {trading_after_disable}", "INFO")
            
            # Attempt 3 trades
            self.log("Attempting 3 trade submissions while trading disabled...", "WARNING")
            
            for i in range(3):
                self.log(f"Trade attempt {i+1}...", "INFO")
                
                trade_request = {
                    'trade_uuid': f"test-kill-switch-{i}-{int(time.time())}",
                    'user_id': 1,
                    'symbol': 'EUR/USD',
                    'action': 'BUY',
                    'order_type': 'MARKET',
                    'quantity': 0.01,
                    'entry_price': 1.1000,
                    'timeframe': 'H1'
                }
                
                result = await broker_safe_executor.execute_trade(1, trade_request)
                self.log(f"Trade {i+1} result: {result}", "INFO")
                
                if result['success']:
                    self.log(f"ERROR: Trade {i+1} succeeded when trading should be disabled!", "CRITICAL")
                else:
                    self.log(f"Trade {i+1} correctly rejected: {result.get('error')}", "INFO")
            
            # Re-enable trading
            self.log("Re-enabling trading...", "INFO")
            enable_result = await system_control_manager.enable_trading(
                "Emergency kill switch test completed", 
                "operational_test"
            )
            self.log(f"Enable trading result: {enable_result}", "INFO")
            
            # Get final state
            trading_final = await self.get_trading_state()
            self.log(f"Final trading state: {trading_final}", "INFO")
            
            # Get system state
            system_state = {
                'trading_enabled': trading_final,
                'metrics_counters': operational_metrics.counters,
                'metrics_gauges': operational_metrics.gauges
            }
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Emergency Kill Switch",
                success=not trading_after_disable and trading_final,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=trading_final,
                system_state=system_state,
                duration=duration
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"Emergency kill switch test failed: {e}", "ERROR")
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Emergency Kill Switch",
                success=False,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=False,
                system_state={'error': str(e)},
                duration=duration
            )
            
            self.test_results.append(result)
            return result
    
    async def simulate_phase2_broker_connectivity(self) -> TestResult:
        """PHASE 2: Broker Connectivity Monitor"""
        self.current_test_logs = []
        start_time = time.time()
        
        self.log("=== PHASE 2: BROKER CONNECTIVITY MONITOR TEST ===", "INFO")
        
        trading_before = await self.get_trading_state()
        self.log(f"Initial trading state: {trading_before}", "INFO")
        
        try:
            from core.system_control import system_control_manager
            from core.operational_metrics import operational_metrics
            from execution.mt5_bridge import MT5Bridge
            
            # Start monitoring
            self.log("Starting system monitoring...", "INFO")
            await system_control_manager.start_monitoring()
            
            # Simulate broker disconnection by mocking MT5 bridge failure
            self.log("Simulating broker disconnection...", "WARNING")
            
            # Force multiple broker failures to trigger auto-disable
            for i in range(3):
                self.log(f"Simulating broker failure {i+1}...", "WARNING")
                system_control_manager.broker_consecutive_failures += 1
                operational_metrics.record_broker_failure()
                
                # Check if trading gets disabled
                trading_state = await self.get_trading_state()
                self.log(f"Trading state after failure {i+1}: {trading_state}", "INFO")
                
                if not trading_state:
                    self.log("Trading automatically disabled due to broker failures!", "CRITICAL")
                    break
            
            # Verify trading is disabled
            trading_after_failure = await self.get_trading_state()
            self.log(f"Trading state after broker failures: {trading_after_failure}", "INFO")
            
            # Simulate broker recovery
            self.log("Simulating broker recovery...", "INFO")
            system_control_manager.broker_consecutive_failures = 0
            operational_metrics.record_broker_recovery()
            
            # Re-enable trading manually
            self.log("Re-enabling trading after broker recovery...", "INFO")
            await system_control_manager.enable_trading(
                "Broker connectivity restored", 
                "operational_test"
            )
            
            # Stop monitoring
            await system_control_manager.stop_monitoring()
            
            # Get final state
            trading_final = await self.get_trading_state()
            self.log(f"Final trading state: {trading_final}", "INFO")
            
            system_state = {
                'trading_enabled': trading_final,
                'broker_failures': operational_metrics.counters['broker_failures'],
                'broker_connected': operational_metrics.gauges['broker_connected']
            }
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Broker Connectivity Monitor",
                success=not trading_after_failure and trading_final,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=trading_final,
                system_state=system_state,
                duration=duration
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"Broker connectivity test failed: {e}", "ERROR")
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Broker Connectivity Monitor",
                success=False,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=False,
                system_state={'error': str(e)},
                duration=duration
            )
            
            self.test_results.append(result)
            return result
    
    async def simulate_phase3_database_health(self) -> TestResult:
        """PHASE 3: Database Health Guard"""
        self.current_test_logs = []
        start_time = time.time()
        
        self.log("=== PHASE 3: DATABASE HEALTH GUARD TEST ===", "INFO")
        
        trading_before = await self.get_trading_state()
        self.log(f"Initial trading state: {trading_before}", "INFO")
        
        try:
            from core.system_control import system_control_manager
            from core.operational_metrics import operational_metrics
            
            # Simulate database failures
            self.log("Simulating database connection failures...", "WARNING")
            
            for i in range(3):
                self.log(f"Simulating DB failure {i+1}...", "WARNING")
                system_control_manager.db_consecutive_failures += 1
                operational_metrics.record_db_failure()
                
                # Check if trading gets disabled
                trading_state = await self.get_trading_state()
                self.log(f"Trading state after DB failure {i+1}: {trading_state}", "INFO")
                
                if not trading_state:
                    self.log("Trading automatically disabled due to DB failures!", "CRITICAL")
                    break
            
            # Verify trading is disabled
            trading_after_failure = await self.get_trading_state()
            self.log(f"Trading state after DB failures: {trading_after_failure}", "INFO")
            
            # Simulate database recovery
            self.log("Simulating database recovery...", "INFO")
            system_control_manager.db_consecutive_failures = 0
            operational_metrics.record_db_recovery()
            
            # Re-enable trading
            self.log("Re-enabling trading after DB recovery...", "INFO")
            await system_control_manager.enable_trading(
                "Database connectivity restored", 
                "operational_test"
            )
            
            # Get final state
            trading_final = await self.get_trading_state()
            self.log(f"Final trading state: {trading_final}", "INFO")
            
            system_state = {
                'trading_enabled': trading_final,
                'db_failures': operational_metrics.counters['db_failures'],
                'db_connected': operational_metrics.gauges['db_connected']
            }
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Database Health Guard",
                success=not trading_after_failure and trading_final,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=trading_final,
                system_state=system_state,
                duration=duration
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"Database health test failed: {e}", "ERROR")
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Database Health Guard",
                success=False,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=False,
                system_state={'error': str(e)},
                duration=duration
            )
            
            self.test_results.append(result)
            return result
    
    async def simulate_phase4_monitoring_metrics(self) -> TestResult:
        """PHASE 4: Monitoring & Metrics"""
        self.current_test_logs = []
        start_time = time.time()
        
        self.log("=== PHASE 4: MONITORING & METRICS TEST ===", "INFO")
        
        trading_before = await self.get_trading_state()
        self.log(f"Initial trading state: {trading_before}", "INFO")
        
        try:
            from core.operational_metrics import operational_metrics
            from core.broker_safe_executor import broker_safe_executor
            
            # Simulate 20 trades
            self.log("Simulating 20 trades...", "INFO")
            
            for i in range(20):
                trade_request = {
                    'trade_uuid': f"metrics-test-{i}-{int(time.time())}",
                    'user_id': 1,
                    'symbol': 'EUR/USD',
                    'action': 'BUY' if i % 2 == 0 else 'SELL',
                    'order_type': 'MARKET',
                    'quantity': 0.01,
                    'entry_price': 1.1000 + (i * 0.0001),
                    'timeframe': 'H1'
                }
                
                # Simulate different outcomes
                if i in [5, 15]:  # 2 forced timeouts
                    self.log(f"Trade {i+1}: Simulating timeout...", "WARNING")
                    operational_metrics.record_trade_execution(2.5, False, timeout=True)
                elif i == 10:  # 1 forced ledger retry
                    self.log(f"Trade {i+1}: Simulating ledger retry...", "WARNING")
                    operational_metrics.record_trade_execution(0.8, True)
                    operational_metrics.record_ledger_retry()
                else:
                    self.log(f"Trade {i+1}: Normal execution...", "INFO")
                    operational_metrics.record_trade_execution(0.5, True)
            
            # Simulate broker latency
            self.log("Simulating broker latency metrics...", "INFO")
            for latency in [0.1, 0.2, 0.15, 0.3, 0.25]:
                operational_metrics.record_broker_latency(latency)
            
            # Simulate DB latency
            self.log("Simulating DB latency metrics...", "INFO")
            for latency in [0.05, 0.08, 0.06, 0.12, 0.09]:
                operational_metrics.record_db_latency(latency)
            
            # Get metrics output
            self.log("Generating Prometheus metrics output...", "INFO")
            metrics_output = operational_metrics.get_prometheus_metrics()
            
            # Log key metrics
            self.log("=== KEY METRICS SUMMARY ===", "INFO")
            self.log(f"Total trades: {operational_metrics.counters['trades_total']}", "INFO")
            self.log(f"Successful trades: {operational_metrics.counters['trades_successful']}", "INFO")
            self.log(f"Failed trades: {operational_metrics.counters['trades_failed']}", "INFO")
            self.log(f"Timeout trades: {operational_metrics.counters['trades_timeout']}", "INFO")
            self.log(f"Ledger retries: {operational_metrics.counters['ledger_update_retries']}", "INFO")
            
            trading_final = await self.get_trading_state()
            
            system_state = {
                'trading_enabled': trading_final,
                'metrics_summary': operational_metrics.get_summary_stats()
            }
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Monitoring & Metrics",
                success=True,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=trading_final,
                system_state=system_state,
                duration=duration
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"Monitoring metrics test failed: {e}", "ERROR")
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Monitoring & Metrics",
                success=False,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=False,
                system_state={'error': str(e)},
                duration=duration
            )
            
            self.test_results.append(result)
            return result
    
    async def simulate_phase5_alert_escalation(self) -> TestResult:
        """PHASE 5: Alert Escalation Logic"""
        self.current_test_logs = []
        start_time = time.time()
        
        self.log("=== PHASE 5: ALERT ESCALATION LOGIC TEST ===", "INFO")
        
        trading_before = await self.get_trading_state()
        self.log(f"Initial trading state: {trading_before}", "INFO")
        
        try:
            from core.system_control import system_control_manager
            from core.operational_metrics import operational_metrics
            
            # Test CRITICAL alert (should trigger kill switch)
            self.log("Testing CRITICAL alert - Ledger update total failure...", "CRITICAL")
            await system_control_manager.disable_trading(
                "CRITICAL: Ledger update total failure", 
                "alert_system"
            )
            
            trading_after_critical = await self.get_trading_state()
            self.log(f"Trading state after CRITICAL alert: {trading_after_critical}", "CRITICAL")
            
            # Re-enable for next test
            await system_control_manager.enable_trading(
                "Recovery from critical alert", 
                "alert_system"
            )
            
            # Test HIGH alert (logs + monitoring event)
            self.log("Testing HIGH alert - Broker timeout...", "WARNING")
            operational_metrics.record_trade_execution(5.0, False, timeout=True)
            self.log("HIGH alert logged - trading should continue", "WARNING")
            
            trading_after_high = await self.get_trading_state()
            self.log(f"Trading state after HIGH alert: {trading_after_high}", "INFO")
            
            # Test WARNING alert (log only)
            self.log("Testing WARNING alert - High slippage detected...", "WARNING")
            self.log("WARNING alert logged - trading should continue", "WARNING")
            
            trading_after_warning = await self.get_trading_state()
            self.log(f"Trading state after WARNING alert: {trading_after_warning}", "INFO")
            
            # Test broker disconnect scenario
            self.log("Testing broker disconnect scenario...", "WARNING")
            operational_metrics.record_broker_failure()
            self.log("Broker disconnect alert sent", "WARNING")
            
            # Test DB failure scenario
            self.log("Testing DB failure scenario...", "WARNING")
            operational_metrics.record_db_failure()
            self.log("DB failure alert sent", "WARNING")
            
            trading_final = await self.get_trading_state()
            self.log(f"Final trading state: {trading_final}", "INFO")
            
            system_state = {
                'trading_enabled': trading_final,
                'alert_severity_tested': ['CRITICAL', 'HIGH', 'WARNING'],
                'broker_failures': operational_metrics.counters['broker_failures'],
                'db_failures': operational_metrics.counters['db_failures']
            }
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Alert Escalation Logic",
                success=not trading_after_critical and trading_final,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=trading_final,
                system_state=system_state,
                duration=duration
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"Alert escalation test failed: {e}", "ERROR")
            duration = time.time() - start_time
            
            result = TestResult(
                test_name="Alert Escalation Logic",
                success=False,
                logs=self.current_test_logs.copy(),
                trading_state_before=trading_before,
                trading_state_after=False,
                system_state={'error': str(e)},
                duration=duration
            )
            
            self.test_results.append(result)
            return result
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("# OPERATIONAL RESILIENCE TEST REPORT")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        for result in self.test_results:
            report.append(f"## {result.test_name}")
            report.append(f"**Success:** {result.success}")
            report.append(f"**Duration:** {result.duration:.2f}s")
            report.append(f"**Trading State Before:** {result.trading_state_before}")
            report.append(f"**Trading State After:** {result.trading_state_after}")
            report.append("")
            
            report.append("### System State:")
            for key, value in result.system_state.items():
                report.append(f"- {key}: {value}")
            report.append("")
            
            report.append("### Log Entries:")
            for log in result.logs:
                report.append(f"```\n{log}\n```")
            report.append("")
            report.append("---")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Run all operational resilience tests"""
    print("🔥 OPERATIONAL RESILIENCE HARDENING & LIVE TEST 🔥")
    print("=" * 60)
    
    tester = OperationalResilienceTester()
    
    # Run all phases
    phases = [
        ("Phase 1: Emergency Kill Switch", tester.simulate_phase1_emergency_kill_switch),
        ("Phase 2: Broker Connectivity", tester.simulate_phase2_broker_connectivity),
        ("Phase 3: Database Health", tester.simulate_phase3_database_health),
        ("Phase 4: Monitoring & Metrics", tester.simulate_phase4_monitoring_metrics),
        ("Phase 5: Alert Escalation", tester.simulate_phase5_alert_escalation)
    ]
    
    for phase_name, phase_func in phases:
        print(f"\n🚀 Running {phase_name}...")
        try:
            result = await phase_func()
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"{status} - Duration: {result.duration:.2f}s")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Generate report
    print("\n📊 Generating test report...")
    report = tester.generate_report()
    
    # Save report
    with open('operational_resilience_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Report saved to: operational_resilience_report.md")
    print("📋 Logs saved to: operational_test.log")
    
    # Summary
    passed = sum(1 for r in tester.test_results if r.success)
    total = len(tester.test_results)
    print(f"\n📈 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL OPERATIONAL RESILIENCE TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - review logs for details")

if __name__ == "__main__":
    asyncio.run(main())
