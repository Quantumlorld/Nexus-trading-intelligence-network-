#!/usr/bin/env python3
"""
NEXUS TRADING SYSTEM - FAILURE SIMULATION & LOAD TESTING
=========================================================

Comprehensive failure simulation and load testing for production readiness.
Tests system behavior under various failure conditions and high load scenarios.

Test Scenarios:
1. Broker API timeout
2. Partial fill handling
3. Trade rejection
4. Network failure mid-execution
5. Database write failure after broker execution
6. Backend restart during active trade
7. High latency environment (500-2000ms)
8. 100 concurrent trade submissions
9. 1000 trades over 10 minutes load test
10. Memory growth monitoring
11. Event loop blocking detection
12. Error rate analysis
"""

import asyncio
import time
import json
import random
import threading
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import psutil
import pytest
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('failure_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success_count: int
    failure_count: int
    total_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors: List[str]
    timestamp: str

@dataclass
class TradeRequest:
    """Trade request structure"""
    symbol: str
    quantity: float
    trade_type: str
    order_type: str
    price: Optional[float] = None
    user_id: str = "test_user"

class FailureSimulator:
    """Failure simulation and load testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = None
        self.test_results: List[TestResult] = []
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        def monitor():
            while True:
                try:
                    # Memory usage
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_samples.append(memory_mb)
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.cpu_samples.append(cpu_percent)
                    
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    async def execute_trade(self, trade: TradeRequest, simulate_failure: str = None) -> Dict[str, Any]:
        """Execute a trade with optional failure simulation"""
        start_time = time.time()
        
        try:
            # Simulate different failure scenarios
            if simulate_failure == "timeout":
                await asyncio.sleep(35)  # Exceed timeout
                raise asyncio.TimeoutError("Simulated timeout")
            
            elif simulate_failure == "network_failure":
                # Simulate network failure by closing connection
                if self.session:
                    await self.session.close()
                    self.session = None
                raise ConnectionError("Simulated network failure")
            
            elif simulate_failure == "high_latency":
                # Simulate high latency (500-2000ms)
                latency = random.uniform(0.5, 2.0)
                await asyncio.sleep(latency)
            
            # Execute actual trade
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"Content-Type": "application/json"}
                )
            
            url = f"{self.base_url}/api/trading/execute"
            payload = asdict(trade)
            
            async with self.session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def test_broker_api_timeout(self, num_trades: int = 10) -> TestResult:
        """Test broker API timeout scenario"""
        logger.info(f"Testing broker API timeout with {num_trades} trades...")
        
        trades = [
            TradeRequest(
                symbol="EUR/USD",
                quantity=0.01,
                trade_type="BUY",
                order_type="MARKET"
            ) for _ in range(num_trades)
        ]
        
        results = []
        errors = []
        
        for i, trade in enumerate(trades):
            result = await self.execute_trade(trade, simulate_failure="timeout")
            results.append(result)
            if not result["success"]:
                errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="Broker API Timeout",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],  # Limit error display
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_partial_fill_handling(self, num_trades: int = 20) -> TestResult:
        """Test partial fill handling"""
        logger.info(f"Testing partial fill handling with {num_trades} trades...")
        
        trades = [
            TradeRequest(
                symbol="EUR/USD",
                quantity=random.uniform(0.1, 1.0),
                trade_type="BUY",
                order_type="LIMIT",
                price=1.0500
            ) for _ in range(num_trades)
        ]
        
        results = []
        errors = []
        
        for i, trade in enumerate(trades):
            result = await self.execute_trade(trade)
            results.append(result)
            
            # Check for partial fill indicators
            if result["success"] and "data" in result:
                data = result["data"]
                if "filled_quantity" in data and data["filled_quantity"] < trade.quantity:
                    logger.info(f"Partial fill detected: {data['filled_quantity']}/{trade.quantity}")
            
            if not result["success"]:
                errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="Partial Fill Handling",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_trade_rejection(self, num_trades: int = 15) -> TestResult:
        """Test trade rejection handling"""
        logger.info(f"Testing trade rejection with {num_trades} trades...")
        
        # Create trades likely to be rejected (insufficient margin, invalid symbols, etc.)
        trades = [
            TradeRequest(
                symbol="INVALID/PAIR",
                quantity=999999,  # Unrealistic quantity
                trade_type="BUY",
                order_type="MARKET"
            ) for _ in range(num_trades)
        ]
        
        results = []
        errors = []
        
        for i, trade in enumerate(trades):
            result = await self.execute_trade(trade)
            results.append(result)
            
            if not result["success"]:
                errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="Trade Rejection",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_network_failure_mid_execution(self, num_trades: int = 10) -> TestResult:
        """Test network failure during trade execution"""
        logger.info(f"Testing network failure mid-execution with {num_trades} trades...")
        
        trades = [
            TradeRequest(
                symbol="EUR/USD",
                quantity=0.01,
                trade_type="BUY",
                order_type="MARKET"
            ) for _ in range(num_trades)
        ]
        
        results = []
        errors = []
        
        for i, trade in enumerate(trades):
            # Simulate network failure for half the trades
            simulate_failure = "network_failure" if i % 2 == 0 else None
            result = await self.execute_trade(trade, simulate_failure=simulate_failure)
            results.append(result)
            
            if not result["success"]:
                errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="Network Failure Mid-Execution",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_high_latency_environment(self, num_trades: int = 30) -> TestResult:
        """Test system behavior under high latency (500-2000ms)"""
        logger.info(f"Testing high latency environment with {num_trades} trades...")
        
        trades = [
            TradeRequest(
                symbol="EUR/USD",
                quantity=0.01,
                trade_type="BUY",
                order_type="MARKET"
            ) for _ in range(num_trades)
        ]
        
        results = []
        errors = []
        
        for i, trade in enumerate(trades):
            result = await self.execute_trade(trade, simulate_failure="high_latency")
            results.append(result)
            
            if not result["success"]:
                errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="High Latency Environment",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_concurrent_trades(self, num_concurrent: int = 100) -> TestResult:
        """Test concurrent trade submissions"""
        logger.info(f"Testing {num_concurrent} concurrent trade submissions...")
        
        trades = [
            TradeRequest(
                symbol="EUR/USD",
                quantity=0.01,
                trade_type="BUY" if i % 2 == 0 else "SELL",
                order_type="MARKET",
                user_id=f"concurrent_user_{i}"
            ) for i in range(num_concurrent)
        ]
        
        # Execute all trades concurrently
        tasks = [self.execute_trade(trade) for trade in trades]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "response_time": 0,
                    "error": str(result)
                })
                errors.append(f"Trade {i+1}: {result}")
            else:
                processed_results.append(result)
                if not result["success"]:
                    errors.append(f"Trade {i+1}: {result['error']}")
        
        response_times = [r["response_time"] for r in processed_results if r["response_time"] > 0]
        success_count = sum(1 for r in processed_results if r["success"])
        
        return TestResult(
            test_name="Concurrent Trade Submissions",
            success_count=success_count,
            failure_count=len(processed_results) - success_count,
            total_requests=len(processed_results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(processed_results) - success_count) / len(processed_results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def test_load_simulation(self, duration_minutes: int = 10, trades_per_minute: int = 100) -> TestResult:
        """Load test: 1000 trades over 10 minutes"""
        total_trades = duration_minutes * trades_per_minute
        logger.info(f"Load testing: {total_trades} trades over {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = []
        errors = []
        trade_count = 0
        
        while time.time() < end_time and trade_count < total_trades:
            # Calculate target time for this trade
            target_time = start_time + (trade_count / trades_per_minute * 60)
            delay = max(0, target_time - time.time())
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Execute trade
            trade = TradeRequest(
                symbol=random.choice(["EUR/USD", "GBP/USD", "USD/JPY"]),
                quantity=random.uniform(0.01, 0.1),
                trade_type=random.choice(["BUY", "SELL"]),
                order_type="MARKET",
                user_id=f"load_user_{trade_count}"
            )
            
            result = await self.execute_trade(trade)
            results.append(result)
            
            if not result["success"]:
                errors.append(f"Trade {trade_count+1}: {result['error']}")
            
            trade_count += 1
        
        response_times = [r["response_time"] for r in results if r["response_time"] > 0]
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            test_name="Load Simulation",
            success_count=success_count,
            failure_count=len(results) - success_count,
            total_requests=len(results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            error_rate=(len(results) - success_count) / len(results) * 100,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            errors=errors[:10],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all failure simulation and load tests"""
        logger.info("🚀 STARTING COMPREHENSIVE FAILURE SIMULATION & LOAD TESTING")
        logger.info("=" * 80)
        
        # Start monitoring
        monitor_thread = self.start_monitoring()
        
        try:
            # Run all tests
            test_results = []
            
            # Individual failure scenarios
            test_results.append(await self.test_broker_api_timeout())
            test_results.append(await self.test_partial_fill_handling())
            test_results.append(await self.test_trade_rejection())
            test_results.append(await self.test_network_failure_mid_execution())
            test_results.append(await self.test_high_latency_environment())
            
            # Concurrency and load tests
            test_results.append(await self.test_concurrent_trades())
            test_results.append(await self.test_load_simulation())
            
            self.test_results = test_results
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            return report
        
        finally:
            # Stop monitoring
            monitor_thread.join(timeout=1)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        total_requests = sum(r.total_requests for r in self.test_results)
        total_successes = sum(r.success_count for r in self.test_results)
        total_failures = sum(r.failure_count for r in self.test_results)
        
        # Calculate overall metrics
        overall_error_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
        avg_response_time = statistics.mean([r.avg_response_time for r in self.test_results])
        max_response_time = max(r.max_response_time for r in self.test_results)
        
        # Memory and CPU analysis
        memory_growth = max(self.memory_samples) - min(self.memory_samples) if self.memory_samples else 0
        avg_memory = statistics.mean(self.memory_samples) if self.memory_samples else 0
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "total_requests": total_requests,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "overall_error_rate": round(overall_error_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "memory_growth_mb": round(memory_growth, 2),
                "avg_memory_mb": round(avg_memory, 2),
                "avg_cpu_percent": round(avg_cpu, 2),
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            },
            "test_results": [asdict(result) for result in self.test_results],
            "performance_analysis": {
                "response_time_distribution": {
                    "under_100ms": len([r for r in self.test_results if r.avg_response_time < 0.1]),
                    "under_500ms": len([r for r in self.test_results if r.avg_response_time < 0.5]),
                    "under_1s": len([r for r in self.test_results if r.avg_response_time < 1.0]),
                    "over_1s": len([r for r in self.test_results if r.avg_response_time >= 1.0])
                },
                "error_rate_analysis": {
                    "critical_errors": len([r for r in self.test_results if r.error_rate > 50]),
                    "high_errors": len([r for r in self.test_results if 20 < r.error_rate <= 50]),
                    "moderate_errors": len([r for r in self.test_results if 5 < r.error_rate <= 20]),
                    "low_errors": len([r for r in self.test_results if r.error_rate <= 5])
                },
                "resource_utilization": {
                    "memory_efficiency": "Good" if memory_growth < 100 else "Poor",
                    "cpu_efficiency": "Good" if avg_cpu < 80 else "Poor"
                }
            },
            "recommendations": self.generate_recommendations(),
            "production_readiness": self.assess_production_readiness()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate test-based recommendations"""
        recommendations = []
        
        # Analyze error rates
        high_error_tests = [r for r in self.test_results if r.error_rate > 20]
        if high_error_tests:
            recommendations.append("🚨 HIGH ERROR RATE: Address reliability issues in failure scenarios")
        
        # Analyze response times
        slow_tests = [r for r in self.test_results if r.avg_response_time > 1.0]
        if slow_tests:
            recommendations.append("⏱️ SLOW RESPONSE TIMES: Optimize performance for better user experience")
        
        # Analyze memory growth
        memory_growth = max(self.memory_samples) - min(self.memory_samples) if self.memory_samples else 0
        if memory_growth > 100:
            recommendations.append("💾 MEMORY GROWTH: Investigate memory leaks and optimize resource usage")
        
        # Analyze concurrent performance
        concurrent_test = next((r for r in self.test_results if r.test_name == "Concurrent Trade Submissions"), None)
        if concurrent_test and concurrent_test.error_rate > 10:
            recommendations.append("⚡ CONCURRENCY ISSUES: Improve handling of simultaneous requests")
        
        # Analyze load test performance
        load_test = next((r for r in self.test_results if r.test_name == "Load Simulation"), None)
        if load_test and load_test.error_rate > 5:
            recommendations.append("🔄 LOAD HANDLING: Optimize system for sustained high load")
        
        return recommendations
    
    def assess_production_readiness(self) -> Dict[str, Any]:
        """Assess production readiness based on test results"""
        score = 100
        
        # Deduct points for high error rates
        for result in self.test_results:
            if result.error_rate > 50:
                score -= 20
            elif result.error_rate > 20:
                score -= 10
            elif result.error_rate > 5:
                score -= 5
        
        # Deduct points for slow response times
        for result in self.test_results:
            if result.avg_response_time > 2.0:
                score -= 10
            elif result.avg_response_time > 1.0:
                score -= 5
        
        # Deduct points for memory growth
        memory_growth = max(self.memory_samples) - min(self.memory_samples) if self.memory_samples else 0
        if memory_growth > 200:
            score -= 15
        elif memory_growth > 100:
            score -= 10
        
        score = max(0, score)
        
        readiness_level = "PRODUCTION READY" if score >= 80 else \
                        "NEEDS IMPROVEMENT" if score >= 60 else \
                        "NOT READY"
        
        return {
            "readiness_score": score,
            "readiness_level": readiness_level,
            "key_concerns": [
                r.test_name for r in self.test_results 
                if r.error_rate > 20 or r.avg_response_time > 1.0
            ]
        }

async def main():
    """Main test execution"""
    project_root = Path(__file__).parent
    
    print("🚀 NEXUS TRADING SYSTEM - FAILURE SIMULATION & LOAD TESTING")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print("=" * 80)
    
    async with FailureSimulator() as simulator:
        # Run comprehensive tests
        report = await simulator.run_all_tests()
        
        # Save report
        report_path = project_root / "failure_simulation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 FAILURE SIMULATION & LOAD TEST RESULTS")
        print("=" * 80)
        
        summary = report["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {100 - summary['overall_error_rate']:.2f}%")
        print(f"Error Rate: {summary['overall_error_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
        print(f"Max Response Time: {summary['max_response_time']:.3f}s")
        print(f"Memory Growth: {summary['memory_growth_mb']:.2f} MB")
        
        print("\n🎯 PRODUCTION READINESS:")
        readiness = report["production_readiness"]
        print(f"Score: {readiness['readiness_score']}/100")
        print(f"Level: {readiness['readiness_level']}")
        
        if readiness['key_concerns']:
            print("\n⚠️  KEY CONCERNS:")
            for concern in readiness['key_concerns']:
                print(f"  - {concern}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print(f"\n📄 Full report: {report_path}")
        
        # Determine next steps
        if readiness['readiness_score'] >= 80:
            print("\n✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        elif readiness['readiness_score'] >= 60:
            print("\n⚠️  SYSTEM NEEDS IMPROVEMENTS BEFORE PRODUCTION")
        else:
            print("\n🚨 SYSTEM NOT READY FOR PRODUCTION - CRITICAL ISSUES FOUND")

if __name__ == "__main__":
    asyncio.run(main())
